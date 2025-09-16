#!/usr/bin/env python3
"""
All-in-one ROS2 node for Ballbot using ESP32 (serial)
- Reads IMU + encoder lines from ESP32 over serial
- Publishes /ballbot/imu (sensor_msgs/Imu) and /ballbot/joint_states
- Runs a linear Kalman filter (IMU + encoders)
- Runs LQR controller -> motor torques -> PWM commands to ESP32
- Safety: tilt emergency stop, command timeout
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import threading
from collections import deque

from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist

# Serial comms
import serial

# Try to import CARE solver for LQR; fallback to identity K
try:
    from scipy.linalg import solve_continuous_are
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---------------------------
# LQR controller (adapted)
# ---------------------------
class BallbotLQRController:
    def __init__(self, max_torque=12.0):
        # --- physical params (tune to your robot)
        self.m_ball = 0.5
        self.m_body = 1.8
        self.R_ball = 0.111
        self.h = 0.22
        self.g = 9.81
        self.I_body_xx = 1.2
        self.I_body_yy = 1.2

        self.n_motors = 3
        self.motor_angles = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
        self.motor_radius = 0.12
        self.max_torque = max_torque

        # geometry transforms
        self.motor_to_ball_x = np.cos(self.motor_angles)
        self.motor_to_ball_y = np.sin(self.motor_angles)

        # build system
        self._build_system_matrices()
        self._compute_lqr_gains()

    def _build_system_matrices(self):
        m_total = self.m_ball + self.m_body
        denom_trans = m_total
        denom_rot = self.I_body_xx + self.m_body * self.h**2

        A = np.zeros((8,8))
        A[0,2] = 1
        A[1,3] = 1
        A[2,4] = -self.m_body * self.g * self.h / denom_trans
        A[3,5] = -self.m_body * self.g * self.h / denom_trans
        A[4,6] = 1
        A[5,7] = 1
        A[6,4] = m_total * self.g * self.h / denom_rot
        A[7,5] = m_total * self.g * self.h / denom_rot
        self.A = A

        B = np.zeros((8,3))
        B[2,:] = self.motor_to_ball_x / denom_trans
        B[3,:] = self.motor_to_ball_y / denom_trans
        B[6,:] = -self.motor_to_ball_x * self.h / denom_rot
        B[7,:] = -self.motor_to_ball_y * self.h / denom_rot
        self.B = B

    def _compute_lqr_gains(self):
        Q = np.diag([200.,200.,20.,20.,500.,500.,100.,100.])
        R = np.diag([1.,1.,1.])
        if SCIPY_AVAILABLE:
            try:
                P = solve_continuous_are(self.A, self.B, Q, R)
                self.K = np.linalg.inv(R) @ self.B.T @ P
                self._ok = True
                print("LQR K computed.")
            except Exception as e:
                print("LQR compute error:", e)
                self.K = np.zeros((3,8))
                self._ok = False
        else:
            print("scipy not available — using fallback zero K (must install scipy for proper LQR).")
            self.K = np.zeros((3,8))
            self._ok = False

    def compute_control(self, state, setpoint=None):
        if setpoint is None:
            setpoint = np.zeros(8)
        error = state - setpoint
        motor_forces = -self.K @ error  # N
        motor_torques = motor_forces * self.R_ball
        motor_torques = np.clip(motor_torques, -self.max_torque, self.max_torque)
        return motor_forces, motor_torques

# ---------------------------
# Linear Kalman Filter (simple)
# ---------------------------
class BallbotKalmanFilter:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.x = np.zeros(8)  # [x,y,vx,vy,theta_x,theta_y,wx,wy]
        self.P = np.eye(8)*0.1
        self.Q = np.diag([1e-4,1e-4,1e-2,1e-2,1e-4,1e-4,1e-3,1e-3])
        self.R_imu = np.diag([1e-3,1e-3,1e-3,1e-3])
        self.R_enc = np.diag([1e-2,1e-2,1e-2,1e-2])

    def set_dt(self, dt):
        self.dt = dt

    def predict(self):
        dt = self.dt
        F = np.eye(8)
        F[0,2] = dt
        F[1,3] = dt
        F[4,6] = dt
        F[5,7] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update_imu(self, tilt_x, tilt_y, wx, wy):
        H = np.zeros((4,8))
        H[0,4] = 1
        H[1,5] = 1
        H[2,6] = 1
        H[3,7] = 1
        z = np.array([tilt_x, tilt_y, wx, wy])
        self._update(z, H, self.R_imu)

    def update_enc(self, pos_x, pos_y, vx, vy):
        H = np.zeros((4,8))
        H[0,0] = 1
        H[1,1] = 1
        H[2,2] = 1
        H[3,3] = 1
        z = np.array([pos_x, pos_y, vx, vy])
        self._update(z, H, self.R_enc)

    def _update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

# ---------------------------
# Main Node
# ---------------------------
class BallbotAllInOneNode(Node):
    def __init__(self):
        super().__init__('ballbot_all_in_one')

        # Parameters (tune as needed)
        self.declare_parameters(namespace='',
            parameters=[
                ('serial_port', '/dev/ttyUSB0'),
                ('baudrate', 115200),
                ('control_frequency', 100.0),
                ('ticks_per_rev', 360),
                ('wheel_radius', 0.03),  # m
                ('torque_to_pwm_scale', 20.0),
                ('max_pwm', 255),
                ('max_torque', 12.0),
                ('safety_max_tilt_deg', 40.0),
                ('command_timeout_ms', 500),
            ])
        self.serial_port = self.get_parameter('serial_port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.control_frequency = float(self.get_parameter('control_frequency').value)
        self.dt = 1.0 / self.control_frequency
        self.ticks_per_rev = float(self.get_parameter('ticks_per_rev').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.torque_to_pwm_scale = float(self.get_parameter('torque_to_pwm_scale').value)
        self.max_pwm = int(self.get_parameter('max_pwm').value)
        self.max_torque = float(self.get_parameter('max_torque').value)
        self.safety_max_tilt = math.radians(float(self.get_parameter('safety_max_tilt_deg').value))
        self.command_timeout_ms = int(self.get_parameter('command_timeout_ms').value)

        # Components
        self.lqr = BallbotLQRController(max_torque=self.max_torque)
        self.kf = BallbotKalmanFilter(dt=self.dt)

        # State containers
        self.latest_imu = {'accel': np.zeros(3), 'gyro': np.zeros(3), 'quat': None}
        self.latest_enc = {'pos': np.zeros(3), 'vel': np.zeros(3)}
        self.last_command_time = time.time()
        self.emergency_stop = False

        # ROS publishers/subscribers
        self.imu_pub = self.create_publisher(Imu, '/ballbot/imu', 10)
        self.joint_pub = self.create_publisher(JointState, '/ballbot/joint_states', 10)
        self.state_pub = self.create_publisher(Float64MultiArray, '/ballbot/system_state', 10)
        self.diag_pub = self.create_publisher(String, '/ballbot/diagnostics', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Serial setup (ESP32)
        self.ser = None
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=0.1)
            time.sleep(1.0)  # allow ESP32 to boot
            self.get_logger().info(f"Serial connected to {self.serial_port} @ {self.baudrate}")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {self.serial_port}: {e}")

        # Start serial reader thread
        self.serial_lock = threading.Lock()
        self.serial_thread_running = True
        self.serial_thread = threading.Thread(target=self.serial_reader_thread, daemon=True)
        self.serial_thread.start()

        # Control timer
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        # Encoder -> ball mapping helpers (same geometry as LQR)
        self.motor_to_ball_x = self.lqr.motor_to_ball_x
        self.motor_to_ball_y = self.lqr.motor_to_ball_y

        # Diagnostics timer
        self.create_timer(1.0, self.publish_diagnostics)

        self.get_logger().info("Ballbot all-in-one node started")

    # ---------------------------
    # Serial reader: parse lines from ESP32
    # ---------------------------
    def serial_reader_thread(self):
        buf = b""
        while self.serial_thread_running:
            try:
                if self.ser is None:
                    time.sleep(0.2)
                    continue
                line = self.ser.readline()
                if not line:
                    continue
                try:
                    s = line.decode('utf-8', errors='ignore').strip()
                except Exception:
                    continue
                if not s:
                    continue
                # Example lines:
                # IMU,ax,ay,az,gx,gy,gz
                # ENC,pos1,vel1,pos2,vel2,pos3,vel3
                if s.startswith("IMU"):
                    parts = s.split(',')
                    if len(parts) >= 7:
                        ax = float(parts[1]); ay = float(parts[2]); az = float(parts[3])
                        gx = float(parts[4]); gy = float(parts[5]); gz = float(parts[6])
                        # store (convert units)
                        self.latest_imu['accel'] = np.array([ax, ay, az]) * 9.81  # g -> m/s^2
                        # gyro given in deg/s from ESP32; convert to rad/s
                        self.latest_imu['gyro'] = np.radians(np.array([gx, gy, gz]))
                        # Publish Imu message (basic)
                        imu_msg = Imu()
                        imu_msg.header.stamp = self.get_clock().now().to_msg()
                        imu_msg.header.frame_id = "imu_link"
                        # no orientation quaternion (we'll compute tilt from acc/gyro in KF)
                        # fill angular_velocity and linear_acceleration
                        imu_msg.angular_velocity.x = self.latest_imu['gyro'][0]
                        imu_msg.angular_velocity.y = self.latest_imu['gyro'][1]
                        imu_msg.angular_velocity.z = self.latest_imu['gyro'][2]
                        imu_msg.linear_acceleration.x = self.latest_imu['accel'][0]
                        imu_msg.linear_acceleration.y = self.latest_imu['accel'][1]
                        imu_msg.linear_acceleration.z = self.latest_imu['accel'][2]
                        # minimal covariance
                        imu_msg.orientation_covariance[0] = -1.0
                        self.imu_pub.publish(imu_msg)
                        # mark IMU arrival time
                        self.latest_imu['time'] = time.time()
                elif s.startswith("ENC"):
                    parts = s.split(',')
                    # Expect: ENC,pos1,vel1,pos2,vel2,pos3,vel3
                    if len(parts) >= 7:
                        pos = np.zeros(3)
                        vel = np.zeros(3)
                        try:
                            pos[0] = float(parts[1]); vel[0] = float(parts[2])
                            pos[1] = float(parts[3]); vel[1] = float(parts[4])
                            pos[2] = float(parts[5]); vel[2] = float(parts[6])
                        except Exception:
                            continue
                        # store
                        self.latest_enc['pos'] = pos
                        self.latest_enc['vel'] = vel
                        self.latest_enc['time'] = time.time()
                        # publish JointState (positions in ticks; velocities as provided)
                        js = JointState()
                        js.header.stamp = self.get_clock().now().to_msg()
                        js.name = ['motor1', 'motor2', 'motor3']
                        js.position = pos.tolist()
                        js.velocity = vel.tolist()
                        self.joint_pub.publish(js)
                elif s.startswith("STATUS"):
                    # optional: log status lines from ESP32
                    self.get_logger().debug(f"ESP32 STATUS: {s}")
                else:
                    # debug / unrecognized
                    self.get_logger().debug(f"SERIAL >> {s}")
            except Exception as e:
                self.get_logger().debug(f"Serial thread error: {e}")
                time.sleep(0.05)

    # ---------------------------
    # Utility: convert encoder ticks -> linear pos/vel of ball
    # We'll interpret encoder vel as ticks/sec. Convert to motor angular vel (rad/s),
    # then to ball velocities vx, vy using motor geometry and wheel radius.
    # ---------------------------
    def encoder_to_ball_motion(self, ticks_pos, ticks_vel):
        # ticks -> motor revolutions: revs = ticks / ticks_per_rev
        # angular velocity (rad/s) = revs/sec * 2π = ticks_vel / ticks_per_rev * 2π
        omega = (np.array(ticks_vel) / self.ticks_per_rev) * 2.0 * math.pi  # rad/s
        # ball linear velocity contribution from each motor = omega * motor_contact_radius
        # map motors -> x,y using geometry
        vx = np.dot(self.motor_to_ball_x, omega) * self.motor_radius_factor()
        vy = np.dot(self.motor_to_ball_y, omega) * self.motor_radius_factor()
        # positions (integrated) — convert ticks to radians -> linear via contact radius
        pos_rad = (np.array(ticks_pos) / self.ticks_per_rev) * 2.0 * math.pi
        pos_x = np.dot(self.motor_to_ball_x, pos_rad) * self.motor_radius_factor()
        pos_y = np.dot(self.motor_to_ball_y, pos_rad) * self.motor_radius_factor()
        return pos_x, pos_y, vx, vy

    def motor_radius_factor(self):
        # effective factor to convert motor angular to ball linear
        # if motor contacts ball directly with radius R_ball, v = omega_motor * R_ball
        # but if gearing or wheel radius differs, adjust here. Use ball radius by default.
        return self.lqr.R_ball

    # ---------------------------
    # Control: main loop called by ROS timer
    # ---------------------------
    def control_loop(self):
        try:
            # safety: serial available
            if self.ser is None:
                self.get_logger().warn("Serial not available to ESP32.")
                return

            # Timeout safety: if no command from higher-level (cmd_vel) recently, zero targets
            if (time.time() - self.last_command_time) * 1000.0 > self.command_timeout_ms:
                self.target_velocity = np.array([0.0, 0.0])
            # check for emergency tilt (KF state)
            # run KF predict
            self.kf.set_dt(self.dt)
            self.kf.predict()

            # If IMU data available: compute tilt from accel+gyro (simple)
            # We'll compute tilt_x (pitch) and tilt_y (roll) from accel (small-angle ~) and gyro integration
            if 'time' in self.latest_imu and time.time() - self.latest_imu['time'] < 0.5:
                ax, ay, az = self.latest_imu['accel']
                # compute approximate tilt from accel (safe when quasi-static)
                tilt_x = math.atan2(-ax, math.sqrt(ay*ay + az*az))  # pitch
                tilt_y = math.atan2(ay, az)  # roll
                wx = self.latest_imu['gyro'][0]
                wy = self.latest_imu['gyro'][1]
                self.kf.update_imu(tilt_x, tilt_y, wx, wy)

            # If encoder data available, convert to ball motion and update KF
            if 'time' in self.latest_enc and time.time() - self.latest_enc['time'] < 0.5:
                ticks_pos = self.latest_enc['pos']
                ticks_vel = self.latest_enc['vel']
                pos_x, pos_y, vx, vy = self.encoder_to_ball_motion(ticks_pos, ticks_vel)
                self.kf.update_enc(pos_x, pos_y, vx, vy)

            # Read fused state
            state = self.kf.x.copy()  # 8-vector

            # Safety check: tilt magnitude
            total_tilt = math.sqrt(state[4]**2 + state[5]**2)
            if total_tilt > self.safety_max_tilt:
                if not self.emergency_stop:
                    self.emergency_stop = True
                    self.get_logger().error(f"EMERGENCY STOP: tilt {math.degrees(total_tilt):.1f}° > {math.degrees(self.safety_max_tilt):.1f}°")
                    self.send_motor_command([0,0,0])
                return

            # Build setpoint from cmd_vel (we keep it small; cmd_vel handler updates target_velocity)
            setpoint = np.zeros(8)
            if hasattr(self, 'target_position'):
                setpoint[0] = getattr(self, 'target_position')[0]
                setpoint[1] = getattr(self, 'target_position')[1]
            if hasattr(self, 'target_velocity'):
                setpoint[2] = self.target_velocity[0]
                setpoint[3] = self.target_velocity[1]

            # Compute LQR control
            motor_forces, motor_torques = self.lqr.compute_control(state, setpoint)

            # Convert torques -> PWM using torque_to_pwm_scale
            pwm = motor_torques * (self.torque_to_pwm_scale)  # linear mapping; tune this constant
            pwm = np.clip(pwm, -self.max_pwm, self.max_pwm)
            pwm_int = [int(round(v)) for v in pwm.tolist()]

            # Send PWM command to ESP32
            self.send_motor_command(pwm_int)

            # Publish fused system state for debug
            msg = Float64MultiArray()
            msg.data = state.tolist()
            self.state_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            # emergency stop on error
            self.send_motor_command([0,0,0])
            self.emergency_stop = True

    def send_motor_command(self, pwm_list):
        # Format: MOTOR,p1,p2,p3\n
        if self.ser is None:
            return
        p = [int(max(-self.max_pwm, min(self.max_pwm, int(x)))) for x in pwm_list]
        cmd = f"MOTOR,{p[0]},{p[1]},{p[2]}\n"
        try:
            with self.serial_lock:
                self.ser.write(cmd.encode('utf-8'))
        except Exception as e:
            self.get_logger().debug(f"Failed to write motor command: {e}")

    # ---------------------------
    # cmd_vel callback
    # ---------------------------
    def cmd_vel_callback(self, msg: Twist):
        # convert linear x,y to target velocities (m/s)
        self.target_velocity = np.array([msg.linear.x, msg.linear.y])
        self.last_command_time = time.time()
        # clear emergency if manual override
        if np.linalg.norm(self.target_velocity) > 0.05 and self.emergency_stop:
            self.get_logger().info("Manual override - clearing emergency stop")
            self.emergency_stop = False

    # ---------------------------
    # Diagnostics publish
    # ---------------------------
    def publish_diagnostics(self):
        try:
            total_tilt = math.sqrt(self.kf.x[4]**2 + self.kf.x[5]**2)
            diag = {
                'timestamp': time.time(),
                'emergency_stop': self.emergency_stop,
                'tilt_deg': math.degrees(total_tilt),
                'state': self.kf.x.tolist()
            }
            msg = String()
            msg.data = str(diag)
            self.diag_pub.publish(msg)
        except Exception as e:
            self.get_logger().debug(f"Diag error: {e}")

    # ---------------------------
    # Clean shutdown
    # ---------------------------
    def destroy_node(self):
        self.serial_thread_running = False
        try:
            if self.ser:
                with self.serial_lock:
                    # tell ESP32 to stop motors
                    self.ser.write(b"MOTOR,0,0,0\n")
                time.sleep(0.05)
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BallbotAllInOneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
