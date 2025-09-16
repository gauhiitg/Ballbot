from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ballbot_control',
            executable='ballbot_all_in_one',
            name='ballbot_controller',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',   # adjust to your ESP32 port
                'baudrate': 115200,
                'control_frequency': 100.0,
                'ticks_per_rev': 360,
                'wheel_radius': 0.03,
                'torque_to_pwm_scale': 20.0,
                'max_pwm': 255,
                'max_torque': 12.0,
                'safety_max_tilt_deg': 40.0,
                'command_timeout_ms': 500,
            }]
        ),
    ])
