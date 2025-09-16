from setuptools import setup

package_name = 'ballbot_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    py_modules=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['ballbot_control/launch/ballbot_all.launch.py']),
    ],
    install_requires=['setuptools', 'numpy', 'pyserial', 'scipy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Ballbot ROS2 control with IMU, encoders, KF, LQR',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ballbot_all_in_one = ballbot_control.nodes.ballbot_all_in_one_ros2:main',
        ],
    },
)
