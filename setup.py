from setuptools import find_packages, setup

package_name = 'ball_tracking_machine_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditi',
    maintainer_email='lagisettyaditi@gmail.com',
    description='Stereo ball tracking and landing-point prediction with a ZED 2i.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = ball_tracking_machine_vision.tracker_node:main',
        ],
    },
)
