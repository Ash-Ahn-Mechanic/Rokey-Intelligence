from setuptools import find_packages, setup

package_name = 'webcam_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', [
            'resource/best_turtlebot.pt',
            'resource/best_webcam.pt',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junsu',
    maintainer_email='junsoo122@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cv_cam_pub  = webcam_detect.cv_cam_pub:main',
            'yolo_webcam = webcam_detect.yolo_webcam:main',
            'fov_nav_node = webcam_detect.fov_nav_node:main',
            'turtlebot_state = webcam_detect.turtlebot_state:main',
            'turtlebot_track = webcam_detect.turtlebot_track:main',
        ],
    },
)
