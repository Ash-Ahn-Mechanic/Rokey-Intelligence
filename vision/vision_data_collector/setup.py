from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vision_data_collector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'rqt_plugin'), glob('rqt_plugin/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahn',
    maintainer_email='domcove9@gmail.com',
    description='YOLO training data collector for TurtleBot4 + OAK-D',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mission_node = vision_data_collector.mission_node:main',
            'camera_node = vision_data_collector.camera_node:main',
        ],
    },
)
