from setuptools import find_packages, setup
from glob import glob

package_name = 'drl_px4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='icfoss22',
    maintainer_email='icfoss22@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
    'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'uav_env = drl_px4.uav_env:main',
            'drone_env1 = drl_px4.drone_env1:main',
            'train_uav = drl_px4.train_uav:main',
            'sac_per = drl_px4.sac_per:main',
            'test_uav = drl_px4.test_uav:main',

        ],
    },
)
