from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='It Classifies Actions',
    author='britohyago',
    license='',
    install_requires=['albumentations', 'opencv-python', 'mediapipe', 'pandas', 'pytube', 'moviepy']
)