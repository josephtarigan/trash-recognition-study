# gcloud setup file
# josepht 10 aug 2019
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.2.4',
                     'h5py==2.9.0',
                     'matplotlib==3.0.3',
                     'tensorflow-gpu==1.14.0',
                     'Pillow==6.2.1']

setup(
    name='trash-recognition-trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trash recogntion trainer application'
)