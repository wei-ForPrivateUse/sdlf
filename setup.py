import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'pylib'))

from setuptools import setup, find_packages
from pylib.sdlf.train import MAJOR_VERSION, MINOR_VERSION

packages = find_packages(where='pylib')
setup(
    name='sdlf',
    version=f'{MAJOR_VERSION}.{MINOR_VERSION}',
    description='Simple Deep Learning Framework',
    author='Yufei Wei',
    author_email='931995113@qq.com',
    url='https://github.com/wei-ForPrivateUse/sdlf.git',
    license='MIT',
    packages=packages,
    package_dir={'': 'pylib'},
)
