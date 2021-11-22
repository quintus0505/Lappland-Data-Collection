import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='we_envs',
    version='0.1.1',
    packages=find_packages(),
    description='environments simulated in MuJoCo',
    url='git@git.lug.ustc.edu.cn:fhln/we_ur5epush-v2.git',
    author='USTC fhln',
    install_requires=[
        'termcolor',
    ],
)
