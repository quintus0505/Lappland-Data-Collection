#!/usr/bin/env python3
import os

from os.path import join, dirname, realpath
from setuptools import find_packages, setup
from distutils.command.build import build as DistutilsBuild


with open(join("mujoco_py", "version.py")) as version_file:
    exec(version_file.read())


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dirname(realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'mujoco_py' or p.startswith('mujoco_py.')


class Build(DistutilsBuild):
    def run(self):
        os.environ['MUJOCO_PY_FORCE_REBUILD'] = 'True'
        os.environ['MUJOCO_PY_SKIP_ACTIVATE'] = 'True'
        import mujoco_py  # noqa: force build
        DistutilsBuild.run(self)


setup(
    name='mujoco-py',
    version=__version__,  # noqa
    author='OpenAI Robotics Team',
    author_email='robotics@openai.com',
    url='https://github.com/openai/mujoco-py',
    packages=packages,
    include_package_data=True,
    cmdclass={'build': Build},
    package_dir={'mujoco_py': 'mujoco_py'},
    package_data={'mujoco_py': ['generated/*.so']},
    install_requires=[
	'glfw>=1.4.0',
	'numpy>=1.11'
	'Cython>=0.27.2',
	'imageio>=2.1.2',
	'cffi>=1.10',
	'fasteners~=0.15',
	'Cython',],
    tests_require=[
	'imagehash>=3.4',
	'ipdb',
	'Pillow>=4.0.0',
	'pycparser>=2.17.0',
	'pytest>=3.0.5',
	'pytest-instafail==0.3.0',
	'sphinx',
	'sphinx_rtd_theme',
	'numpydoc',],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
