#!/usr/bin/env python3

import os
import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile

# see https://github.com/pybind/python_example


def readme():
  with open('README.md') as f:
    return f.read()


def list_files(dirs, exts, exclude=[]):
  files = []
  if isinstance(exclude, str):
    exclude = [exclude]
  for directory in dirs:
    for ext in exts:
      files.extend(glob.glob(os.path.join(directory, "*." + ext)))
  [f in files and files.remove(f) for f in exclude]
  return files

# def cxxflags(platform):
#   if platform == "unix":
#     return [
#       "-Wall",
#       "-pedantic",
#       "-pthread",
#       "-Wsign-compare",
#       "-fstack-protector-strong",
#       "-Wformat",
#       "-Werror=format-security",
#       "-Wdate-time",
#       "-fPIC",
#       "-std=c++17",
#       "-fvisibility=hidden"
#     ]
#   elif platform == "msvc":
#     return ['/EHsc']
#   else:
#     return []

# def ldflags(_platform):
#   return []


ext_modules = [
  Pybind11Extension(
    '_neworder_core',
    sources=list_files(['src'], ["cpp"]),
    depends=["setup.py", "neworder/__init__.py"] + list_files(["src"], ["h"]),
    cxx_std=17
  ),
]

ParallelCompile().install()

setup(
  name='neworder',
  # version set from __init__.py via setup.cfg,
  author='Andrew P Smith',
  author_email='a.p.smith@leeds.ac.uk',
  url='https://neworder.readthedocs.io',
  description='A dynamic microsimulation framework',
  long_description=readme(),
  long_description_content_type="text/markdown",
  packages=["neworder"],
  package_data={"neworder": ["py.typed", "*.pyi"]},
  ext_modules=ext_modules,
  install_requires=['numpy>=1.19.1', 'pandas>=1.0.5', 'scipy'],
  setup_requires=['pybind11>=2.5.0', 'pytest-runner'],
  tests_require=['pytest', 'mpi4py>=3.0.3'],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  zip_safe=False,
)
