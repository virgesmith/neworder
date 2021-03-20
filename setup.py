#!/usr/bin/env python3

import os
import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile
import sys
import setuptools

# see https://github.com/pybind/python_example

def readme():
  with open('README.md') as f:
    return f.read()

def version():
  """ The file VERSION in the project root is now the single source of version info """
  with open("VERSION") as fd:
    return fd.readline().rstrip()

def list_files(dirs, exts, exclude=[]):
  files = []
  if isinstance(exclude, str):
    exclude = [exclude]
  for directory in dirs:
    for ext in exts:
      files.extend(glob.glob(os.path.join(directory, "*." + ext)))
  [f in files and files.remove(f) for f in exclude]
  return files

def cxxflags(platform):

  if platform == "unix":
    return [
      "-Wall",
      "-pedantic",
      "-pthread",
      "-Wsign-compare",
      "-fstack-protector-strong",
      "-Wformat",
      "-Werror=format-security",
      "-Wdate-time",
      "-fPIC",
      "-std=c++17",
      "-fvisibility=hidden"
    ]
  elif platform == "msvc":
    return ['/std:c++17', '/EHsc']
  else:
    return []

def ldflags(_platform):
  return []

def defines(platform):
  return [
    ("NEWORDER_VERSION", version())
  ]

ext_modules = [
  Pybind11Extension(
    'neworder',
    sources=list_files(['src'], ["cpp"]),
    define_macros=[("NEWORDER_VERSION", "0.3.0")],
    depends=["setup.py", "VERSION"] + list_files(["src"], ["h"]),
    cxx_std=17
  ),
]

# class BuildExt(build_ext):
#   """A custom build extension for adding compiler-specific options."""
#   # c_opts = {
#   #     'msvc': ['/EHsc'],
#   #     'unix': [],
#   # }
#   # l_opts = {
#   #     'msvc': [],
#   #     'unix': [],
#   # }

#   # if sys.platform == 'darwin':
#   #   darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
#   #   c_opts['unix'] += darwin_opts
#   #   l_opts['unix'] += darwin_opts

#   def build_extensions(self):
#     ct = self.compiler.compiler_type

ParallelCompile(default=4).install()

setup(
  name='neworder',
  version=version(),
  author='Andrew P Smith',
  author_email='a.p.smith@leeds.ac.uk',
  url='https://neworder.readthedocs.io',
  description='A dynamic microsimulation framework',
  long_description = readme(),
  long_description_content_type="text/markdown",
  ext_modules=ext_modules,
  install_requires=['numpy>=1.19.1', 'pandas>=1.0.5'],
  setup_requires=['pybind11>=2.5.0', 'pytest-runner'],
  tests_require=['pytest', 'mpi4py>=3.0.3'],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  zip_safe=False,
)
