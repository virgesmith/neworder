#!/usr/bin/env python3

import os
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
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

class get_pybind_include(object):
  """Helper class to determine the pybind11 include path

  The purpose of this class is to postpone importing pybind11
  until it is actually installed, so that the ``get_include()``
  method can be invoked. """

  def __str__(self):
    import pybind11
    return pybind11.get_include()


ext_modules = [
  Extension(
    'neworder',
    # Sort input source files to ensure bit-for-bit reproducible builds
    # (https://github.com/pybind/python_example/pull/53)
    sources=list_files(['src/lib'], ["cpp"]),
    #define_macros=defines(),
    include_dirs=[
      "./src/include",
      "./src/lib",
      # Path to pybind11 headers
      get_pybind_include(),
    ],
    #extra_compile_args=cxxflags(),
    depends=["VERSION"] + list_files(["src/include", "src/lib"], ["h"]),
    language='c++'
  ),
]

class BuildExt(build_ext):
  """A custom build extension for adding compiler-specific options."""
  # c_opts = {
  #     'msvc': ['/EHsc'],
  #     'unix': [],
  # }
  # l_opts = {
  #     'msvc': [],
  #     'unix': [],
  # }

  # if sys.platform == 'darwin':
  #   darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
  #   c_opts['unix'] += darwin_opts
  #   l_opts['unix'] += darwin_opts

  def build_extensions(self):
    ct = self.compiler.compiler_type

    # opts = self.c_opts.get(ct, [])
    # link_opts = self.l_opts.get(ct, [])
    # if ct == 'unix':
    #   if True: #has_flag(self.compiler, '-fvisibility=hidden'):
    #     opts.append('-fvisibility=hidden')

    for ext in self.extensions:
      print(self.distribution.get_version())
      ext.define_macros = defines(ct) #[('NEWORDER_VERSION', self.distribution.get_version())]
      ext.extra_compile_args = cxxflags(ct)
      ext.extra_link_args = ldflags(ct)
    
    build_ext.build_extensions(self)

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
  install_requires=['numpy>=1.19.2', 'pandas>=1.1.2'],
  setup_requires=['pybind11>=2.5.0', 'pytest-runner'],
  tests_require=['pytest', 'mpi4py>=3.0.3'],    
  cmdclass={'build_ext': BuildExt},
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],    
  zip_safe=False,
)
