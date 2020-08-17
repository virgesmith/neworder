#!/usr/bin/env python3

import os
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# see https://github.com/pybind/python_example


def version():
  """ The file VERSION in the project root is now the single source of version info """
  with open("VERSION") as fd:
    return fd.readline().strip("\n")

def list_files(dirs, exts, exclude=[]):
  files = []
  if isinstance(exclude, str):
    exclude = [exclude]
  for directory in dirs:
    for ext in exts:
      files.extend(glob.glob(os.path.join(directory, "*." + ext)))
  [f in files and files.remove(f) for f in exclude]
  return files

def cxxflags():

  return [
    "-pthread", 
    "-Wno-unused-result", 
    "-Wsign-compare", 
    "-fstack-protector-strong", 
    "-Wformat", 
    "-Werror=format-security", 
    "-Wdate-time", 
    "-fPIC", 
    "-std=c++17", 
    "-fvisibility=hidden"
  ]

#def ld_flags():

def defines():
  v = version().split(".")
  return [ 
    ("NEWORDER_VERSION_MAJOR", v[0]),
    ("NEWORDER_VERSION_MINOR", v[1]),
    ("NEWORDER_VERSION_PATCH", v[2])
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
    define_macros=defines(),
    include_dirs=[
      "./src/include",
      "./src/lib",
      # Path to pybind11 headers
      get_pybind_include(),
    ],
    extra_compile_args=cxxflags(),
    depends=list_files(["src/include", "src/lib"], ["h"]),
    language='c++'
  ),
]


# # # cf http://bugs.python.org/issue26689
# def has_flag(compiler, flagname):
#   """Return a boolean indicating whether a flag name is supported on
#   the specified compiler.
#   """
#   import tempfile
#   import os
#   with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
#       f.write('int main (int argc, char **argv) { return 0; }')
#       fname = f.name
#   try:
#       compiler.compile([fname], extra_postargs=[flagname])
#   except setuptools.distutils.errors.CompileError:
#       return False
#   finally:
#       try:
#           os.remove(fname)
#       except OSError:
#           pass
#   return True




class BuildExt(build_ext):
  """A custom build extension for adding compiler-specific options."""
  c_opts = {
      'msvc': ['/EHsc'],
      'unix': [],
  }
  l_opts = {
      'msvc': [],
      'unix': [],
  }

  if sys.platform == 'darwin':
    darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    c_opts['unix'] += darwin_opts
    l_opts['unix'] += darwin_opts

  def build_extensions(self):
    ct = self.compiler.compiler_type
    opts = self.c_opts.get(ct, [])
    link_opts = self.l_opts.get(ct, [])
    # if ct == 'unix':
    #   if True: #has_flag(self.compiler, '-fvisibility=hidden'):
    #     opts.append('-fvisibility=hidden')

    # for ext in self.extensions:
    #   ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
    #   ext.extra_compile_args = opts
    #   ext.extra_link_args = link_opts
    build_ext.build_extensions(self)

setup(
    name='neworder',
    version=version(),
    author='Andrew P Smith',
    author_email='a.p.smith@leeds.ac.uk',
    url='https://github.com/virgesmith/neworder',
    description='A microsimulation framework',
    long_description='',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
