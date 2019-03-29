# this can interfere with virtualenv?
##!/usr/bin/env python3

import os
import glob
import warnings
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import pybind11

def readme():
  with open('README.md') as f:
    return f.read()

def version():
  import neworder
  return neworder.__version__

# def list_files(dirs, exts, exclude=[]):
#   files = []
#   if isinstance(exclude, str):
#     exclude = [exclude]
#   for directory in dirs:
#     for ext in exts:
#       files.extend(glob.glob(os.path.join(directory, "*." + ext)))
#   [f in files and files.remove(f) for f in exclude]
#   return files

# define_macros = [('MAJOR_VERSION', version().split(".")[0]),
#                   ('MINOR_VERSION', version().split(".")[1]),
#                   ('PATCH_VERSION', version().split(".")[2]),
#                   ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
#                 ]
# include_path = ["src/include", "src/lib", pybind11.get_include()]
# base_compiler_args = ["-O2", "-Werror", "-Wno-error=deprecated-declarations", "-fPIC", "-std=c++14", "-pedantic"]

# # TODO linker settings from:
# # python3.6-config --ldflags
# # -L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu -L/usr/lib -lpython3.6m -lpthread -ldl  -lutil -lm  -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions

# library_dirs = ["/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu", "/usr/lib"] 
# libraries = ["python3.6m", "pthread", "dl", "util", "m"] 
# extra_link_args = ["-Xlinker", "-export-dynamic", "-Wl,-O1", "-Wl,-Bsymbolic-functions"]

# TODO MPI build?

# neworderlib = Extension(
#   'libneworder',
#   define_macros = define_macros,
#   extra_compile_args = ['-shared'] + base_compiler_args,
#   sources = list_files(["src/lib"], ["cpp"]),
#   depends = list_files(["src/include", "src/lib"], ["h"]),
#   include_dirs = include_path,
#   library_dirs = library_dirs,
#   libraries = libraries,
#   extra_link_args = extra_link_args
# )

# neworderbin = Extension(
#   'neworder', 
#   define_macros = define_macros,
#   extra_compile_args = base_compiler_args,
#   sources = list_files(["src/bin"], ["cpp"], exclude="src/bin/main_mpi.cpp"),
#   depends = list_files(["src/include", "src/lib", "src/bin"], ["h"]),
#   include_dirs = include_path,
#   library_dirs = library_dirs,
#   libraries = libraries,
#   extra_link_args = extra_link_args
# )

# newordertest = Extension(
#   'newordertest', 
#   define_macros = define_macros,
#   extra_compile_args = base_compiler_args,
#   sources = list_files(["src/test"], ["cpp"], exclude="src/test/main_mpi.cpp"),
#   depends = list_files(["src/include", "src/lib", "src/test"], ["h"]),
#   include_dirs = include_path,
#   library_dirs = library_dirs,
#   libraries = libraries,
#   extra_link_args = extra_link_args
# )

# use make for now
dummy = Extension("dummy", [])

class BuildBinaries(build_ext):
  def run(self):
    os.system("make -j4")
    os.system("make -f MPI.mk -j4")

# TODO run test binary

import unittest
def test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('tests', pattern='test_wrapper.py')
  return test_suite

setup(
  name = 'neworder',
  version = version(),
  description = 'Parallel Generic Microsimulation Framework',
  author = 'Andrew P Smith',
  author_email = 'a.p.smith@leeds.ac.uk',
  url = 'http://github.com/virgesmith/neworder',
  long_description = readme(),
  long_description_content_type="text/markdown",
  cmdclass = {'build_ext': BuildBinaries},
  ext_modules = [dummy],
  #ext_scripts = [neworderbin, newordertest],
  setup_requires=['numpy', 'pybind11'],
  install_requires=[],
  test_suite='setup.test_suite'
)
