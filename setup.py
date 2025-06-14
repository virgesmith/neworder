#!/usr/bin/env python3

import glob
import os

import numpy
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension
from setuptools import setup

# see https://github.com/pybind/python_example


def readme():
    with open("README.md") as f:
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


ext_modules = [
    Pybind11Extension(
        "_neworder_core",
        sources=list_files(["src"], ["cpp"]),
        include_dirs=[numpy.get_include()],
        depends=["setup.py", "neworder/__init__.py"] + list_files(["src"], ["h"]),
        cxx_std=20,
    ),
]

ParallelCompile().install()

setup(
    name="neworder",
    packages=["neworder"],
    package_data={"neworder": ["py.typed", "*.pyi"]},
    ext_modules=ext_modules,
    zip_safe=False,
)
