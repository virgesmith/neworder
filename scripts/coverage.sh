#!/bin/bash

# deprecated - see coverage-ci.sh

# clean: ./setup.py clean doesnt work, so...
touch VERSION

# build with instrumentation
CFLAGS="-fprofile-arcs -ftest-coverage" ./setup.py install
pytest

# m=demangle, r="relative" (ignores <> includes (but then lcov doesn't?))
gcov -abcjmru build/temp.linux-x86_64-3.8/src/*.gcda
lcov --capture --directory . --output-file coverage.info --exclude "/usr/include/*" --exclude "*/pybind11/*"
#lcov -r coverage.info -o coverage-filtered.info "/usr/include/*"

# process raw output into html
mkdir -p test-coverage/
genhtml coverage.info --output-directory test-coverage

# clean up
rm -f *.gcov
rm coverage.info
