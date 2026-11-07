cmake_minimum_required(VERSION 3.18...3.31)

project(neworder LANGUAGES CXX)

# NumPy is only needed for the numpy/random/bitgen.h header
find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(nanobind CONFIG REQUIRED)

# Module_docstr.cpp is deliberately not listed: it is #included inside the module definition in Module.cpp
#
# NOMINSIZE: nanobind builds for size by default, which would also apply to the sampling loops in this module
# NB_DOMAIN: isolates the global leak warning flag from any other nanobind extensions in the process
nanobind_add_module(
  _neworder_core
  NOMINSIZE
  FREE_THREADED
  NB_DOMAIN neworder
  src/DataFrame.cpp
  src/Log.cpp
  src/Model.cpp
  src/Module.cpp
  src/MonteCarlo.cpp
  src/NPArray.cpp
  src/SplitMix64.cpp
  src/Timeline.cpp
)

target_compile_features(_neworder_core PRIVATE cxx_std_20)
target_include_directories(_neworder_core PRIVATE ${Python_NumPy_INCLUDE_DIRS})

install(TARGETS _neworder_core LIBRARY DESTINATION .)
