cmake_minimum_required(VERSION 3.10)
project(neworder)
#set(CMAKE_BUILD_TYPE Release)

# TODO: build configs
# debug / release
# serial / MPI
# linux / windows / osx
# icpc/clang?

set(CMAKE_CXX_COMPILER g++)
message("COMPILER=" ${CMAKE_CXX_COMPILER})

# TODO configurable PYVER: set(PYVER 3)
set(BOOST_PYTHON_LIB boost_python3)
set(BOOST_NUMPY_LIB boost_numpy3-py36)

# TODO override for custom boost lib/location
#set(BOOST_EXTRA_CXXFLAGS)
#set(BOOST_EXTRA_LDFLAGS=

execute_process(COMMAND bash "-c" "python3-config --cflags | sed 's/-Wstrict-prototypes//g' | sed 's/-O3//g'" OUTPUT_VARIABLE PY_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
message("PY_CXXFLAGS=" ${PY_CXXFLAGS})

execute_process(COMMAND python3 -c "import numpy;print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
message("NUMPY_INC=" ${NUMPY_INC})

execute_process(COMMAND python3-config --ldflags OUTPUT_VARIABLE PY_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
message("PY_LDFLAGS=" ${PY_LDFLAGS})

set(CMAKE_CXX_FLAGS "${PY_CXXFLAGS} -I ${NUMPY_INC} -Werror -Wno-error=deprecated-declarations -fPIC -std=c++14 -pedantic -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DBOOST_NO_AUTO_PTR ${BOOST_EXTRA_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O2")

set(CMAKE_SHARED_LINKER_FLAGS "${PY_LDFLAGS} ${BOOST_EXTRA_LDFLAGS}")

include_directories(src/lib, src/include)

file(GLOB LIB_SRC "src/lib/*.cpp")

add_library(neworder SHARED ${LIB_SRC})
