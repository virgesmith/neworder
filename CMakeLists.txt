# This is a work-in-progress...

cmake_minimum_required(VERSION 3.9)
project(neworder)
#set(CMAKE_BUILD_TYPE Release)

# TODO: build configs
# debug / release
# serial / MPI
# linux / windows / osx
# icpc/clang?

set(CMAKE_CXX_COMPILER g++)
message("COMPILER=" ${CMAKE_CXX_COMPILER})

# I don't like this one bit - if the final string argument is omitted you get an extremely unhelpful "set given invalid arguments for CACHE mode" error
# The cached value also persists so you need to explicitly set it back to the default 
set(PYVER 3 CACHE STRING "python version")


execute_process(COMMAND bash "-c" "python${PYVER}-config --cflags | sed 's/-Wstrict-prototypes//g' | sed 's/-O3//g'" OUTPUT_VARIABLE PY_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
message("PY_CXXFLAGS=" ${PY_CXXFLAGS})

execute_process(COMMAND python3 -c "import numpy;print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
message("NUMPY_INC=" ${NUMPY_INC})

execute_process(COMMAND python3-config --ldflags OUTPUT_VARIABLE PY_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
message("PY_LDFLAGS=" ${PY_LDFLAGS})

set(CMAKE_CXX_FLAGS "${PY_CXXFLAGS} -I ${NUMPY_INC} -Werror -Wno-error=deprecated-declarations -fPIC -std=c++14 -pedantic -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
#set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O2")
message("CMAKE_CXX_FLAGS=" ${CMAKE_CXX_FLAGS})

set(CMAKE_SHARED_LINKER_FLAGS "${PY_LDFLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS=" ${CMAKE_SHARED_LINKER_FLAGS})

# this 
include_directories(src/lib, src/include)

file(GLOB LIB_SRC "src/lib/*.cpp")

add_library(neworder SHARED ${LIB_SRC})
target_link_libraries(neworder)
