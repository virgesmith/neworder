#!/bin/bash

# Gets boost and builds the python libs

if [ ! -f ./boost_1_67_0.tar.bz2 ]; then
  wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.bz2
fi

tar jxf boost_1_67_0.tar.bz2

cd boost_1_67_0
./bootstrap.sh --prefix=/usr/local --with-libraries=python --with-python=$(which python3)
./b2 cxxflags=-DBOOST_NO_AUTO_PTR install
