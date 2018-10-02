# arc3.sh

# make overrides for arc3

echo make BOOST_EXTRA_CXXFLAGS=-I../../../boost_1_67_0/ \
          BOOST_EXTRA_LDFLAGS=-L../../src/lib \
          BOOST_PYTHON_LIB=boost_python36 \
          BOOST_NUMPY_LIB=boost_numpy36 \
          $@