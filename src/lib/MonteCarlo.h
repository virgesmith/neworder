
#pragma once

#include "numpy.h"

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

namespace neworder {

// Uniform random [0,1) fed from the environment's PRNG stream
np::ndarray ustream(size_t n);

// single-prob hazard 
np::ndarray hazard(double prob, size_t n);

// vector hazard 
np::ndarray hazard_v(const np::ndarray& prob);

// compute stopping times 
np::ndarray stopping(double prob, size_t n);

// vector stopping 
np::ndarray stopping_v(const np::ndarray& prob);

// vector stopping for non-homogeneous poisson process (i.e. time-dependent hazard rate)
np::ndarray stopping_nhpp(const np::ndarray& lambda_t, double dt, size_t n);

}