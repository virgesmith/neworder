
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
std::vector<double> stopping(double prob, size_t n);

// vector stopping 
std::vector<double> stopping_v(const std::vector<double>& prob);

// vector stopping for non-homogeneous poisson process
std::vector<double> stopping_nhpp(const std::vector<double>& lambda_t, size_t n);

}