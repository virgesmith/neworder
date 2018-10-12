
#pragma once

#include "numpy.h"

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

namespace neworder {

// Uniform random [0,1) fed from the environment's PRNG stream
np::ndarray ustream(size_t n);

// TODO use overloads not different names

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

// multiple-arrival (0+) process (requires that final hazard rate is zero)
np::ndarray arrivals(const np::ndarray& lambda_t, double dt, double gap, size_t n);

np::ndarray first_arrival(const np::ndarray& lambda_t, double dt, size_t n, double minval = 0.0);

// given an array of arrival times at one state, sample times of arrival of subsequent event
// TODO flag to mark lambda_t as abs/rel to startingpoint
np::ndarray next_arrival(const np::ndarray& startingpoints, const np::ndarray& lambda_t, double dt, double gap);

}