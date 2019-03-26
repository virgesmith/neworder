
#pragma once

#include "numpy.h"

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

namespace no {

// Uniform random [0,1) fed from the environment's PRNG stream
np::array ustream(size_t n);

// TODO use overloads not different names

// single-prob hazard 
np::array hazard(double prob, size_t n);

// vector hazard 
np::array hazard(const np::array& prob);

// compute stopping times 
np::array stopping(double prob, size_t n);

// vector stopping 
np::array stopping(const np::array& prob);

// multiple-arrival (0+) process (requires that final hazard rate is zero)
np::array arrivals(const np::array& lambda_t, double dt, double gap, size_t n);


// compute arrival times given a nonhomogeneous Poisson process specified by lambd
np::array first_arrival(const np::array& lambda_t, double dt, size_t n, double minval);

// as above with minval defaulted to 0.0
inline np::array first_arrival(const np::array& lambda_t, double dt, size_t n)
{
  return first_arrival(lambda_t, dt, n, 0.0);
}

// given an array of arrival times at one state, sample times of arrival of subsequent event optionally with a minumum separation minsep
// relative = true means lambda_t is relative to the stating point *plus minsep*
np::array next_arrival(const np::array& startingpoints, const np::array& lambda_t, double dt, bool relative, double minsep);

// as above with relative defaulted to false and minval defaulted to 0.0
inline np::array next_arrival(const np::array& startingpoints, const np::array& lambda_t, double dt)
{
  return next_arrival(startingpoints, lambda_t, dt, false, 0.0);
}

// as above with minval defaulted to 0.0
inline np::array next_arrival(const np::array& startingpoints, const np::array& lambda_t, double dt, bool relative)
{
  return next_arrival(startingpoints, lambda_t, dt, relative, 0.0);
}

}