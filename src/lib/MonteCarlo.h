
#pragma once

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

void test_mc();

namespace no {

class NEWORDER_EXPORT MonteCarlo
{
public:

  // some basic seeding strategies
  static int64_t deterministic_independent_stream(int r);

  static int64_t deterministic_identical_stream(int);

  static int64_t nondeterministic_stream(int);

  // constructs given a seed
  MonteCarlo(int64_t seed);

  int64_t seed() const;

  void reset();

  // used by python __repr__
  std::string repr() const;

  // Uniform random [0,1) fed from the environment's PRNG stream
  py::array ustream(py::ssize_t n);

  // single-prob hazard 
  py::array hazard(double prob, py::ssize_t n);

  // vector hazard 
  py::array hazard(const py::array& prob);

  // compute stopping times 
  py::array stopping(double prob, py::ssize_t n);

  // vector stopping 
  py::array stopping(const py::array& prob);

  // multiple-arrival (0+) process (requires that final hazard rate is zero)
  py::array arrivals(const py::array& lambda_t, double dt, double gap, size_t n);

  // compute arrival times given a nonhomogeneous Poisson process specified by lambd
  py::array first_arrival(const py::array& lambda_t, double dt, size_t n, double minval);

  // given an array of arrival times at one state, sample times of arrival of subsequent event optionally with a minumum separation minsep
  // relative = true means lambda_t is relative to the starting point *plus minsep*
  py::array next_arrival(const py::array& startingpoints, const py::array& lambda_t, double dt, bool relative, double minsep);

private:

  // Use this over std::uniform_real_distribution as can make C++ and rust implementations produce identical streams
  double u01();

  int64_t m_seed;
  std::mt19937 m_prng;
};


}