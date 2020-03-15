
#pragma once

#include "ArrayHelpers.h"

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

void test_mc();

namespace no {

class MonteCarlo
{
public:
  MonteCarlo(int rank, int size, bool indep);

  bool indep() const;

  int64_t seed() const;

  void reset();

  // Uniform random [0,1) fed from the environment's PRNG stream
  py::array ustream(size_t n);

  // single-prob hazard 
  py::array hazard(double prob, size_t n);

  // vector hazard 
  py::array hazard(const py::array& prob);

  // compute stopping times 
  py::array stopping(double prob, size_t n);

  // vector stopping 
  py::array stopping(const py::array& prob);

  // multiple-arrival (0+) process (requires that final hazard rate is zero)
  py::array arrivals(const py::array& lambda_t, double dt, double gap, size_t n);

  // compute arrival times given a nonhomogeneous Poisson process specified by lambd
  py::array first_arrival(const py::array& lambda_t, double dt, size_t n, double minval);

  // given an array of arrival times at one state, sample times of arrival of subsequent event optionally with a minumum separation minsep
  // relative = true means lambda_t is relative to the stating point *plus minsep*
  py::array next_arrival(const py::array& startingpoints, const py::array& lambda_t, double dt, bool relative, double minsep);

private:
  bool m_indep;
  int64_t m_seed;
  std::mt19937 m_prng;
};


}