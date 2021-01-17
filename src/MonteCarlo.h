
#pragma once

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <vector>
#include <random>
#include <functional>
#include <cstddef>
#include <cstdint>

void test_mc();

namespace no {

class NEWORDER_EXPORT MonteCarlo
{
public:

  // some basic seeding strategies
  static int32_t deterministic_independent_stream(int r);

  static int32_t deterministic_identical_stream(int);

  static int32_t nondeterministic_stream(int);

  // constructs given a seed
  MonteCarlo(int32_t seed);

  int32_t seed() const;

  void reset();

  // used by python __repr__
  std::string repr() const;

  // Uniform random [0,1) fed from the environment's PRNG stream
  py::array_t<double> ustream(py::ssize_t n);

  // returns a hash of the internal state. (The raw string representation's length varies, making MPI comms trickier)
  size_t state() const;

  // raw unsigned 64-bit ints, can be used to (un)deterministically seed another generator (e.g. np.random)
  uint64_t raw();

  // randomly sample categories with weights
  py::array_t<int64_t> sample(py::ssize_t n, const py::array_t<double>& cat_weights);

  // single-prob hazard
  py::array_t<double> hazard(double prob, py::ssize_t n);

  // vector hazard
  py::array_t<double> hazard(const py::array_t<double>& prob);

  // compute stopping times
  py::array_t<double> stopping(double prob, py::ssize_t n);

  // vector stopping
  py::array_t<double> stopping(const py::array_t<double>& prob);

  // multiple-arrival (0+) process (requires that final hazard rate is zero)
  py::array_t<double> arrivals(const py::array_t<double>& lambda_t, double dt, py::ssize_t n, double gap);

  // compute arrival times given a nonhomogeneous Poisson process specified by lambda
  py::array_t<double> first_arrival(const py::array_t<double>& lambda_t, double dt, py::ssize_t n, double minval);

  // given an array of arrival times at one state, sample times of arrival of subsequent event optionally with a minumum separation minsep
  // relative = true means lambda_t is relative to the starting point *plus minsep*
  py::array_t<double> next_arrival(const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt, bool relative, double minsep);

  // returns sampled arrival counts (within time interval dt) for each intensity lambda
  py::array_t<int64_t> counts(const py::array_t<double>& lambda, double dt);

private:

  // Use this over std::uniform_real_distribution as can make C++ and rust implementations produce identical streams
  double u01();

  int32_t m_seed;
  std::mt19937 m_prng;
};


// helper functions

// construct cumulative from probabilities
std::vector<double> cumulative(const double* p, size_t n);

// get index of largest value of cumprob smaller than x
size_t interp(const std::vector<double>& cumprob, double x);


}