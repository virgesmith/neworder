
#pragma once

#include "ArrayHelpers.h"
#include "NewOrder.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

void test_mc();

namespace no {

class NEWORDER_EXPORT MonteCarlo {
public:
  // some basic seeding strategies
  static int32_t deterministic_independent_stream() noexcept;

  static int32_t deterministic_identical_stream() noexcept;

  static int32_t nondeterministic_stream();

  // constructs given a seeder callable, invoked on construction and on each reset()
  MonteCarlo(std::function<int32_t()> seeder) noexcept;

  void init_bitgen(nb::capsule capsule);

  int32_t seed() const noexcept;

  void reset();

  // used by python __repr__
  std::string repr() const noexcept;

  // Uniform random [0,1) fed from the environment's PRNG stream
  NpArray<double> ustream(Py_ssize_t n);

  // returns a hash of the internal state. (The raw string representation's length varies, making MPI comms trickier)
  size_t state() const noexcept;

  // raw unsigned 64-bit ints, can be used to (un)deterministically seed another generator (e.g. np.random)
  uint64_t raw() noexcept;

  // randomly sample categories with weights
  NpArray<int64_t> sample(Py_ssize_t n, const ArrayLike<double>& cat_weights);

  // single-prob hazard
  NpArray<double> hazard(double prob, Py_ssize_t n);

  // vector hazard
  NpArray<double> hazard(const ArrayLike<double>& prob);

  // compute stopping times
  NpArray<double> stopping(double prob, Py_ssize_t n);

  // vector stopping
  NpArray<double> stopping(const ArrayLike<double>& prob);

  // multiple-arrival (0+) process (requires that final hazard rate is zero)
  NpArray<double> arrivals(const ArrayLike<double>& lambda_t, double dt, Py_ssize_t n, double gap);

  // compute arrival times given a nonhomogeneous Poisson process specified by lambda
  NpArray<double> first_arrival(const ArrayLike<double>& lambda_t, double dt, Py_ssize_t n, double minval);

  // given an array of arrival times at one state, sample times of arrival of subsequent event optionally with a minumum
  // separation minsep relative = true means lambda_t is relative to the starting point *plus minsep*
  NpArray<double> next_arrival(const ArrayLike<double>& startingpoints, const ArrayLike<double>& lambda_t, double dt,
                               bool relative, double minsep);

  // returns sampled arrival counts (within time interval dt) for each intensity lambda
  NpArray<int64_t> counts(const ArrayLike<double>& lambda, double dt);

private:
  // Use this over std::uniform_real_distribution as can make C++ and rust implementations produce identical streams
  double u01() noexcept;

  std::function<int32_t()> m_seeder;
  int32_t m_seed;
  std::mt19937 m_prng;
};

// helper functions

// construct cumulative from probabilities
std::vector<double> cumulative(const double* p, size_t n);

// get index of largest value of cumprob smaller than x
size_t interp(const std::vector<double>& cumprob, double x);

} // namespace no