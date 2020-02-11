
#pragma once

#include "numpy.h"

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>


namespace {

// compute the RNG seed
int64_t compute_seed(int rank, int size, bool indep)
{
  // ensure stream (in)dependence w.r.t. sequence and MPI rank/sizes
  return 77027473 * 0 + 19937 * size + rank * indep;  
}

}

namespace no {

class MonteCarlo
{
public:
  MonteCarlo(int rank, int size, bool indep) : m_indep(indep), m_seed(compute_seed(rank, size, indep)), m_prng(m_seed) { }

  bool indep() const { return m_indep; }

  int64_t seed() const { return m_seed; }

  void reset() 
  {
    m_prng.seed(m_seed); 
  }

  // Uniform random [0,1) fed from the environment's PRNG stream
  np::array ustream(size_t n);

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

  // given an array of arrival times at one state, sample times of arrival of subsequent event optionally with a minumum separation minsep
  // relative = true means lambda_t is relative to the stating point *plus minsep*
  np::array next_arrival(const np::array& startingpoints, const np::array& lambda_t, double dt, bool relative, double minsep);

private:
  bool m_indep;
  int64_t m_seed;
  std::mt19937 m_prng;
};


}