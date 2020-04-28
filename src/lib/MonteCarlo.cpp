
#include "MonteCarlo.h"
#include "Environment.h"
//#include "Log.h"

#include "ArrayHelpers.h"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace {

// compute the RNG seed
int64_t compute_seed(int rank, int size, bool indep)
{
  // ensure stream (in)dependence w.r.t. sequence and MPI rank/sizes
  return 77027473 * 0 + 19937 * size + rank * indep;  
}

}

no::MonteCarlo::MonteCarlo(int rank, int size, bool indep) 
  : m_indep(indep), m_seed(compute_seed(rank, size, indep)), m_prng(m_seed) { }

bool no::MonteCarlo::indep() const 
{ 
  return m_indep; 
}

int64_t no::MonteCarlo::seed() const 
{ 
  return m_seed; 
}

void no::MonteCarlo::reset() 
{
  m_prng.seed(m_seed); 
}

// uniform [0,1)
double no::MonteCarlo::u01()
{
  static const double SCALE = 1.0 / (1ull << 32);
  return m_prng() * SCALE;
}

NEWORDER_EXPORT py::array no::MonteCarlo::ustream(py::ssize_t n)
{
  return no::make_array<double>({n}, [&](){ return u01(); });
}

// simple hazard constant probability 
NEWORDER_EXPORT py::array no::MonteCarlo::hazard(double prob, py::ssize_t n)
{
  return no::make_array<int>({n}, [&]() { return (u01() < prob) ? 1 : 0; });
}

// hazard with varying probablities 
py::array no::MonteCarlo::hazard(const py::array& prob)
{
  return no::unary_op<int, double>(prob, [&](double p){ return u01() < p ? 1 : 0; });
}

// computes stopping times 
NEWORDER_EXPORT py::array no::MonteCarlo::stopping(double prob, py::ssize_t n)
{
  double rprob = 1.0 / prob;

  return no::make_array<double>({n}, [&]() { return -::log(u01()) * rprob; });
}

py::array no::MonteCarlo::stopping(const py::array& prob)
{
  return no::unary_op<double, double>(prob, [&](double p) { return -::log(u01()) / p; });
}


// multiple-arrival (0+) process 
py::array no::MonteCarlo::arrivals(const py::array& lambda_t, double dt, double gap, size_t n)
{
  const double* pl = no::cbegin<double>(lambda_t);
  size_t nl = lambda_t.size();

  // validate lambdas - but what exactly is valid?
  if (pl[nl-1] != 0.0)
  {
    throw std::runtime_error("Multiple-arrival Non-homogeneous Poisson process requires a zero final hazard rate");
  }

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  std::vector<std::vector<double>> times(n);

  double tmax = (nl - 1) * dt;
  size_t imax = 0;

  for (size_t i = 0; i < n; ++i)
  {
    // rejection sampling
    double pt = 0.0;
    do 
    {
      do 
      {
        pt += -::log(u01()) / lambda_u;
        // final entry in lambda_t is flat extrapolated...
        lambda_i = pl[ std::min((size_t)(pt / dt), nl-1) ];
        if (pt > tmax && lambda_i == 0.0)
        {
          pt = no::Timeline::never();
          break;
        }
      } while (u01() > lambda_i / lambda_u);
      times[i].push_back(pt);
      pt += gap;
    } while (pt < tmax);
    imax = std::max(times[i].size(), imax);
    //no::log("%%: %%"_s % i % times[i]);
  }

  py::array nptimes = no::empty<double>({n, imax - 1});
  no::fill(nptimes, no::Timeline::never());
  double* pa = no::begin<double>(nptimes);

  for (size_t i = 0; i < times.size(); ++i)
  {
    for (size_t j = 0; j < times[i].size() - 1; ++j)
    {
      pa[j] = times[i][j];
    }
    pa += imax - 1;
  }

  return nptimes;
}

py::array no::MonteCarlo::first_arrival(const py::array& lambda_t, double dt, size_t n, double minval)
{
  const double* pl = no::cbegin<double>(lambda_t);
  size_t nl = lambda_t.size();

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  py::array times = no::empty_array<double>({static_cast<py::ssize_t>(n)});
  double* pt = no::begin<double>(times);
  double tmax = (nl - 1) * dt;

  for (size_t i = 0; i < n; ++i)
  {
    // rejection sampling
    pt[i] = minval;
    do 
    {
      pt[i] += -::log(u01()) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[ std::min((size_t)(pt[i] / dt), nl-1) ];
      // deal with open case (event not certain to happen)
      if (pt[i] > tmax && lambda_i == 0.0)
      {
        pt[i] = no::Timeline::never();
        break;
      }
    } while (u01() > lambda_i / lambda_u);
  }
  return times;
}

// next-arrival process - times of transition from a state arrived at at startingpoints to a subsequent state, with an optional deterministic minimum separation
// if the state hasn't been arrived at (no::never())
py::array no::MonteCarlo::next_arrival(const py::array& startingpoints, const py::array& lambda_t, double dt, bool relative, double minsep)
{
  py::ssize_t n = startingpoints.size();

  const double* pl = no::cbegin<double>(lambda_t);
  size_t nl = lambda_t.size();
  double tmax = (nl - 1) * dt;

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  py::array times = no::empty_array<double>({n});
  double* pt = no::begin<double>(times);

  for (py::ssize_t i = 0; i < n; ++i)
  {
    // account for any deterministic time lag (e.g. 9 months between births)
    double offset = no::at<double>(startingpoints, {i}) + minsep;
    // skip if we haven't actually arrived at the state to transition from
    if (no::Timeline::isnever(offset))
    {
      pt[i] = no::Timeline::never();
      continue;
    }
    // offset if lambdas in absolute time (not relative to start point)
    pt[i] = relative ? 0.0 : offset;
    do 
    {
      pt[i] += -::log(u01()) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[ std::min((size_t)(pt[i] / dt), nl-1) ];
      if (pt[i] > tmax && lambda_i == 0.0)
      {
        pt[i] = no::Timeline::never();
        break;
      }
    } while (u01() > lambda_i / lambda_u);
    // restore offset if required
    pt[i] += relative ? offset : 0.0;
  }
  return times;
}

