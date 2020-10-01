
#include "MonteCarlo.h"
#include "Environment.h"
#include "Log.h"
#include "ArrayHelpers.h"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace {

void validate_prob(double prob)
{
  if (prob < 0.0 || prob > 1.0)
  {
    throw py::value_error("probabilities must be in [0,1], got %%"_s % prob);
  }
}

void validate_prob(const py::array_t<double>& prob)
{
  std::for_each(no::cbegin(prob), no::cend(prob), [](double p) { 
    if (p < 0.0 || p > 1.0)
    {
      throw py::value_error("probabilities must be in [0,1], got %%"_s % p);
    }
  });
}

void validate_lambda(const py::array_t<double>& lambda)
{
  std::for_each(no::cbegin(lambda), no::cend(lambda), [](double l) { 
    if (l < 0.0)
    {
      throw py::value_error("hazard rates must be in >=0, got %%"_s % l);
    }
  });
}

}


// helper functions for basic seeding strategies
int64_t no::MonteCarlo::deterministic_independent_stream(int r)
{
  return 19937 + r;
}

int64_t no::MonteCarlo::deterministic_identical_stream(int)
{
  return 19937;  
}

int64_t no::MonteCarlo::nondeterministic_stream(int)
{
  std::random_device rand;
  return (static_cast<int64_t>(rand()) << 32) + rand();
}

//
no::MonteCarlo::MonteCarlo(int64_t seed) 
  : m_seed(seed), m_prng(m_seed) { }


int64_t no::MonteCarlo::seed() const 
{ 
  return m_seed; 
}

void no::MonteCarlo::reset() 
{
  m_prng.seed(m_seed); 
}

std::string no::MonteCarlo::repr() const
{  
  return "<neworder.MonteCarlo seed=%%>"_s % seed();
}

size_t no::MonteCarlo::state() const
{
// disable as causes segfault in mpi tests on travis OSX build
#ifdef __APPLE__
  return 0;
#else
  std::ostringstream s;
  s << m_prng;
  return std::hash<std::string>{}(s.str());
#endif
}

// raw unsigned 64-bit ints (for enabling numpy to use this generator)
uint64_t no::MonteCarlo::raw()
{
  return (static_cast<uint64_t>(m_prng()) << 32) | static_cast<uint64_t>(m_prng()); 
}

// uniform [0,1)
double no::MonteCarlo::u01()
{
  static const double SCALE = 1.0 / (1ull << 32);
  return m_prng() * SCALE;
}

py::array_t<double> no::MonteCarlo::ustream(py::ssize_t n)
{
  return no::make_array<double>({n}, [this](){ return u01(); });
}

py::array_t<int64_t> no::MonteCarlo::sample(py::ssize_t n, const py::array_t<double>& cat_weights)
{
  py::ssize_t m = cat_weights.size();
  const double* p = no::cbegin(cat_weights);

  std::vector<double> cumul(m);
  double running_sum = 0.0;
  for (size_t i = 0; i < static_cast<size_t>(m); ++i)
  {
    if (p[i] < 0.0) 
      throw py::value_error("category weights must be positive, got %%"_s % p[i]);
    running_sum += p[i];
    cumul[i] = running_sum;     
  }

  if (fabs(cumul[m-1] - 1.0) > std::numeric_limits<double>::epsilon())  
    throw py::value_error("category weights must sum to unity, got %%"_s % cumul[m-1]);

  return no::make_array<int64_t>({n}, [this, &cumul]() { 
      double r = u01();
      int64_t k = 0;
      while (cumul[k] < r) ++k;
      return k; 
    });

}

// simple hazard constant probability 
py::array_t<double> no::MonteCarlo::hazard(double prob, py::ssize_t n)
{
  validate_prob(prob);
  return no::make_array<int>({n}, [&]() { return (u01() < prob) ? 1 : 0; });
}

// hazard with varying probablities 
py::array_t<double> no::MonteCarlo::hazard(const py::array_t<double>& prob)
{
  validate_prob(prob);
  return no::unary_op<int, double>(prob, [this](double p){ return u01() < p ? 1 : 0; });
}

// computes stopping times 
py::array_t<double> no::MonteCarlo::stopping(double prob, py::ssize_t n)
{
  validate_prob(prob);
  double rprob = 1.0 / prob;

  // if p=0 will return no::time::far_future() (NOT no::time::never())
  // the argument being that stopping is an event that must happen, as opposed to an arrival, which might not
  return no::make_array<double>({n}, [&]() { return -::log(u01()) * rprob; });
}

py::array_t<double> no::MonteCarlo::stopping(const py::array_t<double>& prob)
{
  validate_prob(prob);
  return no::unary_op<double, double>(prob, [this](double p) { return -::log(u01()) / p; });
}


// multiple-arrival (0+) process 
py::array_t<double> no::MonteCarlo::arrivals(const py::array_t<double>& lambda_t, double dt, py::ssize_t n, double gap)
{
  validate_lambda(lambda_t);
  const double* pl = no::cbegin(lambda_t);
  size_t nl = lambda_t.size();

  // validate lambdas - but what exactly is valid?
  if (pl[nl-1] != 0.0)
  {
    throw py::value_error("Multiple-arrival Non-homogeneous Poisson process requires a zero final hazard rate");
  }

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  std::vector<std::vector<double>> times(n);

  double tmax = (nl - 1) * dt;
  size_t imax = 0;

  for (size_t i = 0; i < static_cast<size_t>(n); ++i)
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
          pt = no::time::never();
          break;
        }
      } while (u01() > lambda_i / lambda_u);
      times[i].push_back(pt);
      pt += gap;
    } while (pt < tmax);
    imax = std::max(times[i].size(), imax);
    //no::log("%%: %%"_s % i % times[i]);
  }

  py::array_t<double> nptimes({n, static_cast<py::ssize_t>(imax - 1)});
  no::fill(nptimes, no::time::never());
  double* pa = no::begin(nptimes); //.begin();

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

py::array_t<double> no::MonteCarlo::first_arrival(const py::array_t<double>& lambda_t, double dt, py::ssize_t n, double minval)
{
  validate_lambda(lambda_t);
  const double* pl = no::cbegin(lambda_t);
  size_t nl = lambda_t.size();

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);

  double lambda_i;

  py::array_t<double> times(static_cast<size_t>(n));
  double* pt = no::begin(times);
  double tmax = (nl - 1) * dt;

  for (size_t i = 0; i < static_cast<size_t>(n); ++i)
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
        pt[i] = no::time::never();
        break;
      }
    } while (u01() > lambda_i / lambda_u);
  }
  return times;
}

// next-arrival process - times of transition from a state arrived at at startingpoints to a subsequent state, with an optional deterministic minimum separation
// if the state hasn't been arrived at (no::never())
py::array_t<double> no::MonteCarlo::next_arrival(const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt, bool relative, double minsep)
{
  validate_lambda(lambda_t);
  size_t n = startingpoints.size();

  const double* pl = no::cbegin(lambda_t);
  size_t nl = lambda_t.size();
  double tmax = (nl - 1) * dt;

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  py::array_t<double> times(n);
  double* pt = no::begin(times);

  for (size_t i = 0; i < n; ++i)
  {
    // account for any deterministic time lag (e.g. 9 months between births)
    double offset = no::at(startingpoints, {static_cast<py::ssize_t>(i)}) + minsep;
    // skip if we haven't actually arrived at the state to transition from
    if (no::time::isnever(offset))
    {
      pt[i] = no::time::never();
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
        pt[i] = no::time::never();
        break;
      }
    } while (u01() > lambda_i / lambda_u);
    // restore offset if required
    pt[i] += relative ? offset : 0.0;
  }
  return times;
}

