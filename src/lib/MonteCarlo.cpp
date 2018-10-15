
#include "MonteCarlo.h"
#include "Environment.h"
//#include "Log.h"

#include "numpy.h"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// Realised random outcomes based on vector of hazard rates
struct Hazard : pycpp::UnaryArrayOp<int, double>
{
  Hazard() : m_prng(neworder::getenv().prng()), m_dist(0.0, 1.0) { }      

  int operator()(double p)
  {
    return (m_dist(m_prng) < p) ? 1 : 0;
  }

  // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
  // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
  // force it to be visible:
  using pycpp::UnaryArrayOp<int, double>::operator();

private:
  std::mt19937& m_prng;
  std::uniform_real_distribution<double> m_dist;  
};


// Turns vector of hazard rates into random stopping times
struct Stopping : pycpp::UnaryArrayOp<double, double>
{
  Stopping() : m_prng(neworder::getenv().prng()), m_dist(0.0, 1.0) { }      

  double operator()(double p)
  {
    return -::log(m_dist(m_prng)) / p;
  } 

  // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
  // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
  // force it to be visible:
  using pycpp::UnaryArrayOp<double, double>::operator();

private:
  std::mt19937& m_prng;
  std::uniform_real_distribution<double> m_dist;  
};

np::ndarray neworder::ustream(size_t n)
{
  std::mt19937& prng = neworder::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  return pycpp::make_array<double>(n, [&](){ return dist(prng); });
}

// simple hazard constant probability 
np::ndarray neworder::hazard(double prob, size_t n)
{
  std::mt19937& prng = neworder::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  return pycpp::make_array<int>(n, [&]() { return (dist(prng) < prob) ? 1 : 0; });
}

// hazard with varying probablities 
np::ndarray neworder::hazard(const np::ndarray& prob)
{
  Hazard f;
  return f(prob);
}

// computes stopping times 
np::ndarray neworder::stopping(double prob, size_t n)
{
  std::mt19937& prng = neworder::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);
  double rprob = 1.0 / prob;

  return pycpp::make_array<double>(n, [&]() { return -::log(dist(prng)) * rprob; });
}

np::ndarray neworder::stopping(const np::ndarray& prob)
{
  Stopping f;
  return f(prob);
}

// DEPRECATED - NO LONGER EXPOSED TO PYTHON (use first_arrival)
// MC stopping time for a non-homogeneous poisson process, given
// a piecewise-constant hazard rate with spacing dt=1
// uses the thinning algorithm described in: 
// Lewis, Peter A., and Gerald S. Shedler. "Simulation of nonhomogeneous Poisson processes by thinning." Naval Research Logistics (NRL) 26.3 (1979): 403-413.
// See also explanation in Glasserman, Monte-Carlo Methods in Financial Engineering, 2003, pp140-141
np::ndarray neworder::stopping_nhpp(const np::ndarray& lambda_t, double dt, size_t n)
{
  std::mt19937& prng = neworder::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double* pl = pycpp::begin<double>(lambda_t);
  size_t nl = pycpp::size(lambda_t);

  // validate lambdas - but what exactly is valid?
  if (pl[nl-1] == 0.0)
  {
    throw std::runtime_error("Non-homogeneous Poisson process requires a nonzero final hazard rate");
  }
  // for (size_t i = 0; i < nl; ++i)
  // {
  //   if (pl[i] <= 0.0 || pl[i] >= 1.0)
  //     throw std::runtime_error("Lewis-Shedler algorithm requires probabilities in (0,1): element %% is %%"_s % i % pl[i]);
  // }

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  np::ndarray times = pycpp::empty_1d_array<double>(n);
  double* pt = pycpp::begin<double>(times);

  for (size_t i = 0; i < n; ++i)
  {
    // rejection sampling
    pt[i] = 0.0;
    do 
    {
      pt[i] += -::log(dist(prng)) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[ std::min((size_t)(pt[i] / dt), nl-1) ];
    } while (dist(prng) > lambda_i / lambda_u);
  }
  return times;
}


// multiple-arrival (0+) process 
np::ndarray neworder::arrivals(const np::ndarray& lambda_t, double dt, double gap, size_t n)
{
  std::mt19937& prng = neworder::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double* pl = pycpp::begin<double>(lambda_t);
  size_t nl = pycpp::size(lambda_t);

  // validate lambdas - but what exactly is valid?
  if (pl[nl-1] != 0.0)
  {
    throw std::runtime_error("Multiple-arrival Non-homogeneous Poisson process requires a zero final hazard rate");
  }
  // for (size_t i = 0; i < nl; ++i)
  // {
  //   if (pl[i] <= 0.0 || pl[i] >= 1.0)
  //     throw std::runtime_error("Lewis-Shedler algorithm requires probabilities in (0,1): element %% is %%"_s % i % pl[i]);
  // }

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
        pt += -::log(dist(prng)) / lambda_u;
        // final entry in lambda_t is flat extrapolated...
        lambda_i = pl[ std::min((size_t)(pt / dt), nl-1) ];
        if (pt > tmax && lambda_i == 0.0)
        {
          pt = neworder::Timeline::never();
          break;
        }
      } while (dist(prng) > lambda_i / lambda_u);
      times[i].push_back(pt);
      pt += gap;
    } while (pt < tmax);
    imax = std::max(times[i].size(), imax);
    //neworder::log("%%: %%"_s % i % times[i]);
  }

  np::ndarray nptimes = np::empty(py::make_tuple(n, imax- 1), np::dtype::get_builtin<double>());
  pycpp::fill(nptimes, neworder::Timeline::never());
  double* pa = pycpp::begin<double>(nptimes);

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

np::ndarray neworder::first_arrival(const np::ndarray& lambda_t, double dt, size_t n, double minval)
{
  std::mt19937& prng = neworder::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double* pl = pycpp::begin<double>(lambda_t);
  size_t nl = pycpp::size(lambda_t);

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  np::ndarray times = pycpp::empty_1d_array<double>(n);
  double* pt = pycpp::begin<double>(times);
  double tmax = (nl - 1) * dt;

  for (size_t i = 0; i < n; ++i)
  {
    // rejection sampling
    pt[i] = minval;
    do 
    {
      pt[i] += -::log(dist(prng)) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[ std::min((size_t)(pt[i] / dt), nl-1) ];
      // deal with open case (event not certain to happen)
      if (pt[i] > tmax && lambda_i == 0.0)
      {
        pt[i] = neworder::Timeline::never();
        break;
      }
    } while (dist(prng) > lambda_i / lambda_u);
  }
  return times;
}

// next-arrival process - times of transition from a state arrived at at startingpoints to a subsequent state, with an optional deterministic minimum separation
// if the state hasn't been arrived at (neworder::never())
np::ndarray neworder::next_arrival(const np::ndarray& startingpoints, const np::ndarray& lambda_t, double dt, bool relative, double minsep)
{
  size_t n = pycpp::size(startingpoints);

  std::mt19937& prng = neworder::getenv().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double* pl = pycpp::begin<double>(lambda_t);
  size_t nl = pycpp::size(lambda_t);
  double tmax = (nl - 1) * dt;

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  np::ndarray times = pycpp::empty_1d_array<double>(n);
  //np::ndarray times = pycpp::zero_1d_array<double>(n);
  double* pt = pycpp::begin<double>(times);

  for (size_t i = 0; i < n; ++i)
  {
    // account for any deterministic time lag (e.g. 9 months between births)
    double offset = pycpp::at<double>(startingpoints, i) + minsep;
    // skip if we haven't actually arrived at the state to transition from
    if (neworder::Timeline::isnever(offset))
    {
      pt[i] = neworder::Timeline::never();
      continue;
    }
    // offset if lambdas in absolute time (not relative to start point)
    pt[i] = relative ? 0.0 : offset;
    do 
    {
      pt[i] += -::log(dist(prng)) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[ std::min((size_t)(pt[i] / dt), nl-1) ];
      if (pt[i] > tmax && lambda_i == 0.0)
      {
        pt[i] = neworder::Timeline::never();
        break;
      }
    } while (dist(prng) > lambda_i / lambda_u);
    // restore offset if required
    pt[i] += relative ? offset : 0.0;
  }
  return times;
}