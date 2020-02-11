
#include "MonteCarlo.h"
#include "Environment.h"
//#include "Log.h"

#include "numpy.h"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// // Realised random outcomes based on vector of hazard rates
// struct Hazard : pycpp::UnaryArrayOp<int, double>
// {
//   Hazard() : m_prng(no::getenv().prng()), m_dist(0.0, 1.0) { }      

//   int operator()(double p)
//   {
//     return (m_dist(m_prng) < p) ? 1 : 0;
//   }

//   // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
//   // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
//   // force it to be visible:
//   using pycpp::UnaryArrayOp<int, double>::operator();

// private:
//   std::mt19937& m_prng;
//   std::uniform_real_distribution<double> m_dist;  
// };


// // Turns vector of hazard rates into random stopping times
// struct Stopping : pycpp::UnaryArrayOp<double, double>
// {
//   Stopping() : m_prng(no::getenv().prng()), m_dist(0.0, 1.0) { }      

//   double operator()(double p)
//   {
//     return -::log(m_dist(m_prng)) / p;
//   } 

//   // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
//   // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
//   // force it to be visible:
//   using pycpp::UnaryArrayOp<double, double>::operator();

// private:
//   std::mt19937& m_prng;
//   std::uniform_real_distribution<double> m_dist;  
// };

NEWORDER_EXPORT np::array no::MonteCarlo::ustream(size_t n)
{
  std::uniform_real_distribution<> dist(0.0, 1.0);

  return np::make_array<double>(n, [&](){ return dist(m_prng); });
}

// simple hazard constant probability 
NEWORDER_EXPORT np::array no::MonteCarlo::hazard(double prob, size_t n)
{
  std::uniform_real_distribution<> dist(0.0, 1.0);

  return np::make_array<int>(n, [&]() { return (dist(m_prng) < prob) ? 1 : 0; });
}

// hazard with varying probablities 
np::array no::MonteCarlo::hazard(const np::array& prob)
{
  std::uniform_real_distribution<> dist(0.0, 1.0);

  return np::unary_op<double, double>(prob, [&](double p){ return dist(m_prng) < p ? 1 : 0; });

  // py::array_t<int> result(prob.size());
  // const double* pp = (const double*)prob.data(0);
  // int* pr = (int*)result.request().ptr;
  // for (size_t i = 0; i < prob.size(); ++i, ++pp, ++pr)
  // {
  //   *pr = (dist(prng) < *pp) ? 1 : 0;
  // }
  // return result;

  // Hazard f;
  // return f(prob);
}

// computes stopping times 
NEWORDER_EXPORT np::array no::MonteCarlo::stopping(double prob, size_t n)
{
  std::uniform_real_distribution<> dist(0.0, 1.0);
  double rprob = 1.0 / prob;

  return np::make_array<double>(n, [&]() { return -::log(dist(m_prng)) * rprob; });
}

np::array no::MonteCarlo::stopping(const np::array& prob)
{
  std::uniform_real_distribution<> dist(0.0, 1.0);

  return np::unary_op<double, double>(prob, [&](double p) { return -::log(dist(m_prng)) / p; });

  // Stopping f;
  // return f(prob);
}


// multiple-arrival (0+) process 
np::array no::MonteCarlo::arrivals(const np::array& lambda_t, double dt, double gap, size_t n)
{
  std::uniform_real_distribution<> dist(0.0, 1.0);

  const double* pl = np::cbegin<double>(lambda_t);
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
        pt += -::log(dist(m_prng)) / lambda_u;
        // final entry in lambda_t is flat extrapolated...
        lambda_i = pl[ std::min((size_t)(pt / dt), nl-1) ];
        if (pt > tmax && lambda_i == 0.0)
        {
          pt = no::Timeline::never();
          break;
        }
      } while (dist(m_prng) > lambda_i / lambda_u);
      times[i].push_back(pt);
      pt += gap;
    } while (pt < tmax);
    imax = std::max(times[i].size(), imax);
    //no::log("%%: %%"_s % i % times[i]);
  }

  np::array nptimes = np::empty<double>({n, imax- 1});
  np::fill(nptimes, no::Timeline::never());
  double* pa = np::begin<double>(nptimes);

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

np::array no::MonteCarlo::first_arrival(const np::array& lambda_t, double dt, size_t n, double minval)
{
  std::uniform_real_distribution<> dist(0.0, 1.0);

  const double* pl = np::cbegin<double>(lambda_t);
  size_t nl = lambda_t.size();

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  np::array times = np::empty_1d_array<double>(n);
  double* pt = np::begin<double>(times);
  double tmax = (nl - 1) * dt;

  for (size_t i = 0; i < n; ++i)
  {
    // rejection sampling
    pt[i] = minval;
    do 
    {
      pt[i] += -::log(dist(m_prng)) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[ std::min((size_t)(pt[i] / dt), nl-1) ];
      // deal with open case (event not certain to happen)
      if (pt[i] > tmax && lambda_i == 0.0)
      {
        pt[i] = no::Timeline::never();
        break;
      }
    } while (dist(m_prng) > lambda_i / lambda_u);
  }
  return times;
}

// next-arrival process - times of transition from a state arrived at at startingpoints to a subsequent state, with an optional deterministic minimum separation
// if the state hasn't been arrived at (no::never())
np::array no::MonteCarlo::next_arrival(const np::array& startingpoints, const np::array& lambda_t, double dt, bool relative, double minsep)
{
  size_t n = startingpoints.size();

  std::uniform_real_distribution<> dist(0.0, 1.0);

  const double* pl = np::cbegin<double>(lambda_t);
  size_t nl = lambda_t.size();
  double tmax = (nl - 1) * dt;

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  np::array times = np::empty_1d_array<double>(n);
  //np::array times = np::zero_1d_array<double>(n);
  double* pt = np::begin<double>(times);

  for (size_t i = 0; i < n; ++i)
  {
    // account for any deterministic time lag (e.g. 9 months between births)
    double offset = np::at<double>(startingpoints, i) + minsep;
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
      pt[i] += -::log(dist(m_prng)) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[ std::min((size_t)(pt[i] / dt), nl-1) ];
      if (pt[i] > tmax && lambda_i == 0.0)
      {
        pt[i] = no::Timeline::never();
        break;
      }
    } while (dist(m_prng) > lambda_i / lambda_u);
    // restore offset if required
    pt[i] += relative ? offset : 0.0;
  }
  return times;
}