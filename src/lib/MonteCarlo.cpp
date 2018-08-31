
#include "MonteCarlo.h"
#include "Environment.h"
#include "Module.h"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include <iostream>

// Realised random outcomes based on vector of hazard rates
struct Hazard : pycpp::UnaryArrayOp<int, double>
{
  Hazard() : m_prng(pycpp::Environment::get().prng()), m_dist(0.0, 1.0) { }      

  int operator()(double p)
  {
    return (m_dist(m_prng) < p) ? 1 : 0;
  } 

  np::ndarray operator()(const np::ndarray& arg)
  {
    return call_impl(arg);
  }

private:
  std::mt19937& m_prng;
  std::uniform_real_distribution<double> m_dist;  
};


// Turns vector of hazard rates into random stopping times
struct Stopping : pycpp::UnaryArrayOp<double, double>
{
  Stopping() : m_prng(pycpp::Environment::get().prng()), m_dist(0.0, 1.0) { }      

  double operator()(double p)
  {
    return -::log(m_dist(m_prng)) / p;
  } 

  np::ndarray operator()(const np::ndarray& arg)
  {
    return call_impl(arg);
  }

private:
  std::mt19937& m_prng;
  std::uniform_real_distribution<double> m_dist;  
};


// np::ndarray neworder::ustream(size_t n)
// {
//   std::mt19937& prng = pycpp::Environment::get().prng();
//   std::uniform_real_distribution<> dist(0.0, 1.0); // can't make this const, so best not make it static 
//   np::ndarray a = pycpp::empty_1d_array<double>(n); 
//   // TODO ufunc?
//   double* p = reinterpret_cast<double*>(a.get_data());
//   std::generate(p, p + n, [&](){ return dist(prng); });
//   return a;
// }

np::ndarray neworder::ustream(size_t n)
{
  struct PrngFill : pycpp::NullaryArrayOp<double>
  {
    PrngFill() : prng(pycpp::Environment::get().prng()), dist(0.0, 1.0) { }
 
    double operator()()
    {
      return dist(prng);
    }

    np::ndarray operator()(size_t n)
    {
      return call_impl(n);
    }

  private:

    std::mt19937& prng;
    std::uniform_real_distribution<double> dist; 
  };

  PrngFill f;
  return f(n);
  // np::ndarray a = pycpp::empty_1d_array<double>(n); 
  // // TODO ufunc?
  // double* p = reinterpret_cast<double*>(a.get_data());
  // std::generate(p, p + n, [&](){ return dist(prng); });
}

// simple hazard constant probability 
np::ndarray neworder::hazard(double prob, size_t n)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  np::ndarray h = pycpp::empty_1d_array<int>(n); 
  // TODO ufunc?
  int* p = reinterpret_cast<int*>(h.get_data());
  for (size_t i = 0; i < n; ++i)
    p[i] = (dist(prng) < prob) ? 1 : 0;
  return h;
}

// hazard with varying probablities 
np::ndarray neworder::hazard_v(const np::ndarray& prob)
{
  Hazard f;
  return f(prob);
}

// computes stopping times 
np::ndarray neworder::stopping(double prob, size_t n)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double rprob = 1.0 / prob;
  np::ndarray s = pycpp::empty_1d_array<double>(n);
  // TODO ufunc?
  double* p = reinterpret_cast<double*>(s.get_data());
  for (size_t i = 0; i < n; ++i)
    p[i] = -::log(dist(prng)) * rprob;

  return s;
}

np::ndarray neworder::stopping_v(const np::ndarray& prob)
{
  Stopping f;
  return f(prob);
}

// MC stopping time for a non-homogeneous poisson process, given
// a piecewise-constant hazard rate with spacing dt=1
// uses the thinning algorithm described in: 
// Lewis, Peter A., and Gerald S. Shedler. "Simulation of nonhomogeneous Poisson processes by thinning." Naval Research Logistics (NRL) 26.3 (1979): 403-413.
np::ndarray neworder::stopping_nhpp(const np::ndarray& lambda_t, size_t n)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double* pl = reinterpret_cast<double*>(lambda_t.get_data());
  size_t nl = pycpp::size(lambda_t);

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  np::ndarray times = pycpp::empty_1d_array<double>(n);
  double* pt = reinterpret_cast<double*>(times.get_data());

// def generate_times_opt(rate_function,max_t,delta):
//     t = numpy.arange(delta,max_t, delta)
//     avg_rate = (rate_function(t) + rate_function(t + delta)) / 2.0
//     avg_prob = 1 - numpy.exp(-avg_rate * delta / 1000.0)
//     rand_throws = numpy.random.uniform(size=t.shape[0])

//     return t[avg_prob >= rand_throws].tolist()

  // for (size_t i = 0; i < n; ++i)
  // {
  //   do 
  //   {
  //     times[i] -= log(dist(prng)) / lambda_u;
  //     lambda_i = lambda_t[ std::min((size_t)times[i], lambda_t.size()-1) ];
  //   } while (dist(prng) > lambda_i / lambda_u);
  // }

  for(size_t i = 0; i < n; ++i)
  {
    double t = 0;
    for (;;)
    {
      t = t - ::log(dist(prng)) / lambda_u;
      lambda_i = pl[ std::min((size_t)t, nl-1) ];
      if (dist(prng) <= lambda_i / lambda_u)
      {
        pt[i] = t;
        break;
      }
    }
  }

  return times;
}

