
#include "MonteCarlo.h"
#include "Environment.h"

#include <vector>
#include <random>
#include <algorithm>



// std::vector<double> neworder::ustream(int n)
// {
//   std::mt19937& prng = pycpp::Environment::get().prng();
//   std::uniform_real_distribution<> dist(0.0, 1.0); // can't make this const, so best not make it static 
//   std::vector<double> ret(n);
//   std::generate(ret.begin(), ret.end(), [&](){ return dist(prng); });

//   return ret;
// }

np::ndarray neworder::ustream(size_t n)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0); // can't make this const, so best not make it static 
  np::ndarray a = pycpp::empty_1d_array<double>(n); 
  double* p = reinterpret_cast<double*>(a.get_data());
  std::generate(p, p + n, [&](){ return dist(prng); });
  return a;
}



// simple hazard 
np::ndarray neworder::hazard(double prob, size_t n)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  np::ndarray h = pycpp::empty_1d_array<int>(n); 
  int* p = reinterpret_cast<int*>(h.get_data());
  for (size_t i = 0; i < n; ++i)
    p[i] = (dist(prng) < prob) ? 1 : 0;
  return h;
}

// simple hazard 
np::ndarray neworder::hazard_v(const np::ndarray& prob)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  size_t n = pycpp::size(prob);
  np::ndarray h = pycpp::empty_1d_array<int>(n); 
  int* p = reinterpret_cast<int*>(h.get_data());
  for (size_t i = 0; i < n; ++i)
    p[i] = (dist(prng) < prob[i]) ? 1 : 0;
  return h;
}

// computes stopping times 
std::vector<double> neworder::stopping(double prob, size_t n)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double rprob = 1.0 / prob;

  std::vector<double> h(n);
  for (auto& it: h)
  {
    it = -log(dist(prng)) * rprob;
  }

  return h;
}

// MC stopping time for a non-homogeneous poisson process, given
// a piecewise-constant hazard rate with spacing dt=1
// uses the thinning algorithm described in: 
// Lewis, Peter A., and Gerald S. Shedler. "Simulation of nonhomogeneous Poisson processes by thinning." Naval Research Logistics (NRL) 26.3 (1979): 403-413.
std::vector<double> neworder::stopping_nhpp(const std::vector<double>& lambda_t, size_t n)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(lambda_t.begin(), lambda_t.end());
  double lambda_i;

  std::vector<double> times(n);

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
      t = t - log(dist(prng)) / lambda_u;
      lambda_i = lambda_t[ std::min((size_t)t, lambda_t.size()-1) ];
      if (dist(prng) <= lambda_i / lambda_u)
      {
        times[i] = t;
        break;
      }
    }
  }

  return times;
}

// computes stopping times 
std::vector<double> neworder::stopping_v(const std::vector<double>& prob)
{
  std::mt19937& prng = pycpp::Environment::get().prng();
  std::uniform_real_distribution<> dist(0.0, 1.0);  

  size_t n = prob.size();
  std::vector<double> h(n);
  for (size_t i = 0; i < n; ++i)
  {
    h[i] = -log(dist(prng)) / prob[i];
  }

  return h;
}