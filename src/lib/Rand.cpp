
#include "Rand.h"

#include <vector>
#include <random>
#include <algorithm>

// TODO thread/process-safe seeding
namespace {
std::mt19937 prng(77027465);
}

neworder::UStream::UStream(int64_t seed) : m_seed(seed), m_prng(seed), m_dist(0.0, 1.0)
{
}

std::vector<double> neworder::UStream::get(int n)
{
  std::vector<double> ret(n);
  for (int i = 0; i < n; ++i)
  {
    ret[i] = m_dist(m_prng);
  }
  return ret;
}


// simple hazard 
std::vector<int> neworder::hazard(double prob, size_t n)
{
  // TODO thread/process-safe seeding
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::vector<int> h(n);
  for (auto& it: h)
    it = (dist(prng) < prob) ? 1 : 0;
  return h;
}

// simple hazard 
std::vector<int> neworder::hazard_v(const std::vector<double>& prob)
{
  // TODO thread/process-safe seeding
  std::uniform_real_distribution<> dist(0.0, 1.0);

  size_t n = prob.size();
  std::vector<int> h(n);
  for (size_t i = 0; i < n; ++i)
  {
    h[i] = (dist(prng) < prob[i]) ? 1 : 0;
  }
  return h;
}

// computes stopping times 
std::vector<double> neworder::stopping(double prob, size_t n)
{
  // TODO thread/process-safe seeding
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
  // TODO thread/process-safe seeding
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
  // TODO thread/process-safe seeding
  std::uniform_real_distribution<> dist(0.0, 1.0);  

  size_t n = prob.size();
  std::vector<double> h(n);
  for (size_t i = 0; i < n; ++i)
  {
    h[i] = -log(dist(prng)) / prob[i];
  }

  return h;
}