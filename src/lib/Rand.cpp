
#include "Rand.h"

#include <vector>
#include <random>


// simple hazard 
std::vector<int> hazard(double prob, size_t n)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng(77027465);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::vector<int> h(n);
  for (auto& it: h)
    it = (dist(prng) < prob) ? 1 : 0;
  return h;
}

// simple hazard 
std::vector<int> hazard_v(const std::vector<double>& prob)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng(77027465);
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
std::vector<double> stopping(double prob, size_t n)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng(77027465);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double rprob = 1.0 / prob;

  std::vector<double> h(n);
  for (auto& it: h)
  {
    it = -log(dist(prng)) * rprob;
  }

  return h;
}

// computes stopping times 
std::vector<double> stopping_v(const std::vector<double>& prob)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng(77027465);
  std::uniform_real_distribution<> dist(0.0, 1.0);  

  size_t n = prob.size();
  std::vector<double> h(n);
  for (size_t i = 0; i < n; ++i)
  {
    h[i] = -log(dist(prng)) / prob[i];
  }

  return h;
}