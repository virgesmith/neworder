
#pragma once

//#include "Global.h"

#include <vector>
#include <random>


// simple hazard 
inline std::vector<int64_t> hazard(double prob, size_t n)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng;
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::vector<int64_t> h(n);
  for (auto& it: h)
    it = (dist(prng) < prob) ? 1 : 0;
  return h;
}

// computes stopping times within cutoff period (-1 otherwise)
inline std::vector<double> stopping(double prob, double cutoff, size_t n)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng;
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double rprob = 1.0 / prob;

  std::vector<double> h(n);
  for (auto& it: h)
  {
    double t = -log(dist(prng)) * rprob;
    it = (t < cutoff) ? t : -1.0;
  }
  return h;
}

// TODO vectors of stopping times/truncated...