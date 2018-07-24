
#pragma once

//#include "Global.h"

#include <vector>
#include <random>


// simple hazard 
inline std::vector<int> hazard(double prob, size_t n)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng;
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::vector<int> h(n);
  for (auto& it: h)
    it = (dist(prng) < prob) ? 1 : 0;
  return h;
}

// computes stopping times 
inline std::vector<double> stopping(double prob, size_t n)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng;
  std::uniform_real_distribution<> dist(0.0, 1.0);

  double rprob = 1.0 / prob;

  std::vector<double> h(n);
  for (auto& it: h)
  {
    it = -log(dist(prng)) * rprob;
  }

  // if (cutoff > 0)
  // {
  //   for (auto& it: h)
  //   {
  //     it = (it < cutoff) ? it : -1.0;
  //   }
  // }
  return h;
}

