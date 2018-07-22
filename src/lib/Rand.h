
#pragma once

//#include "Global.h"

#include <vector>
#include <random>


// simple hazard 
inline std::vector<int64_t> hazard(double cutoff, size_t n)
{
  // TODO thread/process-safe seeding
  std::mt19937 prng;
  std::uniform_real_distribution<> dist(0.0, 1.0);

  std::vector<int64_t> h(n);
  for (auto& it: h)
    it = (dist(prng) < cutoff) ? 1 : 0;
  return h;
}

// TODO vectors of stopping times/truncated...