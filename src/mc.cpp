#include "mc.h"
#include "Global.h"

#include <random>

std::vector<double> urand01(size_t n)
{
  std::mt19937& rng = Global::instance<std::mt19937>();
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  
  std::vector<double> r(n);
  for (size_t i = 0; i < n; ++i)
    r[i] = u01(rng);
  return r;
}
