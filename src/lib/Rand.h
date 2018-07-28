
#pragma once

//#include "Global.h"

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

// TODO need threadsafe RNG independence/seeding

namespace neworder {

// Uniform random stream in [0,1)
class UStream
{
public:
  UStream(int64_t seed);

  ~UStream() = default;

  // disable copy/assign to avoid non-independent streams
  // TODO work out how to export to python with no copy?
  // RStream(const RStream&) = delete;
  // RStream& operator=(const RStream&) = delete;

  std::vector<double> get(int n);

private:
  int64_t m_seed;
  std::mt19937 m_prng;
  std::uniform_real_distribution<> m_dist;  
};

// single-prob hazard 
std::vector<int> hazard(double prob, size_t n);

// vector hazard 
std::vector<int> hazard_v(const std::vector<double>& prob);

// compute stopping times 
std::vector<double> stopping(double prob, size_t n);

// vector stopping 
std::vector<double> stopping_v(const std::vector<double>& prob);

}