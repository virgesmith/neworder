
#pragma once

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

// TODO rename to MC

namespace neworder {

// TODO this can just be a function now 
// Uniform random [0,1) fed from the environment's PRNG stream
class UStream
{
public:
  UStream();

  ~UStream() = default;

  // disable copy/assign to avoid non-independent streams
  // TODO work out how to export to python with no copy?
  // RStream(const RStream&) = delete;
  // RStream& operator=(const RStream&) = delete;

  std::vector<double> get(int n);

private:
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

// vector stopping for non-homogeneous poisson process
std::vector<double> stopping_nhpp(const std::vector<double>& lambda_t, size_t n);

}