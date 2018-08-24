
#pragma once

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>

// TODO rename to MC

namespace neworder {

// Uniform random [0,1) fed from the environment's PRNG stream
std::vector<double> ustream(int n);

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