
#pragma once

//#include "Global.h"

#include <vector>
#include <cstdint>

// TODO need threadsafe RNG independence/seeding

// single-prob hazard 
std::vector<int> hazard(double prob, size_t n);

// vector hazard 
std::vector<int> hazard_v(const std::vector<double>& prob);

// compute stopping times 
std::vector<double> stopping(double prob, size_t n);

// vector stopping 
std::vector<double> stopping_v(const std::vector<double>& prob);
