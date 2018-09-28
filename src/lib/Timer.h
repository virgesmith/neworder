#pragma once 

// Simple timer functionality

#include <chrono>

class Timer
{
public:
  Timer() 
  { 
    reset();
  }

  ~Timer() = default;

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

  void reset()
  {
    m_start = std::chrono::high_resolution_clock::now();
  }

  double elapsed_s()
  {
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_start).count() / 1'000'000;
  }

private:
  std::chrono::high_resolution_clock::time_point m_start;
};