#pragma once

#include "Log.h"

#include <vector>

class Timeline final
{
public:

  // Default "null" timeline is just one step of arbitrary size
  Timeline() : m_checkpoints{0.0, 1.0}, m_steps(1), m_time(0.0), m_index(0) { }

  Timeline(double begin, double end, int n) : m_checkpoints{begin, end}, m_steps(n), m_time(begin), m_index(0) 
  { 
    // validate
    if (n < 1)
    {
      throw std::runtime_error("invalid timeline: nsteps (%%) is not strictly positive"_s % n);
    }
    if (begin >= end)
    {
      throw std::runtime_error("invalid timeline: begin (%%) is not strictly before end (%%)"_s % begin % end);
    }
    // 0th timestep is at begin, nth timestep is at end
    m_dt = (end-begin) / n; 
  }

  ~Timeline() { }

  Timeline(const Timeline&) = delete;
  //Timeline& operator=(const Timeline&) = delete;

  double time() { return m_time; }
  double index() { return m_index; }

  double dt() { return m_dt; }

  void step()
  {
    if (m_time < m_checkpoints.back())
    {
      m_time += m_dt;
      ++m_index;
    }
  }

  bool checkpoint()
  {
    // TODO...
    // for now only end is a checkpoint
    return end();
  }

  bool end()
  {
    return m_index == m_steps;
  }

private:
  std::vector<double> m_checkpoints;
  double m_dt; // timestep
  size_t m_steps; // total no. of steps
  double m_time; // current time
  size_t m_index; // index of current time
};

