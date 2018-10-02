#pragma once

#include <vector>
#include <cstddef>

namespace neworder {

class Timeline final
{
public:

  // Default "null" timeline is just one step of arbitrary size
  Timeline();

  Timeline(double begin, double end, int n);

  ~Timeline() = default;

  Timeline(const Timeline&) = delete;
  Timeline& operator=(const Timeline&) = default;

  double time();
  size_t index();

  double dt();
  void step();

  bool checkpoint();

  bool end();

  // returns a floating point number that compares less than any other number
  static double distant_past();

  // returns a floating point number that compares greater than any other number
  static double far_future();

private:
  std::vector<double> m_checkpoints;
  double m_dt; // timestep
  size_t m_steps; // total no. of steps
  double m_time; // current time
  size_t m_index; // index of current time
};

}