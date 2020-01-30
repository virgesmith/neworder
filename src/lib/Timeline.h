#pragma once

#include "NewOrder.h"

#include <vector>
#include <cstddef>


namespace no {

class Timeline final
{
public:

  // Default "null" timeline is just one step of zero size
  Timeline();

  Timeline(double start, double end, const std::vector<size_t>& checkpoints);

  ~Timeline() = default;

  Timeline(const Timeline&) = delete;
  Timeline& operator=(const Timeline&) = default;
  Timeline(Timeline&&) = delete;
  Timeline& operator=(Timeline&&) = default;

  double time() const;
  size_t index() const;

  double start() const;
  double end() const;
  size_t nsteps() const;

  double dt() const;
  const std::vector<size_t>& checkpoints() const;

  void next();

  bool at_checkpoint() const;

  bool at_end() const;

  // returns a floating point number that compares unequal (and unordered) to any other number
  // thus the following all evaluate to true: never() != never(), !(x < never()), !(x >= never()) (so be careful!)
  static double never();

  // this MUST be used to correctly compare against never since NaN != NaN
  static bool isnever(double t);

  // returns a floating point number that compares less than any other number
  static double distant_past();

  // returns a floating point number that compares greater than any other number
  static double far_future();

private:
  std::vector<size_t> m_checkpoints;

  double m_start; 
  double m_end;
  double m_dt; // timestep

  size_t m_index; // index of current time
  double m_time; // current time
};

}