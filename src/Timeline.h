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

  Timeline(const Timeline&) = default;
  Timeline& operator=(const Timeline&) = default;
  Timeline(Timeline&&) = default;
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

  // used by python __repr__
  std::string repr() const;

private:
  double m_start; 
  double m_end;

  size_t m_index; // index of current time
  
  std::vector<size_t> m_checkpoints;
};

namespace time {
  
  // returns a floating point number that compares unequal (and unordered) to any other number
  // thus the following all evaluate to true: never() != never(), !(x < never()), !(x >= never()) (so be careful!)
  double never();

  // this MUST be used to correctly compare against never since NaN != NaN
  bool isnever(double t);

  // returns a floating point number that compares less than any other number
  double distant_past();

  // returns a floating point number that compares greater than any other number
  double far_future();
}

}