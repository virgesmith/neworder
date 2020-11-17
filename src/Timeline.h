#pragma once

#include "NewOrder.h"

#include "Log.h"

#include <pybind11/chrono.h>

#include <vector>
#include <chrono>
#include <cstddef>


namespace no {

// // Abstract base class for timelines
// template<typename T>
// class Timeline
// {
// public:
//   virtual ~Timeline() = default;

//   T time() const;
//   size_t index() const;

//   T start() const;
//   T end() const;
//   size_t nsteps() const;

//   double dt() const;
//   const std::vector<size_t>& checkpoints() const;

//   void next();

//   bool at_checkpoint() const;

//   bool at_end() const;

//   // used by python __repr__
//   std::string repr() const;


// }

class NEWORDER_EXPORT Timeline final //: public Timeline<double>
{
public:

  // Default "null" timeline is just one step of zero size
  Timeline();

  Timeline(double start, double end, const std::vector<size_t>& checkpoints);

  virtual ~Timeline() = default;

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


class NEWORDER_EXPORT CalendarTimeline
{
public:
  using time_point = std::chrono::system_clock::time_point;

  // TODO specify time increment in days, months or years
  // TODO specify checkpoints (as multiple of steps)
  CalendarTimeline(time_point start, time_point end);

  time_point time() const;
  size_t index() const;

  time_point start() const;
  time_point end() const;
  size_t nsteps() const;

  double dt() const;
  //const std::vector<size_t>& checkpoints() const;

  void next();

  //bool at_checkpoint() const;
  bool at_end() const;

  // weekday of current time point. Sun=0, Mon=1 etc
  int dow() const;

  std::string repr() const;

private:
  size_t m_index;
  std::vector<time_point> m_times;
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