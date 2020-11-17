#pragma once

#include "NewOrder.h"

#include "Log.h"

#include <pybind11/chrono.h>

#include <vector>
#include <chrono>
#include <cstddef>


namespace no {

// Abstract base class for timelines - implements the basic stepping functionality
class Timeline
{
public:

  Timeline() { }

  virtual ~Timeline() = default;

  Timeline(const Timeline&) = default;
  Timeline& operator=(const Timeline&) = default;
  Timeline(Timeline&&) = default;
  Timeline& operator=(Timeline&&) = default;

  // for logging only
  virtual std::string time() const { return "n/a"; }
  virtual std::string start() const { return "n/a"; }
  virtual std::string end() const { return "n/a"; }

  virtual size_t index() const = 0;
  virtual size_t nsteps() const = 0;
  virtual double dt() const = 0;

  virtual void next() = 0;

  virtual bool at_checkpoint() const = 0;
  virtual bool at_end() const = 0;

  // used by python __repr__
  virtual std::string repr() const = 0;

};

class NEWORDER_EXPORT NoTimeline : public Timeline
{
public:
  NoTimeline() : m_stepped(false) { }

  std::string time() const { return "n/a"; }
  std::string start() const { return "n/a"; }
  std::string end() const { return "n/a"; }

  size_t index() const { return static_cast<size_t>(m_stepped); }
  size_t nsteps() const { return 1; }
  double dt() const { return 0.0; }

  virtual void next() { m_stepped = true; }
  //const std::vector<size_t>& checkpoints() const { return {1}; }

  bool at_checkpoint() const { return m_stepped; }
  bool at_end() const { return m_stepped; }

  // used by python __repr__
  std::string repr() const { return "<NoTimeline>"; }

private:
  bool m_stepped;

};

class NEWORDER_EXPORT LinearTimeline final : public Timeline
{
public:

  LinearTimeline(double start, double end, const std::vector<size_t>& checkpoints);

  virtual ~LinearTimeline() = default;

  LinearTimeline(const LinearTimeline&) = default;
  LinearTimeline& operator=(const LinearTimeline&) = default;
  LinearTimeline(LinearTimeline&&) = default;
  LinearTimeline& operator=(LinearTimeline&&) = default;

  std::string time() const;
  std::string start() const;
  std::string end() const;

  size_t index() const;
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


class NEWORDER_EXPORT CalendarTimeline : public Timeline
{
public:
  using time_point = std::chrono::system_clock::time_point;

  // TODO specify time increment in days, months or years
  // TODO specify checkpoints (as multiple of steps)
  CalendarTimeline(time_point start, time_point end);

  std::string time() const;
  std::string start() const;
  std::string end() const;

  size_t index() const;
  size_t nsteps() const;
  double dt() const;
  //const std::vector<size_t>& checkpoints() const;

  void next();

  bool at_checkpoint() const { return false; }
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