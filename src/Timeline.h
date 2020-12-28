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

  // times may have different types, depending on implementation
  virtual py::object time() const = 0;
  virtual py::object start() const = 0;
  virtual py::object end() const = 0;

  virtual int64_t index() const = 0;
  virtual int64_t nsteps() const = 0;
  virtual double dt() const = 0;

  virtual void next() = 0;

  virtual bool at_end() const = 0;

  // used by python __repr__
  virtual std::string repr() const = 0;

};

// An empty (one arbitrary step) timeline. The model's step method will each be called once only
class NEWORDER_EXPORT NoTimeline final : public Timeline
{
public:
  NoTimeline() : m_stepped(false) { }

  virtual ~NoTimeline() = default;
  NoTimeline(const NoTimeline&) = default;
  NoTimeline& operator=(const NoTimeline&) = default;
  NoTimeline(NoTimeline&&) = default;
  NoTimeline& operator=(NoTimeline&&) = default;

  // the actual types may differ in derived classes
  py::object time() const;
  py::object start() const;
  py::object end() const;

  int64_t index() const;
  int64_t nsteps() const;
  double dt() const;

  virtual void next();

  bool at_end() const;

  // used by python __repr__
  std::string repr() const;

private:
  // flag whether we've done the arbitrary step
  bool m_stepped;
};

// An equally-spaced timeline between 2 numeric time points
class NEWORDER_EXPORT LinearTimeline final : public Timeline
{
public:

  // Fixed length timeline
  LinearTimeline(double start, double end, size_t steps);

  // Open-ended timeline. Requires a call to halt() in the step() method to terminate the model run
  LinearTimeline(double start, double step);

  virtual ~LinearTimeline() = default;

  LinearTimeline(const LinearTimeline&) = default;
  LinearTimeline& operator=(const LinearTimeline&) = default;
  LinearTimeline(LinearTimeline&&) = default;
  LinearTimeline& operator=(LinearTimeline&&) = default;

  py::object time() const;
  py::object start() const;
  py::object end() const;

  int64_t index() const;
  int64_t nsteps() const;
  double dt() const;

  void next();

  bool at_end() const;

  // used by python __repr__
  std::string repr() const;

private:
  size_t m_index;
  double m_start;
  double m_end;
  double m_dt; // store both dt and steps as (re)computing steps from dt is prone to rounding errors
  size_t m_steps;
};


// A generic numeric timeline, the model developer supplies the entire timeline
class NEWORDER_EXPORT NumericTimeline final : public Timeline
{
public:
  NumericTimeline(const std::vector<double>& times);

  virtual ~NumericTimeline() = default;

  NumericTimeline(const NumericTimeline&) = default;
  NumericTimeline& operator=(const NumericTimeline&) = default;
  NumericTimeline(NumericTimeline&&) = default;
  NumericTimeline& operator=(NumericTimeline&&) = default;

  py::object time() const;
  py::object start() const;
  py::object end() const;

  int64_t index() const;
  int64_t nsteps() const;
  double dt() const;

  void next();

  bool at_end() const;

  std::string repr() const;

private:
  size_t m_index;
  std::vector<double> m_times;
};

// A timeline based on calendar dates and intervals (no intraday resolution, ignores DST adjustments)
class NEWORDER_EXPORT CalendarTimeline final : public Timeline
{
public:
  using time_point = std::chrono::system_clock::time_point;

  // Fixed-end
  CalendarTimeline(time_point start, time_point end, size_t step, char unit);

  // Open-ended
  CalendarTimeline(time_point start, size_t step, char unit);

  virtual ~CalendarTimeline() = default;

  CalendarTimeline(const CalendarTimeline&) = default;
  CalendarTimeline& operator=(const CalendarTimeline&) = default;
  CalendarTimeline(CalendarTimeline&&) = default;
  CalendarTimeline& operator=(CalendarTimeline&&) = default;

  py::object time() const;
  py::object start() const;
  py::object end() const;

  int64_t index() const;
  int64_t nsteps() const;
  double dt() const;

  void next();

  bool at_end() const;

  std::string repr() const;

private:

  // advance to next point
  time_point advance(const time_point& time) const;

  size_t m_index;
  size_t m_step;
  char m_unit;
  int m_refDay;

  // this caches timesteps when the end is known (so we know total number of steps)
  std::vector<time_point> m_times;
  // otherwise just store the current step start and end, and a reference day (for monthly increments)
  std::tuple<time_point, time_point> m_currentStep;
};

namespace time {

  // returns a floating point number that compares unequal to (and unordered w.r.t) any other number
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