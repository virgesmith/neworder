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

  virtual size_t index() const = 0;
  virtual size_t nsteps() const = 0;
  virtual double dt() const = 0;

  virtual void next() = 0;

  virtual bool at_checkpoint() const = 0;
  virtual bool at_end() const = 0;

  // used by python __repr__
  virtual std::string repr() const = 0;

};

// An empty (one arbitrary step) timeline. The model's step and checkpoint method will each be called once only
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

  size_t index() const;
  size_t nsteps() const;
  double dt() const;

  virtual void next();
  //const std::vector<size_t>& checkpoints() const { static std::vector<size_t> checkpoints{1}; return checkpoints; }

  bool at_checkpoint() const;
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

  LinearTimeline(double start, double end, const std::vector<size_t>& checkpoints);

  virtual ~LinearTimeline() = default;

  LinearTimeline(const LinearTimeline&) = default;
  LinearTimeline& operator=(const LinearTimeline&) = default;
  LinearTimeline(LinearTimeline&&) = default;
  LinearTimeline& operator=(LinearTimeline&&) = default;

  py::object time() const;
  py::object start() const;
  py::object end() const;

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
  size_t m_index;
  double m_start;
  double m_end;
  std::vector<size_t> m_checkpoints;
};


// A generic numeric timeline, the model developer supplies the entire timeline and the checkpoints
class NEWORDER_EXPORT NumericTimeline final : public Timeline
{
public:
  NumericTimeline(const std::vector<double>& times, const std::vector<size_t>& checkpoints);

  virtual ~NumericTimeline() = default;

  NumericTimeline(const NumericTimeline&) = default;
  NumericTimeline& operator=(const NumericTimeline&) = default;
  NumericTimeline(NumericTimeline&&) = default;
  NumericTimeline& operator=(NumericTimeline&&) = default;

  py::object time() const;
  py::object start() const;
  py::object end() const;

  size_t index() const;
  size_t nsteps() const;
  double dt() const;
  //const std::vector<size_t>& checkpoints() const;

  void next();

  bool at_checkpoint() const;
  bool at_end() const;

  std::string repr() const;

private:
  size_t m_index;
  std::vector<double> m_times;
  std::vector<size_t> m_checkpoints;
};

// A timeline based on calendar dates and intervals (no intraday resolution, ignores DST adjustments)
class NEWORDER_EXPORT CalendarTimeline final : public Timeline
{
public:
  using time_point = std::chrono::system_clock::time_point;

  // TODO specify checkpoints (as multiple of steps)
  CalendarTimeline(time_point start, time_point end, size_t step, char unit, size_t n_checkpoints);

  virtual ~CalendarTimeline() = default;

  CalendarTimeline(const CalendarTimeline&) = default;
  CalendarTimeline& operator=(const CalendarTimeline&) = default;
  CalendarTimeline(CalendarTimeline&&) = default;
  CalendarTimeline& operator=(CalendarTimeline&&) = default;

  py::object time() const;
  py::object start() const;
  py::object end() const;

  size_t index() const;
  size_t nsteps() const;
  double dt() const;
  //const std::vector<size_t>& checkpoints() const;

  void next();

  bool at_checkpoint() const;
  bool at_end() const;

  std::string repr() const;

private:
  size_t m_index;
  std::vector<time_point> m_times;
  std::vector<size_t> m_checkpoints;
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