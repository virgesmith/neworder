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

  Timeline() : m_index(0) { }

  virtual ~Timeline() = default;

  // times may have different types, depending on implementation
  virtual py::object time() const = 0;
  virtual py::object start() const = 0;
  virtual py::object end() const = 0;

  int64_t index() { return m_index; };
  virtual int64_t nsteps() const = 0;
  virtual double dt() const = 0;

  virtual void _next() = 0;

  virtual bool at_end() const = 0;

  // used by python __repr__, default implementation
  virtual std::string repr() const {
    return "<%% index=%%>"s % py::cast(this).attr("__class__").attr("__name__").cast<std::string>() % m_index;
  }

protected:
  size_t m_index;

private:
  friend class Model;
  // this is called internally to ensure index is incremented
  void next()
  {
    ++m_index;
    _next();
  }
};

class PyTimeline: public Timeline
{
  using Timeline::Timeline;
  using Timeline::operator=;

  // trampoline methods
  py::object time() const override { PYBIND11_OVERRIDE_PURE(py::object, Timeline, time); }
  py::object start() const override { PYBIND11_OVERRIDE_PURE(py::object, Timeline, start); }
  py::object end() const override { PYBIND11_OVERRIDE_PURE(py::object, Timeline, end); }

  int64_t nsteps() const override { PYBIND11_OVERRIDE_PURE(int64_t, Timeline, nsteps); }
  double dt() const override { PYBIND11_OVERRIDE_PURE(double, Timeline, dt); }

  void _next() override { PYBIND11_OVERRIDE_PURE(void, Timeline, _next); }

  bool at_end() const override { PYBIND11_OVERRIDE_PURE(bool, Timeline, at_end); }
  std::string repr() const override { PYBIND11_OVERRIDE_NAME(std::string, Timeline, "__repr__", repr); }
};


// An empty (one arbitrary step) timeline. The model's step method will each be called once only
class NEWORDER_EXPORT NoTimeline final : public Timeline
{
public:
  NoTimeline() { }

  ~NoTimeline() override = default;
  NoTimeline(const NoTimeline&) = default;
  NoTimeline& operator=(const NoTimeline&) = default;
  NoTimeline(NoTimeline&&) = default;
  NoTimeline& operator=(NoTimeline&&) = default;

  // the actual types may differ in derived classes
  py::object time() const override;
  py::object start() const override;
  py::object end() const override;

  int64_t nsteps() const override;
  double dt() const override;

  void _next() override;

  bool at_end() const override;

  std::string repr() const override;
};

// An equally-spaced timeline between 2 numeric time points
class NEWORDER_EXPORT LinearTimeline final : public Timeline
{
public:

  // Fixed length timeline
  LinearTimeline(double start, double end, size_t steps);

  // Open-ended timeline. Requires a call to halt() in the step() method to terminate the model run
  LinearTimeline(double start, double step);

  ~LinearTimeline() override = default;

  LinearTimeline(const LinearTimeline&) = default;
  LinearTimeline& operator=(const LinearTimeline&) = default;
  LinearTimeline(LinearTimeline&&) = default;
  LinearTimeline& operator=(LinearTimeline&&) = default;

  py::object time() const override;
  py::object start() const override;
  py::object end() const override;

  int64_t nsteps() const override;
  double dt() const override;

  void _next() override;

  bool at_end() const override;

  std::string repr() const override;

private:
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

  ~NumericTimeline() override = default;

  NumericTimeline(const NumericTimeline&) = default;
  NumericTimeline& operator=(const NumericTimeline&) = default;
  NumericTimeline(NumericTimeline&&) = default;
  NumericTimeline& operator=(NumericTimeline&&) = default;

  py::object time() const override;
  py::object start() const override;
  py::object end() const override;

  int64_t nsteps() const override;
  double dt() const override;

  void _next() override;

  bool at_end() const override;

  std::string repr() const override;

private:
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

  ~CalendarTimeline() override = default;

  CalendarTimeline(const CalendarTimeline&) = default;
  CalendarTimeline& operator=(const CalendarTimeline&) = default;
  CalendarTimeline(CalendarTimeline&&) = default;
  CalendarTimeline& operator=(CalendarTimeline&&) = default;

  py::object time() const override;
  py::object start() const override;
  py::object end() const override;

  int64_t nsteps() const override;
  double dt() const override;

  void _next() override;

  bool at_end() const override;

  std::string repr() const override;

private:

  // advance to next point
  time_point advance(const time_point& time) const;

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