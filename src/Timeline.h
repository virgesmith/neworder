#pragma once

#include "NewOrder.h"

#include "Log.h"

#include <nanobind/trampoline.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace no {

// Abstract base class for timelines - implements the basic stepping functionality
class Timeline {
public:
  Timeline() : m_index(0) {}

  virtual ~Timeline() = default;

  // times may have different types, depending on implementation
  virtual nb::object time() const = 0;
  virtual nb::object start() const = 0;
  virtual nb::object end() const = 0;

  int64_t index() { return m_index; };
  virtual double dt() const = 0;

  virtual void _next() = 0;

  virtual bool at_end() const = 0;

  // used by python __repr__, default implementation
  virtual std::string repr() const {
    return "<%% index=%%>"s % nb::cast<std::string>(nb::find(*this).attr("__class__").attr("__name__")) % m_index;
  }

protected:
  size_t m_index;

private:
  friend class Model;
  // this is called internally to ensure index is incremented
  void next() {
    ++m_index;
    _next();
  }
};

class PyTimeline : private Timeline {
  NB_TRAMPOLINE(Timeline);
  using Timeline::operator=;

  // trampoline methods
  nb::object time() const override { NB_OVERRIDE_PURE(time); }
  nb::object start() const override { NB_OVERRIDE_PURE(start); }
  nb::object end() const override { NB_OVERRIDE_PURE(end); }

  double dt() const override { NB_OVERRIDE_PURE(dt); }

  void _next() override { NB_OVERRIDE_PURE(_next); }

  bool at_end() const override { NB_OVERRIDE_PURE(at_end); }
  std::string repr() const override { NB_OVERRIDE_NAME("__repr__", repr); }
};

// An empty (one arbitrary step) timeline. The model's step method will each be called once only
class NEWORDER_EXPORT NoTimeline final : public Timeline {
public:
  NoTimeline() {}

  ~NoTimeline() override = default;
  NoTimeline(const NoTimeline&) = default;
  NoTimeline& operator=(const NoTimeline&) = default;
  NoTimeline(NoTimeline&&) = default;
  NoTimeline& operator=(NoTimeline&&) = default;

  // the actual types may differ in derived classes
  nb::object time() const override;
  nb::object start() const override;
  nb::object end() const override;

  double dt() const override;

  void _next() override;

  bool at_end() const override;

  std::string repr() const override;
};

// An equally-spaced timeline between 2 numeric time points
class NEWORDER_EXPORT LinearTimeline final : public Timeline {
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

  nb::object time() const override;
  nb::object start() const override;
  nb::object end() const override;

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
class NEWORDER_EXPORT NumericTimeline final : public Timeline {
public:
  NumericTimeline(const std::vector<double>& times);

  ~NumericTimeline() override = default;

  NumericTimeline(const NumericTimeline&) = default;
  NumericTimeline& operator=(const NumericTimeline&) = default;
  NumericTimeline(NumericTimeline&&) = default;
  NumericTimeline& operator=(NumericTimeline&&) = default;

  nb::object time() const override;
  nb::object start() const override;
  nb::object end() const override;

  double dt() const override;

  void _next() override;

  bool at_end() const override;

  std::string repr() const override;

private:
  std::vector<double> m_times;
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
} // namespace time

} // namespace no