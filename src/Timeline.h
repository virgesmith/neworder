#pragma once

#include "NewOrder.h"

#include "Log.h"

#include <cstdint>

namespace no {

// Abstract base class for timelines. Implementations (in python) are iterators over the timeline's steps, each step
// being a (start, end) interval of the implementation's native type (e.g. date). Regardless of the native type, time
// and dt are always floats so they can be used in computations, e.g. with the MonteCarlo functions
class Timeline {
public:
  Timeline() = default;

  virtual ~Timeline() = default;

  virtual int64_t index() const = 0;

  // the start, before any steps are taken, then the end of the most recent step
  virtual double time() const = 0;

  // the length of the most recent step
  virtual double dt() const = 0;

  // takes the next step and returns it as a (start, end) tuple of native values, or raises StopIteration
  virtual py::object next() = 0;

  // used by python __repr__, default implementation
  virtual std::string repr() const {
    return "<%% index=%%>"s % py::cast(this).attr("__class__").attr("__name__").cast<std::string>() % index();
  }
};

class PyTimeline : public Timeline {
  using Timeline::Timeline;
  using Timeline::operator=;

  // PYBIND11_OVERRIDE can't be used for properties: it would fetch the property's value rather than a callable, and
  // would recurse infinitely if the property isn't overridden. So check explicitly that the python subclass overrides
  // the (abstract) attribute. NB caller must hold the GIL
  py::handle overriding_self(const char* name) const {
    py::handle self =
        py::detail::get_object_handle(static_cast<const Timeline*>(this), py::detail::get_type_info(typeid(Timeline)));
    if (py::type::handle_of(self).attr(name).is(py::type::of<Timeline>().attr(name))) {
      throw py::type_error("%% must implement the abstract member '%%'"s %
                           py::type::handle_of(self).attr("__name__").cast<std::string>() % name);
    }
    return self;
  }

  // trampoline methods
  int64_t index() const override {
    py::gil_scoped_acquire gil;
    return overriding_self("index").attr("index").cast<int64_t>();
  }

  double time() const override {
    py::gil_scoped_acquire gil;
    return overriding_self("time").attr("time").cast<double>();
  }

  double dt() const override {
    py::gil_scoped_acquire gil;
    return overriding_self("dt").attr("dt").cast<double>();
  }

  py::object next() override {
    py::gil_scoped_acquire gil;
    return overriding_self("__next__").attr("__next__")();
  }

  std::string repr() const override { PYBIND11_OVERRIDE_NAME(std::string, Timeline, "__repr__", repr); }
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