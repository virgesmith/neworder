#include "Timeline.h"
#include "Log.h"

#include <algorithm>

py::object no::NoTimeline::time() const { return py::float_(time::never()); }
py::object no::NoTimeline::start() const { return py::float_(time::never()); }
py::object no::NoTimeline::end() const { return py::float_(time::never()); }

double no::NoTimeline::dt() const { return 0.0; }

void no::NoTimeline::_next() { /* nothing to do, base class increments index */ }

bool no::NoTimeline::at_end() const { return m_index > 0; }

// used by python __repr__
std::string no::NoTimeline::repr() const { return "<neworder.NoTimeline stepped=%%>"s % (m_index > 0 ? "True": "False"); }


no::LinearTimeline::LinearTimeline(double start, double end, size_t steps)
  : m_start(start), m_end(end), m_steps(steps)
{
  // validate
  // negative timesteps are disallowed as MC functions will misbehave with dt<0
  if (m_end <= m_start)
  {
    throw py::value_error("end time (%%) must be after the start time (%%)"s % m_end % m_start);
  }

  if (m_steps < 1)
  {
    throw py::value_error("timeline must have at least one step");
  }

  // set to start
  m_index = 0;
  m_dt = (m_end - m_start) / m_steps;
}

no::LinearTimeline::LinearTimeline(double start, double step)
  : m_start(start), m_end(no::time::far_future()), m_dt(step), m_steps(-1)
{
  // validate
  if (m_dt <= 0.0)
  {
    throw py::value_error("timeline must have a positive step size, got %%"s % m_dt);
  }
}

double no::LinearTimeline::dt() const
{
  return m_dt;
}

void no::LinearTimeline::_next()
{
}

bool no::LinearTimeline::at_end() const
{
  return m_steps > 0 && m_index >= m_steps;
}

py::object no::LinearTimeline::time() const { return py::float_(m_start + m_dt * m_index); }
py::object no::LinearTimeline::start() const { return py::float_(m_start); }
py::object no::LinearTimeline::end() const { return py::float_(m_end); }


std::string no::LinearTimeline::repr() const
{
  return m_end == no::time::far_future()
    ? "<neworder.LinearTimeline start=%% end=never dt=%% steps=inf time=%% index=%%>"s % m_start % m_dt % time() % m_index
    : "<neworder.LinearTimeline start=%% end=%% dt=%% steps=%% time=%% index=%%>"s % m_start % m_end % m_dt % m_steps % time() % m_index;
}


no::NumericTimeline::NumericTimeline(const std::vector<double>& times)
  : m_times(times)
{
  if (m_times.size() < 2)
  {
    throw py::value_error("timeline must have at least 2 points");
  }
  // check ascending
  for (size_t i = 1; i < m_times.size(); ++i)
  {
    if (m_times[i] <= m_times[i-1])
    {
      throw py::value_error("invalid timeline: time at index %% (%%) is not strictly greater than previous (%%)"s
        % i % m_times[i] % m_times[i-1]);
    }
  }
}

py::object no::NumericTimeline::time() const
{
  return at_end() ? end() : py::float_(m_times[m_index]);
}

py::object no::NumericTimeline::start() const
{
  return py::float_(m_times.front());
}

py::object no::NumericTimeline::end() const
{
  return py::float_(m_times.back());
}

double no::NumericTimeline::dt() const
{
  if (m_index >= m_times.size() - 1)
    return 0.0;
  return m_times[m_index+1] - m_times[m_index];
}

void no::NumericTimeline::_next()
{
}

bool no::NumericTimeline::at_end() const
{
  return m_index >= m_times.size() - 1;
}


std::string no::NumericTimeline::repr() const
{
  return "<neworder.NumericTimeline times=%% steps=%% time=%% index=%%>"s
          % m_times % (m_times.size() - 1) % time() % m_index;
}


// returns a floating point number that compares less than any other number
double no::time::distant_past()
{
  return -std::numeric_limits<double>::infinity();
}

// returns a floating point number that compares greater than any other number
double no::time::far_future()
{
  return std::numeric_limits<double>::infinity();
}

// returns a floating point number that compares unequal to (and unordered w.r.t) any other number
// thus the following all evaluate to true: never() != never(), !(x < never()), !(x >= never()) (so be careful!)
double no::time::never()
{
  return std::numeric_limits<double>::quiet_NaN();
}

// use this rather than direct comparison to never, as NaN != NaN (as above)
bool no::time::isnever(double t)
{
  return std::isnan(t);
}
