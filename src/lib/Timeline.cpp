
#include "Timeline.h"
#include "Log.h"

#include "NewOrder.h"

#include <algorithm>

// Default "null" timeline is just one step of arbitrary size
no::Timeline::Timeline() : m_start(0.0), m_end(0.0), m_index(0), m_checkpoints{1} 
{ 
}

no::Timeline::Timeline(double start, double end, const std::vector<size_t>& checkpoints)
  : m_start(start), m_end(end), m_checkpoints(checkpoints)
{
  size_t n = m_checkpoints.size();
  // validate
  if (n < 1)
  {
    py::value_error("checkpoints must contain at least one value (the last step)");
  }

  // validate checkpoints monotonic and on timeline
  for (size_t i = 1; i < m_checkpoints.size(); ++i)
  {
    if (m_checkpoints[i] <= m_checkpoints[i-1])
    {
      throw py::value_error("invalid timeline: checkpoint %% (%%) is not strictly greater than previous (%%)"_s 
        % i % m_checkpoints[i] % m_checkpoints[i-1]);
    }
  }

  // set to start 
  m_index = 0;
}

double no::Timeline::start() const
{
  return m_start;
}

double no::Timeline::end() const
{
  return m_end;
}


double no::Timeline::time() const 
{ 
  return m_start + m_index * dt(); 
}

size_t no::Timeline::index() const 
{ 
  return m_index;
}

double no::Timeline::dt() const 
{ 
  return (m_end - m_start) / m_checkpoints.back();
}

size_t no::Timeline::nsteps() const 
{ 
  return m_checkpoints.back(); 
}

void no::Timeline::next()
{
  if (m_index < m_checkpoints.back())
  {
    ++m_index;
  }
}

bool no::Timeline::at_checkpoint() const
{
  return std::find(m_checkpoints.begin(), m_checkpoints.end(), m_index) != m_checkpoints.end();
}

const std::vector<size_t>& no::Timeline::checkpoints() const
{
  return m_checkpoints;
}

bool no::Timeline::at_end() const
{
  return m_index == m_checkpoints.back();
}

std::string no::Timeline::repr() const
{
  return "<neworder.Timeline start=%% end=%% checkpoints=%% index=%%>"_s 
          % start() % end() % checkpoints() % index();
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
