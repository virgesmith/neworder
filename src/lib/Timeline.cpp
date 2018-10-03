
#include "Timeline.h"
#include "Log.h"

#include "python.h"

// Default "null" timeline is just one step of arbitrary size
neworder::Timeline::Timeline() : m_checkpoints{1}, m_steps(1), m_begin(0.0), m_index(0), m_time(0.0) 
{ 

}

neworder::Timeline::Timeline(double begin, double end, int n) : m_checkpoints{(size_t)n}, m_steps(n), m_begin(begin), m_index(0), m_time(begin) 
{ 
  // validate
  if (n < 1)
  {
    throw std::runtime_error("invalid timeline: nsteps (%%) is not strictly positive"_s % n);
  }
  if (begin >= end)
  {
    throw std::runtime_error("invalid timeline: begin (%%) is not strictly before end (%%)"_s % begin % end);
  }
  // 0th timestep is at begin, nth timestep is at end
  m_dt = (end-begin) / n; 
  neworder::log("checkpoints=%%"_s % m_checkpoints);
}

// TODO move out of class?
neworder::Timeline::Timeline(const py::tuple& spec) 
{
  size_t n = py::len(spec);
  if (n < 3)
  {
    std::runtime_error("timeline spec is too short, must contain a minimum of (begin, end, steps)");
  }
  std::vector<double> checkpoint_times(n - 1);
  for (size_t i = 0; i < n - 1; ++i)
  {
    checkpoint_times[i] = py::extract<double>(spec[i]);
  }

  m_steps = py::extract<int>(spec[n-1]);
  if (m_steps < 1)
  {
    throw std::runtime_error("invalid timeline: nsteps (%%) is not strictly positive"_s % m_steps);
  }

  m_dt = (checkpoint_times.back() - checkpoint_times.front()) / m_steps;

  // for now checkpoint at closest timestep
  m_checkpoints.resize(checkpoint_times.size());
  m_checkpoints[0] = 0;
  // validate checkpoints monotonic and on timeline
  for (size_t i = 1; i < checkpoint_times.size(); ++i)
  {
    if (checkpoint_times[i] <= checkpoint_times[i-1])
    {
      throw std::runtime_error("invalid timeline: element %% (%%) is not strictly greater than previous (%%)"_s 
        % i % checkpoint_times[i] % checkpoint_times[i-1]);
    }
    m_checkpoints[i] = static_cast<int>((checkpoint_times[i] - checkpoint_times[0]) / m_dt);
  }
  neworder::log("checkpoints=%%"_s % m_checkpoints);
}

double neworder::Timeline::time() const { return m_time; }
size_t neworder::Timeline::index() const { return m_index; }

double neworder::Timeline::dt() const { return m_dt; }

void neworder::Timeline::step()
{
  if (m_index < m_checkpoints.back())
  {
    m_time += m_dt;
    ++m_index;
  }
}

bool neworder::Timeline::is_checkpoint() const
{
  return std::find(m_checkpoints.begin(), m_checkpoints.end(), m_index) != m_checkpoints.end();
}

const std::vector<size_t>& neworder::Timeline::checkpoints() const
{
  return m_checkpoints;
}

bool neworder::Timeline::end() const
{
  return m_index == m_steps;
}

// returns a floating point number that compares less than any other number
double neworder::Timeline::distant_past()
{
  return -std::numeric_limits<double>::max();
}

// returns a floating point number that compares greater than any other number
double neworder::Timeline::far_future()
{
  return std::numeric_limits<double>::max();
}

