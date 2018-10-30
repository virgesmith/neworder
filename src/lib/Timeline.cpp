
#include "Timeline.h"
#include "Log.h"

#include "python.h"

// Default "null" timeline is just one step of arbitrary size
neworder::Timeline::Timeline() : m_checkpoints{1}, m_steps(1), m_begin(0.0), m_dt(0.0), m_index(0), m_time(0.0) 
{ 

}

neworder::Timeline::Timeline(const std::vector<double>& checkpoint_times, size_t nsteps) 
  : m_checkpoints(checkpoint_times.size()), m_steps(nsteps)
{
  size_t n = checkpoint_times.size();
  if (n < 2)
  {
    std::runtime_error("timeline specification is too short, must contain a minimum of (begin, end)");
  }

  // validate
  if (n < 1)
  {
    throw std::runtime_error("invalid timeline: nsteps (%%) is not strictly positive"_s % n);
  }
  m_begin = checkpoint_times[0];
  m_dt = (checkpoint_times.back() - checkpoint_times.front()) / m_steps;

  // for now checkpoint at closest timestep (step 0 is not a checkpoint)
  m_checkpoints.resize(checkpoint_times.size() - 1);
  // validate checkpoints monotonic and on timeline
  for (size_t i = 1; i < checkpoint_times.size(); ++i)
  {
    if (checkpoint_times[i] <= checkpoint_times[i-1])
    {
      throw std::runtime_error("invalid timeline: element %% (%%) is not strictly greater than previous (%%)"_s 
        % i % checkpoint_times[i] % checkpoint_times[i-1]);
    }
    m_checkpoints[i-1] = static_cast<int>((checkpoint_times[i] - checkpoint_times[0]) / m_dt);
  }

  // set to start 
  m_index = 0;
  m_time = m_begin;
}

double neworder::Timeline::time() const 
{ 
  return m_time; 
}

size_t neworder::Timeline::index() const 
{ 
  return m_index; 
}

double neworder::Timeline::dt() const 
{ 
  return m_dt; 
}

size_t neworder::Timeline::nsteps() const 
{ 
  return m_steps; 
}

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

// returns a floating point number that compares unequal to (and unordered w.r.t) any other number
// thus the following all evaluate to true: never() != never(), !(x < never()), !(x >= never()) (so be careful!)
double neworder::Timeline::never()
{
  return std::numeric_limits<double>::quiet_NaN();
}

// use this rather than direct comparison to never, as NaN != NaN (as above) 
bool neworder::Timeline::isnever(double t)
{
  return std::isnan(t);
}
