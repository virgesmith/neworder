
#include "Timeline.h"
#include "Log.h"


// Default "null" timeline is just one step of arbitrary size
neworder::Timeline::Timeline() : m_checkpoints{0.0, 1.0}, m_steps(1), m_time(0.0), m_index(0) { }

neworder::Timeline::Timeline(double begin, double end, int n) : m_checkpoints{begin, end}, m_steps(n), m_time(begin), m_index(0) 
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
}

double neworder::Timeline::time() { return m_time; }
size_t neworder::Timeline::index() { return m_index; }

double neworder::Timeline::dt() { return m_dt; }

void neworder::Timeline::step()
{
  if (m_time < m_checkpoints.back())
  {
    m_time += m_dt;
    ++m_index;
  }
}

bool neworder::Timeline::checkpoint()
{
  // TODO...
  // for now only end is a checkpoint
  return end();
}

bool neworder::Timeline::end()
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

