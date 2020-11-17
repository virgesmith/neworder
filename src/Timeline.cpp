
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
  // negative timesteps are disallowed as MC functions will misbehave with dt<0
  if (end < start) // end==start IS valid (for a null timeline)
  {
    throw py::value_error("end time (%%) must not be before start time (%%)"s % m_end % m_start);
  }

  if (n < 1)
  {
    throw py::value_error("checkpoints must contain at least one value (the last step)");
  }

  // validate checkpoints monotonic and on timeline
  for (size_t i = 1; i < m_checkpoints.size(); ++i)
  {
    if (m_checkpoints[i] <= m_checkpoints[i-1])
    {
      throw py::value_error("invalid timeline: checkpoint %% (%%) is not strictly greater than previous (%%)"s
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
  return "<neworder.Timeline start=%% end=%% checkpoints=%% index=%%>"s
          % start() % end() % checkpoints() % index();
}


namespace {

// for incrementing time in months preserving day of month
// e.g. passing 2020,1 will return 29 (leap)
int daysInFollowingMonth(int year, int month)
{
  month +=1;
  if (month == 12)
  {
    year += 1;
    month -= 12;
  }
  static const int days[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int d = days[month];
  if (month == 1 && ((year % 100 != 0) ^ (year % 400 == 0)) && (year % 4 == 0)) // February of a leap year
    ++d;
  return d;
}

}


no::CalendarTimeline::CalendarTimeline(time_point start, time_point end) : m_index(0)
{
  //size_t i = 0;
  m_times.push_back(start);
  time_point time = start;

  for(;;)
  {
    std::time_t t = std::chrono::system_clock::to_time_t(time);
    tm* local_tm = std::localtime(&t);
    // track whether we cross a DST change
    int dst_prev = local_tm->tm_isdst;
    local_tm->tm_mday += daysInFollowingMonth(local_tm->tm_year, local_tm->tm_mon);
    std::mktime(local_tm);
    //no::log("h: %% dst: %% prev: %%"s % local_tm->tm_hour % local_tm->tm_isdst % dst_prev);

    // adjust so that hour of day is preserved across DST changes
    if (local_tm->tm_isdst == 0 && dst_prev == 1)
    {
      local_tm->tm_hour += 1;
    }
    else if (local_tm->tm_isdst == 1 && dst_prev == 0)
    {
      local_tm->tm_hour -= 1;
    }

    t = std::mktime(local_tm);

    time = std::chrono::system_clock::from_time_t(t);
    if (time >= end)
      break;
    m_times.push_back(time);
  }
  m_times.push_back(end);
}

no::CalendarTimeline::time_point no::CalendarTimeline::start() const
{
  return m_times.front();
}

no::CalendarTimeline::time_point no::CalendarTimeline::end() const
{
  return m_times.back();
}


size_t no::CalendarTimeline::index() const
{
  return m_index;
}


bool no::CalendarTimeline::at_end() const
{
  return m_index >= m_times.size();
}

no::CalendarTimeline::time_point no::CalendarTimeline::time() const
{
  return m_times[m_index];
}

void no::CalendarTimeline::next()
{
  ++m_index;
}

double no::CalendarTimeline::dt() const
{
  if (m_index < m_times.size() - 1)
    return std::chrono::duration_cast<std::chrono::seconds>(m_times[m_index+1] - m_times[m_index]).count();
  return 0.0;
}

size_t no::CalendarTimeline::nsteps() const
{
  return m_times.size();
}


// Sun=0, Mon=1 etc
int no::CalendarTimeline::dow() const
{
  std::time_t t = std::chrono::system_clock::to_time_t(m_times[m_index]);
  tm* local_tm = std::localtime(&t);
  return local_tm->tm_wday;
}

std::string no::CalendarTimeline::repr() const
{
  std::time_t t = std::chrono::system_clock::to_time_t(m_times[m_index]);
  std::string buf(64, 0);
  std::strftime(buf.data(), buf.size(), "%F %T %Z", std::localtime(&t));
  return std::string(buf);
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
