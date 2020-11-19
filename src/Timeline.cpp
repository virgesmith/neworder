
#include "Timeline.h"
#include "Log.h"

#include "NewOrder.h"

#include <pybind11/chrono.h>

#include <algorithm>


py::object no::NoTimeline::time() const { return py::float_(time::never()); }
py::object no::NoTimeline::start() const { return py::float_(time::never()); }
py::object no::NoTimeline::end() const { return py::float_(time::never()); }

size_t no::NoTimeline::index() const { return static_cast<size_t>(m_stepped); }
size_t no::NoTimeline::nsteps() const { return 1; }
double no::NoTimeline::dt() const { return 0.0; }

void no::NoTimeline::next() { m_stepped = true; }

bool no::NoTimeline::at_checkpoint() const { return m_stepped; }
bool no::NoTimeline::at_end() const { return m_stepped; }

// used by python __repr__
std::string no::NoTimeline::repr() const { return "<NoTimeline stepped=%%>"s % (m_stepped ? "True": "False"); }


no::LinearTimeline::LinearTimeline(double start, double end, const std::vector<size_t>& checkpoints)
  : m_index(0), m_start(start), m_end(end), m_checkpoints(checkpoints)
{
  size_t n = m_checkpoints.size();

  // validate
  // negative timesteps are disallowed as MC functions will misbehave with dt<0
  if (end <= start)
  {
    throw py::value_error("end time (%%) must be after the start time (%%)"s % m_end % m_start);
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

size_t no::LinearTimeline::index() const
{
  return m_index;
}

double no::LinearTimeline::dt() const
{
  return (m_end - m_start) / m_checkpoints.back();
}

size_t no::LinearTimeline::nsteps() const
{
  return m_checkpoints.back();
}

void no::LinearTimeline::next()
{
  if (m_index < m_checkpoints.back())
  {
    ++m_index;
  }
}

bool no::LinearTimeline::at_checkpoint() const
{
  return std::find(m_checkpoints.begin(), m_checkpoints.end(), m_index) != m_checkpoints.end();
}

const std::vector<size_t>& no::LinearTimeline::checkpoints() const
{
  return m_checkpoints;
}

bool no::LinearTimeline::at_end() const
{
  return m_index == m_checkpoints.back();
}

py::object no::LinearTimeline::time() const { return py::float_(m_start + dt() * m_index); }
py::object no::LinearTimeline::start() const { return py::float_(m_start); }
py::object no::LinearTimeline::end() const { return py::float_(m_end); }

std::string no::LinearTimeline::repr() const
{
  return "<neworder.LinearTimeline start=%% end=%% checkpoints=%% time=%% index=%%>"s
          % m_start % m_end % m_checkpoints % time() % m_index;
}


no::NumericTimeline::NumericTimeline(const std::vector<double>& times, const std::vector<size_t>& checkpoints)
  : m_index(0), m_times(times), m_checkpoints(checkpoints)
{
  if (m_times.size() < 2)
  {
    throw py::value_error("timeline must have 2 or more points");
  }
  if (m_checkpoints.empty())
  {
    throw py::value_error("checkpoints must have 1 or more values");
  }
  // check ascending
  for (size_t i = 1; i < m_times.size(); ++i)
  {
    if (m_times[i] <= m_times[i-1])
    {
      throw py::value_error("invalid timeline: time %% (%%) is not strictly greater than previous (%%)"s
        % i % m_times[i] % m_times[i-1]);
    }
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
  // ensure final time is a checkpoint
  if (m_checkpoints.back() != m_times.size() - 1)
  {
    throw py::value_error("final checkpoint (%%) does not correspond to the final timestep (index %%)"s % m_checkpoints.back() % (m_times.size() - 1));
  }
}

py::object no::NumericTimeline::time() const
{
  return py::float_(m_times[m_index]);
}

py::object no::NumericTimeline::start() const
{
  return py::float_(m_times.front());
}

py::object no::NumericTimeline::end() const
{
  return py::float_(m_times.back());
}

size_t no::NumericTimeline::index() const
{
  return m_index;
}

size_t no::NumericTimeline::nsteps() const
{
  return m_checkpoints.back();
}

double no::NumericTimeline::dt() const
{
  if (m_index == m_times.size() - 1)
    return 0.0;
  return m_times[m_index+1] - m_times[m_index];
}

//const std::vector<size_t>& no::NumericTimeline::checkpoints() const;

void no::NumericTimeline::next()
{
  if (m_index < m_checkpoints.back())
  {
    ++m_index;
  }
}

bool no::NumericTimeline::at_checkpoint() const
{
  return std::find(m_checkpoints.begin(), m_checkpoints.end(), m_index) != m_checkpoints.end();
}

bool no::NumericTimeline::at_end() const
{
  return m_index == m_checkpoints.back();
}

std::string no::NumericTimeline::repr() const
{
  return "<neworder.NumericTimeline times=%% checkpoints=%% time=%% index=%%>"s
          % m_times % m_checkpoints % m_times[m_index] % m_index;
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

no::CalendarTimeline::time_point addDays(no::CalendarTimeline::time_point time, size_t n)
{
  std::time_t t = std::chrono::system_clock::to_time_t(time);
  tm* local_tm = std::localtime(&t);
  // track whether we cross a DST change
  int dst_prev = local_tm->tm_isdst;
  local_tm->tm_mday += n;
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

  return std::chrono::system_clock::from_time_t(t);
}


no::CalendarTimeline::time_point addMonths(no::CalendarTimeline::time_point time, size_t n, int refDay)
{
  for (size_t i = 0; i < n; ++i)
  {
    std::time_t t = std::chrono::system_clock::to_time_t(time);
    tm* local_tm = std::localtime(&t);
    // track whether we cross a DST change
    int dst_prev = local_tm->tm_isdst;
    // ensure we dont go over end of next month
    int difm = daysInFollowingMonth(local_tm->tm_year, local_tm->tm_mon);
    if (local_tm->tm_mday > difm)
      local_tm->tm_mday = difm;
    else
      local_tm->tm_mday = refDay;
    local_tm->tm_mon += 1;
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
  }
  return time;
}

}


no::CalendarTimeline::CalendarTimeline(time_point start, time_point end, size_t step, char unit, size_t n_checkpoints) : m_index(0)
{
  if (start >= end)
  {
    throw py::value_error("start time (%%) must be after end time (%%)"s % py::cast(start) % py::cast(end));
  }

  unit = tolower(unit);
  if (unit != 'd' && unit != 'm' && unit != 'y')
  {
    throw py::value_error("invalid time unit '%%', must be one of D,d,M,m,Y,y"s % unit);
  }

  std::time_t t = std::chrono::system_clock::to_time_t(start);
  tm* local_tm = std::localtime(&t);
  int refDay = local_tm->tm_mday;

  m_times.push_back(start);
  time_point time = start;

  for(;;)
  {
    if (unit == 'd')
    {
      time = addDays(time, step);
    }
    else if (unit == 'm')
    {
      time = addMonths(time, step, refDay);
    }
    else if (unit == 'y')
    {
      time = addMonths(time, step * 12, refDay); // ensures we deal with leap years correctly
    }
    if (time >= end)
      break;
    m_times.push_back(time);
  }
  m_times.push_back(end);

  // now spread the checkpoints over the timeline
  if (n_checkpoints >= m_times.size())
  {
    throw py::value_error("too many checkpoints requested (%%), timeline only has %% steps"s % n_checkpoints % (m_times.size()-1));
  }

  m_checkpoints.reserve(n_checkpoints);
  double steps_per_checkpoint = static_cast<double>(m_times.size()-1) / n_checkpoints;
  for (size_t i = 0; i < n_checkpoints; ++i)
  {
    m_checkpoints.push_back((i + 1) * steps_per_checkpoint);
  }
  no::log("%%"s % m_checkpoints);
}

size_t no::CalendarTimeline::index() const
{
  return m_index;
}

bool no::CalendarTimeline::at_end() const
{
  return m_index == m_checkpoints.back();
}

void no::CalendarTimeline::next()
{
  if (m_index < m_checkpoints.back())
  {
    ++m_index;
  }
}

bool no::CalendarTimeline::at_checkpoint() const
{
  return std::find(m_checkpoints.begin(), m_checkpoints.end(), m_index) != m_checkpoints.end();
}


double no::CalendarTimeline::dt() const
{
  static const double years_per_sec = 1.0 / (365.2475 * 86400);
  if (m_index < m_times.size() - 1)
    return std::chrono::duration_cast<std::chrono::seconds>(m_times[m_index+1] - m_times[m_index]).count() * years_per_sec;
  return 0.0;
}

size_t no::CalendarTimeline::nsteps() const
{
  return m_checkpoints.back();
}

// namespace {

// py::object to_object(const no::CalendarTimeline::time_point& time)
// {
// //   py::object o = py::cast(time);
// //   std::time_t t = std::chrono::system_clock::to_time_t(time);
// //   std::string buf(64, 0);
// //   std::strftime(buf.data(), buf.size(), "%F %T", std::localtime(&t));
// //   return py::str(buf);
//   return py::cast{time);
// }

// }

py::object no::CalendarTimeline::time() const { return py::cast(m_times[m_index]); }
py::object no::CalendarTimeline::start() const { return py::cast(m_times.front()); }
py::object no::CalendarTimeline::end() const { return py::cast(m_times.back()); }

// // Sun=0, Mon=1 etc
// int no::CalendarTimeline::dow() const
// {
//   std::time_t t = std::chrono::system_clock::to_time_t(m_times[m_index]);
//   tm* local_tm = std::localtime(&t);
//   return local_tm->tm_wday;
// }

std::string no::CalendarTimeline::repr() const
{
  return "<neworder.CalendarTimeline start=%% end=%% checkpoints=%% time=%% index=%%>"s
          % start() % end() % m_checkpoints % time() % m_index;
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
