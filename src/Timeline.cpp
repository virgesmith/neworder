
#include "Timeline.h"
#include "Log.h"
#include "Error.h"

//#include "NewOrder.h"

#include <pybind11/chrono.h>

#include <algorithm>


py::object no::NoTimeline::time() const { return py::float_(time::never()); }
py::object no::NoTimeline::start() const { return py::float_(time::never()); }
py::object no::NoTimeline::end() const { return py::float_(time::never()); }

int64_t no::NoTimeline::index() const { return static_cast<size_t>(m_stepped); }
int64_t no::NoTimeline::nsteps() const { return 1; }
double no::NoTimeline::dt() const { return 0.0; }

void no::NoTimeline::next() { m_stepped = true; }

bool no::NoTimeline::at_end() const { return m_stepped; }

// used by python __repr__
std::string no::NoTimeline::repr() const { return "<NoTimeline stepped=%%>"s % (m_stepped ? "True": "False"); }


no::LinearTimeline::LinearTimeline(double start, double end, size_t steps)
  : m_index(0), m_start(start), m_end(end), m_steps(steps)
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
  : m_index(0), m_start(start), m_end(no::time::far_future()), m_dt(step), m_steps(-1)
{
  // validate
  if (m_dt <= 0.0)
  {
    throw py::value_error("timeline must have a positive step size, got %%"s % m_dt);
  }
}

int64_t no::LinearTimeline::index() const
{
  return m_index;
}

double no::LinearTimeline::dt() const
{
  return m_dt;
}

int64_t no::LinearTimeline::nsteps() const
{
  return m_steps;
}

void no::LinearTimeline::next()
{
  if (m_steps == 0 || (m_steps > 0 && m_index < m_steps))
  {
    ++m_index;
  }
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
  return "<neworder.LinearTimeline start=%% end=%% dt=%% steps=%% time=%% index=%%>"s
          % m_start % m_end % m_dt % m_steps % time() % m_index;
}


no::NumericTimeline::NumericTimeline(const std::vector<double>& times)
  : m_index(0), m_times(times)
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

int64_t no::NumericTimeline::index() const
{
  return m_index;
}

int64_t no::NumericTimeline::nsteps() const
{
  return m_times.size() - 1;
}

double no::NumericTimeline::dt() const
{
  if (m_index == m_times.size() - 1)
    return 0.0;
  return m_times[m_index+1] - m_times[m_index];
}

void no::NumericTimeline::next()
{
  if (m_index < m_times.size())
  {
    ++m_index;
  }
}

bool no::NumericTimeline::at_end() const
{
  return m_index >= m_times.size() - 1;
}

std::string no::NumericTimeline::repr() const
{
  return "<neworder.NumericTimeline times=%% steps=%% time=%% index=%%>"s
          % m_times % (m_times.size() - 1) % m_times[m_index] % m_index;
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

no::CalendarTimeline::time_point no::CalendarTimeline::advance(const no::CalendarTimeline::time_point& time) const
{
  switch (m_unit)
  {
    case 'd': return addDays(time, m_step);
    case 'm': return addMonths(time, m_step, m_refDay);
    default: // m_unit has already been validated
    case 'y': return addMonths(time, m_step * 12, m_refDay); // ensures we deal with leap years correctly
  }
}


no::CalendarTimeline::CalendarTimeline(time_point start, time_point end, size_t step, char unit)
  : m_index(0), m_step(step), m_unit(tolower(unit)), m_times(1, start)
{
  if (m_times[0] >= end)
  {
    throw py::value_error("start time (%%) must be after end time (%%)"s % py::cast(start) % py::cast(end));
  }

  if (m_step < 1)
  {
    throw py::value_error("time unit step (%%) must be at least 1"s % step);
  }

  if (m_unit != 'd' && m_unit != 'm' && m_unit != 'y')
  {
    throw py::value_error("invalid time unit '%%', must be one of D,d,M,m,Y,y"s % unit);
  }

  std::time_t t = std::chrono::system_clock::to_time_t(start);
  tm* local_tm = std::localtime(&t);
  m_refDay = local_tm->tm_mday;

  time_point time = m_times[0];
  for(;;)
  {
    time = advance(time);
    if (time >= end)
      break;
    m_times.push_back(time);
  }
  m_times.push_back(end);

  // for (const auto& t: m_m_endTimestimes)
  // {
  //   std::time_t tt = std::chrono::system_clock::to_time_t(t);
  //   no::log(std::ctime(&tt));
  // }
}

// open-ended timeline
no::CalendarTimeline::CalendarTimeline(time_point start, size_t step, char unit)
  : m_index(0), m_step(step), m_unit(tolower(unit)), m_times(1, start)
{
  if (m_step < 1)
  {
    throw py::value_error("time unit step (%%) must be at least 1"s % step);
  }

  if (m_unit != 'd' && m_unit != 'm' && m_unit != 'y')
  {
    throw py::value_error("invalid time unit '%%', must be one of D,d,M,m,Y,y"s % unit);
  }

  std::time_t t = std::chrono::system_clock::to_time_t(m_times[0]);
  tm* local_tm = std::localtime(&t);
  m_refDay = local_tm->tm_mday;

  m_currentStep = {start, advance(start) };

}


int64_t no::CalendarTimeline::index() const
{
  return m_index;
}

bool no::CalendarTimeline::at_end() const
{
  return m_times.size() > 1 && m_index >= m_times.size() - 1;
}

void no::CalendarTimeline::next()
{
  if (m_times.size() < 2)
  {
    m_currentStep = { std::get<1>(m_currentStep), advance(std::get<1>(m_currentStep)) };
    ++m_index;
  }
  else
  {
    if (m_index < m_times.size())
    {
      ++m_index;
    }
  }
}


double no::CalendarTimeline::dt() const
{
  static const double years_per_sec = 1.0 / (365.2475 * 86400);
  if (m_times.size() < 2)
  {
    return std::chrono::duration_cast<std::chrono::seconds>(std::get<1>(m_currentStep) - std::get<0>(m_currentStep)).count() * years_per_sec;
  }
  else
  {
    if (m_index < m_times.size() - 1)
      return std::chrono::duration_cast<std::chrono::seconds>(m_times[m_index+1] - m_times[m_index]).count() * years_per_sec;
    return 0.0;
  }
}

int64_t no::CalendarTimeline::nsteps() const
{
  return m_times.size() > 1 ? m_times.size() - 1: -1;
}

py::object no::CalendarTimeline::time() const
{
  if (m_times.size() < 2)
  {
    return py::cast(std::get<0>(m_currentStep));
  }
  return py::cast(m_times[m_index]);
}

py::object no::CalendarTimeline::start() const
{
  return py::cast(m_times[0]);
}

py::object no::CalendarTimeline::end() const
{
  if (m_times.size() < 2)
  {
    return py::float_(no::time::never());
  }
  return py::cast(m_times.back());
}

std::string no::CalendarTimeline::repr() const
{
  if (m_times.empty())
  {
    return "<neworder.CalendarTimeline start=%% end=never step=%%%% nsteps=inf time=%% index=%%>"s
            % start() % m_step % m_unit % time() % m_index;
  }
  else
  {
    return "<neworder.CalendarTimeline start=%% end=%% step=%%%% nsteps=%% time=%% index=%%>"s
            % start() % end() % m_step % m_unit % (m_times.size() - 1) % time() % m_index;
  }

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
