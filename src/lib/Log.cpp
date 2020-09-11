
#include "Log.h"
#include "Environment.h"


// User-defined string literal and overloaded % operator allow for easy vaguely pythonic construction of log messages, e.g.
// no::log("the value of %% plus %% is %%"_s % 2 % "2" % 4.0);

// C++14 implements the ""s -> std::string, use this for C++11
std::string operator ""_s(const char* p, size_t s)
{
  return std::string(p, p + s);
}

template<>
std::string to_string_impl(const char* v)
{
  return std::string(v);
}

std::string to_string_impl(const std::string& v)
{
  return v;
}

// not visible to python
void no::log(const std::string& msg, bool override_verbose)
{
  if (override_verbose || no::getenv().m_verbose)
    py::print(no::getenv().context(), msg);
}

// not visible to python
void no::log(const py::handle& msg, bool override_verbose)
{
  if (override_verbose || no::getenv().m_verbose)
    py::print(no::getenv().context(), msg);
}