
#include "Log.h"
#include "Environment.h"


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