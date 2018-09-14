
#pragma once

#include "python.h"

#include <string>
#include <iostream>


// C++14 implements the ""s -> std::string, use this for C++11
// avoiding a using namespace decl in header results in a spurious warning (gcc bug) that we silence
// warning: literal operator suffixes not preceded by ‘_’ are reserved for future standardization [-Wliteral-suffix]
// see e.g. https://stackoverflow.com/questions/41444490/should-a-using-command-issue-a-warning-when-using-a-reserved-identifier
// and https://stackoverflow.com/questions/41444135/how-to-make-stdoperators-visible-in-a-namespace?noredirect=1&lq=1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wliteral-suffix"
using std::literals::string_literals::operator""s;
#pragma GCC diagnostic pop
 

[[deprecated("use C++14  \"...\"s standard literal prefix")]]
std::string operator ""_s(const char* p, size_t s);

template<typename T>
std::string to_string_impl(T v)
{
  return std::to_string(v);
}

template<>
std::string to_string_impl(const char* v);

std::string to_string_impl(const std::string& v);

template<typename T>
std::string to_string_impl(const std::vector<T>& v)
{
  if (v.empty())
    return "[]";
  std::string result = "[" + to_string_impl(v[0]);  

  for (size_t i = 1; i < v.size(); ++i)
    result += ", " + to_string_impl(v[i]);
  result += "]";

  return result;
}


// need an rvalue ref as might/will be a temporary
template<typename T> 
std::string operator%(std::string&& str, T value)
{
  size_t s = str.find("%%");
  if (s != std::string::npos)
  {
    str.replace(s, 2, to_string_impl(value)); 
  }
  return std::move(str);
}

namespace neworder {

// msg is forcibly coerced to a string
void log(const py::object& msg);
void log(const std::string& msg);

}
