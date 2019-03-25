
#include "Log.h"

#include "Environment.h"
#include "Inspect.h"

#include <iostream>

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


// // specialise for const char*
// template<> 
// std::string operator%(std::string&& str, const char* value)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos)
//   {
//     str.replace(s, 2, value); 
//   }
//   return std::move(str);
// }

// // non-template for string (specialisation isn't matched for some reason)
// std::string operator%(std::string&& str, const std::string& value)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos)
//   {
//     str.replace(s, 2, value); 
//   }
//   return std::move(str);
// }

// not visible to python
void no::log(const std::string& msg)
{
  std::cout << no::getenv().context() << msg << std::endl;
}

// not visible to python
void no::log(const py::object& msg)
{
  std::cout << no::getenv().context() << pycpp::as_string(msg.ptr()) << std::endl;
}