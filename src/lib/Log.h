
#pragma once

#include "python.h"

#include <iostream>

// C++14 implements the ""s -> std::string, use this for C++11
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

// // need an rvalue ref as might/will be a temporary
// template<typename T> 
// std::string operator%(std::string&& str, const std::vector<T>& value)
// {
//   size_t s = str.find("%%");
//   if (s != std::string::npos)
//   {
//     str.replace(s, 2, std::to_string(value[0])); 
//   }
//   return std::move(str);
// }

// // specialise for const char*
// template<> 
// std::string operator%(std::string&& str, const char* value);

// // non-template for string (specialisation isn't matched for some reason)
// std::string operator%(std::string&& str, const std::string& value);

namespace neworder {

// msg is forcibly coerced to a string
void log(const py::object& msg);
void log(const std::string& msg);

}
