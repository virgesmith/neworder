
#pragma once

#include "python.h"

#include <string>
#include <iostream>


// C++14 implements the ""s literal -> std::string but there are so many issues with it (namespace, gcc warnings)
// just stick with the home-made version 

std::string operator ""_s(const char* p, size_t s);

template<typename T>
std::string to_string_impl(T v)
{
  return std::to_string(v);
}

// print pointer
template<typename T>
std::string to_string_impl(T* p)
{
  char buf[20];
  std::sprintf(buf, "0x%016zx", reinterpret_cast<size_t>(p));
  return std::string(buf);
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
