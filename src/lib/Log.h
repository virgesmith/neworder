
#pragma once

#include "NewOrder.h"

#include <string>
#include <type_traits>
#include <sstream>
#include <iomanip>


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

// formatters
namespace format {

  // format floating point to __x.yz
  template<typename T>
  std::string decimal(T x, int pad, int places)
  {
    static_assert(std::is_floating_point<T>::value, "decimal formatting requires a floating-point type");
    std::ostringstream str;
    str.precision(places);
    str << std::fixed << std::setw(pad + places + 1) << x;
    return str.str();
  }

  // pad integral types
  template<typename T>
  std::string pad(T x, int width, char padchar=' ')
  {
    static_assert(std::is_integral<T>::value, "padding requires an integral type");
    std::ostringstream str;
    str << std::setfill(padchar) << std::setw(width) << x;
    return str.str();
  }

  // integral types in hex (zero padded, width implied from size of T, prefix is '0x' if specified)
  template<typename T>
  std::string hex(T x, bool prefix=true)
  {
    static_assert(std::is_integral<T>::value, "hex formatting requires an integral type");
    constexpr int width = (sizeof(T) << 1); // e.g. 32bits=4bytes -> 8 chars

    std::ostringstream str;
    str << (prefix ? "0x": "") << std::setfill('0') << std::setw(width) << std::hex << x;
    return str.str();
  }

  // boolean type as string "true" or "false"
  inline std::string boolean(bool x)
  {
    return x ? "true" : "false";
  }

}

namespace no {

// msg is forcibly coerced to a string
NEWORDER_EXPORT void log(const py::handle& msg, bool override_verbose=false);
void log(const std::string& msg, bool override_verbose=false);

}
