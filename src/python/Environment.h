#pragma once

#include <stdexcept>

namespace pycpp {

class Exception : public std::runtime_error
{
public:
  Exception(const std::string& s) : std::runtime_error(s.c_str()) { }
  //Exception(const Exception&) = delete;
  ~Exception() = default;
};

class Environment
{
public:
  Environment();

  ~Environment();

  // Disable any copy/assignment
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;
  Environment(const Environment&&) = delete;
  Environment& operator=(const Environment&&) = delete;

  // check for errors in the python env: if it returns, there is no error
  static void check();

private:

};

}