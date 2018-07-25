#pragma once

#include <string>

namespace pycpp {

struct Environment
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
  static std::string check() noexcept;

};

}