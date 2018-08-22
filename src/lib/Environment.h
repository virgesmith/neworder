#pragma once

#include "Global.h"

#include "python.h"

#include <string>

namespace pycpp {

struct Environment
{
public:
  ~Environment();

  // Disable any copy/assignment
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;
  Environment(const Environment&&) = delete;
  Environment& operator=(const Environment&&) = delete;

  // check for errors in the python env: if it returns, there is no error
  static std::string check() noexcept;

  // returns the python version
  static std::string version();

  void setid(int rank, int size)
  {
    m_procid = rank;
    m_nprocs = size;
    m_self->attr("procid") = rank;
    m_self->attr("nprocs") = size;
  }

  // returns the env as a python object 
  //operator py::object&() { return m_self; } doesnt implicitly cast
  py::object& operator()() { return *m_self; }

private:
  // Singletons only
  Environment();
  friend Environment& Global::instance<Environment>();

  int m_procid;
  int m_nprocs;
  // TODO wor out why this segfaults if the dtor is called (even on exit)
  py::object* m_self;
};

}