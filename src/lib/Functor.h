#pragma once

// Functor.h

#include "Inspect.h"
#include "python.h"

#include <map>
#include <iostream>

namespace pycpp
{

class Functor
{
public:
  Functor(py::object func, py::list args) : m_func(func)
  { 
    std::cout << pycpp::as_string(args) << std::endl;

    // avoids oddd error passing list elements directly as args 
    m_args.reserve(py::len(args));
    for (int i = 0; i < py::len(args); ++i) 
    {
      m_args.push_back(args[i]);
    }
  }

  py::object operator()() const 
  { 
    // annoyingly there doesnt seem to be a way of just sending the tuple directly
    switch(m_args.size()) 
    {
    case 0: return m_func(); 
    case 1: return m_func(m_args[0]); // weird error with m_args[0]
    case 2: return m_func(m_args[0], m_args[1]);
      // { py::object arg0 = m_args[0];
      // py::object arg1 = m_args[1];
      // return m_func(arg0, arg1); }
    case 3: return m_func(m_args[0], m_args[1], m_args[2]); 
    default:
      throw std::runtime_error("TODO enable >3 args");
    }
  }

private:
  py::object m_func;
  /*py::list*/std::vector<py::object> m_args;
};

typedef std::map<std::string, Functor> FunctionTable;

}