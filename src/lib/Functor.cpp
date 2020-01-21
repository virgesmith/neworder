
#include "Functor.h"

#include "Inspect.h"
#include "NewOrder.h"
#include "Log.h"

pycpp::Functor::Functor(py::object func) : m_func(func)
{ 
}

pycpp::Functor::Functor(py::object func, py::args args) : m_func(func), m_args(args)
{ 
}

pycpp::Functor::Functor(py::object func, py::kwargs kwargs) : m_func(func), m_kwargs(kwargs)
{ 
}

pycpp::Functor::Functor(py::object func, py::args args, py::kwargs kwargs) : m_func(func), m_args(args), m_kwargs(kwargs)
{ 
}

py::object pycpp::Functor::operator()() const 
{ 
  // if (m_args.size() == 1) 
  //   return m_func(m_args, **m_kwargs);
  // else 
    return m_func(*m_args, **m_kwargs);
}

