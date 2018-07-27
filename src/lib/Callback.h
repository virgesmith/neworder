#pragma once

#include "python.h"

#include <vector>

namespace neworder {

const char* module_name();

void import_module();

// TODO perhaps better to copy to np.array?
template <class T>
py::list vector_to_py_list(const std::vector<T>& v) 
{
  py::list list;
  for (auto it = v.begin(); it != v.end(); ++it) 
  {
    list.append(*it);
  }
  return list;
}

// TODO perhaps better to copy to np.array?
template <class T>
std::vector<T> py_list_to_vector(const py::list& l) 
{
  std::vector<T> v(py::len(l));
  for (int i = 0; i < py::len(l); ++i) 
  {
    v[i] = py::extract<T>(l[i])();
  }
  return v;
}

}