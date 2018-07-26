
#include "Callback.h"
//#include "Array.h"
#include "Rand.h"

#include "python.h"

#include <iostream>


const char* module_name()
{
  return "neworder";
}

template<typename T>
T vector_get(const std::vector<T>& v, int i)
{
  return v[i];
}

template<typename T>
void vector_set(std::vector<T>& v, int i, T val)
{
  v[i] = val; 
}

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

BOOST_PYTHON_MODULE(neworder)
{
  py::def("name", module_name);

  py::def("hazard", hazard);

  py::def("stopping", stopping);

  py::def("hazard_v", hazard_v);

  py::def("stopping_v", stopping_v);

  py::class_<std::vector<double>>("dvector", py::init<int>())
    .def("__len__", &std::vector<double>::size)
    .def("clear", &std::vector<double>::clear)
    .def("__getitem__", &vector_get<double>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<double>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("tolist", &vector_to_py_list<double>)
    .def("fromlist", &py_list_to_vector<double>)
    ;  
  py::class_<std::vector<int>>("ivector", py::init<int>())
    .def("__len__", &std::vector<int>::size)
    .def("clear", &std::vector<int>::clear)
    .def("__getitem__", &vector_get<int>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<int>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("tolist", &vector_to_py_list<int>)
    .def("fromlist", &py_list_to_vector<int>)
    ;  
  py::class_<std::vector<std::string>>("svector", py::init<int>())
    .def("__len__", &std::vector<std::string>::size)
    .def("clear", &std::vector<std::string>::clear)
    .def("__getitem__", &vector_get<std::string>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<std::string>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("tolist", &vector_to_py_list<std::string>)
    .def("fromlist", &py_list_to_vector<std::string>)
    ;  
}

void callback::register_all()
{
  // First register callback module
  PyImport_AppendInittab(module_name(), &PyInit_neworder);
}


