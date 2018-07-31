
#include "Callback.h"
//#include "Array.h"
#include "Inspect.h"
#include "Rand.h"

#include "python.h"

#include <iostream>


namespace {
// pythonic array access (no bounds check)
// might be a bottleneck...
template<typename T>
T vector_get(const std::vector<T>& v, int i)
{
  return i >= 0 ? v[i] : v[v.size() + i];
}

template<typename T>
void vector_set(std::vector<T>& v, int i, T val)
{
  v[i >= 0 ? i : v.size() + i] = val; 
}

}

namespace no = neworder;

BOOST_PYTHON_MODULE(neworder)
{
  py::def("name", no::module_name);

  py::def("log", no::log);

  py::def("hazard", no::hazard);

  py::def("stopping", no::stopping);

  py::def("hazard_v", no::hazard_v);

  py::def("stopping_v", no::stopping_v);

  py::class_<std::vector<double>>("DVector", py::init<int>())
    .def("__len__", &std::vector<double>::size)
    .def("clear", &std::vector<double>::clear)
    .def("__getitem__", &vector_get<double>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<double>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("tolist", &neworder::vector_to_py_list<double>)
    .def("fromlist", &neworder::py_list_to_vector<double>)
    // operators
    .def(py::self + double())
    .def(double() + py::self)
    // .def(self + self)
    // .def(self - double)
    // .def(double - self)
    // .def(self - self)
    .def(py::self * double())
    .def(double() * py::self)
    // .def(self / double)
    ;  
  py::class_<std::vector<int>>("IVector", py::init<int>())
    .def("__len__", &std::vector<int>::size)
    .def("clear", &std::vector<int>::clear)
    .def("__getitem__", &vector_get<int>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<int>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("tolist", &neworder::vector_to_py_list<int>)
    .def("fromlist", &neworder::py_list_to_vector<int>)
    ;  
  py::class_<std::vector<std::string>>("SVector", py::init<int>())
    .def("__len__", &std::vector<std::string>::size)
    .def("clear", &std::vector<std::string>::clear)
    .def("__getitem__", &vector_get<std::string>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<std::string>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("tolist", &neworder::vector_to_py_list<std::string>)
    .def("fromlist", &neworder::py_list_to_vector<std::string>)
    ;

  py::class_<no::UStream>("UStream", py::init<int64_t>())
    .def("seed", &no::UStream::seed)
    .def("get", &no::UStream::get)
    ;  

  py::class_<no::Callback>("Callback", py::init<std::string>())
    .def("__call__", &no::Callback::operator())
    ;
}

const char* neworder::module_name()
{
  return "neworder";
}

void neworder::log(const py::object& msg)
{
  std::cout << "[py] " << pycpp::as_string(msg.ptr()) << std::endl;
}

void neworder::import_module()
{
  // First register callback module
  PyImport_AppendInittab(module_name(), &PyInit_neworder);
}



