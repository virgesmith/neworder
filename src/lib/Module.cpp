
#include "Module.h"

#include "Version.h"
//#include "Array.h"
#include "Environment.h"
#include "Inspect.h"
#include "MonteCarlo.h"

#include "python.h"

#include <iostream>


neworder::Callback neworder::Callback::eval(const std::string& code)
{
  return Callback(code, false);
}

neworder::Callback neworder::Callback::exec(const std::string& code)
{
  return Callback(code, true);
}

neworder::Callback::Callback(const std::string& code, bool exec/*, const std::string& locals*/) : m_exec(exec), m_code(code)
{
  // TODO (assuming they ref current env)
  m_globals = py::import("__main__").attr("__dict__");
  m_locals = py::import("neworder").attr("__dict__");
}

py::object neworder::Callback::operator()() const 
{
  // see https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/reference/embedding.html#embedding.boost_python_exec_hpp
  // evaluate the global/local namespaces at the last minute? or do they update dynamically?
  if (m_exec)
  {
    return py::exec(m_code.c_str(), m_globals, m_locals);
  }
  else
  {
    return py::eval(m_code.c_str(), m_globals, m_locals);
  }
}

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

BOOST_PYTHON_MODULE(neworder)
{
  namespace no = neworder;

  // utility/diagnostics
  py::def("name", no::module_name);
  py::def("version", no::module_version);
  py::def("python", no::python_version);
  py::def("log", no::log);
  py::def("shell", no::shell);

  // MC
  py::def("ustream", no::ustream);
  py::def("hazard", no::hazard);
  py::def("stopping", no::stopping);
  py::def("stopping_nhpp", no::stopping_nhpp);
  py::def("hazard_v", no::hazard_v);
  py::def("stopping_v", no::stopping_v);

  py::def("lazy_exec", no::Callback::exec);
  py::def("lazy_eval", no::Callback::eval);
  // TODO env?

  // Containers
  py::class_<std::vector<double>>("DVector", py::init<int>())
    .def("__len__", &std::vector<double>::size)
    .def("clear", &std::vector<double>::clear)
    .def("__getitem__", &vector_get<double>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<double>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("__str__", &no::vector_to_string<double>)
    .def("tolist", &no::vector_to_py_list<double>)
    .def("fromlist", &no::py_list_to_vector<double>)
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
    .def("__str__", &no::vector_to_string<int>)
    .def("tolist", &no::vector_to_py_list<int>)
    .def("fromlist", &no::py_list_to_vector<int>)
    ;  
  py::class_<std::vector<std::string>>("SVector", py::init<int>())
    .def("__len__", &std::vector<std::string>::size)
    .def("clear", &std::vector<std::string>::clear)
    .def("__getitem__", &vector_get<std::string>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
    .def("__setitem__", &vector_set<std::string>, py::with_custodian_and_ward<1,2>()) // to let container keep value
    .def("__str__", &no::vector_to_string<std::string>)
    .def("tolist", &no::vector_to_py_list<std::string>)
    .def("fromlist", &no::py_list_to_vector<std::string>)
    ;

  // Deferred eval/exec of Python code
  py::class_<no::Callback>("Callback", py::no_init)
    .def("__call__", &no::Callback::operator())
    .def("__str__", &no::Callback::code)
    ;
}

const char* neworder::module_name()
{
  return "neworder";
}

const char* neworder::module_version()
{
  return NEWORDER_VERSION_STRING;
}

std::string neworder::python_version()
{
  return pycpp::Environment::get().version();
}

void neworder::shell(/*const py::object& local*/)
{
  py::dict kwargs;
  kwargs["banner"] = py::str("[starting neworder debug shell]");
  kwargs["exitmsg"] = py::str("[exiting neworder debug shell]");
  //py::import("neworder");
  //kwargs["local"] = py::handle<>(PyObject_Dir());
  py::object interpreter = py::import("code").attr("interact")(*py::tuple(), **kwargs);
}

// TODO fix hard-coded python context...
void neworder::log(const py::object& msg)
{
  // TODO more efficient way?
  py::object self = py::import("neworder");
  std::cout << pycpp::Environment::get().context(pycpp::Environment::PY) << pycpp::as_string(msg.ptr()) << std::endl;
}

void neworder::import_module()
{
  // First register callback module
  PyImport_AppendInittab(module_name(), &PyInit_neworder);
}


