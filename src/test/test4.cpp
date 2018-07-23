
// test4 - boost.Python

#include "Object.h"
#include "Function.h"
#include "Module.h"
#include "Inspect.h"

#include <Python.h>

#include <boost/python.hpp>
//#include <boost/python/numpy.hpp>

namespace py = boost::python;
//namespace np = boost::python::numpy;

#include <vector>
#include <string>
#include <memory>
#include <iostream>

std::ostream& operator<<(std::ostream& os, const py::object& o)
{
  return os << py::extract<std::string>(py::str(o))();
}

// std::ostream& operator<<(std::ostream& os, const np::ndarray& a)
// {
//   return os << py::extract<std::string>(py::str(a))();
// }

template<typename T, typename R=T>
struct Uinc
{
  typedef T argument_type;
  typedef R result_type;
  R operator()(T x) { return x + 1; }
};

void test4()
{
  //np::initialize();

  std::cout << "[C++] boost.Python.numpy test" << std::endl;
  //pycpp::String filename(PyUnicode_DecodeFSDefault("pop"));

  py::object module = py::import("pop");
  //pycpp::Module module = pycpp::Module::init(filename);

  py::object o = module.attr("population");
  std::cout << "[C++] " << o << std::endl;
  // PyObject* o = module.getAttr("Population");
  // std::cout << "[C++] " << pycpp::type(o) << std::endl;

  // See here but note compile error ‘class boost::python::api::object’ has no member named ‘def’
  // https://boostorg.github.io/python/doc/html/numpy/tutorial/index.html
  // np::ndarray array = np::from_object(o.attr("array"));
  // //py::object array = o.attr("array");
  // std::cout << "[C++] " << array;

  // std::cout << ", adding 1..." << std::endl;

  // py::object uinc = py::class_<Uinc<int, int>, boost::shared_ptr<Uinc<int, int>>>("Uinc");
  // uinc.def("__call__", np::unary_ufunc<Uinc<int, int>>::make());
  // py::object ud_inst = uinc();

  // py::object result = ud_inst.attr("__call__")(array);
  // std::cout << result << std::endl;


}