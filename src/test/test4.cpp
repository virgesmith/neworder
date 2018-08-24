// Deprecated

// test4 - boost.Python
#include "Inspect.h"

#include "python.h"

#include <vector>
#include <string>
#include <memory>
#include <iostream>


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

  py::object module = py::import("pop");

  py::object o = module.attr("population");
  std::cout << "[C++] " << o << std::endl;

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