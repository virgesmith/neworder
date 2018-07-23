
// test4 - boost.Python

#include "Object.h"
#include "Function.h"
#include "Module.h"
#include "Inspect.h"

#include <Python.h>

#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

#include <vector>
#include <string>
#include <iostream>

void test4()
{
  np::initialize();

  std::cout << "[C++] boost.Python.numpy test" << std::endl;
  //pycpp::String filename(PyUnicode_DecodeFSDefault("pop"));

  py::object module = py::import("pop");
  //pycpp::Module module = pycpp::Module::init(filename);

  py::object o = module.attr("Population");
  // PyObject* o = module.getAttr("Population");
  // std::cout << "[C++] " << pycpp::type(o) << std::endl;

  //np::ndarray array = np::from_object(o.attr("array"));

  // pycpp::Array<int64_t> array();
  // std::cout << "[C++] got " << array.type() << " " << array.dim() << " " << array.shape()[0] << ": ";

  // for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
  //   std::cout << *p << " ";
  // std::cout << ", adding 1..." << std::endl;

  // // modify the data
  // for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
  //   ++(*p);

}