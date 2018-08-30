// Deprecated

// test4 - boost.numpy
#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "python.h"
#include "numpy.h"

#include <vector>
#include <string>
#include <memory>
#include <iostream>


void test_np()
{
  std::cout << "[C++] boost.Python.numpy test" << std::endl;

  pycpp::Environment& env = pycpp::Environment::get();

  py::object module = py::import("neworder");

  // create an array and expose to python...
  py::tuple shape = py::make_tuple(3, 3);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray a = np::zeros(shape, dtype);
  module.attr("a") = a;

  // TODO proper test stuff

  neworder::Callback::exec("import neworder;neworder.log(neworder.a);a[1,1]=3.14")();  

  // check its been modified
  neworder::log(py::str(a));

  // modify it again
  // yuck
  double* p = reinterpret_cast<double*>(a.get_data());

  int dim = a.get_nd();
  // assumes dim >=1 
  int s = a.shape(0);
  for (int i = 1; i < dim; ++i)
    s *= a.shape(i);
  for (int i = 0; i < s; ++i)
    p[i] = (double)i / 10;

  neworder::Callback::exec("import neworder;neworder.log(neworder.a)")();  



  // load a DF and try to extract/modify...

}