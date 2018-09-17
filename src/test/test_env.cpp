
// test neworder embedded module

#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "python.h"
#include "numpy.h"

#include <vector>
#include <string>


void test_env()
{
// Only run in single-process mode
#ifndef NEWORDER_MPI 
  // test logging - use (,) operator combo to make it look like one arg returning bool. If a problem, there will be an exception or worse
  CHECK((neworder::log("neworder env test"), true));

  pycpp::Environment& env = pycpp::Environment::init(0, 1);
  //pycpp::Environment& env = pycpp::getenv();

  py::object module = py::import("neworder");

  CHECK(env.rank() == 0);
  CHECK(env.size() == 1);

  // Check default sequence is [0]
  CHECK(pycpp::size(env.sequence()) == 1);
  CHECK(pycpp::at<int64_t>(env.sequence(), 0) == 0);
  CHECK(env.seq() == 0);
  CHECK(!env.next()); // should be off the end of the default seq

  // update sequence & check C++ sees update
  neworder::Callback::exec("import numpy as np; import neworder; neworder.sequence = np.array([5,6,7,8])")();
  CHECK(pycpp::size(env.sequence()) == 4);
  CHECK(pycpp::at<int64_t>(env.sequence(), 0) == 5);
  CHECK(env.seq() == 0);
  CHECK(env.next()); // shouldn't be off the end of the default seq

  // Modify sequence from C++ and check python sees changes
  pycpp::at<int64_t>(env.sequence(), 0) = 55;
  CHECK(py::extract<int64_t>(neworder::Callback::eval("neworder.sequence[0]")()) == 55);

  // Replace sequence from C++ and check python sees changes
  //env.sequence() = pycpp::zero_1d_array<int64_t>(3); is a copy, not visible from python
  env.seed(pycpp::zero_1d_array<int64_t>(3));  
  CHECK(py::extract<int64_t>(neworder::Callback::eval("len(neworder.sequence)")()) == 3);
  CHECK(py::extract<int64_t>(neworder::Callback::eval("neworder.sequence[2]")()) == 0);
  pycpp::at<int64_t>(env.sequence(), 2) = 555;
  CHECK(py::extract<int64_t>(neworder::Callback::eval("neworder.sequence[2]")()) == 555);

  neworder::log("neworder env test complete");


#endif
}