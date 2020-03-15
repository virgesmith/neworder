
// test neworder embedded module

#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <vector>
#include <string>


void test_env()
{
// Only run in single-process mode
#ifndef NEWORDER_MPI 
  // test logging - use (,) operator combo to make it look like one arg returning bool. If a problem, there will be an exception or worse
  CHECK((no::log("neworder env test"), true));

  no::Environment& env = no::Environment::init(0, 1);
  const py::object& neworder = env;

  CHECK(env.rank() == 0);
  CHECK(env.size() == 1);
  CHECK(neworder.attr("rank")().cast<int>() == 0);
  CHECK(neworder.attr("size")().cast<int>() == 1);

  const py::object& mc = neworder.attr("mc"); 
  CHECK(env.mc().indep());
  CHECK(env.mc().seed() == 19937);
  CHECK(mc.attr("indep")().cast<bool>());
  CHECK(mc.attr("seed")().cast<int>() == 19937);

  // check MC object state is shared between C++ and python
  py::array_t<double> h01_cpp = env.mc().ustream(2);
  py::array_t<double> h23_py = mc.attr("ustream")(2);
  // values should not match (0,1) != (2,3)
  CHECK(no::at<double>(h01_cpp, 0) != no::at<double>(h23_py, 0));
  CHECK(no::at<double>(h01_cpp, 1) != no::at<double>(h23_py, 1));
  // reset from C++
  env.mc().reset();
  // sample from python
  py::array_t<double> h01_py = mc.attr("ustream")(2);
  // values should now match (0,1) == (0,1)
  CHECK(no::at<double>(h01_cpp, 0) == no::at<double>(h01_py, 0));
  CHECK(no::at<double>(h01_cpp, 1) == no::at<double>(h01_py, 1));
  // sample from C++
  py::array_t<double> h23_cpp = env.mc().ustream(2);
  // values should match  
  CHECK(no::at<double>(h23_cpp, 0) == no::at<double>(h23_py, 0));
  CHECK(no::at<double>(h23_cpp, 1) == no::at<double>(h23_py, 1));
  // reset from python
  mc.attr("reset")();
  // sample from C++
  h01_cpp = env.mc().ustream(2);
  // values should still match (0,1) == (0,1)
  CHECK(no::at<double>(h01_cpp, 0) == no::at<double>(h01_py, 0));
  CHECK(no::at<double>(h01_cpp, 1) == no::at<double>(h01_py, 1));

  no::log("neworder env test complete");

#endif
}