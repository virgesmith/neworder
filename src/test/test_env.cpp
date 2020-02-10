
// test neworder embedded module

#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "NewOrder.h"
#include "numpy.h"

#include <vector>
#include <string>


void test_env()
{
// Only run in single-process mode
#ifndef NEWORDER_MPI 
  // test logging - use (,) operator combo to make it look like one arg returning bool. If a problem, there will be an exception or worse
  CHECK((no::log("neworder env test"), true));

  no::Environment& env = no::Environment::init(0, 1);
  const py::object& neworder  = env;

  CHECK(env.rank() == 0);
  CHECK(env.size() == 1);
  CHECK(neworder.attr("rank")().cast<int>() == 0);
  CHECK(neworder.attr("size")().cast<int>() == 1);
  CHECK(neworder.attr("INDEP").cast<bool>());
  CHECK(neworder.attr("SEED").cast<int>() == 19937);

  no::log("neworder env test complete");

#endif
}