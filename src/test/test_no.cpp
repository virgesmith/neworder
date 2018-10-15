
// test neworder embedded module

#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "python.h"
#include "numpy.h"

#include <vector>
#include <string>


void test_no()
{
  // test logging - use (,) operator combo to make it look like one arg returning bool. If a problem, there will be an exception or worse
  CHECK((neworder::log("neworder module test"), true));
  CHECK((neworder::log("test logging types: %% %% %% %% %% %% %%"_s % false % 0 % 0.0 % "" % ""_s % std::vector<int>(10) % (void*)nullptr), true));

  /*neworder::Environment& env =*/ neworder::getenv();
  py::object module = py::import("neworder");

  // Check required (but defaulted) attrs visible from both C++ and python
  const char* attrs[] = {"rank", "size"/*, "sequence", "seq"*/}; 

  for (size_t i = 0; i < sizeof(attrs)/sizeof(attrs[0]); ++i)
  {
    CHECK(pycpp::has_attr(module, attrs[i]));
    CHECK(py::extract<bool>(neworder::Callback::eval("'%%' in locals()"_s % attrs[i])()));
  }

  // Check diagnostics consistent
  CHECK(py::extract<bool>(neworder::Callback::eval("name() == '%%'"_s % neworder::module_name())()));
  CHECK(py::extract<bool>(neworder::Callback::eval("version() == '%%'"_s % neworder::module_version())()));
  CHECK(py::extract<bool>(neworder::Callback::eval("python() == '%%'"_s % neworder::python_version()/*.c_str()*/)()));

  double x = -1e10;
  CHECK(neworder::Timeline::distant_past() < x);
  CHECK(neworder::Timeline::far_future() > x);
  x = 1e10;
  CHECK(neworder::Timeline::distant_past() < x);
  CHECK(neworder::Timeline::far_future() > x);

  // dreams never end
  CHECK(neworder::Timeline::never() != neworder::Timeline::never());
  CHECK(neworder::Timeline::never() != x);
  CHECK(!(neworder::Timeline::never() < x));
  CHECK(!(neworder::Timeline::never() == x));
  CHECK(!(neworder::Timeline::never() >= x));
  // no nay never
  CHECK(!neworder::Timeline::isnever(x))
  // no nay never no more
  CHECK(neworder::Timeline::isnever(neworder::Timeline::never()))  
}