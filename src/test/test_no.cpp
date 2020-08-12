
// test neworder embedded module

#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <vector>
#include <string>


void test_no()
{
  // test logging - use (,) operator combo to make it look like one arg returning bool. If a problem, there will be an exception or worse
  CHECK((no::log("neworder module test"), true));
  // cppcheck-suppress zerodiv 
  CHECK((no::log("test logging types: %% %% %% %% %% %% %%"_s % false % 0 % 0.0 % "" % ""_s % std::vector<int>(10) % (void*)nullptr), true));

  // test formatting
  CHECK(format::decimal(3.14, 6, 6) == "     3.140000");
  // ignores the 1 LHS padding as there are 6 digits
  CHECK(format::decimal(1000000.0 / 7, 1, 6) == "142857.142857");
  CHECK(format::pad(3, 4) == "   3");
  CHECK(format::pad(3, 5, '0') == "00003");
  // ignores 3 as number requires 4 chars
  CHECK(format::pad(5000, 3, '0') == "5000");
  CHECK(format::hex<int32_t>(24233) == "0x00005ea9");
  CHECK(format::hex<size_t>(133, false) == "0000000000000085");
  CHECK(format::boolean(false) == "false");
  
  // /*no::Environment& env =*/ no::getenv();
  py::object module = py::module::import("neworder");
  no::Runtime runtime("neworder");

  // Check required (but defaulted) attrs visible from both C++ and python
  const char* attrs[] = {"rank", "size"}; 

  CHECK(pycpp::has_attr(module, "mpi"));
  for (size_t i = 0; i < sizeof(attrs)/sizeof(attrs[0]); ++i)
  {
    CHECK(pycpp::has_attr(module.attr("mpi"), attrs[i]));
    // TODO why doesnt this work
    CHECK(runtime({"'%%' in dir(neworder.mpi)"_s % attrs[i], no::CommandType::Eval}).cast<bool>());
    //CHECK(runtime({"hasattr(neworder, '%%')"_s % attrs[i], no::CommandType::Eval}).cast<bool>());
  }

  // check string conversion
  CHECK(pycpp::as_string(module.attr("time").attr("never")).substr(0, 42) == "<built-in method never of PyCapsule object");
  CHECK(pycpp::as_string(module.attr("time").attr("never")()) == "nan");

  // Check diagnostics consistent
  CHECK(runtime({"neworder.name() == '%%'"_s % no::module_name(), no::CommandType::Eval}).cast<bool>());
  CHECK(runtime({"neworder.version() == '%%'"_s % no::module_version(), no::CommandType::Eval}).cast<bool>());
  CHECK(runtime({"neworder.python() == '%%'"_s % no::python_version(), no::CommandType::Eval}).cast<bool>());

  double x = -1e10;
  CHECK(no::Timeline::distant_past() < x);
  CHECK(no::Timeline::far_future() > x);
  x = 1e10;
  CHECK(no::Timeline::distant_past() < x);
  CHECK(no::Timeline::far_future() > x);

  // dreams never end
  CHECK(no::Timeline::never() != no::Timeline::never());
  CHECK(no::Timeline::never() != x);
  CHECK(!(no::Timeline::never() < x));
  CHECK(!(no::Timeline::never() == x));
  CHECK(!(no::Timeline::never() >= x));
  // no nay never
  CHECK(!no::Timeline::isnever(x))
  // no nay never no more
  CHECK(no::Timeline::isnever(no::Timeline::never()))  
}