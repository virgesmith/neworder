
// test neworder embedded module

#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "NewOrder.h"
#include "numpy.h"

#include <vector>
#include <string>


void test_no()
{
  // test logging - use (,) operator combo to make it look like one arg returning bool. If a problem, there will be an exception or worse
  CHECK((neworder::log("neworder module test"), true));
  CHECK((neworder::log("test logging types: %% %% %% %% %% %% %%"_s % false % 0 % 0.0 % "" % ""_s % std::vector<int>(10) % (void*)nullptr), true));

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
  
  /*neworder::Environment& env =*/ neworder::getenv();
  py::object module = py::module::import("neworder");

  // Check required (but defaulted) attrs visible from both C++ and python
  const char* attrs[] = {"rank", "size"/*, "sequence", "seq"*/}; 

  for (size_t i = 0; i < sizeof(attrs)/sizeof(attrs[0]); ++i)
  {
    CHECK(pycpp::has_attr(module, attrs[i]));
    CHECK(neworder::Callback::eval("'%%' in locals()"_s % attrs[i])().cast<bool>());
  }

  // Check diagnostics consistent
  CHECK(neworder::Callback::eval("name() == '%%'"_s % neworder::module_name())().cast<bool>());
  CHECK(neworder::Callback::eval("version() == '%%'"_s % neworder::module_version())().cast<bool>());
  CHECK(neworder::Callback::eval("python() == '%%'"_s % neworder::python_version()/*.c_str()*/)().cast<bool>());

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