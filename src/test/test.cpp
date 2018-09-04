
#include "test.h"
#include "Environment.h"

#include "python.h"

#include <vector>
#include <string>
#include <iostream>

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& args, const py::object& expected);
void test_no();
void test_np();
void test_errors();


int main(int argc, const char* argv[]) 
{
  pycpp::Environment& env = pycpp::Environment::init(0, 1);
  try
  {
    // load module, call func with args
    test1("op", "mul", {"2", "3"}, py::object(6));
    test1("op", "void", {"2", "3"}, py::object());

    // module
    test_no();

    // boost.Python.numpy
    test_np();

    // doesnt extract the python error type/msg 
    test_errors();

    neworder::log("Testing python modules");
    for (int i = 1; i < argc; ++i)
    {
      py::object module = py::import(argv[i]);
      py::object testfunc = module.attr("test");
      CHECK(py::extract<bool>(testfunc())());
    }

    REPORT()
  }
  catch (py::error_already_set&)
  {
    std::cerr << env.context(pycpp::Environment::PY) << "ERROR:" << env.get_error() << std::endl;
    return 1;
  }
  catch (std::exception& e)
  {
    std::cerr << env.context() << "ERROR:" << e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << env.context() << "ERROR: unknown exception" << std::endl;
    return 1;
  }
  RETURN();
}