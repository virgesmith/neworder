#include "Environment.h"

#include "python.h"

#include <vector>
#include <string>
#include <iostream>

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& args);
void test2(const std::string& modulename, const std::string& objectname, const std::vector<std::string>& methodnames);
void test3(const std::string& modulename, const std::string& objectname, const std::string& membername, const std::string& methodname);
void test_np();
void test_errors();

int main(int argc, const char* argv[]) 
{
  pycpp::Environment& env = pycpp::Environment::init(0, 1);
  try
  {
    // load module, call func with args
    test1("op", "mul", {"2", "3"});
    test1("op", "void", {"2", "3"});
    //test1("pop", "func", {});

    // boost.Python.numpy
    test_np();

    // doesnt extract the python error type/msg 
    test_errors();

    std::cout << env.context() << " running python modules:" << std::endl;
    for (int i = 1; i < argc; ++i)
    {
      py::object module = py::import(argv[i]);
      py::object testfunc = module.attr("test");
      bool success = py::extract<bool>(testfunc())();
      std::cout << env.context() << argv[i] << ":" << (success ? "PASS" : "FAIL") << std::endl;
    }
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
  return 0;
}