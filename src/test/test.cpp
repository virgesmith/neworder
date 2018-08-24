#include "Environment.h"

#include "python.h"

#include <vector>
#include <string>
#include <iostream>

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& args);
void test2(const std::string& modulename, const std::string& objectname, const std::vector<std::string>& methodnames);
void test3(const std::string& modulename, const std::string& objectname, const std::string& membername, const std::string& methodname);
void test4();
void test_errors();

int main(int argc, const char* argv[]) 
{
  pycpp::Environment& env = pycpp::Environment::init(0, 1);
  try
  {
    // load module, call func with args
    test1("op", "mul", {"2", "3"});
    test1("op", "void", {"2", "3"});
    test1("pop", "func", {});

    // load module, object, call methods
    test2("pop", "population", {"size", "print", "size"});

    // load module, object, modify member, call method
    test3("pop", "population", "array", "columns");
    test3("pop", "population", "array", "values");

    // boost.Python.numpy
    //test4();

    // doesnt extract the python error type/msg 
    test_errors();

    // TODO how to determine tests pass/fail?
    std::cout << env.context() << " running python modules:" << std::endl;
    for (int i = 1; i < argc; ++i)
    {
      py::object module = py::import(argv[i]);
      py::object testfunc = module.attr("test");
      bool success = py::extract<bool>(testfunc())();
      std::cout << "[C++] " << argv[i] << ":" << (success ? "PASS" : "FAIL") << std::endl;
    }
  }
  catch (py::error_already_set&)
  {
    std::cerr << env.context(pycpp::Environment::PY) << "ERROR:" << env.get_error() << std::endl;
    return 1;
  }
  catch (std::exception& e)
  {
    std::cerr << "ERROR:" << e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "ERROR: unknown exception" << std::endl;
    return 1;
  }
  return 0;
}