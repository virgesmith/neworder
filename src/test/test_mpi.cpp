
#include "test.h"
#include "Environment.h"
#include "MPIResource.h"

#include "python.h"

#include <vector>
#include <string>
#include <iostream>

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& args, const py::object& expected);
void test_no();
void test_np();
void test_errors();

int run(int rank, int size, int nmodules, const char* testmodules[]) 
{
  pycpp::Environment& env = pycpp::Environment::init(rank, size);
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
    for (int i = 0; i < nmodules; ++i)
    {
      py::object module = py::import(testmodules[i]);
      py::object testfunc = module.attr("test");
      neworder::log("running test %%.py"_s % testmodules[i]);
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


int main(int argc, const char *argv[])
{
  MPIResource mpi(&argc, &argv);

  return run(mpi.rank(), mpi.size(), argc-1, &argv[1]);
}
