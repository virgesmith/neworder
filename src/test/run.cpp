#include "run.h"
#include "test.h"
#include "Environment.h"
#include "MPIResource.h"

#include "NewOrder.h"

#include <vector>
#include <string>
#include <iostream>

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& args, const py::object& expected);
void test_no();
void test_mc();
void test_env();
void test_np();
void test_mpi();
void test_errors();

void test_py(int nmodules, const char* testmodules[])
{
  for (int i = 0; i < nmodules; ++i)
  {
    py::module module = py::module::import(testmodules[i]);
    py::object testfunc = module.attr("test");
    no::log("running test %%.py"_s % testmodules[i]);
    CHECK(testfunc().cast<bool>());
  }
}

int run(int rank, int size, bool indep, int nmodules, const char* testmodules[]) 
{
  no::Environment& env = no::Environment::init(rank, size, indep);
  try
  {

    // load module, call func with args
    test1("op", "mul", {"2", "3"}, py::int_(6));
    test1("op", "void", {"2", "3"}, py::none());

    // module (C++) tests
    test_mc();
    test_no();
    test_env();
    test_np(); 
    test_mpi();
    test_errors();

    // python tests
    test_py(nmodules, testmodules);

    REPORT()
  }
  catch(std::exception& e)
  {
    std::cerr << "%%ERROR %%"_s % env.context() % e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "%% ERROR: unknown exception"_s % env.context() << std::endl;
    return 1;
  }
  RETURN();
}


