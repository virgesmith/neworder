#include "Environment.h"

#include <vector>
#include <string>

#include <iostream>

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& args);
void test2(const std::string& modulename, const std::string& objectname, const std::vector<std::string>& methodnames);
void test3(const std::string& modulename, const std::string& objectname, const std::string& membername, const std::string& methodname);
void test4();

int main() 
{
  try
  {
    pycpp::Environment env;

    // load module, call func with args
    test1("op", "mul", {"2", "3"});
    test1("op", "void", {"2", "3"});
    test1("pop", "func", {});

    // load module, object, call methods
    test2("pop", "population", {"size", "die", "size", "birth", "birth", "size"});

    // load module, object, modify member, call method
    test3("pop", "population", "array", "columns");
    test3("pop", "population", "array", "values");

    // boost.Python.numpy
    //test4();
  }
  catch (pycpp::Exception& e)
  {
    std::cerr << "ERROR: [python] " << e.what() << std::endl;
    return 1;
  }
  catch (std::exception& e)
  {
    std::cerr << "ERROR: [C++] " << e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "ERROR: [C++] unknown expection" << std::endl;
    return 1;
  }
  return 0;
}