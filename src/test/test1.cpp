#include "Object.h"
#include "Function.h"
#include "Module.h"
#include "Inspect.h"
#include "Callback.h"
#include "python.h"

#include <iostream>

// C++-ified version of the example here: https://docs.python.org/3/extending/embedding.html
// also now boostified?


void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& argstrings)
{

  std::cout << "[C++] " << modulename << ":" << functionname;
  for (const auto& arg: argstrings)
    std::cout << " " << arg;
  std::cout << std::endl;

  // pycpp::String filename(PyUnicode_DecodeFSDefault(modulename.c_str()));

  // pycpp::Module module = pycpp::Module::init(filename);

  // pycpp::Function function(module.getAttr(functionname));
  py::object module = py::import(modulename.c_str());
  py::object function(module.attr(functionname.c_str()));

  for (const auto& attr: pycpp::dir(module)) 
  {
    std::cout << "[C++] ::" << attr.first << " [" << attr.second << "]" << std::endl;
    for (const auto& sattr: pycpp::dir(module.attr(attr.first.c_str()))) 
    {
      std::cout << "[C++] " << attr.first << "::" << sattr.first << " [" << sattr.second << "]" << std::endl;
    }
  }

  //std::cout << "callable: " << PyCallable_Check(function.ptr()) << std::endl;

  //std::cout << "type: " << pycpp::type(function) << std::endl;

  std::vector<int> args(argstrings.size());
  for (size_t i = 0; i < argstrings.size(); ++i)
  {
    args[i] = std::stoi(argstrings[i]);
  }
  py::object result = function(args[0], args[1]);
  std::cout << "[C++] Result: " << result << std::endl;
}