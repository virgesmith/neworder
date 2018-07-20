#include "Object.h"
#include "Function.h"
#include "Module.h"
#include "Inspect.h"

#include <Python.h>

#include <iostream>

// C++-ified version of the example here: https://docs.python.org/3/extending/embedding.html

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& argstrings)
{

  std::cout << "[C++] " << modulename << ":" << functionname;
  for (const auto& arg: argstrings)
    std::cout << " " << arg;
  std::cout << std::endl;

  pycpp::String filename(PyUnicode_DecodeFSDefault(modulename.c_str()));

  pycpp::Module module = pycpp::Module::init(filename);

  pycpp::Function function(module.getAttr(functionname));

  for (const auto& attr: pycpp::dir(module.release())) 
  {
    std::cout << "[C++] ::" << attr.first << " [" << attr.second << "]" << std::endl;
    for (const auto& sattr: pycpp::dir(module.getAttr(attr.first))) 
    {
      std::cout << "[C++] " << attr.first << "::" << sattr.first << " [" << sattr.second << "]" << std::endl;
    }
  }

  pycpp::Tuple args(argstrings.size());
  for (size_t i = 0; i < argstrings.size(); ++i)
  {
    args.set(i, pycpp::Int(std::stoi(argstrings[i])));
  }
  PyObject* result = function.call(args) ;
  std::cout << "[C++] Result type: " << pycpp::type(result) << std::endl;

}