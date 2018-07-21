#include "Object.h"
#include "Environment.h"
#include "Function.h"
#include "Module.h"
#include "Inspect.h"

#include <Python.h>

#include <vector>
#include <string>
#include <iostream>


void test2(const std::string& modulename, const std::string& objectname, const std::vector<std::string>& methodnames)
{
  std::cout << "[C++] " << modulename << ":" << objectname << std::endl;
  // for (int i = 3; i < argc; ++i)
  //   std::cout << " " << argv[i];
  // std::cout << std::endl;

  pycpp::String filename(PyUnicode_DecodeFSDefault(modulename.c_str()));

  pycpp::Module module = pycpp::Module::init(filename);

  //pycpp::Function function(module.getAttr(argv[2]));

  PyObject* o = module.getAttr(objectname);

  for (const auto& methodname: methodnames)
  {
    pycpp::Function method(PyObject_GetAttrString(o, methodname.c_str()));

    pycpp::Tuple noargs(0);

    pycpp::Int res(method.call(noargs));
    std::cout << "[C++] " << objectname << "." << methodname << "(): " << (int)res << std::endl;
  }

  // for (const auto& attr: pycpp::dir(module.release())) 
  // {
  //   std::cout << "[C++] ::" << attr << std::endl;
  //   for (const auto& sattr: pycpp::dir(module.getAttr(attr))) 
  //   {
  //     std::cout << "[C++] " << attr << "::" << sattr << std::endl;
  //   }
  // }

  // for (const auto& attr: pycpp::dir(function.release(), false)) 
  // {
  //   std::cout << "[C++] function::" << attr << std::endl;
  // }

  bool has_person = module.hasAttr("Person");
  std::cout << "[C++] Person? " << has_person << std::endl;

  if (has_person)
  {
    for (const auto& attr: pycpp::dir(module.getAttr("Person"))) 
    {
      std::cout << "[C++] Person::" << attr.first << " [" << attr.second << "]" << std::endl;
    }
  }
}