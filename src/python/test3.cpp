#include "Object.h"
#include "Environment.h"
#include "Function.h"
#include "Module.h"
#include "Inspect.h"

#include <Python.h>

#include <vector>
#include <string>
#include <iostream>

// C++-ified version of the example here: https://docs.python.org/3/extending/embedding.html


int test3(const std::string& modulename, const std::string& objectname, const std::vector<std::string>& membernames)
{
  std::cout << "[C++] " << modulename << ":" << objectname << std::endl;
  // for (int i = 3; i < argc; ++i)
  //   std::cout << " " << argv[i];
  // std::cout << std::endl;

  try
  {
    pycpp::Environment env;

    pycpp::String filename(PyUnicode_DecodeFSDefault(modulename.c_str()));

    pycpp::Module module = pycpp::Module::init(filename);

    PyObject* o = module.getAttr(objectname);
    std::cout << "[C++]" << pycpp::type(o) << std::endl;

    for (const auto& membername: membernames)
    {
      pycpp::List member(PyObject_GetAttrString(o, membername.c_str()));

      std::cout << pycpp::type(member[0]) << std::endl;
      // This is hopelessly inefficient
      std::cout << "[C++] " << PyLong_AsLong(member[0]) << " -> ";
      int i = PyLong_AsLong(member[0]);
      ++i;
      member.set(0, pycpp::Int(i));
      std::cout << PyLong_AsLong(member[0]) << std::endl;
    }
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
  return 0;
}