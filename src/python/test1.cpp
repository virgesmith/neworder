#include "Object.h"
#include "Environment.h"
#include "Function.h"
#include "Module.h"

#include <Python.h>

#include <iostream>

// C++-ified version of the example here: https://docs.python.org/3/extending/embedding.html

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << "pythonfile funcname [args...]" << std::endl;
    return 1;
  }

  std::cout << "[C++] " << argv[1] << ":" << argv[2];
  for (int i = 3; i < argc; ++i)
    std::cout << " " << argv[i];
  std::cout << std::endl;

  try
  {
    pycpp::Environment env;

    pycpp::String filename(PyUnicode_DecodeFSDefault(argv[1]));

    pycpp::Module module(filename);

    pycpp::Function function(module.getAttr(argv[2])); 
    pycpp::Function function2(module.getAttr("die")); 

    bool has_person = module.hasAttr("Person");

    std::cout << "[C++] Person? " << has_person << std::endl;

    if (has_person) {
      // PyObject_Repr: <class 'test1.Person'>
      //pycpp::String person(PyObject_Repr(PyObject_Dir(module.getAttr("Person"))));
      pycpp::List person(PyObject_Dir(module.getAttr("Person")));
      for (int i = 0; i < person.size(); ++i)
      {
        std::string attr((const char*)pycpp::String(person[i]));
        if (attr[0] != '_')
          std::cout << "[C++] Person::" << attr << std::endl;      
      }
    }

    pycpp::Tuple args(argc-3);
    for (int i = 0; i < argc - 3; ++i)
    {
      args.set(i, pycpp::Int(std::stoi(argv[i + 3])));
    }
    pycpp::Int result(function.call(args));
    std::cout << "[C++] Result: " << (int)result << std::endl;

    pycpp::Tuple noargs(0);
    function2.call(noargs);

    pycpp::Int result2(function.call(args));
    std::cout << "[C++] Result: " << (int)result2 << std::endl;

  }
  catch(std::exception& e)
  {
    std::cerr << "ERROR:" << e.what() << std::endl;
    return 1;
  }
  return 0;
}