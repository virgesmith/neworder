#include "Object.h"
#include "Function.h"
#include "Module.h"
#include "Inspect.h"

#include <Python.h>

#include <vector>
#include <string>
#include <iostream>


void test3(const std::string& modulename, const std::string& objectname, const std::string& membername, const std::string& methodname)
{
  std::cout << "[C++] " << modulename << ":" << objectname << std::endl;
  // for (int i = 3; i < argc; ++i)
  //   std::cout << " " << argv[i];
  // std::cout << std::endl;

  pycpp::String filename(PyUnicode_DecodeFSDefault(modulename.c_str()));

  pycpp::Module module = pycpp::Module::init(filename);

  PyObject* o = module.getAttr(objectname);
  std::cout << "[C++] " << pycpp::type(o) << std::endl;

  pycpp::Array<int64_t> array(PyObject_GetAttrString(o, membername.c_str()));
  std::cout << "[C++] got " << array.type() << " " << array.dim() << " " << array.shape()[0] << ": ";

  for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
    std::cout << *p << " ";
  std::cout << ", adding 1..." << std::endl;

  // modify the data
  for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
    ++(*p);

  PyObject* f = PyObject_GetAttrString(o, methodname.c_str());
  pycpp::Function method(f);
  pycpp::Tuple noargs(0);

  PyObject* r = method.call(noargs);
  std::cout << "[C++] " << methodname << " return type is " << pycpp::type(r) << std::endl;

  std::cout << "[C++] " << array.type() << " " << array.dim() << " " << array.shape()[0] << ": ";
  for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
    std::cout << *p << " ";
  std::cout << std::endl;
}