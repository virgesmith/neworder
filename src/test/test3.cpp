// Deprecated

//#include "Array.h"
#include "Inspect.h"

#include "python.h"

#include <vector>
#include <string>
#include <iostream>


void test3(const std::string& modulename, const std::string& objectname, const std::string& membername, const std::string& methodname)
{
  std::cout << "[C++] " << modulename << ":" << objectname << std::endl;

  py::object module = py::import(modulename.c_str());

  py::object o = module.attr(objectname.c_str());
  std::cout << "[C++] " << pycpp::type(o) << std::endl;

  // pycpp::Array<int64_t> array(PyObject_GetAttrString(o, membername.c_str()));
  // std::cout << "[C++] got " << array.type() << " " << array.dim() << " " << array.shape()[0] << ": ";

  // for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
  //   std::cout << *p << " ";
  // std::cout << ", adding 1..." << std::endl;

  // // modify the data
  // for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
  //   ++(*p);

  py::object method(o.attr(methodname.c_str()));

  py::object r = method();
  py::list array(r);
  //pycpp::List array(r);
  std::cout << "[C++] " << methodname << " return type is " << pycpp::type(r) << ":" <<  pycpp::type(array[0])
  //          << r 
            << std::endl;
  //           << " dim " << dim;
  // npy_intp* dims = PyArray_DIMS((PyArrayObject*)r);
  // for (int i = 0; i < dim; ++i)
  //   std::cout << " " << dims[i];


  // PyObject* r = method.call(noargs);
  // std::cout << "[C++] " << methodname << " return type is " << pycpp::type(r) << ":" <<  PyArray_TYPE((PyArrayObject*)r)
  //           << " dim " << dim;
  // npy_intp* dims = PyArray_DIMS((PyArrayObject*)r);
  // for (int i = 0; i < dim; ++i)
  //   std::cout << " " << dims[i];

  //pycpp::Array<const char*> cols(r);
  //npy_intp idx[1] = {0};
  // SIGSEGV:
  ///*PyObject* p =*/ (PyObject*)PyArray_GetPtr((PyArrayObject*)r, idx);
  //std::cout << PyArray_TYPE(r) << std::endl;

  // std::cout << "[C++] " << array.type() << " " << array.dim() << " " << array.shape()[0] << ": ";
  // for (int64_t* p = array.rawData(); p < array.rawData() + array.shape()[0]; ++p)
  //   std::cout << *p << " ";
  // std::cout << std::endl;
}