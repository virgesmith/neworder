#include "Error.h"
#include <pybind11/pybind11.h>

const char* no::NotImplementedError::what() const noexcept
{
  return m_msg.c_str();
} 


// map error types defined here to python exceptions
void no::exception_translator(std::exception_ptr p) 
{
  try 
  {
    if (p) std::rethrow_exception(p);
  } 
  catch (const no::NotImplementedError& e) 
  {
    PyErr_SetString(PyExc_NotImplementedError, e.what());
  }
};
