#include <stdexcept>

#include <pybind11/pybind11.h>

namespace no {

class NotImplementedError : public std::exception
{
public:
  explicit NotImplementedError(const char* msg): m_msg(msg) { }
  explicit NotImplementedError(const std::string& msg): m_msg(msg) { }
  virtual ~NotImplementedError() throw () { }

  virtual const char* what() const throw ()
  {
    return m_msg.c_str();
  } 

private:
  std::string m_msg;
};


// map error types defined here to python exceptions
inline void exception_translator(std::exception_ptr p) 
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



}