#pragma once

#include <Python.h>

namespace pycpp {

class Environment
{
public:
  Environment() 
  {
    Py_Initialize();
  } 

  ~Environment() 
  {
    if (Py_FinalizeEx() < 0)
    {
      // report an error...
    }
  }

  // check for errors

private:

};

}