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

  try
  {
    std::cout << argv[1] << ":" << argv[2];
    for (int i = 3; i < argc; ++i)
      std::cout << " " << argv[i];
    std::cout << std::endl;

    Py_Initialize();
    PyObject *filename = PyUnicode_DecodeFSDefault(argv[1]);
    /* Error checking of filename left out */

    PyObject *module = PyImport_Import(filename);
    Py_DECREF(filename);

    if(!module)
    {
      PyErr_Print();
      throw std::runtime_error(std::string("Failed to load ") + argv[1]);
    }

    PyObject* function = PyObject_GetAttrString(module, argv[2]);
    /* function is a new reference */
    // PyObject* str = PyUnicode_AsEncodedString(function, "utf-8", "~E~");
    // //const char *bytes = PyBytes_AS_STRING(str);

    // printf("Result of call: %s\n", PyBytes_AS_STRING(str));

    if (!function || !PyCallable_Check(function))
    {
      // TODO see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
      // function that sticks python error into an exception and throws
      if (PyErr_Occurred())
        PyErr_Print();
      throw std::runtime_error(std::string("Cannot find function ") + argv[2]);
    }    
    PyObject* args = PyTuple_New(argc - 3);
    for (int i = 0; i < argc - 3; ++i)
    {
      PyObject* pValue = PyLong_FromLong(std::stoi(argv[i + 3]));
      if (!pValue)
      {
        Py_DECREF(args);
        Py_DECREF(module);
        std::cerr << "Cannot convert argument to long" << std::endl;
        return 1;
      }
      /* pValue reference stolen here: */
      PyTuple_SetItem(args, i, pValue);
    }
    PyObject* result = PyObject_CallObject(function, args);
    Py_DECREF(args);
    if (result)
    {
      std::cout << "Result of call: " << PyLong_AsLong(result) << std::endl;
      Py_DECREF(result);
    }
    else
    {
      Py_DECREF(function);
      Py_DECREF(module);
      PyErr_Print();
      std::cout << "Call failed" << std::endl;
      return 1;
    }

    Py_XDECREF(function);
    Py_DECREF(module);

    if (Py_FinalizeEx() < 0)
    {
      return 120;
    }

  }
  catch(std::exception& e)
  {
    std::cerr << "ERROR:" << e.what() << std::endl;
    return 1;
  }
  return 0;
}