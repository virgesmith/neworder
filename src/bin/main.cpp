
#include "Object.h"
#include "Module.h"
#include "Function.h"
#include "Inspect.h"

#include <iostream>

// TODO Logger...

int main(int, const char*[])
{
  pycpp::Environment env;

  try
  {
    // ?
    // PyGILState_STATE gstate;
    // gstate = PyGILState_Ensure();

    // TODO JSON? config file?
    pycpp::Module config = pycpp::Module::init(pycpp::String("config"));

    // TODO pycpp::Double from int
    std::vector<int> timespan = pycpp::List(config.getAttr("timespan")).toVector<int>();
    int timestep = pycpp::Int(config.getAttr("timestep"));

    pycpp::Dict transitions(config.getAttr("transitions"));
    // TODO loop over keys

    pycpp::String inputdata(config.getAttr("initial_population"));
    std::cout << inputdata << std::endl;

    pycpp::Module population = pycpp::Module::init(pycpp::String("population"));

    PyObject* classdef = population.getAttr("Population");
    std::cout << "class:" << pycpp::type(classdef) << std::endl;
    for (const auto& kv: pycpp::dir(classdef))
    {
      std::cout << kv.first << "(" << kv.second << ")" << std::endl; 
    }

    pycpp::Tuple ctor_args(1);
    ctor_args.set(0, std::move(inputdata));

    //pycpp::Function ctor(PyObject_GetAttrString(classdef, "__init__"));

    //std::cout << ctor_args 
    //PyObject* obj = ctor.call(ctor_args);
    //PyObject* obj = PyObject_CallFunction(classdef, "s", "bin/test/ssm_E09000001_MSOA11_ppp_2011.csv");
    Py_INCREF(classdef);
    PyObject* obj = PyObject_CallObject(classdef, ctor_args.release());
    std::cout << "object:" << pycpp::type(obj) << std::endl;
    for (const auto& kv: pycpp::dir(obj))
    {
      std::cout << kv.first << "(" << kv.second << ")" << std::endl; 
    }

    //std::cout << PyObject_IsInstance(obj, classdef) << std::endl;

    pycpp::Function mean_age(PyObject_GetAttrString(obj, "mean_age"));
    pycpp::Function age(PyObject_GetAttrString(obj, "age"));

    pycpp::Double res(mean_age.call());
    std::cout << timespan[0] << ": mean_age=" << res << std::endl;
    
    pycpp::Tuple age_arg(1);
    age_arg.set(0, pycpp::Int(config.getAttr("timestep")));

    for (double t = timespan[0] + timestep; t <= timespan[1]; t += timestep)
    {
      std::cout << t << ": "; 
      age.call(age_arg);
      pycpp::Double res(mean_age.call());
      std::cout << "mean_age=" << res << std::endl;
    }
    //PyGILState_Release(gstate);
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
  catch(...)
  {
    std::cerr << "ERROR: [C++] unknown expection" << std::endl;
    return 1;
  }
}