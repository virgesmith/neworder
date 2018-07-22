
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

    pycpp::Module config = pycpp::Module::init(pycpp::String("config"));

    // TODO pycpp::Double from int
    std::vector<int> timespan = pycpp::List(config.getAttr("timespan")).toVector<int>();
    int timestep = pycpp::Int(config.getAttr("timestep"));

    pycpp::Dict transitions(config.getAttr("transitions"));
    // TODO loop over keys

    pycpp::String inputdata(config.getAttr("initial_population"));
    std::cout << inputdata << std::endl;

    pycpp::Module population = pycpp::Module::init(pycpp::String("population"));

    pycpp::Function ctor(population.getAttr("Population"));
    pycpp::Tuple ctor_args(1);
    ctor_args.set(0, std::move(inputdata));

    PyObject* obj = ctor.call(ctor_args);
    // std::cout << "object:" << pycpp::type(obj) << std::endl;
    // for (const auto& kv: pycpp::dir(obj))
    // {
    //   std::cout << kv.first << "(" << kv.second << ")" << std::endl; 
    // }

    // std::cout << PyObject_IsInstance(obj, ctor.release()) << std::endl;

    pycpp::Function size(PyObject_GetAttrString(obj, "size"));
    pycpp::Function mean_age(PyObject_GetAttrString(obj, "mean_age"));
    pycpp::Function age(PyObject_GetAttrString(obj, "age"));
    pycpp::Function deaths(PyObject_GetAttrString(obj, "deaths"));

    pycpp::Double res(mean_age.call());
    std::cout << "[C++] " << timespan[0] << ": mean_age=" << res << std::endl;
    
    pycpp::Tuple age_arg(1);
    age_arg.set(0, pycpp::Int(config.getAttr("timestep")));

    pycpp::Tuple death_arg(1);
    death_arg.set(0, pycpp::Double(config.getAttr("mortality_hazard")));

    for (double t = timespan[0] + timestep; t <= timespan[1]; t += timestep)
    {
      std::cout << "[C++] " << t << ": "; 
      deaths.call(death_arg);
      age.call(age_arg);
      pycpp::Int n(size.call()); 
      std::cout << "size=" << (int)n;
      pycpp::Double res(mean_age.call());
      std::cout << " mean_age=" << res << std::endl;
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