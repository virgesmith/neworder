
#include "Object.h"
#include "Module.h"
#include "Function.h"

#include <iostream>

// TODO Logger...

int main(int, const char*[])
{
  pycpp::Environment env;
  try
  {
    // TODO JSON? config file?
    pycpp::Module config = pycpp::Module::init(pycpp::String("config"));

    // TODO pycpp::Double from int
    std::vector<int> timespan = pycpp::List(config.getAttr("timespan")).toVector<int>();
    int timestep = pycpp::Int(config.getAttr("timestep"));

    pycpp::Dict transitions(config.getAttr("transitions"));

    for (double t = timespan[0]; t <= timespan[1]; t += timestep)
    {
      std::cout << t << std::endl; 
    }
    
    //pycpp::Module population("population");

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