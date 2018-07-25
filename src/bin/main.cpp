
#include "Inspect.h"
#include "Environment.h"

#include "python.h"

#include <iostream>

// TODO Logger...?

template<typename T>
std::vector<T> list_to_vector(const py::list& obj)
{
  py::ssize_t n = py::len(obj);
  std::vector<T> res;
  res.reserve(n);

  for (py::ssize_t i = 0; i < n; ++i)  
  {
    // TODO this could throw?
    res.push_back(py::extract<T>(obj[i])());
  }
  return res;
}

int main(int, const char*[])
{
  pycpp::Environment env;

  try
  {
    py::object config = py::import("config");

    // TODO direct init in python of an ivector?
    ;
    std::vector<int> timespan = list_to_vector<int>(py::list(config.attr("timespan")));
    int timestep = py::extract<int>(config.attr("timestep"))();

    py::object transitions(config.attr("transitions"));
    // TODO loop over keys

    py::object inputdata(config.attr("initial_population"));
    std::cout << inputdata << std::endl;

    py::object population = py::import("population");

    py::object ctor(population.attr("Population"));
    py::object obj = ctor(inputdata);

    py::object size(obj.attr("size"));
    py::object mean_age(obj.attr("mean_age"));
    py::object age(obj.attr("age"));
    py::object deaths(obj.attr("deaths"));

    py::object res = mean_age();
    std::cout << "[C++] " << timespan[0] << ": mean_age=" << res << std::endl;
    
    double mortality_hazard = py::extract<double>(config.attr("mortality_hazard"));

    for (double t = timespan[0] + timestep; t <= timespan[1]; t += timestep)
    {
      deaths(mortality_hazard);
      age(timestep);
      std::cout << "[C++] " << t << ": " << "size=" << size() << " mean_age=" << mean_age() << std::endl;
    }
  }
  catch (py::error_already_set&)
  {
    std::cerr << "ERROR: [python] " << pycpp::Environment::check() << std::endl;
    return 1;
  }
  catch (std::exception& e)
  {
    std::cerr << "ERROR: [C++] " << e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "ERROR: [C++] unknown exception" << std::endl;
    return 1;
  }
}