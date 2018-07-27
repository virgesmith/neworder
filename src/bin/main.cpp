
#include "Inspect.h"
#include "Environment.h"
#include "Functor.h"

#include "python.h"

#include <map>
#include <iostream>

// TODO Logger...?

// TODO needs a string specialisation otherwise gets split by char
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

// pycpp::FunctionTable makeFuncTable(const py::object& module)
// {


// }
py::tuple get(py::object& module, const std::vector<std::string>& parameters)
{
  py::tuple tuple(parameters.size());
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    tuple[i] = py::object(module.attr(parameters[i].c_str()));
    //std::cout << module.attr(parameters[i].c_str()) << std::endl;
  }
  return tuple;
}


//py::object initialise_object(const std::string& modulename, const std::string& class_name, const std::vector<std::string>& parameters)

int main(int, const char*[])
{
  // TODO config.py as a command line arg

  pycpp::Environment env;
  try
  {
    // TODO specify Path(?) on cmd line?
    py::object config = py::import("config");

    // list of module-class-constructor args -> list of objects
    py::list initialisations = py::dict(config.attr("initialisations")).items();
    std::map<std::string, py::object> objects;
    for (int i = 0; i < py::len(initialisations); ++i)
    {
      py::dict spec = py::dict(initialisations[i][1]);
      //std::cout << pycpp::as_string(spec) << std::endl;
      std::string modulename = py::extract<std::string>(spec["module"])();
      std::string class_name = py::extract<std::string>(spec["class_"])();
      py::list args = py::list(spec["parameters"]);

      py::object module = py::import(modulename.c_str());
      py::object class_ = module.attr(class_name.c_str());
      std::cout << pycpp::as_string(args) << std::endl;
      py::object object = pycpp::Functor(class_, args)();

      objects.insert(std::make_pair(py::extract<std::string>(initialisations[i][0])(), object));
    }

    bool do_checks = py::extract<bool>(config.attr("do_checks"))();

    // TODO direct init in python of an ivector?
    std::vector<int> timespan = list_to_vector<int>(py::list(config.attr("timespan")));
    int timestep = py::extract<int>(config.attr("timestep"))();

    std::cout << "[C++] " << timespan[0] << " initialised" << std::endl;

    // See https://stackoverflow.com/questions/6116345/boostpython-possible-to-automatically-convert-from-dict-stdmap
    pycpp::FunctionTable transitionTable;
    py::list transitions = py::dict(config.attr("transitions")).items();

    for (int i = 0; i < py::len(transitions); ++i)
    {
      py::dict spec = py::dict(transitions[i][1]);
      ;
      transitionTable.insert(std::make_pair(
        py::extract<std::string>(transitions[i][0])(), 
        pycpp::Functor(objects[py::extract<std::string>(spec["object"])()].attr(spec["method"]), py::list(spec["parameters"]))
      ));
      //std::cout << py::object(transitions[i][0]) << std::endl;
    }

    // Load check functors only if checks enabled
    pycpp::FunctionTable checkTable;
    if (do_checks)
    {
      py::list checks = py::dict(config.attr("checks")).items();
      for (int i = 0; i < py::len(checks); ++i)
      {
        py::dict spec = py::dict(checks[i][1]);
        checkTable.insert(std::make_pair(
          py::extract<std::string>(checks[i][0])(), 
          pycpp::Functor(objects.begin()->second.attr(spec["method"]), py::list(spec["parameters"]))
        ));
      }
    }

    pycpp::FunctionTable finalisationTable;
    py::list finalisations = py::dict(config.attr("finalisations")).items();
    for (int i = 0; i < py::len(finalisations); ++i)
    {
      py::dict spec = py::dict(finalisations[i][1]);
      finalisationTable.insert(std::make_pair(
        py::extract<std::string>(finalisations[i][0])(), 
        pycpp::Functor(objects.begin()->second.attr(spec["method"]), py::list(spec["parameters"]))
      ));
    }
    
    //double mortality_hazard = py::extract<double>(config.attr("mortality_hazard"));

    for (double t = timespan[0] + timestep; t <= timespan[1]; t += timestep)
    {
      std::cout << "[C++] " << t << " exec transitions:";
      for (auto it = transitionTable.begin(); it != transitionTable.end(); ++it)
      {
        std::cout << it->first << " ";   
        (it->second)();  
      }
      std::cout << std::endl;
      for (auto it = checkTable.begin(); it != checkTable.end(); ++it)
      {
        bool ok = py::extract<bool>((it->second)())();
        if (!ok) 
        {
          throw std::runtime_error("check failed");
        }  
      }
    }

    std::cout << "[C++] exec finalisations: ";
    // Finalisation
    for (auto it = finalisationTable.begin(); it != finalisationTable.end(); ++it)
    {
      std::cout << it->first << " ";   
      (it->second)();  
    }
    std::cout << "DONE" << std::endl;

  }
  catch (py::error_already_set&)
  {
    std::cerr << "ERROR: [py] " << pycpp::Environment::check() << std::endl;
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