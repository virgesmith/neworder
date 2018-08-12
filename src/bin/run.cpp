
#include "run.h"

//#include "Inspect.h"
#include "Functor.h"
#include "Environment.h"
#include "Callback.h"

#include "python.h"

#include <iostream>
#include <cstdlib>

// TODO Logger...?
namespace no = neworder;

void append_model_paths(const char* paths[], size_t n)
{
  if (!paths || !n) return;
  const char* current = getenv("PYTHONPATH");
  std::string pythonpath = std::string(paths[0]);
  for (size_t i = 1; i < n; ++i)
  {
    pythonpath += ":";
    pythonpath += paths[i]; 
  }

  if (current)
    pythonpath = pythonpath + ":" + current;
  setenv("PYTHONPATH", pythonpath.c_str(), 1);
  std::cout << "[C++] PYTHONPATH=" << pythonpath << std::endl;
}


int run(int rank, int size)
{
  std::cout << "[C++ " << rank << "/" << size << "] process init" << std::endl; 
  pycpp::Environment env;
  try
  {
    // TODO move into env?
    py::object self = py::import("neworder");

    self.attr("procid") = rank;
    self.attr("nprocs") = size;

    // TODO specify Path(?) on cmd line?
    py::object config = py::import("config");

    bool do_checks = py::extract<bool>(config.attr("do_checks"))();

    // TODO direct init in python of a DVector?
    const std::vector<double>& timespan = py::extract<std::vector<double>>(self.attr("timespan"))();
    double timestep = py::extract<double>(self.attr("timestep"))();

    // Do not allow a zero timestep as this will result in an infinite loop
    if (timestep == 0.0)
    {
      throw std::runtime_error("Timestep cannot be zero!");
    }

    std::cout << "[C++] t=" << timespan[0] << " init: ";

    // list of module-class-constructor args -> list of objects
    py::list initialisations = py::dict(config.attr("initialisations")).items();
    for (int i = 0; i < py::len(initialisations); ++i)
    {
      py::dict spec = py::dict(initialisations[i][1]);
      //std::cout << pycpp::as_string(spec) << std::endl;
      std::string modulename = py::extract<std::string>(spec["module"])();
      std::string class_name = py::extract<std::string>(spec["class_"])();
      py::list args = py::list(spec["parameters"]);

      py::object module = py::import(modulename.c_str());
      py::object class_ = module.attr(class_name.c_str());
      //std::cout << pycpp::as_string(args) << std::endl;
      py::object object = pycpp::Functor(class_, args)();

      // taking a const ref here to stay results in an empty string, which is bizarre love triangle
      const std::string name = py::extract<std::string>(initialisations[i][0])();
      self.attr(name.c_str()) = object;
      std::cout << name << " ";
    }
    std::cout << std::endl;

    no::CallbackTable transitionTable; 
    py::list transitions = py::dict(config.attr("transitions")).items();
    for (int i = 0; i < py::len(transitions); ++i)
    {
      transitionTable.insert(std::make_pair(py::extract<std::string>(transitions[i][0])(), 
                                            no::Callback(py::extract<std::string>(transitions[i][1])())));
    }

    no::CallbackTable checkTable; 
    if (do_checks)
    {
      py::list checks = py::dict(config.attr("checks")).items();
      for (int i = 0; i < py::len(checks); ++i)
      {
        checkTable.insert(std::make_pair(py::extract<std::string>(checks[i][0])(), 
                                        no::Callback(py::extract<std::string>(checks[i][1])())));
      }
    }

    no::CallbackTable checkpointTable; 
    py::list checkpoints = py::dict(config.attr("checkpoints")).items();
    for (int i = 0; i < py::len(checkpoints); ++i)
    {
      checkpointTable.insert(std::make_pair(py::extract<std::string>(checkpoints[i][0])(), 
                                            no::Callback(py::extract<std::string>(checkpoints[i][1])())));
    }

    // Loop with checkpoints
    double t = timespan[0] + timestep;
    for (size_t i = 1; i < timespan.size(); ++i)
    {
      for (; t <= timespan[i]; t += timestep)
      {
        std::cout << "[C++] t=" << t << " exec: ";
        // TODO is there a way to do this in-place? does it really matter?
        self.attr("time") = py::object(t);

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
      std::cout << "[C++] checkpoint: ";
      for (auto it = checkpointTable.begin(); it != checkpointTable.end(); ++it)
      {
        std::cout << it->first << ": ";   
        // Note: return value is ignored (exec not eval)
        (it->second)();  
      } 
      std::cout << std::endl;
    }
    std::cout << "[C++] SUCCESS" << std::endl;
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
  return 0;
}