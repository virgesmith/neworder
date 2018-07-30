
#include "Inspect.h"
#include "Environment.h"
#include "Functor.h"
#include "Callback.h"

#include "python.h"

#include <map>
#include <iostream>
#include <cstdlib>

// TODO Logger...?
namespace no = neworder;

int main(int argc, const char* argv[])
{
  // Directory containing model (config.py, etc) is specified on the command line
  // It's added to PYTHONPATH
  if (argc != 2)
  {
    std::cerr << "usage: neworder <model-path>\n"
              << "where <model-path> is a directory containing the model config (config.py) plus the model definition python files" << std::endl;
    exit(1);
  }
  std::cout << "[C++] setting PYTHONPATH=" << argv[1] << std::endl;
  setenv("PYTHONPATH", argv[1], 1);

  pycpp::Environment env;
  try
  {
    // TODO specify Path(?) on cmd line?
    py::object config = py::import("config");
    py::object self = py::import("neworder");

    bool do_checks = py::extract<bool>(config.attr("do_checks"))();

    // TODO direct init in python of an ivector?
    //std::vector<int> timespan = no::py_list_to_vector<int>(py::list(self.attr("timespan")));
    const std::vector<double>& timespan = py::extract<std::vector<double>>(self.attr("timespan"))();
//    py::object t = self.attr("time");
    double timestep = py::extract<double>(self.attr("timestep"))();

    std::cout << "[C++] " << timespan[0] << " init: ";

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
      //std::cout << pycpp::as_string(args) << std::endl;
      py::object object = pycpp::Functor(class_, args)();

      // taking a const ref here to stay results in an empty string, which is bizarre love triangle
      const std::string name = py::extract<std::string>(initialisations[i][0])();
      objects.insert(std::make_pair(name, object));
      std::cout << name << " ";
    }
    std::cout << std::endl;

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

    for (double t = timespan[0] + timestep; t <= timespan[1]; t += timestep)
    {
      std::cout << "[C++] " << t << " exec: ";
      // TODO is there a way to do this in-place? does it really matter?
      self.attr("time") = py::object(t);
      //*time = t;

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

    std::cout << "[C++] finally: ";
    // Finalisation
    for (auto it = finalisationTable.begin(); it != finalisationTable.end(); ++it)
    {
      std::cout << it->first << " ";   
      (it->second)();  
    }
    std::cout << std::endl << "SUCCESS" << std::endl;

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