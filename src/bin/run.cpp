
#include "run.h"

#include "Functor.h"
#include "Environment.h"
#include "Module.h"
#include "Log.h"

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
}


int run(int rank, int size, bool indep)
{
  pycpp::Environment& env = pycpp::Environment::init(rank, size, indep);
  try
  {
    // Load (and exec) config file
    py::object config = py::import("config");
    // Update the env accordingly
    //env.configure(config);
    //neworder::shell();

    bool do_checks= py::extract<bool>(env().attr("do_checks"))();

    int log_level = py::extract<int>(env().attr("log_level"))();
    // TODO actually do something with log_level...
    (void)log_level;

    const np::ndarray& timespan = np::from_object(env().attr("timespan"));
    double timestep = py::extract<double>(env().attr("timestep"))();

    // Do not allow a zero timestep as this will result in an infinite loop
    if (timestep == 0.0)
    {
      throw std::runtime_error("Timestep cannot be zero!");
    }

    neworder::log("starting microsimulation t=%%"_s % pycpp::at<double>(timespan, 0));

    // modifiers (exec)
    no::CallbackArray modifierArray; 
    if (pycpp::has_attr(env(), "modifiers"))
    {
      py::list modifiers = py::list(env().attr("modifiers"));
      int n = py::len(modifiers);
      modifierArray.reserve(n);
      for (int i = 0; i < n; ++i)
      {
        modifierArray.push_back(no::Callback::exec(py::extract<std::string>(modifiers[i])()));
      }
    }

    // transiations (exec)
    no::CallbackTable transitionTable; 
    py::list transitions = py::dict(env().attr("transitions")).items();
    for (int i = 0; i < py::len(transitions); ++i)
    {
      transitionTable.insert(std::make_pair(py::extract<std::string>(transitions[i][0])(), 
                                            no::Callback::exec(py::extract<std::string>(transitions[i][1])())));
    }

    // checks (eval)
    no::CallbackTable checkTable; 
    if (do_checks)
    {
      py::list checks = py::dict(env().attr("checks")).items();
      for (int i = 0; i < py::len(checks); ++i)
      {
        checkTable.insert(std::make_pair(py::extract<std::string>(checks[i][0])(), 
                                        no::Callback::eval(py::extract<std::string>(checks[i][1])())));
      }
    }

    // execs
    no::CallbackTable checkpointTable; 
    py::list checkpoints = py::dict(env().attr("checkpoints")).items();
    for (int i = 0; i < py::len(checkpoints); ++i)
    {
      checkpointTable.insert(std::make_pair(py::extract<std::string>(checkpoints[i][0])(), 
                                            no::Callback::exec(py::extract<std::string>(checkpoints[i][1])())));
    }
    // Iterate over sequence(s) //do {

    // reset stuff...
    // initialisations...
    // list of module-class-constructor args -> list of objects
    py::list initialisations = py::dict(env().attr("initialisations")).items();
    for (int i = 0; i < py::len(initialisations); ++i)
    {
      py::dict spec = py::dict(initialisations[i][1]);
      std::string modulename = py::extract<std::string>(spec["module"])();
      std::string class_name = py::extract<std::string>(spec["class_"])();
      py::list args = py::list(spec["parameters"]);

      py::object module = py::import(modulename.c_str());
      py::object class_ = module.attr(class_name.c_str());
      py::object object = pycpp::Functor(class_, args)();

      // taking a const ref here to stay results in an empty string, which is bizarre love triangle
      const std::string name = py::extract<std::string>(initialisations[i][0])();
      env().attr(name.c_str()) = object;
      neworder::log("initialising %%"_s % name);
    }

    // Apply any modifiers for this process
    if (!modifierArray.empty())
    {
      neworder::log("applying modifier: %%"_s % modifierArray[env.rank()].code());
      modifierArray[env.rank()]();
    }

    // Loop with checkpoints
    double t = pycpp::at<double>(timespan, 0) + timestep;
    for (size_t i = 1; i < pycpp::size(timespan); ++i)
    {
      double checkpoint = pycpp::at<double>(timespan, i);
      for (; t <= checkpoint; t += timestep)
      {
        // TODO is there a way to do this in-place? does it really matter?
        env().attr("time") = py::object(t);

        for (auto it = transitionTable.begin(); it != transitionTable.end(); ++it)
        {
          neworder::log("t=%%: %% "_s % t % it->first);
          (it->second)();  
        }
        for (auto it = checkTable.begin(); it != checkTable.end(); ++it)
        {
          bool ok = py::extract<bool>((it->second)())();
          if (!ok) 
          {
            throw std::runtime_error("check failed");
          }  
        }
      }
      t -= timestep; // temporary fix
      for (auto it = checkpointTable.begin(); it != checkpointTable.end(); ++it)
      {
        neworder::log("checkpoint %%: %%"_s % t % it->first);   
        // Note: return value is ignored (exec not eval)
        (it->second)();  
      } 
    }
    neworder::log("SUCCESS");
    //} while (/*env.next()*/false);
  }
  catch(py::error_already_set&)
  {
    std::cerr << "%% ERROR: %%"_s % env.context(pycpp::Environment::PY) % env.get_error() << std::endl;
    return 1;
  }
  catch(std::exception& e)
  {
    std::cerr << "%% ERROR: %%"_s % env.context() % e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "%% ERROR: unknown exception"_s % env.context() << std::endl;
    return 1;
  }
  return 0;
}