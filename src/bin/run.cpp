
#include "run.h"

#include "Functor.h"
#include "Environment.h"
#include "Module.h"
#include "Log.h"
#include "Timer.h"

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
  neworder::Environment& env = neworder::Environment::init(rank, size, indep);
  Timer timer;
  try
  {
    // Load (and exec) config file
    py::object config = py::import("config");
    // Update the env accordingly
    //env.configure(config);

    bool do_checks= py::extract<bool>(env().attr("do_checks"))();

    int log_level = py::extract<int>(env().attr("log_level"))();
    // TODO actually do something with log_level...
    (void)log_level;

    // timeline comes in as a (double, double, int) tuple: (begin, end, n)
    if (pycpp::has_attr(env(), "timeline"))
    {
      py::tuple t = py::extract<py::tuple>(env().attr("timeline"));
      env.timeline() = neworder::Timeline(py::extract<double>(t[0]), py::extract<double>(t[1]), py::extract<int>(t[2]));
    }
    double timestep = env.timeline().dt();
    env().attr("timestep") = timestep;

    neworder::log("starting microsimulation, timestep=%%..."_s % timestep);

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

      const std::string name = py::extract<std::string>(initialisations[i][0])();
      neworder::log("t=%%(%%) initialise: %%"_s % env.timeline().time() % env.timeline().index() % name);
      py::object module = py::import(modulename.c_str());
      py::object class_ = module.attr(class_name.c_str());
      py::object object = pycpp::Functor(class_, args)();

      // taking a const ref here to stay results in an empty string, which is bizarre love triangle
      env().attr(name.c_str()) = object;
    }

    // Apply any modifiers for this process
    if (!modifierArray.empty())
    {
      neworder::log("t=%%(%%) modifier: %%"_s % env.timeline().time() % env.timeline().index() % modifierArray[env.rank()].code());
      modifierArray[env.rank()]();
    }

    // Loop with checkpoints
    do
    {
      env.timeline().step(); 
      // TODO is there a way to do this in-place? does it really matter?
      double t = env.timeline().time();
      int timeindex = env.timeline().index();
      env().attr("time") = t;
      env().attr("timeindex") = timeindex;

      for (auto it = transitionTable.begin(); it != transitionTable.end(); ++it)
      {
        neworder::log("t=%%(%%) transition: %% "_s % t % timeindex % it->first);
        (it->second)();  
      }
      for (auto it = checkTable.begin(); it != checkTable.end(); ++it)
      {
        neworder::log("t=%%(%%) check: %% "_s % t % timeindex % it->first);
        bool ok = py::extract<bool>((it->second)())();
        if (!ok) 
        {
          throw std::runtime_error("check failed");
        }  
      }
      if (env.timeline().is_checkpoint())
      {
        for (auto it = checkpointTable.begin(); it != checkpointTable.end(); ++it)
        {
          neworder::log("t=%%(%%) checkpoint: %%"_s % t % timeindex % it->first);   
          // Note: return value is ignored (exec not eval)
          (it->second)();  
        }
      } 
    }
    while (!env.timeline().end());
    neworder::log("SUCCESS exec time=%%s"_s % timer.elapsed_s());
  }
  catch(py::error_already_set&)
  {
    std::cerr << "%% ERROR: %%"_s % env.context(neworder::Environment::PY) % env.get_error() << std::endl;
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