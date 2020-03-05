
#include "run.h"

#include "Functor.h"
#include "Environment.h"
#include "Module.h"
#include "Log.h"
#include "Timer.h"

#include "NewOrder.h"

#include <iostream>
#include <cstdlib>

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

namespace {

py::tuple get_or_empty(const py::handle& dict, const char* key)
{
  if (dict.contains(key))
  {
    // no::log("type of args is: (len=%%)"_s);
    // no::log(dict[key].get_type());
    // prevent a single string arg being expanded into a tuple of chars
    if (!dict[key].get_type().is(py::tuple().get_type()))
    //if (dict[key].get_type() == py::str().get_type())
    {
      py::tuple t1(1);
      t1[0] = dict[key];
      return t1;  
    }
    return dict[key];
  }
  else
  {
    return py::tuple();
  }
}

}

int run(int rank, int size, bool indep)
{
  no::Environment& env = no::Environment::init(rank, size, indep);
  py::object& neworder = env; 
  Timer timer;
  try
  {
    // Load (and exec) config file
    py::module config = py::module::import("config");
    // Load the root namespace
    py::module root = py::module::import("__main__");

    // this works around an issue in matplotlib where it assumes sys.argv[0] exists
    py::module sys = py::module::import("sys");
    py::list argv(1);
    argv[0] = py::str("neworder");
    sys.attr("argv") = argv;
    root.attr("sys") = sys;

    bool do_checks = neworder.attr("do_checks").cast<bool>();

    int log_level = neworder.attr("log_level").cast<int>();
    // TODO actually do something with log_level...
    (void)log_level;

    // timeline comes in as a tuple: (begin, end, [checkpoint(s)...])
    if (!pycpp::has_attr(neworder, "timeline"))
    {
      throw std::runtime_error("Timeline must be defined in config.py");
    }
    // TODO more info re defaulted timeline and overridden timestep
    no::log("starting microsimulation. start time=%%, timestep=%%, checkpoint(s) at %%"_s 
      % env.timeline().start() % env.timeline().dt() % env.timeline().checkpoints());

    // modifiers (exec)
    no::CommandList modifierArray; 
    if (pycpp::has_attr(neworder, "modifiers"))
    {
      py::list modifiers = py::list(neworder.attr("modifiers"));
      int n = py::len(modifiers);
      modifierArray.reserve(n);
      for (int i = 0; i < n; ++i)
      {        
        modifierArray.push_back(std::make_tuple(modifiers[i].cast<std::string>(), no::CommandType::Exec));
      }
    }

    // transitions (exec)
    no::CommandDict transitionTable; 
    py::dict transitions = py::dict(neworder.attr("transitions")); //.items();
    for (const auto& kv: transitions)
    {
      transitionTable.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Exec}});
    }

    // checks (eval)
    no::CommandDict checkTable; 
    if (do_checks)
    {
      py::dict checks = py::dict(neworder.attr("checks"));
      for (const auto& kv: checks)
      {
        checkTable.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Eval}});
      }
    }
    
    // execs
    no::CommandDict checkpointTable; 
    py::dict checkpoints = py::dict(neworder.attr("checkpoints"));
    for (const auto& kv: checkpoints)
    {
      checkpointTable.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Exec}});
    }

    // initialisations...
    // list of module-class-constructor args -> list of objects
    py::dict initialisations = py::dict(neworder.attr("initialisations"));
    for (const auto& kv: initialisations)
    {
      const std::string& name = kv.first.cast<std::string>();

      // Note: 
      //kv.second.attr("module"); 
      // kv.second["module"];
      std::string modulename = kv.second["module"].cast<std::string>();
      std::string class_name = kv.second["class_"].cast<std::string>();
      //if (kv.second.contains("args"))
      //py::args args = py::args(kv.second["args"]); // from py::tuple
      py::args args = get_or_empty(kv.second, "args");
      // py::args args = py::tuple(1);
      // args[0] = "hello";
      py::kwargs kwargs = kv.second.contains("kwargs") ? py::kwargs(kv.second["kwargs"]) : py::kwargs(); // from py::dict

      no::log("t=%%(%%) initialise: %%"_s % env.timeline().time() % env.timeline().index() % name);
      py::module module = py::module::import(modulename.c_str());
      py::object class_ = module.attr(class_name.c_str());
      py::object object = pycpp::Functor(class_, args, kwargs)();

      // taking a const ref here to stay results in an empty string, which is bizarre love triangle
      // add object to root namespace
      root.attr(name.c_str()) = object;
    }

    // ensure all the python runs in an env with neworder and the stuff we've initialised in the root namespace
    no::Runtime runtime("neworder");

    // Apply any modifiers for this process
    if (!modifierArray.empty())
    {
      no::log("t=%%(%%) modifier: %%"_s % env.timeline().time() % env.timeline().index() % std::get<0>(modifierArray[env.rank()]));
      runtime(modifierArray[env.rank()]);
    }

    // Loop with checkpoints
    while (!env.timeline().at_end())
    {
      env.timeline().next(); 
      double t = env.timeline().time();
      int timeindex = env.timeline().index();

      for (auto it = transitionTable.begin(); it != transitionTable.end(); ++it)
      {
        no::log("t=%%(%%) transition: %% "_s % t % timeindex % it->first);
        runtime(it->second);  
      }
      for (auto it = checkTable.begin(); it != checkTable.end(); ++it)
      {
        no::log("t=%%(%%) check: %% "_s % t % timeindex % it->first);
        bool ok = runtime(it->second).cast<bool>();
        if (!ok) 
        {
          throw std::runtime_error("check failed");
        }  
      }
      if (env.timeline().at_checkpoint())
      {
        for (auto it = checkpointTable.begin(); it != checkpointTable.end(); ++it)
        {
          no::log("t=%%(%%) checkpoint: %%"_s % t % timeindex % it->first);   
          // Note: return value is ignored (exec not eval)
          runtime(it->second);  
        }
      } 
    }
    no::log("SUCCESS exec time=%%s"_s % timer.elapsed_s());
  }
  catch(std::exception& e)
  {
    std::cerr << "%%ERROR:%%"_s % env.context() % e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "%%ERROR: (unknown exception)"_s % env.context() << std::endl;
    return 1;
  }
  return 0;
}