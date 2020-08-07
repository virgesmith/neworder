#pragma once

#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"

namespace no {

class NEWORDER_EXPORT Model
{
public:
  Model(Timeline& timeline, 
  const py::dict& initialisations, 
  const py::dict& transitions,
  const py::dict& checks,
  const py::dict& checkpoints) 
  : m_timeline(timeline)
  { 
    py::module root = py::module::import("__main__");

    for (const auto& kv: initialisations)
    {
      const std::string& name = kv.first.cast<std::string>();

      // Note: 
      //kv.second.attr("module"); 
      // kv.second["module"];
      // std::string modulename = kv.second["module"].cast<std::string>();
      // std::string class_name = kv.second["class_"].cast<std::string>();
      // //if (kv.second.contains("args"))
      // //py::args args = py::args(kv.second["args"]); // from py::tuple
      // py::args args = get_or_empty(kv.second, "args");
      // // py::args args = py::tuple(1);
      // // args[0] = "hello";
      // py::kwargs kwargs = kv.second.contains("kwargs") ? py::kwargs(kv.second["kwargs"]) : py::kwargs(); // from py::dict

      // no::log("t=%%(%%) initialise: %%"_s % env.timeline().time() % env.timeline().index() % name);
      // py::module module = py::module::import(modulename.c_str());
      // py::object class_ = module.attr(class_name.c_str());
      // py::object object = pycpp::Functor(class_, args, kwargs)();

      // // taking a const ref here to stay results in an empty string, which is bizarre love triangle
      // // add object to root namespace
      root.attr(name.c_str()) = kv.second;
    }

    // transitions (exec)
    for (const auto& kv: transitions)
    {
      no::log("transition %%:%%"_s % kv.first.cast<std::string>() % kv.second.cast<std::string>());
      m_transitions.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Exec}});
    }

    // checks (eval)
    //if (do_checks)
    {
      for (const auto& kv: checks)
      {
        m_checks.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Eval}});
      }
    }
    
    // execs
    for (const auto& kv: checkpoints)
    {
      no::log("checkpoint %%:%%"_s % kv.first.cast<std::string>() % kv.second.cast<std::string>());
      m_checkpoints.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Exec}});
    }

  }

  void run() 
  {
    no::Runtime runtime("neworder");
    Timer timer;

    no::log("starting microsimulation. start time=%%, timestep=%%, checkpoint(s) at %%"_s 
      % m_timeline.start() % m_timeline.dt() % m_timeline.checkpoints());

    // Loop with checkpoints
    while (!m_timeline.at_end())
    {
      m_timeline.next(); 
      double t = m_timeline.time();
      int timeindex = m_timeline.index();

      for (auto it = m_transitions.begin(); it != m_transitions.end(); ++it)
      {
        no::log("t=%%(%%) transition: %% "_s % t % timeindex % it->first);
        runtime(it->second);  
      }
      for (auto it = m_checks.begin(); it != m_checks.end(); ++it)
      {
        no::log("t=%%(%%) check: %% "_s % t % timeindex % it->first);
        bool ok = runtime(it->second).cast<bool>();
        if (!ok) 
        {
          throw std::runtime_error("check failed: %s"_s % it->first);
        }  
      }
      if (m_timeline.at_checkpoint())
      {
        for (auto it = m_checkpoints.begin(); it != m_checkpoints.end(); ++it)
        {
          no::log("t=%%(%%) checkpoint: %%"_s % t % timeindex % it->first);   
          // Note: return value is ignored (exec not eval)
          runtime(it->second);  
        }
      } 
    }
    no::log("SUCCESS exec time=%%s"_s % timer.elapsed_s());

  }

  Timeline& timeline() { return m_timeline; }

private:
  Timeline m_timeline;
  // m_checks;
  //py::dict m_initialisations;
  no::CommandDict m_transitions;
  no::CommandDict m_checks;
  no::CommandDict m_checkpoints;

};

}
