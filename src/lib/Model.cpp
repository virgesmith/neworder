#include "Model.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"
#include "Environment.h"


no::Model::Model(Timeline& timeline, 
  const py::list& modifiers, 
  const py::dict& transitions,
  const py::dict& checks,
  const py::dict& checkpoints) 
  : m_timeline(timeline)
  { 
    // modifiers (exec)
    int n = py::len(modifiers);

    m_modifiers.reserve(n);
    for (int i = 0; i < n; ++i)
    {        
      m_modifiers.push_back(std::make_tuple(modifiers[i].cast<std::string>(), no::CommandType::Exec));
    }

    // transitions (exec)
    for (const auto& kv: transitions)
    {
      no::log("registered transition %%: %%"_s % kv.first.cast<std::string>() % kv.second.cast<std::string>());
      m_transitions.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Exec}});
    }

    // checks (eval)
    for (const auto& kv: checks)
    {
      no::log("registered check %%: %%"_s % kv.first.cast<std::string>() % kv.second.cast<std::string>());
      m_checks.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Eval}});
    }
    
    // execs
    for (const auto& kv: checkpoints)
    {
      no::log("registered checkpoint %%: %%"_s % kv.first.cast<std::string>() % kv.second.cast<std::string>());
      m_checkpoints.insert({kv.first.cast<std::string>(), {kv.second.cast<std::string>(), no::CommandType::Exec}});
    }
  }

void no::Model::run(const no::Environment& env) 
{
  no::Runtime runtime("neworder");
  Timer timer;

  no::log("starting microsimulation. start time=%%, timestep=%%, checkpoint(s) at %%"_s 
    % m_timeline.start() % m_timeline.dt() % m_timeline.checkpoints());

  // Apply any modifiers for this process
  if (!m_modifiers.empty())
  {
    if (m_modifiers.size() != (size_t)env.size()) 
    {
      throw std::runtime_error("modifier array size (%%) not consistent with number of processes (%%)"_s % m_modifiers.size() % env.size());
    }
    no::log("applying process-specific modifier: %%"_s % std::get<0>(m_modifiers[env.rank()]));
    runtime(m_modifiers[env.rank()]);
  }
  
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

