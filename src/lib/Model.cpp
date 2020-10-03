#include "Model.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"
#include "Environment.h"
#include "Error.h"

#include <pybind11/pybind11.h>

no::Model::Model(Timeline& timeline, const py::function& seeder) 
  : m_timeline(timeline), m_monteCarlo(seeder(no::getenv().rank()).cast<int64_t>())
{
  no::log("neworder %% model init: timeline=%% mc=%%"_s % module_version() % m_timeline.repr() % m_monteCarlo.repr());
}


void no::Model::modify(int)
{
  // verbose only
  no::log("defaulted to no-op Model::modify()");
}

void no::Model::step()
{
  throw no::NotImplementedError("Model.step() method must be overridden");
}

void no::Model::halt()
{
  no::log("sending halt signal to Model::run()");
  no::getenv().m_halt = true;
}

bool no::Model::check()
{
  // verbose only
  no::log("defaulted to no-op Model::check()");
  return true;
}

void no::Model::checkpoint()
{
  throw no::NotImplementedError("Model.checkpoint() method must be overridden");
}


bool no::Model::run(py::object& model_subclass) 
{
  // extract the base class
  no::Model& base = model_subclass.cast<no::Model&>();
  Timer timer;

  int rank = no::getenv().rank();

  const std::string& subclass_name = py::str(model_subclass).cast<std::string>();

  no::log("starting model run. start time=%%"_s % base.timeline().start());

  // apply the modifier, if implemented in the derived class 
  no::log("t=%%(%%) %%.modify(%%)"_s % base.timeline().time() % base.timeline().index() % subclass_name % rank);
  model_subclass.attr("modify")(rank);

  // Loop with checkpoints
  bool ok = true;
  while (!base.timeline().at_end())
  {
    base.timeline().next(); 
    double t = base.timeline().time();
    int timeindex = base.timeline().index();

    no::log("t=%%(%%) %%.step()"_s % t % timeindex % subclass_name);
    model_subclass.attr("step")();

    // call the check method and stop if necessary
    if (no::getenv().m_checked)
    {
      ok = model_subclass.attr("check")().cast<bool>();
      if (!ok)
      {
        no::log("t=%%(%%) %%.check() FAILED, halting model run"_s % t % timeindex % subclass_name, true );
        break;
      }
      no::log("t=%%(%%) %%.check() [ok]"_s % t % timeindex % subclass_name );
    }

    // check python hasn't signalled early termination
    if (no::getenv().m_halt)
    {
      no::log("t=%%(%%) received halt signal"_s % t % timeindex);
      no::getenv().m_halt = false;
      // reset the flag
      break;
    }

    // call the checkpoint method as required 
    if (base.timeline().at_checkpoint())
    {
      no::log("t=%%(%%) %%.checkpoint()"_s % t % timeindex % subclass_name );
      model_subclass.attr("checkpoint")();
    } 
  }
  no::log("%% exec time=%%s"_s % (ok ? "SUCCESS": "ERRORED") % timer.elapsed_s());
  return ok;
}


