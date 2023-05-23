#include "Model.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"
#include "Error.h"

#include <pybind11/pybind11.h>

no::Model::Model(no::Timeline& timeline, const py::function& seeder)
  : m_timeline(timeline), m_timeline_handle(py::cast(&timeline)),
  m_monteCarlo(seeder(no::env::rank.load(std::memory_order_relaxed)).cast<int32_t>())
{
  no::log("model init: timeline=%% mc=%%"s % "m_timeline.repr()" % m_monteCarlo.repr());
}


void no::Model::modify(int)
{
  // verbose only
  no::log("defaulted to no-op Model::modify()");
}

void no::Model::halt()
{
  no::log("sending halt signal to Model::run()");
  no::env::halt = true;
}

bool no::Model::check()
{
  // verbose only
  no::log("defaulted to no-op Model::check()");
  return true;
}

void no::Model::finalise()
{
  // verbose only
  no::log("defaulted to no-op Model::finalise()");
}


bool no::Model::run(Model& model)
{
  Timer timer;

  int rank = no::env::rank.load(std::memory_order_relaxed);

  if (rank < 0)
  {
    throw std::runtime_error("environment is not correctly initialised, model will not be run");
  }

  // get the Model class name
  const std::string& model_name = py::cast(&model).attr("__class__").attr("__name__").cast<std::string>();

  no::log("starting %% model run. start time=%%"s % model_name % model.timeline().start());

  // apply the modifier, if implemented in the derived class
  no::log("t=%%(%%) %%.modify(%%)"s % model.timeline().time() % model.timeline().index() % model_name % rank);
  model.modify(rank);

  // Loop over timeline
  bool ok = true;
  while (!model.timeline().at_end())
  {
    py::object t = model.timeline().time();
    size_t timeindex = model.timeline().index();

    // call the step method, then increment the timeline
    no::log("t=%%(%%) %%.step()"s % t % timeindex % model_name);
    model.step();

    model.timeline().next();

    // call the check method and stop if necessary
    if (no::env::checked)
    {
      ok = model.check();
      no::log("t=%%(%%) %%.check(): %%"s % t % timeindex % model_name % (ok? "ok": "FAILED"));
      if (!ok)
      {
        // emit warning as well on failure (since the above message only appears when verbose mode is on)
        no::warn("check() FAILED in %%, halting model run at t=%%(%%)"s % model_name % t % timeindex);
        break;
      }
    }

    // check python hasn't signalled early termination
    if (no::env::halt)
    {
      no::log("t=%%(%%) received halt signal"s % t % timeindex);
      // reset the flag so that subsequent model runs don't halt immediately
      no::env::halt = false;
      break;
    }
  }
  // call the finalise method (if not explicitly halted mid-timeline)
  if (model.timeline().at_end())
  {
    no::log("t=%%(%%) %%.finalise()"s % model.timeline().time() % model.timeline().index() % model_name );
    model.finalise();
    no::log("%% exec time=%%s"s % (ok ? "SUCCESS": "ERRORED") % timer.elapsed_s());
  }
  return ok;
}

