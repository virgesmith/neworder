#include "Model.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"
#include "Error.h"

#include <pybind11/pybind11.h>

no::Model::Model(std::unique_ptr<Timeline> timeline, const py::function& seeder)
  : m_timeline(std::move(timeline)), m_monteCarlo(seeder(no::env::rank.load(std::memory_order_relaxed)).cast<int32_t>())
{
  no::log("neworder %% model init: timeline=%% mc=%%"s % module_version() % m_timeline->repr() % m_monteCarlo.repr());
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


bool no::Model::run(py::object& model_subclass)
{
  // extract the base class
  no::Model& base = model_subclass.cast<no::Model&>();
  Timer timer;

  int rank = no::env::rank.load(std::memory_order_relaxed);

  if (rank < 0)
  {
    throw std::runtime_error("environment is not correctly initialised, model will not be run");
  }

  const std::string& subclass_name = py::str(model_subclass).cast<std::string>();

  no::log("starting model run. start time=%%"s % base.timeline().start());

  // apply the modifier, if implemented in the derived class
  no::log("t=%%(%%) %%.modify(%%)"s % base.timeline().time() % base.timeline().index() % subclass_name % rank);
  model_subclass.attr("modify")(rank);

  // Loop over timeline
  bool ok = true;
  while (!base.timeline().at_end())
  {
    py::object t = base.timeline().time();
    size_t timeindex = base.timeline().index();

    // call the step method, then incement the timeline
    no::log("t=%%(%%) %%.step()"s % t % timeindex % subclass_name);
    model_subclass.attr("step")();

    base.timeline().next();

    // call the check method and stop if necessary
    if (no::env::checked)
    {
      ok = model_subclass.attr("check")().cast<bool>();
      no::log("t=%%(%%) %%.check(): %%"s % t % timeindex % subclass_name % (ok? "ok": "FAILED"));
      if (!ok)
      {
        // emit warning as well on failure (since the above message only appears when verbose mode is on)
        no::warn("check() FAILED in %%, halting model run at t=%%(%%)"s % subclass_name % t % timeindex);
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
  if (base.timeline().at_end())
  {
    no::log("t=%%(%%) %%.finalise()"s % base.timeline().time() % base.timeline().index() % subclass_name );
    model_subclass.attr("finalise")();
    no::log("%% exec time=%%s"s % (ok ? "SUCCESS": "ERRORED") % timer.elapsed_s());
  }
  return ok;
}


