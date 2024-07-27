#include "Model.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"
#include "Helpers.h"

#include <pybind11/pybind11.h>

no::Model::Model(no::Timeline& timeline, const py::function& seeder)
  : m_runState(no::Model::NOT_STARTED), m_timeline(timeline), m_timeline_handle(py::cast(&timeline)),
  m_monteCarlo(seeder().cast<int32_t>())
{
  no::log("model init: timeline=%% mc=%%"s % m_timeline.repr() % m_monteCarlo.repr());
}


void no::Model::modify()
{
  // verbose only
  no::log("defaulted to no-op Model::modify()");
}

void no::Model::halt()
{
  no::log("sending halt signal to Model::run()");
  m_runState = Model::HALTED;
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

  // access the timeline properties via the python object for consistency
  // (we can use the methods for C++ implementations, but not for python implementations)
  auto pytimeline = PyAccessor(model.timeline());

  // check the model hasn't already completed
  if (pytimeline.get_as<bool>("at_end")) {
    throw py::stop_iteration("Model has already run to completion. Reinstantiate the timeline if necessary");
  }

  model.m_runState = no::Model::RUNNING;

  // get the Model class name
  const std::string& model_name = py::cast(&model).attr("__class__").attr("__name__").cast<std::string>();


  no::log("starting %% model run. start time=%%"s % model_name % pytimeline.get("time"));

  // apply the modifier, if implemented in the derived class
  no::log("t=%%(%%) %%.modify(%%)"s % pytimeline.get("time") % pytimeline.get("index") % model_name % rank);
  model.modify();

  // Loop over timeline
  bool ok = true;
  while (model.m_runState == Model::RUNNING)
  {
    py::object t = pytimeline.get("time");
    int64_t timeindex = pytimeline.get_as<int64_t>("index");

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
    if (model.m_runState == no::Model::HALTED)
    {
      no::log("t=%%(%%) received halt signal"s % t % timeindex);
    }

    // normal completion if not explicitly halted
    if (model.m_runState == no::Model::RUNNING && pytimeline.get_as<bool>("at_end"))
    {
      model.m_runState = Model::COMPLETED;
      no::log("t=%%(%%) %%.finalise()"s % pytimeline.get("time") % pytimeline.get("index") % model_name );
      model.finalise();
    }
  }
  no::log("%% exec time=%%s"s % (ok ? "SUCCESS": "ERRORED") % timer.elapsed_s());
  return ok;
}

