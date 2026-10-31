#include "Model.h"
#include "Log.h"
#include "Module.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Timer.h"

#include <pybind11/pybind11.h>

no::Model::Model(no::Timeline& timeline, const py::function& seeder)
    : m_runState(no::Model::NOT_STARTED), m_timeline(timeline), m_timeline_handle(py::cast(&timeline)),
      m_monteCarlo([seeder]() { return seeder().cast<int32_t>(); }) {
  no::log("model init: timeline=%% mc=%%"s % m_timeline.repr() % m_monteCarlo.repr());
}

void no::Model::modify() {
  // verbose only
  no::log("defaulted to no-op Model::modify()");
}

void no::Model::halt() {
  no::log("sending halt signal to Model::run()");
  m_runState = Model::HALTED;
}

bool no::Model::check() {
  // verbose only
  no::log("defaulted to no-op Model::check()");
  return true;
}

void no::Model::finalise() {
  // verbose only
  no::log("defaulted to no-op Model::finalise()");
}

namespace {

// returns false if the timeline is exhausted
bool advance(no::Timeline& timeline) {
  try {
    timeline.next();
  } catch (py::stop_iteration&) {
    // this code path is hit by C++ subclass implementations (currently none)
    return false;
  } catch (py::error_already_set& e) {
    // this code path is hit by python subclass implementations
    if (e.matches(PyExc_StopIteration)) {
      return false;
    }
    throw;
  }
  return true;
}

} // namespace

bool no::Model::run(Model& model) {
  Timer timer;
  int rank = no::env::rank.load(std::memory_order_relaxed);
  if (rank < 0) {
    throw std::runtime_error("environment is not correctly initialised, model will not be run");
  }

  Timeline& timeline = model.timeline();

  // get the Model class name
  const std::string& model_name = py::cast(&model).attr("__class__").attr("__name__").cast<std::string>();

  no::log("starting %% model run. start time=%%"s % model_name % timeline.time());

  // apply the modifier, if implemented in the derived class
  no::log("t=%%(%%) %%.modify()"s % timeline.time() % timeline.index() % model_name);
  model.modify();

  if (!advance(timeline)) {
    throw py::stop_iteration("Model has already run to completion. Reinstantiate the timeline if necessary");
  }

  model.m_runState = no::Model::RUNNING;

  // Loop over timeline
  bool ok = true;
  while (model.m_runState == Model::RUNNING) {
    double t = timeline.time();
    int64_t timeindex = timeline.index();

    no::log("t=%%(%%) %%.step()"s % t % timeindex % model_name);
    model.step();

    // call the check method and stop if necessary
    if (no::env::checked) {
      ok = model.check();
      no::log("t=%%(%%) %%.check(): %%"s % t % timeindex % model_name % (ok ? "ok" : "FAILED"));
      if (!ok) {
        // emit warning as well on failure (since the above message only appears when verbose mode is on)
        no::warn("check() FAILED in %%, halting model run at t=%%(%%)"s % model_name % t % timeindex);
        break;
      }
    }

    // check python hasn't signalled early termination. NB the timeline isn't advanced, so a subsequent run resumes
    // at the next step
    if (model.m_runState == no::Model::HALTED) {
      no::log("t=%%(%%) received halt signal"s % t % timeindex);
      break;
    }

    // normal completion if timeline is exhausted
    if (!advance(timeline)) {
      model.m_runState = Model::COMPLETED;
      no::log("t=%%(%%) %%.finalise()"s % timeline.time() % timeline.index() % model_name);
      model.finalise();
    }
  }
  no::log("%% exec time=%%s"s % (ok ? "SUCCESS" : "ERRORED") % timer.elapsed_s());
  return ok;
}
