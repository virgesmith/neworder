#include "Model.h"
#include "Helpers.h"
#include "Log.h"
#include "Module.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Timer.h"

#include <pybind11/pybind11.h>

namespace {

// A python subclass of Timeline that doesn't implement one of these properties inherits the base class property,
// whose trampoline looks the override up on the instance, finds the base property again and recurses (which can
// overflow the stack). So check the class dicts, which doesn't access the properties.
void check_timeline_implemented(no::Timeline& timeline) {
  // the supplied C++ timelines implement everything
  if (!dynamic_cast<no::PyTimeline*>(&timeline))
    return;

  py::object self = py::cast(&timeline);
  py::handle base = py::type::of<no::Timeline>();
  std::string missing;
  for (const char* name : {"time", "start", "end", "dt", "at_end", "_next"}) {
    for (py::handle cls : self.get_type().attr("__mro__")) {
      if (cls.is(base)) {
        missing += (missing.empty() ? ""s : ", "s) + name;
        break;
      }
      if (cls.attr("__dict__").contains(name))
        break;
    }
  }
  if (!missing.empty()) {
    const std::string& timeline_name = self.get_type().attr("__name__").cast<std::string>();
    py::set_error(PyExc_NotImplementedError, ("%% does not implement Timeline member(s): %%"s % timeline_name % missing).c_str());
    throw py::error_already_set();
  }
}

} // namespace

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

bool no::Model::run(Model& model) {
  Timer timer;
  int rank = no::env::rank.load(std::memory_order_relaxed);
  if (rank < 0) {
    throw std::runtime_error("environment is not correctly initialised, model will not be run");
  }

  check_timeline_implemented(model.timeline());

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
  no::log("t=%%(%%) %%.modify()"s % pytimeline.get("time") % pytimeline.get("index") % model_name);
  model.modify();

  // Loop over timeline
  bool ok = true;
  while (model.m_runState == Model::RUNNING) {
    py::object t = pytimeline.get("time");
    int64_t timeindex = pytimeline.get_as<int64_t>("index");

    // call the step method, then increment the timeline
    no::log("t=%%(%%) %%.step()"s % t % timeindex % model_name);
    model.step();

    model.timeline().next();

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

    // check python hasn't signalled early termination
    if (model.m_runState == no::Model::HALTED) {
      no::log("t=%%(%%) received halt signal"s % t % timeindex);
    }

    // normal completion if not explicitly halted
    if (model.m_runState == no::Model::RUNNING && pytimeline.get_as<bool>("at_end")) {
      model.m_runState = Model::COMPLETED;
      no::log("t=%%(%%) %%.finalise()"s % pytimeline.get("time") % pytimeline.get("index") % model_name);
      model.finalise();
    }
  }
  no::log("%% exec time=%%s"s % (ok ? "SUCCESS" : "ERRORED") % timer.elapsed_s());
  return ok;
}
