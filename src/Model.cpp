#include "Model.h"
#include "Helpers.h"
#include "Log.h"
#include "Module.h"
#include "NewOrder.h"
#include "Timer.h"

no::Model::Model(const nb::object& timeline)
    : m_runState(no::Model::NOT_STARTED), m_timeline(timeline), m_steps(nb::iter(timeline)), m_timestep(nb::none()) {}

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
  int rank = no::env::rank.load(std::memory_order_relaxed);
  if (rank < 0) {
    throw std::runtime_error("environment is not correctly initialised, model will not be run");
  }

  // get the Model class name
  nb::object pymodel = nb::find(model);
  const std::string model_name = nb::cast<std::string>(pymodel.attr("__class__").attr("__name__"));

  if (model.m_runState == Model::COMPLETED) {
    throw nb::stop_iteration("Model has already run to completion. Construct a new model to rerun it");
  }
  if (model.m_runState == Model::FAILED) {
    throw std::runtime_error("Model run failed (an exception was raised, or check() failed) and cannot be resumed. "
      "Fix the problem and re-run.");
  }

  // an exception leaves the model's state, and possibly its timeline, part way through a step, so it can't resume
  try {
    return run_steps(model, pymodel, model_name);
  } catch (...) {
    model.m_runState = Model::FAILED;
    throw;
  }
}

bool no::Model::run_steps(Model& model, const nb::object& pymodel, const std::string& model_name) {
  Timer timer;
  const bool resuming = model.m_runState != no::Model::NOT_STARTED;
  no::log("%% %% model run"s % (resuming ? "resuming" : "starting") % model_name);
  model.m_runState = no::Model::RUNNING;

  // modify applies to the whole run, so a resumed run mustn't apply it again
  if (!resuming) {
    no::log("%%.modify()"s % model_name);
    model.modify();
    if (model.m_runState == no::Model::HALTED) {
      // NB no step has been taken, so a subsequent run starts at the first step
      no::log("received halt signal in modify()");
    }
  }

  // imported here, as neworder.timeline itself imports this module
  nb::object timestep_t = nb::module_::import_("neworder.timeline").attr("Timestep");

  bool ok = true;
  while (model.m_runState == Model::RUNNING) {
    PyObject* next = PyIter_Next(model.m_steps.ptr());
    if (!next) {
      if (PyErr_Occurred()) {
        throw nb::python_error();
      }
      // a model halted on the final step only finalises when it resumes
      model.m_runState = Model::COMPLETED;
      no::log("%%.finalise()"s % model_name);
      model.finalise();
      break;
    }
    nb::object step = nb::steal(next);
    if (!nb::isinstance(step, timestep_t)) {
      throw nb::type_error(("timeline must yield neworder.Timestep, got %% (%%)"s % nb::type_name(step.type()).c_str() %
                            nb::repr(step).c_str())
                               .c_str());
    }
    model.m_timestep = std::move(step);

    // repr is only built when it's logged, as it's costly relative to a trivial step()
    if (no::env::verbose) {
      no::log("%% %%.step()"s % nb::repr(model.m_timestep).c_str() % model_name);
    }
    model.step();

    // call the check method and stop if necessary
    if (no::env::checked) {
      // via python rather than the trampoline, which would reject a numpy.bool
      ok = no::truthy(pymodel.attr("check")());
      no::log("%%.check(): %%"s % model_name % (ok ? "ok" : "FAILED"));
      if (!ok) {
        // emit warning as well on failure (since the above message only appears when verbose mode is on)
        no::warn("check() FAILED in %%, model run failed at %%"s % model_name % nb::repr(model.m_timestep).c_str());
        model.m_runState = Model::FAILED;
        break;
      }
    }

    // check python hasn't signalled early termination. A subsequent run resumes at the next step
    if (model.m_runState == no::Model::HALTED && no::env::verbose) {
      no::log("%% received halt signal"s % nb::repr(model.m_timestep).c_str());
    }
  }
  no::log("%% exec time=%%s"s % Model::run_state_name(model.m_runState) % timer.elapsed_s());
  return ok;
}
