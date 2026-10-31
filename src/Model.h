#pragma once

#include "Module.h"
#include "MonteCarlo.h"
#include "NewOrder.h"

#include <nanobind/trampoline.h>

namespace no {

class Environment;

class NEWORDER_EXPORT Model {
public:
  // deliberately use pre-C++11 enum - scope will be e.g. Model::RUN
  enum RunState {
    NOT_STARTED,
    RUNNING,
    HALTED,   // immediate exit without calling finalise
    COMPLETED // reached end of timeline, finalise called
  };

  // timeline is any iterable of steps
  Model(const nb::object& timeline, const nb::callable& seeder);

  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  static bool run(Model& model);

  // getters
  nb::object timeline() const { return m_timeline; }
  // None before the first step
  nb::object timestep() const { return m_timestep; }
  MonteCarlo& mc() { return m_monteCarlo; }

  // functions to override
  virtual void modify();   // optional, parallel runs only
  virtual void step() = 0; // compulsory
  virtual bool check();    // optional
  virtual void finalise(); // optional

  // set the halt flag
  void halt();

  // get the run state
  RunState runState() const { return m_runState; }

private:
  RunState m_runState;
  nb::object m_timeline;
  // kept between runs, so that a halted model resumes at the next step
  nb::object m_steps;
  nb::object m_timestep;
  MonteCarlo m_monteCarlo;
};

class PyModel : private Model {
  NB_TRAMPOLINE(Model);
  using Model::operator=;

  // trampoline methods
  void modify() override { NB_OVERRIDE(modify); }

  void step() override { NB_OVERRIDE_PURE(step); }

  // NB Model::run calls check() via python, because a python override might not return a strict bool
  bool check() override { NB_OVERRIDE(check); }

  void finalise() override { NB_OVERRIDE(finalise); }
};

} // namespace no