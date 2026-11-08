#pragma once

#include "Module.h"
#include "MonteCarlo.h"
#include "NewOrder.h"
#include "Timeline.h"

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

  Model(no::Timeline& timeline, const nb::callable& seeder);

  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  static bool run(Model& model);

  // getters
  Timeline& timeline() { return m_timeline; }
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
  Timeline& m_timeline;
  nb::object m_timeline_handle; // ensures above ref isnt deleted during the lifetime of this object
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