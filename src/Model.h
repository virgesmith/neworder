#pragma once

#include "NewOrder.h"
#include "Timeline.h"
#include "MonteCarlo.h"
#include "Module.h"

namespace no
{

  class Environment;

  class NEWORDER_EXPORT Model
  {
  public:
    // deliberately use pre-C++11 enum - scope will be e.g. Model::RUN
    enum RunState
    {
      NOT_STARTED,
      RUNNING,
      HALTED, // immediate exit without calling finalise
      COMPLETED // reached end of timeline, finalise called
    };

    Model(no::Timeline &timeline, const py::function &seeder);

    virtual ~Model() = default;

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) = delete;
    Model &operator=(Model &&) = delete;

    static bool run(Model &model);

    // getters
    Timeline &timeline() { return m_timeline; }
    MonteCarlo &mc() { return m_monteCarlo; }

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
    Timeline &m_timeline;
    py::object m_timeline_handle; // ensures above ref isnt deleted during the lifetime of this object
    MonteCarlo m_monteCarlo;
  };

  class PyModel : private Model
  {
    using Model::Model;
    using Model::operator=;

    // trampoline methods
    void modify() override { PYBIND11_OVERRIDE(void, Model, modify); }

    void step() override { PYBIND11_OVERRIDE_PURE(void, Model, step); }

    bool check() override { PYBIND11_OVERRIDE(bool, Model, check); }

    void finalise() override { PYBIND11_OVERRIDE(void, Model, finalise); }
  };

}