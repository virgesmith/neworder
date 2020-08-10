#pragma once

#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"

namespace no {

class Environment;

class NEWORDER_EXPORT Model
{
public:
  Model(Timeline& timeline);

  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  void run(py::object& subclass, const Environment& env);

  // getters
  Timeline& timeline() { return m_timeline; }

  // functions to override
  virtual void modify(int rank); // optional, parallel runs only
  virtual void transition(); // compulsory
  virtual bool check(); // optional
  virtual void checkpoint(); // compulsory

  // const no::CommandList modifiers() const { return m_modifiers; }
  // const no::CommandDict transitions() const { return m_transitions; }
  // const no::CommandDict checks() const { return m_checks; }
  // const no::CommandDict checkpoints() const { return m_checkpoints; }

private:
  Timeline m_timeline;
  // no::CommandList m_modifiers;
  // no::CommandDict m_transitions;
  // no::CommandDict m_checks;
  // no::CommandDict m_checkpoints;

};

}
