#pragma once

#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"

namespace no {

class Environment;

class NEWORDER_EXPORT Model
{
public:
  Model(Timeline& timeline, 
        const py::list& modifiers, 
        const py::dict& transitions,
        const py::dict& checks,
        const py::dict& checkpoints);

  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  void run(const Environment& env);

  // getters
  Timeline& timeline() { return m_timeline; }
  const no::CommandList modifiers() const { return m_modifiers; }
  const no::CommandDict transitions() const { return m_transitions; }
  const no::CommandDict checks() const { return m_checks; }
  const no::CommandDict checkpoints() const { return m_checkpoints; }

private:
  Timeline m_timeline;
  no::CommandList m_modifiers;
  no::CommandDict m_transitions;
  no::CommandDict m_checks;
  no::CommandDict m_checkpoints;

};

}
