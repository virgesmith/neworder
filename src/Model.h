#pragma once

#include "NewOrder.h"
#include "Timeline.h"
#include "MonteCarlo.h"
#include "Module.h"

namespace no {

class Environment;

class NEWORDER_EXPORT Model
{
public:
  Model(std::unique_ptr<Timeline> timeline, const py::function& seeder);

  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

  static bool run(py::object& subclass);

  // getters
  Timeline& timeline() { return *m_timeline; }
  MonteCarlo& mc() { return m_monteCarlo; }

  // functions to override
  virtual void modify(int rank); // optional, parallel runs only
  virtual void step(); // compulsory
  virtual bool check(); // optional
  virtual void finalise(); // optional

  // set the halt flag
  void halt();

private:
  std::unique_ptr<Timeline> m_timeline;
  MonteCarlo m_monteCarlo;

};

}
