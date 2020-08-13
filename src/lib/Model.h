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
  Model(Timeline& timeline);

  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  static void run(py::object& subclass);

  // getters
  Timeline& timeline() { return m_timeline; }
  MonteCarlo& mc() { return m_monteCarlo; }

  // functions to override
  virtual void modify(int rank); // optional, parallel runs only
  virtual void transition(); // compulsory
  virtual bool check(); // optional
  virtual void checkpoint(); // compulsory

private:
  Timeline m_timeline;
  MonteCarlo m_monteCarlo;

};

}
