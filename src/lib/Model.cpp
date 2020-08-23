#include "Model.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"
#include "Environment.h"
#include "Inspect.h"


no::Model::Model(Timeline& timeline) 
  : m_timeline(timeline), m_monteCarlo()
{ 
  no::log("model init: mc={indep:%%, seed:%%}"_s % no::getenv().indep() % m_monteCarlo.seed());
}


void no::Model::modify(int)
{
  // verbose only
  no::log("defaulted to no-op Model::modify()");
}

void no::Model::step()
{
  throw std::runtime_error("Model.step() method must be overridden");
}

bool no::Model::check()
{
  // verbose only
  no::log("defaulted to no-op Model::check()");
  return true;
}

void no::Model::checkpoint()
{
  throw std::runtime_error("Model.checkpoint() method must be overridden");
}


void no::Model::run(py::object& model_subclass) 
{
  // extract the base class
  no::Model& base = model_subclass.cast<no::Model&>();
  no::Runtime runtime("neworder");
  Timer timer;

  int rank = no::getenv().rank();

  const std::string& subclass_name = py::str(model_subclass).cast<std::string>();

  no::log("starting model run. start time=%%, timestep=%%, checkpoint(s) at %%"_s 
    % base.timeline().start() % base.m_timeline.dt() % base.timeline().checkpoints());

  // apply the modifier, if implemented in the derived class 
  no::log("t=%%(%%) calling: %%.modify(%%)"_s % base.timeline().time() % base.timeline().index() % subclass_name % rank);
  model_subclass.attr("modify")(rank);

  // Loop with checkpoints
  while (!base.timeline().at_end())
  {
    base.timeline().next(); 
    double t = base.timeline().time();
    int timeindex = base.timeline().index();

    no::log("t=%%(%%) calling %%.step()"_s % t % timeindex % subclass_name );
    model_subclass.attr("step")();

    // call the modifier if implemented
//    if (pycpp::has_attr(model_subclass, "check"))
    {
      no::log("t=%%(%%) calling %%.check()"_s % t % timeindex % subclass_name );
      model_subclass.attr("check")();
    }
    if (base.timeline().at_checkpoint())
    {
      no::log("t=%%(%%) calling %%.checkpoint()"_s % t % timeindex % subclass_name );
      model_subclass.attr("checkpoint")();
    } 
  }
  no::log("SUCCESS exec time=%%s"_s % timer.elapsed_s());
}


