#include "Model.h"
#include "NewOrder.h"
#include "Timeline.h"
#include "Module.h"
#include "Timer.h"
#include "Log.h"
#include "Environment.h"
#include "Inspect.h"


no::Model::Model(Timeline& timeline) 
  : m_timeline(timeline)
  { 
  }


void no::Model::modify(int)
{
  // verbose only
  no::log("defaulted to Model::modify()");
}

void no::Model::transition()
{
  throw std::runtime_error("Model.transition() method must be overridden");
}

bool no::Model::check()
{
  // verbose only
  no::log("defaulted to Model::check()");
  return true;
}

void no::Model::checkpoint()
{
  throw std::runtime_error("Model.checkpoint() method must be overridden");
}


void no::Model::run(py::object& model_subclass, const no::Environment& env) 
{
  no::Runtime runtime("neworder");
  Timer timer;

  int rank = env.rank();

  const std::string& subclass_name = py::str(model_subclass).cast<std::string>();

  no::log("starting model run. start time=%%, timestep=%%, checkpoint(s) at %%"_s 
    % m_timeline.start() % m_timeline.dt() % m_timeline.checkpoints());

  // apply the modifier, if implemented in the derived class 
  no::log("t=%%(%%) calling: %%.modify(%%)"_s % m_timeline.time() % m_timeline.index() % subclass_name % rank);
  model_subclass.attr("modify")(rank);

  // Loop with checkpoints
  while (!m_timeline.at_end())
  {
    m_timeline.next(); 
    double t = m_timeline.time();
    int timeindex = m_timeline.index();

    no::log("t=%%(%%) calling %%.transition()"_s % t % timeindex % subclass_name );
    model_subclass.attr("transition")();

    // call the modifier if implemented
//    if (pycpp::has_attr(model_subclass, "check"))
    {
      no::log("t=%%(%%) calling %%.check()"_s % t % timeindex % subclass_name );
      model_subclass.attr("check")();
    }
    if (m_timeline.at_checkpoint())
    {
      no::log("t=%%(%%) calling %%.checkpoint()"_s % t % timeindex % subclass_name );
      model_subclass.attr("checkpoint")();
    } 
  }
  no::log("SUCCESS exec time=%%s"_s % timer.elapsed_s());
}


