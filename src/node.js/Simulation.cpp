
#include "Simulation.h"

#include <v8.h>

Simulation::Simulation() : Simulation(2011.0, 2018.0, 1.0) { }

Simulation::Simulation(double start, double end, double stepsize) : m_time(start), m_end(end), m_stepsize(stepsize) 
{ 
  m_isolate = v8::Isolate::GetCurrent();
  // Create a new context.
  m_context = v8::Context::New(m_isolate);
}

Simulation::~Simulation() 
{
  //m_isolate->Dispose();
}

v8::Isolate* Simulation::isolate() const
{
  return m_isolate;
}

v8::Local<v8::Context>& Simulation::context() 
{
  return m_context;
}
