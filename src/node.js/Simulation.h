#pragma once

#include "Global.h"

//#include "v8_fwd.h"
#include <v8.h>

class Simulation
{
public:
  Simulation();
  Simulation(double start, double end, double stepsize);

  v8::Isolate* isolate() const;

  v8::Local<v8::Context>& context();

  ~Simulation();
  Simulation(const Simulation&) = delete;

private:
  // TODO reset VM
  // TODO context?

  double m_time;
  double m_end;
  double m_stepsize;
  v8::Isolate* m_isolate; 

  v8::Local<v8::Context> m_context;
};