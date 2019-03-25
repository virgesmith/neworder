

#include "test.h"
#include "Environment.h"
#include "MPIResource.h"
#include "MPIComms.h"
//#include "Log.h"

template<typename T>
bool send_recv(const T& x, no::Environment& env)
{
  if (env.rank() == 0)
  {
    no::mpi::send(x, 1);
  }
  if (env.rank() == 1)
  {
    T y;
    no::mpi::receive(y, 0);
    //no::log("MPI: 0 sent x=%% 1 recd y=%%"_s % x % y);
    if (y != x)
     return false;
  }
  return true;
}

void test_mpi()
{
#ifdef NEWORDER_MPI
  no::Environment& env = no::getenv();

  CHECK(env.size() > 1);

  CHECK(send_recv(false, env));
  CHECK(send_recv('a', env));
  CHECK(send_recv(1, env));
  CHECK(send_recv((int64_t)-1, env));
  CHECK(send_recv(71.25, env));
  double x = 71.22 + env.rank();
  //no::log("bx=%%"_s % x);
  x = no::mpi::sendrecv(x);  
  // TODO why is this same on all procs?
  //no::log("ax=%%"_s % x);
  CHECK(x == 71.22); // - env.rank());
  //CHECK(send_recv("const char*", env));
  CHECK(send_recv("std::string"_s, env));
  int i = env.rank();
  // will set i to 0 for all procs
  //no::log("proc %% i=%%"_s % env.rank() % i);
  no::mpi::broadcast(i,0);
  //no::log("proc %% i=%%"_s % env.rank() % i);
  CHECK(i == 0);

  std::string s = "env.rank()=%%"_s % env.rank();
  // will set i to 0 for all procs
  //no::log("proc %% i=%%"_s % env.rank() % s);
  no::mpi::broadcast(s,0);
  //no::log("proc %% i=%%"_s % env.rank() % s);
  CHECK(s == "env.rank()=0");

  no::mpi::sync();

  x = 10.0 * env.rank() + env.size();

  std::vector<double> g(env.size(), -1.0);
  no::mpi::gather(x, g, 0);
  if (env.rank() == 0)
  {
    for (size_t i = 0; i < g.size(); ++i)
    {
      CHECK(g[i] == 10.0 * i + env.size());
      //no::log("gather element %%=%%"_s % i % g[i]);
    }
  }
  else 
  {
    CHECK(g.empty());
  }

  std::vector<double> sv(env.size(), -1.0);
  if (env.rank() == 0)
    for (size_t i = 0; i < sv.size(); ++i)
      sv[i] = i * 10.0 + env.size();
  no::mpi::scatter(sv, x, 0);
  CHECK(x == 10.0 * env.rank() + env.size());
  //no::log("scatter rank %% x=%%"_s % env.rank() % x);

  std::vector<double> agv(env.size(), -1.0);
  // give agv one positive element
  agv[env.rank()] = 10.0 * env.rank() + env.size();
  agv = no::mpi::allgather(agv);
  // agv now all positive
  for (size_t i = 0; i < agv.size(); ++i)
  {
    CHECK(agv[i] == 10.0 * i + env.size());
    //no::log("allgather element %%=%%"_s % i % agv[i]);
  }

#endif
}

