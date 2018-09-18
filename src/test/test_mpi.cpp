

#include "test.h"
#include "Environment.h"
#include "MPIResource.h"
#include "MPISendReceive.h"
//#include "Log.h"

template<typename T>
bool send_recv(const T& x, pycpp::Environment& env)
{
  if (env.rank() == 0)
  {
    neworder::mpi::send(x, 1);
  }
  if (env.rank() == 1)
  {
    T y;
    neworder::mpi::receive(y, 0);
    neworder::log("MPI: 0 sent %%=%% 1 recd %%=%%"_s % x % y);
    if (y != x)
     return false;
  }
  return true;
}

void test_mpi()
{
#ifdef NEWORDER_MPI
  pycpp::Environment& env = pycpp::getenv();

  CHECK(env.size() > 1);

  CHECK(send_recv(false, env));
  CHECK(send_recv('a', env));
  CHECK(send_recv(1, env));
  CHECK(send_recv((int64_t)-1, env));
  CHECK(send_recv(71.25, env));
//  CHECK(send_recv("const char*", env));
//  CHECK(send_recv("std::string"_s, env));
  int i = env.rank();
  // will set i to 0 for all procs
  //neworder::log("proc %% i=%%"_s % env.rank() % i);
  neworder::mpi::broadcast(i,0);
  //neworder::log("proc %% i=%%"_s % env.rank() % i);
  CHECK(i == 0);

  std::string s = "env.rank()=%%"_s % env.rank();
  // will set i to 0 for all procs
  //neworder::log("proc %% i=%%"_s % env.rank() % s);
  neworder::mpi::broadcast(s,0);
  //neworder::log("proc %% i=%%"_s % env.rank() % s);
  CHECK(s == "env.rank()=0");


  neworder::mpi::sync();

#endif
}

