""" MPI tests """

# DEPRECATED


import numpy as np
import pandas as pd
import neworder

import test as test_

def send_recv(x):
  if neworder.mpi.rank() == 0:
    neworder.mpi.send(x, 1)
  if neworder.mpi.rank() == 1:
    y = neworder.mpi.receive(0)
    neworder.log("MPI: 0 sent {}={} 1 recd {}={}".format(type(x), x, type(y), y))
    if y != x:
     return False
  return True

def test():
  t = test_.Test()

  if neworder.mpi.size() == 1:
    neworder.log("Skipping MPI tests")
    return True

  t.check(send_recv(True))
  t.check(send_recv(10))
  t.check(send_recv(10.01))
  t.check(send_recv("abcdef"))
  t.check(send_recv([1,2,3]))
  t.check(send_recv({"a": "fghdfkgh"}))

  x = np.array([1,4,9,16])
  if neworder.mpi.rank() == 0:
    neworder.mpi.send(x, 1)
  if neworder.mpi.rank() == 1:
    y = neworder.mpi.receive(0)
    neworder.log("MPI: 0 sent {}={} 1 recd {}={}".format(type(x), x, type(y), y))
    t.check(np.array_equal(x,y))

  df = pd.read_csv("../../test/df2.csv")
  if neworder.mpi.rank() == 0:
    neworder.log("sending (as csv) df len %d rows from 0" % len(df))
    neworder.mpi.send_csv(df, 1)
  if neworder.mpi.rank() == 1:
    dfrec = neworder.mpi.receive_csv(0)
    neworder.log("got (as csv) df len %d rows from 0" % len(dfrec))
    t.check(dfrec.equals(df))

  if neworder.mpi.rank() == 0:
    neworder.log("sending (pickle) df len %d rows from 0" % len(df))
    neworder.mpi.send(df, 1)
  if neworder.mpi.rank() == 1:
    dfrec = neworder.mpi.receive(0)
    neworder.log("got (pickle) df len %d rows from 0" % len(dfrec))
    t.check(dfrec.equals(df))

  # TODO how to test?
  neworder.log("process %d syncing..." % neworder.mpi.rank())
  neworder.mpi.sync()
  neworder.log("process %d synced" % neworder.mpi.rank())

  i = "rank " + str(neworder.mpi.rank())
  root = 0
  if root == neworder.mpi.rank():
    neworder.log("broadcasting '%s' from %d" % (i, root))
  i = neworder.mpi.broadcast(i, root)
  neworder.log("%d got broadcast: '%s' from %d" % (neworder.mpi.rank(), i, root))

  t.check(i == "rank 0")

  # a0 will be different for each proc
  a0 = np.random.rand(2,2)
  if root == neworder.mpi.rank():
    neworder.log("broadcasting '%s' from %d" % (str(a0), root))
  a1 = neworder.mpi.broadcast(a0, root)
  # a1 will equal a0 on rank 0 only
  neworder.log("%d got broadcast: '%s' from %d" % (neworder.mpi.rank(), str(a1), root))
  if neworder.mpi.rank() == 0:
    t.check(np.array_equal(a0, a1))
  else:
    t.check(not np.array_equal(a0, a1))

  # base model for MC engine
  model = neworder.Model(neworder.Timeline.null(), neworder.MonteCarlo.deterministic_independent_stream)

  # test ustream/sequence
  if root == neworder.mpi.rank():
    u0 = model.mc().ustream(1000)
    u1 = np.zeros(1000)
  else:
    u0 = np.zeros(1000)
    u1 = model.mc().ustream(1000)
  # broadcast u1 from 1
  neworder.mpi.broadcast(u1,1)
  # proc 0 should have 2 different random arrays
  # proc 1 should have zeros and a random array
  t.check(not np.array_equal(u0, u1))

  # check independent streams
  u = model.mc().ustream(1000)
  v = neworder.mpi.broadcast(u, root)

  # u == v on broadcasting process only
  t.check(np.array_equal(u, v) == (neworder.mpi.rank() == root))

  # test gather
  x = (neworder.mpi.rank() + 1) ** 2 / 8
  a = neworder.mpi.gather(x, 0)
  if neworder.mpi.rank() == 0:
    t.check(np.array_equal(a, [0.125, 0.5]))
  else:
    t.check(len(a) == 0)
  #neworder.log(a)

  # test scatter
  if neworder.mpi.rank() == 0:
    a = (np.array(range(neworder.mpi.size())) + 1) ** 2 / 8
  else:
    a = np.zeros(neworder.mpi.size())
  neworder.log(a)
  x = neworder.mpi.scatter(a, 0)
  t.check(x == (neworder.mpi.rank() + 1) ** 2 / 8)

  # test allgather
  a = np.zeros(neworder.mpi.size()) - 1
  a[neworder.mpi.rank()] = (neworder.mpi.rank() + 1) ** 2 / 8
  a = neworder.mpi.allgather(a)
  t.check(np.array_equal(a, np.array([0.125, 0.5])))

  # this should probably fail (gather not implemented for int)
  x = neworder.mpi.rank() + 100
  a = neworder.mpi.gather(x, 0)
  #neworder.log(type(x))
  #neworder.log(type(a))

  return not t.any_failed