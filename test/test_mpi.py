""" MPI tests """

import numpy as np
import pandas as pd
import neworder as no

if no.mpi.size() != 1:

  from mpi4py import MPI
  comm = MPI.COMM_WORLD

  def send_recv(x):
    if no.mpi.rank() == 0:
      comm.send(x, dest=1)
    if no.mpi.rank() == 1:
      y = comm.recv(source=0)
      no.log("MPI: 0 sent {}={} 1 recd {}={}".format(type(x), x, type(y), y))
      if y != x:
        return False
    return True

  def test_scalar():
    if no.mpi.size() == 1:
      no.log("Skipping MPI tests")
      return True

    assert send_recv(True)
    assert send_recv(10)
    assert send_recv(10.01)
    assert send_recv("abcdef")
    assert send_recv([1,2,3])
    assert send_recv({"a": "fghdfkgh"})

  def test_arrays():

    x = np.array([1,4,9,16])
    if no.mpi.rank() == 0:
      comm.send(x, dest=1)
    if no.mpi.rank() == 1:
      y = comm.recv(source=0)
      assert np.array_equal(x,y)

    df = pd.read_csv("./test/df2.csv")
    if no.mpi.rank() == 0:
      comm.send(df, dest=1)
    if no.mpi.rank() == 1:
      dfrec = comm.recv(source=0)
      assert dfrec.equals(df)

  #   # TODO how to test barrier?
  #   no.log("process %d syncing..." % no.mpi.rank())
  #   no.mpi.sync()
  #   no.log("process %d synced" % no.mpi.rank())

    i = "rank %d" % no.mpi.rank()
    root = 0
    i = comm.bcast(i, root=root)
    # all procs should now have root process value
    assert i == "rank 0"

    # a0 will be different for each proc
    a0 = np.random.rand(2,2)
    a1 = comm.bcast(a0, root)
    # a1 will equal a0 on rank 0 only
    if no.mpi.rank() == 0:
      assert np.array_equal(a0, a1)
    else:
      assert not np.array_equal(a0, a1)

    # base model for MC engine
    model = no.Model(no.Timeline.null(), no.MonteCarlo.deterministic_identical_stream)

    # # check identical streams (independent=False)
    u = model.mc().ustream(1000)
    v = comm.bcast(u, root=root)
    # u == v on all processes
    assert np.array_equal(u, v)

    # base model for MC engine
    model = no.Model(no.Timeline.null(), no.MonteCarlo.deterministic_independent_stream)

    # # check identical streams (independent=False)
    u = model.mc().ustream(1000)
    v = comm.bcast(u, root=root)
    # u != v on all non-root processes
    if no.mpi.rank() != root:
      assert not np.array_equal(u, v)
    else:
      assert np.array_equal(u, v)

  #   # test gather
  #   x = (no.mpi.rank() + 1) ** 2 / 8
  #   a = no.mpi.gather(x, 0)
  #   if no.mpi.rank() == 0:
  #     assert np.array_equal(a, [0.125, 0.5])
  #   else:
  #     assert len(a) == 0
  #   #no.log(a)

  #   # test scatter
  #   if no.mpi.rank() == 0:
  #     a = (np.array(range(no.mpi.size())) + 1) ** 2 / 8
  #   else:
  #     a = np.zeros(no.mpi.size())
  #   no.log(a)
  #   x = no.mpi.scatter(a, 0)
  #   assert x == (no.mpi.rank() + 1) ** 2 / 8

  #   # test allgather
  #   a = np.zeros(no.mpi.size()) - 1
  #   a[no.mpi.rank()] = (no.mpi.rank() + 1) ** 2 / 8
  #   a = no.mpi.allgather(a)
  #   assert np.array_equal(a, np.array([0.125, 0.5]))

  #   # this should probably fail (gather not implemented for int)
  #   x = no.mpi.rank() + 100
  #   a = no.mpi.gather(x, 0)
  #   #no.log(type(x))
  #   #no.log(type(a))
