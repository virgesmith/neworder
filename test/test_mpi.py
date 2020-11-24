""" MPI tests """

import numpy as np
import pandas as pd
import neworder as no

if no.mpi.size() == 1:
  no.log("No MPI env detected, skipping MPI tests")

else:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD

  no.log("MPI env detected, running MPI tests")

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
    model = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

    # # check identical streams (independent=False)
    u = model.mc().ustream(1000)
    v = comm.bcast(u, root=root)
    # u == v on all processes
    assert np.array_equal(u, v)

    # base model for MC engine
    model = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_independent_stream)

    # # check identical streams (independent=False)
    u = model.mc().ustream(1000)
    v = comm.bcast(u, root=root)
    # u != v on all non-root processes
    if no.mpi.rank() != root:
      assert not np.array_equal(u, v)
    else:
      assert np.array_equal(u, v)

