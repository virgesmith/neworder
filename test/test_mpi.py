"""MPI tests"""

from typing import Any

import numpy as np
import pandas as pd

import neworder as no

if no.mpi.SIZE == 1:
    no.log("Not in parallel mode, skipping MPI tests")
else:
    no.log("Parallel mode enabled, running MPI tests")

    def send_recv(x: Any) -> bool:
        if no.mpi.RANK == 0:
            no.mpi.COMM.send(x, dest=1)
        if no.mpi.RANK == 1:
            y = no.mpi.COMM.recv(source=0)
            no.log("MPI: 0 sent {}={} 1 recd {}={}".format(type(x), x, type(y), y))
            if y != x:
                return False
        return True

    def test_scalar() -> None:
        assert send_recv(True)
        assert send_recv(10)
        assert send_recv(10.01)
        assert send_recv("abcdef")
        assert send_recv([1, 2, 3])
        assert send_recv({"a": "fghdfkgh"})

    def test_arrays() -> None:
        x = np.array([1, 4, 9, 16])
        if no.mpi.RANK == 0:
            no.mpi.COMM.send(x, dest=1)
        if no.mpi.RANK == 1:
            y = no.mpi.COMM.recv(source=0)
            assert np.array_equal(x, y)

        df = pd.read_csv("./test/df2.csv")
        if no.mpi.RANK == 0:
            no.mpi.COMM.send(df, dest=1)
        if no.mpi.RANK == 1:
            dfrec = no.mpi.COMM.recv(source=0)
            assert dfrec.equals(df)

        i = "rank %d" % no.mpi.RANK
        root = 0
        i = no.mpi.COMM.bcast(i, root=root)
        # all procs should now have root process value
        assert i == "rank 0"

        # a0 will be different for each proc
        a0 = np.random.rand(2, 2)
        a1 = no.mpi.COMM.bcast(a0, root)
        # a1 will equal a0 on rank 0 only
        if no.mpi.RANK == 0:
            assert np.array_equal(a0, a1)
        else:
            assert not np.array_equal(a0, a1)

        # base model for MC engine
        model = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

        # # check identical streams (independent=False)
        u = model.mc.ustream(1000)
        v = no.mpi.COMM.bcast(u, root=root)
        # u == v on all processes
        assert np.array_equal(u, v)

        # base model for MC engine
        model = no.Model(
            no.NoTimeline(), no.MonteCarlo.deterministic_independent_stream
        )

        # # check identical streams (independent=False)
        u = model.mc.ustream(1000)
        v = no.mpi.COMM.bcast(u, root=root)
        # u != v on all non-root processes
        if no.mpi.RANK != root:
            assert not np.array_equal(u, v)
        else:
            assert np.array_equal(u, v)
