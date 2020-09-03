import numpy as np
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD

import neworder

class Parallel(neworder.Model):
  def __init__(self, timeline, p, n):
    # initialise base model (essential!)
    super().__init__(timeline, neworder.MonteCarlo.deterministic_independent_stream)

    # enumerate possible states
    self.s = np.array(range(neworder.mpi.size()))

    # create transition matrix
    self.p = np.identity(neworder.mpi.size()) * (1 - neworder.mpi.size() * p) + p

    # record initial population size
    self.n = n

    # individuals use the index as a unique id and their initial state is the MPI rank
    self.pop = pd.DataFrame({"id": np.array(range(neworder.mpi.rank() * n, (neworder.mpi.rank() + 1) * n)),
                             "state": np.full(n, neworder.mpi.rank()) }).set_index("id")

  def step(self):
    # generate some movement
    neworder.dataframe.transition(self, self.s, self.p, self.pop, "state")

    # send emigrants to other processes
    for s in range(neworder.mpi.size()):
      if s != neworder.mpi.rank():
        emigrants = self.pop[self.pop.state == s]
        neworder.log("sending %d emigrants to %d" % (len(emigrants), s))
        comm.send(emigrants, dest=s)

    # remove the emigrants
    self.pop = self.pop[self.pop.state == neworder.mpi.rank()]

    # receive immigrants
    for s in range(neworder.mpi.size()):
      if s != neworder.mpi.rank():
        immigrants = comm.recv(source=s)
        if len(immigrants):
          neworder.log("received %d immigrants from %d" % (len(immigrants), s))
          self.pop = self.pop.append(immigrants)

  def check(self):
    """ Ensure we havent lost (or gained) anybody """
    totals = comm.gather(len(self.pop), root=0)
    if neworder.mpi.rank() == 0:
      return sum(totals) == self.n * neworder.mpi.size()
    return True

  def checkpoint(self):
    comm.Barrier()
    neworder.log("len(pop)=%d" % len(self.pop))

    # check we only have status = rank now
    assert len(self.pop[self.pop.state != neworder.mpi.rank()]) == 0
