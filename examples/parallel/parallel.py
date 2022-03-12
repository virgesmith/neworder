# !constructor! this is a tag for inserting code snippets into the documentation
from typing import cast
import numpy as np
import pandas as pd  # type: ignore
from mpi4py import MPI

comm = MPI.COMM_WORLD

import neworder

class Parallel(neworder.Model):
  def __init__(self, timeline: neworder.Timeline, p: float, n: int):
    # initialise base model (essential!)
    super().__init__(timeline, neworder.MonteCarlo.nondeterministic_stream)

    # enumerate possible states
    self.s = np.arange(neworder.mpi.size())

    # create transition matrix with all off-diagonal probabilities equal to p
    self.p = np.identity(neworder.mpi.size()) * (1 - neworder.mpi.size() * p) + p

    # record initial population size
    self.n = n

    # individuals get a unique id and their initial state is the MPI rank
    self.pop = pd.DataFrame({"id": neworder.df.unique_index(n),
                             "state": np.full(n, neworder.mpi.rank()) }).set_index("id")
#!constructor!

  # !step!
  def step(self) -> None:
    # generate some movement
    neworder.df.transition(self, self.s, self.p, self.pop, "state")

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
          self.pop = pd.concat((self.pop, immigrants))
  # !step!

  # !check!
  def check(self) -> bool:
    # Ensure we haven't lost (or gained) anybody
    totals = comm.gather(len(self.pop), root=0)
    if totals:
      if sum(totals) != self.n * neworder.mpi.size():
        return False
    # And check each process only has individuals that it should have
    out_of_place = comm.gather(len(self.pop[self.pop.state != neworder.mpi.rank()]))
    if out_of_place and any(out_of_place):
      return False
    return True
  # !check!

  # !finalise!
  def finalise(self) -> None:
    # process 0 assembles all the data and prints a summary
    pops = comm.gather(self.pop, root=0)
    if pops:
      pop = pd.concat(pops)
      neworder.log("State counts (total %d):\n%s" % (len(pop), pop["state"].value_counts().to_string()))
  # !finalise!
