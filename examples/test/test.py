import numpy as np
import pandas as pd

import neworder

class Test(neworder.Model):
  def __init__(self, timeline, p, n):
    super().__init__(timeline)
    # states
    self.s = np.array(range(neworder.mpi.size()))
    # transition matrix
    self.p = np.identity(neworder.mpi.size()) * (1 - neworder.mpi.size() * p) + p
    self.n = n

    # all begin with unique id and state = rank
    self.pop = pd.DataFrame({"id": np.array(range(neworder.mpi.rank() * n, neworder.mpi.rank() * n + n)), "state": np.full(n, neworder.mpi.rank()) })

  def transition(self):
    # generate some movement
    neworder.dataframe.transition(self, self.s, self.p, self.pop, "state")

    # send migrants
    for s in range(neworder.mpi.size()):
      if s != neworder.mpi.rank():
        emigrants = self.pop[self.pop.state == s]
        #neworder.log("sending %d emigrants to %d" % (len(emigrants), s))
        neworder.mpi.send(emigrants, s)

    # remove the emigrants
    self.pop = self.pop[self.pop.state == neworder.mpi.rank()]

    # receive migrants
    for s in range(neworder.mpi.size()):
      if s != neworder.mpi.rank():
        immigrants = neworder.mpi.receive(s)
        #neworder.log("received %d immigrants from %d" % (len(immigrants), s))
        self.pop = self.pop.append(immigrants)

  def checkpoint(self):
    neworder.mpi.sync()
    neworder.log("len(pop)=%d" % len(self.pop))

    # check we only have status = rank now
    assert len(self.pop[self.pop.state != neworder.mpi.rank()]) == 0
