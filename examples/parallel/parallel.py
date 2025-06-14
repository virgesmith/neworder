# !constructor! this is a tag for inserting code snippets into the documentation
import numpy as np
import pandas as pd  # type: ignore

import neworder


class Parallel(neworder.Model):
    def __init__(self, timeline: neworder.Timeline, p: float, n: int):
        # initialise base model (essential!)
        super().__init__(timeline, neworder.MonteCarlo.nondeterministic_stream)

        # enumerate possible states
        self.s = np.arange(neworder.mpi.SIZE)

        # create transition matrix with all off-diagonal probabilities equal to p
        self.p = np.identity(neworder.mpi.SIZE) * (1 - neworder.mpi.SIZE * p) + p

        # record initial population size
        self.n = n

        # individuals get a unique id and their initial state is the MPI rank
        self.pop = pd.DataFrame({"id": neworder.df.unique_index(n), "state": np.full(n, neworder.mpi.RANK)}).set_index(
            "id"
        )

    #!constructor!

    # !step!
    def step(self) -> None:
        # generate some movement
        neworder.df.transition(self, self.s, self.p, self.pop, "state")

        # send emigrants to other processes
        for s in range(neworder.mpi.SIZE):
            if s != neworder.mpi.RANK:
                emigrants = self.pop[self.pop.state == s]
                neworder.log("sending %d emigrants to %d" % (len(emigrants), s))
                neworder.mpi.COMM.send(emigrants, dest=s)

        # remove the emigrants
        self.pop = self.pop[self.pop.state == neworder.mpi.RANK]

        # receive immigrants
        for s in range(neworder.mpi.SIZE):
            if s != neworder.mpi.RANK:
                immigrants = neworder.mpi.COMM.recv(source=s)
                if len(immigrants):
                    neworder.log("received %d immigrants from %d" % (len(immigrants), s))
                    self.pop = pd.concat((self.pop, immigrants))

    # !step!

    # !check!
    def check(self) -> bool:
        # Ensure we haven't lost (or gained) anybody
        totals = neworder.mpi.COMM.gather(len(self.pop), root=0)
        if totals:
            if sum(totals) != self.n * neworder.mpi.SIZE:
                return False
        # And check each process only has individuals that it should have
        out_of_place = neworder.mpi.COMM.gather(len(self.pop[self.pop.state != neworder.mpi.RANK]))
        if out_of_place and any(out_of_place):
            return False
        return True

    # !check!

    # !finalise!
    def finalise(self) -> None:
        # process 0 assembles all the data and prints a summary
        pops = neworder.mpi.COMM.gather(self.pop, root=0)
        if pops:
            pop = pd.concat(pops)
            neworder.log("State counts (total %d):\n%s" % (len(pop), pop["state"].value_counts().to_string()))

    # !finalise!
