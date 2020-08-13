""" RiskPaths model """

import numpy as np
import pandas as pd
import neworder
from matplotlib import pyplot as plt

# dynamics data
from data import UnionState
from data import Parity
import data

class RiskPaths(neworder.Model):
  def __init__(self, timeline, n):

    super().__init__(timeline)

    # initialise population - time of death only
    self.population = pd.DataFrame(data={"TimeOfDeath": self.mc().first_arrival(data.mortality_rate, timeline.dt(), n, 0.0),
                                         "TimeOfPregnancy": np.full(n, neworder.time.never()),
                                         "Parity": np.full(n, Parity.CHILDLESS),
                                         "Unions": np.zeros(n, dtype=int),
                                        })

    # Construct a timeline of unions for each person
    # first union - probabilities start at 15, so we add this on afterwards
    self.population["T_Union1Start"] = self.mc().first_arrival(data.p_u1f, data.delta_t, len(self.population)) + data.min_age
    self.population["T_Union1End"] = self.mc().next_arrival(self.population["T_Union1Start"].values, data.r_diss2[0], data.delta_t_u, True, data.min_u1)

    # second union
    self.population["T_Union2Start"] = self.mc().next_arrival(self.population["T_Union1End"].values, data.r_u2f, data.delta_t, True)
    # no mimimum time of 2nd union
    self.population["T_Union2End"] = self.mc().next_arrival(self.population["T_Union2Start"].values, data.r_diss2[1], data.delta_t_u, True)

    # and discard events happening after death
    self.population.loc[self.population["T_Union1Start"] > self.population["TimeOfDeath"], "T_Union1Start"] = neworder.time.never()
    self.population.loc[self.population["T_Union1End"] > self.population["TimeOfDeath"], "T_Union1End"] = neworder.time.never()
    self.population.loc[self.population["T_Union2Start"] > self.population["TimeOfDeath"], "T_Union2Start"] = neworder.time.never()
    self.population.loc[self.population["T_Union2End"] > self.population["TimeOfDeath"], "T_Union2End"] = neworder.time.never()

    # count unions entered into
    self.population.Unions = (~neworder.time.isnever(self.population["T_Union1Start"].values)).astype(int) \
                           + (~neworder.time.isnever(self.population["T_Union2Start"].values)).astype(int)

    neworder.log("RiskPaths initialised")


  def transition(self):
    #"a": "import sys; neworder.log(sys.argv)",
    self.pregnancy()

  def checkpoint(self):
    self.stats()
    self.plot()

  def pregnancy(self):
    # We're interested in the first pregnancy that occurs for each individual
    # fmin ignores nan (np.minimum is a problem as it doesnt deal with nan well)

    # pre-union1 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.NEVER_IN_UNION.value]
    # sample
    t_pregnancy1 = self.mc().first_arrival(p_preg, data.delta_t, len(self.population)) + data.min_age
    # remove pregnancies that happen after union1 formation
    t_pregnancy1[t_pregnancy1 > self.population["T_Union1Start"]] = neworder.time.never()

    # union1 phase1 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.FIRST_UNION_PERIOD1.value]
    # sample
    t_pregnancy1_u1a = self.mc().next_arrival(self.population["T_Union1Start"].values, p_preg, data.delta_t)
    # discard those that happen after union1 transition
    t_pregnancy1_u1a[t_pregnancy1_u1a > self.population["T_Union1Start"] + data.min_u1] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_u1a)

    # union1 phase2 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.FIRST_UNION_PERIOD2.value]
    # sample
    t_pregnancy1_u1b = self.mc().next_arrival(self.population["T_Union1Start"].values + data.min_u1, p_preg, data.delta_t)
    # discard those that happen after union1
    t_pregnancy1_u1b[t_pregnancy1_u1b > self.population["T_Union1End"]] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_u1b)

    # post union1 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.AFTER_FIRST_UNION.value]
    # sample
    t_pregnancy1_postu1 = self.mc().next_arrival(self.population["T_Union1End"].values, p_preg, data.delta_t)
    # discard those that happen after union2 formation
    t_pregnancy1_postu1[t_pregnancy1_postu1 > self.population["T_Union2Start"]] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_postu1)

    # union2 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.SECOND_UNION.value]
    # sample
    t_pregnancy1_u2 = self.mc().next_arrival(self.population["T_Union2Start"].values, p_preg, data.delta_t)
    # discard those that happen after union2 dissolution
    t_pregnancy1_u2[t_pregnancy1_u2 > self.population["T_Union2End"]] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_u2)

    # # post union2 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.AFTER_SECOND_UNION.value]
    t_pregnancy1_postu2 = self.mc().next_arrival(self.population["T_Union2End"].values, p_preg, data.delta_t)
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_postu2)

    # add the times to pregnancy1 to the population, removing those pregnancies that occur after death
    self.population["TimeOfPregnancy"] = t_pregnancy1
    self.population.loc[self.population["TimeOfPregnancy"] > self.population["TimeOfDeath"], "TimeOfPregnancy"] = neworder.time.never()
    # and update parity column
    self.population.loc[~neworder.time.isnever(self.population["TimeOfPregnancy"].values), "Parity"] = Parity.PREGNANT

    # save population
    self.population.to_csv("./population.csv", index=False)

  def stats(self):
    neworder.log("mean unions = %f" % np.mean(self.population.Unions))
    neworder.log("pregnancy ratio = %f" % np.mean(self.population.Parity == Parity.PREGNANT))

  def plot(self):
    b = [ self.population.T_Union1Start[~neworder.time.isnever(self.population.T_Union1Start.values)],
          self.population.T_Union1End[~neworder.time.isnever(self.population.T_Union1End.values)],
          self.population.T_Union2Start[~neworder.time.isnever(self.population.T_Union2Start.values)],
          self.population.T_Union2End[~neworder.time.isnever(self.population.T_Union2End.values)] ]

    plt.hist(b, bins=range(100), stacked=True)
    plt.hist(self.population.TimeOfPregnancy[~neworder.time.isnever(self.population.TimeOfPregnancy.values)], range(100), color='purple')
    plt.legend(["Union 1 starts", "Union 1 ends", "Union 2 starts", "Union 2 ends", "Pregnancy"])
    plt.xlabel("Age (y)")
    plt.ylabel("Frequency")
    #plt.savefig("./doc/examples/img/riskpaths.png")
    plt.show()
