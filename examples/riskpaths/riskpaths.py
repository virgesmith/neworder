""" RiskPaths model """

import numpy as np
import pandas as pd
import neworder

# dynamics data
from data import UnionState, Parity
import data

# !ctor!
class RiskPaths(neworder.Model):
  def __init__(self, n):

    super().__init__(neworder.NoTimeline(), neworder.MonteCarlo.deterministic_identical_stream)

    # initialise population - time of death only
    self.population = pd.DataFrame(index=neworder.df.unique_index(n),
                                   data={"TimeOfDeath": self.mc.first_arrival(data.mortality_rate, data.mortality_delta_t, n, 0.0),
                                         "TimeOfPregnancy": neworder.time.never(),
                                         "Parity": Parity.CHILDLESS,
                                         "Unions": 0,
                                        })
# !ctor!

  # !step!
  def step(self):

    # first sample union state transitions, which influence pregnancy
    self.__union()
    # now sample state-dependent pregnancies
    self.__pregnancy()
  # !step!

  def __union(self):

    dt_u = data.union_delta_t

    # Construct a timeline of unions for each person
    # first union - probabilities start at 15, so we add this on afterwards
    self.population["T_Union1Start"] = self.mc.first_arrival(data.p_u1f, dt_u, len(self.population)) + data.min_age
    self.population["T_Union1End"] = self.mc.next_arrival(self.population["T_Union1Start"].values, data.r_diss2[0], dt_u, True, data.min_u1)

    # second union
    self.population["T_Union2Start"] = self.mc.next_arrival(self.population["T_Union1End"].values, data.r_u2f, dt_u, True)
    # no mimimum time of 2nd union
    self.population["T_Union2End"] = self.mc.next_arrival(self.population["T_Union2Start"].values, data.r_diss2[1], dt_u, True)

    # and discard events happening after death
    self.population.loc[self.population["T_Union1Start"] > self.population["TimeOfDeath"], "T_Union1Start"] = neworder.time.never()
    self.population.loc[self.population["T_Union1End"] > self.population["TimeOfDeath"], "T_Union1End"] = neworder.time.never()
    self.population.loc[self.population["T_Union2Start"] > self.population["TimeOfDeath"], "T_Union2Start"] = neworder.time.never()
    self.population.loc[self.population["T_Union2End"] > self.population["TimeOfDeath"], "T_Union2End"] = neworder.time.never()

    # count unions entered into
    self.population.Unions = (~neworder.time.isnever(self.population["T_Union1Start"].values)).astype(int) \
                           + (~neworder.time.isnever(self.population["T_Union2Start"].values)).astype(int)

  # !finalise!
  def finalise(self):
    neworder.log("mean unions = %f" % np.mean(self.population.Unions))
    neworder.log("pregnancy ratio = %f" % np.mean(self.population.Parity == Parity.PREGNANT))
  # !finalise!

  def __pregnancy(self):
    # We're interested in the first pregnancy that occurs for each individual
    # fmin ignores nan (np.minimum is a problem as it doesnt deal with nan well)

    dt_f = data.fertility_delta_t

    # pre-union1 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.NEVER_IN_UNION.value]
    # sample
    t_pregnancy1 = self.mc.first_arrival(p_preg, dt_f, len(self.population)) + data.min_age
    # remove pregnancies that happen after union1 formation
    t_pregnancy1[t_pregnancy1 > self.population["T_Union1Start"]] = neworder.time.never()

    # union1 phase1 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.FIRST_UNION_PERIOD1.value]
    # sample
    t_pregnancy1_u1a = self.mc.next_arrival(self.population["T_Union1Start"].values, p_preg, dt_f)
    # discard those that happen after union1 transition
    t_pregnancy1_u1a[t_pregnancy1_u1a > self.population["T_Union1Start"] + data.min_u1] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_u1a)

    # union1 phase2 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.FIRST_UNION_PERIOD2.value]
    # sample
    t_pregnancy1_u1b = self.mc.next_arrival(self.population["T_Union1Start"].values + data.min_u1, p_preg, dt_f)
    # discard those that happen after union1
    t_pregnancy1_u1b[t_pregnancy1_u1b > self.population["T_Union1End"]] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_u1b)

    # post union1 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.AFTER_FIRST_UNION.value]
    # sample
    t_pregnancy1_postu1 = self.mc.next_arrival(self.population["T_Union1End"].values, p_preg, dt_f)
    # discard those that happen after union2 formation
    t_pregnancy1_postu1[t_pregnancy1_postu1 > self.population["T_Union2Start"]] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_postu1)

    # union2 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.SECOND_UNION.value]
    # sample
    t_pregnancy1_u2 = self.mc.next_arrival(self.population["T_Union2Start"].values, p_preg, dt_f)
    # discard those that happen after union2 dissolution
    t_pregnancy1_u2[t_pregnancy1_u2 > self.population["T_Union2End"]] = neworder.time.never()
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_u2)

    # # post union2 pregnancy
    p_preg = data.p_preg * data.r_preg[UnionState.AFTER_SECOND_UNION.value]
    t_pregnancy1_postu2 = self.mc.next_arrival(self.population["T_Union2End"].values, p_preg, dt_f)
    t_pregnancy1 = np.fmin(t_pregnancy1, t_pregnancy1_postu2)

    # add the times to pregnancy1 to the population, removing those pregnancies that occur after death
    self.population["TimeOfPregnancy"] = t_pregnancy1
    self.population.loc[self.population["TimeOfPregnancy"] > self.population["TimeOfDeath"], "TimeOfPregnancy"] = neworder.time.never()
    # and update parity column
    self.population.loc[~neworder.time.isnever(self.population["TimeOfPregnancy"].values), "Parity"] = Parity.PREGNANT
