"""
 population.py
"""

import pandas as pd
import numpy as np
import neworder

from helpers import *
import ethpop
import pyramid


# TODO only support single LAD...? (LAD-specific dynamics)
class Population(neworder.Model):
  def __init__(self, timeline, inputdata, asfr, asmr, asir, asor, ascr, asxr):

    super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_seed)

    self.lad = inputdata.split("_")[1]

    self.data = pd.read_csv(inputdata)

    self.fertility = ethpop.create(pd.read_csv(asfr), self.lad)
    self.mortality = ethpop.create(pd.read_csv(asmr), self.lad)
    # assume the in-migration rates are based on the national population and need to be rescaled...
    base_pop = len(self.data)
    # deal with census-merged LADs
    if self.lad == "E09000001" or self.lad == "E09000033":
      base_pop = 219340 + 7397
    elif self.lad == "E06000052" or self.lad == "E06000053":
      raise NotImplementedError("Cornwall CM LAD adj")
    self.in_migration = ethpop.local_rate_from_national_rate(ethpop.create(pd.read_csv(asir), self.lad), base_pop)
    # assume the out-migration rates don't require adjustment
    self.out_migration = ethpop.create(pd.read_csv(asor), self.lad)
    self.immigration = ethpop.local_rate_rescale_from_absolute(ethpop.create(pd.read_csv(ascr), self.lad), base_pop)
    self.emigration = ethpop.local_rate_rescale_from_absolute(ethpop.create(pd.read_csv(asxr), self.lad), base_pop)

    # Force flat rates for testing purposes
    #self.in_migration.Rate = 0.05
    #self.out_migration.Rate = 0.05
    self.immigration.Rate = 0.01
    self.emigration.Rate = 0.005

    # use this to identify people (uniquely only within this table)
    self.counter = len(self.data)

    # Reformatting of input data is required to match Ethpop categories
    # actual age is randomised within the bound of the category
    self.data["Age"] = self.data.DC1117EW_C_AGE - self.mc().ustream(len(self.data))
    self.data = ethpop.from_census_eth(self.data)

  def step(self):
    self.births()
    self.deaths()
    self.migrations()
    self.age()

  def checkpoint(self):
    self.write_table()
    self.plot_pyramid()

  def age(self):
    # Increment age by timestep and update census age categorty (used for ASFR/ASMR lookup)
    # NB census age category max value is 86 (=85 or over)
    self.data.Age = self.data.Age + self.timeline().dt()
    # reconstruct census age group
    self.data.DC1117EW_C_AGE = np.clip(np.ceil(self.data.Age), 1, 86).astype(int)

  def births(self):
    # First consider only females
    females = self.data[self.data.DC1117EW_C_SEX == 2].copy()

    # Now map the appropriate fertility rate to each female
    # might be a more efficient way of generating this array
    rates = females.join(self.fertility, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].values
    # Then randomly determine if a birth occurred (neworder callback)
    h = self.mc().hazard(rates * self.timeline().dt())

    # The babies are a clone of the new mothers, with with changed PID, reset age and randomised gender (keeping location and ethnicity)
    newborns = females[h == 1].copy()
    newborns.PID = range(self.counter, self.counter + len(newborns))
    newborns.Age = self.mc().ustream(len(newborns)) # born within the last 12 months
    newborns.DC1117EW_C_AGE = 1 # this is 0-1 in census category
    # NOTE: do not convert to pd.Series here to stay as this has its own index which conflicts with the main table
    newborns.DC1117EW_C_SEX = self.mc().hazard(0.5, len(newborns)) + 1

    # Finally append newborns to main population and adjust counter
    self.data = self.data.append(newborns)
    self.counter = self.counter + len(newborns)

  def deaths(self):
    # neworder.log("deaths({:.3f})".format(self.timeline().dt()))

    # Map the appropriate mortality rate to each female
    # might be a more efficient way of generating this array
    rates = self.data.join(self.mortality, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"]

    # Then randomly determine if a birth occurred
    h = self.mc().hazard(rates.values * self.timeline().dt())

    # Finally remove deceased from table
    self.data = self.data[h!=1]

  def migrations(self):

    # internal immigration:
    # - assign the rates to the incumbent popultion appropriately by age,sex,ethnicity
    # - randomly sample this population, clone and append
    in_rates = self.data.join(self.in_migration, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].values

    # in-migration should be sampling from the whole population ex-LAD, instead do an approximation by scaling up the LAD population
    # NOTE this is wrong for a number of reasons esp. as it cannot sample category combinations that don't already exist in the LAD
    h_in = self.mc().hazard(in_rates * self.timeline().dt())

    incoming = self.data[h_in == 1].copy()

    # Append incomers to main population and adjust counter
    # Assign a new id
    incoming.PID = range(self.counter, self.counter + len(incoming))
    incoming.Area = self.lad
    # assign a new random fractional age based on census age category
    incoming.Age = incoming.DC1117EW_C_AGE - self.mc().ustream(len(incoming)).tolist()
    self.data = self.data.append(incoming)
    self.counter = self.counter + len(incoming)

    # internal emigration:
    out_rates = self.data.join(self.out_migration, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].values
    h_out = self.mc().hazard(out_rates * self.timeline().dt())
    # remove outgoing migrants
    self.data = self.data[h_out!=1]

    # international immigration
    # Sampling from local population is not ideal
    intl_in_rates = self.data.join(self.immigration, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].values
    h_intl_in = self.mc().hazard(intl_in_rates * self.timeline().dt())

    intl_incoming = self.data[h_intl_in == 1].copy()
    intl_incoming.PID = range(self.counter, self.counter + len(intl_incoming))
    intl_incoming.Area = "INTL" #self.lad
    # assign a new random fractional age based on census age category
    intl_incoming.Age = intl_incoming.DC1117EW_C_AGE - self.mc().ustream(len(intl_incoming))
    self.data = self.data.append(intl_incoming)
    self.counter = self.counter + len(intl_incoming)

    # international emigrtion
    intl_out_rates = self.data.join(self.emigration, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].values
    h_intl_out = self.mc().hazard(intl_out_rates * self.timeline().dt())
    # remove outgoing migrants
    self.data = self.data[h_intl_out!=1]

    # record net migration
    self.in_out = (h_in.sum(), h_out.sum(), h_intl_in.sum(), h_intl_out.sum())

  def mean_age(self):
    return self.data.Age.mean()

  def gender_split(self):
    # this is % female
    return self.data.DC1117EW_C_SEX.mean() - 1.0

  def net_migration():
    # TODO named tuple
    return self.inout[0] - self.in_out[1] + self.in_out[2] - self.in_out[3]

  def size(self):
    return len(self.data)

  def check(self):
    """ State of the nation """
    check(self.data)
    neworder.log("check OK: time={:.3f} size={} mean_age={:.2f}, pct_female={:.2f} net_migration={} ({}-{}+{}-{})" \
      .format(self.timeline().time(), self.size(), self.mean_age(), 100.0 * self.gender_split(),
      self.in_out[0] - self.in_out[1] + self.in_out[2] - self.in_out[3],
      self.in_out[0], self.in_out[1], self.in_out[2], self.in_out[3]))
    return True # Faith

  def write_table(self):
    filename = "./examples/people/dm_{}_{:.3f}.csv".format(self.lad, self.timeline().time())
    neworder.log("writing %s" % filename)
    return self.data.to_csv(filename, index=False)

  def plot_pyramid(self):
    s = self.data.groupby(by=["DC1117EW_C_SEX", "DC1117EW_C_AGE"])["PID"].agg("count").reset_index()
    a = range(85) #s.DC1117EW_C_AGE.unique() - 1
    m = s[s.DC1117EW_C_SEX == 1].PID
    f = s[s.DC1117EW_C_SEX == 2].PID

    pyramid.plot(a, m, f)
