"""
 population.py
"""

import pandas as pd
import numpy as np
import neworder

from helpers import *

def _col(age, sex):
  col = "M" if sex == 1 else "F" if sex == 2 else "?"
  col = col + str(age-1) + "." + str(age)
  return col

 
# TODO only support single LAD...? (LAD-specific dynamics)
class Population:
  def __init__(self, inputdata, asfr, asmr, asir, asor):

    # guard for no input data (if more MPI processes than input files)
    if not len(inputdata):
      raise ValueError("proc {}/{}: no input data".format(neworder.procid, neworder.nprocs))

    self.lad = inputdata[0].split("_")[1]

    self.data = pd.DataFrame()
    for file in inputdata: 
      self.data = self.data.append(pd.read_csv(file))

    self.fertility = pd.read_csv(asfr, index_col=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"]) 
    self.mortality = pd.read_csv(asmr, index_col=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])
    self.in_migration = create_migration_table(pd.read_csv(asir), self.lad)
    self.out_migration = create_migration_table(pd.read_csv(asor), self.lad)

    # seed RNG: for now, rows in data * sum(DC1117EW_C_AGE) - TODO add MPI rank/size?
    seed = int(len(self.data) * self.data.DC1117EW_C_AGE.sum()) 
    neworder.log("{} seed: {}".format(self.lad, seed)) 
    self.rstream = neworder.UStream(seed)

    # use this to identify people (uniquely only within this table)
    self.counter = len(self.data)

    # Reformatting of input data is required to match Ethpop categories
    # actual age is randomised within the bound of the category
    # TODO segfault can occur if mix ops with DVector and array/list...
    self.data["Age"] = self.data.DC1117EW_C_AGE - self.rstream.get(len(self.data)).tolist()
    self.data = census_eth_to_newethpop_eth(self.data)

  def age(self, deltat):
    # Increment age by timestep and update census age categorty (used for ASFR/ASMR lookup)
    # NB census age category max value is 86 (=85 or over)
    self.data.Age = self.data.Age + deltat
    # reconstruct census age group
    self.data.DC1117EW_C_AGE = np.clip(np.ceil(self.data.Age), 1, 86)

  def births(self, deltat):
    # First consider only females
    females = self.data[self.data.DC1117EW_C_SEX == 2].copy()

    # Now map the appropriate fertility rate to each female
    # might be a more efficient way of generating this array
    rates = females.join(self.fertility, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()
    # Then randomly determine if a birth occurred (neworder callback)
    # python disallows scalar float mult of float list, but neworder.DVector does support this
    h = np.array(neworder.hazard_v(neworder.DVector.fromlist(rates) * deltat).tolist())

    # The babies are a clone of the new mothers, with with changed PID, reset age and randomised gender (keeping location and ethnicity)
    newborns = females[h == 1].copy()
    newborns.PID = range(self.counter, self.counter + len(newborns))
    newborns.Age = self.rstream.get(len(newborns)).tolist() # born within the last 12 months
    newborns.DC1117EW_C_AGE = 1 # this is 0-1 in census category
    # NOTE: do not convert to pd.Series here to stay as this has its own index which conflicts with the main table
    newborns.DC1117EW_C_SEX = np.array(neworder.hazard(0.5, len(newborns)).tolist()) + 1

    # Finally append newborns to main population and adjust counter
    self.data = self.data.append(newborns)
    self.counter = self.counter + len(newborns)
  
  def deaths(self, deltat):
    # neworder.log("deaths({:.3f})".format(deltat))

    # Map the appropriate mortality rate to each female
    # might be a more efficient way of generating this array
    rates = self.data.join(self.mortality, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()

    # Then randomly determine if a birth occurred
    # neworder callback (requires inefficient conversions: Series/np.array -> list -> DVector -> list -> np.array)
    # python disallows scalar float mult of float list, but neworder.DVector does support this
    h = np.array(neworder.hazard_v(neworder.DVector.fromlist(rates) * deltat).tolist())

    # Finally remove deceased from table
    self.data = self.data[h!=1]
    
  def migrations(self, deltat):

    # in-migrations: 
    # - assign the rates to the incumbent popultion appropriately by age,sex,ethnicity
    # - randomly sample this population, clone and append
    in_rates = self.data.join(self.in_migration, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()

    # in-migration should be sampling from the whole population ex-LAD, instead do an approximation by scaling up the LAD population
    # NOTE this is wrong for a number of reasons esp. as it cannot sample category combinations that don't already exist in the LAD
    scale = 50000000.0 / len(self.data)
    h_in = np.array(neworder.hazard_v(neworder.DVector.fromlist(in_rates) * scale * deltat).tolist())
    
    incoming = self.data[h_in == 1].copy()

    out_rates = self.data.join(self.out_migration, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()

    h_out = np.array(neworder.hazard_v(neworder.DVector.fromlist(out_rates) * deltat).tolist())

    # remove outgoing migrants
    self.data = self.data[h_out!=1]

    # Finally append incomers to main population and adjust counter
    # Assign a new id
    incoming.PID = range(self.counter, self.counter + len(incoming))
    incoming.Area = self.lad
    # assign a new random fractional age based on census age category
    incoming.Age = incoming.Age - self.rstream.get(len(incoming)).tolist()
    self.data = self.data.append(incoming)
    self.counter = self.counter + len(incoming)

    # TODO incorporate international migration

    # record net migration
    self.in_out = (h_in.sum(), h_out.sum())

  def mean_age(self):
    return self.data.Age.mean()

  def gender_split(self):
    # this is % female
    return self.data.DC1117EW_C_SEX.mean() - 1.0

  def net_migration():
    return self.inout[0] - self.in_out[1]

  def size(self):
    return len(self.data)

  def check(self):
    """ State of the nation """
    neworder.log("check OK: time={:.3f} size={} mean_age={:.2f}, pct_female={:.2f} net_migration={}-{}" \
      .format(neworder.time, self.size(), self.mean_age(), 100.0 * self.gender_split(), self.in_out[0], self.in_out[1]))
    return True # Faith

  def write_table(self, output_file_callback):
    filename = output_file_callback()
    neworder.log("writing %s" % filename)
    return self.data.to_csv(filename, index=False)
