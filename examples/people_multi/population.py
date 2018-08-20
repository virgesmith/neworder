"""
Large-scale version of population.py
"""

import pandas as pd
import numpy as np
import neworder

from helpers import *


# TODO only support single LAD...? (LAD-specific dynamics)
class Population:
  def __init__(self, inputdata, asfr, asmr, asir, asor, ascr, asxr):

    # guard for no input data (if more MPI processes than input files)
    if not len(inputdata):
      raise ValueError("proc {}/{}: no input data".format(neworder.procid, neworder.nprocs))

    self.lads = [file.split("_")[2] for file in inputdata]

    self.data = pd.DataFrame()
    for file in inputdata: 
      data = pd.read_csv(file)
      data["LAD"] = file.split("_")[2]
      self.data = self.data.append(data)

    neworder.log("Preprocessing transition data for %s" % ", ".join(self.lads))
    self.fertility = create_multi_from_ethpop_data(pd.read_csv(asfr), self.lads)
    self.mortality = create_multi_from_ethpop_data(pd.read_csv(asmr), self.lads)
    self.in_migration = local_rates_from_national_rate(create_multi_from_ethpop_data(pd.read_csv(asir), self.lads), self.data)
    self.out_migration = create_multi_from_ethpop_data(pd.read_csv(asor), self.lads)
    self.immigration = create_multi_from_ethpop_data(pd.read_csv(ascr), self.lads)
    self.emigration = create_multi_from_ethpop_data(pd.read_csv(asxr), self.lads)

    # # converts fractional category totals into individuals
    # self.immigrants = generate_intl_migrants(self.immigration, True)
    # # for emigrants don't need individuals
    # self.emigrants = generate_intl_migrants(self.emigration, False)

    # seed RNG: for now, rows in data * sum(DC1117EW_C_AGE) - TODO add MPI rank/size?
    seed = int(len(self.data) * self.data.DC1117EW_C_AGE.sum()) 
    neworder.log("{} seed: {}".format(", ".join(self.lads), seed)) 
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
    self.data.DC1117EW_C_AGE = np.clip(np.ceil(self.data.Age), 1, 86).astype(int)

  def births(self, deltat):
    # First consider only females
    females = self.data[self.data.DC1117EW_C_SEX == 2].copy()

    # Now map the appropriate fertility rate to each female
    # might be a more efficient way of generating this array
    rates = females.join(self.fertility, on=["LAD", "NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()
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
    self.data = self.data.append(newborns, sort=False)
    self.counter = self.counter + len(newborns)
  
  def deaths(self, deltat):
    # neworder.log("deaths({:.3f})".format(deltat))

    # Map the appropriate mortality rate to each female
    # might be a more efficient way of generating this array
    rates = self.data.join(self.mortality, on=["LAD", "NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()

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
    in_rates = self.data.join(self.in_migration, on=["LAD", "NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()

    # in-migration should be sampling from the whole population ex-LAD, instead do an approximation by scaling up the LAD population
    # NOTE this is wrong for a number of reasons esp. as it cannot sample category combinations that don't already exist in the LAD
    h_in = np.array(neworder.hazard_v(neworder.DVector.fromlist(in_rates) * deltat).tolist())
    
    incoming = self.data[h_in == 1].copy()

    out_rates = self.data.join(self.out_migration, on=["LAD", "NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()

    h_out = np.array(neworder.hazard_v(neworder.DVector.fromlist(out_rates) * deltat).tolist())

    # remove outgoing migrants
    self.data = self.data[h_out!=1]

    # Append incomers to main population and adjust counter
    # Assign a new id
    incoming.PID = range(self.counter, self.counter + len(incoming))
    incoming.Area = incoming.LAD
    # assign a new random fractional age based on census age category
    incoming.Age = incoming.DC1117EW_C_AGE - self.rstream.get(len(incoming)).tolist()
    self.data = self.data.append(incoming, sort=False)
    self.counter = self.counter + len(incoming)
    
    # international disabled for now
    # intl_incoming = self.immigrants.copy()
    # intl_incoming["PID"] = range(self.counter, self.counter + len(intl_incoming))
    # intl_incoming["Area"] = self.lad
    # # assign a new random fractional age based on census age category
    # intl_incoming["Age"] = intl_incoming.DC1117EW_C_AGE - self.rstream.get(len(intl_incoming)).tolist()
    # self.data = self.data.append(intl_incoming, sort=False)
    # self.counter = self.counter + len(intl_incoming)

    # # TODO emigration
    # intl_outgoing = self.emigrants.copy()
    # #print(intl_outgoing.head())
    # for i in range(len(intl_outgoing)):
    #   ref = self.data[(self.data.NewEthpop_ETH == intl_outgoing.iloc[i].NewEthpop_ETH)
    #     & (self.data.DC1117EW_C_SEX == intl_outgoing.iloc[i].DC1117EW_C_SEX)
    #     & (self.data.DC1117EW_C_AGE == intl_outgoing.iloc[i].DC1117EW_C_AGE)]
    #   count = intl_outgoing.iloc[i].Rate
    #   print(count, len(ref))
    #   print(str(ref.sample(min(count, len(ref)), replace=False)))
      
    #print(len(self.data), len(intl_outgoing))
    #self.data

    # record net migration
    self.in_out = (h_in.sum(), h_out.sum(), 0, 0) # len(self.immigrants), len(self.emigrants))

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
    #check(self.data)
    neworder.log("check OK: time={:.3f} size={} mean_age={:.2f}, pct_female={:.2f} net_migration={} ({}-{}+{}-{})" \
      .format(neworder.time, self.size(), self.mean_age(), 100.0 * self.gender_split(), 
      self.in_out[0] - self.in_out[1] + self.in_out[2] - self.in_out[3], 
      self.in_out[0], self.in_out[1], self.in_out[2], self.in_out[3]))
    return True # Faith

  def write_table(self):
    # TODO define path in config
    filename = "./examples/people_multi/data/dm_{:.3f}_{}-{}.csv".format(neworder.time, neworder.procid, neworder.nprocs)
    neworder.log("writing %s" % filename)
    return self.data.to_csv(filename, index=False)
