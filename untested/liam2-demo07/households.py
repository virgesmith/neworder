""" households.py """

import numpy as np
import pandas as pd
import neworder

class Households():

  def __init__(self, input_data, key):
    #links:
    #  persons: {type: one2many, target: person, field: hh_id}

    self.hh = pd.read_hdf(input_data, key)
    # cols are just: period, id

    # iterator for new unique ids
    self.iditer = max(self.hh.id) + 1

    # TODO probably more efficient ways to do this?
    nsteps = neworder.ntimesteps + 1
    self.output = pd.DataFrame({'period': np.empty(nsteps), 'N_persons': np.empty(nsteps), 'N_children': np.empty(nsteps)})

    # 
    self.output.period[neworder.timeline.index()] = neworder.timeline.time()
    # TODO...
    self.output.N_persons[neworder.timeline.index()] = 0
    self.output.N_children[neworder.timeline.index()] = 0

  def new(self, n):
    assert n > 0
    self.iditer = self.iditer + n
    newids = np.array(range(self.iditer - n, self.iditer))
    #neworder.log(self.hh.head())
    self.hh = self.hh.append(pd.DataFrame({"period": np.full(n, neworder.timestep), "id": newids}))
    return newids

  def csv_output(self):
    # - csv(period,
    #       avg(persons.count()),
    #       avg(persons.count(age < 18)),
    #       fname='hh_size.csv', mode='a')
    self.output.period[neworder.timeline.index()] = neworder.timeline.time()
    # TODO...
    self.output.N_persons[neworder.timeline.index()] = 0
    self.output.N_children[neworder.timeline.index()] = 0

  def init_reports(self):
    # done in constructor
    pass
      # - csv('period', 'N persons', 'N children',
      #       fname='hh_size.csv')

  def write_reports(self, output_data):
    #self.output.to_hdf(output_data, "hh_size")
    self.output.to_csv(output_data, index=False)

  def clean_empty(self, people):
    # list hh ids in the people array
    active = people.pp.hh_id.unique()
    neworder.log('Number of empty households: %d' % len(self.hh[~self.hh.id.isin(active)]))
    self.hh = self.hh[self.hh.id.isin(active)]
    neworder.log('Number of filled households: %d' % len(self.hh))

