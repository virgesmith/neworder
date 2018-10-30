""" households.py """

import pandas as pd
import neworder

class Households():

  def __init__(self, input_data, key):
    #links:
    #  persons: {type: one2many, target: person, field: hh_id}

    self.hh = pd.read_hdf(input_data, key)

    neworder.log("HH init:")
    neworder.log(self.hh.head())

  def csv_output():
    pass
    # - csv(period,
    #       avg(persons.count()),
    #       avg(persons.count(age < 18)),
    #       fname='hh_size.csv', mode='a')

  def init_reports(self):
    pass
      # - csv('period', 'N persons', 'N children',
      #       fname='hh_size.csv')

  def clean_empty(self):
    neworder.log('Number of empty households:' % len(self.hh[self.hh.X == 0]))
    self.hh = self.hh[self.hh.X != 0]

