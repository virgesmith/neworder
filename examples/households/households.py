""" households.py """

import numpy as np
import pandas as pd
import neworder as no

class Households:
  def __init__(self, init_pop):
    self.pop = pd.read_csv(init_pop)
    #no.log(self.pop.columns.values)
    no.log(self.pop.LC4408_C_AHTHUK11.unique())
    c = self.pop.LC4408_C_AHTHUK11.unique()
    m = np.identity(len(c))
    

  def age(self, dt):

    #LC4408_C_AHTHUK11
    no.directmod(self.pop, "LC4408_C_AHTHUK11")

  def check(self):
    return True

  def write_table(self):
    no.log(self.pop.LC4408_C_AHTHUK11.unique())
