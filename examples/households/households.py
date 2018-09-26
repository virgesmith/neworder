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
    t = np.array([[0.9,  0.05, 0.05, 0.0,  0.0,  0.0], 
                  [0.05, 0.9,  0.04, 0.01, 0.0,  0.0], 
                  [0.0,  0.05, 0.9,  0.05, 0.0,  0.0], 
                  [0.0,  0.0,  0.05, 0.9,  0.05, 0.0], 
                  [0.1,  0.1,  0.1,  0.1,  0.5,  0.1], 
                  [0.0,  0.0,  0.00, 0.0,  0.0,  1.0]])
    #t = np.identity(len(c))
    
    no.transition(c, np.cumsum(t, axis=1), self.pop.LC4408_C_AHTHUK11.values)

  def age(self, dt):

    #LC4408_C_AHTHUK11
    no.directmod(self.pop, "LC4408_C_AHTHUK11")

  def check(self):
    return True

  def write_table(self):
    no.log(self.pop.LC4408_C_AHTHUK11.unique())
