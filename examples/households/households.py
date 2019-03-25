""" households.py """

import os
import numpy as np
import pandas as pd
import neworder as no
import ukpopulation.snhpdata as SNHPData

class Households:
  def __init__(self, cache_dir, init_pop):
    self.cache_dir = cache_dir
    self.pop = pd.read_csv(os.path.join(self.cache_dir, init_pop))
    self.pop["LC4408_C_AHTHUK11_orig"] = self.pop.LC4408_C_AHTHUK11
    # no.log(self.pop.LC4408_C_AHTHUK11.unique())
    # self.cat = self.pop.LC4408_C_AHTHUK11.unique()
    # "C_AHTHUK11": {
    #   "0": "All categories: Household type",
    #   "1": "One person household",
    #   "2": "Married or same-sex civil partnership couple household",
    #   "3": "Cohabiting couple household",
    #   "4": "Lone parent household",
    #   "5": "Multi-person household"
    # }
    self.cat = {"LC4408_C_AHTHUK11": np.array([1,2,3,4,5]) }


    # NOTE: pandas stores column-major order but numpy view is row major so the matrix looks right but is actually transposed
    # (no amount of transposing actually changes the memory layout (it just changes the view)
    # the C++ code assumes the transition matrix is column major (col sums to unity not rows)
    self.t = pd.read_csv(os.path.join(self.cache_dir, "w_hhtype_dv-tpm.csv")).set_index("initial state").values / 100.0
    # check rows sum to unity
    assert np.allclose(np.sum(self.t, 1), np.ones(len(self.t)))

    # TODO get snhp
    self.snhp = SNHPData.SNHPData(self.cache_dir)

    self.projection = self.snhp.aggregate(no.area)
    
  def age(self, dt):
    col = "LC4408_C_AHTHUK11"

    actual = len(self.pop)
    projected = int(self.projection.loc[self.projection["PROJECTED_YEAR_NAME"] == int(no.time), "OBS_VALUE"].values[0])
    #no.log(self.cat[col])
    no.transition(self.cat[col], self.t, self.pop, "LC4408_C_AHTHUK11")
    if actual < projected:
      no.log("sampling deficit %d households (vs projection)" % (projected - actual))
      deficit = int(projected) - actual
      newbuilds = self.pop.sample(deficit)
      self.pop = self.pop.append(newbuilds, ignore_index=True)

  def check(self):
    return True

  def write_table(self, final_population):
    #no.log(self.pop.LC4408_C_AHTHUK11.unique())
    no.log("writing final population to " + final_population)
    self.pop.to_csv(os.path.join(self.cache_dir, final_population), index=False)

