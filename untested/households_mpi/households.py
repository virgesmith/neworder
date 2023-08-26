""" households.py """

import os
import numpy as np
import pandas as pd
import neworder as no
import ukpopulation.snhpdata as SNHPData

class Households:
  def __init__(self, input_files, ht_trans, cache_dir):

    self.cache_dir = cache_dir
    # guard for no input data (if more MPI processes than input files)
    if not len(input_files):
      raise ValueError("proc {}/{}: no input data".format(no.mpi.RANK(), no.mpi.SIZE()))
    self.lads = [file.split("_")[1] for file in input_files]
    # assumes all files in same dir
    self.data_dir = os.path.dirname(input_files[0])

    # store as dict of DFs
    self.pop = pd.DataFrame()

    for file in input_files:
      no.log("reading initial population: %s" % file)
      data = pd.read_csv(file)
      data["LAD"] = file.split("_")[1]
      self.pop = self.pop.append(data)
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
    self.t = pd.read_csv(ht_trans).set_index("initial state").values / 100.0
    # check rows sum to unity
    assert np.allclose(np.sum(self.t, 1), np.ones(len(self.t)))

    # TODO get snhp
    self.snhp = SNHPData.SNHPData(self.cache_dir)

    self.projection = self.snhp.aggregate(self.lads)

  def age(self, dt):
    col = "LC4408_C_AHTHUK11"
    no.df.transition(self.cat[col], self.t, self.pop, "LC4408_C_AHTHUK11")

    # ensure area totals match projections
    for lad in self.pop["LAD"].unique():
      lad_pop = self.pop[self.pop["LAD"] == lad]
      actual = len(lad_pop)
      # TODO LAD
      projected = self.projection.loc[(self.projection["PROJECTED_YEAR_NAME"] == int(no.time)) &
                                          (self.projection["GEOGRAPHY_CODE"] == lad), "OBS_VALUE"]
      if len(projected) == 0:
        no.log("WARNING %s cannot find household projection data for %d", (lad, no.time))
      projected = int(projected.values[0])
      if actual < projected:
        no.log("sampling deficit %d households (vs projection)" % (projected - actual))
        deficit = int(projected) - actual
        self.pop = self.pop.append(lad_pop.sample(deficit), ignore_index=True)

  def check(self):
    return True

  def write_table(self):
    file = os.path.join(self.data_dir, "dm_{:.3f}_{}-{}.csv".format(no.time, no.mpi.RANK(), no.mpi.SIZE()))
    no.log("writing final population: %s" % file)
    self.pop.to_csv(file, index=False)

    #no.log(self.pop.LC4408_C_AHTHUK11.unique())

