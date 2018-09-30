"""
Test framework expects modules with a function called test that
- takes no arguments
- returns True on success and False on failure
"""
import neworder as no
import numpy as np
import pandas as pd

import test as test_

def test():
  t = test_.Test()

  #t.check(False)
  s = no.ustream(10000)
  t.check(isinstance(s, np.ndarray))

  t.check(len(s) == 10000)

  t.check(abs(np.mean(s) - 0.5) < 0.02)

  f = no.lazy_eval("2 + 2")
  t.check(f() == 4)

  # TODO this overlaps/duplicates tests in op.py - reorganise

  # test thinning algorithm for non-homogeneous Poisson process
  h = np.array([0.014] * 10)
  #l = no.stopping_v(h)
  l = no.stopping_nhpp(h, 1.0, 10000)
  t.check(abs(np.mean(l) * 0.014 - 1.0) < 0.03)
  # varying timestep should make no difference
  l = no.stopping_nhpp(h, 0.1, 10000)
  t.check(abs(np.mean(l) * 0.014 - 1.0) < 0.03)

  # test a certain(ish) hazard rate
  h = np.array([0.99, 0.99, 0.01])
  l = no.stopping_nhpp(h, 1.0, 10000)
  no.log("TODO NHPP appears broken: %f" % np.mean(l))

  # test a zero(ish) hazard rate
  h = np.array([1e-30, 1e-30, 1e-30, .9999])
  l = no.stopping_nhpp(h, 1.0, 10000)
  no.log("TODO NHPP appears broken: %f" % np.mean(l))

  # this also tests a zero hazard rate 
  h = np.array([i/3000 for i in range(100)])
  #no.log(h)
  le = no.stopping_nhpp(h, 1.0, 10000)
  no.log(sum(le)/len(le))

  # not convinced this is working correctly
  h = np.array([0.999, 0.1])
  le = no.stopping_nhpp(h, 1.0, 1000)
  no.log(sum(le)/len(le))

  # modify df passing column 
  df = pd.read_csv("../../tests/df.csv")

  # modify df passing directly
  no.directmod(df, "DC2101EW_C_ETHPUK11")
  t.check(np.array_equal(df["DC2101EW_C_ETHPUK11"].values, np.zeros(len(df)) + 3))

  df = pd.read_csv("../../tests/df.csv")
  cats = np.array(range(4))
  transitions = np.identity(len(cats)) * 0 + 0.25
  #no.log(transitions)
  no.transition(cats, transitions, df, "DC2101EW_C_ETHPUK11")
  # it's possible this could fail depending on random draw
  t.check(np.array_equal(np.sort(df["DC2101EW_C_ETHPUK11"].unique()), np.array(range(4))))

  df2 = df.copy()
  df3 = no.append(df,df2)
  t.check(len(df3) == len(df) + len(df2)) 

  return not t.any_failed
