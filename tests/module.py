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

  x = -1e10
  t.check(no.distant_past() < x)
  t.check(no.far_future() > x)
  x = 1e10
  t.check(no.distant_past() < x)
  t.check(no.far_future() > x)

  # dreams never end
  t.check(no.never() != no.never())
  t.check(not no.never() == x)
  t.check(no.never() != x)
  t.check(not x < no.never())
  t.check(not x >= no.never())
  # no nay never: 
  t.check(not no.isnever(x))
  # no nay never no more:
  t.check(no.isnever(no.never()))

  #t.check(False)
  s = no.mc.ustream(10000)
  t.check(isinstance(s, np.ndarray))

  t.check(len(s) == 10000)

  t.check(abs(np.mean(s) - 0.5) < 0.02)

  # # TODO this overlaps/duplicates tests in op.py - reorganise

  # # test thinning algorithm for non-homogeneous Poisson process
  # h = np.array([0.014] * 10)
  # #l = no.stopping(h)
  # l = no.first_arrival(h, 1.0, 10000)
  # t.check(abs(np.mean(l) * 0.014 - 1.0) < 0.03)
  # # varying timestep should make no difference
  # l = no.first_arrival(h, 0.1, 10000)
  # t.check(abs(np.mean(l) * 0.014 - 1.0) < 0.03)

  # # test a certain(ish) hazard rate
  # h = np.array([0.99, 0.99, 0.01])
  # l = no.first_arrival(h, 1.0, 10000)
  # no.log("TODO NHPP appears broken: %f" % np.mean(l))

  # # test a zero(ish) hazard rate
  # h = np.array([1e-30, 1e-30, 1e-30, .9999])
  # l = no.first_arrival(h, 1.0, 10000)
  # no.log("TODO NHPP appears broken: %f" % np.mean(l))

  # # this also tests a zero hazard rate 
  # h = np.array([i/3000 for i in range(100)])
  # #no.log(h)
  # le = no.first_arrival(h, 1.0, 10000)
  # no.log(sum(le)/len(le))

  # # y
  # h = np.array([0.999, 0.1])
  # le = no.first_arrival(h, 1.0, 1000)
  # no.log(sum(le)/len(le))

  sometime = no.isnever(np.full(10, 1.0))
  t.check(np.all(~sometime))
  never = no.isnever(np.full(10, no.never()))
  no.log(never)
  t.check(np.all(never))

  # # DataFrame ops

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

  # df2 = df.copy()
  # df3 = no.append(df,df2)
  # t.check(len(df3) == len(df) + len(df2)) 

  x = np.zeros(1)
  p = no.logistic(x)
  t.check(p == no.logistic(x, 0.0))
  t.check(p == no.logistic(x, 0.0, 1.0))

  t.check(no.logistic(x) == np.ones(1) * 0.5)
  t.check(no.logit(p) == x)
  #t.check(no.isnever(np.ones(1))[0] == False)

  return not t.any_failed
