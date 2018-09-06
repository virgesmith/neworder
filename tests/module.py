"""
Test framework expects modules with a function called test that
- takes no arguments
- returns True on success and False on failure
"""
import neworder as no
import numpy as np
import pandas as pd

def test():

  s = no.ustream(10000)
  if not isinstance(s, np.ndarray):
    return False
  if not len(s) == 10000:
    return False
  if not abs(np.mean(s) - 0.5) < 0.01:
    return False

  f = no.lazy_eval("2 + 2")
  if not f() == 4: 
    return False

  # TODO this overlaps/duplicates tests in op.py - reorganise

  # test thinning algorithm for non-homogeneous Poisson process
  h = np.array([0.014] * 10)
  #l = no.stopping_v(h)
  l = no.stopping_nhpp(h, 10000)
  if not abs(np.mean(l) * 0.014 - 1.0) < 0.01:
    return False

  # test a certain(ish) hazard rate
  h = np.array([0.99, 0.99, 0.01])
  l = no.stopping_nhpp(h, 10000)
  no.log("TODO NHPP appears broken: %f" % np.mean(l))

  # test a zero(ish) hazard rate
  h = np.array([1e-30, 1e-30, 1e-30, .9999])
  l = no.stopping_nhpp(h, 10000)
  no.log("TODO NHPP appears broken: %f" % np.mean(l))

  # this also tests a zero hazard rate 
  h = np.array([i/3000 for i in range(100)])
  #no.log(h)
  le = no.stopping_nhpp(h, 10000)
  no.log(sum(le)/len(le))

  # not convinced this is working correctly
  h = np.array([0.999, 0.1])
  le = no.stopping_nhpp(h, 1000)
  no.log(sum(le)/len(le))

  # pass df
  df = pd.read_csv("../../tests/df.csv")
  no.transition(df["DC2101EW_C_ETHPUK11"].values)
  no.log(df.head())
  no.directmod(df, "DC2101EW_C_ETHPUK11")
  no.log(df.head())

  no.log(len(df))
  df2 = df.copy()
  df3 = no.append(df,df2)
  no.log(len(df3))
  no.log(df3.index)

  # 6.3MB file
  import pickle
  bigdf = pd.read_csv("../../examples/people/ssm_E08000021_MSOA11_ppp_2011.csv")
  no.log("data {} len={}".format(type(bigdf), len(bigdf))) # rows in DF
  pickled = pickle.dumps(bigdf)
  no.log("pickled {} len={}".format(type(pickled), len(pickled))) # 9.5MB binary serialised

  unpickled = pickle.loads(pickled)
  no.log("unpickled {} len={}".format(type(unpickled), len(unpickled))) # rows in DF

  # from io import StringIO
  # buf = StringIO()
  # bigdf.to_csv(buf, index=False)
  # csvbuf = buf.getvalue()
  # no.log("pickled {} len={}".format(type(csvbuf), len(csvbuf))) # 6.3MB csv

  return True
