""" MPI tests """

import numpy as np
import pandas as pd
import neworder

import test as t

def send_recv(x):
  if neworder.procid == 0:
    neworder.send(x, 1)
  if neworder.procid == 1:
    y = neworder.receive(0)
    neworder.log("MPI: 0 sent {}={} 1 recd {}={}".format(type(x), x, type(y), y))
    if y != x:
     return False
  return True

def test():
  if neworder.nprocs == 1:
    neworder.log("Skipping MPI tests")
    return True
  
  t.check(send_recv(True))
  t.check(send_recv(10))
  t.check(send_recv(10.01))
  t.check(send_recv("abcdef"))
  t.check(send_recv([1,2,3]))
  t.check(send_recv({"a": "fghdfkgh"}))

  x = np.array([1,4,9,16])
  if neworder.procid == 0:
    neworder.send(x, 1)
  if neworder.procid == 1:
    y = neworder.receive(0)
    neworder.log("MPI: 0 sent {}={} 1 recd {}={}".format(type(x), x, type(y), y))
    t.check(np.array_equal(x,y))

  df = pd.read_csv("../../tests/ssm_E09000001_MSOA11_ppp_2011.csv")
  if neworder.procid == 0:
    neworder.log("sending (as csv) df len %d rows from 0" % len(df))
    neworder.send_csv(df, 1)
  if neworder.procid == 1:
    dfrec = neworder.receive_csv(0)
    neworder.log("got (as csv) df len %d rows from 0" % len(dfrec))
    t.check(dfrec.equals(df))

  if neworder.procid == 0:
    neworder.log("sending (pickle) df len %d rows from 0" % len(df))
    neworder.send(df, 1)
  if neworder.procid == 1:
    dfrec = neworder.receive(0)
    neworder.log("got (pickle) df len %d rows from 0" % len(dfrec))
    t.check(dfrec.equals(df))

  return t.any_failed