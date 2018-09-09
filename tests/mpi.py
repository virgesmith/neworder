""" MPI tests """

import numpy as np
import pandas as pd
import neworder

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
  
  if not send_recv(True):
    return False

  if not send_recv(10):
    return  
    
  if not send_recv(10.01):
    return False

  if not send_recv("abcdef"):
    return False

  if not send_recv([1,2,3]):
    return False

  if not send_recv({"a": "fghdfkgh"}):
    return False

  x = np.array([1,4,9,16])
  if neworder.procid == 0:
    neworder.send(x, 1)
  if neworder.procid == 1:
    y = neworder.receive(0)
    neworder.log("MPI: 0 sent {}={} 1 recd {}={}".format(type(x), x, type(y), y))
    if not np.array_equal(x,y):
      return False

  df = pd.read_csv("../../tests/ssm_E09000001_MSOA11_ppp_2011.csv")
  if neworder.procid == 0:
    neworder.log("sending (as csv) df len %d rows from 0" % len(df))
    neworder.send_csv(df, 1)
  if neworder.procid == 1:
    dfrec = neworder.receive_csv(0)
    neworder.log("got (as csv) df len %d rows from 0" % len(dfrec))
    if not dfrec.equals(df):
      neworder.log(df.head())
      neworder.log(dfrec.head())
      return False

  if neworder.procid == 0:
    neworder.log("sending (pickle) df len %d rows from 0" % len(df))
    neworder.send(df, 1)
  if neworder.procid == 1:
    dfrec = neworder.receive(0)
    neworder.log("got (pickle) df len %d rows from 0" % len(dfrec))
    if not dfrec.equals(df):
      neworder.log(df.head())
      neworder.log(dfrec.head())
      return False

  return True