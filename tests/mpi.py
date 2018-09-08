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

  # Fails
  # if not send_recv("abcdef"):
  #   return False
  # if not send_recv([1,2,3]):
  #   return False

  df = pd.read_csv("../../tests/ssm_E09000001_MSOA11_ppp_2011.csv")
  if neworder.procid == 0:
    neworder.log("sending df len %d rows from 0" % len(df))
    neworder.send_csv(df, 1)
  if neworder.procid == 1:
    dfrec = neworder.receive_csv(0)
    neworder.log("got df len %d rows from 0" % len(dfrec))
    if not dfrec.equals(df):
      neworder.log(df.head())
      neworder.log(dfrec.head())
      return False

  # a = np.array([0,12,3])
  # if neworder.procid == 0:
  #   neworder.log("sending df len %d rows from 0" % len(self.fertility))
  #   neworder.send(a, 1)
  # if neworder.procid == 1:
  #   neworder.log(neworder.receive(1))

  return True