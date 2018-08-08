""" option.py
Defines and values a European vanilla option using Monte-Carlo
"""

from helpers import *
from math import *
import numpy as np

import neworder

class Option():
  def __init__(self, stock, callput, strike, expiry):

    self.stock = stock()
    self.callput = callput
    self.strike = strike
    self.expiry = expiry

    self.pv = 0.0
    self.nsims = 0

  def mc(self, nsims):
    # get the time from the environment
    dt = neworder.time
    # compute underlying prices at dt
    underlyings = self.stock.spot * np.exp((self.stock.rate - self.stock.divy - 0.5 * self.stock.vol * self.stock.vol) * dt + np.random.normal(size=nsims) * self.stock.vol * sqrt(dt))
    # compute option prices at dt
    if self.callput == "CALL":
      option = (underlyings - self.strike).clip(min=0.0).mean()
    else:
      option = (self.strike - underlyings).clip(min=0.0).mean()

    # discount back to val date
    self.pv = option * exp(-self.stock.rate * dt)
    self.nsims = nsims
  
  def price(self):
    return self.pv

  def check(self):
    ref = bs_euro_option(self.stock.spot, self.strike, self.stock.rate, self.stock.divy, self.expiry, self.stock.vol, self.callput)
    err = self.pv / ref - 1.0
    neworder.log("mc: {:.6f} / ref: {:.6f} err={:.2%}".format(self.pv, ref, err))
    # relative error should be within O(1/(sqrt(sims))) of analytic solution
    return True if abs(err) <= 2.0/sqrt(self.nsims) else False