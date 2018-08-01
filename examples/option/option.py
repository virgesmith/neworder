""" option.py
Prices a European vanilla option using Monte-Carlo
"""

from math import *
import numpy as np

import neworder

def _norm_cdf(x):
  return (1.0 + erf(x / sqrt(2.0))) / 2.0

class Stock():
  def __init__(self, spot, rate, divy, vol):
    self.spot = spot
    self.rate = rate
    self.divy = divy
    self.vol = vol

    # persist by inserting into neworder
    neworder.stock = self

class Option():
  def __init__(self, stock, callput, strike, expiry):

    self.stock = stock()
    self.callput = callput
    self.strike = strike
    self.expiry = expiry

    self.pv = 0.0
    self.nsims = 0

  def mc(self, nsims, dt):
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

  # implement analytic Black-Scholes pricing as a check... 
  def bs(self):
    srt = self.stock.vol * sqrt(self.expiry)
    rqs2t = (self.stock.rate - self.stock.divy + 0.5 * self.stock.vol * self.stock.vol) * self.expiry
    d1 = (log(self.stock.spot/self.strike) + rqs2t) / srt
    d2 = d1 - srt
    df = exp(-self.stock.rate * self.expiry)
    qf = exp(-self.stock.divy * self.expiry)

    if self.callput == "CALL":
      return self.stock.spot * qf * _norm_cdf(d1) - self.strike * df * _norm_cdf(d2)
    else:
      return -self.stock.spot * df * _norm_cdf(-d1) + self.strike * df * _norm_cdf(-d2)
  
  def price(self):
    return self.pv

  def check(self):
    ref = self.bs()
    err = self.pv / ref - 1.0
    neworder.log("mc: {:.6f} / ref: {:.6f} err={:.2%}".format(self.pv, ref, err))
    # relative error should be within ~1/(sqrt(sims)) of analytic solution
    return True if abs(err) <= 2.0/sqrt(self.nsims) else False