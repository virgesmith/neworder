""" Black-Scholes model implementations: analytic and MC """

import neworder
from helpers import *
from math import *

# Subclass neworder.Model
class BlackScholes(neworder.Model):
  def __init__(self, timeline, modifiers, transitions, checks, checkpoints, nsims=100000):
    super().__init__(timeline, modifiers, transitions, checks, checkpoints)
    self.nsims = nsims

  def mc(self, option, market):
    # get the time from the environment
    dt = self.timeline().time()
    normals = nstream(self.nsims)
    # compute underlying prices at dt
    underlyings = market.spot * np.exp((market.rate - market.divy - 0.5 * market.vol * market.vol) * dt + normals * market.vol * sqrt(dt))
    # compute option prices at dt
    if option.callput == "CALL":
      fv = (underlyings - option.strike).clip(min=0.0).mean()
    else:
      fv = (option.strike - underlyings).clip(min=0.0).mean()

    # discount back to val date
    return fv * exp(-market.rate * dt)

  def analytic(self, option, market):
    """ Compute Black-Scholes European option price """
    S = market.spot
    K = option.strike
    r = market.rate
    q = market.divy
    T = option.expiry
    vol = market.vol

    srt = vol * sqrt(T)
    rqs2t = (r - q + 0.5 * vol * vol) * T
    d1 = (log(S/K) + rqs2t) / srt
    d2 = d1 - srt
    df = exp(-r * T)
    qf = exp(-q * T)

    if option.callput == "CALL":
      return S * qf * norm_cdf(d1) - K * df * norm_cdf(d2)
    else:
      return -S * df * norm_cdf(-d1) + K * df * norm_cdf(-d2)

  def compare(self, pv_mc, option, market):
    """ Compare MC price to analytic """
    ref = self.analytic(option, market)
    err = pv_mc / ref - 1.0
    neworder.log("mc: {:.6f} / ref: {:.6f} err={:.2%}".format(pv_mc, ref, err))
    # relative error should be within O(1/(sqrt(sims))) of analytic solution
    return True if abs(err) <= 2.0/sqrt(self.nsims) else False
