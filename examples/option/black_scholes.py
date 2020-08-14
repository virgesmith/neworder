""" Black-Scholes model implementations: analytic and MC """

import neworder
from math import exp, log
from helpers import *

# Subclass neworder.Model
class BlackScholes(neworder.Model):
  def __init__(self, option, market, nsims=100000):

    # Using exact MC calc of GBM requires only 1 timestep
    timeline = neworder.Timeline(0.0, option.expiry, [1])
    super().__init__(timeline)

    self.option = option
    self.market = market
    self.nsims = nsims

  def modify(self, rank):
    # hmmm....
    if rank == 1:
      self.market.spot = self.market.spot * 1.01 # delta/gamma up bump
    elif rank == 2:
      self.market.spot = self.market.spot * 0.99 # delta/gamma down bump
    elif rank == 3:
      self.market.vol = self.market.vol + 0.001 # 10bp upward vega

  def transition(self):
    self.pv = self.simulate()

  def checkpoint(self):
    self.compare()
    # compute some market risk
    self.greeks()

  def simulate(self):
    # get the time from the environment
    dt = self.timeline().time()
    normals = nstream(self.mc().ustream(self.nsims))
    # compute underlying prices at dt
    S = self.market.spot
    r = self.market.rate
    q = self.market.divy
    sigma = self.market.vol
    underlyings = S * np.exp((r - q - 0.5 * sigma * sigma) * dt + normals * sigma * sqrt(dt))
    # compute option prices at dt
    if self.option.callput == "CALL":
      fv = (underlyings - self.option.strike).clip(min=0.0).mean()
    else:
      fv = (self.option.strike - underlyings).clip(min=0.0).mean()

    # discount back to val date
    return fv * exp(-r * dt)

  def analytic(self):
    """ Compute Black-Scholes European option price """
    S = self.market.spot
    K = self.option.strike
    r = self.market.rate
    q = self.market.divy
    T = self.option.expiry
    vol = self.market.vol

    srt = vol * sqrt(T)
    rqs2t = (r - q + 0.5 * vol * vol) * T
    d1 = (log(S/K) + rqs2t) / srt
    d2 = d1 - srt
    df = exp(-r * T)
    qf = exp(-q * T)

    if self.option.callput == "CALL":
      return S * qf * norm_cdf(d1) - K * df * norm_cdf(d2)
    else:
      return -S * df * norm_cdf(-d1) + K * df * norm_cdf(-d2)

  def compare(self):
    """ Compare MC price to analytic """
    ref = self.analytic()
    err = self.pv / ref - 1.0
    neworder.log("mc: {:.6f} / ref: {:.6f} err={:.2%}".format(self.pv, ref, err))
    # relative error should be within O(1/(sqrt(sims))) of analytic solution
    return True if abs(err) <= 2.0/sqrt(self.nsims) else False

  def greeks(self):
    # get all the results
    pvs = neworder.mpi.gather(self.pv, 0)
    # compute sensitivities on rank 0
    if neworder.mpi.rank() == 0:
      neworder.log("gathered results: %s" % pvs)
      neworder.log("PV=%f" % pvs[0])
      neworder.log("delta=%f" % ((pvs[1] - pvs[2])/2))
      neworder.log("gamma=%f" % ((pvs[1] - 2*pvs[0] + pvs[2])))
      neworder.log("vega 10bp=%f" % (pvs[3] - pvs[0]))
