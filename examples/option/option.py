""" option.py
Defines and values a European vanilla option using Monte-Carlo
"""

import neworder

# TODO make named tuple?
class Option():
  def __init__(self, callput, strike, expiry):
    self.callput = callput
    self.strike = strike
    self.expiry = expiry
  
  # TODO move somewhere more appropriate (market?)
  def greeks(self, pv):
    neworder.sync()
    if neworder.rank() == 0:
      neworder.log(pv)
      neworder.log("PV=%f" % pv[0])
      neworder.log("delta=%f" % ((pv[1] - pv[2])/200))
      neworder.log("gamma=%f" % ((pv[1] - 2*pv[0] + pv[2])/10000))
      neworder.log("vega 10bp=%f" % (pv[3] - pv[0]))
