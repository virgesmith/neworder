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
    pvs = neworder.gather(pv, 0)
    if neworder.rank() == 0:
      neworder.log("PV=%f" % pvs[0])
      neworder.log("delta=%f" % ((pvs[1] - pvs[2])/2))
      neworder.log("gamma=%f" % ((pvs[1] - 2*pvs[0] + pvs[2])))
      neworder.log("vega 10bp=%f" % (pvs[3] - pvs[0]))
