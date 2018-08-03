""" market.py
Defines a simple market with a single underlying
"""

import neworder

class Market():
  def __init__(self, spot, rate, divy, vol):
    self.spot = spot
    self.rate = rate
    self.divy = divy
    self.vol = vol

    # persist by inserting into neworder
    # TODO can this be more automatic?
    neworder.market = self

