""" market.py
Defines a simple market with a single underlying
"""

import neworder

class Market():
  """ This is just a simple container for market data """
  def __init__(self, spot, rate, divy, vol):
    self.spot = spot
    self.rate = rate
    self.divy = divy
    self.vol = vol

