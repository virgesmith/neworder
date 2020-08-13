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

