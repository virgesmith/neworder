""" Example - pricing a simple option
  
The main vanishing point of this example is to illustrate how different objects can interact within the model. In this case the objects
are the option itself, and the underlying stock (essentially the market) that governs its price.
"""

import numpy as np
import neworder

from option import Option
from market import Market
from black_scholes import BlackScholes 

# market data
spot = 100.0 # underlying spot price
rate = 0.02  # risk-free interest rate
divy = 0.01  # (continuous) dividend yield
vol = 0.2    # stock volatility

# (European) option instrument data
callput = "CALL" 
strike = 100.0   
expiry = 0.75   

# use 4 identical sims with perturbations to compute market sensitivities (a.k.a. Greeks)
assert neworder.size() == 4 and not neworder.mc.indep(), "This example requires 4 processes with identical RNG streams"

# Using exact MC calc of GBM requires only 1 timestep 
timeline = neworder.Timeline(0.0, expiry, [1])

# rust requires nsims in root namespace (or modify transitions/checkpoints)
nsims = 100000 # number of prices to simulate

#neworder.log_level = 1

# create an array for the results from each model run 
neworder.pv = np.zeros(neworder.size())

# initialisation
market = Market(spot, rate, divy, vol)
option = Option(callput, strike, expiry)

# process-specific modifiers (for sensitivities)
modifiers = [
  "pass", # base valuation
  "neworder.shell(); market.spot = market.spot * 1.01", # delta/gamma up bump
  "market.spot = market.spot * 0.99", # delta/gamma down bump
  "market.vol = market.vol + 0.001" # 10bp upward vega
]

transitions = { 
  # compute the option price using a Monte-Carlo simulation
  "compute_mc_price": "neworder.pv = neworder.model.mc(option, market)"
}

checkpoints = {
  # compare the MC price to the analytic solution
  "compare_mc_price": "neworder.model.compare(neworder.pv, option, market)",
  # compute some market risk
  "compute_greeks": "option.greeks(neworder.pv)"
}

neworder.model = BlackScholes(timeline, modifiers, transitions, {}, checkpoints, nsims)

neworder.log(dir())