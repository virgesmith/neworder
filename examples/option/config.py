""" Example - pricing a simple option
  
The main point of this example is to illustrate how different objects can interact within the model. In this case the objects
are the option itself, and the underlying stock (essentially the market) that governs its price.
"""

import numpy as np
import neworder

# run 4 times NB contained type is int64
#neworder.sequence = np.array([10,11,12,13])

# market data
spot = 100.0 # underlying spot price
rate = 0.02  # risk-free interest rate
divy = 0.01  # (continuous) dividend yield
vol = 0.2    # stock volatility

# (European) option data
callput = "CALL" 
strike = 100.0   
expiry = 0.75   

# Using exact MC calc of GBM requires only 1 timestep 
neworder.timeline = (0, expiry, 1)

neworder.nsims = 100000 # number of prices to simulate
neworder.sync_streams = True # all procs use same RNG stream

neworder.log_level = 1
neworder.do_checks = False
# no per-timestep checks implemented since there is only one timestep
neworder.checks = { }

# use 4 identical sims with perturbations
assert neworder.size() == 4 and not neworder.indep(), "This example requires 4 processes with identical RNG streams"

neworder.pv = np.zeros(neworder.size())

# initialisation
neworder.initialisations = {
  "market": { "module": "market", "class_": "Market", "parameters": [spot, rate, divy, vol] },
  "option": { "module": "option", "class_": "Option", "parameters": [callput, strike, expiry] },
  # TODO import module without creating a class instance?
  "model": { "module": "black_scholes", "class_": "BS", "parameters": [] }
}

# process-specific modifiers (for sensitivities)
neworder.modifiers = [
  "pass", # base valuation
  "market.spot = market.spot * 1.01", # delta up bump
  "market.spot = market.spot * 0.99", # delta up bump
  "market.vol = market.vol + 0.001" # 10bp upward vega
]

neworder.transitions = { 
  # compute the option price
  # To use QRNG (Sobol), set quasi=True
  "compute_mc_price": "pv = model.mc(option, market, nsims, quasi=False)"
}

neworder.checkpoints = {
  "compare_mc_price": "model.compare(pv, nsims, option, market)",
  "compute_greeks": "option.greeks(pv)"
}
