""" Example - pricing a simple option
  
The main vanishing point of this example is to illustrate how different objects can interact within the model. In this case the objects
are the option itself, and the underlying stock (essentially the market) that governs its price.
"""

import numpy as np
import neworder

# market data
spot = 100.0 # underlying spot price
rate = 0.02  # risk-free interest rate
divy = 0.01  # (continuous) dividend yield
vol = 0.2    # stock volatility

# (European) option data
callput = "CALL" 
strike = 100.0   
expiry = 0.75   

# variables defined in the module are not accessible by the embedded environment

# Using exact MC calc of GBM requires only 1 timestep 
neworder.timeline = neworder.Timeline(0.0, expiry, [1])

# rust requires nsims in root namespace (or modify transitions/checkpoints)
neworder.nsims = 100000 # number of prices to simulate
# #neworder.sync_streams = True # all procs use same RNG stream

neworder.log_level = 1
neworder.do_checks = False
# # no per-timestep checks implemented since there is only one timestep
neworder.checks = { }

# use 4 identical sims with perturbations
assert neworder.size() == 4 and not neworder.mc.indep(), "This example requires 4 processes with identical RNG streams"

neworder.pv = np.zeros(neworder.size())

# initialisation
neworder.initialisations = {
  "market": { "module": "market", "class_": "Market", "args": (spot, rate, divy, vol) },
  "option": { "module": "option", "class_": "Option", "args": (callput, strike, expiry) },
  # TODO import module without creating a class instance?
  "model": { "module": "black_scholes", "class_": "BS", "args": () } # thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:110:16
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
  # rust 
  "compute_mc_price": "neworder.pv = model.mc(option, market, neworder.nsims, quasi=False)"
}

neworder.checkpoints = {
  "compare_mc_price": "model.compare(neworder.pv, neworder.nsims, option, market)",
  "compute_greeks": "option.greeks(neworder.pv)"
}
