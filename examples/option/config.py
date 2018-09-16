""" Example - pricing a simple option
  
The main point of this example is to illustrate how different objects can interact within the model. In this case the objects
are the option itself, and the underlying stock (essentially the market) that governs its price.
"""

import numpy as np
import neworder

# run 4 times NB contained type is int64
#neworder.sequence = np.array([0,1,2,3])

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
neworder.timespan = np.array([0, expiry])
neworder.timestep = expiry
neworder.nsims = 100000 # number of prices to simulate
neworder.sync_streams = True # all procs use same RNG stream

neworder.log_level = 1
neworder.do_checks = False
# no per-timestep checks implemented since there is only one timestep
neworder.checks = { }

# delayed evaluation for initialisations
get_stock = neworder.lazy_eval("market")
get_option = neworder.lazy_eval("option")

# initialisation
neworder.initialisations = {
  "market": { "module": "market", "class_": "Market", "parameters": [spot, rate, divy, vol] },
  "option": { "module": "option", "class_": "Option", "parameters": [callput, strike, expiry] },
  # TODO import module without creating a class instance?
  "model": { "module": "black_scholes", "class_": "BS", "parameters": [] }
}

neworder.transitions = { 
  # compute the option price
  # To use QRNG (Sobol), set quasi=True
  "compute_mc_price": "pv = model.mc(option, market, nsims, quasi=False)",
  "compute_greeks": "sync() #..."
}

neworder.checkpoints = {
   "compare_mc_price": "model.compare(pv, nsims, option, market)",
}
