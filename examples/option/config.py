""" Example - pricing a simple option
  
The main point of this example is to illustrate how different objects can interact within the model. In this case the objects
are the option itself, and the underlying stock (essentially the market) that governs its price.
"""

import numpy as np
import neworder

# run 4 times
neworder.sequence = neworder.IVector.fromlist([0,1,2,3])

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
neworder.timespan = neworder.DVector.fromlist([0, expiry])
neworder.timestep = expiry
neworder.nsims = 100000 # number of prices to simulate

loglevel = 1
do_checks = False
# no per-timestep checks implemented since there is only one timestep
checks = { }

# delayed evaluation for initialisations
get_stock = neworder.lazy_eval("market")

# initialisation
initialisations = {
  "market": { "module": "market", "class_": "Market", "parameters": [spot, rate, divy, vol] },
  "option": { "module": "option", "class_": "Option", "parameters": [get_stock, callput, strike, expiry] }
}

transitions = { 
  # compute the option price
  #"compute_mc_price": "option.mc(nsims)"
  # use QRNG
  "compute_mc_price": "option.mc(nsims, quasi=True)" 
}

checkpoints = {
   "compare_mc_price": "option.check()"
}
