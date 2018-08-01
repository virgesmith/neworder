# Option example - pricing an Option 

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

# Using exact MC calc of GBM requires only 1 timestep 
neworder.timespan = neworder.DVector.fromlist([0, expiry])
neworder.timestep = expiry
nsims = 100000 # number of prices to simulate

loglevel = 1
do_checks = True
checks = { 
  "rel_error": { "module": "option", "method": "check", "parameters": [] }
}
 
# initialisation
initialisations = {
  "option": { "module": "option", "class_": "Option", "parameters": [spot, rate, divy, vol, callput, strike, expiry] }
}

transitions = { 
  "compute_mc_price": { "object": "option", "method": "mc", "parameters": [nsims, expiry] }
}

finalisations = {
  # "object": "people" # TODO link to module when multiple
  "compare_mc_price" : { "object": "option", "method": "price", "parameters": [] }
}
