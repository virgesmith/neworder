""" Example - pricing a simple option
  
The main point of this example is to illustrate how different objects can interact within the model. In this case the objects
are the option itself, and the underlying stock (essentially the market) that governs its price.
"""

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
  "rel_error": { "object": "option", "method": "check", "parameters": [] }
}

# delayed evaluation
get_stock = neworder.Callback("neworder.market")

# initialisation
initialisations = {
  # TODO 
  "xmarket": { "module": "market", "class_": "Market", "parameters": [spot, rate, divy, vol] },
  "option": { "module": "option", "class_": "Option", "parameters": [get_stock, callput, strike, expiry] }
}

transitions = { 
  #"touch_stock": { "object": "2option", "method": "mc", "parameters": [] },
  "compute_mc_price": { "object": "option", "method": "mc", "parameters": [nsims, expiry] }
}

finalisations = {
  # "object": "people" # TODO link to module when multiple
  "compare_mc_price" : { "object": "option", "method": "price", "parameters": [] }
}
