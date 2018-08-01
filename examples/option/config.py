""" Example - pricing a simple option
  
Monte-Carlo simulation is a common technique in quantitative finance. 
For more info see e.g. https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance

A European call option is a derivative contract that grants the holder the right (but not the obilgation) 
to buy an underlying stock S at a fixed "strike" price K at some given future time T (the expiry). Similarly,
a put option grants the right (but not obligation) to sell, rather than buy, at a fixed price.
See https://en.wikipedia.org/wiki/Call_option

Framing derivative pricing in terms of a microsimulation model:
- start with an intiial (t=0) population of N (identical) underlying prices
- evolve each price using Monte-Carlo simulation of the stochastic differential equation (SDE)
     dS/S = (r-q)dt + vdW
  where S is price, r is risk-free rate, q is continuous dividend yield, v is volatility and dW a Wiener process
- at expiry (t=T) compute the option prices for each underlying and take the mean
- discount the option price back to valuation date (t=0)

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
get_stock = neworder.Callback("neworder.stock")

# initialisation
initialisations = {
  "stock": { "module": "option", "class_": "Stock", "parameters": [spot, rate, divy, vol] },
  "option": { "module": "option", "class_": "Option", "parameters": [get_stock, callput, strike, expiry] }
}

transitions = { 
  #"touch_stock": { "object": "2option", "method": "mc", "parameters": [] },
  "compute_mc_price": { "object": "option", "method": "mc", "parameters": [nsims, expiry] }
}

finalisations = {
  # "object": "people" # TODO link to module when multiple
  "compare_mc_price" : { "object": "2option", "method": "price", "parameters": [] }
}
