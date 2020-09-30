""" Example - pricing a simple option

The main vanishing point of this example is to illustrate how different processes 
can interact within the model.

"""

import neworder
from black_scholes import BlackScholes

# neworder.verbose() # defaults to False
# neworder.checked() # defaults to True

# requires 4 identical sims with perturbations to compute market sensitivities 
# (a.k.a. Greeks)
assert neworder.mpi.size() == 4, "This example requires 4 processes"

# initialisation

# market data
market = {
  "spot": 100.0, # underlying spot price
  "rate": 0.02,  # risk-free interest rate
  "divy": 0.01,  # (continuous) dividend yield
  "vol": 0.2    # stock volatility
}
# (European) option instrument data
option = {
  "callput": "CALL",
  "strike": 100.0,
  "expiry": 0.75 # years
}

# model parameters
nsims = 1000000 # number of underlyings to simulate

# instantiate model
bs_mc = BlackScholes(option, market, nsims)

# run model
neworder.run(bs_mc)
