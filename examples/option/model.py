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
assert neworder.mpi.size() == 4 and not neworder.mpi.indep(), "This example requires 4 processes with identical RNG streams"

# rust requires nsims in root namespace (or modify transitions/checkpoints)
nsims = 100000 # number of prices to simulate

#neworder.log_level = 1

# create an array for the results from each model run
neworder.pv = np.zeros(neworder.mpi.size())

# initialisation
market = Market(spot, rate, divy, vol)
option = Option(callput, strike, expiry)

neworder.model = BlackScholes(option, market, nsims)

