"""
Example - pricing a simple option

The main vanishing point of this example is to illustrate how different processes
can interact within the model, and how to synchronise the random streams in each process
"""

from black_scholes import BlackScholes
from helpers import Market, Option

import neworder

"""model inputs"""

# market data
market = Market(spot=100.0, rate=0.02, divy=0.01, vol=0.2)

# (European) option instrument data
option = Option(callput="CALL", strike=100.0, expiry=0.75)

# model parameters
nsims = 1000000  # number of underlyings to simulate

# neworder.verbose() # uncomment for verbose logging
# neworder.checked(False) # uncomment to disable checks

# requires 4 identical sims with perturbations to compute market sensitivities
# (a.k.a. Greeks)
assert neworder.mpi.SIZE == 4, "This example requires 4 processes"

# instantiate model
bs_mc = BlackScholes(option, market, nsims)

# run model
neworder.run(bs_mc)
