""" Example - pricing a simple option

The main vanishing point of this example is to illustrate how different objects can interact within the model. In this case the objects
are the option itself, and the underlying stock (essentially the market) that governs its price.
"""

import neworder

from option import Option
from market import Market
from black_scholes import BlackScholes

if not neworder.embedded():
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  neworder.module_init(comm.Get_rank(), comm.Get_size(), False, True)


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

# initialisation
market = Market(spot, rate, divy, vol)
option = Option(callput, strike, expiry)

bs_mc = BlackScholes(option, market, nsims)

neworder.run(bs_mc)
