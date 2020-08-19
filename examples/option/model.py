""" Example - pricing a simple option

The main vanishing point of this example is to illustrate how different processes can interact within the model.

"""

import neworder

from option import Option
from market import Market
from black_scholes import BlackScholes

neworder.module_init(independent=False, verbose=False)

# requires 4 identical sims with perturbations to compute market sensitivities (a.k.a. Greeks)
assert neworder.mpi.size() == 4 and not neworder.mpi.indep(), "This example requires 4 processes with identical RNG streams"

# market data
spot = 100.0 # underlying spot price
rate = 0.02  # risk-free interest rate
divy = 0.01  # (continuous) dividend yield
vol = 0.2    # stock volatility

# (European) option instrument data
callput = "CALL"
strike = 100.0
expiry = 0.75


# rust requires nsims in root namespace (or modify transitions/checkpoints)
nsims = 100000 # number of prices to simulate

# initialisation
market = Market(spot, rate, divy, vol)
option = Option(callput, strike, expiry)

bs_mc = BlackScholes(option, market, nsims)

neworder.run(bs_mc)
