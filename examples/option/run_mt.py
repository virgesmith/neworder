"""
Example - pricing a simple option

The main vanishing point of this example is to illustrate how work can be split across multiple
tasks, and how to synchronise the random streams between them

This version implements the model using multithreading rather than MPI
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from black_scholes_mt import BlackScholesMT, greeks
from helpers import Market, Option

import neworder

# neworder.verbose()  # uncomment for verbose logging
# neworder.checked(False) # uncomment to disable checks

# requires 4 identical sims with perturbations to compute market sensitivities
# (a.k.a. Greeks)
assert neworder.mpi.SIZE == 1, "This example should not be run via MPI, it uses multiple threads in a single process"

# market data
market = Market(spot=100.0, rate=0.02, divy=0.01, vol=0.2)

# (European) option instrument data
option = Option(callput="CALL", strike=100.0, expiry=0.75)

# model parameters
nsims = 1000000  # number of underlyings to simulate


def run_thread(index: int) -> BlackScholesMT:
    model = BlackScholesMT(option, market, nsims, index)
    neworder.run(model)
    return model


print(f"neworder FT: {neworder.freethreaded()}, python FT:{not sys._is_gil_enabled()}", sys.version)

# instantiate and run threads
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_thread, i) for i in range(4)]
    results = [f.result() for f in futures]
    # unlike MPI is is difficult for model threads to compare their state with each other, so the check is deferred to
    # the main thread
    assert len({r.mc.state() for r in results}), "Thread RNGs have diverged"
    greeks(results)
