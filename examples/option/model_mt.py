"""
Example - pricing a simple option

The main vanishing point of this example is to illustrate how work can be split across multiple
tasks, and how to synchronise the random streams between them

This version implements the model using multithreading rather than MPI
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

from black_scholes_mt import BlackScholesMT, greeks

import neworder

# neworder.verbose()  # uncomment for verbose logging
# neworder.checked(False) # uncomment to disable checks

# requires 4 identical sims with perturbations to compute market sensitivities
# (a.k.a. Greeks)
assert neworder.mpi.SIZE == 1, "This example should not be run via MPI, it uses multiple threads in a single process"

# initialisation

# market data
market = {
    "spot": 100.0,  # underlying spot price
    "rate": 0.02,  # risk-free interest rate
    "divy": 0.01,  # (continuous) dividend yield
    "vol": 0.2,  # stock volatility
}
# (European) option instrument data
option = {
    "callput": "CALL",
    "strike": 100.0,
    "expiry": 0.75,  # years
}

# model parameters
nsims = 10000000  # number of underlyings to simulate


def run_thread(index: int) -> BlackScholesMT:
    model = BlackScholesMT(option, market, nsims, index)
    neworder.run(model)
    return model


print(f"neworder FT: {neworder.freethreaded()}, python FT:{not sys._is_gil_enabled()}", sys.version)

# instantiate models
t = time.perf_counter()
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_thread, i) for i in range(4)]
    results = [f.result() for f in futures]
    greeks(results)
print(time.perf_counter() - t)

# # sequential execution
# t = time.perf_counter()
# results = [run_thread(i) for i in range(4)]
# greeks(results)
# print(time.perf_counter() - t)
