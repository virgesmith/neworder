

# Derivative Pricing

## Background

Monte-Carlo simulation is a [common technique in quantitative finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance).

A [European call option](https://en.wikipedia.org/wiki/Call_option) is a derivative contract that grants the holder the right (but not the obligation) to buy an underlying stock S at a fixed "strike" price K at some given future time T (the expiry). Similarly, a put option grants the right (but not obligation) to sell, rather than buy, at a fixed price.

In order to calculate the fair value of a derivative contract one can simulate a (large) number of paths the underlying stock may take (according to current market conditions and some model assumptions). We then take the mean of the derivative price for
each simulated path to get the value of the derivative _at expiry_. Finally this price is discounted to get the current fair value.

We can easily frame a derivative derivative pricing problem in terms of a microsimulation model:

- start with an intial (t=0) population of N (identical) underlying prices. Social scientists could refer to this as a 'cohort'.
- evolve each price to option expiry time (t=T) using Monte-Carlo simulation of the stochastic differential equation (SDE):

  dS/S = (r-q)dt + &sigma dW

  where S is price, r is risk-free rate, q is continuous dividend yield, v is volatility and dW a Wiener process (a 1-d Brownian motion).
- compute the option prices for each of the underlyings and take the mean
- discount the option price back to valuation date (t=0)

For this simple option we can also compute an analytic fair value under the Black-Scholes model, and use this to determine the accuracy of the Monte-Carlo simulation. We also demonstrate the capabilities neworder has in terms of sensitivity analysis.

## Implementation

We use an implementation of the Monte-Carlo technique described above, and also, for comparision, the analytic solution.

Additionally, we compute some market risk: sensitivities to the underlying price and volatility. In order to do this we need to run the simulation multiple times with perturbations to market data. To eliminate random noise we also want to use identical random streams in each simulation. The model is run over 4 processes in the MPI framework to achieve this.

The `model.py` file sets up the run, providing input data, constructing, and the running the model. The input data consists of a `Dict` describing the market data, another describing the option contract, and a single model parameter (the number of paths).

```python
import neworder
from black_scholes import BlackScholes

# neworder.verbose() # defaults to False
# neworder.checked() # defaults to True

# requires 4 identical sims with perturbations to compute market sensitivities (a.k.a. Greeks)
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
```

The file [black_scholes.py](../../examples/option/black_scholes.py) contains the model implementation (subclassing `neworder.Model`), with both analytic option formula and the Monte-Carlo simulation, with [helpers.py](../../examples/option/helpers.py) providing some additional functionality.

#### Constructor

The constructor takes copies of the parameters, and defines a simple timeline [0, T] corresponding to [valuation date, expiry date] and a single timestep, which is all we require for this example. It initialises the base class with the timeline, and specifies that each process use the same random stream (which reduces noise in our risk calculations):

```python
class BlackScholes(neworder.Model):
  def __init__(self, option, market, nsims):

    # Using exact MC calc of GBM requires only 1 timestep
    timeline = neworder.Timeline(0.0, option["expiry"], [1])
    super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_stream)

    self.option = option
    self.market = market
    self.nsims = nsims
```

#### Modifier

This method defines the 'modifiers' for each process: the perturbations applied to the market data in each process in order to calculate the option price sensitivity to that market data. In this case we bump the spot up and down and the volatility up in the non-root processes allowing, calculation of delta, gamma and vega by finite differencing:

```python
  def modify(self, rank):
    if rank == 1:
      self.market["spot"] *= 1.01 # delta/gamma up bump
    elif rank == 2:
      self.market["spot"] *= 0.99 # delta/gamma down bump
    elif rank == 3:
      self.market["vol"] += 0.001 # 10bp upward vega
```

#### Step

This method actually runs the simulation and stores the result for later use. The calculation details are not shown here for brevity (see the source file):

```python
  def step(self):
    self.pv = self.simulate()
```

#### Check

Even though we explicitly requested that each process has identical random streams, this does not guarantee the streams will stay identical, as different process could sample more or less than others, and the streams get out of step.

This method samples one uniform from each stream and will return `False` if any of them are different, which will halt the model (for that process).

!!! danger "Deadlocks"
    The implementation needs to be careful here is if some processes stop and others continue, a deadlock can occur when a process tries to communicate with a process that has ended. The check method must therefore ensure that ALL processes either pass or fail.

In the below implementation, all samples are sent to a single process (0) for comparison and the result is broadcast back to every process, which can then all fail simultaneously if necessary.

```python
  def check(self):
    # check the rng streams are still in sync by sampling from each one, comparing, and broadcasting the result
    # if one process fails the check and exits without notifying the others, deadlocks can result
    r = self.mc().ustream(1)[0]
    # send the value to process 0)
    a = comm.gather(r, 0)
    # process 0 checks the values
    if neworder.mpi.rank() == 0:
      ok = all(e == a[0] for e in a)
    else:
      ok = True
    # broadcast process 0's ok to all processes
    ok = comm.bcast(ok, root=0)
    return ok
```

#### Checkpoint

Finally the checkpoint method is called at end of the timeline. Again, the calculation detail is omitted for clarity, but the method performs two tasks:

- checks the Monte-Carlo result against the analytic formula and displays the simulated price and the random error, for each process.
- computes the sensitivities: process 0 gathers the results from the other processes and computes the finite-difference formulae.

```python
  def checkpoint(self):
    # check and report accuracy
    self.compare()
    # compute and report some market risk
    self.greeks()
```

## Execution

By default, the model has verbose mode off and checked mode on. These settings can be changed in [model.py]()

To run the model,

```bash
mpiexec -n 4 python examples/option/model.py
```

which will produce something like

```text
[py 0/4] check() ok: True
[py 0/4] mc: 7.182313 / ref: 7.201286 err=-0.26%
[py 2/4] mc: 6.646473 / ref: 6.665127 err=-0.28%
[py 1/4] mc: 7.740759 / ref: 7.760108 err=-0.25%
[py 3/4] mc: 7.216204 / ref: 7.235288 err=-0.26%
[py 0/4] PV=7.182313
[py 0/4] delta=0.547143
[py 0/4] gamma=0.022606
[py 0/4] vega 10bp=0.033892
```

Note that the order of the output will vary, and log messages may even get intermingled.
