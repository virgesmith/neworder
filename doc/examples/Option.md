# Derivative Pricing

## Background

Monte-Carlo simulation is a [common technique in quantitative finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance).

A [European call option](https://en.wikipedia.org/wiki/Call_option) is a derivative contract that grants the holder the right (but not the obligation)
to buy an underlying stock S at a fixed "strike" price K at some given future time T (the expiry). Similarly,
a put option grants the right (but not obligation) to sell, rather than buy, at a fixed price.

In order to calculate the fair value of a derivative contract one can simulate a (large) number of paths the underlying stock may take
(according to current market conditions and some model assumptions). We then take the mean of the derivative price for
each simulated path to get the value of the derivative _at expiry_. Finally this price is discounted to get the current fair value.

We can easily frame a derivative derivative pricing problem in terms of a microsimulation model:
- start with an intial (t=0) population of N (identical) underlying prices. Social scientists could refer to this as a 'cohort'.
- evolve each price to option expiry time (t=T) using Monte-Carlo simulation of the stochastic differential equation (SDE):

  dS/S = (r-q)dt + vdW

  where S is price, r is risk-free rate, q is continuous dividend yield, v is volatility and dW a Wiener process (a 1-d Brownian motion).
- compute the option prices for each of the underlyings and take the mean
- discount the option price back to valuation date (t=0)

For this simple option we can also compute an analytic fair value under the Black-Scholes model, and use this to determine the accuracy of the Monte-Carlo simulation. We also demonstrate the capabilities neworder has in terms of sensitivity analysis.

## Implementation

We use an implementation of the Monte-Carlo technique described above, and also, for comparision, the analytic solution.

Additionally, we compute some market risk: sensitivities to the underlying price and volatility. In order to do this we need to run the simulation multiple times with perturbations to market data. To eliminate random noise we also want to use identical random streams in each simulation.

We run the model over 4 processes in the MPI framework to achieve this:

```
$ ./run_example.sh option 4 -c
```
where the `-c` flag ensures the random streams are identical.

The [config.py](../../examples/option/config.py) file:
- sets the parameters for the market and the option, and describes how to initialise the [market](../../examples/option/market.py) and [option](../../examples/option/option.py) objects with these parameters.

- defines a simple timeline [0, T] corresponding to [valuation date, expiry date] and a single timestep, which is al we require for this example.

- describes the 'modifiers' for each process: the perturbations applied to the market data in order to calculate the option price sensitivity to that market data. In this case we bump the spot up and down and the volatility up, allowing calculation of delta, gamma and vega.

- defines the "transition", which in this case is simply running the Monte-Carlo simulation in one step from time zero to time T.

- and finally the "checkpoints" run at the end of the timeline:
  - check the Monte-Carlo result against the analytic formula and displays the price and the random error.
  - process 0 gathers the results from the other processes and computes the sensitivities described above.

The file [black_scholes.py](../../examples/option/black_scholes.py) implements the Model (by subclassing `neworder.Model`), with both analytic option formula and the Monte-Carlo simulation, with [helpers.py](../../examples/option/helpers.py) providing some additional functionality.

The simulation must be run with 4 processes and, to eliminate Monte-Carlo noise from the sensitivities, with each process using identical random number streams (the -c flag):

```bash
$ ./run_example.sh option 4 -c
[no 2/4] neworder 0.0.0 env: mc=(indep:0, seed:79748) python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0]
[no 1/4] neworder 0.0.0 env: mc=(indep:0, seed:79748) python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0]
[no 0/4] neworder 0.0.0 env: mc=(indep:0, seed:79748) python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0]
[no 3/4] neworder 0.0.0 env: mc=(indep:0, seed:79748) python 3.8.2 (default, Jul 16 2020, 14:00:26)  [GCC 9.3.0]
[no 1/4] initialise: registered object 'market'
[no 1/4] initialise: registered object 'option'
[no 1/4] registered transition compute_mc_price: neworder.pv = neworder.model.mc(option, market)
[no 1/4] registered checkpoint compare_mc_price: neworder.model.compare(neworder.pv, option, market)
[no 1/4] registered checkpoint compute_greeks: option.greeks(neworder.pv)
[no 1/4] starting microsimulation. start time=0.000000, timestep=0.750000, checkpoint(s) at [1]
[no 1/4] applying process-specific modifier: market.spot = market.spot * 1.01
[no 1/4] t=0.750000(1) transition: compute_mc_price
[no 3/4] initialise: registered object 'market'
[no 3/4] initialise: registered object 'option'
[no 3/4] registered transition compute_mc_price: neworder.pv = neworder.model.mc(option, market)
[no 3/4] registered checkpoint compare_mc_price: neworder.model.compare(neworder.pv, option, market)
[no 3/4] registered checkpoint compute_greeks: option.greeks(neworder.pv)
[no 3/4] starting microsimulation. start time=0.000000, timestep=0.750000, checkpoint(s) at [1]
[no 3/4] applying process-specific modifier: market.vol = market.vol + 0.001
[no 3/4] t=0.750000(1) transition: compute_mc_price
[no 2/4] initialise: registered object 'market'
[no 2/4] initialise: registered object 'option'
[no 2/4] registered transition compute_mc_price: neworder.pv = neworder.model.mc(option, market)
[no 2/4] registered checkpoint compare_mc_price: neworder.model.compare(neworder.pv, option, market)
[no 2/4] registered checkpoint compute_greeks: option.greeks(neworder.pv)
[no 2/4] starting microsimulation. start time=0.000000, timestep=0.750000, checkpoint(s) at [1]
[no 2/4] applying process-specific modifier: market.spot = market.spot * 0.99
[no 2/4] t=0.750000(1) transition: compute_mc_price
[no 0/4] initialise: registered object 'market'
[no 0/4] initialise: registered object 'option'
[no 0/4] registered transition compute_mc_price: neworder.pv = neworder.model.mc(option, market)
[no 0/4] registered checkpoint compare_mc_price: neworder.model.compare(neworder.pv, option, market)
[no 0/4] registered checkpoint compute_greeks: option.greeks(neworder.pv)
[no 0/4] starting microsimulation. start time=0.000000, timestep=0.750000, checkpoint(s) at [1]
[no 0/4] applying process-specific modifier: pass
[no 0/4] t=0.750000(1) transition: compute_mc_price
[no 3/4] t=0.750000(1) checkpoint: compare_mc_price
[py 3/4] mc: 7.244002 / ref: 7.235288 err=0.12%
[no 3/4] t=0.750000(1) checkpoint: compute_greeks
[no 1/4] t=0.750000(1) checkpoint: compare_mc_price
[py 1/4] mc: 7.768708 / ref: 7.760108 err=0.11%
[no 1/4] t=0.750000(1) checkpoint: compute_greeks
[no 1/4] SUCCESS exec time=0.026183s
[no 3/4] SUCCESS exec time=0.025125s
[no 2/4] t=0.750000(1) checkpoint: compare_mc_price
[py 2/4] mc: 6.673223 / ref: 6.665127 err=0.12%
[no 2/4] t=0.750000(1) checkpoint: compute_greeks
[no 2/4] SUCCESS exec time=0.028023s
[no 0/4] t=0.750000(1) checkpoint: compare_mc_price
[py 0/4] mc: 7.209954 / ref: 7.201286 err=0.12%
[no 0/4] t=0.750000(1) checkpoint: compute_greeks
[py 0/4] PV=7.209954
[py 0/4] delta=0.547743
[py 0/4] gamma=0.022023
[py 0/4] vega 10bp=0.034048
[no 0/4] SUCCESS exec time=0.027773s
```
