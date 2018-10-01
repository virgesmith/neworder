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

The [config.py](../../examples/option/config.py) file: 
- defines a simple timeline [0, T] corresponding to [valuation date, expiry date] and a single timestep.
- sets the parameters for the market and the option, and describes how to initialise the [market](../../examples/option/market.py) and [option](../../examples/option/option.py) objects with these parameters.
- describes the 'modifiers' for each process: the perturbations applied to the market data in order to calculate the option price sensitivity to that market data
- defines the "transition", which in this case is simply running the Monte-Carlo simulation from time zero to time T.
- checks the Monte-Carlo result against the analytic formula and displays the price and the random error.
- finally, gathers the results from the other processes and computes some sensitivities.

The file [black_scholes.py](../../examples/option/black_scholes.py) implements the both analytic option formula and the Monte-Carlo simulation, with [helpers.py](../../examples/option/helpers.py) providing some additional functionality. 

The simulation must be run with 4 processes and, to eliminate Monte-Carlo noise from the sensitivities, with each process using identical random number streams (the -c flag): 

```bash
$ ./run_example.sh option 4 -c
[no 3/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 3/4] starting microsimulation t=0.000000
[no 3/4] initialising market
[no 3/4] initialising option
[no 1/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 1/4] starting microsimulation t=0.000000
[no 1/4] initialising market
[no 1/4] initialising option
[no 2/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 2/4] starting microsimulation t=0.000000
[no 2/4] initialising market
[no 2/4] initialising option
[no 0/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 0/4] starting microsimulation t=0.000000
[no 0/4] initialising market
[no 0/4] initialising option
[no 3/4] initialising model
[no 3/4] applying modifier: exec("market.vol = market.vol + 0.001")
[no 3/4] t=0.750000: compute_mc_price
...
[no 2/4] SUCCESS
[no 3/4] SUCCESS
[no 1/4] SUCCESS
[py 0/4] PV=7.213875
[py 0/4] delta=0.548820
[py 0/4] gamma=0.022923
[py 0/4] vega 10bp=0.034061
[no 0/4] SUCCESS
```
