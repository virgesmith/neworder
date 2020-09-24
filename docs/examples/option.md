# Derivative Pricing

This example showcases how to run parallel simulations, each with slightly different input data, in order to compute sensitivities to the input parameters.

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Background

Monte-Carlo simulation is a [common technique in quantitative finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance).

A [European call option](https://en.wikipedia.org/wiki/Call_option) is a derivative contract that grants the holder the right (but not the obligation) to buy an underlying stock \(S\) at a fixed "strike" price \(K\) at some given future time \(T\) (the expiry). Similarly, a put option grants the right (but not obligation) to sell, rather than buy, at a fixed price.

In order to calculate the fair value of a derivative contract one can simulate a (large) number of paths the underlying stock may take (according to current market conditions. The model assumes that the evolution of the underlying is given by the stochastic differential equation (SDE):

\[
\frac{dS}{S} = (r-q)dt + \sigma dW
\]

where \(S\) is price, \(r\) is risk-free rate, \(q\) is continuous dividend yield, \(\sigma\) is volatility and \(dW\) a Wiener process (a 1-d Brownian motion), and the value of the option \(V\) is

\[
V(0) = e^{-rT}.\text{max}\left( S(T)-K,0 \right)
\]

We can compute this by simulating paths to get \(S(T)\) and taking the mean. The first term above discounts back to \(t=0\), so we get the *current* fair value.

We can easily frame this derivative pricing problem in terms of a microsimulation model:

- start with an intial \(t=0\) population of \(N\) (identical) underlying prices \(S(0)\). Social scientists could refer to this as a 'cohort'.
- evolve each price to option expiry time \(S(T)\) using Monte-Carlo simulation

We then compute the mean of the discounted option prices for each of the underlying prices to get the result.

For this simple option we can also compute an analytic fair value under the Black-Scholes model, and use this to determine the accuracy of the Monte-Carlo simulation. We also demonstrate the capabilities neworder has in terms of sensitivity analysis, by using multiple processes to compute finite-difference approximations to the following risk measures:

- delta: \(\Delta=\frac{dV}{dS}\)
- gamma: \(\Gamma=\frac{d^2V}{dS^2}\)
- vega: \(\frac{dV}{d\sigma}\)

## Implementation

We use an implementation of the Monte-Carlo technique described above, and also, for comparision, the analytic solution.

Additionally, we compute some market risk: sensitivities to the underlying price and volatility. In order to do this we need to run the simulation multiple times with perturbations to market data. To eliminate random noise we also want to use identical random streams in each simulation. The model is run over 4 processes in the MPI framework to achieve this.

The `model.py` file sets up the run, providing input data, constructing, and the running the model. The input data consists of a `Dict` describing the market data, another describing the option contract, and a single model parameter (the number of paths).

{{ include_snippet("examples/option/model.py")}}

### Constructor

The constructor takes copies of the parameters, and defines a simple timeline \([0, T]\) corresponding to the valuation and expiry dates, and a single timestep, which is all we require for this example. It initialises the base class with the timeline, and specifies that each process use the same random stream (which reduces noise in our risk calculations):

{{ include_snippet("examples/option/black_scholes.py", "constructor") }}

### Modifier

This method defines the 'modifiers' for each process: the perturbations applied to the market data in each process in order to calculate the option price sensitivity to that market data. In this case we bump the spot up and down and the volatility up in the non-root processes allowing, calculation of delta, gamma and vega by finite differencing:

{{ include_snippet("examples/option/black_scholes.py", "modifier") }}

### Step

This method actually runs the simulation and stores the result for later use. The calculation details are not shown here for brevity (see the source file):

{{ include_snippet("examples/option/black_scholes.py", "step") }}

### Check

Even though we explicitly requested that each process has identical random streams, this doesn't guarantee the streams will stay identical, as different process could sample more or less than others, and the streams get out of step.

This method samples one uniform from each stream and will return `False` if any of them are different, which will halt the model (for that process).

!!! danger "Deadlocks"
    The implementation needs to be careful here is if some processes stop and others continue, a deadlock can occur when a process tries to communicate with a process that has ended. The check method must therefore ensure that ALL processes either pass or fail.

In the below implementation, all samples are sent to a single process (0) for comparison and the result is broadcast back to every process, which can then all fail simultaneously if necessary.

{{ include_snippet("examples/option/black_scholes.py", "check") }}

### Checkpoint

Finally the checkpoint method is called at end of the timeline. Again, the calculation detail is omitted for clarity, but the method performs two tasks:

- checks the Monte-Carlo result against the analytic formula and displays the simulated price and the random error, for each process.
- computes the sensitivities: process 0 gathers the results from the other processes and computes the finite-difference formulae.

{{ include_snippet("examples/option/black_scholes.py", "checkpoint") }}

## Execution

By default, the model has verbose mode off and checked mode on. These settings can be changed in [model.py]()

To run the model,

```bash
mpiexec -n 4 python examples/option/model.py
```
which will produce something like

```text
[py 2/4]  mc: 6.646473 / ref: 6.665127 err=-0.28%
[py 3/4]  mc: 7.216204 / ref: 7.235288 err=-0.26%
[py 1/4]  mc: 7.740759 / ref: 7.760108 err=-0.25%
[py 0/4]  check() ok: True
[py 0/4]  mc: 7.182313 / ref: 7.201286 err=-0.26%
[py 0/4]  PV=7.182313
[py 0/4]  delta=0.547143
[py 0/4]  gamma=0.022606
[py 0/4]  vega 10bp=0.033892
```

{{ include_snippet("./docs/examples/src.md") }}
