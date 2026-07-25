# Markov Chain

This example illustrates how to model a process that consists of probabilistic transitions between discrete states, and showcases how *neworder* can drastically increase performance on certain operations on dataframes.

Firstly we have 3 arbitrary states: 0, 1 and 2. The initial population starts in state 0, and the following transitions are permitted, as described by this transition matrix:

\[
\begin{pmatrix}
1-p_{01}-p_{02} & p_{01}   & p_{02}   \\
0               & 1-p_{12} & p_{12}   \\
p_{20}          & 0        & 1-p_{20}
\end{pmatrix}
\]

Each transition is modelled as a Poisson process with different mean arrival times \(\mu_{ij}=1/\lambda_{ij}\), which generate the probabilities above by \(p_{ij}=\lambda_{ij}.\delta t\)

We use a time horizon of 100 (arbitrary units) with 100 steps and a population of 100000. This equates to computing ten million possible transitions during the model run. The sizes of the populations in each state, as the model progresses, is illustrated below. As you can see an equilibrium state is reached. (NB This means balanced transitions rather than no transitions)

As well as simulating the population, the model computes the *analytic* equilibrium (stationary) distribution of the Markov chain directly from the transition matrix (as the eigenvector corresponding to eigenvalue 1). This is plotted alongside the simulated proportions (dashed lines, below) as a check that the simulation is behaving correctly, and is also logged at the end of the run for comparison against the simulated proportions in the final timestep.

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Performance

The model also implements a pure-python equivalent of the `no.df.transition()` function (`MarkovChain.transition_py()`), for comparison. Set `use_python_impl = True` in `model.py` to use it instead of *neworder*'s C++ implementation.

For the full 100000-person, 100-step model, the C++ implementation takes about 2.5s (depending on platform), versus about 8s for the python implementation - a factor of 3 or so. Note that the C++ implementation can only operate on integer state data - if the state is expressed as another type, e.g. a string, consider changing the format, or just use the python implementation.

## Input

{{ include_snippet("./examples/markov_chain/model.py") }}

## Implementation

{{ include_snippet("./examples/markov_chain/markov_chain.py") }}

## Output

![population evolution](./img/markov-chain.png)
