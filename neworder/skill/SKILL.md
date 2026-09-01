---
name: neworder
description: >
  Use when writing, editing, running or debugging a microsimulation model built on the neworder
  framework — subclassing neworder.Model, defining a timeline, sampling from the MonteCarlo or
  SplitMix64 engines, transitioning pandas categorical data with neworder.df, spatial domains, and
  MPI/multithreaded parallel runs. Triggers: "neworder", "no.Model", "neworder.run",
  "microsimulation", NoTimeline/LinearTimeline/NumericTimeline/CalendarTimeline, mc.hazard,
  mc.stopping, mc.arrivals, no.df.transition, SplitMix64, StateGrid, neworder.mpi.
---

# Building models with neworder

`neworder` is a dynamic microsimulation framework: a C++ core (via pybind11) exposed as a Python
module. You supply a `Model` subclass and a timeline; the runtime iterates the timeline, calling
your methods at each step. It is data-agnostic — populations are normally `pandas` DataFrames and
the library functions operate directly on them (and on `numpy` arrays) without copying.

**The canonical documentation is the project site, <https://neworder.readthedocs.io>.** This skill
is a working reference for the shape of a correct model; when you need detail beyond it, read the
relevant page rather than guessing:

| Page | What's there |
|------|--------------|
| [Overview](https://neworder.readthedocs.io/en/stable/overview/) | the framework's model, timelines, spatial domains, data/performance notes |
| [Tips and Tricks](https://neworder.readthedocs.io/en/stable/tips/) | seeding strategies, reproducibility, `SplitMix64`, halting, deadlocks, time comparison |
| [Examples](https://neworder.readthedocs.io/en/stable/examples/) | 16 runnable models, each with a walkthrough — the best source of idiom |
| [API Reference](https://neworder.readthedocs.io/en/stable/api/) | note: the API is documented by the package's own type annotations/stubs, not on the site |
| [Developer](https://neworder.readthedocs.io/en/stable/developer/) | building from source, running tests, generating stubs |

Use `/en/stable/` for the released version (what `pip install neworder` gives you) and
`/en/latest/` for the current `main`.

## Model lifecycle

`neworder.run(model)` drives this sequence, and nothing else should advance the timeline:

1. `modify()` — once, before the run. Optional; typically used to perturb inputs per MPI process.
2. `step()` — once per timestep. **Required.**
3. `check()` — after each `step()`, if implemented. Returning `False` aborts the run.
4. `finalise()` — once, on reaching the end of the timeline. Optional.

`run` returns `False` if the model failed — check it. Disable `check()` calls globally with
`neworder.checked(False)`; turn on runtime logging with `neworder.verbose()`. Log from a model
with `neworder.log(...)`, which prefixes process/timestep context.

## Minimal model

```py
import neworder as no
import pandas as pd


class MyModel(no.Model):
    def __init__(self, n: int, mortality_rate: float) -> None:
        # ESSENTIAL: initialise the base class with a timeline (and optionally a seeder).
        # Omitting this is the single most common error — it fails at runtime.
        super().__init__(no.LinearTimeline(0.0, 100.0, 100), no.MonteCarlo.deterministic_identical_stream)
        self.population = pd.DataFrame(index=no.df.unique_index(n), data={"age": 0.0, "alive": True})
        self.mortality_rate = mortality_rate

    def step(self) -> None:
        died = self.mc.hazard(self.mortality_rate, len(self.population)).astype(bool)
        self.population.loc[died, "alive"] = False
        self.population.age += self.timeline.dt

    def check(self) -> bool:
        return bool((self.population.age >= 0.0).all())

    def finalise(self) -> None:
        no.log(f"survivors: {self.population.alive.sum()}")


if __name__ == "__main__":
    model = MyModel(10000, 0.01)
    if not no.run(model):
        no.log("model failed")
```

`self.mc` (the `MonteCarlo` engine), `self.timeline` and `self.run_state` are provided by the base
class. Vectorise: prefer whole-array `mc`/`numpy`/`pandas` operations over per-agent Python loops.

## Timelines

| Class | Use for |
|-------|---------|
| `NoTimeline()` | continuous-time / case-based models evaluated in one instantaneous step |
| `LinearTimeline(start, end, nsteps)` | equally-spaced non-calendar steps; `LinearTimeline(start, step)` is open-ended |
| `NumericTimeline(times)` | explicit, unequally-spaced non-calendar times |
| `CalendarTimeline(start, relativedelta(...), end=...)` | calendar dates (day/month/year steps, ACT/365 year fractions) |

Properties: `index`, `time`, `dt`, `start`, `end`, `nsteps`, `at_end`. Custom timelines subclass
`no.Timeline` and override `start`, `end`, `time`, `dt`, `_next` (and optionally `__repr__`);
`index` is provided and must not be overridden. `CalendarTimeline` (pure Python, in
`neworder/timeline.py`) is the reference implementation to copy.

Open-ended timelines stop via `model.halt()` from within `step()`. Note that `halt()` does *not*
return immediately — the rest of `step()` and `check()` still run — and `finalise()` is **not**
called for a halted model; call it explicitly if needed. A halted model can be resumed by passing
it to `no.run` again.

## Randomness

### `MonteCarlo` — the model's sequential stream (`self.mc`)

| Method | Returns |
|--------|---------|
| `ustream(n)` | `n` U[0,1) variates |
| `hazard(p, n)` / `hazard(p_array)` | Bernoulli outcomes (0.0/1.0) for a constant or per-agent probability |
| `stopping(lambda_, n)` / `stopping(lambda_array)` | times to a stopping event, constant hazard rate |
| `arrivals(lambda_, dt, n, mingap)` | arrival times from a non-homogeneous Poisson process |
| `first_arrival(...)` / `next_arrival(...)` | first/subsequent arrivals in a non-homogeneous Poisson process |
| `counts(lambda_, dt)` | event counts |
| `sample(n, cat_weights)` | categorical sampling |
| `raw()` / `state()` / `seed()` | a random 64-bit integer (for seeding other generators) / internal state / the seed |
| `reset()` | re-invoke the seeder |

Non-arrivals are returned as `neworder.time.NEVER` (NaN) — test with `no.time.isnever(x)`, never
with `==`. `no.time.DISTANT_PAST`/`FAR_FUTURE` are `-inf`/`+inf`.

Seeding strategies (passed as the second argument to `super().__init__`, a callable returning an
`int` that fits `int32`): `MonteCarlo.deterministic_identical_stream`,
`deterministic_independent_stream` (keyed on MPI rank), `nondeterministic_stream`, or your own.
Identical streams + perturbed inputs → sensitivity analysis; independent streams + identical
inputs → convergence analysis.

For all of `numpy`'s distributions on neworder's stream, use the adapter `no.as_np(self.mc)`.

### `SplitMix64` — stateless, keyed draws

`MonteCarlo` is sequential: a draw depends on how many draws preceded it, so adding, removing or
reordering agents changes every subsequent variate. `SplitMix64` hashes integer keys instead, so
the draw for agent *i* is the same whether or not any other agent was drawn:

```py
self.rng = no.SplitMix64(no.MonteCarlo.deterministic_identical_stream)
draws = self.rng.uarray(person_ids, no.SplitMix64.hash64("mortality"), self.timeline.index)
```

Keys are scalars (context, adding no dimension) or **1-D** integer arrays (each adding one output
dimension); multi-dimensional key arrays raise `TypeError`. `raw(...)` returns the underlying
`int64` hashes instead of U[0,1) — used for seeding (e.g. per-timestep re-seeding of `mc`, or
`np.random.PCG64(rng.raw(np.arange(4), MODEL_ID).view(np.uint64))`; view as `uint64` because
numpy rejects negative entropy). `use_counter=True` makes repeated identical calls differ, but the
counter is not thread-safe — give each thread its own instance, or key on a thread-specific
scalar. Choose `MonteCarlo` for non-uniform sampling (it has no `SplitMix64` equivalent) and
`SplitMix64` for uniform draws that must survive sub-sampling or reordering.

## DataFrame operations (`neworder.df`)

- `no.df.unique_index(n)` — `n` index values unique across MPI processes.
- `no.df.transition(mc, matrix, series)` — Markov transition of a **categorical** series. Returns
  a new `pd.Categorical`; it does **not** modify in place, so assign the result back:
  `df[col] = no.df.transition(model.mc, m, df[col])`. `matrix` rows must follow
  `series.cat.categories` order; convert with `.astype("category")` first if needed.
- `no.df.transition_conditional(mc, matrices, group, series)` — as above with a different matrix
  per row, selected by a second categorical column (e.g. by age band or sex). Rows whose group is
  missing are untouched; every other group category needs an entry in `matrices`.

`no.df.transition` is orders of magnitude faster than an equivalent Python loop — use it rather
than iterating rows.

## Spatial domains

Optional. `Space` (continuous, arbitrary dimension; `move`, `dists`, `dists2`, `in_range`),
`StateGrid` (discrete grid; `count_neighbours`, `shift`, indexing), and `GeospatialGraph` (a
`networkx`/`osmnx` wrapper for shortest paths, isochrones, subgraphs — needs the `geospatial`
extra). Edge behaviour is `no.Edge.UNBOUNDED`, `WRAP`, `CONSTRAIN` or `BOUNCE` (`Space` supports
all; `StateGrid` supports `WRAP` and `CONSTRAIN` only). See the boids, Conway, Schelling and
wolf-sheep examples.

## Parallel execution

`no.mpi.RANK`, `no.mpi.SIZE`, `no.mpi.COMM` (rank 0, size 1, `COMM` unused in serial). Requires
one of the `parallel-native`/`parallel-openmpi`/`parallel-mpich` extras and `mpiexec -n <N>`.

- Failure must be **all-or-nothing**: a blocking communication whose counterpart has already
  exited deadlocks the whole run. Have one process compute a shared `check()` result and broadcast
  it, or reduce with a logical "and" — never let `check()` return different values per process.
- Use `modify()` for per-rank input perturbation.
- Identically-seeded streams only stay synchronised if each process takes the same number of draws.
- Multithreaded (rather than multiprocess) models: `no.freethreaded()` reports whether the
  interpreter is free-threaded, and `no.thread_id()` identifies the calling thread.

## Gotchas

- **The base class must be initialised** — `super().__init__(timeline, seeder)`. Skipping it is a
  runtime error, not a quiet one.
- **The C++ core is statically typed.** Passing `3.0` where an `int` is expected raises `TypeError`
  immediately, and `dtype` matters as much as the outer type. The package ships annotations and
  `py.typed`, so a type checker catches most of this — use one.
- **`transition`/`transition_conditional` return new data**, they do not mutate the series.
- **Never advance the timeline yourself**; the runtime does it between steps.
- **`NEVER` is NaN** — `NEVER == NEVER` is `False`; use `no.time.isnever`.
- **`mc.reset()` re-invokes the seeder**, so a non-deterministic seeder gives a *different* stream
  after reset, not the original one.
- **Explicit per-agent Python loops are the usual performance bug.** Reach for the vectorised
  `mc.*`, `no.df.*` and numpy equivalents first.

## Running models and examples

Examples are self-contained directories under `examples/`, entry point `model.py` or `run.py`:

```sh
python examples/mortality/model.py
mpiexec -n 2 python examples/parallel/model.py   # parallel examples
```

## Working on neworder itself

If you are changing this framework rather than using it, the repository's `AGENTS.md` governs the
workflow: C++ core in `src/`, Python package in `neworder/`, type stubs regenerated with
`pybind11-stubgen` whenever the pybind11 bindings change, the full gate suite
(`uv run ruff check`, `uv run ruff format --check`, `uv run ty check neworder examples test`,
`uv run pytest`) green before anything is called done, the affected `examples/` run by hand, docs
under `docs/` updated, and a `JOURNAL.md` entry for every substantive change.
