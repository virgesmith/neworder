# Hello World

This simple example illustrates the basic model structure, how it all fits together and how it's executed by the framework.

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Model Definition and Execution

The framework runs a model via the `neworder.run` function, which takes an instance of a `Model` class. All models contain, at a minimum:

- an instance of a timeline, in this case the `NoTimeline` arbitrary single-step timeline
- an instance of a `neworder.MonteCarlo` engine
- user-defined methods to evolve the state (`step`) and report/postprocess results (`finalise`).

In this example the model doesn't have an explicit discrete timeline, so for models of this type a method is provided to construct an empty timeline (which is a single step of length zero).

!!! note "Timelines"
    In some model specifications, each individual's entire history can be constructed in a single pass, and in this type of situation a null timeline is appropriate. In more complex examples, the timeline could either refer to absolute time, or for "case-based" models (to use MODGEN parlance), the age of a cohort.

In this (rather contrived) example we have a population a who possess a sole boolean "talkative" attribute, which is initially `False`. The model randomly transitions this state, according to a given probability, and then those individuals who have become talkative say hello.

## Inputs

The input data for this model are just:

- the size of the population
- the probability of saying hello

## Implementation

Firstly we create our model class, subclassing `neworder.Model`:

{{ include_snippet("./examples/hello-world/model.py", "class") }}

and provide a constructor that initialises the base class and a DataFrame containing the population:

{{ include_snippet("./examples/hello-world/model.py", "constructor") }}

!!! note "Unique Indexing"
    The `neworder.df.unique_index()` provides a mechanism to guarantee unique indices for DataFrames, even for parallel runs. This allows individuals to be exchanged and tracked between processes without conflicting indices.

The `step` method randomly samples new values for the "talkative" attribute, using the `neworder.MonteCarlo.hazard` method

{{ include_snippet("./examples/hello-world/model.py", "step") }}

and at the end of the timeline the `finalise` method prints greetings from the talkative individuals using the `neworder.log` function, which is preferred to plain `print` statements as the output is annotated with useful context for debugging.

{{ include_snippet("./examples/hello-world/model.py", "finalise") }}

## Execution

The model is run by simply constructing an instance of our model and passing it to the `run` method:

{{ include_snippet("./examples/hello-world/model.py", "script") }}

From the command line, run the model:

```bash
python examples/hello-world/model.py
```

which should result in something like

```text
[py 0/1]  Hello from 0
[py 0/1]  Hello from 3
[py 0/1]  Hello from 4
[py 0/1]  Hello from 6
```

## Understanding the workflow and the output

To get a better idea of what's going on, uncomment the line containing `neworder.verbose()` and rerun the model. The output is explained line-by-line below.

!!! note "Annotated Output"
    The `neworder.log` output is prefixed with a source identifier in square brackets, containing the following information for debugging purposes:

      - Source of message: `no` if logged from the framework itself, `py` if logged from python code (via the `neworder.log()` function). The former are only displayed in verbose mode.
      - the process id ('rank' in MPI parlance) and the total number of processes ('size' in MPI parlance) - in serial mode these default to 0/1.


First we get some information about the model initialisation parameters:

```text
[no 0/1]  model init: timeline=<NoTimeline stepped=False> mc=<neworder.MonteCarlo seed=874991939>
[no 0/1]  starting model run. start time=nan
```

When using `NoTimeline()` the start time, end time are undefined and the timestep is zero - i.e. a single arbitrary step in which all computation is done.

The RNG seed is displayed for reproducibility, although in this configuration we explicitly requested a random seed using the `neworder.MonteCarlo.nondeterministic_stream` parameter, so each time you run the model you will get a different seed, and different results. The next output concerns the modify method which is explained in the [Option](./option.md) example.

```text
[no 0/1]  t=nan(0) HelloWorld.modify(0)
[no 0/1]  defaulted to no-op Model::modify()
```

Note that the time value is NaN (not-a-number) since absolute time is not important in this implementation, but the index is zero. Now the `step` method is called, which applies our single random transition - making some individuals "talkative":

```text
[no 0/1]  t=nan(1) HelloWorld.step()
```

and this is followed by the `check` method, which is optional and we haven't implemented:

```text
[no 0/1]  defaulted to no-op Model::check()
[no 0/1]  t=nan(1) HelloWorld.check() [ok]
```
and the time index has updated to 1.

!!! note "Checks"
    Custom data sanity checks can be run after each timestep by overriding the `check` method. The default implementation does nothing. A typical pattern would be to implement checks for debugging a model during development, then disable them entirely to improve performance using `neworder.checked(False)`. See the other examples.

We've now reached the end of our single step "timeline" so the `finalise` method is called:

```text
[no 0/1]  t=nan(1) HelloWorld.finalise()
```

which in this case just prints the results:

```text
[py 0/1] Hello from 0
[py 0/1] Hello from 1
[py 0/1] Hello from 4
[py 0/1] Hello from 7
[py 0/1] Hello from 8
```

and finally the model reports its status and execution time:

```text
[no 0/1] SUCCESS exec time=0.000914s
```

## Next steps

Try re-running the model with different input parameters, or changing the seeding strategy for reproducible results. To reproduce the exact results above replace `neworder.MonteCarlo.nondeterministic_stream` with `lambda _: 874991939` in the model initialiser, or just use `neworder.MonteCarlo.deterministic_independent_stream` to get consistent results for every run.

Then, check out some or all of the other examples...
