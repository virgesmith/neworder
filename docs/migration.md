# Migrating to 2.0

neworder 2.0 replaces the timeline classes with plain data. A timeline is now an iterator (or any iterable) of `neworder.Timestep(index, start, end, dt)`, and the model, not the timeline, holds the current step, as `self.timestep`. See [Timeline](overview.md#timeline) for the full description.

If you can't migrate yet, pin `neworder<2`.

## Constructing a timeline

The classes become factory functions in `neworder.timeline`. Each takes `start` (or the points, for `sequence`) as its first argument, and the rest are keyword-only.

1.x | 2.0
----|----
`NoTimeline()` | `timeline.null()`
`LinearTimeline(0, 10, 10)`, integer unit steps | `timeline.linear(0, end=10)`
`LinearTimeline(0.0, 1.0, 4)`, any other steps | `timeline.numeric(0.0, step=0.25, end=1.0)`
`LinearTimeline(2020, 1)`, open-ended | `timeline.linear(2020)`, or `timeline.numeric(2020.0, step=...)`
`NumericTimeline([0.0, 0.5, 2.0])` | `timeline.sequence([0.0, 0.5, 2.0])`
`CalendarTimeline(start, step, end=end)` | `timeline.calendar(start, step=step, end=end)`
a subclass of `neworder.Timeline` | a generator function, or a list, of `Timestep` (see [custom timelines](overview.md#custom-timelines))

`linear` only accepts integers, and `numeric` requires `end - start` to be a whole number of steps.

`sequence` is not limited to numbers: the points can be of any type, and `start` and `end` of each step are the points themselves, unconverted. The length of a step is computed by the `dt` argument, a function of the step's start and end returning a `float`. It defaults to `end - start`, so it must be supplied for points whose difference isn't a number, e.g. for dates:

```python
dates = pd.date_range("2026-01-31", "2026-12-31", freq="ME")
timeline = neworder.timeline.sequence(dates, dt=lambda start, end: (end - start).days / 365)
```

`dt` must be positive for every step, i.e. the points must be increasing. Unlike `NumericTimeline`, which validated the points on construction, `sequence` raises a `ValueError` only when the run reaches the offending step, and a sequence with fewer than two points is an empty timeline, not an error.

The factories return generators, which a model run consumes. To share a timeline between models, convert it to a list, or call the factory once per model.

In type annotations, replace `neworder.Timeline` with `Iterable[neworder.Timestep]`.

## Reading the current step

Replace `self.timeline` with `self.timestep` in model code:

```python
# 1.x
def step(self) -> None:
    h = self.mc.hazard(self.rates * self.timeline.dt)
    self.population.loc[h == 1, "died"] = self.timeline.time


# 2.0
def step(self) -> None:
    h = self.mc.hazard(self.rates * self.timestep.dt)
    self.population.loc[h == 1, "died"] = self.timestep.end
```

`step()` evolves the model's state from `self.timestep.start` to `self.timestep.end`, over a period of length `self.timestep.dt`. `check()` and `finalise()` see the same step, and are normally considered to happen at its end, `self.timestep.end`.

This differs from 1.x, where the timeline had a single current time, and advanced between `step()` and `check()`:

- **`time`** was the start of the step in `step()`, but its end in `check()` and `finalise()`. Replace it with `self.timestep.start` or `self.timestep.end` accordingly. The 1.x code above dated a death to the start of the step it happened in; the 2.0 version uses the end, which is usually what's meant. Use `self.timestep.start` there to reproduce 1.x results exactly.
- **`index`** was the number of steps completed: 0 in the first `step()` and 1 by the first `check()`. `self.timestep.index` is the number of the current step, counting from 1, throughout the step. So in `step()` the 1.x value is `self.timestep.index - 1`, and in `check()` and `finalise()` it's unchanged.
- **`start`, `end` and `at_end`** described the whole timeline, and have no equivalent, as a step knows nothing about the rest of its timeline. Record `self.timestep.start` in the first step if the start of the timeline is needed, read `self.timestep.end` in `finalise()` for its end, and do end-of-run work in `finalise()`.
- **Before the first step**, i.e. in `__init__()` and `modify()`, there is no current step: `self.timestep` is `None`.

## Silent changes

These changes don't raise errors, so check for them in your model:

- **Step index starts at 1.** Anything keyed on the index, such as a random stream seeded from it or an array lookup, moves by one, and results won't match 1.x exactly.
- **`calendar` shortens the final step.** If `end` isn't a whole number of steps from `start`, the last step ends at `end`; in 1.x it overshot.
- **`null()` has an undefined `dt`.** It's `neworder.time.NEVER` (NaN), not 0.0.
- **`modify()` is called once.** In 1.x it was called again each time a halted model was resumed.
- **`halt()` in `modify()` is honoured.** The run stops before taking any steps.
- **A model halted on its final step** calls `finalise()` when it's resumed.
- **Iterating `self.timeline` takes steps from the run.** A `for` loop, `list()` or `any()` over it in model code skips those steps (see [advancing the timeline manually](overview.md#advancing-the-timeline-manually)). Construct the model with a list if you need to inspect the timeline.
