# Migrating to 2.0

neworder 2.0 replaces the timeline classes with plain data. A timeline is now an iterator (or any iterable) of `neworder.Timestep(index, start, end, dt)`, and the model, not the timeline, holds the current step, as `self.timestep`. See [Timeline](overview.md#timeline) for the full description.

The `stats` module which provided logit and logistic functions has now been removed, since they duplicate functionality in `scipy`. See [statistical functions](#statistical-functions) for the replacements.

If you can't migrate yet, pin `neworder<2`.

## Constructing a timeline

The classes become factory functions in `neworder.timeline`. Their arguments are keyword-only, apart from `start` for `linear` and `times` for `numeric`.

1.x | 2.0
----|----
`NoTimeline()` | `timeline.null()`
`LinearTimeline(0, 10, 10)`, integer unit steps | `timeline.linear(0, end=10)`
`LinearTimeline(0.0, 1.0, 4)`, any other steps | `timeline.numeric(start=0.0, step=0.25, end=1.0)`
`LinearTimeline(2020, 1)`, open-ended | `timeline.linear(2020)`, or `timeline.numeric(start=..., step=...)`
`NumericTimeline([0.0, 0.5, 2.0])` | `timeline.numeric([0.0, 0.5, 2.0])`
`CalendarTimeline(start, step, end=end)` | `timeline.calendar(start=start, step=step, end=end)`
a subclass of `neworder.Timeline` | a generator function, or a list, of `Timestep` (see [custom timelines](overview.md#custom-timelines))

`linear` only accepts integers, and `numeric` requires `end - start` to be a whole number of steps.

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
    self.population.loc[h == 1, "died"] = self.timestep.start
```

What `self.timeline` returned in 1.x depended on where you read it, because the timeline advanced between `step()` and `check()`. In 2.0, `step()`, `check()` and `finalise()` all see the same step:

1.x | in | 2.0
----|----|----
`self.timeline.time` | `step()` | `self.timestep.start`
`self.timeline.time` | `check()`, `finalise()` | `self.timestep.end`
`self.timeline.index` | `step()` | `self.timestep.index - 1` (the index now starts at 1)
`self.timeline.index` | `check()`, `finalise()` | `self.timestep.index`
`self.timeline.dt` | anywhere | `self.timestep.dt`
`self.timeline.start` | anywhere | record `self.timestep.start` in the first step
`self.timeline.end` | anywhere | `self.timestep.end` in `finalise()`
`self.timeline.at_end` | anywhere | no equivalent: do end-of-run work in `finalise()`
any of the above | `__init__()`, `modify()` | none: `self.timestep` is `None` before the first step

## Silent changes

These changes don't raise errors, so check for them in your model:

- **Step index starts at 1.** Anything keyed on the index, such as a random stream seeded from it or an array lookup, moves by one, and results won't match 1.x exactly.
- **`calendar` shortens the final step.** If `end` isn't a whole number of steps from `start`, the last step ends at `end`; in 1.x it overshot.
- **`null()` has an undefined `dt`.** It's `neworder.time.NEVER` (NaN), not 0.0.
- **`modify()` is called once.** In 1.x it was called again each time a halted model was resumed.
- **`halt()` in `modify()` is honoured.** The run stops before taking any steps.
- **A model halted on its final step** calls `finalise()` when it's resumed.
- **Iterating `self.timeline` takes steps from the run.** A `for` loop, `list()` or `any()` over it in model code skips those steps (see [advancing the timeline manually](overview.md#advancing-the-timeline-manually)). Construct the model with a list if you need to inspect the timeline.

## Statistical functions

Replace the `neworder.stats` functions with their `scipy.special` equivalents (`scipy` is installed with neworder):

1.x | 2.0
----|----
`neworder.stats.logistic(x)` | `scipy.special.expit(x)`
`neworder.stats.logistic(x, x0, k)` | `scipy.special.expit(k * (x - x0))`
`neworder.stats.logit(x)` | `scipy.special.logit(x)`

The `neworder.stats` functions accepted lists, but `k * (x - x0)` needs `x` to be a numpy array (or pandas object), so convert a list with `np.asarray(x)` first.
