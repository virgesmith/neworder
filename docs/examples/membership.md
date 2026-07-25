# Membership

A toy open-population model of scheme membership - members join and churn (leave) over time - used to showcase `neworder.SplitMix64`, a hash-based random stream that is an alternative to the default sequential `MonteCarlo` engine. It is aimed at *open* population models, where individuals join and leave over time, and per-individual reproducibility needs to survive changes in population size, membership or row order.

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Why SplitMix64?

`MonteCarlo` is a sequential stream: every draw advances its internal state, so the value returned depends on how many draws were taken before it. That's fine for a fixed, ordered population, but it means the draw a given individual receives is really a property of *when* it was requested, not of the individual itself. If the population changes shape between runs - someone leaves, someone new is inserted earlier in the DataFrame, the rows get reordered - the same individual can end up with a different draw for no modelling reason.

`SplitMix64` has no state to advance. Each variate is computed by hashing a set of integer keys - typically a person id, a process id and a timestep - together with a seed. The draw for a given key is always the same, regardless of which other keys are present or in what order they're processed. See [Tips and Tricks](../tips.md#ultimate-reproducibility) for the full API reference.

## The model

A toy open-population scheme: it starts with a fixed number of members, and each year:

- every existing member churns (leaves) with a fixed annual hazard,
- some number of new members join.

{{ include_snippet("./examples/membership/run.py")}}

## Implementation

The model constructs a `SplitMix64` instance alongside the usual `MonteCarlo` engine, and uses a constant master seed - the "churn" stream via `SplitMix64.hash64` - ensuring
that calls to uarray with identical arguments will produce identical results.:

{{ include_snippet("./examples/membership/membership.py", "constructor")}}

Each year, `step` draws one churn variate per *member id* and *year* - not per row - so the outcome for a given member cannot be affected by who else is in the population that year, and the outcomes for different years are independent:

{{ include_snippet("./examples/membership/membership.py", "step")}}

## Proving the invariance

`finalise` doesn't just report the membership history - it demonstrates the property the example is built around, using the final surviving population:

{{ include_snippet("./examples/membership/membership.py", "invariance")}}

Two checks against `SplitMix64`:

1. **Sub-population invariance** - hashing the whole population and hashing a single member in isolation give the same answer for that member. Nothing else in the population affects it.
2. **Reordering invariance** - shuffling the order of the member ids changes their position in the output array, but every id still maps back to the same value.

Both are asserted directly in the code, so the example fails loudly if either property is ever broken.

For contrast, the same reordering is then applied to a plain `MonteCarlo.ustream()` draw, resetting the engine (via `self.mc.reset()`) between the two calls so both start from the same seed. Because `ustream` output is positional, every single member ends up with a different value purely because of where it sits in the shuffled array.

## Output

```bash
python examples/membership/run.py
```

```text
[py 0/1(2956824)] 2026: 1000 members
[py 0/1(2956824)] 2027: 995 members
[py 0/1(2956824)] 2028: 892 members
...
[py 0/1(2956824)] 2036: 984 members
[py 0/1(2956824)] SplitMix64: a member's draw is unchanged by sub-sampling or reordering
[py 0/1(2956824)] MonteCarlo: reordering alone changed the draw for 984/984 members
```

The exact membership counts will vary between runs of this example, but the last two lines will always read the same way: `SplitMix64` is unaffected by the shuffle, and `MonteCarlo` changes for (close to) every member.

## Next steps

Compare this against the [Mortality](./mortality.md) or [People](./people.md) examples, which use the sequential `MonteCarlo` stream for a fixed, closed population where positional draws are not a concern. See [Tips and Tricks](../tips.md#ultimate-reproducibility) for the complete `SplitMix64` API, including `use_counter` for getting independent repeated draws.
