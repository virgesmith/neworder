# Development Journal

A running log of every task/PR: *why* it was done and the *design decisions* made.
Newest entries at the top. This is the durable record of intent that keeps the
maintainer in control of the codebase's direction — see the
[Task & Design Summaries](AGENTS.md#task--design-summaries) policy in `AGENTS.md`.

Entry template:

```markdown
## YYYY-MM-DD — <title> (#PR)

**Why** — the motivation and the problem this solves.

**What** — high-level description of the change.

**Design decisions**
- <decision> — alternatives considered, why this was chosen.

**Follow-ups** — anything deferred, known limitations.
```

---

## 2026-07-26 — markov_chain example: demonstrate transition_conditional alongside transition (uncommitted)

**Why** — the new `no.df.transition_conditional` (see the entry below) had no example exercising it. `examples/markov_chain` was the natural place, since it already runs the single-matrix `no.df.transition` case with an analytic equilibrium cross-check to validate the simulation.

**What** — extended `MarkovChain` with an optional `group_transition_matrices` constructor argument. When supplied, the population is additionally split into groups (an interleaved, non-random assignment - deterministic, no extra RNG draw), each following its own transition matrix via `no.df.transition_conditional`, tracked in a second `state_conditional` column/`summary_conditional` table alongside (but independently of) the original single-matrix `state`/`summary`. Added `MarkovChain.mixed_stationary_distribution()` (population-share-weighted average of each group's own analytic stationary distribution, factored via a shared `_stationary_distribution` static method) as the equilibrium the grouped run should actually converge to, for comparison against the existing pooled-matrix `stationary_distribution()`. `model.py` now defines two group matrices and logs both pairs of simulated-vs-analytic equilibria; `docs/examples/markov-chain.md` gained a "Conditional transitions" section explaining the comparison. `visualisation.py` now renders both runs side by side (one panel per matrix regime, each with its own analytic-equilibrium dashed lines), and `docs/examples/img/markov-chain.png` was regenerated from an actual 100000-person/100-step run to match.

**Design decisions**
- The two group matrices bias the two routes out of state 0 in *opposite* directions (`mu_01/2, mu_02*2` vs `mu_01*2, mu_02/2`), not just scaled by different constant factors. First attempt scaled every rate in a group's matrix by a uniform factor (2x/0.5x) — verified numerically that this leaves the stationary distribution unchanged (a Markov chain's equilibrium depends on the ratios between rates, not their magnitude), which made the grouped and pooled analytic equilibria come out numerically identical and defeated the point of the example. Biasing the routes asymmetrically instead gives each group a genuinely different equilibrium.
- Kept the grouped run's RNG draws strictly *after* the pooled run's within `step()`, so adding the conditional branch doesn't perturb the existing pooled-run numbers (both draw from the same shared `self.mc` stream) — the pre-existing documented equilibrium values for the pooled case stay reproducible.
- `visualisation.py` initially shipped with no second chart (comparison logged as numbers only), to keep the diff scoped. Reversed that — a picture is the actual point of the comparison, and the two-panel figure makes the pooled-vs-grouped equilibrium gap (visible in the dashed lines) obvious in a way the log lines don't. Factored the single-panel plotting logic out into `_plot_occupancy(ax, ...)` so `show()` can call it once (pooled only) or twice (pooled + grouped, side by side via `plt.subplots(1, 2)`) depending on whether `group_transition_matrices` was supplied, keeping the no-groups case visually unchanged.

**Follow-ups** — none.

## 2026-07-26 — no.df.transition_conditional, and MonteCarlo& instead of Model& for df:: RNG consumers (uncommitted)

**Why** — after the recent `no.df.transition` hardening work, the natural next gap was a transition whose matrix depends on another column's value per row (e.g. probabilities that differ by age band or sex) — a stratified/conditional Markov step, not currently expressible without hand-rolling it in Python.

**What** — added `no.df.transition_conditional(mc, matrices, group, series)`: applies a different (square, category-count-sized) transition matrix per row depending on the value of a second categorical column (`group`). `matrices` is a `dict` keyed by `group`'s category labels; `group` and `series` must both be `category`-dtype and the same length. Rows with a missing/NaN `group` value are left untouched, mirroring the existing missing-category handling in `transition()`. Also changed `transition()` (and the new function) to take `no::MonteCarlo&` instead of `no::Model&` as their first argument, since that was the only thing they used off `Model` — callers now pass `model.mc` instead of `model`. Updated all call sites (`examples/markov_chain`, `examples/parallel/parallel_mpi.py`, `test/benchmark.py`, `test/test_df.py`) and regenerated the `df` stubs (`neworder/df.pyi`, `stubs/_neworder_core/df.pyi`).

**Design decisions**
- `matrices` as a `dict` keyed by group label, not a 3D `(n_groups, m, m)` array — considered the array form since it would match the existing all-numpy convention used by `transition()`'s matrix argument, but rejected it because it silently depends on `group.cat.categories` order matching array axis 0; a label-keyed dict is self-documenting and doesn't create that alignment hazard.
- Stub typing for `matrices` first landed as bare `dict` (untyped) because `dict[Any, ArrayLike]` made `ty` reject any concretely-typed dict literal a caller passes (e.g. `dict[str, np.ndarray]`) — `dict` is invariant in its value type parameter, so a concrete `ndarray` value type isn't assignable to the `ArrayLike` value type it expects even though every `ndarray` is a valid `ArrayLike`. Fixed properly by typing the parameter as `collections.abc.Mapping[Any, Annotated[ArrayLike, float64]]` instead — `Mapping` is covariant in its value type, so this keeps the precise typing without the false-positive. Also tightened `series`/`group` from `typing.Any` to `pandas.Series`, and both functions' return type from `typing.Any` to `pandas.Categorical`, in both `neworder/df.pyi` and `stubs/_neworder_core/df.pyi`.
- Considered also adding `no.df.sample` (draw categories from a fixed, unconditional distribution, for initializing a column) as a pandas-Categorical-returning wrapper — dropped it because `MonteCarlo.sample(n, cat_weights)` already exists and returns raw codes, and the only thing a wrapper would add is a one-line `pd.Categorical.from_codes(...)` call callers can just write themselves. Not enough value to justify a new bound function.
- Migrated `transition()`'s signature too (not just the new function), trading a breaking API change (Model& → MonteCarlo&) for consistency across all `no.df` RNG-consuming functions rather than leaving one on `Model&` and one on `MonteCarlo&`.

**Follow-ups** — `examples/parallel/parallel_mpi.py`'s updated call site is type-checked (`ty check` passes) but not executed here — this sandbox has no MPI (`mpiexec`/`mpi4py` unavailable). `test/benchmark.py` similarly type-checks but wasn't run — it depends on a large, non-committed CSV (`ssm_hh_E09000001_OA11_2011.csv`) not present in this environment. Both should be spot-checked wherever MPI/the data file are actually available.

## 2026-07-25 — Task & Design Summaries policy (uncommitted)

**Why** — task context (motivation, rejected alternatives, design rationale) was living entirely in chat sessions with AI agents, with nothing durable left behind once a session ended. `JOURNAL.md` already existed as a template for this, but nothing required it to be used, and `AGENTS.md` didn't reference it.

**What** — added a "Task & Design Summaries" section to `AGENTS.md` mandating a `JOURNAL.md` entry for every substantive development task, and a corresponding step in the `Workflow` checklist. Backfilled `JOURNAL.md` with entries for the (until now unrecorded) work already done this session.

**Design decisions**
- Kept the policy as its own `AGENTS.md` section rather than folding it into `Workflow`, because `JOURNAL.md`'s header already linked to the anchor `AGENTS.md#task--design-summaries` — the heading text was chosen to match that anchor exactly.
- Scoped the requirement to substantive changes (features, fixes, refactors, dependency/CI/design changes) and explicitly exempted trivial diffs, to avoid journal noise.

**Follow-ups** — none.

## 2026-07-25 — Examples nav: link section header to the index page (uncommitted)

**Why** — the new `docs/examples/index.md` landing page was reachable only via a duplicated child link inside the "Examples" dropdown; clicking the "Examples" section header itself did nothing.

**What** — enabled the `navigation.indexes` theme feature in `zensical.toml`.

**Design decisions**
- No template/plugin work needed: reading zensical's `templates/partials/nav-item.html` showed it already implements the standard mkdocs-material behaviour — when `navigation.indexes` is enabled, a nav section whose first child resolves to an `index.md`/`README.md` page (already the case for the `{ Examples = "examples/index.md" }` entry added alongside the index page) gets its header turned into a link to that page, and the duplicate child entry is omitted from the rendered list automatically.

**Follow-ups** — none.

## 2026-07-25 — Examples index page with card gallery (uncommitted)

**Why** — `docs/examples/index.md` was a placeholder (`# Examples` / `blah blah`); there was no landing page summarising what's available before diving into the nav dropdown.

**What** — replaced the placeholder with a Material-style `grid cards` gallery: one card per example (icon, title, one-line description, "Walkthrough" link), covering all examples in nav order. Wired an `{ Examples = "examples/index.md" }` entry into `zensical.toml`'s nav as the first child of the Examples section.

**Design decisions**
- Used the extensions already enabled in `zensical.toml` (`md_in_html`, `attr_list`, `pymdownx.emoji`) rather than adding new dependencies — confirmed the bundled zensical theme ships the `.grid.cards` CSS and the `material`/`octicons` icon sets needed for this pattern.
- Card descriptions were condensed from each example's own doc-page intro rather than invented, to stay accurate to the existing docs.

**Follow-ups** — none.

## 2026-07-25 — Membership example: SplitMix64 showcase (uncommitted)

**Why** — `neworder.SplitMix64` (the hash-based RNG added in #112) was documented in `docs/tips.md` but had no runnable example, despite `AGENTS.md` stating examples are the primary user-facing demonstration of the framework's features.

**What** — added `examples/membership/model.py`: a toy open-population "scheme membership" model where members join and churn (leave) each year. `finalise()` empirically asserts `SplitMix64`'s two defining properties — sub-population invariance and reordering invariance — and contrasts them against equivalent, positional `MonteCarlo.ustream()` draws. Added a `docs/examples/membership.md` walkthrough, a `zensical.toml` nav entry, and a cross-link from `docs/tips.md`'s "Ultimate Reproducibility" section.

**Design decisions**
- Modelled as an *open* population (members joining/leaving) rather than a closed one, because that's the exact scenario `docs/tips.md` already cites as `SplitMix64`'s motivating use case — makes the property concrete instead of abstract.
- The two invariance properties are directly `assert`ed in `prove_invariance()`, not just logged, so the example fails loudly if either regresses.
- The `MonteCarlo` contrast resets the engine (`self.mc.reset()`) between the two draws so both start from the same seed — an apples-to-apples comparison rather than an assertion made only in prose.
- Named the example (and its directory) `membership`, not `splitmix64` — this repo names examples after the domain/model (`mortality`, `people`, `schelling`, ...), not the technique being showcased. `churn` was considered as a more evocative alternative but `membership` was chosen to match the model's own class name and scenario.

**Follow-ups** — none currently known.

