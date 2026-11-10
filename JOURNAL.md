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

## 2026-11-10 — Document advancing the timeline from model code (#134)

**Why** — For an iterator timeline, `self.timeline` is the iterator the run takes its steps from, so model code can
advance it. The 2026-11-07 guard against this (`RuntimeError`) went with the timeline classes, so it's now silent.
Advancing deliberately can be useful, e.g. to skip steps, but iterating `self.timeline` for some other reason
consumes steps without warning.

**What**
- The overview has a section on advancing the timeline manually, with a warning about doing it by accident. The
  bundled skill, and the `Model.timeline` docstring and stub, say the same.
- Tests pin the behaviour: `next(self.timeline, None)` skips a step and leaves `self.timestep` unchanged, skipping
  the remaining steps finalises after the current one, iterating a list timeline has no effect, a bare `next()` on
  an exhausted timeline raises `StopIteration` out of the run, and a partly consumed iterator starts at its next step.
- The skill's stale "`timeline.time` is the end of the step" pitfall is replaced.

**Design decisions**
- *Document and support it rather than reinstate a guard.* The run loop only sees an iterator, so detecting a step
  taken elsewhere would mean wrapping it, and that would also block deliberate use. A list timeline is the way to
  make the timeline safe to inspect.
- Examples narrow `self.timeline` with `assert isinstance(self.timeline, Iterator)` rather than suppressing the type
  error, since `Model.timeline` is correctly typed `Iterable`.

**Follow-ups** — a bare `next()` past the end leaves the model `RUNNING`; it could be finalised instead, but code
that does this has a bug, so raising is the more useful outcome.

---

## 2026-11-10 — Resuming a halted model doesn't call `modify` again (#134)

**Why** — `Model::run` called `modify()` unconditionally, so every resume of a halted model applied it again (this
dates from 1.x). `modify` is for one-off changes to the model's inputs, e.g. per-process perturbations in a parallel
run or sensitivity analysis, and applying it twice corrupts them. A model halted on its final step called `modify`
again when resumed, before finalising.

**What**
- `Model::run` calls `modify()` only when the run state is `NOT_STARTED`. A resumed run carries on with the next step;
  if there is none, it just calls `finalise()`.
- Tests assert `modify` is called once across halt/resume, halt in `modify`, and halt on the final step.
- Docs, the `modify` docstring and stub, the timeline contract and the agent skill say it's called once, on the first
  run.

**Design decisions**
- The run state is the guard, not a separate flag: any state other than `NOT_STARTED` means `modify` has run. This
  covers a model halted in `modify` itself, and one whose `check()` failed, which is left `RUNNING`.

**Follow-ups** — none.

---

## 2026-11-10 — Timeline v2: timelines are data; the model owns the current step (#134)

**Why** — In the first v2 design, timelines were stateful iterator objects whose `__next__` returned each step as a
native `(start, end)` tuple. The only caller was `Model::run`, which discarded the tuple, so model code couldn't reach
the native values (the people example rebuilt dates from the float time). The iterator also invited model code to
advance the timeline, which the run loop then had to guard against. Several intermediate designs kept the timeline as
a stateful clock: a C++ base class with abstract properties dispatched through a hand-written trampoline, then a C++
clock computing everything from `_point(i)`/`_time(i)`/`_at_end(i)` hooks. Each added mechanism to the C++/Python
boundary, and the step state lived in the wrong place: the position in the timeline is part of a model run, like its
halt state.

**What**
- A timeline is plain data: an iterator of `neworder.timeline.Timestep(index, start, end, dt)` named tuples (or any
  iterable of them). `start` and
  `end` are native (e.g. `int` or `date`), and `dt` is a float.
- `neworder.timeline` provides factories, `null()`, `linear()`, `numeric()` and `calendar()`, which always return a
  generator, open-ended if `end` is omitted. They replace `NoTimeline`, `LinearTimeline`,
  `NumericTimeline` and `CalendarTimeline`.
- `calendar()` shortens a final step that would overshoot `end`, rather than ending after it.
- The C++ `Timeline` class, its trampoline, bindings and stub are gone. `Model` holds the iterable, an iterator over
  it, and the current step, exposed as `Model.timestep` (`None` before the first step). `Model::run` takes steps with
  `PyIter_Next`, so there's no exception-driven control flow, and finalises when the iterator is exhausted. Keeping
  the iterator between runs is what makes halt and resume work.
- Examples read `self.timestep` instead of `self.timeline`. Models that need the start time record it from the first
  step's `start` (markov_chain, membership).

**Design decisions**
- Data rather than objects: the run loop needs nothing from a timeline but its next step, so a timeline needn't be a
  class, and a custom timeline is just a list or a generator function.
- The current step lives on the model (`self.timestep`), not passed to `step()`: method signatures are unchanged, and
  `check()` and `finalise()` see it too.
- `timestep` is typed as `Timestep` in the stub, although it is `None` before the first step. Typing it `Timestep | None`
  would make every `step()` narrow a value that is always set there.
- No float time: steps carry native `start`/`end` and a float `dt`, which covers the Monte-Carlo functions. For
  `linear` and `numeric` the native values are numbers anyway.
- `dt` is exact: the step size for `linear`/`numeric`, and whole days / 365 for `calendar`.
- The factories always return an iterator, rather than a tuple when fixed and a generator when open-ended: one return
  type, and no overloads. An iterator can only be used by one model; `list()` it to share it.
- Factories rather than classes with the old names: the names change anyway, and lowercase functions match what they
  are.

**Behaviour changes**
- `self.timestep.index` is 1 in the first step (1.x `timeline.index` was 0), so random streams keyed on it are
  shifted by one.
- The null timeline's `dt` is `NEVER` (it was 0.0).
- A seeder can't read `self.timestep`, as it's first called while the base class is being initialised. The tips page
  shows keeping the step index in a plain attribute instead.

**Follow-ups** — `test_bindings.py` no longer covers timelines beyond pass-through, as there's no C++ timeline left
to pin.

---

## 2026-11-08 — Upgrade locked dependencies; fix ty 0.0.99 diagnostics

**Why** — Refreshed `uv.lock` (pandas 3.0.6 → 3.1.1, ty 0.0.85 → 0.0.99, ruff → 0.17.4, among others). The newer ty
reported 45 diagnostics, including errors that would fail CI's `ty check`.

**What**
- Renamed `ty:ignore[unresolved-import]` to `ty:ignore[import-not-found]` (rule renamed in ty).
- `neworder/time.pyi`: added an `NDArray -> NDArray[bool]` overload of `isnever` ahead of the `SupportsFloat -> bool`
  one. `ndarray` defines `__float__`, so arrays matched the scalar overload and were typed as `bool`, which made
  `~isnever(arr)` look like a deprecated bool inversion.
- Examples: passed histogram `bins` as lists (`bins.tolist()`), since matplotlib's stubs reject `ndarray`, and cast
  `plt.hist`'s patches to `BarContainer` in the mortality example (one dataset always gives a single container).
- Bumped the ruff and ty pre-commit pins to the locked versions so local hooks and CI agree.

**Design decisions**
- Fixed the `isnever` stub rather than suppressing the warnings in examples: the type was wrong for every array caller.

**Follow-ups** — The pre-commit pins and `uv.lock` can drift again; dependabot only updates the lockfile.

---

## 2026-11-08 — Fix infection example colour lookup

**Why** — The infection example crashed in its visualisation with `AttributeError: 'int' object has no attribute
'rgba'`. pandas stores an `IntEnum` assigned to a column as plain `int64`, so `status.apply(lambda c: c.rgba)` received
`numpy.int64` values, not `Status` members. It went unnoticed because the example's OSM download (Overpass API) had
been failing first.

**What** — Removed the `Status.rgba` property and replaced it with a module-level `COLOURS` array indexed by status
value, so colours are looked up for the whole column in one step: `COLOURS[self.agents.status]`.

**Design decisions**
- Lookup table, not a fix to the property (`status.map(lambda s: Status(s).rgba)`): the column holds integers, so going
  back through the enum one row at a time was the wrong shape. Indexing an array is vectorised and simpler.
- Kept `Status` as an `IntEnum` for the named states, and kept `status` as an integer column instead of forcing `object`
  dtype to preserve the enum, so comparisons such as `status == Status.INFECTED` stay vectorised.
- Matplotlib colour names (`"white"`, `"red"`, ...) instead of RGBA tuples, for readability.

**Follow-ups** — osmnx's defaults (180s timeout, unlimited 55s retries on 429/504, logging off) make an overloaded
Overpass server look like a hang; the example could enable `ox.settings.log_console` and use a shorter timeout.

---

## 2026-11-07 — Timeline v2 review fixes: louder breaks, run-loop fixes (#134)

**Why** — A review of #134 found that several 1.x models would keep running under 2.0 but silently give different
results, plus two bugs: `halt()` called from `modify()` was ignored, and an equal-step `NumericTimeline` ending at
0 was rejected. Breaking changes should fail loudly wherever they can.

**What**
- `LinearTimeline(start=0, *, end=None)`: `end` is keyword-only, because `LinearTimeline(0, 1)` meant "open-ended,
  step 1" in 1.x and would otherwise silently become a one-step timeline. `bool` start/end values are rejected.
- `CalendarTimeline` `time` and `dt` are in **years** (ACT/365), not days. This replaces the 2026-10-31 decision.
  Annual rates are the common case, and with days a `hazard(rate * dt)` only failed for rates above ~1/365; below
  that, probabilities were silently 365x too high. `dt` is computed from the whole number of days in the step, so
  it's exact (e.g. 31/365). The people example goes back to using `timeline.dt` directly.
- `dt` is `NaN` (`NEVER`) before the first step for every timeline, including the fixed-step ones and `NoTimeline`.
  This replaces "the step size if fixed, NEVER otherwise". The base class handles index 0, and subclasses implement
  `_dt(i)`.
- `Model::run`:
  - `halt()` in `modify()` is honoured: no steps are taken and the timeline isn't advanced. `RUNNING` was
    previously set after `modify()`, which overwrote the halt.
  - Re-running a completed model raises `StopIteration` *before* calling `modify()`, which resolves the earlier
    follow-up. A model halted on its final step now finalises when resumed, where it previously raised.
  - It raises `ValueError` if a new model's timeline has already been advanced (e.g. by `list(timeline)`), and
    `RuntimeError` if `modify`/`step`/`check` advance it. Either would otherwise silently skip steps.
- Equal-step `NumericTimeline` uses an absolute tolerance when checking that `end` is a whole number of steps, so
  `start=-0.3, step=0.1, end=0.0` works. `NumericTimeline(times)` rejects `NaN`s.
- `CalendarTimeline` finds its number of steps by exponential/binary search rather than a linear scan. Daily
  steps over 200 years took 0.3s to construct.
- Docs: a "Migrating from 1.x" warning in the overview lists the remaining silent changes (`time`/`index` in
  `step()`, index-keyed random streams, `NumericTimeline(times)`). The overview and the bundled skill also
  document halt/resume and the advance guard.

**Design decisions**
- *Guard against advancing the timeline in model code* rather than only documenting it. It costs two extra
  `index` reads per step: the framework's own overhead is ~3.6µs/step, vs ~2.8µs without the guard and ~1.5µs in
  1.7, which is negligible next to any real `step()`. A fresh model on a partly-iterated timeline is rejected
  rather than run from there, as there's no known use for it and it was the cause of the confusing "already run
  to completion" error.
- *A model halted on its final step finalises when resumed.* Resuming means "continue from the next step"; when
  there isn't one, that's completion. The alternative was to keep raising `StopIteration`, which left no way to
  finalise.
- *Kept:* in the infection example, the initially infected recover one step earlier than in 1.x, because their
  infection time (index 0) is now measured against end-of-step indices. This is consistent with the clock
  semantics, so the example wasn't changed.

**Follow-ups**
- The MPI examples and tests weren't run (no MPI available locally). The infection example couldn't be run
  (no access to the Overpass API).
- #135 (nanobind): `PyTimeline`'s abstract-member check uses pybind11 internals and must be ported, and
  #135's `test_bindings.py` Timeline tests target the 1.x API.

## 2026-10-31 — Timelines as clock-like iterators over native steps, with float time; implementations moved to Python (v2.0.0)

**Why** — The old `Timeline` interface was large and awkward: `start`, `end`, `time`, `dt`, `at_end`, `_next`,
plus an `index` managed by the C++ base class. You couldn't iterate it outside a model, and `time` meant the start of
the step in `step()` but the end of it in `check()` (after the timeline had advanced). A Python subclass missing a
property recursed infinitely, which crashed the interpreter on Windows. Implementing the concrete timelines in C++
gained nothing, as each access goes through Python anyway. The design follows the maintainer's prototype (`t.py`,
not committed), with its bugs fixed (see below).

**What** — Breaking change, so a major version bump to 2.0.0.
- `neworder.Timeline` is still the C++ abstract base class, and `Model` still takes a `no::Timeline&`. Its
  abstract interface is now `index` (int), `time` and `dt` (both `double`), and `__next__`. It also has a concrete
  `__iter__` (which returns self) and a default `__repr__`. `start`, `end`, `at_end` and `_next` are gone.
- A timeline behaves like a clock:
  - Before the run, `index` is 0 and `time` is the start.
  - Each `__next__` takes a step: it increments `index`, sets `time` to the end of the step, and returns the step as
    a `(start, end)` tuple of the timeline's native type.
  - `time` and `dt` are always floats. `dt` is the length of the most recent step. Before the first step, `dt` is
    the step size if that's fixed, and `NEVER` otherwise.
- `NoTimeline`, `LinearTimeline` and `NumericTimeline` moved from C++ to `neworder/timeline.py`, alongside
  `CalendarTimeline`, and share a private `_SteppedTimeline[T]` base. Subclasses give each point's native value
  (`_point`) and its float value (`_numeric`):
  - `NoTimeline()`: native `(None, None)`; `time` is `NEVER` and `dt` is 0.
  - `LinearTimeline(start=0, end=None)`: native `int`, unit steps, open-ended if `end` is omitted.
  - `NumericTimeline(start=..., step=..., end=None)`: native `float`, equal steps, open-ended if `end` is omitted;
    `end - start` must be a whole number of steps.
  - `NumericTimeline(times)`: native `float`, explicit (possibly unequal) points.
  - `CalendarTimeline(start=..., step=..., end=None)`: native `date`, keyword-only arguments; `time` is days since
    the start and `dt` is in days.
  - Reprs follow the prototype (`<ClassName start=... end=... step=...>`).
- `Model::run` logs the start and calls `modify` (which sees the start). It then takes the first step, raising
  `StopIteration` if the timeline is already exhausted, and loops step → check → halt test → advance. `finalise`
  runs when advancing raises `StopIteration`. It calls the C++ virtuals directly, which made `PyAccessor`
  (in `src/Helpers.h`, previously used to read Python timelines' properties) redundant, so it was deleted.
- Examples updated:
  - Clock semantics: conway's glider is placed when `index == 1`, mortality looks up hazards by the age at the
    start of the step (`time - dt`), and markov_chain, membership and wolf_sheep take the start from `time` before
    the run.
  - people: `Population` takes the start date, to turn the float time back into a date for its labels, and converts
    `dt` from days to years for its annual rates.
  - option uses `NumericTimeline([0.0, expiry])`, and boids an open-ended `NumericTimeline`.
  - Every `LinearTimeline` lost its `nsteps`/step argument, and mortality's `max_age` is now an int.
  - Docs (overview, hello-world, tips) and the bundled skill were updated to match.

**Design decisions**
- *Native values via iteration; float `time`/`dt` on the timeline.* Model code always gets a number it can compute
  with, pass to the MonteCarlo functions and compare with `no.time.NEVER`/`FAR_FUTURE`/`DISTANT_PAST`. Those
  constants don't work with `None` or `date`. Rejected along the way:
  - `time` as the native `(start, end)` tuple: this broke those constants for `NoTimeline` and calendar dates.
  - No `dt` at all: models need step lengths for hazards and ageing.
- *Clock semantics, as in the prototype.* `index` counts the steps taken, and `modify` sees the start. Rejected: a
  timeline "positioned at" its first step, with index 0 during the first step. That kept existing `index` usage
  unchanged, but it made the state before the run (which `modify` sees) the end of the first step, which is
  unintuitive.
- *Calendar float time is in days* (the prototype's choice). This is exact, and doesn't build in a year-length
  convention: models with annual rates convert, as the people example does with ACT/365. Rejected: years since
  the start on an ACT/365 basis, and decimal years.
- *Bugs fixed relative to the prototype*:
  - `Numeric.dt` returned the start of the step.
  - `if self._end and ...` treated an end of 0 as open-ended.
  - Accumulating `+= step` drifted (floats) and lost the day of the month (calendar).
  - The calendar `dt` was computed as `t - (t - step)`, which is wrong for month-end steps.
  - `start + n * step` falling just short of `end` gave an extra step. Equal-step `NumericTimeline` now fixes the
    number of steps up front, and rejects an `end` that isn't a whole number of steps away.
  
  Points are computed from the start (`start + i * step`).
- *Halting doesn't advance the timeline.* A resumed run calls `modify` again, then continues with the next step.
- *Abstract members are enforced in the trampoline.* `NB_OVERRIDE` can't dispatch to Python properties: it fetches
  the property's value and then calls it. (`PYBIND11_OVERRIDE` had the same problem, and recursed on an
  un-overridden property, before the move to nanobind.) So `PyTimeline` checks that the subclass's attribute isn't
  the base's own, and raises a `TypeError` that names the missing member. Values are converted with
  `no::checked_cast`, so a property returning the wrong type raises e.g. `TypeError: MyTimeline.index must be an
  int64, got 'zero'`. This replaces the "pure virtual function" `RuntimeError` documented for 1.x custom
  timelines.
- *Missing members are reported on first use, not at instantiation.* `Timeline` can't be a true Python ABC:
  CPython's abstract check is in `object.__new__`, which nanobind's own `__new__` bypasses, so even `ABCMeta`'s
  `__abstractmethods__` is ignored. A metaclass whose `__call__` checks for overrides was shown to work, but
  wasn't adopted: the runtime `TypeError` (raised at the latest when a `Model` is constructed) plus static
  checking via the stub's `@abstractmethod`s is enough.
- *Rebased onto the nanobind migration (#135).* The design was written against pybind11 and ported when rebasing.
  `Timeline.__repr__` stays bound non-virtually, like `Model`'s methods. The timeline tests that #135 added to
  `test/test_bindings.py` were rewritten for the new interface (numpy scalars from `index`/`time`/`dt`, a
  wrongly-typed `index`, `__repr__`, identity and lifetime). The tests of the C++ timeline constructors' argument
  conversion were dropped, as `test/test_timeline.py` covers the Python constructors.
- *`LinearTimeline` is integer with unit steps*, so it's an exact counter. Anything else goes through
  `NumericTimeline`, so float handling lives in one place.

**Follow-ups**
- `Model::run` calls `modify` before discovering that a timeline is already exhausted (there's no way to peek at an
  iterator).
- A Python subclass that implements `index`/`time`/`dt` as methods rather than properties gets a cast error
  (`MyTimeline.time must be a float, got <bound method ...>`) rather than one that says they must be properties.
- The examples weren't run (per project preference); they're covered by `ty` only.

---

## 2026-11-08 — Cast errors, nanobind pin, and other review follow-ups (#135)

**Why** — The remaining findings from reviewing the nanobind port:
- A failed `nb::cast` is a bare `std::bad_cast`, so a seeder or a timeline's `index` returning the wrong type raised
  `RuntimeError: std::bad_cast`, without saying what or why.
- `nanobind>=3.1` had no upper bound, but the `ArrayLike` caster uses nanobind's detail API, which has changed between
  major versions. Free-threaded users always build from source, so a nanobind 4 could break their installs.
- Developer checkouts from before the move to scikit-build-core still contain an in-place `_neworder_core.*.so`. It's
  gitignored, and shadows the current build whenever the repo root is on `sys.path`.
- `unary_op` heap-allocated its result's shape on every call.
- The `stats` deprecation said "removed in v1.8", which is now this release.

**What**
- `no::checked_cast<T>(value, what)` (`src/Helpers.h`) raises `TypeError: <what> must be <type>, got <repr>`. It's used
  for both seeders (`Model` and `SplitMix64`) and `PyAccessor::get_as`, e.g. `MyTimeline.index must be an int64, got
  'zero'`. An out of range scalar argument to `SplitMix64` raises the same `TypeError` as an out of range array element.
- `nanobind>=3.1,<4` in `build-system.requires`.
- `docs/developer.md` warns about the old extension modules, and says how to remove them.
- `unary_op` builds the shape in a `std::array` (numpy allows at most 64 dimensions). It made no measurable difference:
  `hazard`, `isnever` and `stopping` on 1- and 9-element arrays were within noise of before (about 550-850 ns a call).
- The `stats` deprecation now says it will be removed in v2.0.

**Design decisions**
- *Fixed the call sites rather than adding a global translator for `nb::cast_error`.* A translator can't know the value
  or the expected type, so its message would be no better. The remaining `nb::cast`s convert values neworder produces
  itself, the thread id, or MPI's rank and size.
- *Kept the `unary_op` change* despite no measurable gain, as it's simpler than the vector it replaces.
- *Didn't delete the stale extension modules* from the checkout this was developed in; that's the developer's call.

**Follow-ups** — `no.log`, and logging in verbose mode, raise when `sys.stdout` is None (e.g. under `pythonw.exe`),
where pybind11's `py::print` did nothing.

---

## 2026-11-07 — Fix crashes on empty input, bad seeders and unencodable strings (#135)

**Why** — Reviewing the nanobind port found inputs that crash the interpreter or abort the process, rather than raise:
- `no.log("x\udc80y")` segfaulted, because `nb::str::c_str()` returns null for a string that can't be encoded as UTF-8
  (e.g. a lone surrogate, as in a filename decoded with `surrogateescape`). `std::to_string(nb::object)`, which builds
  error and log messages, had the same bug: `basic_string` from null, which is UB on MSVC.
- `mc.sample(n, [])` segfaulted, as `no::cumulative` read the first and last elements of an empty array.
- A seeder that raised, or returned something other than an int32, called `std::terminate` from the `noexcept`
  `MonteCarlo` constructor.
- `first_arrival`, `next_arrival` and `arrivals` read out of bounds on empty intensities, and `arrivals` with `n=0`
  underflowed its column count to -1.

**What**
- `log_obj` (`src/Module.cpp`) joins the arguments as Python strings and prints the result, so it behaves as `print`
  does: an unencodable string fails (or not) according to `sys.stdout`'s error handler, and a NUL no longer truncates.
- `std::to_string(nb::object)` (`src/Log.cpp`) falls back to `backslashreplace` when the string can't be encoded.
- `no::cumulative` raises `ValueError` on empty probabilities. The arrival processes raise `ValueError` on empty
  intensities (`validate_lambda_t`), and `arrivals(..., n=0, ...)` returns a `(0, 0)` array.
- `MonteCarlo`'s constructor is no longer `noexcept`. A seeder's result that isn't an int32 raises `TypeError` from
  construction or `reset()` (see the next entry), and exceptions raised by the seeder propagate unchanged.

**Design decisions**
- *Different fixes for the two string conversions.* `log` is Python's logging, so it should match `print`. That
  includes raising on a strict stdout, e.g. pytest's `capsys`, which is what pybind11's `py::print` did.
  `std::to_string` builds messages, often for an exception that's already being raised, so it must not fail itself.
- *Empty intensities are an error, not an empty result.* The processes sample from the largest intensity and
  extrapolate the final one, neither of which exists. `counts` still accepts an empty array, as it maps one intensity
  to one count.

---

## 2026-11-07 — Merge `NPArray` into `ArrayHelpers`

**Why** — The split didn't follow the code. `ArrayHelpers.h` declared the numpy helpers, but `NPArray.cpp` implemented
them, and `NPArray.h` only declared `isnever_a`, `logistic` and `logit`, which only `Module.cpp` uses.

**What** — `NPArray.cpp` renamed to `ArrayHelpers.cpp`, with the declarations from `NPArray.h` moved to
`ArrayHelpers.h` and `NPArray.h` removed. No functional change.

**Design decisions**
- *Kept the `ArrayHelpers` name*, as 7 files already include it, against 1 for `NPArray.h`.
- *Left the three functions in `ArrayHelpers`* rather than moving `isnever_a` to `Timeline` and the other two to a
  stats file. That would also be reasonable, but it's a separate change.

---

## 2026-11-07 — Inline the element functions in `make_array` and `unary_op`

**Why** — Both took a `std::function`, so every element of every array they filled cost an indirect call that the
compiler couldn't inline. They underlie most of `MonteCarlo`, `isnever`, `logistic`, `logit` and `df.unique_index`.

**What** — `src/ArrayHelpers.h`: the function is now a template parameter. Callers are unchanged. Measured per element
on 1M-element arrays, against a build of the same source using `std::function` (abi3, Linux, CPython 3.15):
`df.unique_index` -86%, `isnever` -39%, `ustream` -19%, `hazard(p, n)` -18%, `sample` -10%, `hazard(array)` -7%,
`stopping(p, n)` -6%, `stopping(array)` -3%. `ustream(1)` per call -3%. The random streams are unchanged
(`test_bindings.py`).

**Design decisions**
- *Corrected the comment on `NpArray`.* It said that wrapping memory allocated in C++ was several times slower than
  `np.empty`, but a stock nanobind module measured 318 ns vs 289 ns for a 1-element array, and the same for 1000
  elements. The 2026-11-06 entry's "about 3x slower" was relative to pybind11. The reason to keep `np.empty` is that the
  result owns its data, which `test_bindings.py` checks.
- *Kept `ArrayLike` and its caster.* The same module showed that a plain `nb::ndarray` argument converts dtypes and
  strided arrays, but rejects lists, tuples and `pandas.Series`, and takes `numpy.float64` ahead of a `double`
  overload. The tests require all of those to work.

---

## 2026-11-07 — Publish abi3 wheels, and stop publishing free-threaded wheels

**Why** — The release built a wheel per interpreter per platform: 12 wheels, with more for every new CPython. The
2026-09-02 investigation on `claude/publish-ci-abi3-builds-ely7z4` found abi3 impossible with pybind11, and nanobind
removes that blocker. One abi3 wheel per platform takes the total to 3.

**What**
- `CMakeLists.txt`: `STABLE_ABI` and `${SKBUILD_SABI_COMPONENT}`. `pyproject.toml`: `wheel.py-api = "cp313"`, which
  sets the wheel tag. Both are needed: `STABLE_ABI` alone gives an abi3 binary in a version-specific wheel, and
  `wheel.py-api` alone gives a version-specific binary in a wheel claiming to be abi3.
- `src/NPArray.cpp`: `PyTuple_SET_ITEM` isn't in the limited API, so it's now `PyTuple_SetItem`.
- `cp314t-*` removed from the cibuildwheel build list. `3.14t` stays in the `build-test.yml` matrix.

**Design decisions**
- *Accept the per-call cost.* Measured on Linux with CPython 3.15, against native wheels from the same source:
  `isnever(float)` unchanged, `mc.ustream(1)` +8%, `isnever` on a 1-element array +9%, `mc.sample(1, weights)` +11%,
  `Model.run` per step +2%, `mc.ustream(1000000)` -7%. 3.13 and 3.14 were similar, except `isnever` on a 1000-element
  array, which was +22% on 3.13 and +1-2% on the others. Neither that outlier nor the large `ustream` gain is explained.
- *Linked mode rather than split mode.* In nanobind 3.1, `abi3t` (PEP 803, Python 3.15+) is only available in split
  mode, so linked mode has no stable ABI for free-threaded builds. Split mode wasn't tried, as it adds a runtime
  dependency on a backend package.
- *Stop publishing free-threaded wheels, but keep testing free-threaded builds.* Publishing them would need 3 more
  wheels for each free-threaded version, which is the matrix growth this change removes. Free-threading is still a
  supported feature, so `FREE_THREADED` stays in `CMakeLists.txt` and the `3.14t` jobs in `build-test.yml` keep testing
  it, using a source build. `pip install neworder` on a free-threaded interpreter now builds from source too, which
  needs a C++ toolchain. Wheels built on 3.14t and 3.15t also passed the tests locally, on Linux.
- *Keep cp314 and cp315 in the cibuildwheel build list.* cibuildwheel 4.2.0 reuses the cp313 abi3 wheel for them rather
  than rebuilding, and still runs the tests on each, so the wheel is tested on every interpreter it's published for.
- *Tag `cp313` rather than `cp312`.* nanobind's linked mode targets the 3.12 limited API, but the tag matches
  `requires-python`.

**Follow-ups**
- Windows and macOS are untested: run `pypi-release.yml` manually to check those wheels, and the wheel reuse, before
  releasing.
- Revisit free-threaded wheels if nanobind's linked mode gains `abi3t`, or split mode becomes worthwhile.

---

## 2026-11-06 — Migrate from pybind11 to nanobind, and from setuptools to scikit-build-core

**Why** — nanobind produces smaller, faster bindings. Measured against pybind11 on Linux with CPython 3.14: the
extension is 362 KB rather than 754 KB, the wheel 181 KB rather than 304 KB, `Model.run` and scalar calls take 55-62%
of the time, and small array calls 44-70%. Functions on large arrays are about the same or faster, as most of their
time isn't in the bindings. The public Python API is unchanged, and the random streams match the pybind11 build's (see
`test_bindings.py`).

**What**
- `src/`: all bindings ported. Removed unused array helpers (`binary_op`, `at`, `fill`) and `no::log(handle)`.
- Build: `setup.py` and `MANIFEST.in` replaced by `CMakeLists.txt` and `[tool.scikit-build]`. `[tool.uv] cache-keys`
  means that a plain `uv sync` now rebuilds after a C++ change. The extension no longer lands in the repo root.
- `pybind11-stubgen` replaced by `nanobind.stubgen`, which no longer needs `--ignore-all-errors`. The hand-maintained
  stubs needed almost no change.
- `test/test_bindings.py` pins the boundary's behaviour and the values of the random streams. It was written first and
  passes on the pybind11 build, so it acted as the contract for the migration.
- Coverage: the extension is built with instrumentation in the project environment. Only the build step changed.
- `ty`: `allowed-unresolved-imports` covers the packages from the optional extras, so type checking works with or
  without them installed, and `neworder/mpi.pyi` no longer needs a `ty:ignore`.

**Behaviour changes**
- Fixed: strided input arrays were silently read as if contiguous, e.g. `isnever(a[::2])` read the wrong elements. They
  are now copied to a contiguous layout. Results change for any caller affected by the bug.
- Fixed: a `Timeline` subclass missing a property recursed unboundedly, crashing the interpreter on Windows. It now
  raises a `RuntimeError`. The trampoline's re-entrancy guard stops the recursion after one round.
- A subclass that doesn't call `super().__init__` now raises a `TypeError` (and a `RuntimeWarning`) when first used
  rather than on construction, as nanobind has no metaclass in which to check.
- A virtual method assigned to an instance (`m.step = f`) is only called from C++ if the class also overrides it, as
  nanobind checks the class for an override before fetching it from the instance. Overriding in a subclass is
  unaffected, and so is `check()`, which `Model::run` calls via Python.
- `bool` parameters (`relative`, `use_counter`, `verbose`, `checked`) now accept only `True`/`False`, so e.g. a
  `numpy.bool` raises `TypeError`. Values returned from `check()` and `at_end` still only need to be truthy.
- The wording of argument errors differs; the exception types don't.

**Design decisions**
- *scikit-build-core and CMake.* nanobind's documentation says that custom build systems are unsupported, so staying
  on setuptools would have meant owning that logic. `NOMINSIZE` is set, because nanobind optimises for size by default,
  which would also apply to the sampling loops in this module.
- *`no::ArrayLike<T>` for array arguments, rather than `nb::ndarray`.* `nb::ndarray` only accepts objects that already
  expose their memory, so it rejects lists and pandas objects. It's a small custom caster with the semantics of
  pybind11's `array_t`: the no-convert pass accepts only real `ndarray`s, and the convert pass uses `np.asarray`.
  Without the first restriction, a `numpy.float64` passed to `isnever` is claimed by the array overload through the
  buffer protocol, and it returns a 0-d array rather than a `bool`.
- *Returned arrays are allocated with `np.empty`, then passed through `nb::ndarray`.* The documented approach, wrapping
  memory allocated in C++ in a capsule, was tried first. Compared to pybind11 it was about 3x slower for 1-element
  arrays, which matters to case-based models that make a call per person, and the result didn't own its data. With
  `np.empty`, nanobind's export returns the original array, so it's a plain numpy array that owns its data. A custom
  array class and caster bypassing `nb::ndarray` altogether was also tried: it saves about 130 ns per call, by avoiding
  the import of the new array, but isn't worth owning a caster built on nanobind's `detail` API. numpy's C API would be
  faster still, but would add a binary dependency on numpy.
- *Base methods are bound non-virtually.* nanobind's trampolines only recognise a `super()` call if C++ made the outer
  call. So with `.def("check", &Model::check)`, an override called from Python that calls `super().check()` dispatches
  back to itself and runs twice. `test_bindings.py` caught this. As a result `Timeline.__repr__` is non-virtual, and
  each concrete timeline binds its own.
- *`Model::run` calls `check` via Python*, because `NB_OVERRIDE` casts the result strictly, and models commonly return
  a `numpy.bool` (e.g. the mortality example). Rejected: using nanobind's `detail::ticket` API to write a lenient
  trampoline.
- *Leak warnings are off by default* (set `NEWORDER_LEAK_WARNINGS` to enable), as a model can legitimately hold a cycle
  that the garbage collector can't see through C++, e.g. a seeder bound to the model. The test suite reports no leaks.
  `NB_DOMAIN` isolates this global flag from other nanobind extensions.
- *Split mode not adopted.* It adds a runtime dependency on a backend package. The stable ABI doesn't need split mode,
  and was adopted separately (see the 2026-11-07 entry).
- *One PR rather than class by class.* The two libraries don't share types, and `Model`, `MonteCarlo` and the array
  helpers cross every file.

**Follow-ups**
- The `infection` example fails with `AttributeError: 'int' object has no attribute 'rgba'` in its plotting code. It
  fails the same way on the pybind11 build, so isn't related to this change.
- Storing seeders as `nb::callable` rather than in a `std::function` would allow `tp_traverse` on `Model` and
  `SplitMix64`, so that Python can collect self-referencing models.
- If PR #134 (Timeline API v2) goes ahead it will need rebasing onto this. Its `overriding_self` uses pybind11
  internals, and `catch (py::stop_iteration&)` won't compile, as `nb::stop_iteration` is a function rather than a type.

---

## 2026-10-17 — Raise the coverage target to 95%

**Why** — The coverage-workflow change (2026-10-08, below) temporarily lowered the Codecov project target to 60%,
pending fixes before the next release. The aim is now 95%.

**What**
- `codecov.yml`: `parsers: cobertura: partials_as_hits: true`. Separate `cpp` and `python` flag statuses alongside
  the overall one, all with a 95% target (validated with Codecov's validator).
- `coverage.yml`: install the `geospatial` extra (its tests no longer need Overpass, #125). Pass
  `--exclude-unreachable-branches --exclude-throw-branches` to gcovr. Mirrored in `docs/developer.md`.
- New tests. These cover `Space` validation, `repr`, `extent`, BOUNCE movement, tuple `to_points` and
  `in_range`; `StateGrid` item access; and `GeospatialGraph` without a CRS and `from_point` (offline, with
  `ox.graph_from_point` monkeypatched). On the C++ side they cover 1d and non-square matrices and missing series
  categories in `df`; an invalid non-leading probability in `mc.sample`; verbose logging during a model run; a
  failed `check()` raised as an error under `warnings.simplefilter("error")`; and `freethreaded()`.
- `docs/overview.md`: added the missing `at_end` to the custom timeline table and class diagram. Added a warning
  that a Python `Timeline` subclass missing a property can crash with a stack overflow (see Follow-ups).
  Mirrored in the bundled skill (`neworder/skill/SKILL.md`), which also lacked `at_end` in its override list.
- Fixed `test_space2d`, whose "bounce" section moved points with the WRAP space, so BOUNCE was never exercised.
- Widened the `to_points` annotation of `Space.dists2`/`dists`/`in_range` to accept a tuple of coordinate arrays,
  which the code already handled.

**Design decisions**
- Most of the drop to 60.8% was Codecov's scoring, not missing tests. C++ line coverage was already 94.9%, but
  any line with an untaken branch counted as a miss. Scoring a local gcovr report the way Codecov does gives:
  originally about 53%; 89.7% with the gcovr exclusions; 95.3% with `partials_as_hits`; above that with the new
  tests. Python goes from 89% to about 98%.
- Timeline tests were dropped from this change. A test for incomplete Python timelines crashed CI on Windows by
  exposing the stack overflow below. A fix (binding the timeline properties on the concrete timelines rather than
  the abstract base) was tried on this branch and reverted, to keep this change to coverage only.
- Per-flag statuses, so that high C++ coverage can't hide a Python regression (or vice versa). Both flags use 95%
  rather than the `target: auto` suggested earlier for Python, which now has enough margin.
- Patch target raised from 75% to 90%, not 95%. A 95% patch target is too strict: in a small patch, one
  uncovered line can drop it below 95%. Some lines also can't be covered in the coverage job: MPI-only paths,
  and the closing braces and destructors that gcov reports as missed.

**Follow-ups**
- The remaining C++ misses: the mpi4py branch of `init_env` (runs only under MPI, which the coverage job doesn't
  use); `std::to_string(char)` and `no::log(const py::handle&)` in `Log.cpp` (no callers, could be deleted); and
  gcov artefacts on closing braces and defaulted destructors.
- `test_check_flag` in `test_module.py` leaves `no.checked(False)` set for any later tests. The new check-failure
  test re-enables it explicitly.
- Known bug, documented rather than fixed for now: a Python `Timeline` subclass that doesn't implement `time`,
  `start`, `end`, `dt` or `at_end` recurses when that property is accessed. `PYBIND11_OVERRIDE_PURE` looks the
  override up on the instance and finds the base class property again. On Linux this ends in a `RuntimeError`,
  but on Windows it overflows the stack and crashes. Fixing it would also let the `Timeline.h` trampolines and
  `CalendarTimeline` stepping be tested.

---

## 2026-10-07 — Make geospatial tests independent of the Overpass API

**Why** — `test_geospatial` downloaded a road network from the public Overpass API in every CI matrix job (12
concurrent requests). Overpass rate-limits and is sometimes overloaded, so CI failed intermittently with
timeouts and refused connections (e.g. on #124).

**What**
- Committed the network the test used (drive network within 2km of Coniston, 69 nodes, 110KB) as
  `test/coniston.graphml`. The tests now build `GeospatialGraph` from it offline and cover every method: CRS
  projection, nodes/edges, `edges_to`/`edges_from`, `shortest_path`, `subgraph` and `isochrone`.
- Kept one live test of `from_point`. It only runs when `NEWORDER_NETWORK_TESTS` is set (on the ubuntu/3.13 job
  in `build-test.yml`), and skips rather than fails on network errors and Overpass error responses.
- The new `isochrone` test found that it used geopandas' deprecated `unary_union`. It now uses
  `shapely.union_all`, which needs no bump to the geopandas minimum version (`union_all()` needs geopandas 1.0).

**Design decisions**
- Rejected: caching osmnx responses in GitHub Actions (a cold cache still means 12 concurrent requests),
  retries/longer timeouts (failures become rarer, not impossible), and skipping on failure without a fixture
  (CI would stay green while the code went untested).

**Follow-ups** — The infection example still queries Overpass. CI doesn't run it, but it can fail for users for
the same reason.

---

## 2026-10-17 — Deprecate `stats.logistic` and `stats.logit` (#125)

**Why** — `neworder.stats.logistic` and `logit` duplicate `scipy.special.expit` and `scipy.special.logit`, and
scipy is already a runtime dependency.

**What** — Deprecated both functions ahead of their removal, without touching the C++ bindings.
`neworder/stats.py` replaces `stats.pyi`: a pure-Python module whose `logistic` and `logit` call
`_neworder_core.stats` and are decorated with `warnings.deprecated` (PEP 702). That one decorator emits a
`DeprecationWarning` pointing at the caller on each call, and makes `ty`, pyright and IDEs flag the calls. Each
docstring has a "Deprecated" admonition giving its scipy equivalent, shown on the API page.

**Design decisions**
- Deprecate the functions, not the module. A module-level `__getattr__` (PEP 562) that warned on any access to
  `neworder.stats` was tried first and dropped: it warned on access rather than use (including `hasattr` and
  star imports), and changed how `stats` is imported.
- A Python module rather than wrapping the extension functions in place. `warnings.deprecated` can't decorate a
  pybind11 function directly (it sets `__deprecated__` on it, and builtins have no `__dict__`). An earlier
  version used a custom `functools.wraps` helper to replace the functions in `_neworder_core.stats`, plus
  `typing_extensions.deprecated` in the stub. Dropping Python 3.12 (#126) made `warnings.deprecated`
  available, so runtime and static deprecation now come from one decorator, with no helper, no monkeypatching
  of the extension module, and no `typing_extensions` import.
- Python rather than C++ (`py::warnings::warn` in each binding): no rebuild, and the bindings are left
  unchanged until they are removed.

**Follow-ups** — Remove the functions, along with the C++ bindings, `neworder/stats.py`, `test_stats.py` and
the `docs/api.md` section, in a later release.

---

## 2026-10-08 — Drop Python 3.12; add 3.15 to CI (#126)

**Why** — 3.12 support was due to be dropped. Doing it now unblocks `warnings.deprecated` (3.13+) for the
`stats` deprecation without a `typing_extensions` dependency, and lets CI show how ready the dependencies are
for 3.15.

**What**
- `requires-python >= 3.13`; classifiers drop 3.12 and add 3.15; cibuildwheel builds cp315 wheels and no longer builds cp312 wheels.
- CI matrix: 3.13, 3.14, 3.14t, 3.15, all gating and all with the geospatial extra. The geospatial extra now
  requires `shapely>=2.2.0`, the first release with cp315 wheels.
- Removed code that only existed for 3.12: the `sys.version_info.minor > 12` guards in the multithreaded
  examples, and a `ty: ignore` on `types.CapsuleType` (new in 3.13) in `__init__.pyi`.
- Fixed the option example, which reported "python FT" inverted (`sys._is_gil_enabled()` without `not`).
- Bumped GitHub Actions: `actions/checkout` v6 → v7 in all workflows, `astral-sh/setup-uv` v8.1.0 → v10.2.0.
- Removed the `draft-pdf.yml` workflow, which built the JOSS paper PDF on every push. The `paper/` sources are kept.
- Moved the docs toolchain (`zensical`, `mkdocstrings-python`, `requests`) from the `dev` dependency group to a new
  `docs` group. zensical only ships `abi3` wheels, which free-threaded builds can't use, so every 3.14t CI job
  compiled its Rust core from source. No CI job builds the docs, and ReadTheDocs installs from
  `docs/requirements.txt`. Build the docs locally with `uv run --group docs zensical serve`.

**Design decisions**
- 3.15 started out experimental (`continue-on-error`, no geospatial extra) because `shapely` had no cp315
  wheel. shapely 2.2.0 added them, and with it every 3.15 job passed, so 3.15 is now gating. The `shapely`
  floor was raised to 2.2.0, rather than relying on the lockfile alone, so a re-lock or a plain pip install
  can't select a version without 3.15 support.
- cp315 wheels are built for release, since 3.15 is tested in CI. cp315t wheels are not: cibuildwheel only
  builds the interpreters tested in `build-test.yml`, and 3.15t isn't in that matrix yet.

**Follow-ups** — Add 3.15t to the CI matrix and cp315t wheels to the release build, together.

---

## 2026-10-08 — Coverage workflow: uv, codecov-action and Python coverage

**Why** — `coverage.yml` was never converted to uv when the rest of CI was. It also uploaded with Codecov's
deprecated bash uploader, and collected no Python coverage, despite `AGENTS.md` saying it did.

**What**
- The workflow uses `setup-uv`. After `uv sync --dev`, it rebuilds the extension in place with
  `CXXFLAGS=--coverage` (`setup.py build_ext --inplace`, with setuptools supplied by `--with`). Tests then run
  with `uv run --no-sync`, so uv doesn't replace the instrumented build.
- C++ coverage is turned into Cobertura XML with `gcovr` (run via `uvx`). Python coverage comes from
  `pytest-cov` (via `--with`).
- Both reports are uploaded with `codecov/codecov-action@v7.1.1`, flagged `cpp` and `python`, with its file
  search and plugins disabled so only these two reports are sent.
- Updated the "Test Coverage" section of `docs/developer.md` with the commands to reproduce it locally.

**Design decisions**
- An in-place rebuild, rather than `CXXFLAGS=--coverage uv sync`: uv builds in a temporary directory that is
  deleted afterwards, and gcov writes `.gcda` files next to the object files, so a uv build leaves no coverage
  data. Verified locally: the in-place build gives 94.9% C++ line coverage, and Python coverage is 78%.
- `gcovr` and `pytest-cov` are run with `uvx`/`--with` rather than added as dev dependencies, since only the
  coverage workflow needs them.


**Follow-ups (before the next release)** — The first CI run reported 60.8% (C++ 52.8%, Python 78.2%), against
95.6% before, so the project target in `codecov.yml` was temporarily lowered from 90% to 60%. The drop has two
causes:
- gcovr's Cobertura report includes branch data, which the old uploader never sent. Codecov scores any line with
  an untaken branch as partial, not hit, and C++ has many (exception paths, inlined std library code). Setting
  `parsers: cobertura: partials_as_hits: true` in `codecov.yml` restores line-coverage scoring. gcovr's
  `--exclude-unreachable-branches --exclude-throw-branches` would reduce the noise in the branch data.
- Python coverage is now included, and is lower (78%; the coverage job doesn't install the geospatial extra, so
  `geospatial.py` counts as uncovered). Use separate per-flag project statuses (`flags: [cpp]` with 90%,
  `flags: [python]` with `target: auto`), and install the geospatial extra once its tests no longer need
  Overpass (#125).
Then restore the 90% target. The per-flag config above was validated with Codecov's validator, but not yet run
in CI.

---

## 2026-10-07 — Document submodules in the API reference

**Why** — The API page rendered only top-level members of `neworder`; the `time`, `mpi`, `stats` and `df`
submodules were missing entirely.

**What** — Added explicit `::: neworder.<submodule>` directives to `docs/api.md`. Fixed three further gaps in
the hand-maintained stubs that kept members out of the rendered docs:
- `stats.logistic` was declared as three `@overload`s with no implementation. Griffe silently drops overloads
  that have no implementation signature. The overloads were also wrong: the `(x, k)` form doesn't exist, and a
  positional second argument is `x0`. They are replaced by the real binding signature
  `logistic(x, x0=0.0, k=1.0)`.
- `time.isnever` really is overloaded in C++ (scalar/array), so an implementation signature was added after
  the overloads, purely so griffe picks it up.
- Constants in `time` and `mpi` had no docstrings, so mkdocstrings (`show_if_no_docstring = false`) hid
  them. Added attribute docstrings.

**Design decisions**
- Explicit directives rather than `show_submodules = true`: the latter also renders internal modules
  (`skill_cli`, `logging`) and duplicates classes already re-exported at the top level.
- Attribute docstrings rather than a per-directive `show_if_no_docstring`, so constants get descriptions.

Also merged a fresh `pybind11-stubgen _neworder_core` run into `neworder/__init__.pyi`. It added the
`Model.run()` instance method, removed the non-existent `Timeline.nsteps`, and brought in the fuller
`SplitMix64` docstrings. The generated output was not copied over wholesale, because it would have lost the
hand-typed signatures (`SplitMix64` args, pandas types in `df`) and the pure-Python re-exports.

`mpi.COMM` is now typed `mpi4py.MPI.Intracomm | None`. This is accurate, because it is `None` without
`mpi4py`, but it broke type checking wherever `COMM` is used. All of those call sites are already guarded at
runtime by `SIZE > 1` or an MPI-only entry point, so each function that uses `COMM` now starts with
`assert ... COMM is not None` to narrow the type. Alternatives considered: keeping `COMM` non-optional (wrong
in serial mode), or casting (hides real misuse).

**Follow-ups** — Regenerating stubs with `pybind11-stubgen` overwrites these manual edits; reapply them, or
consider a griffe extension that promotes implementation-less overloads.

Also corrected the `SplitMix64` docstrings in `Module_docstr.cpp` and the stub. The old `__init__` text claimed
the seeder was called on construction and on `reset()`. In fact the constructor only stores it, it is called on
every `uarray()`/`raw()` call, and `reset()` only zeroes the counter. `raw()` also increments the counter.

Rewrote the stub-regeneration section of `AGENTS.md`. It had a flag that current pybind11-stubgen rejects as
ambiguous (`--ignore-invalid all`, now `--ignore-all-errors`), a wrong output path, and a non-existent
`stubPackages` setting. It now documents the merge-don't-copy workflow, the manual corrections to preserve, and
that `uv sync` drops extras not named on the command line. Brought `docs/developer.md` up to date in the same way. It was
still on `pip install -e .[dev]`, and now covers the `uv` workflow and quality gates, building the docs, and how
the hand-maintained stubs relate to `pybind11-stubgen` output.

---

## 2026-10-06 — Embed example videos via the mkdocs-video shim

**Why** — The boids and infection example pages embedded their animations with raw `<video>` HTML and
inline flex styles, which is verbose and duplicates playback attributes on every page.

**What** — Enabled Zensical's built-in emulation of the `mkdocs-video` plugin (configured in
`zensical.toml`: native `<video>` rather than iframe, webm, autoplay, muted) and replaced the raw HTML
with `![type:video](...)` markup. The two boids videos sit side by side using the theme's `grid` class.
Bumped `zensical` to 0.0.68.

**Design decisions**
- Zensical doesn't run MkDocs plugins; it maps the `mkdocs-video` config onto its own media extension,
  so no extra docs dependency is needed.
- The shim wraps each video in a block-level `div.video-container`, so styling the image via attr_list
  can't place two videos on one row — a `grid` (theme built-in, collapses to one column on narrow
  screens) is used instead of hand-rolled flex styles.
- `video_type = "webm"` is required: the shim defaults to `mp4` and would mislabel the sources.

**Follow-ups** — The shim nests its `div` inside a `<p>`, which is invalid HTML; browsers tolerate it
but may add a little vertical space.

---

## 2026-09-29 — Typed API reference in the docs

**Why** — The API reference page rendered class and method docs from the type stubs, but
mkdocstrings-python hides annotations by default, so signatures appeared as e.g.
`sample(n, cat_weights)` with no types. The docs build on ReadTheDocs would also have failed, since
`mkdocstrings-python` was missing from `docs/requirements.txt`.

**What** — Enabled `show_signature_annotations`, `separate_signature` and `signature_crossrefs`
(plus google docstring style, source member order and root headings) for the mkdocstrings python
handler in [zensical.toml](zensical.toml). Added `mkdocstrings-python` to the pinned doc
requirements generated by `write_requirements()` in [docs/macros.py](docs/macros.py) and
regenerated [docs/requirements.txt](docs/requirements.txt), which also bumps the `zensical` pin to
the version in the dev env.

**Design decisions**
- Rely on griffe's static analysis of the `.pyi` stubs rather than importing the compiled
  extension — the RTD build installs only `docs/requirements.txt` and never builds `neworder`.
  Verified by building in a clean venv with only those requirements.
- Escaped the citation numbers in [docs/references.md](docs/references.md) (`\[1\] [text](url)`).
  mkdocstrings pulls in the autorefs plugin, which claims `[1] [text]` as an unresolved
  reference-style cross-ref and renders the line as literal text; previously Python-Markdown fell
  back to parsing the inline link. An ordered list would also work but changes the page's look.

**Follow-ups** — Some numpy types in the stubs render verbosely (e.g.
`Annotated[ArrayLike, float64]`); tidying them would improve both the docs and IDE hints. Only
`Model`, `MonteCarlo` and `SplitMix64` are currently documented in [docs/api.md](docs/api.md).

---

## 2026-08-31 — Installable agent skill (`neworder-skill`)

**Why** — Coding agents write *neworder* models much more reliably when given a compact,
purpose-built reference than when left to infer the framework's shape from source, stubs or a
partial reading of the docs site. The recurring failures are all framework-specific and cheap to
prevent in writing: forgetting `super().__init__(timeline, seeder)`, mutating the result of
`no.df.transition` in place instead of assigning it back, comparing against `NEVER` with `==`,
looping over agents in python instead of vectorising, and letting `check()` return different
values per MPI process. Bundling the reference in the package makes it available in any downstream
project that installs *neworder*, not just in this repo. Modelled on the equivalent change in
[virgesmith/xenoform-rs#24](https://github.com/virgesmith/xenoform-rs/pull/24).

**What** — Added [neworder/skill/SKILL.md](neworder/skill/SKILL.md), an agent skill covering the
model lifecycle, the four timeline types, the `MonteCarlo` and `SplitMix64` engines and their
seeding strategies, the `neworder.df` operations, spatial domains, MPI patterns and the framework's
recurring pitfalls. Added [neworder/skill_cli.py](neworder/skill_cli.py), a `neworder-skill` console
script (registered under `[project.scripts]`) with `--install [PATH]` / `--remove [PATH]`, default
`PATH=.agents`; `package_data` in [setup.py](setup.py) ships the skill in the wheel and sdist. Tests
in [test/test_skill_cli.py](test/test_skill_cli.py), a new
[docs/agent-skill.md](docs/agent-skill.md) page with a nav entry in [zensical.toml](zensical.toml),
and a pointer to it from [README.md](README.md).

**Design decisions**
- **The skill points at the documentation site rather than restating it.** Its header table links
  to the overview, tips, examples and developer pages at `neworder.readthedocs.io/en/stable/`, and
  the body is deliberately a summary an agent can hold in context, not a second copy of the docs
  that would drift out of sync with them. That is also why the user-facing documentation for the
  feature is a docs-site page, with only a two-line pointer in `README.md` — the README is
  inlined into `docs/index.md` via `include_snippet`, so anything longer would duplicate the new
  page on the site's front page.
- **Install by symlink where possible, copy where not.** A symlink to `neworder/skill/` inside the
  installed package always matches the version in use, with nothing to keep up to date — the
  approach `xenoform-rs` took, and Streamlit's `streamlit skills` before it. That repo could stop
  there; this one cannot, because CI (and the classifiers) cover Windows, where `symlink_to` needs
  developer mode or elevation. So `_link_or_copy` catches `OSError` and falls back to
  `shutil.copytree`, and `--install` over an existing copy refreshes it rather than reporting it
  up to date, since a copy — unlike a symlink — goes stale on upgrade. The link target is relative where a
  relative path exists, so a symlinked skill survives the project being moved; on Windows there is
  no relative path between different drives (`os.path.relpath` raises `ValueError`, caught by CI
  with the package on `D:` and the project's temp dir on `C:`), so the link target falls back to
  an absolute path there.
- **Ownership is checked before anything is overwritten or deleted.** A symlink is ours if it
  resolves to the bundled directory; a directory is ours only if every entry is a file whose name
  we ship. Anything else — a user's own file, directory or foreign symlink at the target — is left
  untouched and the command exits 1. The subset rule means a copy with a user-added file in it is
  no longer considered ours, which is the safe direction to err in.
- **`.agents/skills/<name>` as the default target, overridable by `PATH`.** Matches the sibling
  repo and the emerging cross-harness convention, and `neworder-skill --install .claude` covers a
  specific harness without needing per-harness detection logic in the installer.
- **A console-script entry point rather than a loose script.** `Path(__file__).parent / "skill"`
  resolves from wherever `neworder` is importable, so the script naturally targets the environment
  it is invoked from with no `.venv` detection. Verified against a built wheel and sdist that
  `package_data` ships `neworder/skill/SKILL.md` and that the entry point is registered — this
  matters because `[tool.cibuildwheel]` runs the test suite against the installed wheel, so
  `test_skill_cli.py` would fail there if the skill were not packaged.

**Follow-ups** — The skill's content is maintained by hand and can drift from the docs site; the
type-level details it quotes (method signatures, timeline properties) are the most likely to go
stale, and nothing checks them. If it proves worth it, the tables could be generated from the
stubs, or a test could assert that every method named in the skill exists on the corresponding
class. `--install` targets one directory at a time; multi-harness install (writing both `.agents`
and `.claude`) is deferred until someone asks for it.

---

## 2026-08-01 — SplitMix64.raw (#120)

**Why** — `SplitMix64` exposed only `uarray`, so the underlying 64-bit hashes were unreachable, and two things needed them. Seeding an external generator previously meant `MonteCarlo.raw()` or the `as_np` bitgen adapter, both of which tie the external generator to the sequential mt19937 stream — what it gets depends on how many draws were taken before it, so there was no order-independent way to initialise one. And variates `uarray` cannot express (a uniform integer over an arbitrary range, say) need the hash itself, not its image in `U[0,1)`.

**What** — added `SplitMix64.raw(*keys)` in [src/SplitMix64.cpp](src/SplitMix64.cpp): identical to `uarray` in argument handling, output shape and counter semantics, returning the raw 64-bit hashes as `int64` instead of mapping them onto `U[0,1)`. Binding, docstrings, the stub entry in [neworder/\_\_init\_\_.pyi](neworder/__init__.pyi) and 9 tests. [docs/tips.md](docs/tips.md) gains two sections: re-seeding a model's `MonteCarlo` per timestep, so a step's draws no longer depend on how many were taken before it, and seeding a numpy bit generator.

**Design decisions**
- **Keyed like `uarray`, not as numpy's `ISeedSequence`.** The obvious alternative was to implement numpy's protocol — `generate_state(n_words, dtype)` — which would make `np.random.PCG64(sm)` work directly. Rejected because it is shaped by numpy rather than by how models key their draws: it takes a word *count* and derives words from an index, so it cannot be keyed on person id / process id / timestep the way `uarray` is. Seeding an external generator is only one use of the hashes. `raw(np.arange(4), MODEL_ID).view(np.uint64)` covers the numpy case anyway, at the cost of an explicit view.
- **int64 only, no dtype parameter.** SplitMix64 is a 64-bit generator; a dtype selector would buy callers an `.astype` at the price of a real validation surface — it would have to check `kind()` *and* itemsize, since `float64`/`int32` share itemsizes with `uint64`/`uint32` and would otherwise be silently reinterpreted. Signed rather than unsigned for consistency with `hash64`, which already returns `int64`.
- **Extracted `mix_keys`/`hash_at` rather than duplicating the argument parsing.** `raw` and `uarray` share the salt computation, shape derivation and per-element hashing, differing only in the final mapping, so the two cannot drift. `test_raw_matches_uarray` pins the relationship `uarray == (raw viewed as uint64 >> 11) * 2**-53`; the pre-existing `test_uarray_known_values_*` goldens are untouched and still pass.
- **`MonteCarlo::reset()` is no longer `noexcept`.** It invokes a Python callable, so any exception — e.g. a seeder returning a value that doesn't fit `int32` — terminated the process via `std::terminate` instead of propagating. Surfaced by the per-timestep seeding pattern the new docs recommend, which calls `mc.reset()` inside `step()`. Note the constructor is still `noexcept` and carries the same exposure.
- **`MonteCarlo` is not deprecated, and `docs/tips.md` keeps its neutral "when to use which" note.** `hazard`/`stopping`/`arrivals`/`sample`/`counts` have no `SplitMix64` equivalent, so for non-uniform sampling `MonteCarlo` is not a legacy path but the only option — telling users it is deprecated would point them at a replacement that cannot do the job. The two engines are complementary rather than successive: choose by whether draws must be order-independent, not by which is newer.
- [docs/index.md](docs/index.md) carries an unrelated drive-by wording fix to the examples section.

**Follow-ups** — `MonteCarlo`'s non-uniform samplers have no `SplitMix64` equivalent, so the class cannot be retired until those are ported.

If the `ISeedSequence` interop is ever wanted, implement `generate_state` as a thin wrapper over `raw` keyed on the word index, and add `ISeedSequence.register(SplitMix64)`. The constraints, all verified: `BitGenerator.__init__` isinstance-checks `ISeedSequence` and *then* calls `generate_state` by name, so registering without the method turns a clear `TypeError` at construction into an `AttributeError` thrown from inside numpy — registration is a type assertion only and carries no behaviour. A pybind11 extension type cannot inherit the Python-side ABC, so virtual-subclass registration is the only route. Each bit generator asks for a different shape of state — `PCG64` 4×uint64, `Philox` 2×uint64, `SFC64` 3×uint64, `MT19937` 624×uint32. And numpy's `SeedSequence` rejects negative entropy (`ValueError: expected non-negative integer`), which is why the docs view `raw`'s output as `uint64` first — roughly half of it is negative.

**Follow-up (larger): unify the seeding interface across `MonteCarlo` and `SplitMix64`.** The two engines are seeded through incompatible interfaces, all of them scalar and none wide enough:

- `MonteCarlo` takes `std::function<int32_t()>`, `seed()` returns `int32_t`, and the built-in strategies (`deterministic_independent_stream` and friends) are `int32_t` — [src/MonteCarlo.h](src/MonteCarlo.h).
- `SplitMix64` takes `std::function<int64_t()>` and casts the result to `uint64_t` for the salt — [src/SplitMix64.h](src/SplitMix64.h).
- `Model` forwards its `py::function` seeder to `MonteCarlo` only, so a model's `SplitMix64` instances have to be seeded separately by hand.

The cost today: `int32` is the narrowest link, so [docs/tips.md](docs/tips.md) has to tell users to `astype(np.int32)` the output of `raw` (wrapping, and flagged as a `ty` diagnostic at the call site) to drive a `Model` seeder; and a single `int32` is a weak `mt19937` seed.

What a unified interface should provide:

1. **One width and one signedness** — `uint64` words throughout, so no seeder value is unrepresentable and nothing narrows at a boundary.
2. **Non-scalar seeds** — a seeder may return a sequence/array of words, not just one. `mt19937` has 624×uint32 of state and cannot be seeded to full strength from one scalar (`std::seed_seq` or a `SeedSequence`-derived spread is the route); `SplitMix64`'s salt is one word by construction but should accept a vector and fold it.
3. **`np.random.SeedSequence` compatibility, both directions** — accept a `SeedSequence` (or its entropy) as a seed, and emit words that seed a numpy bit generator without the `.view(np.uint64)`/`.astype` dance, subject to the `ISeedSequence` constraints above. Note numpy declares its seed parameter as the concrete `SeedSequence`, not the ABC, so *no* third-party implementation can satisfy it statically — suppression at the call site is unavoidable.

**The blocker is reproducibility, not design.** Changing what `MonteCarlo` derives from a given seed changes every existing model's stream, which is the one guarantee the framework sells — widening it to `std::seed_seq` was tried once and abandoned for exactly this reason. Any unification must either keep the current scalar `int32` → `mt19937` path bit-exact and use the wider path only for new-style (vector / `SeedSequence`) seeds, or land as an explicit opt-in with the golden values in `test_mc.py` regenerated in the same commit.

**Note: `AGENTS.md` is misleading about stubs**, on two counts found while updating them here:

- The stubgen command is stale — `pybind11-stubgen --ignore-invalid all` is now ambiguous (`--ignore-invalid-expressions` / `--ignore-invalid-identifiers`); `--ignore-all-errors` is the current spelling. A full regen also drops the hand-added `CalendarTimeline` and the condensed docstrings, so stubs are better hand-merged than regenerated wholesale.
- **`stubs/` is gitignored and is not what `ty` reads.** The authoritative, version-controlled stub is [neworder/\_\_init\_\_.pyi](neworder/__init__.pyi); editing `stubs/_neworder_core/__init__.pyi` alone changes nothing (verified by perturbing a signature there and observing `ty`'s output was unaffected). `AGENTS.md` describes a `stubs/_neworder_core-stubs/` layout and a `stubPackages` setting, neither of which exists — there is no `[tool.ty]` section in `pyproject.toml` at all.

## 2026-07-30 — remove Docker image (#118)

**Why** — issue #118: the Docker image added unnecessary complexity (a `Dockerfile` to maintain, a manual rebuild-and-push step tacked onto every release) for a job the release CI already does — packaging and uploading the examples archive as a GitHub release artifact.

**What** — deleted `Dockerfile` and `.dockerignore`. Removed the "Docker" section and the manual docker-push release-checklist step from [docs/developer.md](docs/developer.md), the docker-pull instructions from [docs/index.md](docs/index.md) and [docs/examples/src.md](docs/examples/src.md) (both now just point at the release examples archive), and the `Dockerfile` layout entry / manual-rebuild rule from [AGENTS.md](AGENTS.md).

**Design decisions**
- Left `paper/paper.md` untouched — it's the JOSS-published paper (frozen since the 2021 review, per its git history), a historical record of what was true at publication rather than living documentation.

**Follow-ups** — the `virgesmith/neworder` image on Docker Hub itself is out of scope for this repo change; it will simply stop being updated.

## 2026-07-26 — markov_chain example: split into MarkovChain + ConditionalMarkovChain siblings (uncommitted)

**Why** — the group-conditional comparison (see the entry below) had been bolted onto `MarkovChain` behind an optional `group_transition_matrices` constructor argument, which meant `__init__`, `step()`, and `finalise()` all carried an `if self.group_transition_matrices is not None:` branch, and `mixed_stationary_distribution()` needed a defensive `assert` for a case that should have been unreachable by construction. Flagged (correctly) as not liking the shape of that file.

**What** — went through two shapes before landing on the final one. First cut: `ConditionalMarkovChain(MarkovChain)`, calling `super().step()` for the inherited pooled simulation and layering a second `state_conditional`/`summary_conditional` pair alongside it in the same instance. Flagged again (correctly) - `state`/`summary` vs `state_conditional`/`summary_conditional` inside one instance is the same "two things glued into one class" smell as the original flag argument, just moved down a level. Final shape: `MarkovChain` and `ConditionalMarkovChain` are now siblings, both extending a new `MarkovChainBase(no.Model)` that owns exactly the shared scaffolding (population setup, `_state_counts()`, `finalise()`'s `t`-column cleanup, the `_stationary_distribution()` static helper) - nothing pooled-matrix-specific lives there. Each subclass has exactly one `state` column and one `summary`, using the inherited names directly, and provides its own `step()` and its own matrix validation. `model.py` now constructs and runs *two* separate model instances (each needs its own `LinearTimeline` - they're stateful iterators, sharing one across two `no.run()` calls raises `StopIteration` on the second run) and passes both into `visualisation.show(pooled_model, grouped_model)`, which no longer needs an `isinstance` check - it just takes an optional second argument.

**Design decisions**
- Rejected the `ConditionalMarkovChain(MarkovChain)` subclass shape (first cut above) specifically because it required two parallel state columns/summaries per instance to keep the pooled and grouped simulations from clobbering each other - a sign that one instance was doing two models' worth of work. Splitting into two real instances removes the need for the `_conditional`-suffixed pair entirely.
- Extracted `MarkovChainBase` rather than either (a) duplicating population/state-counting setup across two unrelated classes or (b) keeping the parent-child relationship - there's genuine shared logic (not just superficially similar lines), so a common base earns its keep here without over-abstracting.
- Two separate `no.Model` instances necessarily means two independent RNG streams (each seeded via `deterministic_identical_stream`, but consumed independently rather than interleaved within one shared stream as in the first cut) - the logged equilibrium proportions are no longer bit-for-bit identical to earlier runs. Verified this is expected, not a regression: the grouped run's simulated proportions still track its own analytic (mixed) prediction closely and still diverge from the pooled analytic prediction, which is the property that actually matters. Regenerated `docs/examples/img/markov-chain.png` to match.

**Follow-ups** — none.

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

