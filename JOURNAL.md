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

