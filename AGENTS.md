# Agent Guidelines for `neworder`

This file instructs AI agents acting as developer, reviewer, and QA for this repository.

## Project Overview

`neworder` is a dynamic microsimulation framework implemented as a Python extension module. The core is written in C++ (in [src/](src/)) and exposed to Python via [pybind11](https://pybind11.readthedocs.io/). The Python package lives in [neworder/](neworder/), with type stubs in [stubs/](stubs/), tests in [test/](test/), documentation in [docs/](docs/), and runnable examples in [examples/](examples/).

The documentation site is at [neworder.readthedocs.io](https://neworder.readthedocs.io). Source is in [docs/](docs/) (MkDocs). ReadTheDocs builds automatically on pushes to `main` ("latest") and on `v*` tags ("stable").

## Toolchain

| Tool | Command |
|------|---------|
| Package manager | `uv` |
| Linter / formatter | `ruff` (`uv run ruff check`, `uv run ruff format`) |
| Type checker | `ty` (`uv run ty check neworder examples test`) |
| Tests | `uv run pytest` |
| MPI tests | `mpiexec -n 2 uv run pytest` |
| Install dev deps | `uv sync --dev` |
| Install with geospatial | `uv sync --dev --extra geospatial` |
| Install with MPI (system) | `uv sync --dev --extra parallel-native` |
| Install with OpenMPI | `uv sync --dev --extra parallel-openmpi` |
| Install with MPICH | `uv sync --dev --extra parallel-mpich` |

Pre-commit hooks run `uv-lock`, `ruff-check --fix`, `ruff-format`, and `ty` automatically on commit.

## Quality Gates

All of the following must pass before any change is considered complete:

```sh
uv run ruff check                       # zero lint errors
uv run ruff format --check              # zero formatting issues
uv run ty check neworder examples test  # zero type errors
uv run pytest                           # all tests pass
```

## Developer Rules

- **Mixed C++/Python codebase.** The C++ extension (`_neworder_core`) is built by `uv sync`. C++ source lives in [src/](src/); do not edit the compiled `.so` files directly.
- **Type stubs must stay in sync.** If the C++ extension API changes, regenerate stubs (see [Regenerating Type Stubs](#regenerating-type-stubs) below).
- **Runtime dependencies are intentional.** `pandas`, `scipy`, and `matplotlib` are runtime dependencies. New runtime deps need a strong justification. Optional extras are defined in `[project.optional-dependencies]` in [pyproject.toml](pyproject.toml): `geospatial` (pyogrio, networkx, osmnx, geopandas, shapely) and three parallel variants — `parallel-native` (uses a system MPI installation via `mpi4py`), `parallel-openmpi` (bundles OpenMPI), and `parallel-mpich` (bundles MPICH). Choose the parallel extra that matches the MPI implementation available on the target system.
- **Line length is 120** (configured in [pyproject.toml](pyproject.toml) under `[tool.ruff]`).
- **Ruff ignore list.** `E501` (line too long), `N803` and `N806` (non-lowercase variable names) are suppressed — matrix/array variable naming conventions take precedence.
- **No comments explaining what the code does.** Only add a comment when the *why* is non-obvious (hidden constraint, workaround, subtle invariant).
- **Type annotations required** on all Python function signatures. `ty` will catch missing or incorrect ones.

## Reviewer Checklist

When reviewing a PR or diff, check:

1. **Correctness** — does the logic match the documented behaviour? Consider edge cases: empty populations, single-process vs. MPI, zero-length timelines.
2. **C++/Python boundary** — pybind11 bindings in [src/Module.cpp](src/Module.cpp) must accurately expose C++ behaviour. Check for ownership, lifetime, and GIL issues.
3. **Type stubs** — if the C++ API changed, verify stubs in [stubs/](stubs/) and [neworder/](neworder/) are updated.
4. **MPI correctness** — functions touching `mpi` must work correctly in both serial (1 process) and parallel (N > 1) contexts. Don't introduce rank-0-only logic without guarding.
5. **Test coverage** — new behaviour needs a test in [test/](test/). The relevant test files map to modules: `test_mc.py` → MonteCarlo, `test_timeline.py` → timelines, `test_model.py` → Model, `test_df.py` → dataframe helpers, etc.
6. **Examples** — examples are the primary user-facing demonstration of the framework and must always work. Any change to the public API, CLI behaviour, or data structures must be checked against all affected examples. Run them manually: `python examples/<name>/model.py` (or `mpiexec -n <N> python examples/<name>/model.py` for parallel examples). A change is not complete until all examples run cleanly.
7. **Ruff rules** — no rule in the `select` list should be suppressed without justification. Active rules: `ARG, B, C, E, F, I, N, PERF, PTH, RUF, SIM, UP, W` (E501, N803, N806 are ignored).
8. **Docs** — if a public API is added or changed, update [docs/api.md](docs/api.md) and relevant pages under [docs/](docs/).

## QA Rules

- Run the full gate suite (`ruff check`, `ruff format --check`, `ty check`, `pytest`) before declaring any task done.
- CI runs the matrix: Python 3.12, 3.13, 3.14, 3.14t × ubuntu, windows, macos. Flag anything that might be platform- or version-specific (e.g. freethreaded 3.14t, compiler flags).
- MPI tests run separately via [mpi-test.yml](.github/workflows/mpi-test.yml). If touching parallel functionality, ensure `mpiexec -n 2 uv run pytest` passes locally.
- Coverage (C++ via gcov + Python) is reported to [codecov.io](https://codecov.io/gh/virgesmith/neworder). Ensure `gcc` and `gcov` versions are consistent if building coverage locally.
- If a test is skipped or marked `xfail`, leave a comment explaining why and when it can be removed.
- **Run the examples.** The test suite does not exercise [examples/](examples/). After any code change, manually run the affected examples (`python examples/<name>/model.py`). Parallel examples require an MPI prefix: `mpiexec -n <N> python examples/<name>/model.py`. The CI release workflow packages the examples as artifacts — they must all work against the published code.

## Task & Design Summaries

Every development task — feature, fix, refactor, or notable design decision — must be recorded in [JOURNAL.md](JOURNAL.md) before the task is considered complete. This is the durable record of *why* the codebase looks the way it does: unlike a diff or a PR description, it survives outside any single conversation or session, so a maintainer (or another agent, with no memory of how the change came about) can understand the intent and the alternatives that were rejected.

- **When**: once the change is implemented and the quality gates pass, before (or as part of) committing.
- **Where**: prepend a new entry to the top of [JOURNAL.md](JOURNAL.md) — newest entries go first — using the template at the top of that file.
- **What to capture**: the motivation (*why*), a high-level summary of the change (*what*), any non-obvious design decisions and the alternatives considered, and known follow-ups or limitations.
- **Scope**: substantive work — new features, bug fixes, refactors, dependency/CI changes, and design decisions. Trivial changes (typo fixes, formatting-only diffs) don't need an entry.

## Repository Layout

```
src/                     # C++ extension source (pybind11)
  Model.cpp / Model.h
  MonteCarlo.cpp / ...
  Module.cpp             # pybind11 bindings entry point
  Timeline.cpp / ...
neworder/                # Python package
  __init__.py            # public exports (re-exports _neworder_core + pure-Python)
  domain.py              # Domain, Edge, Space, StateGrid
  mc.py                  # as_np helper
  timeline.py            # CalendarTimeline
  *.pyi                  # type stubs for C++ submodules (time, mpi, stats, df)
  py.typed               # PEP 561 marker
stubs/
  _neworder_core-stubs/  # generated type stubs for the extension module
test/
  test_df.py
  test_domain.py
  test_geospatial.py
  test_mc.py
  test_model.py
  test_module.py
  test_mpi.py
  test_multithreaded.py
  test_splitmix64.py
  test_stats.py
  test_timeline.py
docs/                    # MkDocs documentation source
  api.md
  developer.md
  contributing.md
  examples/
examples/                # runnable microsimulation examples
.github/workflows/
  build-test.yml         # CI: lint + type check + test matrix
  coverage.yml           # CI: codecov upload
  mpi-test.yml           # CI: MPI parallel tests
  pypi-release.yml       # CI: PyPI publish on v* tag
pyproject.toml
.pre-commit-config.yaml
Dockerfile
```

## Regenerating Type Stubs

Type stubs for the C++ extension are generated with `pybind11-stubgen` (a dev dependency). Run this after any change to the pybind11 bindings in [src/Module.cpp](src/Module.cpp):

```sh
# Regenerate top-level stubs (output goes to stubs/)
pybind11-stubgen _neworder_core --ignore-invalid all

# Regenerate a submodule stub, then move it into the package
pybind11-stubgen _neworder_core.time
mv stubs/_neworder_core/time-stubs/__init__.pyi neworder/time.pyi
# Repeat for other submodules: mpi, stats, df
```

`--ignore-invalid all` is required because pybind11-stubgen cannot parse default arguments that are functions (a pattern used in `_neworder_core`).

After generation, check the output manually:
- Numpy array types often come out as `Any` — fix them to `np.ndarray` with appropriate dtype annotations.
- The top-level stub lands in [stubs/\_neworder\_core-stubs/](stubs/) and is picked up by `ty` via the `stubPackages` setting.
- Submodule stubs go directly into [neworder/](neworder/) as `*.pyi` files alongside the package.

## Branch and Release Policy

- **`main` is the integration branch.** All changes must go through a pull request; direct pushes are blocked for non-admins.
- **Releases are triggered by a `v*` tag** (e.g. `v1.2.3`). Pushing such a tag to GitHub runs [pypi-release.yml](.github/workflows/pypi-release.yml), which builds a source distribution and publishes it to PyPI using trusted publishing (OIDC — no API token needed). It also packages and uploads example archives as release artifacts.
- **Do not push a `v*` tag** unless the release is fully ready, the version in [pyproject.toml](pyproject.toml) matches the tag, and all CI checks pass.
- Version bumps go in `pyproject.toml` (`version = "x.y.z"`).
- The Docker image must be rebuilt and pushed manually after a release (see [docs/developer.md](docs/developer.md)).

## Workflow

1. Create a feature branch off `main` — never commit directly to `main`.
2. Make C++ changes in [src/](src/) and/or Python changes in [neworder/](neworder/).
3. Rebuild the extension: `uv sync --dev`. Add `--extra geospatial` if touching geospatial code, or the appropriate `--extra parallel-*` if testing MPI functionality.
4. If the extension API changed, regenerate type stubs (see [Regenerating Type Stubs](#regenerating-type-stubs)).
5. Add or update tests in [test/](test/) in the relevant file.
6. Run the full gate suite locally.
7. If the public API changed, update [docs/api.md](docs/api.md) and any affected example.
8. Add an entry to [JOURNAL.md](JOURNAL.md) (see [Task & Design Summaries](#task--design-summaries)).
9. Commit — pre-commit hooks will auto-fix formatting and re-lock `uv.lock`.
10. Open a PR targeting `main`; CI must pass before merging.
11. To release: bump the version in [pyproject.toml](pyproject.toml), merge to `main`, create a GitHub release with a new tag `vX.Y.Z` and release notes (e.g. `git log vX.Y.W..HEAD --oneline`) — PyPI publish triggers automatically. Copy generated example artifacts to the release.
