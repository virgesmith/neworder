# Agent Skill

*neworder* ships an [agent skill](https://code.claude.com/docs/en/skills) — a compact, purpose-built
reference for AI coding agents (Claude Code and others that read `SKILL.md` files) covering the
model lifecycle, timelines, the `MonteCarlo` and `SplitMix64` engines, the DataFrame operations,
spatial domains, parallel execution and the framework's common pitfalls.

The skill is a *pointer into these docs*, not a replacement for them: it gives an agent enough to
write a correctly-shaped model without loading the whole site into its context, and links back to
the relevant page here whenever more detail is needed.

## Installing

The skill is bundled inside the installed package, and a console script installs it into a project:

```sh
neworder-skill --install [PATH]  # default PATH: .agents
neworder-skill --remove [PATH]   # default PATH: .agents
```

This creates (or removes) `PATH/skills/neworder`. Run it from the environment *neworder* is
installed in — it's a normal console-script entry point, so it's only on `PATH` while that
virtualenv is active (or via `uv run neworder-skill --install`).

!!! note "Choosing a target directory"
    The default `.agents/skills/` is read by agents supporting the cross-tool convention. Pass an
    explicit `PATH` for a specific harness, e.g. `neworder-skill --install .claude` installs to
    `.claude/skills/neworder`.

Where the platform allows it, the skill is installed as a **symlink** to the copy inside the
installed package, so it always matches the version of *neworder* actually in use and there is
nothing to keep up to date. On platforms where symlinks aren't permitted (Windows without
developer mode) the files are copied instead, and re-running `--install` refreshes that copy —
do this after upgrading *neworder*.

!!! warning "Installing into a version-controlled project"
    A symlinked skill points outside the repository, so it will not work for anyone else who
    checks it out. Either add `.agents/` to `.gitignore` and have each developer install it, or
    commit a copy.

Both commands refuse to touch anything they don't recognise as their own: an existing file,
directory or symlink at the target that isn't an installed copy of this skill is left alone and
the command exits non-zero.

## What the agent gets

The skill's frontmatter tells an agent when to load it — mentions of *neworder*, `no.Model`,
`neworder.run`, the timeline classes, `mc.hazard`/`stopping`/`arrivals`, `no.df.transition`,
`SplitMix64`, `StateGrid` or `neworder.mpi`. Its body covers:

- the `modify` → `step`/`check` → `finalise` lifecycle, and a minimal working model
- the four timeline types, custom timelines, and open-ended timelines with `halt()`
- the `MonteCarlo` sampling methods, the seeding strategies, and the numpy adapter `as_np`
- `SplitMix64`'s keyed, order-independent draws and when they're preferable
- `no.df.transition`/`transition_conditional`/`unique_index`, including the categorical-dtype and
  assign-the-result-back requirements
- spatial domains and edge behaviours
- MPI patterns, and the all-or-nothing failure rule that avoids deadlocks
- the framework's recurring pitfalls: an uninitialised base class, strict C++ typing, `NEVER`
  being NaN, and explicit per-agent Python loops

The source is [`neworder/skill/SKILL.md`](https://github.com/virgesmith/neworder/blob/main/neworder/skill/SKILL.md)
in the repository — fixes and additions are welcome, see [Contributing](./contributing.md).
