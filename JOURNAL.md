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

