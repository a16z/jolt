# Spec: Agent DX

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @markosg04                     |
| Created     | 2026-07-27                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

AI agents (interactive Claude Code sessions, CI routines, jolt-eval harnesses) perform a large share of development on Jolt, but the repo gives them a human-shaped surface: validation commands are documented only as full-workspace two-feature-mode invocations; the real CI clippy matrix (allocative combos, `no-default-features`, `jolt-verifier` feature products, akita, field-inline) is discoverable only by reading workflows; architectural boundaries live in spec prose (`specs/clean-slate-prover.md` "dependency direction is load-bearing") with no mechanical enforcement, so an agent can silently regress the legacy decomposition; operational knowledge is rediscovered every session; and always-loaded agent context (CLAUDE.md, skills) has no budget guard — it was last rightsized by hand in #1700. This spec defines the agent-DX program, modeled on openai/codex's agent infrastructure: semgrep-enforced architecture invariants derived from a dependency analysis of `crates/`, a canonical command surface with a tiered validation ladder, context-budget governance with per-directory instructions, operational skills backed by tested scripts, deterministic agent primitives (PR watcher, bench differ), a CI latency budget, an agent-legibility lint, and AI-in-CI workflows (label-triggered agent verbs, decomposed review, nightly digest, issue triage) under a strict privilege-split safety architecture.

## Intent

### Goal

Make the repo mechanically self-describing and self-checking for agents: boundaries and conventions enforced by fast deterministic CI (semgrep + small tested scripts) with error messages that name the sanctioned alternative, commands exposed as a tiered verb set, knowledge captured as on-demand skills, and agent judgment applied only where computation cannot decide (review, triage, narration) under least-privilege CI wiring.

### Invariants

- **Green-on-adoption ratchet:** every semgrep rule passes on `main` at merge. Pre-existing violations are either fixed in the implementation PR when comment/doc-level, or carried as per-file allowlist entries each linked to a burn-down issue. Allowlists only shrink; an allowlist entry that no longer matches anything fails CI (self-policing, per codex `verify_cargo_workspace_manifests.py`).
- **Sanctioned-alternative messages:** every rule's `message` states what to do instead and links the governing doc (CLAUDE.md section or spec).
- **Prover/verifier decoupling:** prover-side development must not be able to change verifier behavior — this is what permits a relaxed review bar on prover code. Enforced three ways: the verifier's production dependency closure is pinned in a checked-in lockfile and additions are an explicit reviewed diff; `jolt-verifier` source must not import prove-side modules of shared crates; and any PR whose diff touches closure-crate sources is mechanically labeled `verifier-closure`, routing it to the strict-review path. The asymmetry is process, but the detection is deterministic.
- **Gate equivalence:** `just gate` runs exactly the pre-handoff checks CLAUDE.md documents (fmt, two-feature-mode workspace clippy `-D warnings`, muldiv e2e in both modes); CI remains the authoritative superset. Weakening the gate must be an explicit justfile diff.
- **Script determinism:** agent-consumed scripts in `scripts/agent/` emit versioned JSON; all quantities (deltas, severities, thresholds) are computed in code, never left for a model to estimate; every script has tests running in CI.
- **Context budget:** always-loaded agent context is governed: root CLAUDE.md must not grow past its at-merge line count (recorded in the budget file); each skill ≤ 150 lines; adding any new always-loaded file requires editing the checked-in budget file, which is the review flag.
- **AI-in-CI privilege split:** any CI job invoking a model runs read-only (no write permissions, sandboxed, fork-gated); its output is schema-validated by a separate job that holds the write permission; malformed model output is a no-op, never a failure that blocks humans.
- **No behavioral changes:** prover/verifier behavior and proof bytes are untouched. The single production-source slice is the prove-gating refactor, which moves existing code behind `cfg(feature = "prove")` without modifying it, gated by muldiv in both modes plus the feature-invariance fixtures. Consequently no `jolt-eval/src/invariant/` or `objective/` entries change — semgrep invariants are static source-tree properties, a different genre from jolt-eval's runtime invariants.

### Non-Goals

- Fixing the architectural debts the analysis found (the `zk`/`prover-fixtures`/`field-inline` feature leaks to `jolt-prover-legacy`, the hyperkzg ambient `OsRng`, dead `transcript-*` features on `jolt-verifier`) — each becomes a linked issue; this spec only prevents new instances.
- E2E failure localizer (stage-level prover instrumentation) — requires prover changes; separate spec.
- Bazel or remote build caching; the cache item here is local and opt-in.
- jolt-eval framework changes (the digest *consumes* its outputs).
- Replacing human review or making any AI check merge-blocking; AI-in-CI outputs are advisory comments/labels.

## Evaluation

### Acceptance Criteria

- [ ] `.semgrep/agent-dx/` contains the rule catalog below; `semgrep scan --config .semgrep/agent-dx --error` passes on `main`; the legacy `.semgrep/jolt-verifier-boundaries.yml` is deleted, with any still-valid rules ported into the catalog (terminology-era rules dropped).
- [ ] A `semgrep` CI job (pinned version, `--metrics=off`) runs on every PR in rust.yml, wall-clock < 2 minutes, alongside a pytest-tested `scripts/agent/verify_dep_kinds.py` asserting the manifest-level edges semgrep cannot parse (dep *kinds* via `cargo metadata`).
- [ ] Seeding a violation of each rule class (one source rule, one manifest rule, one stale-allowlist entry) fails CI with a message naming the sanctioned alternative.
- [ ] The verifier production closure is pinned in a checked-in lockfile; `verify_dep_kinds.py` fails on a seeded closure addition; a deterministic labeler applies `verifier-closure` to PRs whose diff touches closure-crate sources (and leaves prover-side PRs unlabeled).
- [ ] `jolt-verifier` fixture-verification tests pass identically standalone and under full workspace feature unification (both configurations wired in CI).
- [ ] `jolt-sumcheck`, `jolt-openings`, and `jolt-blindfold` expose a `prove` feature gating their prove-side modules; a standalone `-p jolt-verifier` default-features build compiles none of them (compile probe in CI); `jolt-kernels`/`jolt-prover`/`jolt-akita`/legacy enable it; `jolt-verifier` enables it only in dev-dependencies; muldiv passes in both modes; the new with/without-`prove` clippy combos join the CI matrix and the justfile `matrix` tier.
- [ ] `just --list` shows documented verbs `fmt`, `check <crate>`, `lint [tier]`, `test <crate> [filter]`, `gate`, `muldiv`, `profile <name>`; `just -n gate` matches the CLAUDE.md pre-handoff set; a lint tier exposes the full CI clippy matrix.
- [ ] CLAUDE.md contains the Validation Ladder and Agent Behavior sections; root CLAUDE.md net line count does not increase; `subprotocols/blindfold/` (and the zkvm subsystem root) carry per-directory CLAUDE.md files holding the invariants moved out of the root.
- [ ] `scripts/agent/verify_context_budget.py` + checked-in budget file enforce the context caps in CI.
- [ ] Skills `debug-transcript-mismatch`, `guest-builds`, `profiling` exist under `.claude/skills/`; every command in each executes on a clean checkout.
- [ ] `scripts/agent/pr_watch.py` emits JSONL snapshots with the action-verb vocabulary (`diagnose_ci_failure`, `process_review_comment`, `retry_failed_checks`, `ready_to_merge`, `stop_pr_closed`, `idle`), tracks retry budget per head SHA, and has pytest coverage; a `babysit-pr` skill dispatches over those verbs and embeds the GitHub mutation policy.
- [ ] `scripts/agent/bench_diff.py` compares two Criterion baseline trees and emits JSON percent deltas, pytest-covered.
- [ ] `scripts/agent/setup-build-cache.sh` exists; the adopt-or-drop measurement (below) is recorded in the implementation PR.
- [ ] The dylint scaffold under `tools/` builds and runs an argument-comment lint (mismatch = error, uncommented-literal = allow) in CI.
- [ ] `.github/workflows/README.md` documents the PR-path latency budget and which checks belong PR-side vs post-merge; a fan-in required job treats skipped/cancelled as failure.
- [ ] Labels `claude-attempt` and `claude-triage` fire routines via the existing `_fire-claude-routine.yml`; the review flow is decomposed into `code-review-*` dimension skills fanned out by an orchestrator; a nightly digest workflow runs a tested collector over fuzz/bench/fs-soundness/arch-test artifacts and posts a model-narrated summary whose severity markers come only from the collector; an issue-labeler (and optional deduplicator) runs under the privilege-split invariant.

### Testing Strategy

Rust suite unaffected (no behavioral changes); muldiv in both modes stays the correctness gate wrapped by `just gate`/`just muldiv`. New tests: pytest for every `scripts/agent/` script (dep-kind fixtures incl. a seeded normal-dep-on-legacy and a seeded verifier-closure addition; labeler fixtures for closure-touching vs prover-only diffs; watcher fixtures for review-before-CI ordering, retry budget reset on new SHA, no-stop-while-open; bench fixtures for regression/improvement/missing-baseline; context-budget over/under fixtures); a CI meta-test that seeds one violation per semgrep rule class in a scratch tree and asserts nonzero exit; `just --list` / `just -n gate` smoke. AI-in-CI workflows are validated by sentinel-label dry-runs on a test issue/PR before enabling on `opened` events.

### Performance

Dev-loop targets (recorded as a table in the implementation PR): semgrep job < 2 min; `just check <crate>` seconds-scale; single-crate single-mode `just lint` ≤ 1/4 of full two-mode workspace clippy; `just gate` unchanged by construction. Build cache experiment: cold `cargo check --features host` in a second worktree must improve ≥ 40% with the cache and warm in-worktree iteration must not regress > 10% (sccache disables incremental — that is the measured risk); miss either → drop the item and record numbers. PR-path CI: the workflows README states the budget; any check moved post-merge must show the PR-path p50 improvement in its PR. No `jolt-eval` objectives move (code-quality objectives measure `crates/jolt-prover-legacy/src/`, untouched).

## Design

### Architecture analysis of `crates/` (basis for the invariants)

The modular stack is 25 crates in a strict layering — foundations (`jolt-field`, `jolt-riscv`, `jolt-poly`, `jolt-transcript`, `jolt-program`, `jolt-lookup-tables`), crypto/PCS abstraction (`jolt-crypto`, `jolt-openings`), PIOP components (`jolt-claims`, `jolt-r1cs`, `jolt-witness`, `jolt-sumcheck`), PCS implementations (`jolt-dory`, `jolt-hyperkzg`, `jolt-akita`), ZK (`jolt-blindfold`), then `jolt-verifier` → `jolt-kernels` → `jolt-prover`, with `jolt-prover-legacy` consuming the stack. The authoritative layering rules live in `specs/clean-slate-prover.md` ("dependency direction is load-bearing": `jolt-sumcheck` must not name `ConcreteSumcheck`; kernels are transcript-free, FS-free, RNG-free; `jolt-prover` names only traits) and in scattered lib.rs docs (`jolt-profiling` "leaf crate"; `jolt-witness` "no id vocabulary of its own"). None of it is enforced.

Current ground truth (2026-07-27, verified via `cargo metadata` dep kinds + source sweep):

- **Source boundary to legacy holds:** zero `use jolt_prover_legacy` in `crates/*/src`; all real imports are under `tests/` (4 files). Every `crates/* → jolt-prover-legacy` manifest edge is dev-kind.
- **Manifest boundary leaks:** four feature entries in three crates forward to the monolith or tracer — `jolt-verifier` `zk = ["jolt-prover-legacy/zk"]` and `prover-fixtures`, `jolt-prover` `prover-fixtures`, `jolt-lookup-tables` `field-inline` (names legacy *and* tracer). The verifier case is a feature-level cycle: legacy depends on `jolt-verifier` while `--features zk` reaches back through a dev-dep edge.
- **PCS abstraction holds:** `jolt-verifier` and `jolt-openings` link no concrete PCS crate in production (dory edges are dev-only); the `akita`/`zk` exclusion is a real `compile_error!` (`crates/jolt-verifier/src/config.rs:12`).
- **Prover→verifier influence is indirect, not a dep edge:** `jolt-prover`, `jolt-kernels`, and legacy sit above `jolt-verifier`, so Cargo already forbids the direct dependency (it would cycle). The channels that remain are (i) shared crates inside the verifier closure that host prove-side code by design — `jolt-sumcheck` contains the prove-side recording seam (`SumcheckRecorder`, `ProveRounds`, `prove_batch`) — so an edit made for prover convenience there is a verifier change even though no prover crate is touched; (ii) feature unification — workspace builds with prover features recompile closure crates under different cfgs; (iii) prover-convenience helpers creeping into `jolt-verifier` itself (`prover-fixtures` is the sanctioned, test-only exception).
- **Pattern conventions are nearly clean:** transcripts are constructed only in `jolt-transcript`'s factories plus two defensible adapters (`JoltToDoryTranscript` wraps a borrowed transcript; `jolt-akita/src/native_batching.rs:181` bridges a foreign Akita transcript); randomness is injected (`R: RngCore`) everywhere except one ambient `OsRng` in `jolt-hyperkzg/src/scheme.rs:266`; `std::time` outside `jolt-profiling`, and print macros in library paths, are already clean outside `#[cfg(test)]`; `#[allow(` survives only in two *generated* akita schedule files; `jolt-kernels` has zero transcript/RNG references, matching its spec invariant.
- **Incidental findings** routed to issues, not rules: `jolt-verifier`'s three `transcript-*` features are declared, defaulted, and never read; `XLEN` is hardcoded in three places; rayon gating is inconsistent (5 crates unconditional); `jolt-lookup-tables/src/lib.rs` lacks a crate doc; stale `jolt-zkvm` name in the hyperkzg README.

### Semgrep rule catalog (`.semgrep/agent-dx/`)

Why semgrep rather than more clippy/dylint: semgrep scans text across *all* cfg combinations (clippy sees only compiled ones — Jolt's feature matrix makes this a real gap), rules are reviewable data an agent can extend without writing Rust, and per-rule `paths`/allowlists give ratchet ergonomics. Dylint complements it where name resolution is required (WS-6). The existing `.semgrep/jolt-verifier-boundaries.yml` predates this design (terminology bans and stage-layout rules from the sumcheck-relations refactor), is wired into no workflow, and is superseded: port only rules that still pass with a current rationale.

| Rule id | Enforces | State on main |
|---|---|---|
| `boundary-no-legacy-in-modular-src` | no `jolt_prover_legacy::` in `crates/*/src` (tests excluded) | clean — locks the decomposition |
| `boundary-no-legacy-in-features` | no `crates/*` feature forwards to `jolt-prover-legacy/*` or `tracer/*` (with `verify_dep_kinds.py` asserting no normal-kind dep edges) | 4 allowlisted entries, issue-linked |
| `boundary-verifier-pcs-agnostic` | no `jolt_dory::`/`jolt_hyperkzg::`/`jolt_akita::` in `jolt-verifier` or `jolt-openings` src | clean |
| `boundary-verifier-no-prove-side-imports` | `jolt-verifier/src` never references prove-side seam symbols of shared crates (`SumcheckRecorder`, `ProveRounds`, `prove_batch`, `recorder::`) outside test modules | clean in production (test-module hits only); compiler-enforced once prove-gating lands |
| `boundary-sumcheck-below-verifier` | `jolt-sumcheck/src` never names `ConcreteSumcheck`/generated verifier types | clean — encodes the load-bearing direction rule |
| `boundary-kernels-transcript-free` | no `Transcript`, `Rng`/`rand`, FS references in `jolt-kernels/src` | clean — locks clean-slate-prover invariant #5 |
| `boundary-profiling-leaf` | library crates don't `use jolt_profiling` (tracing only) | verify at implementation |
| `pattern-transcript-passed-not-built` | `*Transcript::new` outside `jolt-transcript` + tests forbidden | 2 allowlisted adapters |
| `pattern-rng-injected-not-ambient` | no `OsRng`/`thread_rng()`/`from_entropy` construction in non-test src | 1 allowlisted (hyperkzg setup), issue-linked |
| `pattern-no-wallclock` | no `std::time::{Instant,SystemTime}` outside `jolt-profiling` (non-test) | clean |
| `pattern-no-print` | no `println!`/`eprintln!`/`dbg!` in `crates/*/src` (bins/tests excluded) | clean; covers cfgs clippy never compiles |
| `pattern-no-allow-attr` | `#[expect]` over `#[allow]` in `crates/*/src` | 2 allowlisted (generated schedules), issue on the generator |
| `pattern-lib-doc-required` | every `crates/*/src/lib.rs` starts with `//!` | 1 fixed in-PR (doc-only) |

Three decoupling mechanisms live outside semgrep because they need `cargo metadata` or diff context, all in the same CI job family:

- **Closure lockfile:** the verifier's production closure (today: `jolt-verifier` + its 14 normal deps, transitively) is pinned in a checked-in file; `verify_dep_kinds.py` fails when the computed closure differs. Growing the verifier's reach is then a reviewed one-line diff, never a side effect of a Cargo.toml edit elsewhere.
- **Closure-diff labeler:** a deterministic job intersects each PR's changed files with closure-crate sources and applies the `verifier-closure` label. This is what makes the asymmetric review policy real: prover-side PRs (outside the closure) ride the relaxed bar; anything touching what the verifier links is routed to strict review (and can auto-trigger the review-dimension routine in WS-8).
- **Feature-invariance check:** `jolt-verifier`'s fixture-verification tests run both standalone (`-p jolt-verifier`, minimal features) and under full workspace feature unification, asserting that enabling prover features cannot change a verification outcome.

**Prove-gating the shared stack (decision).** Rather than leaving prove-side code permanently resident in the verifier closure, the shared crates that host it gain a `prove` feature and gate it: `jolt-sumcheck` (`prover.rs`, `recorder.rs`, `committed.rs` — already file-separated), `jolt-openings` (the packed-opening prover half), `jolt-blindfold` (`prove.rs`; the split from `verify.rs` already exists). `jolt-verifier` depends on all three with `prove` **off** in normal dependencies and **on** in dev-dependencies (its round-trip tests in `stages/relations.rs` are the only prove-side references, all in test modules). Enablers: `jolt-kernels`, `jolt-prover`, `jolt-akita`, legacy. This is consistent with the repo's feature policy ("features may gate linkage, never choice" — clean-slate-prover.md): the feature gates the prover *surface*, not behavior. The payoff is that the compiler, not process, guarantees the standalone verifier build contains no prove-side code; `boundary-verifier-no-prove-side-imports` and the closure labeler become defense-in-depth for unified builds, and prove-gated modules are exempt from the `verifier-closure` label — restoring the relaxed review path for prove-seam edits. Known cascade for implementation: `jolt-verifier-derive`'s generated code references prove-side symbols, so the generated prove surface must itself be `cfg(feature = "prove")`-gated (or split into a prover-side derive); this is the one open design point.

Rules the analysis argues for but that would flag broad code today (rayon-behind-`parallel`, `unsafe`-density caps) are deferred, listed in the catalog README with their current violation counts.

### Remaining workstreams

**WS-2 Command surface.** Root justfile (verbs above; `just lint` tiers: `touched <crate>` single-mode → `full` two-mode → `matrix` = the CI clippy feature products); CLAUDE.md Validation Ladder + Agent Behavior rules (never kill running cargo commands; scope tests `-p`; no full suite unprompted; no re-run after pure fmt; batch `host`↔`host,zk` switches; gate once before handoff; >30 min rediscovering an environmental fact → capture it in a skill in the same PR). Makefile keeps arch-tests; justfile is the sole agent entry point.

**WS-3 Context governance.** Budget file + `verify_context_budget.py` in CI; per-directory CLAUDE.md for `crates/jolt-prover-legacy/src/subprotocols/blindfold/` (claim/constraint sync rules) and the zkvm subsystem, shrinking the root; AGENTS.md symlink keeps one contract for all agent brands.

**WS-4 Operational skills.** `debug-transcript-mismatch` (reproduce in standard mode first; walk stage order to first divergent claim; prover vs `new_from_verifier` values; claim/constraint sync), `guest-builds` (CLI reinstall, ZeroOS musl, "Built ELF not found"), `profiling` (commands + trace interpretation).

**WS-5 Agent primitives.** `pr_watch.py` + `babysit-pr` skill: script owns polling, state, retry budgets keyed by head SHA, trusted-author filtering; skill owns judgment and embeds the mutation policy (push only to the PR's head branch; never reply to human review comments without user-confirmed text; agent comments prefixed; never act so it's unclear whether agent or human did something visible to others; flake handling never edits tests/CI/pins — rerun within budget or stop). Plus `bench_diff.py` and the build-cache experiment.

**WS-6 Legibility lint.** Dylint scaffold under `tools/` with an argument-comment lint modeled on codex's (`/*param*/` validated against the resolved parameter name; mismatch = error, uncommented literal = allow initially); the scaffold is the future home for Jolt-specific type-resolved lints.

**WS-7 CI latency budget.** `.github/workflows/README.md` documenting PR-path vs post-merge placement rules and budgets; `check_ci_results`-style required fan-in (`if: always()`, skipped/cancelled = failure). Moves of existing checks happen individually with measured PR-path improvement.

**WS-8 AI-in-CI.** On the existing routine plumbing: `claude-attempt` (issue → branch + fix PR) and `claude-triage` (issue → reproduce, comment findings) labels; `ci-code-review` refactored into an orchestrator fanning `code-review-*` dimension skills (claim/constraint sync, zk/non-zk cfg parity, hot-path perf, change-size ≤800/≤500 with staged-split proposals grounded in the diff, proof-format breaking changes, test quality); nightly digest (tested collector aggregates fuzz/bench/fs-soundness/arch nightlies, computes severity from thresholds, stamps script version + git head; model writes prose and may only copy computed markers); issue labeler with a curated taxonomy, deduplicator optional second. All under the privilege-split invariant: read-only sandboxed model jobs, strict output schemas (`additionalProperties: false`), separate write jobs, sentinel labels for re-runs, idempotency markers on posted comments, fork-gated.

### Alternatives Considered

- **Enforce boundaries in clippy/dylint instead of semgrep:** rejected as primary — clippy misses uncompiled cfg combinations and lint logic is Rust code agents can't cheaply extend; dylint is retained (WS-6) for checks needing name resolution.
- **Python verifier scripts (codex-style) instead of semgrep:** rejected — semgrep gives per-rule paths/allowlists, IDE integration, and a rules-as-data catalog for free; Python remains only where semgrep can't reach (dep kinds, token budgets).
- **Fix the four feature leaks now instead of allowlisting:** rejected — unwinding `jolt-verifier`'s `zk` feature cycle is a functional change out of scope for a tooling PR; the allowlist + issue makes the debt visible without blocking.
- **Leave the prove-side seam ungated and rely on lockfile + labeler alone:** rejected — every prove-seam edit would land on the strict-review path, defeating the point of relaxed prover-side development; a compiler guarantee beats a process guarantee.
- **Extract a `jolt-sumcheck-prove` crate instead of a feature:** rejected for now — heavier than needed, and clean-slate-prover.md records the explicit flip conditions for crate extraction; a linkage-gating feature is the lighter form consistent with the repo's stated feature policy.
- **Extend the Makefile / commit `.cargo/config.toml` with sccache / put knowledge in CLAUDE.md:** rejected as in v1 — no self-documenting verbs; breaks contributors without sccache; always-loaded context is a per-session tax (hence the budget invariant).
- **Make AI review checks required:** rejected — model output gates a human workflow only advisorily; determinism gates (semgrep, scripts) are the only new required checks.

## Documentation

No `book/` changes (contributor-facing). CONTRIBUTING.md command blocks replaced by `just` verbs (also fixes the existing CLAUDE.md/CONTRIBUTING clippy-scope drift); `.semgrep/agent-dx/README.md` documents the catalog, allowlist policy, and deferred rules; `.github/workflows/README.md` is new (WS-7).

## Execution

Slices, roughly in order: (1) semgrep catalog + CI job + `verify_dep_kinds.py` with the verifier-closure lockfile + closure-diff labeler + burn-down issues; (2) justfile + CLAUDE.md/CONTRIBUTING rewrite + behavior rules; (3) context budget check + per-directory CLAUDE.md moves; (4) operational skills; (5) `pr_watch.py`/babysit-pr + `bench_diff.py` + cache experiment; (6) dylint scaffold; (7) workflows README + fan-in; (8) AI-in-CI: labels first (existing plumbing), then review decomposition, digest, labeler; (9) prove-gating the shared stack — the only production-source slice, landed as its own PR after the closure machinery exists to verify it, resolving the derive question first. Each slice is independently landable; 1–2 are the highest-leverage pair. Unverified until implementation: semgrep's Rust-parser fidelity on the few macro-heavy files (fallback: `generic` mode per rule), `boundary-profiling-leaf` current state, sccache numbers, and single-crate lint timing.

## References

- openai/codex: `verify_tui_core_boundary.py` / `verify_cargo_workspace_manifests.py` (boundary scripts, self-policing allowlists), root `AGENTS.md`, `.codex/skills/` (`babysit-pr` watcher + mutation policy, `codex-issue-digest` computed-severity collector), `.github/workflows/README.md` (latency budget), issue-labeler/deduplicator/translator workflows (privilege split, output schemas), `tools/argument-comment-lint`.
- `specs/clean-slate-prover.md` (layering, invariants #1–8, feature-policy line "features may gate linkage, never choice"), `specs/self-contained-sumcheck-relations.md` (origin of the superseded semgrep file).
- [semgrep](https://semgrep.dev), [just](https://github.com/casey/just), [dylint](https://github.com/trailofbits/dylint), [sccache](https://github.com/mozilla/sccache).
- `jolt-eval/README.md` — "maximize agent productivity" as an existing repo goal.
