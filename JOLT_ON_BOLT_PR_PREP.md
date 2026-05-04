# Jolt-on-Bolt PR Prep

This document tracks the cleanup plan for turning the `refactor/crates`
working branch into a reviewable PR stack. The initial scope is the
Jolt-on-Bolt path only:

```text
full Fr
non-zk
generated prover/verifier artifacts
jolt-core retained as the reference implementation
```

Do not expand the first stack into zk, wrapping, blindfold, or jolt-core
replacement work.

## Branch Policy

`refactor/crates` is the dirty integration branch. Do not PR it directly. Cut
one review branch per crate or per cleanup boundary, with each branch rebased
over current `origin/main` and any prerequisite crate PRs.

Keep the active verifier-cleanup work isolated until it stabilizes. That work
may feed the Bolt and generated-artifact PRs, but it should not obscure the
crate landing sequence.

## Upstream PRs To Watch

Before cutting or rebasing stack branches, check open PRs that may affect the
foundation crates:

- `jolt-v2/jolt-hyperkzg`: HyperKZG crate.
- `quang/pcs-prover-verifier-split`: PCS trait split over `refactor/crates`.
- `quang/bytecode-expand-spec`: bytecode expansion design.
- `jolt-sumcheck` fuzzing and `jolt-poly` perf PRs.
- `jolt-crypto` perf and transcript spongefish specs.

If these land first, rebase the stack and remove duplicated work.

## Concrete PR Stack

Use small review branches cut from `origin/main`, rebased over their prerequisite
branches only. Each PR should be understandable without reading the integration
branch.

| Review | Branch | Scope | Depends on |
| --- | --- | --- | --- |
| 1 | `prep/restore-process-docs` | Restore process/spec/agent files that disappeared from `origin/main`; no crate code changes. | `origin/main` |
| 2 | `prep/remove-stale-experimental-crates` | Delete `crates/jolt-compiler/` and `crates/jolt-zkvm/`; remove stale active references. | `prep/restore-process-docs` |
| 3 | `prep/stub-deferred-crates` | Keep `crates/jolt-wrapper/` and `crates/jolt-blindfold/` as buildable stubs with explicit deferred status. | cleanup PRs |
| 4 | `stack/jolt-witness` | Witness/oracle materialization crate for the full-`Fr`, non-zk path; prover-side only. | cleanup PRs |
| 5 | `stack/jolt-kernels` | Shared prover kernels used below generated prover code; no verifier dependency. | cleanup PRs |
| 6 | `stack/bolt` | Bolt compiler, Jolt protocol package, generation command, schema/version headers, local gates, README updates. | `jolt-witness`, `jolt-kernels` as needed |
| 7 | `stack/generated-jolt-verifier` | Generated verifier crate, verifier-owned proof/output types, negative gates, boundary checks. | `bolt` |
| 8 | `stack/generated-jolt-prover` | Generated prover crate; may construct verifier-owned proof/output types; no tracer internals. | `bolt`, `jolt-verifier`, `jolt-witness`, `jolt-kernels` |
| 9 | `stack/jolt-equivalence` | E2E semantic oracle comparing the generated path to `jolt-core`. | generated prover/verifier |
| 10 | `stack/jolt-profiling` | Land reusable core-vs-Bolt measurement primitives and perf-oracle gates; keep protocol-specific harnesses near their owning semantic oracle or CI job. | generated path, `jolt-equivalence` |

If the generated verifier and prover cannot be separated cleanly at first,
collapse reviews 7 and 8 into one generated-artifact PR, but keep the dependency
direction explicit: verifier code must remain witness/prover/trace/core-free,
and prover code may depend on verifier-owned proof/output types only.

## Origin/Main Restoration Audit

Current comparison command:

```bash
git diff --name-status --diff-filter=D origin/main HEAD -- .github .claude agent-skills specs book docs CLAUDE.md AGENTS.md README.md
```

Restore these process/spec files before crate PRs:

```text
.claude/skills/analyze-spec/SKILL.md
.claude/skills/ci-code-review/SKILL.md
.claude/skills/implement-spec/SKILL.md
.claude/skills/new-invariant/SKILL.md
.claude/skills/new-objective/SKILL.md
.claude/skills/new-spec/SKILL.md
.claude/skills/update-docs/SKILL.md
CLAUDE.md
agent-skills/jolt/SKILL.md
agent-skills/jolt/install.sh
specs/1370-spec-driven-workflow.md
specs/1402-ecdsa-inputs-sanitation.md
specs/TEMPLATE.md
specs/act4-tests.md
specs/unify-field-hierarchy.md
```

No `.github/workflows/**`, book, README, or `AGENTS.md` deletions showed up in
that scoped comparison. Source-code deletions outside the process/spec scope
should be reviewed by their owning crate PRs rather than restored in the prep
hygiene branch.

## Generated Artifact Policy

`jolt-prover` and `jolt-verifier` are generated artifacts. Treat them as
checked-in outputs, not hand-maintained source.

Required before landing the generated crates:

- Add a documented regeneration command under `crates/bolt/README.md`.
- Add a CI or local hook that regenerates the artifacts and fails if
  `crates/jolt-prover` or `crates/jolt-verifier` differ from the checked-in
  tree.
- Keep generated files deterministic and formatted.
- In PR descriptions, state the Bolt commit or command that produced the
  generated artifacts.

Suggested CI shape:

```text
run Bolt artifact generation
git diff --exit-code crates/jolt-prover crates/jolt-verifier
```

The current local regeneration command documented in `crates/bolt/README.md` is:

```bash
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules --cargo-quiet
```

The generated-artifact PR should either keep that command as the source of truth
or replace it with a smaller public command, then wire the same command into CI.

Generated files should also carry an obvious generated header and artifact
schema/version marker so reviewers know which files are source of truth and
which files are compiler output.

## Crate Boundary Rails

Add Semgrep or equivalent static checks for crate-boundary rules. These checks
should run in CI and should be easy to run locally before opening a PR.

Verifier boundaries:

```text
jolt-verifier must not import jolt-prover
jolt-verifier must not import jolt-kernels
jolt-verifier must not import jolt-witness
jolt-verifier must not import jolt-trace
jolt-verifier must not import jolt-core
jolt-verifier must not import tracer internals
```

Prover and trace boundaries:

```text
jolt-prover must not import tracer internals directly
jolt-prover may import jolt-verifier proof/output types intentionally
jolt-prover may import jolt-witness for oracle materialization
jolt-trace is the boundary crate that may know tracer/emulator details
jolt-witness should stay prover-side and verifier-free
```

Bolt/compiler boundaries:

```text
generic Bolt modules must not branch on Jolt-specific names
Jolt protocol facts must live under crates/bolt/src/protocols/jolt/**
generated role crates must not contain stale jolt-host/jolt-instructions names
generated verifier code must not contain prover-only imports or witness paths
```

Suggested Semgrep checks:

- forbidden imports by crate path
- stale crate names in active manifests/generated code
- generated verifier importing prover/kernel/witness/trace/core crates
- generic Bolt code matching `jolt`, `Jolt`, stage names, or relation names
  outside an explicit migration allowlist
- use of `panic!`, `todo!`, `unimplemented!`, `dbg!`, `println!`, `eprintln!`,
  `unwrap`, and `expect` in non-test prover/verifier/compiler paths

Keep the allowlist small, named, and shrinking. If a rule fires in production
code, either fix it or document why it is temporarily allowed.

Initial Semgrep rule inventory:

```text
crate-boundary/verifier-no-prover-imports
crate-boundary/verifier-no-witness-imports
crate-boundary/verifier-no-trace-imports
crate-boundary/verifier-no-core-imports
crate-boundary/prover-no-tracer-internals
crate-boundary/witness-no-verifier-imports
crate-boundary/bolt-generic-no-jolt-special-cases
crate-boundary/generated-verifier-no-prover-only-symbols
crate-boundary/no-stale-crate-names
crate-boundary/no-debug-or-panicking-macros-in-prod
crate-boundary/no-unwrap-expect-in-prod
```

Rule scopes should be explicit:

```text
prod source:
  crates/bolt/src/**
  crates/jolt-prover/src/**
  crates/jolt-verifier/src/**
  crates/jolt-witness/src/**
  crates/jolt-trace/src/**
  crates/jolt-kernels/src/**

excluded or separately linted:
  **/tests/**
  **/benches/**
  crates/bolt/tests/fixtures/**
  crates/bolt/tests/generated/**
  historical docs
```

Add a CI job that runs the Semgrep rules before the generated-artifact diff
check. Boundary failures should be treated as architecture regressions, not
style nits.

Current status:

- `.semgrep/jolt-rules.yaml` already contains general production-code hygiene
  checks for panic/debug macros, unwrap/expect, transmute, println/eprintln,
  broad `#[allow(...)]`, expensive clones, ark dependency leaks, and HashMap
  review prompts.
- Add the crate-boundary inventory above as named rules before landing the
  generated role crates. The existing generic hygiene rules are useful but do
  not yet prove the verifier/prover/witness/Bolt dependency boundaries.
- CI should run the boundary rules on the explicit prod-source scope above, then
  run the broader hygiene rules either as warnings or as hard errors per crate
  once each crate has documented its remaining debt.

## Error And Clippy Hardening

Move toward stricter clippy and typed errors before productionizing the stack.

Cleanup targets:

- Replace `assert!`, `panic!`, `unwrap`, and `expect` in non-test paths with
  typed errors.
- Separate user/input errors from internal compiler invariant errors.
- Make generated prover/verifier APIs return explicit error enums.
- Convert ad hoc string errors into structured error variants where practical.
- Keep `clippy::pedantic` enabled and reduce broad suppressions over time.
- Use `#[expect(...)]` for narrow, intentional lint exceptions rather than
  broad `#[allow(...)]`.

The initial stack does not need to eliminate every warning, but it should
remove hard-deny violations from production paths and document remaining lint
debt.

Clippy gate shape:

```text
initial PR stack:
  cargo clippy -p bolt --no-deps -- -D warnings
  cargo clippy -p jolt-witness --no-deps -- -D warnings
  cargo clippy -p jolt-prover --no-deps -- -D warnings
  cargo clippy -p jolt-verifier --no-deps -- -D warnings

production hardening:
  add clippy::pedantic per crate once broad lint debt is removed
  make new allow/expect entries fail review unless justified inline
  keep generated-code lint policy documented in the Bolt README
```

Prefer making generated code clean rather than suppressing lints in the
generated crates. If the generator needs an exception, encode the exception in
one place in Bolt and include a comment explaining why generated output needs
it.

## Verifier And Soundness Hardening

The generated verifier must become small, deterministic, witness-free, and
hard to misuse.

Required negative/equivalence gates should cover:

```text
missing proof
missing stage proof
reordered stage proof
stage proof in wrong slot
tampered commitment
tampered sumcheck coefficient
tampered sumcheck point
tampered named eval
wrong transcript state
missing opening claim
extra opening claim
opening claims in wrong order
opening equality mismatch
wrong PCS proof
missing evaluation setup
missing evaluation proof
wrong evaluation proof
```

Transcript binding audits should verify that public config, proof shape,
commitments, openings, stage boundaries, and evaluation claims are bound in the
expected order. The verifier must never consume prover-only hints, witness
material, trace rows, or kernel-only data.

Hardening checklist:

- Normalize verifier inputs before transcript absorption where ordering matters.
- Reject duplicate, missing, or unexpected named claims rather than silently
  ignoring them.
- Keep all proof length and stage-count checks ahead of cryptographic work.
- Add negative tests for every public verifier entrypoint.
- Ensure verifier errors are deterministic and do not depend on prover-only
  state.
- Keep any compatibility shims temporary and tracked in this document.

## Compiler Fixture Bloat

Compiler-generated fixtures should not create large PR diffs by default.

Cleanup steps:

1. Move compiler scratch/generation fixtures under ignored paths such as
   `crates/bolt/tests/generated/` or `crates/bolt/tests/fixtures/`.
2. Keep only intentional golden fixtures tracked.
3. If bulky generated fixtures are already tracked, remove them with
   `git rm --cached` in a dedicated cleanup commit after confirming equivalent
   regeneration coverage exists.
4. Prefer small structural fixture assertions over checked-in full generated
   Rust files unless the fixture is a deliberate golden.

The root `.gitignore` should ignore future compiler-generated fixture
directories so local regeneration does not inflate diffs.

## Naming Cleanup

Remove stale names from new crate PRs:

- `jolt-host` is superseded by `jolt-trace`.
- `jolt-instructions` is superseded by the current split instruction crates.
- `jolt-compiler` and `jolt-zkvm` should not remain in manifests, comments, or
  tests except in historical notes.

Use current crate names in manifests, docs, tests, and generated code.

## Stale Path And Code Removal

Remove stale code before cutting crate PRs so reviewers only see the current
Jolt-on-Bolt design.

Delete these stale crate directories from the working branch:

```text
crates/jolt-compiler/
crates/jolt-zkvm/
```

Current stale rationale:

- Both directories are additions relative to `origin/main`.
- Neither directory is a root workspace member in `Cargo.toml`.
- `Cargo.lock` and active crate manifests do not require `jolt-compiler` or
  `jolt-zkvm`; the only manifest references are inside the stale directories
  themselves.
- `jolt-compiler` is superseded in this stack by `crates/bolt/`.
- `jolt-zkvm` is an old top-level runtime/prover shape that overlaps with the
  current split through `jolt-trace`, `jolt-witness`, generated `jolt-prover`,
  generated `jolt-verifier`, and `jolt-equivalence`.
- Remaining references in active files should be removed, renamed, or marked as
  historical context in the stale-cleanup PR.

Prep cleanup status:

- `crates/jolt-compiler/` and `crates/jolt-zkvm/` have been removed from this
  prep branch.
- Active docs/comments have been moved to current terminology: generated path,
  `jolt-trace`, and current prover/verifier crates.
- The old legacy modular-stack runner and bespoke Bolt stage
  benchmark were removed. The reusable timing, median, and peak RSS pieces are
  folded into `jolt-profiling`; future perf gates should use those primitives
  instead of a standalone benchmark crate.

Keep these only as minimal stubs unless a later PR explicitly implements them:

```text
crates/jolt-wrapper/
crates/jolt-blindfold/
```

Remove stale references from:

- root `Cargo.toml` workspace members and `[workspace.dependencies]`
- crate manifests under `crates/*/Cargo.toml`
- tests and benches, especially old modular-stack code paths
- generated role crates and Bolt emitted imports
- README/book docs
- comments that describe the current architecture

Current replacements:

```text
jolt-host         -> jolt-trace
jolt-instructions -> current split instruction crates / jolt-riscv path
jolt-compiler     -> remove from this stack
jolt-zkvm         -> remove from this stack
```

Before opening each PR, run:

```bash
rg -n "jolt-host|jolt-instructions|jolt-compiler|jolt-zkvm|jolt_compute|jolt-cpu|jolt_cpu" Cargo.toml crates book README.md
```

Any remaining match must be either deleted, renamed to the current crate, or
explicitly documented as historical context. Do not leave stale names in active
manifests, generated code, or e2e/perf paths.

## Docs

Before the Bolt PR lands:

- Expand `crates/bolt/README.md` into a real crate README.
- Add a book page for Bolt and the generated Jolt path.
- Mark the initial status clearly: full `Fr`, non-zk, experimental.
- Document the generation command, artifact ownership, local gates, and LLVM
  environment requirements.
- Add an architecture diagram for:

```text
jolt-trace -> jolt-witness -> jolt-prover/jolt-kernels -> jolt-verifier
```

- Explain why `jolt-core` remains the reference implementation while
  Jolt-on-Bolt matures.
- Add per-crate status notes: generated, production, experimental, or stub.
- Document the intended public APIs and which modules are internal.

The Bolt README should cover:

- What Bolt owns as a compiler and what the generated crates own as artifacts.
- The supported initial Jolt-on-Bolt mode: full `Fr`, non-zk.
- How to regenerate prover/verifier artifacts.
- How generated headers/schema versions map back to Bolt.
- Which fixture directories are ignored scratch output versus tracked goldens.
- Required LLVM/MLIR setup for local development.
- The local pre-PR command sequence.

The book should cover:

- Where Bolt fits in the Jolt architecture.
- The crate pipeline from trace extraction through verification.
- Why `jolt-core` remains the reference implementation during Bolt maturation.
- How `jolt-equivalence` and `jolt-profiling` act as semantic and performance
  oracles.
- The deprecation criteria for eventually replacing `jolt-core`.

## Testing And Gates

Minimum gates per relevant PR:

```text
cargo fmt --check
cargo check -p <crate> --offline
cargo test -p <crate> <focused tests>
```

For Bolt and generated artifacts, also run:

```text
cargo check -p bolt -p jolt-prover -p jolt-verifier -p jolt-equivalence --offline
generated-artifact regeneration diff check
jolt-equivalence e2e tests as the semantic oracle
```

The local MLIR/LLVM environment currently needs `llvm-config` on `PATH`. Record
the exact env in the Bolt README so reviewers can reproduce checks.

Expand CI in stages:

```text
fast PR gate:
  cargo fmt --check
  cargo metadata
  targeted cargo check
  targeted unit tests
  Semgrep crate-boundary checks
  focused clippy with -D warnings

generated-artifact gate:
  regenerate jolt-prover and jolt-verifier
  git diff --exit-code crates/jolt-prover crates/jolt-verifier

semantic oracle gate:
  run focused jolt-equivalence tests on small real traces
  run verifier negative/tamper tests
  compare generated path outputs against jolt-core reference outputs

perf oracle gate:
  use jolt-profiling to instrument core-vs-Bolt runs with matching spans
  run small perf smoke gates on every relevant PR
  run larger/nightly perf gates for realistic traces
  fail on configured proof-size, prove-time, verify-time, or memory regressions
```

Use `jolt-equivalence` as the semantics rail and `jolt-profiling` as the
performance/observability rail. These gates should protect continued Bolt
compiler development, not just the initial crate landing.

CI should distinguish required PR gates from scheduled maturity gates. Keep the
fast PR gate cheap enough to run on every change, then use nightly or manual
jobs for larger equivalence suites and realistic perf traces.

## Equivalence And Profiling

Keep `jolt-equivalence` for now. It is the e2e sandbox proving the generated
path matches `jolt-core`, and `jolt-core` remains the reference until the Bolt
path is mature enough to replace it confidently.

The old separate benchmark crate is gone. Keep reusable measurement primitives
in `jolt-profiling`, expand tracing coverage in generated/codegen paths, and
define perf gates around paired core-vs-Bolt runs.

Tracing/profiling cleanup:

- Add spans around generated prover stage execution.
- Add spans around kernel calls and expensive oracle materialization paths.
- Keep setup, prove, verify, commitment, stage, opening, and evaluation-proof
  timings separable.
- Define at least one small perf smoke program and one realistic perf program.
- Track proof size, peak memory, prove time, verify time, and stage breakdowns.
- Require the same named span families for the `jolt-core` reference path and
  the generated Bolt path, including:

```text
core.setup
core.prove
core.verify
bolt.setup
bolt.commitment
bolt.stage1 ... bolt.stage8
bolt.evaluate
bolt.verify
```

After folding benchmark primitives into `jolt-profiling`, add named gates for:

```text
semantic oracle:
  generated trace/proof/verifier behavior matches jolt-core for selected cases

perf oracle:
  generated prover stage timing stays within configured thresholds versus core
  generated verifier timing stays within configured thresholds versus core
  proof size and memory do not regress unexpectedly versus core

observability oracle:
  required tracing spans are present for stage, kernel, oracle, and PCS work
```

## Public API Polish

Before declaring the generated path production-ready:

- Keep stable top-level APIs for prove, verify, setup, and trace extraction.
- Hide generated internals unless tests or debugging require exposure.
- Keep verifier APIs independent of prover, witness, trace, and core crates.
- Avoid exposing stage-specific generated types unless they are part of a
  deliberate debugging or artifact-inspection API.
- Document which APIs are stable and which are internal/compiler-owned.

## Long-Term Maturity Path

The goal is to continue maturing Bolt until it can replace and eventually
deprecate `jolt-core`. Do not couple that end state to the initial crate stack.

Near term:

```text
land full-Fr non-zk Jolt-on-Bolt
keep jolt-core as reference
use jolt-equivalence for semantic confidence
use jolt-profiling for perf confidence
```

Medium term:

```text
develop new protocol/compiler work on Bolt
expand compiler rails and generic protocol support
expand compiler coverage while preserving crate-boundary checks
reduce generated verifier surface
harden verifier and artifact-generation CI
```

Long term:

```text
replace jolt-core only after equivalence, perf, soundness, and API gates are mature
deprecate jolt-core deliberately, with migration docs and compatibility policy
```
