# Spec: Production Fuzz Infrastructure

| Field | Value |
|-------|-------|
| Author(s) | Jolt maintainers |
| Created | 2026-07-23 |
| Status | proposed |
| PR | #1696 |

## Scope

One runner discovers crate-local cargo-fuzz workspaces, validates their
configuration, and runs the same build, replay, mutation, minimization, and
coverage commands locally and in CI. Runtime proving and verification behavior
must not change.

[The operator guide](../FUZZING.md) owns commands and state lifecycles.
`uv run python scripts/fuzz.py inventory` owns the current target list and
budgets; neither is duplicated here.

## Contracts

- Fuzz manifests declare targets, focus, feature flags, and positive PR/daily/weekly
  budgets. Discovery rejects missing sources, duplicate names, missing policies,
  and policies without targets. Standard TOML syntax is accepted.
- Each workspace pins its Rust toolchain and commits a lockfile. Resolution uses
  `cargo metadata --locked` and must fail on a nonzero exit. Every selected target
  is compiled with its own declared features, including ZK-only targets.
  Execution uses repository-root cwd and an explicit fuzz manifest directory so
  targets that build guests resolve packages in the main workspace.
- Every target has a checked-in seed. File presence is mechanically checked;
  reaching the input parser and the intended oracle requires semantic review.
- Seeds and regressions are immutable inputs. Replay executes each file once
  before mutation. Coverage-guided corpora, crash artifacts, and build products
  stay outside Git.
- A target failure is reported without skipping remaining selected targets.
  The command exits nonzero if any target fails. Configuration errors stop the
  command before fuzzing that workspace.

## CI and state ownership

```text
manifests -> discovery -> workspace jobs
                            |
                      build -> replay -> mutate
                                            |
                          weekly: minimize -> coverage
                                            |
                          scheduled main: save corpus
```

PR and manual runs may restore trusted corpora but cannot publish new corpus
state. Only successful scheduled main runs save corpora and prune superseded
cache entries. Crashes are uploaded for triage; important fixed cases become
reviewed regression inputs.

Checkout credentials are not persisted. Fuzz jobs have read access to contents
and write access to Actions for cache pruning; the repository token is passed to
that pruning step only. Public artifacts are not an embargoed disclosure channel.

ASan and coverage builds use separate target directories and caches. The runner
passes the coverage directory through cargo-fuzz's explicit `--target-dir` option.
Per-input length, timeout, and RSS limits bound mutation work. Target budgets
must leave room for compilation, replay, minimization, and coverage under the
workflow's job timeout.

## Oracle requirements

Prefer Jolt-owned optimized/reference comparisons, accepted objects with
semantically invalid mutations, and algebraic constraints checked against direct
evaluation. An honest baseline must pass before a must-reject target starts
mutating it. No-op mutations are skipped; changed values are not automatically
invalid statements.

Proof fixture selectors, stage selectors, and round selectors must vary
independently. Scalar payloads should alter existing coefficients without
unnecessarily breaking proof shape. A small fixed mutation catalogue warrants a
short budget; longer campaigns require input-dependent values or measured growth.

## Budget calibration

Budgets are priors until measured. Compare three independent runs per target
using the same starting corpus; record execution rate, edge/features, corpus
size, peak RSS, and semantic depth at 5, 10, and 30 minutes. Increase time only
where additional runs continue reaching useful states. Preserve a smoke budget
for fixed mutation catalogues and stable rejection checks.

Line coverage alone is not a soundness oracle. A parser rejecting arbitrary
bytes quickly does not substitute for reaching later verification stages.

## Verification

Runner tests cover discovery, TOML syntax, feature propagation, resolution
failure, replay inputs, target failures, and coverage-directory selection.
`check --compile` checks every configured target. Real seed replay checks that
fixtures deserialize, honest baselines verify, and known mutations reject.
Regenerate verifier fixtures whenever the proof or preprocessing format changes.

## Alternatives

A single root fuzz crate couples unrelated dependencies and feature sets.
Crate-local workspaces retain those boundaries; manifest discovery removes the
need for a second CI target list. Mutable corpora remain caches rather than Git
history. OSS-Fuzz or ClusterFuzzLite deployment is separate from this local/CI
workflow.
