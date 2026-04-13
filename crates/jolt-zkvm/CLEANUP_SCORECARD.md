# Cleanup Scorecard

**Status**: IN PROGRESS — MIGRATION MODE
**Last updated**: 2026-04-13
**MODE**: SWEEP (finish Tier 1-2), then MIGRATION (Tier 4 via MIGRATION_PLAN.md)
**NEXT ACTION**: 1.13 (unreachable! dispatch paths)

## Loop Protocol

Three modes — see `ARCHITECTURE.md` and `MIGRATION_PLAN.md` for design docs.

```
SWEEP MODE (Tiers 1-3, 5, 6):
  1. Read this file → find NEXT ACTION
  2. Execute it
  3. Run test gate
  4. Pass → commit, mark PASS, advance NEXT ACTION
  5. Fail → fix, goto 3

MIGRATION MODE (Tier 4):
  1. Read MIGRATION_PLAN.md → find NEXT STEP
  2. Execute it
  3. Run test gate + verification greps
  4. Pass → commit, mark step DONE, advance pointer
  5. Fail → fix plan if needed, goto 2
  6. All steps DONE → mark Tier 4 criteria PASS, return to SWEEP

DESIGN MODE (when new architectural issues surface):
  1. Deep-read relevant code
  2. Update ARCHITECTURE.md with target design
  3. Update MIGRATION_PLAN.md with ordered steps
  4. Switch to MIGRATION MODE
```

## Test Gate

```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence zkvm_proof_accepted --cargo-quiet
cargo fmt --check -q
cargo clippy -p jolt-compiler -p jolt-compute -p jolt-cpu -p jolt-zkvm -p jolt-dory -p jolt-openings -p jolt-verifier --message-format=short -q --all-targets -- -D warnings
```

---

## Tier 1 — Hygiene (mechanical, no logic changes)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1.1 | Clippy clean: jolt-compiler | PASS | |
| 1.2 | Clippy clean: jolt-compute | PASS | |
| 1.3 | Clippy clean: jolt-cpu | PASS | |
| 1.4 | Clippy clean: jolt-zkvm | PASS | |
| 1.5 | Clippy clean: jolt-dory | PASS | |
| 1.6 | Clippy clean: jolt-openings | PASS | |
| 1.7 | Clippy clean: jolt-verifier | PASS | |
| 1.8 | cargo fmt clean | PASS | |
| 1.9 | Clippy clean: jolt-core (blocker for equivalence) | PASS | |
| 1.10 | No TODO/FIXME/HACK/XXX comments | PASS | |
| 1.11 | Section separators removed | PASS | 107 removed across 15 files |
| 1.12 | No #[allow(dead_code)] suppressions (delete the dead code) | PASS | 1 legit RAII remain (TracingGuards) |
| 1.13 | No unreachable!() for dispatch paths | FAIL | 7 in jolt-cpu, 5 in jolt-metal — resolved by Tier 4 migration |

## Tier 2 — Simplification (Occam's razor — every line earns its place)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 2.1 | Unnecessary .clone() eliminated in runtime | FAIL | runtime.rs |
| 2.2 | unwrap_or_else → unwrap_or where applicable | FAIL | |
| 2.3 | Verbose match/if-let simplified | TODO | |
| 2.4 | Magic numbers named as constants | FAIL | |
| 2.5 | #[allow(clippy::too_many_arguments)] eliminated | FAIL | 4 occurrences — resolved by Tier 4 migration |
| 2.6 | No placeholders or janky code | TODO | |
| 2.7 | No hacky workarounds | FAIL | SnapshotEval |

## Tier 3 — Crate Boundaries & Visibility

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 3.1 | pub(crate) used for internal-only items | FAIL | Most crates lack discipline |
| 3.2 | Consistent naming across crate boundaries | PASS | |
| 3.3 | Type aliases for repeated generic bounds | TODO | |
| 3.4 | jolt-compiler: pure protocol→ops lowering | PASS | |
| 3.5 | jolt-compute: zero protocol knowledge in trait | FAIL | → MIGRATION_PLAN Step 6 |
| 3.6 | jolt-cpu: zero protocol types in public API | FAIL | → MIGRATION_PLAN Step 6 |
| 3.7 | jolt-zkvm: zero protocol logic in runtime | FAIL | → MIGRATION_PLAN Step 6 |
| 3.8 | jolt-dory: zero Jolt-specific leakage | PASS | |
| 3.9 | Coordinated constants use shared source | FAIL | |
| 3.10 | Clean dependency DAG | PASS | |

## Tier 4 — Architectural (MIGRATION_PLAN.md)

### 4A — Protocol-Unaware Runtime

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.1 | Op enum: no algorithm-named variants | FAIL | → MIGRATION Steps 3-5 |
| 4.2 | ComputeBackend: 1 InstanceState + 4 methods | FAIL | → MIGRATION Steps 1-2 |
| 4.3 | RuntimeState: 1 instance_states map | FAIL | → MIGRATION Step 6 |
| 4.4 | CpuBackend: unified instance dispatch | FAIL | → MIGRATION Step 2 |
| 4.5 | No SnapshotEval workaround needed | FAIL | Separate design needed |
| 4.6 | Scoped evaluation model | FAIL | Separate design needed |

### 4B — Type Safety

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.7 | Challenge indices: typed newtype | TODO | |
| 4.8 | Batch/instance keys: typed | TODO | |
| 4.9 | Dispatch paths compile-time provable | FAIL | → MIGRATION Step 7 |

### 4C — Extensibility

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.10 | New subprotocol = compiler + backend only | FAIL | → MIGRATION Step 6 |
| 4.11 | New backend = one crate only | FAIL | → MIGRATION Step 6 |

### 4D — Abstraction Quality

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.12 | Each abstraction single-purpose | FAIL | → MIGRATION |
| 4.13 | No leaky abstractions | FAIL | → MIGRATION Step 6 |
| 4.14 | Minimal indirection | TODO | |
| 4.15 | Traits have minimal surface area | FAIL | → MIGRATION Step 6 |

## Tier 5 — Production Quality

### 5A — Comments & Documentation

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 5.1 | WHY comments on non-obvious logic | TODO | |
| 5.2 | SAFETY comments on unsafe blocks | TODO | |
| 5.3 | No doc comments restating item name | TODO | |
| 5.4 | Public API docs explain behavior | TODO | |

### 5B — Testing

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 5.5 | All tests compile and pass | PASS | |
| 5.6 | No #[ignore] without justification | TODO | |
| 5.7 | Fuzz targets compile | PASS | |

### 5C — Anti-Patterns

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 5.8 | Runtime never imports protocol types | FAIL | → MIGRATION Step 6 |
| 5.9 | Backend trait uses only generic names | FAIL | → MIGRATION Step 6 |
| 5.10 | State machines encoded in compiler | FAIL | → MIGRATION |
| 5.11 | No out-of-band state | FAIL | SnapshotEval |
| 5.12 | Module is self-describing | TODO | |

## Tier 6 — Systems Engineering Principles

### 6A — Cohesion

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.1 | ComputeBackend orthogonal concerns | FAIL | → MIGRATION |
| 6.2 | runtime.rs < 500 LOC | FAIL | → MIGRATION |
| 6.3 | Each file one reason to change | TODO | |

### 6B — Dependency Inversion

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.4 | High-level depends on abstractions | FAIL | |
| 6.5 | No upward dependencies | PASS | |

### 6C — Information Hiding

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.6 | DeviceBuffer no panics on wrong variant | FAIL | |
| 6.7 | Internal types not in pub API | FAIL | |
| 6.8 | Impl details behind interfaces | FAIL | |

### 6D — Error Handling

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.9 | No .expect() in non-test code | FAIL | |
| 6.10 | Error types for fallible ops | FAIL | |
| 6.11 | Panics only for invariant violations | TODO | |

### 6E — Simplicity

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.12 | Explainable in 5 sentences | FAIL | |
| 6.13 | No speculative abstractions | TODO | |
| 6.14 | Unidirectional data flow | TODO | |

---

## Progress

- **Tier 1**: 12/13 passing (1.13 deferred to migration)
- **Tier 2**: 0/7 passing
- **Tier 3**: 4/10 passing
- **Tier 4**: 0/15 passing (all blocked on MIGRATION_PLAN)
- **Tier 5**: 2/12 passing
- **Tier 6**: 1/14 passing
- **Overall: 19/71 passing**

## Execution Order

1. SWEEP: Tier 2 simplifications (clone, verbose patterns, magic numbers)
2. MIGRATION: Tier 4 via MIGRATION_PLAN.md Steps 1-7
3. SWEEP: Tier 3 remaining (pub visibility, constants)
4. SWEEP: Tier 5 production polish (comments, docs, anti-patterns)
5. SWEEP: Tier 6 systems principles (error handling, information hiding)
