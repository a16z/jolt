# Cleanup Scorecard

**Status**: IN PROGRESS
**Last updated**: 2026-04-13
**NEXT ACTION**: 1.11 (Section separators)

## Loop Protocol

```
1. READ this file
2. Execute NEXT ACTION
3. Run test gate
4. If pass: commit, mark criterion PASS, update NEXT ACTION to next FAIL
5. If fail: diagnose, fix, goto 3
6. If all PASS: DONE
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
| 1.10 | No TODO/FIXME/HACK/XXX comments | PASS | 0 found |
| 1.11 | Section separators removed (// ──, // ===, // ---) | FAIL | 107 across 15 files |
| 1.12 | No #[allow(unused)]/dead_code suppressions | FAIL | 4 across 4 files |
| 1.13 | No unreachable!() for dispatch paths | FAIL | 7 in jolt-cpu, 5 in jolt-metal |

## Tier 2 — Simplification (Occam's razor — every line earns its place)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 2.1 | Unnecessary .clone() eliminated in runtime | FAIL | 6 clones in runtime.rs |
| 2.2 | unwrap_or_else → unwrap_or where applicable | FAIL | jolt-cpu generic.rs, jolt-openings mock.rs |
| 2.3 | Verbose match/if-let simplified | TODO | |
| 2.4 | Magic numbers named as constants | FAIL | jolt-compiler params.rs: hardcoded 4, 8 |
| 2.5 | #[allow(clippy::ptr_arg)] reviewed | FAIL | 5 in jolt-cpu backend.rs |
| 2.6 | No placeholders or janky code | TODO | Full scan needed |
| 2.7 | No hacky workarounds (prefer clean redesign) | FAIL | SnapshotEval is a workaround for flat eval map |

## Tier 3 — Crate Boundaries & Visibility (each crate does one job, hides its internals)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 3.1 | pub(crate) used for internal-only items | FAIL | Only jolt-dory has good discipline (11 pub(crate)) |
| 3.2 | Consistent naming across crate boundaries | PASS | Minor: ProverConfig vs OneHotConfig |
| 3.3 | Type aliases for repeated generic bounds | TODO | |
| 3.4 | jolt-compiler: zero runtime code, only protocol→ops lowering | PASS | Clean |
| 3.5 | jolt-compute: zero protocol knowledge in trait | FAIL | 38-method trait with protocol-specific state machines |
| 3.6 | jolt-cpu: zero protocol types in public API | FAIL | Exposes Cpu{Booleanity,HwReduction,PrefixSuffix}State |
| 3.7 | jolt-zkvm: zero protocol logic in runtime | FAIL | 12 match arms + 3 state fields |
| 3.8 | jolt-dory: zero Jolt-specific leakage | PASS | Clean |
| 3.9 | Coordinated constants use shared source | FAIL | LOG_K=128 duplicated |
| 3.10 | Clean dependency DAG (no circular, minimal coupling) | PASS | No cycles detected |

## Tier 4 — Architectural (ML Compiler of Cryptography)

### 4A — Protocol-Unaware Runtime

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.1 | Op enum: no algorithm-named variants | FAIL | 12 variants: PrefixSuffix(4) + Booleanity(4) + HwReduction(4) |
| 4.2 | ComputeBackend: single InstanceState type + 4 methods | FAIL | Currently 3 types + 38 methods |
| 4.3 | RuntimeState: 1 instance_states map | FAIL | 3 protocol-specific HashMaps |
| 4.4 | Iteration enum: no protocol-specific variants | FAIL | PrefixSuffix, Booleanity, HammingWeightReduction |
| 4.5 | CpuBackend: unified instance dispatch | FAIL | 12 separate impl methods |
| 4.6 | No SnapshotEval workaround needed | FAIL | 12 occurrences across 4 files |
| 4.7 | Scoped evaluation model (not flat HashMap) | FAIL | Single-slot per poly, stages overwrite |

### 4B — Type Safety (compile-time correctness > runtime checks)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.8 | Challenge indices: typed newtype not raw usize | TODO | challenges[usize] throughout |
| 4.9 | Batch/instance keys: typed not (usize, usize) | TODO | |
| 4.10 | Dispatch paths compile-time provable (no unreachable!) | FAIL | 7 unreachable! in jolt-cpu |

### 4C — Extensibility (new subprotocol = local change, not shotgun surgery)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.11 | Adding a subprotocol touches only compiler + backend impl | FAIL | Currently: Op + Backend trait + RuntimeState + CpuBackend |
| 4.12 | Adding a backend touches only the new backend crate | FAIL | Must implement 38 protocol methods |

### 4D — Abstraction Quality (Occam's razor applied to design)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.13 | Each abstraction has a clear, single purpose | FAIL | ComputeBackend mixes kernel dispatch + buffer ops + state machines |
| 4.14 | No leaky abstractions (protocol doesn't leak through generic interfaces) | FAIL | BooleanityConfig visible through ComputeBackend |
| 4.15 | Minimal indirection (fewest layers between intent and execution) | TODO | |
| 4.16 | Traits have minimal surface area | FAIL | ComputeBackend: 38 methods |

## Tier 5 — Production Quality

### 5A — Comments & Documentation

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 5.1 | WHY comments on all non-obvious logic | TODO | |
| 5.2 | SAFETY comments on all unsafe blocks | TODO | |
| 5.3 | No doc comments that restate the item name | TODO | |
| 5.4 | Public API docs explain behavior + constraints | TODO | |

### 5B — Testing

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 5.5 | All tests compile and pass | PASS | Both equivalence tests pass |
| 5.6 | No #[ignore] without justification | TODO | |
| 5.7 | Fuzz targets compile | PASS | Fixed transcript_no_panic type inference |

### 5C — Anti-Patterns (ML compiler design violations)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 5.8 | Runtime never imports protocol-specific types | FAIL | Imports BooleanityConfig, HwReductionConfig |
| 5.9 | Backend never sees algorithm names (only generic operations) | FAIL | ps_init, bool_bind, hw_reduce |
| 5.10 | State machines encoded in compiler, not runtime | FAIL | PS/Bool/HW state machines live in runtime |
| 5.11 | No out-of-band state (all state flows through Module) | FAIL | SnapshotEval patches state outside Module flow |
| 5.12 | Module is self-describing (executable without external context) | TODO | |

## Tier 6 — Systems Engineering Principles

### 6A — Single Responsibility & Cohesion

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.1 | jolt-compute trait separable into orthogonal concerns | FAIL | KernelDispatch + BufferOps + ReductionState mixed in 1 trait |
| 6.2 | jolt-zkvm runtime.rs < 500 LOC | FAIL | ~1600 LOC, multiple concerns |
| 6.3 | Each file has one reason to change | TODO | |

### 6B — Dependency Inversion

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.4 | High-level modules depend on abstractions, not concretions | FAIL | jolt-zkvm imports jolt-compiler IR types directly |
| 6.5 | No upward dependencies (leaf crates stay leaves) | PASS | No cycles |

### 6C — Information Hiding

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.6 | DeviceBuffer doesn't panic on wrong variant access | FAIL | 4 panics in accessor methods |
| 6.7 | Internal state types not in pub API | FAIL | CpuKernel, BoxedEvalFn, EvalFn exposed |
| 6.8 | Implementation details behind well-defined interfaces | FAIL | runtime.rs pattern-matches on compiler IR inline |

### 6D — Error Handling

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.9 | No .expect() in non-test, non-setup code | FAIL | 23 in runtime.rs, 10 in jolt-dory |
| 6.10 | Error types for all fallible operations | FAIL | jolt-compute has no error type |
| 6.11 | Panics only for true invariant violations | TODO | |

### 6E — Simplicity (the design you'd draw on a whiteboard)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 6.12 | Could explain the architecture in 5 sentences | FAIL | Protocol-aware escape hatches complicate the story |
| 6.13 | No abstraction exists just for "flexibility" | TODO | |
| 6.14 | Data flows in one direction through the pipeline | TODO | |

---

## Progress

- **Tier 1**: 10/13 passing
- **Tier 2**: 0/7 passing
- **Tier 3**: 4/10 passing
- **Tier 4**: 0/16 passing
- **Tier 5**: 2/12 passing
- **Tier 6**: 1/14 passing
- **Overall: 17/72 passing**
