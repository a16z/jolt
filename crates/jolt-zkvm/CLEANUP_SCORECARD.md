# Cleanup Scorecard

**Status**: IN PROGRESS
**Last updated**: 2026-04-13
**NEXT ACTION**: Fix 1.8 (cargo fmt jolt_core_module.rs), then 1.9 (jolt-core clippy)

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

## Tier 1 — Hygiene

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1.1 | Clippy clean: jolt-compiler | PASS | Fixed: is_multiple_of, into_iter, cloned→copied, loop indexing |
| 1.2 | Clippy clean: jolt-compute | PASS | Fixed: #[allow(too_many_arguments)] on bool_init |
| 1.3 | Clippy clean: jolt-cpu | PASS | Fixed: loop indexing, num_traits imports, too_many_arguments |
| 1.4 | Clippy clean: jolt-zkvm | PASS | Fixed: unused variable poly→_poly |
| 1.5 | Clippy clean: jolt-dory | PASS | |
| 1.6 | Clippy clean: jolt-openings | PASS | |
| 1.7 | Clippy clean: jolt-verifier | PASS | |
| 1.8 | cargo fmt clean (all crates) | FAIL | jolt_core_module.rs has 8 formatting diffs |
| 1.9 | Clippy clean: jolt-core (blocker for equivalence) | FAIL | 2 errors: manual_range_contains in read_raf_checking.rs |
| 1.10 | No TODO/FIXME/HACK comments | FAIL | 6 across jolt-dory, jolt-compiler |
| 1.11 | Section separators removed | FAIL | ~104 `// ──` in traits/runtime, ~207 `// [=-]{4,}` in compiler |
| 1.12 | No #[allow(unused)]/dead_code suppressions | FAIL | 12 across 9 files |
| 1.13 | No unreachable!() for dispatch paths | FAIL | 7 in jolt-cpu/src/backend.rs (protocol-aware unreachable arms) |

## Tier 2 — Simplification

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 2.1 | Unnecessary .clone() eliminated | FAIL | 6 in runtime.rs |
| 2.2 | unwrap_or_else → unwrap_or where applicable | FAIL | jolt-cpu generic.rs, jolt-openings mock.rs |
| 2.3 | Verbose match/if-let simplified | TODO | |
| 2.4 | Magic numbers named | FAIL | jolt-compiler params.rs: hardcoded 4, 8 for log_k_chunk |
| 2.5 | #[allow(clippy::ptr_arg)] reviewed | FAIL | 5 in jolt-cpu backend.rs |

## Tier 3 — API & Crate Boundaries

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 3.1 | Unnecessary pub visibility tightened | TODO | |
| 3.2 | Consistent naming across crate boundaries | TODO | |
| 3.3 | Type aliases for repeated generic bounds | TODO | |
| 3.4 | Crate purity: jolt-compiler has zero runtime code | TODO | |
| 3.5 | Crate purity: jolt-compute has zero protocol knowledge | FAIL | 3 protocol types + 12 methods |
| 3.6 | Crate purity: jolt-cpu has zero protocol types in public API | FAIL | Exposes CpuBooleanityState, CpuHwReductionState, CpuPrefixSuffixState |
| 3.7 | Crate purity: jolt-zkvm has zero protocol logic | FAIL | 12 protocol match arms + 3 state fields |
| 3.8 | Crate purity: jolt-dory has zero Jolt-specific leakage | TODO | |
| 3.9 | Coordinated constants (LOG_K, thresholds) use shared source | FAIL | LOG_K=128 in both jolt-cpu and jolt-verifier independently |

## Tier 4 — Architectural (ML Compiler Standard)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.1 | Op enum: collapse 12 protocol variants → 4 generic Instance ops | FAIL | PrefixSuffix(4) + Booleanity(4) + HwReduction(4) |
| 4.2 | ComputeBackend: collapse to 1 InstanceState type + 4 methods | FAIL | Currently 3 types + 12 methods |
| 4.3 | RuntimeState: 1 instance_states map (not 3 typed maps) | FAIL | 3 protocol-specific HashMaps |
| 4.4 | Iteration enum: no protocol-specific variants | FAIL | PrefixSuffix, Booleanity, HammingWeightReduction |
| 4.5 | CpuBackend: unified instance dispatch | FAIL | 12 separate impl methods |
| 4.6 | Eliminate SnapshotEval workaround | FAIL | 12 occurrences across 4 files |
| 4.7 | Scoped evaluation model (not flat HashMap) | FAIL | evaluations: HashMap<PolynomialId, F> single-slot |
| 4.8 | Type-safe challenge indices (newtype not raw usize) | TODO | challenges[usize] throughout |
| 4.9 | Type-safe batch/instance keys | TODO | (usize, usize) throughout |
| 4.10 | Dispatch paths compile-time provable (no unreachable!) | FAIL | 7 unreachable! in jolt-cpu for protocol dispatch |

## Tier 5 — ML Compiler Generality

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 5.1 | Any valid Module executes correctly (runtime = correct interpreter) | FAIL | Protocol-specific ops break this |
| 5.2 | New subprotocol = local change (no shotgun surgery) | FAIL | Currently: Op + Backend + RuntimeState + CpuBackend changes |
| 5.3 | Clean dependency DAG (no circular deps, minimal coupling) | TODO | |
| 5.4 | Zero protocol concepts visible to backend implementors | FAIL | Backend must know PrefixSuffix/Booleanity/HW |
| 5.5 | Module is self-describing (no out-of-band state needed) | TODO | |

---

## Progress

- **Tier 1**: 7/13 passing
- **Tier 2**: 0/5 passing
- **Tier 3**: 0/9 passing
- **Tier 4**: 0/10 passing
- **Tier 5**: 0/5 passing
- **Overall: 7/42 passing**
