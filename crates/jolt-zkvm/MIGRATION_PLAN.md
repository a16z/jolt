# Migration Plan: Unified Instance State

**Status**: NOT STARTED
**NEXT STEP**: Step 1

## Overview

Collapse 3 protocol-specific state machines into 1 generic instance abstraction.
See `ARCHITECTURE.md` for the target design.

**Invariant**: Every step produces a compilable, test-passing codebase.

---

## Step 1: Define new types (additive only)

**What**: Add `InstanceConfig`, `InstanceOutput`, unified `Op` variants, and
new `ComputeBackend` methods alongside the existing ones. Nothing is removed.

**Files**:
- `crates/jolt-compiler/src/module.rs` — add `InstanceConfig` enum, 4 new `Op` variants
- `crates/jolt-compute/src/traits.rs` — add `InstanceState<F>` type, `InstanceOutput`,
  4 new methods (`instance_init/bind/reduce/finalize`) with default impls that panic
  (so existing backends compile without changes)

**Verify**: `cargo clippy -p jolt-compiler -p jolt-compute` clean. All tests pass
(nothing uses the new types yet).

**Commit**: `refactor(compute): add unified InstanceState trait interface`

---

## Step 2: Implement for CpuBackend

**What**: Implement the 4 new methods on CpuBackend. Each delegates to the
existing internal implementations. Define `CpuInstanceState<F>` enum.

**Files**:
- `crates/jolt-cpu/src/backend.rs` — add `CpuInstanceState` enum, implement
  `instance_init/bind/reduce/finalize` by matching on config/state variant
  and delegating to existing `ps_*/bool_*/hw_*` private methods

**Key details**:
- `instance_reduce` for PrefixSuffix must compute `eval_1 = previous_claim - eval_0`
  internally (currently done in runtime.rs). This moves protocol logic into the
  backend where it belongs.
- `instance_finalize` for PrefixSuffix returns `InstanceOutput { buffers: ..., evaluations: [] }`.
  For Booleanity/HwReduction returns `InstanceOutput { buffers: [], evaluations: ... }`.

**Verify**: `cargo clippy -p jolt-cpu` clean. Existing tests still pass
(old methods still exist and are still called).

**Commit**: `refactor(cpu): implement unified instance dispatch`

---

## Step 3: Migrate PrefixSuffix through the stack

**What**: Change compiler emission + runtime dispatch for PrefixSuffix from
old ops to new generic ops. Delete old PS-specific code paths.

**Files**:
- `crates/jolt-compiler/src/compiler/emit.rs` — emit `Op::InstanceInit { config: InstanceConfig::PrefixSuffix { kernel } }` instead of `Op::PrefixSuffixInit`
  - Same for Bind/Reduce/Materialize → InstanceBind/Reduce/Finalize
- `crates/jolt-zkvm/src/runtime.rs` — add dispatch for new `Op::InstanceInit/Bind/Reduce/Finalize`,
  remove dispatch for `Op::PrefixSuffixInit/Bind/Reduce/Materialize`
  - `RuntimeState`: add `instance_states` HashMap, keep old 3 maps for now
- `crates/jolt-compiler/src/module.rs` — can keep old Op variants (dead but present)
  until all 3 are migrated

**Verify**: Transcript parity + cross-system verification pass. The bytes on
the wire are identical because the same math runs in the same order.

**Commit**: `refactor(zkvm): migrate PrefixSuffix to unified instance ops`

---

## Step 4: Migrate Booleanity through the stack

**What**: Same as Step 3 but for Booleanity.

**Files**: Same files as Step 3.

**Key detail**: Challenge resolution (the 5 `iter().map(|&i| challenges[i]).collect()`
calls in the current `BooleanityInit` handler) moves into `CpuBackend::instance_init`.
The `InstanceConfig::Booleanity` variant carries challenge slot indices; the backend
resolves them using the `challenges` slice passed to `instance_init`.

**Verify**: Transcript parity + cross-system verification pass.

**Commit**: `refactor(zkvm): migrate Booleanity to unified instance ops`

---

## Step 5: Migrate HwReduction through the stack

**What**: Same pattern. After this step, all 3 state machines use the unified path.

**Files**: Same files.

**Key detail**: The dummy `bool_claims` and `virt_claims` (currently `vec![F::zero(); total]`)
move into the backend's init. The long comment block (lines 1235-1282 in runtime.rs)
about claim computation also moves — it's protocol knowledge that belongs in the backend.

**Verify**: Transcript parity + cross-system verification pass.

**Commit**: `refactor(zkvm): migrate HwReduction to unified instance ops`

---

## Step 6: Delete old code

**What**: Now that all 3 use the unified path:
- Delete 12 old `Op` variants from module.rs
- Delete 3 associated types + 12 methods from `ComputeBackend`
- Delete 3 HashMaps from `RuntimeState`
- Delete old method impls from `CpuBackend`
- Delete `BooleanityConfig` and `HwReductionConfig` types (absorbed into InstanceConfig)
- Remove `#[allow(clippy::too_many_arguments)]` on deleted methods
- Update jolt-metal if it implements these (likely stubs/unreachable)

**Files**:
- `crates/jolt-compiler/src/module.rs`
- `crates/jolt-compute/src/traits.rs`
- `crates/jolt-cpu/src/backend.rs`
- `crates/jolt-zkvm/src/runtime.rs`
- `crates/jolt-metal/src/backend.rs` (if applicable)

**Verify**:
```bash
# Protocol names gone from runtime:
grep -c "PrefixSuffix\|Booleanity\|HwReduction" crates/jolt-zkvm/src/runtime.rs  # → 0
# Old methods gone from trait:
grep -c "ps_init\|bool_init\|hw_init" crates/jolt-compute/src/traits.rs  # → 0
# Tests still pass:
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence zkvm_proof_accepted --cargo-quiet
```

**Commit**: `refactor(compute): remove protocol-specific state machine interfaces`

---

## Step 7: Simplify and tighten

**What**: Post-migration cleanup:
- Remove `unreachable!()` dispatch paths in old CpuBackend (now replaced by enum match)
- Tighten `pub` → `pub(crate)` on internal types
- Remove stale comments referencing old architecture
- Update CLEANUP_SCORECARD.md: mark Tier 4A criteria as PASS

**Verify**: Full test gate.

**Commit**: `refactor(zkvm): post-migration cleanup`

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Transcript divergence after migration | Each step runs both equivalence tests. Same math, same order. |
| CpuBackend instance_init signature too wide | `provider` + `challenges` + `lookup_trace` covers all 3 subprotocols |
| Metal backend breaks | Metal stubs these methods; update stubs to match new signature |
| Context loss mid-migration | This file tracks NEXT STEP; each step is self-contained |

## LOC Impact Estimate

| Area | Before | After | Delta |
|------|--------|-------|-------|
| Op variants (protocol-specific) | 12 | 0 | -12 |
| Op variants (generic) | 0 | 4 | +4 |
| ComputeBackend methods | 12 | 4 | -8 |
| ComputeBackend assoc types | 3 | 1 | -2 |
| RuntimeState fields | 3 | 1 | -2 |
| runtime.rs match arms | ~250 LOC | ~30 LOC | -220 |
| Config types | 2 standalone | 0 (absorbed) | -2 types |
