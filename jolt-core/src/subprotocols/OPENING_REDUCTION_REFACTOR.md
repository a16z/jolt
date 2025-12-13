# Opening Reduction Refactor - Progress Tracker

This document tracks the progress of the opening reduction optimization. The goal is to eliminate the expensive generic opening reduction sumcheck (Stage 7) by:
1. Using `IncReduction` to align dense polynomial claims (RamInc, RdInc) in Stage 6
2. Moving HammingWeight to a new Stage 7 that uses `r_cycle_stage6` from Stage 6
3. Fusing HammingWeight with Address Reduction to align all ra_i claims

## 🎉 STATUS: E2E TESTS PASSING (Dec 13, 2025)

The major refactoring is done. All e2e tests pass when run individually.

**Latest Fix:** Moved `RamHammingBooleanity` from Stage 5 to Stage 6 so it shares `r_cycle_stage6`.
This allows the RAM HW claims to be fetched from the accumulator directly, eliminating the need for
`ram_hw_claims` as a separate field in the proof.

**Verified passing tests:**
- `fib_e2e_dory` ✅
- `memory_ops_e2e_dory` ✅  
- `muldiv_e2e_dory` ✅
- `small_trace_e2e_dory` ✅
- `sha2_e2e_dory` ✅
- `sha3_e2e_dory` ✅
- `advice_e2e_dory` ✅
- `btreemap_e2e_dory` ✅

**Note:** Tests fail when run together due to pre-existing `DoryGlobals` singleton issue (not related to this refactor).

Remaining work:
- Unit tests for sumcheck correctness
- Optional cleanup of deprecated code

## High-Level Goal

**Before optimization:**
- Stage 7 runs a generic opening reduction sumcheck over ALL committed polynomials
- This is expensive: O(N × log(K) × log(T)) where N = number of polynomials

**After optimization:**
- Stage 6 aligns dense claims (Inc polynomials) to `r_cycle_stage6`
- Stage 7 runs a cheap fused HammingWeight + Address Reduction over only `log_k_chunk ≤ 8` rounds
- All committed polynomials share `r_cycle_stage6` → go directly to Dory

## Current Stage Layout (IMPLEMENTED ✅)

```
Stage 4:
  - RegistersReadWriteChecking (emits RdInc claim at s_cycle_stage4)
  - RamValEvaluation (emits RamInc claim at r_cycle_stage4)
  - RamValFinal (emits RamInc claim at r_cycle_stage4)

Stage 5:
  - RegistersValEvaluation (emits RdInc claim at s_cycle_stage5)
  - RamRaReduction
  - InstructionReadRaf (emits InstructionRa virtual claim)

Stage 6: (all sumchecks share r_cycle_stage6) ✅
  - BytecodeReadRaf
  - BytecodeBooleanity
  - RamHammingBooleanity (moved from Stage 5 to share r_cycle_stage6!) ✅
  - RamBooleanity
  - RamRaVirtualization
  - InstructionRaVirtualization
  - InstructionBooleanity
  - IncReduction

Stage 7: (uses r_cycle_stage6 via accumulator, produces r_address_stage7 via log_k_chunk rounds) ✅
  - HammingWeightClaimReduction (fused HammingWeight + Address Reduction)
  - Produces DoryOpeningState with unified opening point
  - RAM HW claims fetched from Stage 6's RamHammingBooleanity via accumulator ✅

Stage 8: ✅
  - Dory opening proof (on aligned claims via DoryOpeningState)
```

## Claim Flow After Optimization

**After Stage 6:**
- RamInc: 1 claim at r_cycle_stage6 (from IncReduction)
- RdInc: 1 claim at r_cycle_stage6 (from IncReduction)
- BytecodeRa(i): 2 claims at (r_addr_bool, r_cycle_stage6), (r_addr_readraf, r_cycle_stage6)
- InstructionRa(i): 2 claims at (r_addr_bool, r_cycle_stage6), (r_addr_virt, r_cycle_stage6)
- RamRa(i): 2 claims at (r_addr_bool, r_cycle_stage6), (r_addr_virt, r_cycle_stage6)

**After Stage 7:**
- RamInc: 1 claim at r_cycle_stage6 (unchanged)
- RdInc: 1 claim at r_cycle_stage6 (unchanged)
- BytecodeRa(i): 1 claim at (ρ_addr, r_cycle_stage6)
- InstructionRa(i): 1 claim at (ρ_addr, r_cycle_stage6)
- RamRa(i): 1 claim at (ρ_addr, r_cycle_stage6)

All committed polynomials share r_cycle_stage6!

## Implementation Progress

### ✅ COMPLETED (Dec 13, 2025)

1. **`inc_reduction.rs`** - Dense polynomial reduction sumcheck
   - Reduces 5 claims (3 RamInc + 2 RdInc) to 2 claims at `r_cycle_stage6`
   - Uses prefix-suffix optimization for first half of rounds
   - Integrated into Stage 6 in `prover.rs` and `verifier.rs`
   - **STATUS**: ✅ WORKING (e2e tests pass)

2. **`hamming_weight_claim_reduction.rs`** - Fused sumcheck
   - `HammingWeightClaimReductionParams` - parameters for the fused sumcheck
   - `HammingWeightClaimReductionProver` - prover implementation
   - `HammingWeightClaimReductionVerifier` - verifier implementation
   - `compute_all_G` - computes pushforward polynomials from trace
   - **STATUS**: ✅ WORKING (e2e tests pass)

3. **`opening_proof.rs`** changes:
   - Added `DoryOpeningState<F>` - minimal state for Dory opening
   - Added `SumcheckId::HammingWeightClaimReduction`
   - Added `dory_opening_state` field to both accumulators

4. **Stage 6 changes** (`prover.rs` and `verifier.rs`):
   - ✅ Removed `bytecode_hamming_weight` from instances
   - ✅ Removed `ram_hamming_weight` from instances
   - ✅ Removed `lookups_ra_hamming_weight` from instances
   - ✅ Added `gen_ra_booleanity_prover` and `new_ra_booleanity_verifier` helper functions

5. **Stage 7 changes** (`prover.rs` and `verifier.rs`):
   - ✅ Replaced generic opening reduction with HammingWeightClaimReduction
   - ✅ Gets `r_cycle_stage6` via opening accumulator
   - ✅ Runs sumcheck for `log_k_chunk` rounds only
   - ✅ Constructs `DoryOpeningState` with unified opening point

6. **Stage 8 changes** (`prover.rs` and `verifier.rs`):
   - ✅ Uses `DoryOpeningState` instead of `OpeningReductionState`
   - ✅ Builds streaming RLC directly from trace (no witness poly regeneration!)
   - ✅ Uses `RLCPolynomial::new_streaming` for true streaming

### 🔲 TODO (Remaining Work)

#### Testing

1. **Unit tests for HammingWeightClaimReduction**:
   - [ ] Test `compute_all_G` against naive implementation
   - [ ] Test sumcheck prover/verifier correctness

2. **Unit tests for IncReduction**:
   - [ ] Test sumcheck correctness

3. **E2E tests**:
   - [x] Run existing Jolt e2e tests (fibonacci, sha2, etc.) - ALL PASSING ✅
   - [x] Fix integration bugs - FIXED ✅

#### Cleanup (Optional)

4. **Remove obsolete code**:
   - [ ] Remove or deprecate `opening_reduction.rs`
   - [ ] Remove `OpeningReductionState` once fully migrated
   - [ ] Clean up unused HammingWeight sumcheck code

5. **Optional: Unify Booleanity sumchecks**:
   - [ ] Create single `BooleanitySumcheck` for all families (Instruction, Bytecode, Ram)
   - [ ] Share eq tables across all ra_i within a family
   - [ ] May improve cache efficiency

## Key Design Decisions

### r_cycle_stage6 propagation

**Decision**: Pass `r_cycle_stage6` via opening accumulator, NOT prover state.

After Stage 6, the Booleanity/Virtualization sumchecks append claims with opening points
containing r_cycle_stage6. Stage 7 retrieves r_cycle by getting any RA claim's opening 
point from the accumulator and extracting the cycle portion.

Example:
```rust
// In Stage 7 params initialization:
let (point, _) = accumulator.get_committed_polynomial_opening(
    CommittedPolynomial::BytecodeRa(0),
    SumcheckId::BytecodeBooleanity,
);
let r_cycle_stage6 = point.r[log_k_chunk..].to_vec();
```

### G_i computation

**Decision**: Compute G_i in Stage 7's `initialize()`, NOT in Stage 6.

G_i polynomials are the "pushforward" of ra_i over r_cycle:
```
G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
```

These are ONLY needed for Stage 7. Computing them requires:
1. Access to the trace (for ra_i indices)
2. r_cycle_stage6 (from Stage 6 via accumulator)

The `compute_all_G` function in `hamming_weight_claim_reduction.rs` does this in a 
single streaming pass over the trace.

### HammingWeight claims

**Decision**: HammingWeight claims are NOT stored separately.

In the fused sumcheck, HammingWeight is verified implicitly:
- `input_claim` includes `Σ_i γ^{3i} · H_i` where H_i is the expected hamming weight
- The sumcheck proves `Σ_k G_i(k) = H_i` as part of the fused relation
- No need to store separate HammingWeight openings in the accumulator

### Why fuse HammingWeight with Address Reduction?

The G_i polynomial (pushforward of ra_i over r_cycle) is shared:
- HammingWeight: Σ_k G_i(k) = H_i
- Booleanity reduction: Σ_k eq(r_addr_bool, k)·G_i(k) = claim_bool_i
- Virtualization reduction: Σ_k eq(r_addr_virt_i, k)·G_i(k) = claim_virt_i

By batching with γ, we fuse all three into one degree-2 sumcheck!

### Degree analysis

- G_i(k) contributes degree 1
- eq(r_addr, k) contributes degree 1 (or 0 for HammingWeight constant term)
- Maximum: 1 + 1 = 2

Same degree as address reduction alone - fusion is free!

## Stage 7/8 Dory Integration (Important!)

### How Dory Gets the Opening Point Currently

**Stage 7 (current):**
1. Regenerate witness polynomials
2. Run opening reduction sumcheck (`log_K + log_T` rounds)
3. Creates `OpeningReductionState`:
   - `r_sumcheck` = unified point from sumcheck challenges
   - `gamma_powers` = RLC coefficients (sampled after sumcheck)
   - `sumcheck_claims` = per-polynomial claims
   - `polynomials` = list of committed polynomials

**Stage 8 (current):**
1. Build RLC polynomial: `Σ γ_i · poly_i`
2. Dory prove: Opens RLC at `state.r_sumcheck`

**Key: `compute_joint_claim` handles different-sized polynomials:**
```rust
// Dense polys (log_T vars): first log_K challenges contribute Lagrange factor
let r_slice = &state.r_sumcheck[..max_rounds - poly_rounds];
let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
```

### New Stage 7/8 Flow

**After new Stage 6:**
- Dense polys: opening at `r_cycle_stage6` (length `log_T`)
- Sparse polys: openings at `(r_addr_*, r_cycle_stage6)` (different r_addr per claim type)

**After new Stage 7 (HammingWeightClaimReduction):**
- Dense polys: opening at `r_cycle_stage6` (unchanged)
- Sparse polys: opening at `(ρ_addr, r_cycle_stage6)` (SAME r_addr now!)

**Unified opening point: `(ρ_addr, r_cycle_stage6)`** of length `log_K + log_T`.

**New Stage 7:**
1. Run `HammingWeightClaimReduction` sumcheck (only `log_K_chunk` rounds!)
   - Produces `ρ_addr` challenges
   - Produces `G_i(ρ_addr)` claims for each ra_i via `cache_openings`
2. Collect all claims:
   - From IncReduction: `RamInc(r_cycle_stage6)`, `RdInc(r_cycle_stage6)`
   - From HammingWeightClaimReduction: `G_i(ρ_addr)` for each ra_i
3. Construct `OpeningReductionState` **directly** (NO opening reduction sumcheck!):
   - `r_sumcheck = (ρ_addr, r_cycle_stage6)` (concatenated, big-endian)
   - `sumcheck_claims` = collected claims
   - `gamma_powers` = sample from transcript
   - `polynomials` = [RamInc, RdInc, InstructionRa(0), ..., RamRa(d-1)]

**New Stage 8:**
- **Regenerate witness polynomials** (still needed for Dory proof)
- Build RLC polynomial using the state
- Dory prove at `state.r_sumcheck`

**For `compute_joint_claim` (verifier):**
- Dense polys (log_T vars) get Lagrange factor: `∏_{i<log_K} (1 - ρ_addr_i)`
- Sparse polys (log_K + log_T vars) get Lagrange factor: `1`

This works because all sparse polys NOW share the same `ρ_addr`!

### Witness Regeneration & Streaming

**Key optimization**: Stage 8 now streams directly from trace via `RLCPolynomial::new_streaming`. 
No witness polynomial regeneration is needed!

- OLD: Regen witnesses (Stage 7) → Run opening reduction sumcheck → Build RLC → Dory
- NEW: Run HammingWeightClaimReduction (trace-based) → Build streaming RLC directly → Dory

Both the expensive opening reduction sumcheck AND witness regeneration are eliminated!

## Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `subprotocols/inc_reduction.rs` | ✅ Wired | Dense Inc reduction sumcheck, needs testing |
| `subprotocols/hamming_weight_claim_reduction.rs` | ✅ Wired | Fused HW + Address reduction |
| `subprotocols/mod.rs` | ✅ Updated | Exports new modules, deprecation comments |
| `poly/opening_proof.rs` | ✅ Updated | Added `DoryOpeningState`, `SumcheckId::HammingWeightClaimReduction` |
| `poly/rlc_polynomial.rs` | ✅ Updated | Added `TraceSource` enum, `new_streaming` with single-pass VMV for materialized trace |
| `subprotocols/opening_reduction.rs` | ⚠️ DEPRECATED | No longer used, can be removed in cleanup PR |
| `zkvm/prover.rs` | ✅ Updated | New Stage 7 (HammingWeightClaimReduction), Stage 8 (streaming) |
| `zkvm/verifier.rs` | ✅ Updated | New Stage 7/8 verification |
| `zkvm/bytecode/mod.rs` | ✅ Updated | Added `gen_ra_booleanity_prover`, `new_ra_booleanity_verifier` |
| `zkvm/instruction_lookups/mod.rs` | ✅ Updated | Added `gen_ra_booleanity_prover`, `new_ra_booleanity_verifier` |

## Dead Code (can be removed in cleanup PR)

- `OpeningReductionState` struct (replaced by `DoryOpeningState`)
- `opening_reduction_state` fields in accumulators
- `sumchecks: Vec<OpeningProofReductionSumcheckProver>` (populated but never used)
- Old Stage 7 methods: `prove_batch_opening_sumcheck`, `verify_batch_opening_sumcheck`, etc.
- `opening_reduction.rs` module

## Dead Code Already Removed

- [x] `EqAddressState`, `EqCycleState`, `OneHotPolynomialProverOpening` from `poly/one_hot_polynomial.rs`
- [x] Associated tests (`dense_polynomial_equivalence`, `sumcheck_K_less_than_T`, etc.)
- [x] Unused fields (`G`, `H`, `num_variables_bound`) from `OneHotPolynomial` struct
- [x] `generate_witness_batch` function from `zkvm/witness.rs`
- [x] `WitnessData`, `SharedWitnessData` structs from `zkvm/witness.rs`
- [x] `RLCPolynomial::linear_combination` function (replaced by `new_streaming`)
- [x] Added `TraceSource` enum with `Materialized(Arc<Vec<Cycle>>)` and `Lazy(LazyTraceIterator)` variants
- [x] Updated `new_streaming` to take `TraceSource` instead of `LazyTraceIterator`
- [x] Implemented single-pass parallel VMV for materialized trace (default, efficient)

## Testing

- [ ] Unit tests for `HammingWeightClaimReductionProver` sumcheck correctness
- [ ] Unit tests comparing `compute_all_G` against naive computation
- [ ] Unit tests for `IncReduction` sumcheck correctness
- [ ] Integration test: full e2e proof with new Stage 7
- [ ] Benchmark comparison: old vs new opening reduction cost

## Documentation

- [ ] Update `book/src/how/architecture/opening-proof.md` - references old `OneHotPolynomialProverOpening`
      and the old "Layer 2" opening reduction sumcheck which no longer exists
