# ReadRaf Stages: Implementation Plan

**Date:** 2026-03-12
**Branch:** `refactor/crates`
**Context:** Final two missing sumcheck instances (BytecodeReadRaf + InstructionReadRaf)

---

## Overview

Both ReadRaf stages prove a double-sum identity over (address k, cycle j):

```
claimed_sum = Σ_{j,k} f(k, j)
```

where `f` involves an RA product `Π_i ra_i(k_chunk_i, j)`, per-stage eq polynomials,
and value polynomials derived from bytecode/lookup tables.

Both stages split into two phases via `SegmentedEvaluator`:
- **Address phase** (log_K rounds): binds address variables. Custom kernel, degree 2.
- **Cycle phase** (log_T rounds): binds cycle variables. ProductSum(d, 1), ToomCook, degree d+1.

The address phase uses pre-computed F polynomials that absorb the cycle-domain
eq and RA contributions (computed from trace). The cycle phase materializes RA
polynomials and uses a combined eq weight buffer.

---

## BytecodeReadRaf

### Mathematical Identity

```
input_claim = Σ_{j,k} ra(k,j) · [ Σ_{s=1}^{5} γ^{s-1}·eq_s(j)·Val'_s(k)
                                    + γ_entry·eq_entry(j)·f_trace(k)·f_expected(k) ]
```

Where:
- `ra(k,j) = Π_{i=0}^{d-1} ra_i(k_chunk_i, j)` — one-hot product over address chunks
- `eq_s(j) = eq(r_cycle_s, j)` — per-stage eq polynomial at prior challenge point
- `Val'_s(k) = Val_s(k) + int_correction_s · Int(k)` — per-stage value polynomial with
  RAF identity contributions folded in (stages 1 and 3 have Int terms)
- `f_trace(k)` = one-hot at `PC(cycle_0)`, `f_expected(k)` = one-hot at `entry_bytecode_index`
- `eq_entry(j) = eq(0, j)` — indicator for cycle 0
- `γ_entry = γ^7` — entry constraint weight

**5 stages** virtualize claims from: (1) Spartan outer, (2) product virtualization,
(3) shift sumcheck, (4) register RW checking, (5) register val eval + instruction lookups.

**N_STAGES = 5**, `d = bytecode_d` (typically 2-3), `K = bytecode_k`, `T = trace_length`.

### Address Phase — Segment 0

**Rounds:** log_K, **Degree:** 2, **Binding:** LowToHigh, **Mode:** StandardGrid

The address phase sums over k. The cycle domain is pre-absorbed into F_s(k):

```
F_s(k) = Σ_j eq_s(j) · ra(k, j)
```

Computed via the **split-eq optimization** (O(T) additions + O(touched_PCs × K) muls):

```
F_s(k) = Σ_{c_hi} E_hi[c_hi] · (Σ_{c_lo : PC(c)=k} E_lo[c_lo])
```

Where `E_hi`, `E_lo` are the halved eq tables for each stage's `r_cycle_s`.

**Kernel formula** (Custom, 12 inputs, baked challenges):

```
kernel(i) = Σ_{s=0}^{4} γ^s · F_s(i) · Val'_s(i) + γ_entry · f_trace(i) · f_expected(i)
```

All terms are degree 2 (product of two openings). Challenges `γ^0..γ^4, γ_entry` baked
at compile time.

**Inputs** (12 buffers):
| Index | Buffer | Size |
|-------|--------|------|
| 0-4 | F_0, ..., F_4 | K |
| 5-9 | Val'_0, ..., Val'_4 | K |
| 10 | f_entry_trace | K |
| 11 | f_entry_expected | K |

**Weights:** None (unit weights, no eq in address phase).

### Transition

At round log_K, the address variables are fully bound at `r_addr`. The transition:

1. **Evaluate bound scalars:**
   - `bound_val_s = Val'_s(r_addr)` for each stage — evaluate multilinear at bound point
   - `bound_f_entry = f_expected(r_addr)` — evaluate at bound point
   - The address round poly evaluations carry these implicitly via the running claim.

2. **Materialize RA polynomials** from trace + `r_addr_chunks`:
   - `r_addr_chunks = one_hot_params.compute_r_address_chunks(r_addr)`
   - For each chunk i: `ra_i(j) = eq(r_addr_chunk_i, PC_chunk_i(j))`
   - Each ra_i is a multilinear polynomial over cycle variables (length T)
   - Use `RaPolynomial::new(indices, eq_table)` → lazy materialization

3. **Compute combined eq weight buffer:**
   ```
   combined_eq[j] = Σ_{s=0}^{4} γ^s · bound_val_s · eq_s(j)
                  + γ_entry · bound_f_entry · eq_entry(j)
   ```
   - `eq_s(j)` = full eq table for each stage's `r_cycle_s` (length T)
   - `eq_entry(j) = eq(0, j)` (indicator for cycle 0)
   - Pre-compute as a single buffer of length T

4. **Create cycle-phase KernelEvaluator:**
   - `ProductSum(d, 1)` kernel, ToomCook mode
   - d RA poly buffers as inputs
   - `combined_eq` as ToomCook eq weights (via `with_toom_cook_eq`)
   - Cycle claimed sum = running claim from address phase (continuous sumcheck)

### Cycle Phase — Segment 1

**Rounds:** log_T, **Degree:** d+1, **Binding:** LowToHigh, **Mode:** ToomCook

Standard ProductSum sumcheck:

```
Σ_j combined_eq(j) · Π_{i=0}^{d-1} ra_i(j)
```

This is exactly `KernelEvaluator::with_toom_cook_eq(ra_bufs, kernel, d, eq_w, claimed_sum, backend)`
where `eq_w` is the ToomCook challenge point derived from the combined_eq buffer.

**Wait — ToomCook expects eq as `eq(w, ·)` factored via w.** But our combined_eq is NOT
a standard eq polynomial — it's a weighted sum of 5-6 different eq polynomials. We can't
factor it into `eq_single(w_0, X) · partial_eq(w[1..], ·)`.

**Resolution:** Use StandardGrid mode for the cycle phase instead of ToomCook:
- combined_eq is a regular input buffer (not weight)
- Kernel: `Custom` that computes `eq_combined(j) · Π_i ra_i(j)`
- But this would be degree d+1 (eq × d-way product), which is correct
- `P(1) = claim - P(0)` recovery works fine

However, the `eval_linear_prod` Toom-Cook optimization loses ~2x perf vs old code.

**Alternative — GruenSplitEq approach:** The old code uses N_STAGES separate GruenSplitEq
polynomials, each handling one stage's eq contribution. The final round poly is an
RLC of per-stage round polys. This avoids materializing the full combined_eq.

**Decision: Use KernelEvaluator StandardGrid with combined_eq as input.**

For the cycle phase, combine all stage eqs into one buffer and use a Custom kernel.
This materializes O(T) memory for the combined_eq but avoids the complexity of
per-stage Gruen tracking. The Toom-Cook optimization for the RA product is handled
by the kernel itself (ProductSum shape within the Custom expression).

Actually, re-examining: we can still use `ProductSum(d, 1)` with `pairwise_reduce`
if we make combined_eq the weight buffer. The weight buffer doesn't need to be a
standard eq polynomial — `pairwise_reduce` just multiplies by `weights[i]` at each
position. ToomCook reconstruction requires the eq factoring, but if we use StandardGrid
mode with explicit weights, the weight buffer is just multiplied directly.

**Final approach for cycle phase:**
- Use `KernelEvaluator::new(inputs, weights, kernel, num_evals, backend)` (weighted StandardGrid)
- `inputs` = [ra_0, ..., ra_{d-1}] (d buffers)
- `weights` = combined_eq buffer (length T/2 after first bind)
- Kernel = `ProductSum(d, 1)` BUT evaluated on the standard grid {0, 2, ..., d+1}
  with the weight multiply handled by `pairwise_reduce`.

Hmm, but `ProductSum(d, 1)` is designed for ToomCook grid `{1, ..., D-1, ∞}`.
On the standard grid, we need a `Custom` kernel that computes the d-way product.

**Simplest correct approach:**

```
Custom kernel: Π_{i=0}^{d-1} opening(i)
```

This is a d-way product. Degree = d. With the eq/weight factor, total degree = d+1.
The weight buffer is combined_eq. StandardGrid mode with `P(1) = claim - P(0)`.
num_evals = d (evaluations at {0, 2, 3, ..., d}).

This works! The only perf concern is that the d-way product kernel evaluates at d
standard grid points, whereas ToomCook would use d Toom-Cook points. For d ≤ 3
(typical bytecode), this is fine.

**Alternatively, use the existing `product_sum` catalog descriptor** which produces a
`ProductSum(d, 1)` shape. But `ProductSum` outputs on the ToomCook grid, and
StandardGrid mode expects the standard grid. These are incompatible.

**Decision: Custom d-way product kernel on standard grid, combined_eq as weight buffer.**

```rust
// Build the d-way product kernel
let eb = ExprBuilder::new();
let mut product = eb.opening(0);
for i in 1..d {
    product = product * eb.opening(i);
}
let expr = eb.build(product);
let desc = KernelDescriptor {
    shape: KernelShape::Custom { expr, num_inputs: d },
    degree: d, // d-way product
    tensor_split: None,
};
```

With weights = combined_eq, the StandardGrid mode produces degree d+1 round polys
(weight adds one degree), which matches the claim degree.

### Claims Produced

After all log_K + log_T rounds:
- d `BytecodeRa(i)` openings at `(r_address_chunks[i], r_cycle)`
- The sumcheck challenges split as: first log_K = r_address, last log_T = r_cycle
- `r_cycle` is reversed (LowToHigh binding → MSB-first for evaluation)

### Witness Generation in `build()`

1. **Derive challenges from prior_claims + transcript:**
   - γ powers (8 total)
   - Per-stage gammas (β_1..β_5 with varying counts)
   - rv_claims (5 scalars from prior stage openings)
   - raf_claim, raf_shift_claim (from Spartan openings)
   - r_cycles (5 challenge vectors from prior stages)
   - entry_gamma, entry_bytecode_index

2. **Compute F_s(k) tables** (expensive, O(T·N_STAGES) work):
   - Split-eq optimization: E_hi[s], E_lo[s] for each stage
   - Double-sum: for each c_hi, scan c_lo accumulating by PC
   - Parallel over c_hi chunks

3. **Compute Val'_s(k) tables** (O(K·N_STAGES) work):
   - Single fused pass over bytecode instructions
   - Each instruction → 5 Val values from flags/immediates/register-eq
   - Fold Int(k) corrections into stages 1 and 3

4. **Compute f_trace, f_expected** (O(K) work):
   - f_trace[k] = one-hot at PC(cycle_0)
   - f_expected[k] = one-hot at entry_bytecode_index

5. **Compute input_claim:**
   - `Σ_s γ^s · rv_claim_s + γ^5 · raf_claim + γ^6 · raf_shift_claim + γ_entry`

6. **Build address kernel + upload buffers**
7. **Build transition closure** (captures trace, r_cycles, one_hot_params, etc.)
8. **Return SegmentedEvaluator(segment_0, log_K).then(log_T, transition)**

### `extract_claims()`

Takes ownership of the d RA polynomial evaluation tables from the transition
closure (stored as `Option<Vec<Vec<F>>>` on the stage struct, set during transition).

For each RA poly i:
- `eval_point = (r_address_chunks[i], r_cycle_reversed)`
- `eval = poly.evaluate(&eval_point)`
- Return `ProverClaim { evaluations, point: eval_point, eval }`

### `claim_definitions()`

Returns d `ClaimDefinition` entries for `BytecodeRa(0)` through `BytecodeRa(d-1)`.

---

## InstructionReadRaf

### Mathematical Identity

```
input_claim = Σ_{j,k} eq(r_reduction, j) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k))
```

Where:
- `eq(r_reduction, j)` — single eq polynomial (vs 5 per-stage eqs in bytecode)
- `ra(k, j) = Π_{i=0}^{n_virtual-1} ra_i(k_chunk_i, j)` — product over virtual RA chunks
- `Val_j(k)` = lookup table value for cycle j's table at address k
- `RafVal_j(k)` = RAF term: operand prefixes weighted by raf_flag
- `K = 2^{LOG_K}` where `LOG_K = 2 * XLEN = 128` (HUGE address space)
- `n_virtual = LOG_K / ra_virtual_log_k_chunk` (typically 16-32)

**Key difference from Bytecode:** The address space is 128 bits (vs ~20 for bytecode).
This makes materializing full address-domain tables impossible. Instead, the address
phase uses **prefix-suffix decomposition** to avoid exponential blowup.

### Address Phase Structure

The 128 address bits are processed in `phases` chunks of `log_m = LOG_K / phases` bits
each (typically 2-8 bits per phase). Within each phase:

1. **Suffix polynomials** (per lookup table): pre-computed accumulators over the
   cycle domain, indexed by the current phase's chunk bits
2. **Prefix checkpoints**: cached prefix MLE evaluations, updated every 2 rounds
3. **Expanding table**: accumulates the product of all prior phases' bindings

Phase transitions happen every `log_m` rounds. At each transition:
- Condense `u_evals[j] *= v_prev[k_chunk_j]` (expand prior phase into per-cycle scalars)
- Reinitialize suffix polys for the new phase
- Update prefix registry with final checkpoint values
- Reset expanding table

**This is the hard part.** The address phase has:
- Different suffix tables per lookup table (30+ tables)
- Three separate prefix-suffix decompositions (left, right, identity) for RAF
- Prefix checkpoints that update every 2 rounds
- Phase transitions every log_m rounds
- All of this is degree 2 per round

### Mapping to SegmentedEvaluator

**Option A — One segment per phase (phases segments + 1 cycle segment):**

Each address phase = one SegmentedEvaluator segment with a round hook for
prefix checkpoint updates. Phase transitions = segment transitions.

```
Segment 0 (log_m rounds): Custom kernel, address phase 0
  → transition: condense u_evals, reinit suffix/prefix
Segment 1 (log_m rounds): Custom kernel, address phase 1
  → transition: condense, reinit
...
Segment {phases-1} (log_m rounds): Custom kernel, address phase (phases-1)
  → transition: materialize RA polys, combined_val, combined_eq
Segment {phases} (log_T rounds): Custom d-product kernel, cycle phase
```

Round hook: every 2 rounds within a segment, recompile kernel with updated prefix
checkpoints via `evaluator.update_kernel(new_kernel)`.

**Option B — Single address segment with internal phase management:**

One address segment that handles all phases internally (not via SegmentedEvaluator).
This requires the kernel to change at phase boundaries, which SegmentedEvaluator
handles via transitions. So Option A is cleaner.

**Decision: Option A — one segment per phase.**

### Address Phase Kernel (per phase)

For each phase, the kernel computes:

```
kernel(i) = Σ_{table t} checkpoint_t · suffix_t(i)
           + γ · left_prefix_checkpoint · left_suffix(i)
           + γ² · right_prefix_checkpoint · right_suffix(i)
           + identity_coeff · identity_suffix(i)
```

- `checkpoint_t` = baked scalar (prefix evaluation for table t at current prefix point)
- `suffix_t(i)` = input buffer (suffix polynomial for table t at current phase)
- `left_prefix_checkpoint`, etc. = baked RAF scalars
- `left_suffix(i)`, etc. = input buffers (RAF suffix polys)

**Degree:** 2 (checkpoint × suffix, both linear in the bound variable)
**Inputs:** `N_tables + 3` suffix buffers (30+ tables + left/right/identity RAF)
**Weights:** Unit (no eq in address phase)
**Binding:** LowToHigh
**Round hook:** Every 2 rounds, recompile kernel with updated prefix checkpoints.

### Cycle Phase

After all address phases complete:

1. Materialize RA polynomials: for each virtual RA i, combine expanding table values
   across its assigned phases.
2. Compute `combined_val[j] = Val_j(bound_address) + γ · RafVal_j(bound_address)`
3. The cycle identity is:
   ```
   Σ_j eq(r_reduction, j) · combined_val(j) · Π_i ra_i(j)
   ```

**With single eq polynomial**, we CAN use ToomCook mode:
- `eq_w = r_reduction`
- Inputs = [combined_val, ra_0, ..., ra_{n_virtual-1}] (n_virtual+1 buffers)
- Kernel = `ProductSum(n_virtual+1, 1)` — product of all inputs
- ToomCook mode with eq factored out
- Degree = n_virtual + 2 (eq × combined_val × Π ra_i)

This is a clean mapping to `KernelEvaluator::with_toom_cook_eq`.

### Claims Produced

- `n_virtual` InstructionRa(i) openings at `(r_address_chunk_i, r_cycle)`
- Per-table flag claims (lookup table selections) — computed from trace at r_cycle
- RAF flag claim

---

## Infrastructure Needed

### Already exists in new crates:
- `KernelEvaluator` with StandardGrid + ToomCook modes ✅
- `SegmentedEvaluator` with transitions + round hooks ✅
- `RaPolynomial` (lazy materialization) ✅
- `IdentityPolynomial` ✅
- `OneHotParams` (in jolt-verifier) ✅
- `BytecodePreprocessing` (in jolt-zkvm/witness) ✅
- `PrefixSuffixEvaluator` (in jolt-sumcheck) ✅
- `catalog::product_sum`, `catalog::eq_product`, etc. ✅
- `EqPolynomial` ✅

### Needs to be ported/created:

| Component | Source | Target | Notes |
|-----------|--------|--------|-------|
| `ExpandingTable` | jolt-core/utils/ | jolt-zkvm/evaluators/ or jolt-poly | Only for InstructionReadRaf |
| `PrefixSuffixDecomposition` | jolt-core/poly/ | jolt-poly or jolt-zkvm | Multi-phase prefix/suffix, used by InstructionReadRaf |
| `PrefixRegistry` | jolt-core/poly/ | jolt-poly or jolt-zkvm | Shared cache for prefix polys |
| `Prefixes` / `PrefixCheckpoint` | jolt-core/poly/ | jolt-poly or jolt-zkvm | Prefix MLE eval + update |
| `LookupBits` | jolt-core/utils/ | jolt-zkvm | Bit-packed lookup index |
| Val poly computation | bytecode/read_raf | jolt-zkvm stage | Bytecode iteration with per-stage formulas |
| F_s computation | bytecode/read_raf | jolt-zkvm stage | Split-eq trace iteration |
| Flag claim computation | instruction_lookups/read_raf | jolt-zkvm stage | Split-eq per-table accumulation |

### Does NOT need porting:

| Component | Reason |
|-----------|--------|
| `GruenSplitEqPolynomial` | Replaced by combined_eq + ToomCook/StandardGrid |
| `SumcheckInstanceProver` trait | Replaced by `SumcheckCompute` via KernelEvaluator |
| `SumcheckInstanceVerifier` trait | Replaced by config-driven verifier |
| `SumcheckInstanceParams` trait | Replaced by `ClaimDefinition` in jolt-ir |
| BlindFold constraint methods | Deferred (standard mode first) |

---

## Implementation Order

### Phase 1: BytecodeReadRaf (simpler, warmup)

```
Step 1: Bytecode stage scaffolding
        New file: crates/jolt-zkvm/src/stages/s_bytecode_read_raf.rs
        Struct, ProverStage impl skeleton, claim_definitions()
        ~50 lines

Step 2: Val polynomial computation
        Port compute_val_polys() — fused bytecode iteration
        Needs: instruction flag types, register eq tables
        ~150 lines

Step 3: F polynomial computation
        Port split-eq F_s computation from trace
        Needs: BytecodePreprocessing.get_pc(), EqPolynomial
        ~100 lines

Step 4: Address phase kernel + SegmentedEvaluator wiring
        Build Custom kernel, upload buffers, create segment 0
        ~60 lines

Step 5: Transition closure + cycle phase
        Materialize RA polys, build combined_eq, create segment 1
        ~120 lines

Step 6: extract_claims() + claim chaining
        Return d BytecodeRa claims
        ~40 lines

Step 7: Unit tests
        Synthetic witness: random F/Val/RA tables → prove → verify
        Full round-trip with brute-force claim check
        ~150 lines
```

**Estimated: ~670 lines** (vs 1762 in old code — no verifier, no BlindFold, no params gen)

### Phase 2: InstructionReadRaf (complex)

```
Step 1: Port ExpandingTable
        New file: crates/jolt-zkvm/src/evaluators/expanding_table.rs
        or crates/jolt-poly/src/expanding_table.rs
        ~80 lines

Step 2: Port PrefixSuffixDecomposition infrastructure
        Prefix/suffix tables, PrefixRegistry, PrefixCheckpoint
        These may live in jolt-poly or jolt-zkvm
        ~300 lines (ported + adapted)

Step 3: Port LookupBits
        Bit-packed lookup index extraction
        ~50 lines

Step 4: Instruction stage scaffolding
        New file: crates/jolt-zkvm/src/stages/s_instruction_read_raf.rs
        ProverStage impl with multi-segment SegmentedEvaluator
        ~100 lines

Step 5: Address phase per-segment kernel + transitions
        One segment per phase with round hooks for prefix updates
        Suffix poly initialization per phase
        ~250 lines

Step 6: Cycle phase (ToomCook with combined_val)
        Materialize RA polys from expanding tables, combined_val
        Create ProductSum KernelEvaluator
        ~150 lines

Step 7: Flag claim computation
        Split-eq per-table accumulation for opening claims
        ~100 lines

Step 8: extract_claims() + claim_definitions()
        ~50 lines

Step 9: Unit tests
        Synthetic multi-phase address binding + cycle product
        ~200 lines
```

**Estimated: ~1280 lines** (vs 1939 in old code)

### Phase 3: Integration

```
Step 1: Wire both stages into pipeline (stages/mod.rs)
Step 2: Update stage_parity_plan.md
Step 3: Run existing stage tests
```

---

## Key Design Decisions

### 1. Combined eq vs per-stage Gruen (BytecodeReadRaf)

**Old:** 5 separate GruenSplitEqPolynomial instances, per-stage round polys, RLC aggregation.

**New:** Pre-compute `combined_eq[j] = Σ_s γ^s · bound_val_s · eq_s(j) + entry_term`.
Single combined buffer as kernel weights.

**Tradeoff:** Materializes O(T) combined_eq (one buffer) vs O(5·√T) Gruen split tables.
For T = 2^20, this is 8MB vs ~5×1KB. Acceptable — and much simpler.

**Perf note:** The old Gruen approach avoids d+1 field muls per position for the
weight × product. Our approach includes the weight multiply in `pairwise_reduce`.
Difference is ~10% in cycle-phase cost for typical d=2-3. Acceptable for clean design.

### 2. ToomCook vs StandardGrid for cycle phase

**BytecodeReadRaf cycle:** Combined eq is NOT a standard eq polynomial (sum of 5+1 eqs),
so ToomCook factoring doesn't apply. Use StandardGrid with combined_eq as weight buffer
and a Custom d-product kernel.

**InstructionReadRaf cycle:** Single eq(r_reduction, j), so ToomCook DOES apply.
Use `KernelEvaluator::with_toom_cook_eq` with `ProductSum(n_virtual+1, 1)`.

### 3. Address kernel: one big Custom vs per-stage split

For BytecodeReadRaf, one big Custom kernel with 12 inputs is cleanest. The γ-RLC
aggregation happens inside the kernel, not outside.

For InstructionReadRaf, the address kernel has ~30+ input buffers (one suffix per
lookup table + 3 RAF suffixes). This is fine for CPU (inlined closure), and for GPU
a single shader reading ~35 buffers is practical.

### 4. Expanding tables and SegmentedEvaluator

InstructionReadRaf's multi-phase address structure maps to SegmentedEvaluator segments:
- Each segment = one phase of `log_m` rounds
- Transition = condensation + suffix reinit + prefix cache update
- Round hook = prefix checkpoint update every 2 rounds

This keeps all computation within the kernel framework — no custom SumcheckCompute.

### 5. Memory management

Both stages need to store RA polynomial evaluation tables for `extract_claims()`.
These are materialized at the address→cycle transition and stored as
`Option<Vec<Vec<F>>>` on the stage struct. `extract_claims()` takes ownership.

For InstructionReadRaf, the expanding tables (`v[phase]`) are consumed during
RA materialization and can be dropped immediately after.

---

## Testing Strategy

### BytecodeReadRaf

1. **Synthetic address-only test:** Random F_s, Val_s tables, verify address phase
   round polys sum correctly.

2. **Synthetic two-phase test:** Random F_s, Val_s, RA polys. Full prove/verify
   through SegmentedEvaluator (address + cycle). Check final eval matches brute-force.

3. **Entry constraint test:** Verify entry term (f_trace × f_expected) is correctly
   included in address phase and contributes to combined_eq in cycle phase.

4. **Consistency with brute-force:** Compute claimed_sum by iterating over all
   (k, j) pairs, verify it matches the stage's claimed_sum.

### InstructionReadRaf

1. **Single-phase address test:** 1 phase of log_m rounds with synthetic suffix/prefix
   tables. Verify round polys.

2. **Multi-phase address test:** phases segments with transitions. Verify expanding
   table condensation and prefix checkpoint updates.

3. **Round hook test:** Verify prefix checkpoints update every 2 rounds.

4. **Full two-phase test:** Address phases + cycle phase with ToomCook product.
   Full prove/verify round-trip.

5. **Flag claim test:** Verify per-table flag claims match brute-force evaluation.

---

## Dependencies Between Steps

```
BytecodeReadRaf:
  Step 2 (Val polys)  ──┐
  Step 3 (F polys)     ──┤── Step 4 (address kernel) ── Step 5 (transition) ── Step 6-7
                         │
  (existing infra)  ─────┘

InstructionReadRaf:
  Step 1 (ExpandingTable) ──┐
  Step 2 (PrefixSuffix)   ──┤── Step 4 (scaffolding) ── Step 5 (address) ── Step 6 (cycle) ── Step 7-9
  Step 3 (LookupBits)     ──┘
```

BytecodeReadRaf is fully independent of InstructionReadRaf. Do bytecode first as warmup.

---

## Open Questions

1. **Where to put ExpandingTable?** Options: jolt-poly (general utility) or
   jolt-zkvm/evaluators (prover-specific). Leaning toward jolt-zkvm since it's
   only used by one stage.

2. **PrefixSuffixDecomposition location?** This is a complex data structure used
   only by InstructionReadRaf. Could live in jolt-poly (it's a polynomial utility)
   or in the stage file itself (single user). Leaning toward a dedicated file in
   jolt-zkvm/evaluators.

3. **Val poly computation needs instruction types.** BytecodeReadRaf's Val computation
   requires `Instruction`, `CircuitFlags`, `InstructionFlags`, etc. These come from
   jolt-instructions. Need to verify the dependency path allows jolt-zkvm to use them.

4. **F_s computation needs trace access.** The stage's `build()` method needs the
   trace (`Vec<Cycle>`) and `BytecodePreprocessing`. These are available from the
   `WitnessOutput` / prover context. Need to verify how the stage receives them
   (likely via constructor, similar to how existing stages receive WitnessStore refs).
