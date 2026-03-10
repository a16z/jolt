# Prover Pipeline Divergence Report: jolt-core vs jolt-zkvm

**Date:** 2026-03-10
**Branch:** `refactor/crates` vs `main`

This report compares the reference jolt-core prover pipeline against the new modular jolt-zkvm implementation. It covers protocol-level correctness, Fiat-Shamir synchronization, stage composition, batching, opening proofs, and kernel-level performance parity.

---

## Executive Summary

The new crates achieve **strong algorithmic parity** on sumcheck kernels and batching mechanics. However, there are **three critical protocol-level divergences** and **two performance regressions** that must be addressed before E2E muldiv can pass against the old verifier or reach performance parity:

| # | Severity | Area | Issue |
|---|----------|------|-------|
| **D1** | **CRITICAL** | Fiat-Shamir | Input claims not absorbed before batching coefficient derivation |
| **D2** | **CRITICAL** | Fiat-Shamir | Opening claims not flushed to transcript between stages |
| **D3** | **CRITICAL** | Fiat-Shamir | RLC claims not absorbed before gamma sampling in stage 8 |
| **D4** | MAJOR | Stage composition | Stage 2 has 1 instance (new) vs 5 instances (old) — different batching |
| **D5** | MAJOR | Stage composition | Stages 5–7 reorganized — different instance grouping |
| **P1** | PERF | Opening phase | Eager RLC materialization — no streaming from trace |
| **P2** | PERF | Opening phase | Lost Dory streaming evaluation (DorySourceAdapter) |

---

## 1. CRITICAL: Fiat-Shamir Transcript Divergences (D1, D2, D3)

### D1 — Input claims not absorbed before batching coefficients

**Old code** (`jolt-core/src/subprotocols/sumcheck.rs:44-48`):
```rust
sumcheck_instances.iter().for_each(|sumcheck| {
    let input_claim = sumcheck.input_claim(opening_accumulator);
    transcript.append_scalar(b"sumcheck_claim", &input_claim);
});
let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
```

**New code** (`crates/jolt-sumcheck/src/batched.rs:65`):
```rust
let alpha = challenge_fn(transcript.challenge());
// No claim absorption before this line
```

**Impact:** The old prover absorbs each instance's input claim into the transcript BEFORE deriving the batching coefficient α. The new code skips this entirely. Since the transcript state differs, α will differ, all subsequent challenges will differ, and proofs are **cryptographically incompatible**.

**Fix:** Before line 65 in `batched.rs`, the caller (or the batched prover itself) must absorb the `claimed_sum` of each claim into the transcript. The cleanest approach: add a loop in `prove_with_handler` that calls `transcript.append_scalar(b"sumcheck_claim", &claim.claimed_sum)` for each claim before deriving α.

**Note:** In ZK mode, the old code intentionally skips claim absorption (line 219). The new code should match: absorb in standard mode, skip in ZK mode.

### D2 — Opening claims not flushed to transcript between stages

**Old code** (`jolt-core/src/subprotocols/sumcheck.rs:176`):
```rust
opening_accumulator.flush_to_transcript(transcript);
```
This is called at the end of every batched sumcheck. `flush_to_transcript` drains `pending_claims` and calls `transcript.append_scalar(b"opening_claim", &claim)` for each cached opening.

**New code** (`crates/jolt-zkvm/src/pipeline.rs:106-158`):
The stage loop has no transcript flush between stages. `extract_claims()` produces `ProverClaim` objects that are accumulated in `all_opening_claims`, but their evaluations are never absorbed into the transcript before the next stage's `build()` call.

**Impact:** The old pipeline absorbs all opening evaluations into the Fiat-Shamir transcript after each stage. The next stage's input claims and batching coefficients are derived from a transcript that includes these bindings. Without this, the new pipeline has weaker Fiat-Shamir binding — and proofs are incompatible.

**Fix:** After `extract_claims()` in `prove_stages()`, append each new claim's `eval` to the transcript:
```rust
for claim in &new_claims {
    transcript.append_scalar(b"opening_claim", &claim.eval);
}
```

### D3 — RLC claims not absorbed before gamma sampling

**Old code** (`jolt-core/src/zkvm/prover.rs:1944-1946`):
```rust
#[cfg(not(feature = "zk"))]
self.transcript.append_scalars(b"rlc_claims", &claims);
let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());
```

**New code** (`crates/jolt-openings/src/reduction.rs:117`):
```rust
let rho = challenge_fn(transcript.challenge());
// No claim absorption before rho derivation
```

The `OpeningReduction` trait's docstring even says "the transcript must already contain all claim data" (line 50-52), but nothing enforces this. The caller (s8_opening.rs) does not absorb claims before calling `reduce_prover`.

**Impact:** Same as D1 — rho derived from different transcript state.

**Fix:** In `OpeningStage::prove()` (s8_opening.rs), before calling `reduce_prover`, absorb all claim evaluations:
```rust
for claim in &claims {
    transcript.append_scalar(b"rlc_claims", &claim.eval);
}
```

---

## 2. MAJOR: Stage Composition Divergences (D4, D5)

### D4 — Stage 2 has fundamentally different instance composition

**Old Stage 2** (5 instances batched together):
1. `RamReadWriteChecking` — degree 4, log_T + log_K rounds
2. `ProductVirtualRemainder` — degree 2, log_T rounds (Spartan product continuation)
3. `InstructionLookupsClaimReduction` — degree 2, log_T rounds
4. `RamRafEvaluation` — degree 2, log_K rounds
5. `OutputCheck` — degree 2, log_K rounds

**New Stage 2** (1 instance):
1. `RaVirtualStage` — single RA virtual sumcheck, degree d+1

**Impact:** This is an intentional reorganization, not a bug. The new pipeline distributes old Stage 2's instances across multiple stages (S2 for RA virtual, S3 for claim reductions, S4 for RW checking, S5 for output/RAF). This changes the batching — different α coefficients, different combined round polynomials, different proofs.

**Consequence:** The new pipeline produces **structurally different proofs** from the old. This is acceptable if the new verifier (jolt-verifier) is built to match the new stage layout. However, it means old proofs cannot be verified by the new verifier and vice versa.

### D5 — Stages 5–7 reorganized

| Old Stage | Old Instances | New Stage | New Instances |
|-----------|--------------|-----------|---------------|
| S5 | InstrReadRaf + RamRaReduction + RegsValEval (3) | S5 | RamOutputCheck + RamRafEval (2) |
| S6 | BytecodeRaf + Booleanity + HammingBool + RamRaVirtual + InstrRaVirtual + IncReduction + Advice×2 (6-8) | S6 | HammingBooleanity only (1) |
| S7 | HammingWeightReduction + Advice×2 (1-3) | S7 | HammingReduction (1) |

The old Stage 6 was a "mega-batch" of 6-8 heterogeneous instances. The new pipeline breaks these into smaller, focused stages. The total sumcheck work is the same, but the batching structure differs (fewer instances per batch = different α weighting).

**Consequence:** Same as D4 — structurally different proofs. The new verifier must be built to match.

**Missing from new pipeline:**
- BytecodeRaf checking — not visible in any new stage
- IncClaimReduction — not visible as a separate instance
- AdviceClaimReduction (phases 1 & 2) — not implemented yet (noted as deferred)
- InstructionRaVirtual — subsumed by S2's RaVirtualStage?
- RamRaVirtual — subsumed by S2?
- ProductVirtualRemainder (Spartan product continuation) — where did this go?
- ShiftSumcheck (old S3) — not visible in new S3
- InstructionInput (old S3) — not visible in new S3

**Action required:** Verify that every sumcheck instance from the old pipeline has a corresponding instance in the new pipeline. The total set of checks must be equivalent for soundness.

---

## 3. Batching Mechanics: Matched ✓

Setting aside the Fiat-Shamir absorption issues, the core batching algorithm is correct:

| Aspect | Old | New | Match |
|--------|-----|-----|-------|
| Front-loaded batching | ✓ (Posen) | ✓ (Posen) | ✓ |
| Claim scaling by 2^(N−n) | `mul_pow_2(max-n)` | `mul_pow_2(offset)` | ✓ |
| Dummy rounds (constant claim/2) | ✓ | ✓ | ✓ |
| α derivation | `challenge_vector(n)` | `challenge_fn(challenge())` | ✓ (modulo D1) |
| Round poly combination | Coefficient-space addition | Evaluation-space + interpolation | ✓ (math equiv) |
| Round absorption + challenge | Compress → append → squeeze | `absorb_round_poly` → `challenge` | ✓ |
| Bind active witnesses | `ingest_challenge(r, round)` | `bind(challenge)` | ✓ |

The new evaluation-based combination (evaluate each instance poly at 0..max_degree, combine, interpolate) is mathematically equivalent to the old coefficient-space addition, and handles mixed degrees more naturally.

---

## 4. Opening Phase Divergences (P1, P2)

### P1 — Eager RLC materialization (performance regression)

**Old code:** Builds a streaming `RLCPolynomial` that combines all committed polynomials lazily. During `Dory::prove()`, the RLC polynomial is evaluated on-the-fly via `vector_matrix_product()` without ever materializing the full table. Memory: O(√T).

**New code:** `rlc_combine()` in `reduction.rs:186-209` materializes the full combined evaluation table into a `Vec<F>`. For k polynomials of size 2^n, this allocates O(2^n) field elements. This is then passed to `PCS::open()` as a concrete `Polynomial`.

**Impact:** For a trace of length T with ~20 committed polynomials, the old code never holds more than one row of the RLC polynomial (~√T elements). The new code holds the entire RLC table (~T elements per evaluation point group). For T = 2^20, this is ~32MB per group vs ~32KB.

**Fix:** Implement a streaming RLC path:
- Option A: Add a `StreamingRlcReduction` that wraps `(poly_id, coefficient)` pairs and implements `EvaluationSource` for Dory
- Option B: Override `PCS::open()` with a specialized method that accepts `&[(coefficient, &[F])]` instead of a materialized polynomial
- Option C: Accept the regression for now (correctness first, optimize later)

### P2 — Lost Dory streaming evaluation

**Old code:** `DorySourceAdapter` implements `EvaluationSource<Fr>` and bridges the old polynomial types to Dory's internal evaluation. This enables streaming `vector_matrix_product` during opening without materializing the full polynomial.

**New code:** `DoryScheme::open()` receives a `Polynomial<Fr>` (dense evaluation table). While Dory internally may use streaming for its own operations, the input is already fully materialized.

**Impact:** Combined with P1, Dory no longer benefits from streaming at the polynomial level. The evaluation table is materialized in `rlc_combine()`, then Dory processes it. The old pipeline avoided this materialization entirely.

---

## 5. Kernel-Level Performance: Matched ✓

The inner-loop hot paths are algorithmically equivalent:

| Aspect | Old | New | Match |
|--------|-----|-----|-------|
| Unreduced accumulation | `UnreducedProductAccum` | `FieldAccumulator::fmadd` | ✓ |
| Split-eq factorization | `GruenSplitEqPolynomial` | `SplitEqEvaluator` | ✓ |
| RA poly lazy materialization | `RaPolynomial<u8,F>` state machine | Same design | ✓ |
| Toom-Cook specializations | D ∈ {16, 32} | D ∈ {2,3,4,5,6,7,8,16,32} | ✓+ (more) |
| Parallel threshold | Implicit rayon | Explicit `PAR_THRESHOLD=1024` | ✓ (better) |
| Fold pattern | `par_fold_out_in` | `par_fold_out_in` | ✓ (identical) |
| Product evaluation kernel | `eval_prod_D_assign` | `eval_linear_prod_assign` | ✓ |

The new code has **more** Toom-Cook specializations (D=2 through 8 in addition to 16 and 32), which covers all practical chunk sizes without falling back to the naive path. This is a performance improvement.

---

## 6. Spartan Integration: Partially Matched

**Old:** Uses `UniformSpartanKey` with lazy bilinear evaluation, univariate skip for the first round, and a specialized `OuterRemainingStreamingSumcheck` for remaining rounds. The outer and product sumchecks are in Stage 1 and Stage 2 respectively.

**New:** `UniformSpartanStage` in `s1_spartan.rs` wraps `UniformSpartanProver::prove_dense_with_challenges()`. The uniform key and constraint structure (24 constraints × 41 vars) are preserved.

**Gap:** Need to verify:
- Does the new Spartan prover implement univariate skip?
- Is the product virtual sumcheck (old Stage 2, instance 2) accounted for?
- Does the new `UniformSpartanKey` use the same lazy evaluation as the old?

---

## 7. Missing Protocol Components

These items from the old pipeline need verification in the new crates:

| Component | Old Location | New Location | Status |
|-----------|-------------|--------------|--------|
| Univariate skip (S1-S2) | `univariate_skip.rs` | `jolt-spartan/uniform_prover.rs` | Needs verification |
| BytecodeRaf checking | Stage 6 | ? | **NOT FOUND** |
| ShiftSumcheck | Stage 3 | ? | **NOT FOUND** |
| InstructionInput sumcheck | Stage 3 | ? | **NOT FOUND** |
| IncClaimReduction | Stage 6 | ? | **NOT FOUND** |
| AdviceClaimReduction (2 phases) | Stages 6-7 | Deferred per spec | Known deferred |
| ProductVirtual remainder | Stage 2 | ? | **NOT FOUND** |
| RamRaVirtual | Stage 6 | S2? | Needs verification |
| OutputCheck | Stage 2 | S5 | Likely moved |
| Lagrange padding for Inc polys | Stage 8 RLC | ? | **NOT VERIFIED** |
| Streaming witness commitment | Tier-1/tier-2 | WitnessSink | Architecture exists, needs wiring |

---

## 8. Recommendations (Priority Order)

### Must-fix before E2E

1. **Fix D1:** Add claim absorption in `BatchedSumcheckProver::prove_with_handler` before α derivation. Gate with `#[cfg(not(feature = "zk"))]` to match old behavior.

2. **Fix D2:** Add `flush_to_transcript` equivalent in `prove_stages()` after `extract_claims()`.

3. **Fix D3:** Add claim absorption in `OpeningStage::prove()` before `reduce_prover()`.

4. **Audit missing instances:** Systematically verify every sumcheck instance from the old pipeline has a counterpart. The stage layout can differ, but the total set of polynomial identity checks must be equivalent.

### Should-fix for performance parity

5. **P1/P2:** Implement streaming RLC for the opening phase, or accept the regression with a tracking issue.

6. **Verify Spartan uni-skip:** Confirm the new `UniformSpartanProver` implements the first-round univariate skip optimization.

### Nice-to-have

7. **Benchmark stage 8:** Compare peak memory and wall time between old and new at the opening phase.

8. **Verify Lagrange padding:** Confirm dense polynomial evaluation tables are correctly padded before RLC combination.

---

## Appendix: Old vs New Fiat-Shamir Transcript Order

### Old Pipeline (per stage)
```
1. append_scalar("sumcheck_claim", input_claim_i)  ×N instances
2. challenge_vector(N) → batching coefficients
3. For each round:
   a. compress round poly → append coefficients
   b. challenge_scalar → round challenge r_k
4. cache_openings → pending_claims
5. flush_to_transcript: append_scalar("opening_claim", v_i)  ×K openings
```

### New Pipeline (per stage)
```
1. (MISSING — no claim absorption)
2. challenge() → α  (single batching coefficient)
3. For each round:
   a. absorb_round_poly → append coefficients
   b. challenge() → round challenge r_k
4. extract_claims → ProverClaim objects
5. (MISSING — no opening claim flush)
```

### Old Stage 8
```
1. append_scalars("rlc_claims", &claims)
2. challenge_scalar_powers(N) → gamma powers
3. Build streaming RLC polynomial
4. Dory::prove() with streaming evaluation
5. bind_opening_inputs: append point + eval
```

### New Stage 8
```
1. (MISSING — no claim absorption)
2. challenge() → rho (per group)
3. rlc_combine() → materialized Vec<F>
4. PCS::open() with materialized polynomial
```
