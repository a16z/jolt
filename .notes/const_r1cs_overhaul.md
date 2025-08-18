## Compile-time R1CS and Streaming Spartan: Refactor Plan

This document specifies the end-to-end plan to remove runtime R1CS construction, eliminate unnecessary materialization, and streamline the Spartan prover/verifier around compile-time uniform constraints and streaming evaluation. It is intended to be executable guidance for edits across `jolt-core/src/zkvm/r1cs/` and a few adjacent modules.

### High-level goals

- Replace dynamic, runtime-built uniform constraints with the compile-time table in `r1cs/const_constraints.rs` as the single source of truth.
- Remove the need to materialize uniform constraints into sparse matrices for the verifier and prover. Compute A/B/C evaluations directly from constants.
- Avoid building all witness polynomials up front. Stream witness access from the execution trace for Stage 1 (outer sumcheck) and compute the Stage 1 witness-evaluation vector at `r_cycle` via a single streaming pass.
- Keep the PC-shift sumcheck as-is or minimally adapted; it only needs a tiny subset of inputs.
- Preserve existing transcript semantics and openings layout so downstream code remains compatible.

---

## Progress summary (current state)

- `UniformSpartanKey` no longer depends on dynamic `UniformR1CS`; it stores counts and a digest and evaluates using `UNIFORM_R1CS`.
- Digest now hashes a canonical serialization of `UNIFORM_R1CS` plus `num_steps` (domain-tagged) instead of serializing sparse matrices.
- Verifier evaluators (`evaluate_uniform_{a,b,c}_at_point`, `evaluate_small_matrix_rlc`, `evaluate_z_mle_with_segment_evals`) are backed by constants.
- `SpartanInterleavedPolynomial::new_with_precompute` evaluates A/B rows directly from `ConstraintConst::{a,b}.evaluate_row` (no dynamic LC path).
- `SumcheckInstanceProof::prove_spartan_small_value` uses constants (`UNIFORM_R1CS`) for Stage 1.
- Legacy builders removed from production; gated under `#[cfg(test)]`.
- Added constant helpers in `constraints.rs`: `ConstLC::{serialize_canonical, dot_eq_ry, for_each_term, accumulate_evaluations}`.

---

## Target end-state architecture

- Uniform constraints exist only as `UNIFORM_R1CS: &'static [ConstraintConst]`.
- Verifier-side evaluators for `A_small/B_small/C_small` and `z(ry)` operate over constants and the vector of per-input evaluations at `r_cycle` (no sparse matrices).
- Stage 1 (outer) prover consumes `UNIFORM_R1CS` and a trace-backed witness accessor to compute A/B per-step values and SVO accumulators. No `flattened_polynomials` for the 42 inputs.
- Stage 1 then computes `claimed_witness_evals = [P_i(r_cycle)]` via a single pass over the trace (no `batch_evaluate` over dense polynomials).
- Stage 3 (PC sumcheck) keeps `UnexpandedPC`, `PC`, `IsNoop` as tiny dense polynomials or uses a minimal streaming wrapper.

---

## Detailed changes by file/module

### r1cs/const_constraints.rs

- Keep as the sole ground truth for the 28 uniform rows.
- Minor micro-optimizations (optional):
  - Provide internal iteration helpers that avoid per-term `match` inside tight loops (e.g., match the enum once and iterate raw arrays).
  - Mark hot helpers `#[inline(always)]` where it helps.
- Ensure the term order is canonical (by input index) for stable digesting and predictable cache behavior.

### r1cs/constraints.rs

- Delete the dynamic `constraints.rs` path. We will not construct constraints at runtime anymore. Keep only `const_constraints.rs` as source of truth.
- Remove exports and all call sites; keep old files under `old_*` names strictly for tests/validation.

### r1cs/builder.rs

- Sever the connection from `builder.rs` to the rest of the system. No more calls into `materialize()` or `get_constraints()` from production paths. Keep the file renamed under `old_builder.rs` for tests only.

### r1cs/const_builder.rs

- Remove; we will not convert constants back to dynamic constraints. Production code uses `UNIFORM_R1CS` directly.

### r1cs/key.rs

- Replace the key shape:
  - Remove `SparseConstraints` and `UniformR1CS`.
  - New `UniformSpartanKey<F>` fields:
    - `num_steps: usize`
    - `num_rows_per_step: usize` (=`UNIFORM_R1CS.len()`)
    - `num_vars: usize` (=`JoltR1CSInputs::num_inputs()`)
    - `vk_digest: F`
- Re-implement evaluators in terms of `UNIFORM_R1CS`:
  - `evaluate_uniform_a_at_point(rx, ry)`, `evaluate_uniform_b_at_point(rx, ry)`, `evaluate_uniform_c_at_point(rx, ry)`
    - Compute `Σ_row eq(rx,row) * Σ_term coeff * eq(ry, col_idx)`; constants contribute at `const_col = num_vars`.
  - `evaluate_small_matrix_rlc(rx, rlc) -> Vec<F>`
    - Produce per-column vector of `A + r*B + r^2*C` by accumulating across rows using `ConstLC::accumulate_evaluations`.
  - `evaluate_z_mle_with_segment_evals(segment_evals, ry, with_const)` remains as-is.
- New digest: linearize `UNIFORM_R1CS` as a stable list of terms for A/B/C plus `num_steps`; hash; map to `F`.

### r1cs/inputs.rs

- Add a zero-allocation witness accessor for Stage 1:
  - `fn value_at<F, PCS>(input: JoltR1CSInputs, t: usize, trace, preprocessing) -> F` (or equivalent using existing generators but without building full vectors).
- Add a helper to compute `claimed_witness_evals` at `r_cycle` in a single streaming pass:
  - Precompute `eq(r_cycle, t)` for all `t`.
  - For each `t`, for each input `i`, accumulate `claims[i] += weight_t * value_at(i, t, ...)`.
  - Only construct dense polynomials for the 8 committed inputs if the PCS requires it (optional follow-up: streaming commits/openings for those 8).

### r1cs/spartan.rs

- Setup:
  - Replace setup to construct the key with `UniformSpartanKey::new_from_const(num_steps)` (no builders/materialize); `SpartanDag::new` calls this directly.
- Stage 1 (outer sumcheck):
  - Replace the implementation of `prove_outer_sumcheck` in place to use `UNIFORM_R1CS` and a witness accessor; do not add a new function.
  - Eliminate construction of all 42 `input_polys` up front. Instead, use a const SVO precompute with `(trace, preprocessing)` and the witness accessor.
  - For `claimed_witness_evals`, replace `batch_evaluate` with the streaming pass described in `inputs.rs`.
- Stage 2 (inner sumcheck):
  - Keep the external API; it calls the key evaluators to get `A_small/B_small/C_small` and `evaluate_z_mle_with_segment_evals` for `z(ry)`.
- Stage 3 (PC sumcheck):
  - Keep as-is. Optionally switch to small streaming wrappers for `UnexpandedPC`, `PC`, `IsNoop` later.

### poly/spartan_interleaved_poly.rs

- Replace `new_with_precompute` in place so it uses `UNIFORM_R1CS` directly (no dynamic LC).
  - Internals:
    - Evaluate row A/B via `ConstraintConst::{a,b}.evaluate_row(flattened_polynomials, step_idx)` (no dynamic LC).
    - Preserve the SVO accumulators and streaming/linear phases unchanged; only the row evaluation source changes.
- Optionally expose a tiny trait for “row evaluator” so both dynamic (legacy) and const paths can share the SVO logic, then delete legacy after migration.

### subprotocols/sumcheck.rs

- Replace the implementation of `prove_spartan_small_value` in place to consume `UNIFORM_R1CS` and a row evaluator closure (or compute A/B/C directly) without dynamic constraints.

### r1cs/mod.rs

- Remove `pub mod constraints;` (dynamic) and any exports of `builder`/`const_builder`. Only export `old_*` for tests gated under cfg(test) if needed.
- Keep `ops` public for instruction-side use; it is not on the verifier’s hot path.

---

## PCS interactions

- Only 8 inputs are committed. If PCS requires dense polynomials for commits/openings, keep materialization just for these 8. Otherwise, add optional streaming-commit APIs and remove dense buffers entirely.
- Virtual inputs (the rest) are never opened via PCS; continue caching their scalar evaluations in the accumulator as today.

---

## API diffs (changed in place)

- `UniformSpartanKey::new_from_const(num_steps: usize) -> Self` (new constructor), and removal of any builder-based constructors.
- `UniformSpartanKey::{evaluate_uniform_a_at_point, evaluate_uniform_b_at_point, evaluate_uniform_c_at_point}` use constants.
- `UniformSpartanKey::evaluate_small_matrix_rlc(&self, rx: &[F], rlc: F) -> Vec<F>` reimplemented over constants.
- `SumcheckInstanceProof::prove_spartan_small_value` updated in place to operate over constants (signature may change to drop dynamic constraint inputs).
- `SpartanInterleavedPolynomial::new_with_precompute` updated in place to operate over constants.
- `inputs.rs`: `value_at(...)` and `compute_claimed_witness_evals_streaming(r_cycle, trace, preprocessing) -> Vec<F>` added.

---

## Migration plan (incremental)

1. Wire const path for Stage 1 only (prover):
   - Replace `new_with_precompute` and `prove_spartan_small_value` implementations to use constants.
   - Swap Stage 1 call sites to the updated in-place implementations.
   - Keep Stage 2 using the key evaluators (already ported to constants).
2. Replace key evaluators with constant-based ones. Remove `UniformR1CS` and `materialize` usage from Stage 2.
3. Replace `claimed_witness_evals` computation with the streaming pass; drop `batch_evaluate`/`flattened_polynomials` for the 34 virtual inputs.
4. Optional: add streaming commits for the 8 committed inputs (or keep dense just for 8).
5. Remove legacy dynamic builder modules and types from production (`old_builder.rs`, `old_constraints.rs`, any dynamic constraint builders). Gate under `#[cfg(test)]` if still needed for validation; update `mod.rs`.
6. Update digest to constants; add a one-time compatibility check that old/new keys produce identical Stage-2 evaluations and outer claims on sample traces.

---

## No-materialization witness streaming (detailed plan)

Goal: avoid building `flattened_polynomials` for the 42 inputs during proving. Use a streaming accessor over `(trace, preprocessing)` for both Stage 1 and Stage 2 computations.

### New abstraction: WitnessRowAccessor

- Add a trait in `r1cs/inputs.rs` (or `r1cs` mod) to decouple row access from concrete storage:
  - `trait WitnessRowAccessor<F> { fn value_at(&self, input_index: usize, t: usize) -> F; }`
- Provide a concrete impl backed by `(trace, preprocessing)`, mapping `input_index` to `JoltR1CSInputs::from_index(input_index)` and calling specialized getters (reusing existing `generate_witness` logic per-row, without materializing vectors).

### Stage 1 (outer sumcheck) without polynomials

- `SpartanInterleavedPolynomial::new_with_precompute`:
  - Change signature to accept `&dyn WitnessRowAccessor<F>` instead of `flattened_polynomials`.
  - Replace `ConstraintConst::evaluate_row(flattened_polynomials, t)` with a new helper `evaluate_row_with_access(accessor, t)` implemented over `ConstLC`:
    - `ConstLC::for_each_term(|idx, coeff| acc += coeff * accessor.value_at(idx, t));` then add const term.
  - Keep SVO logic and accumulators unchanged.
- `SumcheckInstanceProof::prove_spartan_small_value`:
  - Replace `flattened_polys` param with `&dyn WitnessRowAccessor<F>` and plumb it to `new_with_precompute`.
- `spartan.rs` Stage 1 call sites:
  - Construct a `TraceWitnessAccessor` and pass it through.

### Claimed witness evals at r_cycle without polynomials

- Add `compute_claimed_witness_evals_streaming(r_cycle: &[F], accessor: &dyn WitnessRowAccessor<F>, num_steps: usize) -> Vec<F>`:
  - Precompute `eq(r_cycle, t)` for all `t`.
  - Initialize `claims[num_inputs] = 0`.
  - Loop `t in 0..num_steps`: for each input `i`, do `claims[i] += weight_t * accessor.value_at(i, t)`.
  - Store the constant position evaluation if needed by PCS layout.
- Use this result wherever we previously `batch_evaluate`d variable polynomials.

### Stage 2 (inner sumcheck) inputs (z(ry)) without polynomials

- We only need `segment_evals[i] = P_i(ry)` for all inputs:
  - Add `compute_segment_evals_streaming(ry: &[F], accessor: &dyn WitnessRowAccessor<F>, num_steps: usize) -> Vec<F>` with the same streaming pattern (weights = `eq(ry, t)`).
  - Keep the 8 committed inputs as dense polys if PCS demands; otherwise, extend PCS with a streaming commit/open path later.

### Stage 3 (PC sumcheck)

- Unchanged for now. Optional: replace with a tiny streaming accessor because it only uses a few inputs (`UnexpandedPC`, `PC`, `IsNoop`).

### API diffs for streaming

- `inputs.rs`:
  - Define `WitnessRowAccessor<F>` and `TraceWitnessAccessor<'a, F, PCS>`.
  - Implement `value_at` for all `JoltR1CSInputs` variants by reading from `(trace, preprocessing)` per row.
  - Add `compute_claimed_witness_evals_streaming` and `compute_segment_evals_streaming` helpers.
- `constraints.rs`:
  - Add `ConstLC::evaluate_row_with(accessor, t)` or reuse `for_each_term` to compute row sums given an accessor.
- `poly/spartan_interleaved_poly.rs`:
  - Update `new_with_precompute` to accept `&dyn WitnessRowAccessor<F>` and use `evaluate_row_with`.
- `subprotocols/sumcheck.rs`:
  - Update `prove_spartan_small_value` signature to accept the accessor instead of `flattened_polys`.
- `spartan.rs`:
  - Thread the accessor from the state manager to Stage 1, and use streaming helper for claimed witness evals at `r_cycle`.

### Performance expectations

- Time: eliminate allocation and iteration over 42 dense polynomials; the streaming accessor reads only what is needed per row.
- Memory: drop 42 input polynomials; keep at most the 8 committed (or stream them as a follow-up).
- Cache locality: tight loops over small constant rows and direct trace access.

---

## Testing and validation

- Keep and extend `const_constraints` tests comparing const vs dynamic constraints until dynamic is fully deleted.
- Add tests to compare:
  - Outer sumcheck claims (Az, Bz, Cz) old vs new on a fixed trace.
  - Inner evaluator outputs old vs new at random points (same transcript seed).
  - Streaming `claimed_witness_evals` and `segment_evals` vs `batch_evaluate` equality.
- Fuzz random traces/instructions for a small number of cycles and assert transcript equality up to permutation-invariant ordering.

---

## Performance expectations

- Time: remove LC allocation/sorting and sparse materialization; Stage 1 row evals are tighter via `ConstLC`; Stage 2 avoids walking sparse matrices.
- Memory: drop 42 dense polynomials upfront; keep at most the 8 committed (or stream them too). Remove `UniformR1CS` from the key.
- Cache behavior: tighter loops over small fixed-size rows; fewer indirections.

---

## Risks and mitigations

- Digest stability: define a canonical serialization of `UNIFORM_R1CS` (sort terms by input index; include const term if any); test determinism.
- Constant column index: keep `const_col = JoltR1CSInputs::num_inputs()` convention in all evaluators and tests.
- PCS assumptions: if PCS needs dense buffers for commits, keep the 8 committed polynomials materialized; this still yields large memory/time wins.

---

## Work breakdown (checklist)

- [x] Refactor key to constants, remove builder/UniformR1CS from prod, update digest.
- [x] Port Stage 1 precompute and sumcheck to constants; remove dynamic constraint args.
- [x] Gate legacy modules under tests.
- [ ] Add `WitnessRowAccessor` and implement streaming row access in `inputs.rs`.
- [ ] Replace `flattened_polynomials` with accessor in `spartan_interleaved_poly.rs` and `sumcheck.rs`.
- [ ] Implement streaming `claimed_witness_evals` and `segment_evals`.
- [ ] Keep/optional: streaming commits for 8 committed inputs.
- [ ] Extend tests and benchmarks to validate streaming path and measure RSS/time improvements.


