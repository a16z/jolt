# W4-S review 1b (independent, fresh reviewer; same scope as review #1)

Scope: commit `226c8f3d7` — `crates/jolt-wrapper/src/stream.rs`, `stream/types.rs`, `spartan.rs`,
`tests/`, journal `w4s-stream-spartan.md`. Line numbers refer to the commit, not the working tree
(the implementer was already editing `stream.rs`/`spartan.rs`/`types.rs` in place while this
review ran; `.journals/lanes/w4s-review-1.md` already held the codex review, so this file is 1b).

## Checks run (on the commit's tree, before the WIP edits landed)

- `cargo nextest run -p jolt-wrapper --cargo-quiet`: 2 passed, 1 ignored (`n3_g_shape_timing`).
- `cargo clippy -p jolt-wrapper --all-targets -q --message-format=short -- -D warnings`: clean.
- `cargo fmt -p jolt-wrapper --check`: clean.
- Ignored gate not run: it executes `jolt-wrapper-bench`, not this crate (finding 4).

## What holds (verified against jolt-sumcheck / jolt-hyperkzg / jolt-poly sources)

- Compressed rounds: the verifier reconstructs `c1` from the running claim
  (`jolt-sumcheck/src/verifier.rs:117-127`), so `s(0)+s(1)=claim` holds by construction; sound
  because `degree() > claim.degree` and empty polys are rejected (`verifier.rs:111-120`) and the
  degree comes from the verifier's `StageMemberSpec`/hard-coded 2, never from the proof. Round
  count is pinned by `WrongNumberOfRounds` (`verifier.rs:99`), so `EqPolynomial::mle` length
  asserts downstream cannot panic on a malformed proof. No verifier-side panic path found.
- Final-claim binding: `verify_stage_with` checks `final == Σ coeff_i·out_i`
  (`stream.rs:395-403`); `verify_stream` derives the opened value as `final/coeff` (`:912-915`),
  so the prover cannot pick the opened evaluation.
- Stage C: claim values absorbed before every `rho` (`:823-826` / `:879-882`); verifier recomputes
  `q_g(r) = Σ rho_i·eq(p_i, r)` from statement points (`reduction_weights`, `:946-971`) and
  verifies `combine(C_g, q_g(r))` at `final/coeff` — each `v_i = P_{g_i}(p_i)` is bound with error
  `≈ (#claims + rounds)/2^128`. A different column vector with the same packed evaluation at a
  random `(r_row, s_slot)` is a sumcheck/eq collision, negligible — *provided* the verifier derives
  `T̃(s_d) = Σ_g eq(s_d^hi, g)·v_{g,d}` itself (it does not today: finding 1).
- Layout: `(row, slot) → row·k + slot`, row vars high / `log2 k` slot vars low, big-endian bit
  order everywhere (`commit_packed:152-158`, `PackedColumns::point:74-77`, `boolean_point:986`,
  `EqPolynomial::mle/evals`, `Polynomial::evaluate`, HyperKZG fold) — unambiguous and identical
  prover/verifier. Padding slots are zero for the honest prover; a malicious prover can put garbage
  there but `Q̃` has no support on padded column indices and stage A never reads them, so it is
  irrelevant to the relation. `ColumnBatching` padding (`resize(64, 0)`) matches the zero slots.
- `BatchPrelude` scaling: `ScaledRounds.scale = 2^(max−offset−rounds)` (`:329`) equals
  `prove_batch`'s start scale `2^(max−rounds)` after `offset` halvings (`prover.rs:246-280`); the
  verifier's `BatchPrelude::new` uses the same `claimed_sum` law (`batch.rs:59-64`). Consistent
  — but never exercised (finding 3).
- Spartan: `x` absorbed before `tau` (`spartan.rs:62-66`); `az,bz,cz` absorbed before `r_a,r_b,r_c`
  (`:81-84`) so the inner claim binds them individually while the outer check binds `az·bz−cz`;
  public contributions recomputed from `x` (`:86`, `:190`); witness columns `[1+|x|, num_vars)`
  with `witness_len = num_vars − 1 − |x|` enforced (`validate_dimensions`); `W(ry)` bound by
  HyperKZG. Both sides absorb in the same order (`:109-111` vs `:218-227`).
- Edge cases: `k ∤ columns` → zero-filled slots (`:154-157`); a kind with zero columns → empty
  `g1_bit_columns_msm` / generic path; non-power-of-two rows → `RowCount` error (`:117-123`).
- Bytes: Spartan `32·(1+60+4+48) = 3,616`, bincode 3,662 — matches `payload_bytes`/`bincode_bytes`
  and the test.

## Findings

1. **BLOCKER — `stream.rs:854-877`, `tests/stream_synthetic.rs:212-262`: `verify_stream` cannot be
   run by a verifier.** Prior stages are checked against caller-supplied `StageClaims.output`
   (`verify_stage`, `:365-377`) and stage-C points/values against caller-supplied
   `ReductionClaimRef`s; the test feeds both from prover state (`row_result.output_claims`,
   `column_result.output_claims`, `packed.evaluations`). Nothing derives stage-B's output
   `Q̃(s)·Π T̃(s_d)` from the reduced claims, nothing checks `out(A) == in(B)`, and `StageResult`
   has no per-member opening point. Structural corollary: for the planned multi-member stage A
   (T1 row + T2 row + Spartan outer) the per-member outputs are prover data and `WrapperProof`
   has no field for them — `final/coeff` recovers only a single member. Fix: (a) replace
   `prior_stage_claims: &[StageClaims]` with per-stage `FnOnce(&StageResult) -> Result<Vec<Fr>>`
   (or a `StageStatement` trait) mirroring `verify_stage_with`; (b) add `stage_claims: Vec<Fr>`
   to `WrapperProof` for transmitted member outputs, absorbed where `recorder.finish` absorbs them
   today; (c) in the synthetic test compute `T̃(s_d)` from `reduced_claims` + `eq(s_d^hi, g)` and
   `ColumnBatching::expected_final` verifier-side; (d) add `StageResult::member_point(i)`. Same
   as review #1 finding 1; (b) and (d) are additional.

2. **MUST-FIX — `stream.rs:883-891`, `:905-911`: final-stage shape comes from the proof.** Rounds
   = `proof.opening.com.len() + 1` and polynomial count = `proof.commitments.len()` are
   prover-controlled. Today a wrong round count is caught only indirectly by
   `reduction_weights`'s point-length check (`:962-967`), which is vacuous when `claims` is
   empty, and extra commitments are silently accepted with zero weight. Fix: pass the packed
   shape (`rows`, `k`, `groups`) as statement; reject `commitments.len() != groups`; set
   `rounds = log2(rows·k)`; let HyperKZG's `WrongCommitmentCount` do the rest.

3. **MUST-FIX — no test exercises `ScaledRounds` / `BatchPrelude` padding.** Every stage in both
   tests is single-member with `offset = 0` and `rounds = max` (scale 1). The plan's stage A
   scales Spartan outer by `2^4` inside an 18-round batch; an untested adapter there is the
   likeliest integration failure. Fix: a two-member stage test (e.g. row relation 2^12 + a 2^10
   head-aligned member, and one tail-aligned) verified end-to-end, plus a tamper on the short
   member's output.

4. **MUST-FIX — `tests/stream_timing.rs:16-40`: the gate benchmarks `jolt-wrapper-bench`, which
   does not depend on `jolt-wrapper`** (`crates/jolt-wrapper-bench/Cargo.toml`; its own
   `sumcheck::{prove_stream, verify_stream}`). The journal's 12,960 B / 4.765 s / 2.68 GB are
   N3's numbers, not this commit's. Fix: build the N3-G fixture on `jolt_wrapper::stream`
   (`commit_packed` → stage A/B → `prove_reduced_opening`) and assert
   `WrapperProof::bincode_bytes()`; until then strike the attributed measurements from the
   journal. Same as review #1 finding 4.

5. **MUST-FIX — `spartan.rs:285-317`: `public_contributions` walks all three matrices three times
   with two zero weights each (9 matrix passes; the verifier's O(nnz) budget in plan §7 is one).**
   Fix: add `ConstraintMatrices::column_range_contributions(row_weights, values, start, count)
   -> MatrixColumnContributions` in jolt-r1cs (one pass per matrix, reusing
   `matrix_bilinear_eval_columns`), and call it here and for the inner `linear_eval` (`:206-214`).

6. **MUST-FIX — tamper coverage.** Present: bit flip via re-commit, stage-0 round-0 coefficient,
   `reduced_claims[0]`, Spartan public input / round / claim / `opening.v[0][0]`. Missing:
   degree raise (append a coefficient to a round poly → must hit `DegreeBoundExceeded`);
   truncated/extended round list; stage-B and stage-C round tampers; stream opening tamper
   (`v`, `com`, `w`); swapped columns within a group (re-commit with two columns exchanged, reuse
   proof) and swapped `reduced_claims` at the same point; `ReductionClaimRef.point` tamper; extra
   commitment appended (currently accepted — finding 2); Spartan `reduced_claims[3]`
   (`witness_eval`), `commitments[0]`, inner-stage round, different satisfiable R1CS of identical
   dimensions under the target key (must reject), unsatisfied witness rejected by `prove_spartan`.

7. **MUST-FIX — `stream.rs:931`, `spartan.rs:61/:163`: no verifier-key / profile digest in the
   transcript** (plan §7 `vk_digest`). Not an acceptance bug here — the verifier's own matrices
   drive `public_contributions` and `linear_form_bilinear_eval`, so a proof for another R1CS fails
   the inner check w.h.p. — but the challenges are statement-independent, which the plan forbids.
   Fix: `new_stream_transcript(key_digest, commitments)` absorbing the digest first. Same as
   review #1 finding 2 (rated blocker there).

8. **MUST-FIX — `stream.rs:147`: no canonical padded group count when `ceil(columns/k)` is not a
   power of two;** the only column-point split lives in the test (`stream_synthetic.rs:208-211`,
   `commitments.len().trailing_zeros()`), wrong for 5 groups. Fix: `PackedColumns { groups_log2,
   .. }` plus `split_column_point(&[Fr]) -> (group, slot)` and `group_weights(&[Fr])` owned by
   the crate; test 33 and 237 columns. Same as review #1 finding 3.

9. **NIT — `stream.rs:317-324`: `StageWindow` unreachable** (`max_rounds` is the max of the same
   sums; `prove_batch` re-checks anyway). Delete branch and variant.

10. **NIT — `stream.rs:264-274`: `scale.inverse()` per round, `None` mapped to
    `RoundCheckFailed { actual: 0 }`.** Scale is `2^n`; precompute `scale_inverse` in the
    constructor and drop the fake error.

11. **NIT — two spellings of "absorb output claims":** `verify_stage_with:404-405` builds a
    `ClearSumcheckRecorder` only to call `finish`; `verify_stream:916` uses
    `append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, ..)`. One helper, one owner.

12. **NIT — avoidable 32 MiB clones at shape G:** `ClaimReduction::new:725` clones each packed
    polynomial once per claim (40 claims ≈ 1.3 GB transient); `commit_packed:225` clones for
    `commit` although `MultilinearPoly` is implemented for `[F]`
    (`jolt-poly/src/multilinear.rs:224`); `prove_reduced_opening:839` clones the combined vector
    for a self-check. Wrap once / pass slices.

13. **NIT — `stream.rs:80-102`: `PackedColumns::column_evaluations` has no caller.** Delete, or use
    it in the synthetic test in place of `RowRelation::finals()`.

14. **NIT — `spartan.rs`:** `:140-161` seven `ok_or(MalformedProof)` after the length check at
    `:137` (destructure `[az, bz, cz, w] = <[Fr; 4]>::try_from(..)`); `:467-475` `InnerSumcheck::new`
    reports a prover-side input mismatch as `InnerFinalClaim`; `:429-443` / `:511-522` and
    `stream.rs:783-799` run one table pass per evaluation point (`degree+1` passes) — fold into one;
    `types.rs:186-190` stringly-typed `Commitment/Relation/HyperKzg(String)` where
    `OpeningsError`, `ConstraintMatrixEvalError`, `HyperKZGError` exist (`#[from]`).

15. **NIT — docs/journal honesty.** `w4s-stream-spartan.md` says "input/output claims are
    statement-derived … not transmitted": false for `out(A) = in(B)` and for any multi-member
    stage (finding 1). `commit_packed:132-136` validates bitness prover-side only; state in the
    `Column` docs that booleanity/range of `Bits`/`U16` columns is the relation's obligation, not
    the commitment kernel's. `stream_synthetic.rs:1-5` enables `indexing_slicing` via `expect`
    while `spartan_core.rs` indexes freely — pick one.

VERDICT: 1 blockers, 7 must-fix, 7 nits
