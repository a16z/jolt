# W4-S review #1

## Fiat–Shamir trace

Generic stream order:

1. Domain label `jolt-wrapper-v1`; packed commitments in declaration order.
2. Per prior stage: input claims; one squeeze per member for batching coefficients; for each round, compressed coefficients then one squeeze; output claims.
3. Reduced claim values; one squeeze per claim for `rho`.
4. Stage C: reduced input claim; one squeeze for its batch coefficient; for each round, compressed coefficients then one squeeze; derived final evaluation.
5. HyperKZG: derived evaluation; fold commitments; squeeze `r`; all three evaluation rows; squeeze `q`; witness commitment `w`.

Spartan inserts public inputs after the witness commitment and squeezes fresh `tau` coordinates before the outer stage. After the outer output claim it absorbs `Az(rx), Bz(rx), Cz(rx)`, then squeezes the three inner-combination weights. After the inner output claim it absorbs `W(ry)`, then follows the HyperKZG order above. Successive `challenge()` calls rekey the BLAKE3 transcript, so no challenge value is reused. The round verifier reconstructs the omitted linear coefficient from the running claim and rejects stored degree above the declared batch maximum.

Gaps: no profile/verifier-key digest enters either transcript; the generic verifier also receives stage outputs and reduction references from its caller instead of reconstructing the stage chain.

## Findings

1. **BLOCKER — `crates/jolt-wrapper/src/stream.rs:870`: the verifier does not bind stages A/B to the reduced opening.** `verify_stream` verifies every prior stage against caller-supplied `StageClaims`, then separately verifies stage C against caller-supplied `ReductionClaimRef`s. It never derives the stage-C points from `r_A` and the stage-B points, reconstructs each column value as `sum_g eq(s_group, g) * P_g(r_A, s_slot)`, calls `ColumnBatching::expected_final`, or checks that stage B's input is stage A's output. The test makes the gap explicit at `tests/stream_synthetic.rs:253`: both output vectors and every reduction point come directly from prover-side state. A verifier without the witness cannot build these trusted arguments, and accepting prover-built values proves only self-consistency of three unrelated sumchecks. **Fix:** replace `prior_stage_claims`/free-form claim references with a relation-specific verifier driver that derives all stage inputs, outputs, points, group weights, and tensor values while replaying the transcript; feed the resulting stage-C values into the one-opening reduction. Add negative tests for a changed group coordinate, packed-polynomial index, stage-A point, and tensor term.

2. **BLOCKER — `crates/jolt-wrapper/src/stream.rs:931`: the transcript cannot be seeded with the profile digest.** `new_stream_transcript` and `verify_stream` absorb only commitments. Spartan adds public-input values at `spartan.rs:62`, but never binds the R1CS/profile identity. This contradicts `profile.rs:1` and permits the same Fiat–Shamir challenges across different profiles or verifier keys. **Fix:** make the transcript constructor require a canonical profile/verifier-key digest and public-statement encoding, absorb them before prover messages on both paths, and add a test proving under a different satisfiable R1CS/profile with identical dimensions and rejecting under the target key.

3. **MAJOR — `crates/jolt-wrapper/src/stream.rs:147`: packing has no defined group width when `ceil(column_count / k)` is not a power of two.** The function accepts such counts and emits only that many commitments, while the only column-point split uses `commitments.len().trailing_zeros()` at `tests/stream_synthetic.rs:209`. For five groups this returns zero group variables, although the padded column domain needs three. The real 237-column, `k = 8` shape hits this case unless callers silently add columns. **Fix:** give `PackedColumns` one canonical padded column/group count and helpers that split a column point and compute group `eq` weights; either pad missing groups as zero polynomials or keep implicit zero groups with the padded width. Test 33 and 237 columns, mixed-type final groups, and all-zero padding.

4. **MAJOR — `crates/jolt-wrapper/tests/stream_timing.rs:16`: the claimed timing/size gate does not execute this implementation.** It runs `jolt-wrapper-bench`, whose private `sumcheck::{prove_stream, verify_stream}` is separate code; the asserted `11936` therefore says nothing about `stream.rs`, `ClaimReduction`, packing, or its HyperKZG opening. The implementation note's 4.765 s and 2.68 GB figures are not evidence for this commit. **Fix:** move the N3-G fixture onto `jolt_wrapper::stream` and measure that path, or delete this test and the attributed measurements until such a benchmark exists.

5. **MINOR — `crates/jolt-wrapper/src/stream.rs:317`: `StageWindow` is unreachable.** `max_rounds` is computed as the maximum of the same `offset + rounds` values immediately before the check, so no member can exceed it. **Fix:** remove the branch and error variant, or accept an independently declared stage width that makes the validation real.

## Checks

- `cargo nextest run -p jolt-wrapper --cargo-quiet`: passed; 2 passed, 1 ignored.
- `cargo clippy -p jolt-wrapper --all-targets -q -- -D warnings`: passed.
- `stream.rs`: 990 lines; no scoped source file exceeds 1,000 lines.
- Compressed-polynomial reconstruction, verifier degree checks, `BatchPrelude` scaling, Spartan public/witness column ranges, fresh `tau`, combined HyperKZG commitment/evaluation verification, and the 3,616/3,662-byte Spartan assertions match the code.

VERDICT: 2 blockers, 2 majors, 1 minor
