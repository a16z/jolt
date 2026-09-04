# PERF-5 independent review — `a9d8828dd..9a6643df5`

**0 blockers / 0 majors / 4 minors.**

Date: 2026-09-04. Zero-trust review of the PERF-5 prover-time lanes on
`wrap/spartan-hyperkzg`. Read-only detached worktree at `9a6643df5`,
`CARGO_TARGET_DIR=/Volumes/Dev/target/perf5-review`. Every claim below was
re-derived from code; journal numbers were not taken on trust. Negatives written
during the review are in `perf5-review-tests.patch` (not committed as code).

The four-ary HyperKZG fold, the verifier's claimed-value recheck, the hybrid MSM,
the stream tail, and the T2/CopyLink changes are sound as written. The k=32
proof/verifier numbers reproduce exactly. The minors are a mis-stated verifier
operation count, a vacuous-by-default trait method, a documentation
classification error, and two coverage gaps the patch closes.

## 1. Four-ary fold soundness — sound

**Root of unity.** `ROOT_LE` (`kzg.rs:32`) decodes to
`21888242871839275217838484774961031246007050428528088939761107053157389710902`.
Independently confirmed: it is canonical (`< p`), equals `5^((p-1)/4) mod p`,
and squares to `-1`. `FoldPoints::new` re-validates `i² = −1` at runtime
(`kzg.rs:37-39`) and both sides construct `FoldPoints` from the same constant, so
prover and verifier cannot disagree on the root — and the sign of `i` cannot
desynchronize the point order from the residue formula, because both come from
`self.root`.

**Inverse DFT.** With `a = P(r), b = P(ir), c = P(−r), d = P(−ir)` and
`scale_k = 1/(4·r^k)` (`kzg.rs:47-53`), `residues` (`kzg.rs:65-70`) returns

```
R0 = (a+b+c+d)/4          R1 = (a − c + i(d−b))/(4r)
R2 = (a − b + c − d)/(4r²) R3 = (a − c − i(d−b))/(4r³)
```

which is exactly `Σ_u u^{-k} P(u) / 4` for `u ∈ {r, ir, −r, −ir}`. Order matches
the fold's weight order, since `fold_polynomials` maps chunk `[a00,a01,a10,a11]`
= evaluation indices `4m..4m+3` against `rchunks_exact(2)` variables `[x,y]`
(`x` = bit 1, `y` = bit 0). The decomposition `P(X) = Σ_k X^k R_k(X⁴)` is exact
for **any** degree, so the residue step needs no degree bound.

**Degenerate challenges.** `kzg.rs:44` rejects `r = 0` and any `r` with
`r⁴ ∈ {r, ir, −r, −ir}` (i.e. `r¹² = 1`), which is exactly the condition making
the five points distinct and `s⁴ − s` invertible in `interpolate`. Fail-closed on
both sides (`DegenerateChallenge`); there is no re-draw, so a degenerate `r`
makes the statement unprovable rather than unsound. Covered by
`four_point_dft_recovers_residues` for `r ∈ {0, ±1, ±i}`.

**Divisor and pairing.** `divisor()` = `[s², −s, 0, 0, −s, 1]` =
`X⁵ − sX⁴ − sX + s²` = `(X⁴ − r⁴)(X − r⁴)`, monic, roots exactly the five points.
Expanding the multi-pairing (`kzg.rs:289-297`) gives
`B(β) − R(β) − W·(β⁵ − sβ⁴ − sβ + s²)`, i.e. the standard batch identity; VK
powers `β⁴/β⁵` in G2 and `β³/β⁴` in G1 are exactly what the sparse divisor and
the degree-4 remainder need. `interpolate` returns
`cubic(X) + correction·(X⁴ − s)`, which agrees with the four points (where
`X⁴ = s`) and hits `y` at `X = s`.

**Fiat–Shamir order.** `open` and `verify` both append every fold commitment
before drawing `r` (`scheme.rs:187-191` / `259-262`); `kzg_open_batch` and
`kzg_verify_batch` both append rows `v[0..4]` and `p0_at_r_fourth`, then draw
`q`, then append `w` — identical order on both sides. The statement is bound
earlier: the wrapper appends `claim.value` before calling `open`/`verify`, and
the opened commitment is `combine(commitments, weights)` over commitments
already absorbed by `assembly_transcript`.

### Fold-level ledger — what each of the five openings pins

| opening | pins |
|---|---|
| `P_j(r)`, `P_j(ir)`, `P_j(−r)`, `P_j(−ir)` | the four residues `R_0..R_3(r⁴)` of `P_j`, exactly (invertible 4×4 Vandermonde; `r ≠ 0`, `i² = −1`) |
| `P_{j+1}(r⁴)` (fifth row, **derived by the verifier**) | `P_{j+1}(r⁴) = Σ_k w_k R_k^{(j)}(r⁴)`, i.e. `P_{j+1}` is the two-variable fold of `P_j` at `(x,y)` |
| `P_0(r⁴)` (fifth row, entry 0, prover-supplied) | nothing structurally; it only completes the batch. It is pinned to the truth by the batch itself, and it is transcribed before `q` |
| terminal row (`.last()` of each `v` row) | the claimed evaluation: even `ℓ` folds four residues at `point[0..2]`; odd `ℓ` uses `binary_residues(P(r), P(−r))` at `point[0]` |

Chain coverage: `y_fourth` has `levels + 1` entries and the batch binds row 4 for
**every** polynomial, so each level `j → j+1` is pinned. Deviation from the true
fold is a fixed nonzero polynomial evaluated at the random `r⁴`, so it is caught
except with probability `O(deg/|F|)`. No per-level degree bound is required.

**Negatives run** (`review_unbounded_and_predicted_point_folds_reject`, ℓ = 5 and
6). All polynomials are zero-padded to one length before committing, so
`kzg_open_batch` produces a *genuinely valid* five-point opening of exactly what
was committed — the test asserts `kzg_verify_batch` accepts it — and only the
protocol-level fold binding rejects:

| attack | rejection |
|---|---|
| middle fold given a degree-127 term (no per-level degree bound exists) | `PairingCheckFailed` |
| terminal fold given a degree-127 term | `FoldingConsistencyFailed` |
| fold shifted by `(X⁴ − ĝ⁴)(X − ĝ⁴)` for a *predicted* challenge `ĝ` — agrees with the honest fold at all five points of `ĝ` | `PairingCheckFailed` (committing the tampered fold moves `r ≠ ĝ`; asserted) |
| residue order swap inside a fold | `PairingCheckFailed` |

The existing `inconsistent_fold_with_valid_kzg_openings_rejects` reproduces
(`PairingCheckFailed`, ℓ = 5 and 6).

**Degree-bound honesty.** `scheme.rs:270-274` and `protocol.rs:913-916` both say
the fold commitments carry only HyperKZG's SRS-wide bound and that the
shifted-commitment proof bounds the *round* polynomials, not the folds. That is
honest, and the two unbounded-fold attacks above show no per-level bound is
needed.

## 2. Claimed-value check — verifier-side, non-vacuous

`verify_observed` recomputes `terminal` from `proof.v` and compares against its
own `claimed_eval` (`scheme.rs:308-310`) before the pairing check; `open`'s
`derived_claim` check (`scheme.rs:176-178`) is a prover convenience, not the
enforcement point. Lane 1 replaced a redundant full MLE re-evaluation in
`prove_direct_opening` with this free check — equivalent because every fold level
evaluates to the same value (pinned by
`odd_and_even_folds_preserve_geometry_and_evaluation`).

Mutation test: deleting the verifier's `terminal != claimed_eval` check fails
exactly four tests — `wrong_eval_rejects`, `wrong_eval_rejected`, and the two
review negatives — and nothing else. The pre-existing pair only reaches the
**odd** terminal branch; `review_even_terminal_rechecks_claim` (ℓ = 2, 4, 6)
covers the even branch and asserts the specific
`FoldingConsistencyFailed`.

## 3. Hybrid MSM — correct

- **Signed-digit bucket lists.** `heads[bucket]` packs `index+1` with the sign of
  *that* node's digit in the top bit, and `next[index]` holds the previous head,
  so each node carries its own sign through traversal (`msm.rs:189-197`,
  `207-217`, `232-242`). `index+1 ≤ len` never collides with the sign bit.
- **Booth digits.** For width `w`, `booth_digit` returns `[−2^{w−1}, 2^{w−1}]`
  (hand-checked for `w = 3` across all 16 inputs, including the `bits = 2^w − 1`
  carry case that yields 0), so `digit.unsigned_abs() − 1 < 2^{w−1}` =
  `bucket_count`. `small_msm_16_bit` passes unsigned digits `≤ 65535` with
  `bucket_count = 65535`; both stay in range.
- **Affine batch adds.** `x` equal with `y` differing sets the bucket to identity
  and skips scheduling (BN254 has no 2-torsion, so the doubling denominator `2y`
  is nonzero); identity points are skipped; `batch_invert` never sees a zero.
- **Dispatch.** `as_u16_scalar` requires the top three limbs zero and
  `limb0 ≤ 0xFFFF` on the **canonical** (`into_bigint`) representation, so a
  scalar `≥ 2^16` — including `−1` — can never reach the u16 kernel; it goes to
  `pippenger_bigints`. `g1_msm_small`'s 16-bit path is a *bucket-width* choice,
  not a scalar-width one: for `S = u32` it folds `WINDOWS.div_ceil(2) = 2`
  sixteen-bit windows covering all four bytes, and `u8` is excluded by
  `S::WINDOWS >= 2`. `small_scalars_are_skewed` only picks between two kernels
  that compute the same value.
- **Empty-slice edges.** All-small dispatches before the split; an all-zero small
  side yields an empty `small_msm` (identity); `pippenger_bigints`' non-empty
  `debug_assert` cannot fire.

Coverage: the pre-existing suite reaches the long-chain projective path only via
`full_width_msm_handles_skewed_digits`. `review_hybrid_signed_chains_match_naive`
adds a randomized differential against the naive reference at `n ∈ {64, 65, 129,
1025}` driving `hybrid_buckets` + `sum_buckets` directly with mixed ±1 long
chains and `[−128,128]` digits, over point sets containing identities, duplicates
and negated duplicates, plus a mixed-width `g1_msm` case
(`0, 1, 65535, 65536, 65537, u32::MAX`, random) and a direct `small_msm_16_bit`
u32 case. `jolt-crypto` + `jolt-hyperkzg`: **178/178 pass**.

## 4. Stream tail — no live column can be dropped

Structural guarantees, then measurement.

- `column_evaluations_from_bound` (`packing.rs:297-319`) starts from
  `vec![None; column_count]` and fails with `OpeningClaim` unless **every**
  column is filled, and fails on two different values for one column. A dropped
  live column is impossible, not merely unlikely.
- `zero_columns` is the exact complement of the live set: `key.rs:211-220` marks
  `member_columns ∪ hash_vk_columns` live over `total_groups * packing` and takes
  the rest. One owner, one enumeration.
- A member that under-reports is caught by
  `values.len() != columns.len()` → `StageMemberCount` (`protocol.rs:236-238`).
- Even if a live column were mis-classified, it is a completeness failure, not a
  soundness one: `rlc_evaluations_skipping` feeds `HyperKZGScheme::open`, whose
  claim check fires, and the verifier opens `combine(commitments, weights)` over
  the unskipped commitments.

Measured with the review hook (`PERF5_REVIEW_TAIL_CHECK=1`), real k=32 shape:

```
review_tail: slots=832 member_bound=592 unbound_live=6 padding=234
review_tail: padding-skipped RLC == unskipped RLC over 8388608 slots
```

`592 + 6 + 234 = 832`, and `592 + 6 = 598` live columns matches lane 6's
ownership table (307 T1 + 140 CopyLink + 144 T2 + 1 carry + 6 T1 VK). Three
assertions held on every test that reaches the tail (64/64 wrapper tests):

1. reused stage-A bound values `==` a full `column_evaluations(r_A)` scan;
2. padding-skipped RLC `==` unskipped RLC over all 8,388,608 slots;
3. the typed/skipping RLC `==` an independent naive oracle that promotes every
   packed entry to a field element (4,096-slot prefix).

## 5. Representation-only lanes — **two are not** (minor 3)

- **Lane 3 (typed T2 rounds) is representation-only.** Verified by reading:
  `phi`/`linear` now take a hoisted `z_xi` (previously computed twice,
  identically); the `fp` fingerprint loop accumulates once and snapshots at
  `FP_SLOTS_G1 = 2` and `FP_SLOTS_G2 = 4`, reproducing `fp(2)`, `fp(4)`,
  `fp(22)`; the inlined `mask_g1_g2 / mask_g2 / mask_gt` reproduce the old
  `copy_mask(v, s)` case-for-case. No term added or removed.
- **Lane 2 (CopyLink) is not.** Beyond the sparse prover (`CopyLinkValueSource`,
  lazy per-row values — prover-only), it repacked the 20 helper columns into the
  T2 final fill (−3 commitments, −96 B payload) **and moved challenge indices**:
  `t1_challenge_offset` `2·copies → 0`, and copy `β/γ` moved from phase 1a to the
  first T2 phase (`copy_challenge_offset = t2_challenge_offset +
  PHASE_CHALLENGES[0]`, with that phase's `challenge_count += 2·copies + 1`).
  **This is safe and strictly stronger**: `β/γ` are now drawn *after* the
  phase-1b commitments and still before the last-phase helper commitments that
  consume them; T1's challenges stay in phase 1a (re-indexed only); and the extra
  block is appended at the end of the T2 phase-0 draw so `ξ, α` keep their
  positions. `T2Challenges::from_transcript` splices around the interleaved block
  identically for prover and verifier.
- **Lane 5a (s = 4) is a protocol parameter change**, not a representation
  change: 22 → 17 helpers, round degree `GROUP_SIZE + 2` = 6, five final factors.
  Soundness re-derived: `HELPER_COLUMNS = RANGE_COLUMNS.div_ceil(GROUP_SIZE)` and
  `range_group(g) = 4g..min(4g+4, RANGE_COLUMNS)` **tile `[0, 66)` exactly** —
  16 full groups plus a 2-wide tail holding digit bits `e1, e2` — so no range
  column escapes the LogUp. `range_logup_numerator` computes
  `e_{|g|−1}(f)` by prefix/suffix for the actual group length (equal to the old
  hardcoded `f1f2 + f0f2 + f0f1` at length 3), giving `Σ_i 1/(α − c_i)` per group.
  `leading_coefficient` skips short groups, which is correct because their degree
  is `2 + |g| < GROUP_SIZE + 2`. `range_group` is the single owner used by `phi`,
  `linear`, `export_terms` and `leading_coefficient`, and `export_terms` is
  shared by prover and verifier.

**Tamper coverage.** `tamper_suite` is proof-level and unchanged except the
`p0_at_r_squared → p0_at_r_fourth` rename; it mutates every commitment, every
round polynomial coefficient, every stage claim, every `v` entry, `w`, and each
fold commitment — all rejected in the gate run. Witness-level range escape is
covered separately by `tampered_witnesses_are_rejected`
(`c[Col::CHUNKS + 5] += 1 << 16`, a full group under s = 4) and CopyLink helper
mutation by `assert_t2_commitment_row_tamper_rejected` and the T2 row tampers.

## 6. Numbers — reproduced, one real gate run

`cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet
real_wrapper --no-capture`, fixture `fibonacci_2_18_blake3.bin`, default k = 32
(the review target predates lane 7), pristine `9a6643df5` (review edits stashed).
Mutex `/tmp/wrapper-gate.lock` held 00:01:24–00:01:58 ET, acquired at one-minute
load 3.53 on the second attempt.

| item | required | measured |
|---|---:|---:|
| payload | 7,104 B | **7,104 B** |
| bincode | 7,232 B | **7,232 B** |
| statement | 352 B | **352 B** |
| ecMul | 216 | **216** |
| pairing pairs | 8 | **8** |
| Fr mul | 123,121 | **123,121** |
| modeled gas | 4,800,225 | **4,800,225** |

Also observed: HyperKZG opening 1,952 B, 11 fold commitments, ecAdd 216,
`fr_inv` 8, Keccak 839, 500 terms, 9 term rounds; every tamper rejected.
Honest online wall 19.803 s (five-minute load was 8.75 at entry, so this is a
correctness gate, not a clean idle timing point — lane 5b's 19.671 s stands).

The gas figure decomposes exactly:
`21,000 + 16·9,696 + 7,700·216 + 20·123,121 + 1,769 + 100·839 + 3·46,000 +
34,350·8 = 4,800,225`, with calldata `9,696 = 7,104 + 32·74 + 224` (the
`32·proof_g1` term upgrades the 74 compressed G1 points to on-chain 64-byte
form — correct, not double counting). The model's re-parameterization in this
range (`2·114,700 + 183,400` → `3·46,000 + 34,350·pairs`) is value-preserving at
8 pairs and now scales with the pairing count.

## Minors

1. **`kzg_verify_batch` over-reports ecAdd by 2** (`kzg.rs:287`). The routine
   performs `(k−1) + 4 + 2 = k+5` group additions; it charges `k+7`.
   `verify_direct_opening` (`protocol.rs:948`) likewise charges
   `commitments.len()` adds for an MSM that does `len−1`. The published
   "216 ecAdd" is therefore ~3 high. No gas impact — `estimated_gas` does not
   price ecAdd at all — and the direction is conservative, but the statistic
   quoted in the lane journals and PR table is not the operation count the code
   performs. Either charge `k+5` or document the count as an upper bound.
2. **`ProveRounds::append_bound_values` defaults to a no-op**
   (`jolt-sumcheck/src/prover.rs:73`). The `values.len() != columns.len()` guard
   makes a forgotten override loud only for members with ≥ 1 planned column;
   member index 1 currently has an empty `member_columns` entry, so that slot
   passes vacuously today. A future member that owns columns but forgets the
   override *and* is planned with zero columns would silently contribute zeros.
   Consider making the method required, or asserting that only known
   column-less members have empty plans.
3. **Lane classification.** Lanes 2 and 5a are described as prover-side work but
   are protocol-visible (challenge re-indexing plus a commitment-layout change;
   a LogUp group-size parameter change). Both are sound, and the journals do
   report the byte/gas deltas — but "representation-only" should not be used for
   them in the PR narrative.
4. **Coverage gaps closed by this review** (patch, not committed as code): no
   negative reached the even terminal claim recheck; no negative showed that a
   fold with unbounded degree and a fully valid five-point batch is rejected
   (the existing `inconsistent_fold` test tampers a coefficient, not the degree);
   no randomized differential covered the hybrid bucket path across signed
   digits, identity points and duplicate/negated points. Recommend adopting the
   three tests in `perf5-review-tests.patch` (the env-gated tail hooks are review
   scaffolding and should be dropped).

## Gates run

| command | result |
|---|---|
| `cargo nextest run -p jolt-hyperkzg -p jolt-crypto --cargo-quiet` (with review negatives) | 178/178 pass |
| mutation: delete verifier `terminal != claimed_eval`, `--no-fail-fast` | 4 fail (2 pre-existing + 2 review), 28 pass — check is load-bearing |
| `PERF5_REVIEW_TAIL_CHECK=1 cargo nextest run -p jolt-wrapper --no-fail-fast` | 64/64 pass, all three tail differentials hold |
| `PERF5_REVIEW_TAIL_CHECK=1 … real_wrapper --no-capture` (synthetic) | pass; 7,104 / 7,232 / 352 B; tail ledger printed |
| real k=32 gate under `/tmp/wrapper-gate.lock`, pristine tree | pass; all six required numbers exact; every tamper rejects |

Independent verification of `ROOT_LE` used `uv run python` against the BN254 Fr
modulus. Scratch worktree removed after handoff.
