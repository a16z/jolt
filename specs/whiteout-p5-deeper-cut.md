# P5: the deeper cut, k = 2 → 3

| Field | Value |
|---|---|
| Ticket | P5 (candidate #1 of `specs/whiteout-p1-profile.md`) |
| Status | design and security analysis; no production code |
| Pins | jolt `whiteout/dev` `a23f63464`; Akita `~/akita-whiteout` `b7889ad52`; paper `akita-paper/sections/whiteout/*.tex` as read on 2026-11-05 |
| Indexing | paper fold k = Akita `cut_level + 1`. k = 2 is `ZK_CUT_LEVEL = 1` today; k = 3 is `cut_level = 2` |

## Decision

**Go, with conditions.** k = 3 is admissible for every measured shape, and
its security follows from the paper's existing lemmas. The new terms are
one more masked level's slices in δ_LHL and its fold errors in ε_≤k. Today's
planner cannot produce an admissible k = 3 row. The gain is also
smaller than P1 estimated at small traces. The conditions:

1. **Planner fix (blocking).** The selective-ℓ2 path
   (`akita-planner/src/schedule_params/candidate/recursive.rs:657`,
   `append_selective_l2_candidates`) skips `Masked` levels but not
   `AfterCut`. At cut 2 it replaces the residual's Σ-bound A_a (rank 9)
   with a rank-2 ℓ2 matrix, and admission refuses the row: "level 3
   commits the residual with A_a of rank 2, below its Σ-binding rank 9".
   It is latent at cut 1. A scratch copy that also skips `AfterCut` there
   plans rows that pass `ZkOpeningSchedule::admit` at 23–30 variables.
2. **Land P1 #7 (the ν = 24 response row) first.** At k = 3, fib 2^16
   drops from ν = 25 to ν = 24. On today's ν = 24 row that costs about
   +93 KB per proof (170 KB against 77 KB of response proof).
3. **The cut becomes per-row data, k ∈ {2, 3}**: k = 3 wherever a row
   admits it, k = 2 otherwise. k = 2 stays tested.
4. **A paper delta**, with author sign-off on the slice count in δ_LHL
   (D5 is updated; no new theorem).

Expected effect: −0.08 to −0.12 s on muldiv, −0.55 to −0.6 s on fib 2^16
and −1.05 to −1.15 s on fib 2^20 of prover tail. P1 estimated −0.45 to −0.6
s for muldiv. Muldiv gains little because the residual has a floor of about
1.05 M coefficients: Akita's late folds shrink their witness by small
factors.

## 1. Measured level shapes

**Probe.** A scratch copy of Akita at
`/private/tmp/claude-502/-Users-mgeorghiades-jolt/7158a861-26e9-46b3-bc41-16db65c02976/scratchpad/p5/akita`
has three edits: `validate_cut` accepts cuts 1–3, the planner fix of
condition 1 sits behind `P5_FIX`, and admission failures print. Its example
`p5_cut` plans `ZkCommitmentConfig<RecursiveCommitmentConfig<fp128::OneHot>>`
for a singleton one-hot group at cuts 1 and 2 and prints each level and
`ResidualShape`'s Gaussian coefficients:

```text
P5_FIX=1 RAYON_NUM_THREADS=4 ./target/release/examples/p5_cut 23 24 26 28 30
```

Jolt's keys carry precommitted groups that the probe omits. Each workload
is therefore matched to the singleton row whose k = 2 residual equals P1's
`|s|` minus the pads (about 20 K coefficients). fib 2^16 matches 26 variables
to within 1%. fib 2^20 sits between 28 and 30 variables (nearer 30), and
muldiv between 23 and 24, both of which give the same k = 3 row.

| variables | k = 2: w_2, blocks × rank, \|s_res\| | k = 3: masked L2 (d, rank A) | k = 3: w_3, blocks × rank, \|s_res\| | ratio |
|---|---|---|---|---:|
| 23 | 1.05 M, 4 × 10 (d 128), 1,253,504 | 64, 7 | 0.79 M, 6 × 9, 1,048,704 | 0.84 |
| 24 | 1.18 M, 9 × 9, 1,552,512 | 64, 6 | 0.79 M, 6 × 9, 1,048,704 | 0.68 |
| 26 | 1.70 M, 13 × 9, 2,224,256 | 64, 7 | 1.05 M, 4 × 10, 1,253,504 | 0.56 |
| 28 | 2.49 M, 19 × 9, 3,231,872 | 64, 7 | 1.18 M, 9 × 9, 1,552,512 | 0.48 |
| 30 | 3.67 M, 14 × 10, 4,284,544 | 64, 7 | 1.31 M, 10 × 9, 1,720,448 | 0.40 |

Every row has seven levels at k = 3, so the cut is never the last
nonterminal fold. No level after the cut carries a setup prefix.

**Applied to the workloads.** `|e|` ≈ 10·|s| + V + 0.35 M.

| | muldiv | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| \|s\| at k = 2 (P1, measured) | 1.44 M | 2.24 M | 4.18 M |
| \|s\| at k = 3 (derived) | 1.07 M (×0.74) | 1.27 M (×0.57) | 1.74 M (×0.42; ×0.38 at 28 variables) |
| \|e\| / ν | 11.3 M / 24 (same) | 13.6 M / 24 (from 25) | 18.4 M / 25 (from 26) |
| Σ (∝ \|s\|) | −66 ms | −182 ms | −368 ms |
| response, live-length steps (52 / 46 / 43%) | −82 ms | −188 ms | −379 ms |
| response, 2^ν steps (fused, opening, commit) | 0 | −262 ms | −428 ms |
| one more masked fold (P1's estimate) | +30 to +70 ms | +40 to +70 ms | +50 to +100 ms |
| **tail, net** | **−80 to −120 ms** | **−0.56 to −0.59 s** | **−1.08 to −1.13 s** |
| verify, 1 thread (24 ms per M of `N·z`) | −9 ms | −23 ms | −58 ms |
| verify, 16 threads | −2 ms | −3 ms | −8 ms |
| prefix bytes (planner estimate) | +2.8 KB | +2.8 KB | +2.9 KB |
| response-proof bytes | 0 | +93 KB, or 0 after #7 | ≈ 0 (ν = 25 row, 49 KB terminal) |

The step splits are P1's probe shares applied to its main-pass medians.

## 2. Security delta

Labels: **[P]** proven here, **[L]** follows from a cited lemma, **[W]**
needs paper-level work.

**Source cutoff** (`03_setting.tex`, Def. `wo:def:source-cutoff`) **[L]**.
The definition is already generic in k, and the paper lists
k ∈ {2, 3, 4} as admissible for Jolt. At k = 3:

- folds 1–3 work modulo one q (fp128);
- folds 1–3 check no norm in the clear: admission requires the ℓ∞ route at
  levels ≤ the cut;
- fold 3 passes no setup claim: `AfterCut` forbids a prefix, and every
  probe row has none at L3;
- fold 3 is not the last nonterminal fold: L3–L6 follow.

The residual relation is the same object with `w_3` in place of `w_2`.
Lemma `wo:lem:randomized-base` is stated for any k, and covers the widened
L2 maps and its enlarged response range.

**δ_fold** **[P]**. `one_shot_fold_target` gives each witness response at
a masked level the target 2^−136/((cut + 1)·W_ℓ). Summed over the
(cut + 1) levels, that is 2^−136 at any cut. So 2δ_fold ≤ 2^−135, as at
k = 2. The per-fold target tightens by log₂(3/2) bits.
`⌈log₂ masked_folds⌉` goes from 1 to 2 for a single group, so the
balanced union term L grows by at most 1. The probe's masked L2 passed
admission's one-shot digit check.

**δ_LHL and the ZK total** **[L]** (Theorem `wo:thm:zk`; counting **[W]**).
The masked L2 adds its B slices and its D image (45 R_64 columns each, at
2λ + 8 bits beyond the image). Each slice contributes 2^−133. The paper's
k = 2 count is 14 slices; the implementation's count is not tabulated
anywhere (ambiguity A1). The other terms are unchanged: T_att·ε/M =
2^−133.7, 2δ_fold = 2^−135, sampler 2^−149, attempt nonce 2^−148.7.

| slices S | δ_LHL | ZK total |
|---:|---:|---:|
| 14 (k = 2 today) | 2^−129.19 | 2^−129.11 |
| 16 (k = 3: one B slice and one D image) | 2^−129.00 | 2^−128.92 |
| 23 (k = 3, worst case: eight B slices and one D image) | 2^−128.48 | 2^−128.42 |

The total stays below 2^−128 for any plausible count.

**Hiding the extra level** **[L]**. L2's field messages are padded (H7,
exact) and its commitments hidden by LHL (H9). Its fold response never
leaves the prover (H1, inside δ_fold). The Gaussian covers `w_3` the same
way it covers `w_2`. `w_3`'s digits lie in the cut level's open alphabet,
base 8, the same alphabet as at k = 2, so each block's γ_b bound and the
closed-form width σ = αΓ√(Σγ_b²) apply unchanged.

**Knowledge soundness** **[L]**, with a ledger gap **[W]**. Theorem
`wo:thm:knowledge` is generic. ε_≤3 = ε_≤2 + the interactive errors of
fold 3. Clear Akita already pays those same errors, but this repo does not
restate Akita's per-fold ledger (A2). D27's response ledger depends only on
ν, ⌈log₂ m⌉ and ⌈log₂ n_×⌉. ν falls, so it goes from 310/q (≈ 2^−119.7)
to at most 295/q (≈ 2^−119.8). The `ν ≤ 27` bound still holds.

**Residual SIS instance** **[P]**, given the table.
R_Σ = 8κ₁·B(D) depends only on the digit count D = 10. σ shrinks with
√|s|, by 0.2–0.6 bits, which is less than one radix-16 digit (4 bits), so D
and R_Σ are unchanged. The residual commitment stays at d = 128. Its widths
(1,024 or 2,048 per block) are cells of the checked-in 128-bit Σ-binding
table, at rank 9–10 for both A_a and [B_a | B′_a], and admission passed at
those ranks with the fix. **R15's 128-bit Σ table still holds**, and
nothing needs regenerating. The digit-count condition
`⌊L(D)²/k²⌋ ≥ 1 + ⌈log₂(2·2^20·N)⌉` only gets easier as N and k fall.

**Σ and rejection parameters** **[P]**. α, ε = 2^−137, M, T_att, the
acceptance box and the per-attempt rejection rate depend on α and ε, not on
`|s|`, so none of them changes.

**Earlier folds' SIS** **[L]**, checked by the planner and admission. The
masked L2 is re-priced at its one-shot range to rank 6–7 at d = 64 (the
clear L3 had rank 4–5). This matches the paper (`06_evaluation.tex:157-161`:
"rank 6 instead of 5"). Akita's test W2-d, which asserts that the
re-priced ranks equal the clear ranks, holds only at k = 2.

**Transcript and Fiat–Shamir order** **[P]**. The transcript is a strict
insertion after L1's Stage 2:

- L1's next-witness commitment is now hiding, and its randomness sits in
  `w_2`;
- L2's fork at counter 0, with no nonce;
- L2's D image, hiding;
- L2's masked range and relation sumchecks;
- then `cm(u_a)` of L3's residual commitment and Σ, as before.

The pad schedule P grows by L2's slots, and n_× grows by L2's product rows.
The row descriptor already binds the cut (tag 0x5a), and the pads absorb
the catalog digest (D32). A proof for one cut therefore fails Fiat–Shamir
under the other.

## 3. Implementation slices

| # | Repo | Slice | Depends on |
|---|---|---|---|
| A1 | Akita | `validate_cut` accepts {1, 2}. The admission and descriptor tests run at both cuts. Restore a cut-2 masked-prefix test (W1's was deleted). | — |
| A2 | Akita | The planner fix of condition 1, with a test that cut 2 admits at the probe's sizes. | A1 |
| A3 | Akita | The masked prefix at cut 2: the prover and verifier loops are already generic over `0..=cut_level`. Add end-to-end, distinguisher (D30) and tamper tests at cut 2, plus the ZK schedule family rows. | A1, A2 |
| J1 | jolt | `jolt-akita` plans both cuts per row (`WhiteoutCatalog::plan`, `plan_precommitted`) and keeps k = 3 where it admits. Bump `WHITEOUT_PLANS_VERSION`. | A2 |
| J2 | jolt | `jolt-zk`: pad and auxiliary-slot schedule for the third masked level. Assert `n_× ≤ 2^10` in code, which D27 assumes. | J1 |
| J3 | jolt | `jolt-verifier`: the `ZK_CUT_LEVEL` soundness tests at both cuts, regenerated fixtures and fuzz bundles, and the clear, ZK and fixture acceptance suites. | J1, J2 |
| J4 | jolt | P1's main pass rerun on k = 3; D5 rewritten; D27 and δ_LHL recomputed with the slice count from A1. | J3 |

**Shipping.** On `whiteout/dev`, this is one protocol ticket set after P1
#7, separate from the engineering batch (#2–#5, #8, #9), which it compounds
with. Upstream, it folds into A9, A11, J16 and J17 if accepted before
those PRs open. Otherwise it is a follow-up after A13/J16, which are
already gated on author sign-off (R12).

**Acceptance.**

- A row at each workload passes `admit` at cut 2.
- Clear proof bytes are unchanged (C0).
- k = 2 and k = 3 proofs each verify, and fail under the other cut.
- The measured `|s|` lies within ±10% of the derived values above.
- The tail falls by at least 50% of the derived saving at fib 2^20.

## 4. Ambiguity register and what is unverified

- **A1.** The implementation's hiding-slice count at k = 2 and k = 3 is
  not tabulated; the totals above bracket 14–23. A1 counts it.
- **A2.** Akita's per-fold knowledge error at L2 is not restated here. It
  equals the clear Akita fold-3 term, but nobody has summed it into the
  Whiteout ledger.
- **A3.** The workload mapping uses singleton keys matched by k = 2
  residual size, not Jolt's keys with precommitted groups. ν for fib 2^16
  (13.6 M against 16.8 M) has margin. fib 2^20 at 28-variable shapes would
  reach ν = 24 and would need #7.
- **A4.** The planner never prices the Σ and response cost, at either cut.
  A residual-aware k = 2 search might recover part of muldiv's gain without
  a protocol change. That is untested.
- **Unverified.** No prover ran at cut 2. Time savings are P1's linear
  model; only the shapes are planner output. The prefix cost of the extra
  masked fold is P1's estimate, not a measurement. The planner fix was
  checked for admission only, not for optimality or catalog stability.
  The challenge family of the masked L2 at d = 64 is the config's d = 64
  family on the ℓ∞ route; its η² was not printed.

**For the authors.**

- Confirm the δ_LHL slice accounting at k = 3.
- Confirm that the implementation's per-row choice of k (per D5) needs
  no change to Theorem `wo:thm:zk` beyond binding k in the statement.
- Say whether D27's ≈ 2^−119.7 response ledger and the added fold-3 term
  are acceptable for a 128-bit headline.
