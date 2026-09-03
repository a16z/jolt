# W6-RT — verifier relation row table

Date: 2026-09-03. Branch: `wrap/spartan-hyperkzg`.
Code commit: `f845ec8ad`.

## Decisions

- Three-wire lowering: each sparse linear form becomes `acc + coefficient*value - next = 0`;
  each R1CS row ends with `A(z) * B(z) - C(z) = 0`. Constant-one terms fold into `q_C`.
- Columns: 9 VK (`q_L,q_R,q_O,q_M,q_C,sigma_a,sigma_b,sigma_c,active`) and 5 prover
  (`a,b,c,h_id,h_sigma`). `id_w(x) = w * 2^18 + x` stays virtual.
- Internal copy check: grouped size 3. The two well-formedness terms have degree 5 after
  `eq(tau,x)`; the unweighted helper-sum term has degree 1; the gate term has degree 4.
- One combined `RelationTableProver` member, with transcript-random relation weights, avoids four
  copies of the 14-column `2^18` matrix. It enters `prove_kzg_batch_stage` as one degree-5 member;
  this changes neither stage-A round count nor round size.
- Sound phase order differs from W5's current one-phase `wrap`: commit `a,b,c`; squeeze `(beta,
  gamma)`; build and commit `h_id,h_sigma`; then draw `tau` and prove stage A. Helper columns cannot
  be committed before `(beta,gamma)`, and drawing them before wire commitment lets the prover choose
  the permutation witness after the challenge. Standalone protocol implements the two phases.
- The existing seven “known” public entries are proof/program-dependent outsourced evaluations,
  not profile constants (`relation/public_io.rs`). They cannot be baked into a profile VK. W5 must
  bind them with the same `CopyLink` family to a verifier-evaluated public side.

## CopyLink interface

- `CopyLinkSide { selectors: [Vec<Fr>; 3], ids: [Vec<Fr>; 3] }`: VK columns; equal logical IDs on
  both sides identify an edge.
- `CopyLink::witness(left_values, right_values, beta, gamma)`: two grouped inverse-helper columns.
- `CopyLink::prover(..., tau, beta, gamma, weights)`: degree-5 `ProveRounds` member.
- `CopyLink::final_value(...)`: consumes selector/id/value/helper evaluations at `r_A`; committed
  or virtual values are caller-owned. T1 supplies decoded squeeze/message values; public inputs
  use verifier-evaluated values; R supplies `a,b,c` cells. T2 scalars use the weighted link below.

## W5/T2 R-slot contract (published 00:16)

- Cell layout, wire `a`: gate rows `[0, 38_981)`; absorbed-word anchors
  `[38_981, 40_203)` (1,222 schedule `Fr` entries); squeeze anchors `[40_203, 40_579)` (376);
  inactive alignment `[40_579, 40_704)`; Dory scalar block `[40_704, 40_960)` with scalar `s` at
  `40_704 + s`, 175 live cells then 81 zero-constrained padding cells. The 256-cell scalar block
  is dyadic-aligned. Every anchor joins the source R1CS variable's internal sigma cycle.
- VK groups at k=16: relation fixed group = 9 slots (`q_*`, three sigma, active), 7 spare. A
  `CopyLink` side contributes three selector and three logical-ID columns; the caller assigns
  those into VK groups and supplies their stage-B indices.
- Prover phase 1: one group containing `a,b,c` (3 live slots). Draw `(beta,gamma)`. Prover phase 2:
  one group containing `h_id,h_sigma` plus CopyLink helpers (2 relation slots before links). Draw
  `tau` and member batching weights only after phase 2.
- Members, all offset 0 over 18 rounds: `RelationTableProver` degree 5, input 0; each
  `CopyLinkProver` degree 5, input 0; `DoryScalarLinkProver` degree **2**, input
  `sum_s rho^s scalar_s`. Final evaluators take `r_A`, transcript challenges, and stage-B claims
  in the constant index order from `relation_table::FIXED_COLUMNS..TOTAL_COLUMNS`.
- T2 scalar link: W5 compares the R member input with T2's
  `sum_{s,i} rho^s 16^i digit_{s,i}` input. R final claim is
  `weight_scalar_block(r_A) * a(r_A)`; the weight evaluator costs 34 Fr multiplications. It reuses
  R's `a(r_A)` stage-B claim and adds no columns or rounds.
- Written reason for degree 2 versus the requested degree 1: both `weight(x)` and `a(x)` are
  multilinear, so their product has univariate degree 2 in every sumcheck round. Treating their
  pointwise product as one degree-1 polynomial would leave its random-point value unbound to
  `weight(r_A) * a(r_A)`. Stage A is already degree 5, so bytes and rounds do not change.
- Verifier final-evaluator budget, observer-counted: relation gate/internal-copy 103 Fr mul;
  one CopyLink 29; Dory scalar link 34; **166 total**, below the 5,000 cap. Each extra CopyLink adds
  29. No term scans R rows.

## Measurements

Real `/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin`, 10 Rayon threads. Load
average 7.39 before the measured run and 13.06 after it; timings are contended.

- R1CS: 5,254 rows, 6,761 variables, 45 former public entries. Lowering: **38,981 gate rows** in
  the common `2^18` domain; anchors/alignment extend the used R region to **40,960 rows**.
- Columns: 9 VK + 3 wire + 2 inverse helper = **14**. Phase-separated k=16 layout: one fixed VK
  group (9/16 slots), one wire group (3/16), one helper group (2/16); 3 groups enter the final RLC.
- Degrees: gate 4; identity-copy well-formedness 5; sigma-copy well-formedness 5; unweighted helper
  sum 1. `CopyLink` has two degree-5 well-formedness terms and one degree-1 sum term.
- Isolated online additions: wire commit **120 ms**, helper commit **105 ms**, row member including
  matrix construction **245 ms**, scalar-link member **35 ms**; total **505 ms**. Fixed VK commit
  **259 ms** once. The standalone protocol's full prove was **6.404 s**, including stage B and the
  existing-size `2^22` HyperKZG opening. Deterministic SRS setup **43.937 s**, excluded. Scalar-link
  load average was 5.29 after the run; the earlier phase run was 7.39 → 13.06.
- Standalone proof: **4,896 B payload / 4,959 B bincode**: 64 B dynamic phase commitments + 1,824 B
  stage-A committed rounds + 384 B six-round stage B + 448 B column claims + 32 B reduced claim +
  2,144 B HyperKZG (`ell=22`).
- Executed verifier: **87 ecMul, 86 ecAdd, 8 pairing pairs, 7,364 Fr mul, 58 Fr inversions, 326
  Keccak**, **7 ms**. N4 model: **1,390,237 gas** (6,368 EVM calldata bytes after G1 expansion).
  The exact counter assertion is independent of the 38,981 used rows; no verifier row scan.

## Full-stream byte delta

W5 sound fallback: 10,496 B. Removing outer 1,248 + inner 832 + SPARK 1,536 + Az/Bz/Cz 96 +
W/SPARK prior claims 448 + carries 448 + challenge words 608 removes 5,216 B. The no-R core is
5,184 B. Hosting this table adds 64 B phase commitments, 64 B for the extra stage-B group bit, and
448 B column claims; stage-A round wire and `ell=22` opening stay unchanged. **Projected 5,760 B
before T1/T2/public `CopyLink` claim values** (−4,736 B versus fallback).

## Open integration condition

The standalone R proof and real-shape `CopyLink` pass. A sound full-stream byte total is not yet
available: W5's committed `wrap` absorbs one combined commitment phase, while permutation helpers
require the two-phase order above. Also, the projected 5,760 B leaves 240 B for all T1/T2/public
link claims; the current generic `CopyLink` final check exposes more than that unless W5 shares or
folds existing factor claims. Reporting 5.3–5.7 KB as sound before those two items land would omit
proof data.
