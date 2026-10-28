# Whiteout residual commitment over `R_128` (O6, step 1)

Parent: `specs/whiteout-optimization-campaign.md` (O6; "Decisions (owner,
2026-10-25)": d = 128 first). Akita side: `akita:specs/whiteout-akita.md`
§5.5.3. Paper delta: D35.

## Decision

The residual commitment (`A_a`, `[B_a | B′_a]`, r_a) moves from `R_64` to
`R_128`, and the pad commitment `A_p` with it, since the two share the Σ
setup's ring. Akita's zero-knowledge planner gives the level after the cut
the ring dimension `akita_params::zk::RESIDUAL_RING_DIMENSION = 128`,
exempt from the monotone dimension ceiling. Clear catalogs, the transcript
schedule and every message site are unchanged. The security arguments hold
with the numbers below; nothing in the D27 ledger gets worse.

## What changed

- **Akita** (`wo/akita-o6`).
  - `opening_work_domain`: at the `AfterCut` level the dimension domain is
    `{uniform(128)}`, with no ceiling.
  - The three ZK rows are regenerated, and so is the response-digits key
    for the single-group statement (ν 24 → 23).
  - Tests: `checked_in_zk_rows_commit_the_residual_at_its_sigma_binding_ranks`
    pins d = 128 and that every row uses the exception, and
    `the_cut_level_offloads_the_root_setup_from_35_variables` replaces the
    37-variable test.
- **jolt** (`wo/o6-d128`). No production code changes: `jolt-akita` already
  dispatches the Σ setup on the residual's ring dimension (`with_ring_dim!`
  over 64, 128, 256), and the pad commitment's rank is priced at that ring
  (`sigma_matrix_rank`). Two jolt-akita tests now pin `A_p` at d = 128.
- **Jolt's own rows.** The trace-group row (muldiv shape, 16 variables,
  56 polynomials) has cut-level dims (256, 64, 64) and a residual at
  (128, 128, 128). It uses the ceiling exception.

## (a) Why lifting the ceiling at the after-cut level is correct

The ceiling (`dimension_candidates`, `SuffixState::dimension_ceiling`) only
limits the planner's search. Nothing in the protocol depends on it.

1. **No check reads it.** `RingDimensionSchedule` reaches only the planner
   and the policy digest. Row admission, `FoldSchedule::validate_structure`,
   `ZkOpeningSchedule::admit`, the prover and the verifier never compare a
   level's ring dimensions with its predecessor's.
2. **The interface between levels does not depend on the dimension.** A
   level hands its successor a flat coefficient vector. The predecessor's
   Stage 2 geometry takes the successor's ring dimension as a parameter
   (`validate_stage2_successor_capacity`,
   `RelationAddressGeometry::new_with_coefficient_block`), and
   `witness_commitment_domain_len` zero-pads to whole `R_{d_succ}` elements.
   No step assumes `d_succ ≤ d_pred`.
3. **The after-cut level is never folded in zero knowledge.** Both roles
   stop after the cut level's Stage 2 (`finish_through(cut_level)`, Akita
   §4.8.1). The level's parameters are the residual commitment and nothing
   else. Its D matrix, its fold challenges and the levels after it are
   planned but never run, and they keep the monotone ceiling (128 → 64).
4. **The Σ-protocol works over `R_{64m}`** (`whiteout-sigma.md` §0; paper
   `02_preliminaries.tex:69-77`). Restricting scalars along `R_64 ↪ R_128`,
   `X ↦ Y²`, preserves coefficient norms. It also turns `c·v` into
   `c(Y²)·v`, which has the same ℓ1 norm. So
   `‖(c − c′)v‖_∞ ≤ 2κ₁‖v‖_∞` still holds. `Γ_Δ = 106` and
   `R_Σ = 8κ₁·B(10) = 248,636,229,427,520 ≈ 2^47.82` do not change, and the
   binding reduction (`05_security.tex:31-46`) applies to the `R_128`
   matrices as written. The paper itself sizes these matrices over larger
   rings (`06_evaluation.tex:124-133`).
5. **Hardness is priced at the row's own d.** The planner and `admit` use
   the Σ-binding table at d = 128. `sis_occurrences` lists both residual
   matrices at `R_Σ` for `catalog_security --check`.
6. **Clear catalogs cannot change.** The rule fires only at `AfterCut`,
   which only a zero-knowledge search reaches (`SuffixCtx::zk`). It reads
   no policy field. A full regeneration of both artifact sets changed only
   `fp128_dense_zk`, `fp128_onehot_recursive_zk` and the response-digits
   key (§ Evidence).

**Exercised.** Every checked-in ZK row and Jolt's trace row has a cut-level
role ring below 128, so the end-to-end suites (Akita `zk_end_to_end`,
`masked_prefix`, `zero_knowledge`; jolt `whiteout_e2e`, the verifier's
tamper and census suites) prove and verify through the exception.

## (b) Security of the residual and pad commitments at d = 128

**Module-SIS at `R_Σ`.** Akita's ADPS16 quantum ℓ∞ estimator over q128
(`sis_estimator_once --profile exhaustive-serial`; target 128 bits):

| matrix | d | rank × width | n = rank·d | bits | note |
|---|---|---|---|---|---|
| `A_a`, dense 14:1 | 128 | 9 × 1,024 | 1,152 | 128.79 (β 486) | the only rank-9 residual |
| `[B_a \| B′_a]`, 36:1 (widest) | 128 | 10 × 11,841 | 1,280 | 138.07 (β 521) | every other rank-10 residual cell is narrower, so harder |
| `A_p`, rank-9 bracket edge | 128 | 9 × 1,231 | 1,152 | 128.26 (β 484) | the Σ-binding table's certified cell, the largest width at rank 9 |
| `A_p`, muldiv fixture | 128 | 9 × 449 | 1,152 | 132.24 (β 499) | |
| before: `[B_a \| B′_a]`, 36:1 | 64 | 19 × 13,378 | 1,216 | 130.91 (β 494) | for comparison |

`catalog_security --check` passes on every family (all occurrences at
128.26 bits or more by the proven-pruned profile), the residual
occurrences included. `A_p`'s rank comes from the same Σ-binding table
(`sigma_matrix_rank`), whose d = 128 cells were certified at 128 bits when
the table was generated.

**Compression chains.** u_a is `10 × 128 = 1,280` coefficients, and t_p
`9 × 128 = 1,152` or `10 × 128 = 1,280`. Each still fits in 2 balanced
chunks of at most 884 coefficients (O5). The first map of a 640-coefficient
chunk has 5,120 columns of the `R_16` cell's 7,077, which gives **139.13
bits** (β 525, exhaustive). The 608-coefficient chunk was 140.98 bits, and
the cell's 884-coefficient bound is 128.26.

**Σ binding and extraction.** Thm `wo:thm:knowledge` is unchanged. The
radius, the challenge family (`|𝓒| = 2^128.06`, `κ₁ = 53`), the label count
ℓ (labels number operands and relation rows, not N's rows) and the
compressed-binding step for `cm(u_a)` (D29/O4) are all independent of the
residual's ring. Binding forces extracted values to agree through the same
three matrices, now Module-SIS hard over `R_128` at `R_Σ` (above).

**Hiding and statistical zero knowledge.** r_a is sized by the same rule:
`hiding_randomness_columns` gives entropy at least the image bits plus 264,
so 321 `R_128` columns for the 1,280-coefficient image (610 `R_64` before).
r_p follows from the `A_p` fixed point (289 columns at rank 9, 321 at
rank 10). Lemma `wo:lem:stat-hiding` and Thm `wo:thm:zk` keep their
hypotheses: the exact `‖cs‖_σ ≤ 1/α` check, a fresh mask per attempt, a
secret nonce, and the entropy margin. The terms of the statistical total
(δ_LHL over the same 14 slices, `T_att·ε/M`, the PRG and sampler terms)
depend on d only through the sample count. That count changes by the few
percent `|s|` moves, so the total stays 2^−129.1 at the precision O7 §7.4
states it.

## (c) Soundness budget

D27's ledger depends only on ν, ⌈log₂ m⌉ and ⌈log₂ n_×⌉. n_× counts
product pairs of the relation, which O6 does not touch. ν is measured
below. It does not grow on any workload: Akita's single-group statement
goes from 24 to 23, and muldiv, fib 2^16 and fib 2^20 keep theirs. The
ledger stays **310/q ≈ 2^−119.7** at the headline bound (ν ≤ 27, enforced
by `MAX_RESPONSE_NUM_VARS`).

The masked prefix's own error ε_{≤k} can move only through the cut level's
Stage 2 point. Its variable count follows the successor's padded witness
domain. Padding to whole `R_128` elements instead of `R_64` adds at most 64
coefficients, so it adds a Stage-2 round only where the padded length
crosses a power of two, at an extra `deg/q` for that sumcheck's degree. Rows
differ anyway, because the planner re-selects them. Akita prices each
level's error per row, and none of it is in D27's ledger. Whether any
measured row gained a round was not checked.

## Evidence

- **C0, regeneration.** Both artifact sets were regenerated in full
  (`gen_schedule_artifacts`, 26 families each). Only the files and catalog
  rows named above differ.
- **Dump identity** (Akita bar, `AKITA_PROOF_DUMP`): see § Results.
- **Bars:** see § Results.

## Results

**Heads.** jolt `wo/o6-d128`: `[pin]` 979f83d3e, then d6dabc62c (tests),
then this note and D35. Akita `wo/akita-o6`: `[pin]` a95033eab, 2472555de
(planner, rows, tests), fe5853d0f (§5.5.3).

**Bars, all green.**

- *Akita.* clippy on both default lanes, the zk, zk-prover and A46 lanes,
  and 8 planner feature sets. fmt and taplo. CI nextest 1919/1919. The
  no-disk-cache zk lanes: 188, 225 and 37 (`zero_knowledge`,
  `zk_end_to_end`, `masked_prefix`). akita-cpu-backend zk-prover 549.
  akita-planner 154/154 with and without `parallel`.
  `catalog_security --check` passes on every family.
- *Dump identity.* The CI nextest's 113 proof hashes equal post-M1 dev's
  exactly: none missing, none new. Against the W5d base they are the 5 known
  ZK-row omissions and nothing new. No clear proof moved, and no ZK proof in
  that lane moved either. The changed ZK rows are proved only in the zk
  lanes, which the dump does not cover.
- *jolt.* All 16 clippy lanes, among them `akita,zk,prover-fixtures` and
  `akita,zk,profiling,prover-fixtures`. fmt (after one reformat of the test
  commit) and `check_style_invariants --base whiteout/dev`. The profiling
  smoke test. `fs_obligations` (no bless). `whiteout_e2e` 6/6, and 1/1 with
  field-inline. `jolt-akita` 89/89. jolt-verifier
  `akita,zk,prover-fixtures,logging --test-threads 1` 88/88. The census and
  `fs_attacks` with field-inline 11/11. Akita clear verifier 79/79. The
  zk-prover whiteout tests 5/5. jolt-zk and its siblings 345/345.

**Verifier and prover, A/B.** Base is whiteout/dev 48b4db315 with Akita
6ca22bd57; O6 is this branch. Release, `profiling,akita,zk`,
`scripts/whiteout_verify_profile.py`'s harness (`--backend optimized`),
base and O6 interleaved with the first build alternating, 7 runs. The
machine was shared and heavily loaded: 1-minute loads of 4.0–114.6 at
launch, against the < 2 of P0b. Each cell is median [min–max].

| | muldiv base → O6 | fib 2^16 base → O6 | fib 2^20 base → O6 |
|---|---|---|---|
| proof bytes | 93,353 [93,340–93,391] → 93,375 [93,339–93,418] | 95,611 [95,546–95,643] → 95,569 [95,546–95,630] | 97,909 [97,885–97,961] → 97,929 [97,865–97,961] |
| residual products `A_a + [B_a\|B′_a]`, 1 thread (ms) | 46.2 [41.3–55.1] → 30.2 [23.3–205.1] | 76.2 [68.2–184.7] → 46.2 [39.2–285.8] | 167.1 [151.0–179.0] → 99.7 [85.3–105.6] |
| their share of the 1-thread verify | 28.5% → 19.1% | 38.9% → 25.6% | 52.8% → 39.8% |
| `N·z`, 1 thread (ms) | 58.1 → 38.8 | 87.8 → 54.9 | 179.7 → 108.2 |
| steady verify, 1 thread (ms) | 169.7 [144.8–192.1] → 151.8 [121.6–890.2] | 205.1 [183.7–339.4] → 180.0 [150.9–1,118.3] | 314.9 [282.8–344.7] → 253.1 [211.7–270.2] |
| steady verify, 16 threads (ms) | 232.5 [79.4–494.9] → 227.8 [75.5–689.8] | 229.4 [85.8–619.8] → 333.9 [81.0–526.6] | 220.3 [104.0–524.2] → 376.8 [98.0–565.7] |
| first verify, transported (ms) | 410.0 → 371.9 | 307.8 → 363.0 | 332.7 → 419.2 |
| prover tail `whiteout_prove_tail` (ms) | 4,359 → 4,429 | 4,954 → 6,365 | 5,135 → 9,790 |
| ν (`whiteout_plan_response`) | 24 → 24 | 25 → 25 | 26 → 26 |

The 16-thread, first-verify and prover medians track load, not the build:
their ranges span 5–10×, and the same build's runs differ more than the
two builds do. The one round in which all six runs started at load ≤ 6
(round 4) gives:

| round 4 (loads 4.0–5.8) | muldiv base → O6 | fib 2^16 base → O6 | fib 2^20 base → O6 |
|---|---|---|---|
| steady verify, 16 threads (ms) | 82.0 → 75.5 | 85.8 → 81.0 | 105.5 → 98.0 |
| steady verify, 1 thread (ms) | 147.2 → 125.1 | 191.2 → 150.9 | 286.7 → 211.7 |
| first verify (ms) | 128.7 → 98.6 | 151.2 → 113.6 | 170.2 → 139.2 |
| prover tail (ms) | 1,118 → 1,007 | 1,553 → 1,449 | 2,342 → 2,415 |

The base's round-4 numbers match P0b's idle medians (80.1 / 143.0 / 119.1
ms at muldiv, 105.0 / 275.1 / 167.0 at fib 2^20), so that round is close to
idle. One round is not a distribution.

**Reading.**

- *Bytes.* Unchanged within the response proof's run-to-run spread
  (about 100 B). u_a and t_p keep 2 chunks each, and ν is unchanged.
- *Residual products.* 0.60–0.65× on one thread on every workload, against
  about 0.54× modeled (Akita §5.5.2: `n` 1,216 → 1,280, work ∝ `n × input /
  d`). These are CPU sums inside the verify, so they are the least
  load-sensitive measure here.
- *1-thread steady verify.* −10% (muldiv) to −20% (fib 2^20) by median,
  and −15% to −26% in round 4.
- *16-thread verify, first verify, prover tail.* No conclusion from the
  medians. Round 4 shows −5% to −7% (16 threads), −18% to −25% (first
  verify), and −10% to +3% (tail).

## d = 256 outlook

- *The gate.* P0b's rule starts d = 256 only if fib 2^20 still spends at
  least 50% of a 1-thread verify in the residual products. After O6 it is
  39.8% by median (40.3% in round 4, against 53.6% for base in that
  round), so the rule does not start d = 256. A clean idle rerun should
  confirm this before the decision is closed (§ Not verified).
- *What d = 256 would cost.* Blocker (3): the fp128 presets search B only
  over {64, 128}. That is not a problem in itself here, since the after-cut
  domain is a separate constant (`RESIDUAL_RING_DIMENSION`), and `256` is
  supported by the Σ-binding table (rank 5), Akita's main ℓ∞ table, the
  fold-challenge configs and jolt's `with_ring_dim!`. The constant change
  is one line plus regenerated ZK rows. The security work is the same as
  here: the estimator at rank 5 × the residual widths over `R_256`, and
  `A_p` at rank 5.
- *What it would buy.* The products fall by about another 2× (n 1,280 at
  both d), so fib 2^20 would go from ≈ 100 ms to ≈ 50 ms of a ≈ 250 ms
  1-thread verify. The 16-thread verify is spread over the lifted rows, the
  pad-layout replay and the row batch (P0b), which O3b is working on, so
  d = 256 matters there only after those.
- *Side effect to watch.* Every cheaper residual makes the planner offload
  the root's setup at the cut on more rows. On Akita's 36:1 headline that
  already happened at d = 128, and w_k grew 3.4×. Jolt's rows stay direct
  today.

## Not verified

- **Idle A/B.** The 7-run A/B ran at loads of 4–115, so its 16-thread,
  first-verify and prover-tail medians are not attributable to O6. A second,
  load-gated pass (each run waits up to 30 min for load < 6) was started;
  see its addendum below if present.
- **Prover tail at d = 128.** Round 4 shows +3% at fib 2^20 (one sample).
  Not resolved.
- **Stage-2 rounds at the cut.** Whether any row gained a round
  (§ (c)) was not checked row by row.
- **The certified Σ-binding cells.** Only the cells listed in (b) were
  re-estimated exhaustively. The rest of the d = 128 cells are trusted from
  the checked-in certified CSV (`cells_match_the_certified_csv_and_the_current_radius`).
- **Akita dev-protocol test suite.** Not run. The dev artifact set was
  regenerated, and the dev catalog snapshot changes only in the ZK rows and
  the response key.
- **Jolt's other row shapes** (precommitted advice and the committed
  program). They are covered end to end by `whiteout_e2e` and the
  verifier suites, but their cut-level and residual dimensions were not
  printed.
