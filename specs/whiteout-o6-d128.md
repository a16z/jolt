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
  - The priced zero-knowledge search never offloads into the cut level
    (`CutOffload::Direct`; § Resolved below).
  - The three ZK rows are regenerated, and so is the response-digits key
    for the single-group statement (ν 24 → 23).
  - Tests:
    - `checked_in_zk_rows_commit_the_residual_at_its_sigma_binding_ranks`
      pins d = 128 and that every row uses the exception.
    - `the_priced_search_never_offloads_into_the_cut_level` replaces the
      37-variable offload test.
    - `every_checked_in_zk_row_fits_the_response_bound` checks ν ≤ 27 for
      every checked-in ZK row.
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
| `[B_a \| B′_a]`, 36:1 before the cut fix | 128 | 10 × 11,841 | 1,280 | 138.07 (β 521) | a bound for every rank-10 cell: the widest final cell is 10 × 10,881, and narrower cells are harder |
| `A_p`, rank-9 bracket edge | 128 | 9 × 1,231 | 1,152 | 128.26 (β 484) | the Σ-binding table's certified cell, the largest width at rank 9 |
| `A_p`, muldiv fixture | 128 | 9 × 449 | 1,152 | 132.24 (β 499) | |
| before: `[B_a \| B′_a]`, 36:1 | 64 | 19 × 13,378 | 1,216 | 130.91 (β 494) | for comparison |

`catalog_security --check` passes on every family (all occurrences at
128.26 bits or more by the proven-pruned profile), the residual
occurrences included. `A_p`'s rank comes from the same Σ-binding table
(`sigma_matrix_rank`), whose d = 128 cells were certified at 128 bits when
the table was generated.

**Compression chains.** u_a is `10 × 128 = 1,280` coefficients, and t_p
`9 × 128 = 1,152` or `10 × 128 = 1,280`. Each fits in one chunk of at
most 15,115 coefficients (D34), whose first map is 4,608 or 5,120 of the
`R_32` cell's 60,460 columns.

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

The masked prefix's own error ε_{≤k} does not move either. The only
input O6 could change is the cut level's Stage 2 point, whose variable
count follows the successor's padded witness domain
`witness_commitment_domain_len(L, d) = next_pow2(⌈L/d⌉)·d`. For a power of
two d this equals `max(d, next_pow2(L))`. So for any live length
`L ≥ 128` the domain, and with it the Stage-2 round count, is the same at
d = 64 and d = 128. The cut levels' live lengths are 486,080, 2,976,576
and 36,857,344 on Akita's rows, and every witness here is far above 128
coefficients. No row gains a round, and no `deg/q` term enters the budget.

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
6ca22bd57; O6 is this branch. Release, `profiling,akita,zk`, run through
`scripts/whiteout_verify_profile.py`'s harness (`--backend optimized`).
Base and O6 ran interleaved, the first build alternating by round, 7 runs.
The machine was shared. In the gated pass below, each binary waited (up to
30 min) for a 1-minute load below 6 and then ran under `/usr/bin/time -l`.
The harness's launch loads were 4.9–41.5, with a median of 6.0–6.8 per
cell. Each cell is median [min–max].

| | muldiv base → O6 | fib 2^16 base → O6 | fib 2^20 base → O6 |
|---|---|---|---|
| proof bytes | 93,364 [93,329–93,374] → 93,344 [93,314–93,369] | 95,573 [95,544–95,622] → 95,572 [95,541–95,622] | 97,906 [97,870–97,995] → 97,916 [97,894–97,944] |
| steady verify, 16 threads (ms) | 81.1 [78.9–83.7] → 78.2 [74.6–80.1] | 87.7 [85.0–208.0] → 84.0 [81.6–407.1] | 104.9 [103.8–112.6] → 95.3 [92.7–225.9] |
| steady verify, 1 thread (ms) | 144.8 [143.6–149.3] → 122.4 [121.9–124.0] | 180.6 [178.7–250.2] → 146.9 [145.1–198.1] | 274.5 [273.2–310.3] → 206.1 [205.7–288.7] |
| first verify, transported (ms) | 122.9 [118.2–129.7] → 99.4 [96.3–101.5] | 137.2 [134.8–237.6] → 112.1 [108.8–714.9] | 168.4 [165.5–186.2] → 133.5 [127.2–301.6] |
| residual products `A_a + [B_a\|B′_a]`, 1 thread (ms) | 40.9 [40.8–42.4] → 23.4 [23.2–23.6] | 67.4 [66.6–102.1] → 38.0 [37.8–57.0] | 144.9 [144.3–173.1] → 82.9 [82.5–119.8] |
| their share of the 1-thread verify | 28.3% → 19.1% | 37.6% → 26.1% | 52.8% → 40.2% |
| `N·z` (residual + pad), 1 thread (ms) | 51.3 → 30.4 | 77.9 → 45.3 | 155.7 → 90.1 |
| prover tail `whiteout_prove_tail` (ms) | 1,027 [1,017–1,234] → 992 [988–999] | 1,430 [1,402–4,303] → 1,382 [1,368–7,862] | 2,205 [2,187–2,519] → 2,101 [2,078–6,143] |
| — response proof (ms) | 733 → 705 | 1,008 → 978 | 1,573 → 1,507 |
| ν (`whiteout_plan_response`) | 24 → 24 | 25 → 25 | 26 → 26 |

The base column reproduces P0b's idle medians to within 2–3%: muldiv
80.1 / 143.0 / 119.1 ms and fib 2^20 105.0 / 275.1 / 167.0 ms (16 threads /
1 thread / first verify).

**Reading.**

- *Bytes.* Unchanged within the response proof's run-to-run spread
  (about 100 B). u_a and t_p keep 2 chunks each, and ν is unchanged.
- *Residual products.* 0.56–0.57× on every workload. Modeled was about
  0.54× (`n` 1,216 → 1,280 at about half the input ring columns).
- *1-thread steady verify.* −15% (muldiv), −19% (fib 2^16), −25%
  (fib 2^20).
- *16-thread steady verify.* −4%, −4%, −9%. The residual was 11–19% of the
  16-thread wall time (P0b), and the rest of that verify (the lifted rows,
  the pad-layout replay, the row batch) is unchanged.
- *First verify.* −19% to −21%: the residual and pad matrices built on
  first use are half the size.
- *Prover tail.* −3% to −5%, mostly the response proof.

A first, ungated 7-run pass at loads of 4–115 gave the same direction on
one thread (residual products 0.60–0.65×, 1-thread verify −10% to −20% by
median). Its 16-thread, first-verify and prover medians tracked load, not
the build, and are not used.

## Resolved: the cut level is never offloaded (review N3)

**Finding.** A one-off prove of Akita's 36:1 headline ZK row under the
first O6 rows failed with `ResponseTooLarge { num_vars: 29 }`. The
`R_128` residual had made the planner offload the root's setup into the
cut level from 35 variables on, which grew w_k from 8.4M to 36.86M
coefficients. w_k is the residual commitment's input, so it is part of the
Σ vector `s` and of the response vector `e`, and `e` needed ν = 29 against
D27's enforced bound of 27. The prover failed closed, so nothing unsound
was accepted, but the row could not be proved. The pre-O6 planner had the
same exposure from 37 variables. The cause is that the planner's
objective (setup first) does not price `s` or ν.

**Decision (coordinator, option 1).** The priced zero-knowledge search
keeps the edge into the cut level direct. `CutOffload::Priced` became
`CutOffload::Direct`, and the offloaded successor into `cut_level` is
dropped. `CutOffload::Forced` (test support) still plans the masked
Stage 3 at the cut, and admission still accepts such a row. D27 is
unchanged and needs no new delta.

**Why not reject at plan time.** ν depends on the statement: the pad
layout counts the masked claims, the prefix's messages and the auxiliary
block. The planner sees only the schedule. So the bound is checked by a
permanent test over the checked-in rows instead.
`every_checked_in_zk_row_fits_the_response_bound` builds each row's Σ
setup from public data (zero commitments of the row's profile shapes,
since the counting replay reads only shapes) and runs jolt-zk's own
`SigmaSetup::response_layout` and the response prover's `num_vars`. It
fails on the offloading 36:1 row (`ResponseTooLarge { num_vars: 29 }`)
and passes on the final rows. Its statement uses the Akita harness's pad
and `V` ranks (2 and 1), not Jolt's. Jolt's own rows are checked by
`whiteout_e2e`, which proves them.

**Final 36:1 row.** The cut level is direct with dims (128, 64, 64), w_k
is 8.40M coefficients, and the residual is `A_a` 10 × 2,048 and
`[B_a | B′_a]` 10 × 10,881 over `R_128`. Only this row changed, in both
artifact sets.

**Manual re-prove of the final 36:1 row.** A one-off run of Akita's
end-to-end harness, not committed: fresh randomness, no seed, release
build, 1-minute load about 15. Prove and verify pass, and a flipped proof
byte is rejected.

| | value |
|---|---|
| setup + commit | 57.2 s |
| prove (masked prefix, Σ, response proof) | 5.6 s |
| proof (Akita side) | 79,540 B |
| verify | 74.8 ms |
| wall, whole run | 63.6 s |
| peak RSS | 10.1 GiB (`/usr/bin/time -l`) |

The exploratory runs behind the decision ran at loads above 40: the
direct-cut variant proved in 26.2 s with a cold verify of 0.34 s (10.4 GiB),
and base 8c22267ed in 30.3 s with 1.40 s (11.1 GiB). They are not
comparable with this table, only with each other.

## d = 256 outlook

- *The gate.* P0b's rule starts d = 256 only if fib 2^20 still spends at
  least 50% of a 1-thread verify in the residual products. After O6 it is
  40.2% in the gated pass (52.8% for base), so the rule does not start
  d = 256.
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
- *Side effect, closed.* A cheaper residual would make the setup-first
  planner offload into the cut level on more rows. The search no longer
  does that (§ Resolved), and the response-bound test guards every
  checked-in row.

## Not verified

- **Idle machine.** The gated A/B ran at load < 6 for most runs, not at
  P0b's < 2. A few runs launched at loads up to 41 and show as range
  outliers. Base matches P0b's idle medians to within 2–3%.
- **Peak RSS.** It was recorded (`/usr/bin/time -l`, in the scratch run
  directory) but not tabulated.
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
