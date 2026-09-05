# Akita Metal: D128 rank-3 root commit, floor and falsification bar

Outcome, 2026-09-05: C2 passes the production gates and the integrated
three-workload matrix at 10.498382 MHz projected average. See
[acceptance evidence](akita-metal-c2-acceptance-2026-09-05.md). Historical
analysis and the explicitly documented accounting corrections follow;
the acceptance note records the actual catalog-transition behavior.

Status: analysis before code, 2026-09-03. Companion to
`akita-metal-10mhz-attack-strategy.md` section 9 and the geometry re-query in
`benchmark-runs/akita-10mhz-studies/analysis.md`.

## Decision

Replace the T=2^28 one-hot root geometry D512 rank 1 (65,536 accumulator bits per hot
entry) with D128 rank 3 at 2^19 positions per block (49,152 bits), and write a packed
Metal commit kernel that tiles the public matrix per ring element instead of per full
row. Both commit terms then scale with accumulator volume, and the model predicts the
commit falling from 12.9 / 16.4 / 18.2 s to about 9.1 / 13.0 / 14.1 s on BTreeMap /
Fibonacci / SHA-2. The geometry is a public-parameter change under the same SIS policy;
the kernel is new. No code is written until the bar in section 5 is accepted.

## 1. Requirements ledger

- Fact: the planner admits D128 rank 3 at nv = 41 with bound 1,048,575, 4 fold digits,
  7 levels and the same 76,138-byte payload as production, at 2^19 or 2^20 positions per
  block (re-query at planner rev 0e52ebf17; `analysis.md`, "S1 correction"). D64 rank 6
  at 2^20 is admissible at the same volume. Nothing below 49,152 bits is admissible.
- Fact: production D512 rank 1 is forced by `crates/jolt-akita/src/schedules/mod.rs`
  (`regen_k256`: `METAL_ROOT_DIMS`, `inner_output_rank: Some(1)`, positions 2^16 to 2^18),
  not by the SIS tables.
- Fact: the Metal packed kernel exists only for D512
  (`akita-metal/src/packed_onehot_fp128_d512.rs`, `backend.rs:579-588`).
- Requirement: exact parity with the CPU commit (checksum of the reduced inner rows) and
  a verified proof at T=2^28 on all three workloads.
- Requirement: peak RSS at or below 90 GiB. Setup grows from 180.4 M to 270.5 M field
  elements (+1.4 GB) and the prepared matrix from 2 GiB to 3 GiB.
- Invariant: the honest fold policy, bucket set, response digits and level structure are
  unchanged, so the verifier change is a new schedule row and effective-schedule digest.
  Proofs from either catalog must fail closed against the other.
- Constraint: CPU and Metal catalogs regenerate together; the CPU commit path already
  handles any D through `DeferredFp128Ring` and gets a 25% smaller gather per hot entry.
- Assumption (unverified): the D128 kernel reaches the D512 kernel's per-coefficient
  efficiency. This is the whole bet; section 5 makes it falsifiable.
- Open: 2^19 vs 2^20 positions per block. 2^19 has the smaller setup; 2^20 has the
  smaller root output (852.6 vs 899.8 MB). Decide on measured eval-proof time.

## 2. Boundary

Inputs: the cycle-major lane table (1 byte per row per live column, 7.5 GiB at 30
columns), the public matrix A (positions per block x n_a rows x D coefficients x 16 B:
2^19 x 3 x 128 x 16 B = 3 GiB, prepared once), params. Mapping as today: local field
= row x 256 + lane, position = field / D, shift = field mod D. Corrected on
2026-09-05: D128 spans two positions per trace row (`2 * row + lane / 128`);
D512 packs two rows per position. Output: one 384-coefficient fp128 accumulator per
(column, block) task, reduced from 16 position partials, read back as the inner rows.
Tasks per column = T x 256 / (D x P) = 1,024 at 2^19 (512 today). Host keeps the
hybrid tail, reconstruction and merge.

## 3. Lower bound

Correction, 2026-09-05 (user approved): the historical stream estimates below
undercounted the matrix traffic by two. At T28, C30, P19 there are 1,024 tasks
per column. The two-task layout has 64 tasks per threadgroup, so it streams
`30 * 1024 / 64 * 3 GiB = 1440 GiB`, not 720 GiB. The one-task layout
streams 2880 GiB. These are logical traffic volumes, not measured DRAM
traffic; dividing them by a copy benchmark's bandwidth is a proxy, not a
rigorous runtime lower bound. The original estimates below are retained as
the provenance of the error, not as current predictions.

Streaming. Today each threadgroup streams the full 8 KiB row of every position because
a negacyclic rotation mixes all 512 input coefficients into each 256-coefficient output
band; that gives `S = C T K n_a^2 D 16 / 8192` = 2.06 TB. With rank 3 the rotation acts
within one 128-coefficient ring element, so a threadgroup working on element `i` streams
only `P x D x 16` bytes. A SIMD group holds 128 coefficients as 4 per lane (20 words per
thread against 40 today), so it can hold either one task (32 tasks per threadgroup) or
two (64 tasks):

| tasks per threadgroup | bytes streamed | at 575 GB/s | at 412 GB/s |
|---:|---:|---:|---:|
| 32 | 1.55 TB (0.75x) | 2.7 s | 3.75 s |
| 64 | 0.77 TB (0.375x) | 1.35 s | 1.9 s |

Accumulate. 3 x 128 = 384 coefficient adds per hot entry (0.75x). At the calibrated
131 G coefficient-adds/s: 2.93 ns per hot entry, 5.9 / 9.6 / 10.5 s. The gather per hot
entry is 3 x 2 KiB = 6 KiB from threadgroup memory; the 32 KiB tile holds 16 positions of
one element (4 today), so barriers per hot entry fall 4x. Element indexing adds three
address computations per hot entry instead of one; budget 5% on the slope.

Bottomed-out commit (serialized terms plus the measured 1.2 s of stores, reduction and
ballots): 32 tasks per group 9.8 / 13.5 / 14.4 s; 64 tasks per group 8.5 / 12.2 / 13.1 s.
The accumulate binds in all cases; the regime does not flip above 12% density.

## 4. Adjustment candidates

- Two tasks per SIMD group (64 per threadgroup). Halves streaming again. Cost: doubled
  register pressure per lane relative to the one-task layout, still half of today's.
  Correctness unchanged.
- D64 rank 6 at 2^20 positions. Same volume, 2 coefficients per lane, up to 4 tasks per
  SIMD group, 1 KiB rows. Rejected as the primary: six element indices per hot entry and
  64-wide gathers double the bookkeeping the slope budget allows; keep as fallback if
  the D128 lane mapping does not reach the bar.
- Sign handling of the negacyclic wrap within a 128-element (today's quadrant trick at
  `onehot.metal:4211-4249`) carries over unchanged with the modulus D.
- Nothing here changes the protocol beyond the schedule row. Any change to fold digits,
  bucket or policy is out of scope and would need its own soundness note.

## 5. Falsification bar (register before code)

Measured on the Akita `packed_onehot_commit` bench at T=2^28, 30 live columns, density
sweep 13 / 27 / 54%, one warm plus one sample, machine idle:

1. Slope at or below 3.2 ns per hot entry (0.75 x 4.26 = 3.2; anything above means the
   D128 kernel lost per-coefficient efficiency and the geometry gain is eaten).
2. Intercept at or below 2.9 s (one task per SIMD group) or 3.2 s (two tasks).
   The two-task ceiling was corrected from 1.6 s with explicit user approval
   on 2026-09-05 for the twofold traffic error documented in section 3.
   This is an accounting correction, not a fitted hardware optimum. The
   retained candidate's approximately 2.951 s intercept was already known;
   independent full-proof pairs and the workload matrix remain required.
   No other ceiling or acceptance condition changes.
3. Commit at T=2^28 at or below 9.8 / 13.5 / 14.4 s on the production workloads with
   exact checksum parity against the CPU commit and `PROOF_VERIFIED backend=metal`.
4. Peak RSS at or below 90 GiB on BTreeMap; the +1.4 GB setup and +1 GiB matrix must be
   paid for by retiring the D512 matrix and its 2 GiB partial plane.
5. CPU backend commit per hot entry at or below today's 19 ns with the same catalog.

Failing 1 or 3 closes the lever for this kernel design; failing 4 sends it back to the
residency plan, not to the kernel.

## 6. Verification and rollout

Parity: the existing `d512_k256_commit_matches_canonical_accumulate` oracle generalized
to D = 128, plus the CPU/Metal reduced-row checksum in the bench. Acceptance: the section
5 bar, then the three-workload order-reversed matrix used for the campaign, scored
against the fixed-CPU references. Rollback: the K256 catalog keeps the D512 row; the
proof self-describes its schedule, so a mismatched verifier fails closed. Observability:
the `commit_inner` span already records `ring_dimension`, `n_a`, `positions_per_block`.

## 7. Handoff slices

1. Catalog: add D128 to `METAL_ROOT_DIMS`, allow `inner_output_rank: Some(3)`, positions
   2^19 for nv = 41; regenerate; confirm the planner row and digests on both backends.
2. CPU-only end-to-end at T=2^25 and T=2^28 on the new row (the CPU path needs no code);
   this also measures the eval proof and RSS with the larger setup.
3. Bench harness: extend `packed_onehot_commit` to D128 rank 3 with the CPU checksum.
4. Kernel: per-element tile fill, one-task lane mapping first, then the two-task variant.
5. Runtime: task grid for 1,024 blocks per column, partial plane sizing, hybrid tail.

## Ambiguity register

- Whether the 131 G coefficient-adds/s calibration transfers to 128-wide gathers
  (shorter rows may change SIMD-group divergence on the wrap). Resolved by bar item 1.
- Whether the eval proof's root fold changes cost with 4 rows per position (root output
  899.8 vs 942.8 MB; the schedule's positions and levels are otherwise unchanged).
  Resolved by slice 2.
- Whether Akita upstream accepts a rank-3 Metal root or the kernel stays in the fork.
