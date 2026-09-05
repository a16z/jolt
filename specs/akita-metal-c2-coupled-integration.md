# C2 coupled-path integration checkpoint

Outcome, 2026-09-05: the final integrated matrix passed at 10.498382 MHz
projected average. See [acceptance evidence](akita-metal-c2-acceptance-2026-09-05.md)
for the final pin, exact samples and remaining validation limitations.
The checkpoint below records the pre-measurement plan.

2026-09-05. Accepted Jolt b160c87ea / Akita 369a1c7ff. Serial M4 Max;
no subagents. This integrates retained components, not a new kernel design.

## Candidate and invariant

Use Akita f46bbc181: two-task D128/rank3 commitment, D64 D-role relation
rows, row-aligned D128 fold. Apply the earlier C2 Jolt adapter and shared
CPU/Metal K256 catalog selection. Regenerate both catalogs. No change to
SIS policy, challenge distribution, response digits, range topology,
transcript, verifier formulas, or the fixed 1.13 M5 projection.

On this date the user explicitly approved correcting ONLY the synthetic
two-task intercept ceiling from 1.6 to 3.2 s for a twofold logical traffic
error. The canonical floor spec and campaign ledger record that correction.
Slope <=3.2 ns/hot, production commit B/F/S <=9.8/13.5/14.4 s, CPU commit
<=19 ns/hot, correctness, RSS <=90 GiB, watchdog and frozen pair/matrix
requirements remain unchanged. No accepted improvement yet: 9.25135 MHz.

## Evidence and falsifier

Production-family synthetic C2 fold: 0.761633 s cold at T28, versus the
1.90 s component bar. D128 A quotient CPU fallback: 0.773440 s cold,
0.138521 s warm at 524288 columns. Existing overlapped NTT preparation
already includes those A slots; the cold/warm difference is not a new
saving. Do not add another A kernel without contrary full-path evidence.

The original C lost about 3 s overall despite saving about 4 s in commit:
Stage8 regressed about 6.7 s. The retained D-role and fold remove measured
owners of that regression. Root witness remains padded to 2^30, so no
domain-halving credit is assumed. About 2.436 s common saving is needed for
the target from the accepted 29.61/36.25/33.18 s workload matrix.

## Bounded sequence

1. Catalog regeneration and fmt; focused adapter/catalog/mismatch and
   retained-family CPU/Metal parity tests. Reuse unchanged-component
   evidence; record inherited validation failures as non-green.
2. Build the frozen modular_benchmark evaluator, identify its revision and
   SHA256, then one Fibonacci T28 Chrome diagnostic after 120 s idle.
   Process cap 180 s, require PROOF_VERIFIED backend=metal value=true.
   Stop on watchdog/abort, failure, or RSS approaching 90 GiB.
3. Inspect total, commit, Stage8 and A prewarm exposure together. If no
   plausible complete-path gain or production commit misses its ceiling,
   reassess the measured owner; do not run a blind sweep.
4. If feasible, frozen Fibonacci parent/candidate/candidate/parent pair
   with 120 s gaps and the accepted parent binary. Promotion requires
   >=0.20 s saving, all correctness/clippy/fmt gates, both catalogs and
   mismatch rejection, then the all-three integrated matrix before push.

Initial checkpoint budget: 30 minutes through build and the first diagnostic;
report concrete results or the exact failed gate before extending. No tile
variants or protocol changes are part of this checkpoint.
