# Akita prover/memory co-design at 2^26

## Goal

On `perf/optimize-akita-prover` (jolt) + `perf/onehot-commit-sweep` (`~/akita`
path deps), optimize the Akita packed prover at 2^26 sha2-chain on this
machine (M4 Max, 16 cores, 128 GB) until, measured back-to-back:

- **Prove ≤ dory** (≤ 119 s), stretch **≤ 0.75× dory** (≤ 90 s).
- **Peak RSS ≤ 1.1× dory** (≤ 40 GB), minimum bar ≤ 60 GB. RSS also gates
  2^27 on 128 GB machines.
- Verify stays ≤ 350 ms, proof stays within the 1% catalog slack (~100 KB),
  and all correctness gates stay green.

Setup/preprocessing is excluded from the prove objective but must not grossly
regress (dory pays 32 s preprocessing; akita pays ~4 s — keep it single-digit).

## Baselines (2026-07-23/24, this machine, same harness)

| Config | Prove | Verify | Peak RSS | B/cycle |
|---|---|---|---|---|
| dory (day-1 sweep) | **119.49 s** | ~0.4 s | 36.4 GB | 543 |
| akita @ session start (pin fc20716) | 333 s | 148 ms | 80.2 GB | 1,195 |
| akita + A1 kernel + J2 + P1 (n_a=6) — **current** | **136.7 s** | 217 ms | 80.1 GB | 1,193 |

Current traced attribution (155.4 s traced ≈ 136.7 s untraced): commit 82.4 s
(53%), opening/fold 20.0 s, stage1 14.6 s, stage6b lattice booleanity 10.9 s,
stages 3/4/5/6a/7 ~21 s, witness+assemble ~16 s.

Landed so far: A1 sub-block chunking (333→154 s), J2 uni-skip decode dedupe
(stage1 init 15→6.2 s), P1 rank-aware catalogs (root n_a 7→6, 154→136.7 s,
+112 B proof), A5 fused multi-poly sweep (commit 82.4→51.4 s traced;
124.7 s traced prove), J3 sequential decode carry (three trace scans),
merge-tile L1 tuning (commit target ~38-40 s).

## Measured campaign log (traced @2^26 unless noted)

| Milestone | prove | commit | notes |
|---|---|---|---|
| session start | 333 s | ~252 s | safe-path fallback pathology |
| A1 | 154.4 s untraced | — | |
| A1+J2 (n_a=7) | 170.2 s | 96.0 s | stage1 init 15→6.2 s |
| +P1 (n_a=6) | 155.4 s | 82.4 s | 136.7 s untraced; RSS 80.1 GB |
| +A5 fused sweep | 124.7 s | 51.4 s | merge span 797 thread-s = 68 ns/accum (L2-resident tile) |
| dory, same branch (J2+J3 shared) | 122.4 s | 38.0+26.7 (w+c / open) | untraced est ~114 s; RSS 36.6 GB |
| +J3 +self-reducing merge kernel, L1 tiles | **100.34 s untraced** | — | **primary bar (≤119 s) cleared**; RSS 94.9 GB (M2 now top item) |

Detour recorded: naive L1 tiles regressed to 252 s — the 2^15 accumulator cap
was silently splitting every trace-scale block 16×, so small tiles re-streamed
A ~29×. Fixed by cap-triggered self-reduction into canonical partials
(akita 015669b9); block splitting removed from the merge path entirely.

**PIOP verdict (traced stages @2^26):** legacy (dory) 55.4 s vs packed ~42 s
like-for-like (≈1.3×, concentrated in stage5 1.8× and stage6b 1.6×). The
historical "PIOP ≈2×" held against the pre-J2/J3 legacy baseline; J2+J3 are
shared Spartan code and improved both provers (legacy stage1 24→11.9 s).

## Cost model (the math)

Shapes at 2^26/K256: P = 29 one-hot polys, T = 2^26, root n_a = 6,
positions_per_block (ppb) = 2^21, 128 live blocks/poly → 32 blocks each of
2^21 positions per poly; ring D = 64, coefficient = f128 (16 B), A entry =
one ring = 1 KB.

**Commit work.** One ring-accumulate per (poly, position, A-row):
`N = P · T · n_a = 29 · 2^26 · 6 ≈ 11.7 G` accumulates.

**Commit traffic.** Per (poly, block, A-row) the kernel streams that A row
(width × 1 KB); accumulator tiles are L1/L2-resident (row-pass keeps one ring
per block live). A is *identical across all 29 polys*, but the sweep runs
per-poly, so today:

```
A-traffic = P · n_a · T · 1 KB ≈ 11.7 TB  → ~28 s at ~400 GB/s
```

Fusing the sweep across polys reads each A row once per (block, row) and
accumulates into P block-accumulators (29 × 2 KB — still L1/L2-resident):

```
A-traffic(fused) = n_a · T · 1 KB ≈ 0.4 TB  → ~1 s
```

**Commit ALU floor.** Same-session kernel A/B measured the NEON wide
accumulator at **28 ns per ring-accumulate** at a 64 KB tile:

```
commit(fused) ≈ 11.7 G × 28 ns / 16 threads ≈ 21 s      (from 82 s)
```

**Prove floor with everything below landed:** commit ~21-25 s + opening/fold
~15-18 s + stage1 ~10-11 s + stages ~28-30 s + witness ~12 s ≈ **~90-100 s**
≈ 0.75-0.85× dory. Beyond that requires protocol-level work (out of scope).

**Memory accounting (1,193 B/cycle today).**

| Component | Size @2^26 | B/cycle | Note |
|---|---|---|---|
| Expanded A matrix | 12.6 GB | 188 | n_a·ppb·1 KB; was 7.3 GB at ppb 2^20 |
| One-hot block cache | 15.6 GB | 232 | SingleChunkEntry 8 B × 29 polys, held commit→stage 8 |
| Guest trace + witness polys | shared-with-dory baseline | ~300-400? | needs measurement |
| Unattributed (fold buffers, scatter/count, duplicates) | ~25-30 GB | ~400+ | **allocative pass required** |

Floor estimate: trace+witness (~250-350) + indices-as-u8 (29-58) + A (188)
≈ **~500-600 B/cycle ≈ dory parity**. The information-theoretic one-hot
witness is 29 B/cycle — everything above that is engineering.

## Workstreams (ranked)

### A5 — Fused multi-poly column sweep (commit 82 s → ~21-25 s) [BIG]

Batch all P polys' blocks through one A pass: for each (block index, A row),
stream the A row once and accumulate into P per-poly block accumulators
(P × 2 KB wide rings, L1-resident). Requires plumbing `commit_inner` from
per-poly (`api/commitment.rs` cfg-iter over polys → `OneHotPoly::commit_inner`)
to a batched entry point (`OneHotBatchView` already exists for other kernels).
Keep per-poly output shape (t rows per poly per block) identical —
transcript-identical. Watch: parallelization axis moves from (poly ×
block-batch) to (block-batch × row) with an inner poly loop; keep all 16 cores
fed (32 blocks × 6 rows = 192 units ≫ 16 ✓, and tiles can split further).
Gate: akita unit suite (byte-equality vs per-poly path), muldiv both modes,
then traced + untraced 2^26.

### J3 — Stage1 sequential decode cache (−2-4 s)

`R1CSCycleInputs::from_trace(t)` decodes `trace[t]` AND `trace[t+1]` (incl.
two bytecode-PC map lookups); the pair loop visits t sequentially per x_out
chunk, so every step still decodes twice. Split decode out
(`(JoltTraceCycle, pc)`), carry the previous "next" forward within a chunk.
Also applies to `evaluation.rs:867` (`compute_claimed_inputs`) and the
outer-remaining path (outer.rs:861) if they scan sequentially.
Transcript-identical; gate muldiv both modes.

### M1 — Block-cache shrink or early drop (−12-15 GB)

The 15.6 GB SingleChunkEntry cache is retained from commit through stage 8 but
K=256 entries are recomputable from the OneHotTrace indices (8 B stored vs
29 B/cycle information floor as u8 chunk indices + implicit position).
Options: (a) drop after last use and rebuild for the fold pass, (b) store u8
chunk indices only and synthesize entries per block on the fly, (c) keep but
release per-poly as the fold consumes them. Requires finding every consumer
(commit, decompose_fold, onehot_accumulate). Perf-neutrality gate: 2^26 prove
unchanged ±2%.

### M2 — Allocative attribution of the unexplained ~30 GB (then act)

Run the allocative build (`RUST_LOG=debug --features allocative` flamegraph
path) or targeted RSS checkpoints at 2^26 (or 2^24 if too slow) to attribute
peak RSS. Known suspects: pre-folded e/fold scratch scaling with ppb,
duplicate A forms in prepared setup, count/scatter arrays (2^21 × u32 ×
threads), `b_input_flat` digit carriers. Then cut the top items. Target:
−20 GB+, bringing 2^26 under 60 GB and 2^27 within reach.

### O1 — Opening/fold span split + cuts (−3-6 s)

Add spans inside `RingRelationProver::new` (decompose_group_e_hat exists; add
v-rows, grind, w-build) and around `decompose_fold_batched` /
`onehot_accumulate` internals; re-trace; cut what's engineering (the fold
grind is protocol-mandated — don't touch its parameters).

### S1 — Lattice booleanity + bind loops (−3-8 s)

stage6b (10.9 s) and `MultilinearPolynomial::bind_parallel` (6.8 s across
3,141 calls) are generic sumcheck costs; profile compute_message inner loops
before touching. Benefits dory too where shared.

### P2 — Catalog grid hardening (correctness of the P1 trade at scale)

The 1% slack selection reshaped nv 35-38 roots to 1,024-4,096 live blocks;
unrunnable on this machine but plausibly fold-heavy. Before upstreaming:
either cap live-block growth in the rank-aware selection (prefer fewer blocks
among equal-n_a candidates — already the tie-break via bytes; verify), or
validate at 2^27 on a bigger box. Also re-check the K16 regime (T < 2^25).

### Dead ends (documented, do not revisit without new evidence)

- **A4 lazy u128 accumulators**: loses 1.9× to NEON wide (28 vs 53 ns) —
  row-pass tiles are already L1-resident, accumulate is ALU-bound, and scalar
  carry-correction chains lose to 8-lane i32 adds. `Fp128Lazy` + the
  `HasCommitAccum` seam stay in-tree for probes (akita 676ef27b).
- **A2 A-sharing via bigger tile budgets**: tile-budget insensitivity showed
  the win isn't in accumulator tiling; the A-reuse win is across *polys* (A5).

## Protocol (goal mode)

- Iterate at 2^22 for quick signal where possible; confirm at 2^26. Thermal
  discipline: never compare across sessions; same-session back-to-back A/B or
  min-of-N; a single hot e2e number is not evidence (two false alarms so far).
- Gates per landed change: `cargo nextest run -p jolt-prover-legacy muldiv
  --features host` and `--features host,zk`; akita
  `cargo test -p akita-prover --release --features parallel --lib`; jolt-akita
  suite (drift + coverage) when catalogs/planner change; clippy both modes
  (check `pipestatus`, not `$?` after a pipe).
- Transcript identity required except for catalog regenerations (which change
  proofs by design and must pass the full e2e + drift suite).
- Commit per accepted iteration (`perf(akita): …` / `perf(spartan): …`),
  no force-push, no push without ask. Failed attempts: revert, document in
  this spec's dead-ends section.
- Harness: `PERF_LOG_T=N [PERF_TRACE=1] cargo nextest run --release -p
  jolt-prover-legacy --features akita -E 'test(sha2_chain_akita_perf)'
  --run-ignored all --no-capture`; traces land in
  `benchmark-runs/perfetto_traces/`; peak RSS via `/usr/bin/time -l`.

## Acceptance

Done when prove ≤ 119 s AND RSS ≤ 60 GB at 2^26 with all gates green
(stretch: ≤ 90 s / ≤ 40 GB), or when every workstream above is landed or
written off with measurements — whichever comes first. Record final numbers
in this file before upstreaming discussions.
