# Metal wave 9 — lane X9: tier-1 jk_g1_seg_sum headroom (reopened w3 door)

## Verdict

**GO — three RETAINED cuts on `lane/metal-w9-xyzz` (base 71ee2c57e):
tier-1 family CBs @2^25 in-pipeline 3.578 → 2.505 s (−30.0%), e2e @2^25
ABBA −0.92 s (−6.0%), modeled st0 @2^27 ≈ −2.0 s (range −1.5..−2.3).**
The wave-3 "2.52 vs 11.30 Gmul/s" framing was half wrong: the roof stands
(re-measured 11.63 saturated) but the gap is NOT reachable ILP/arithmetic
headroom — it is thread starvation + simdgroup divergence + TG packing,
and memory is a non-factor. Cuts, in commit order:

1. `cf49bff13` — **length-sorted segment bounds + TG width 64.** Kernel ABI
   `seg_starts` prefix → per-thread `[start, end, out_slot]` triples,
   emitted length-sorted descending; `out_slot` restores bucket-walk order
   so the out array is BITWISE identical and host reducers/tier-2 are
   untouched. Dispatch width 256 → 64.
2. `eea04d67e` — attribution rig (bench-only; delete-or-keep at PR handoff,
   E8 precedent): `MetalContext::compile_variant`/`dispatch_variant` +
   `st0-contention --legs g1x` experiment matrix.
3. `1ffe9388c` — **segment cap 256 → 128** (w3 rejection overturned on the
   post-D2/E8 tier-2 lane).

Kill switches: `JOLT_METAL_G1_SORT=0` · `JOLT_METAL_G1_TG_WIDTH=256` ·
`JOLT_METAL_G1_SEGMENT_LEN=256` — the three together restore trunk device
behavior in one binary. KernelId::ALL unchanged (82; no new kernels —
variants are runtime-compiled bench-only).

## Gap attribution (mandate item 1) — where the 4.5× actually goes

All on production-shaped fixtures (real sha2-chain rows, `G1SegBenchCase`),
one superchunk, CB GPU timestamps, gpu-locked. @2^24 shape = 20,651 segs /
4.74 M adds / row 2^14 / one_hot_k 16; @2^25 shape = 11,719 segs / 2.37 M
adds / row 2^17 / one_hot_k 256 (the 2^27-like geometry: 27+8 total vars).

| probe (@2^24 shape) | median | Gmul/s | verdict |
|---|---:|---:|---|
| real kernel, trunk order+width | 7.79 ms | 6.08 | baseline (NOT 2.52 — that was the derated w3 window / in-pipeline co-run) |
| fq-mul chain, same thread count | 6.50 ms | 7.28 | **compute ceiling @20.6k threads = 62% of saturation** |
| fq-mul chain, 2^19 threads | — | **11.63** | w3's 11.30 roof re-verified; ILP-4 variant adds nothing (11.34) — occupancy, not per-thread ILP, is the throughput lever |
| fixed-base (gather removed) | 7.92 ms | — | memory latency ≈ 0 |
| gather-only (EC removed) | 0.19 ms | — | 1.7 TB/s effective — cache-resident |
| 16×-tiled bases (2^27 footprint, 18 MiB) | +2.3% | — | **memory DEAD as a mechanism at any scale** |
| software-pipelined load variant | 7.85 ms | — | compiler already hides the gather |
| length-sorted dispatch order | 5.91 ms | 8.01 | divergence + core imbalance = the biggest single term |
| sorted + width 64 | **5.63 ms** | **8.41** | shipped composition, −27.7% |

Mechanism split: simdgroup utilization (thread runtime = simdgroup max)
is 0.897 unsorted @2^24 shape and **0.794 @2^25 shape** → 0.999 sorted;
width 256 → 64 fixes threadgroup packing at ~10-20k threads (−11% alone);
the residual vs the 11.63 roof is thread-count starvation, which cap 128
attacks directly (2× threads) — in-pipeline it bought another −7.7% commit
wall; cap 64 inverts (−4.6% vs 128) as host absorb + per-thread overheads
catch up. Occupancy/registers: NOT TG-capped (pipeline maxTotal = 1024,
exec width 32) — the w3 "two accumulators +21%" rejection is consistent
with register-pressure sensitivity, but the base kernel itself is not
register-limited at width 64.

**Repriced roof:** the honest ceiling for this kernel family at production
segment counts is the thread-count-limited mul-chain rate — ~5.9 Gmul/s
@11.7k threads, ~7.3 @20.6k, 11.6 saturated — not a flat 11.30. Post-cut
the kernel runs at 91% of the same-threadcount ceiling (@2^25 shape,
sorted+w256 measured 5.42 vs 5.94); remaining in-kernel headroom ≈
threading, not arithmetic. Doors that would beat it must RAISE thread
count (finer caps price flat past 128) or cut muls/add (batched-affine
tree, below).

## Retention A/B (mandate item 2)

Same binary, env kill switches, position-balanced ABBA, 40-50 s cooldowns,
FrBind probe 250 µs (<350 gate), machine solo.

| objective | trunk (OFF) | lane (ON) | delta |
|---|---:|---:|---:|
| isolated kernel @2^24 shape (sort+w64) | 7.79 ms | 5.63 ms | **−27.7%** |
| commit slot @2^24 (sort+w64) | 2.337/2.381 s | 2.063/2.125 s | **−11.2%** |
| commit slot @2^25, +cap128 vs cap256 | 3.341/3.346 | 3.096/3.072 | **−7.7%** |
| tier-1 family CB-s @2^25 e2e (full stack) | 3.578 s | 2.505 s | **−30.0%** |
| **e2e wall @2^25 (full stack)** | 15.32/15.08 | **14.54/14.03** | **−0.92 s / −6.0%** |
| RSS @2^25 (pairs) | 25.35-25.48 GiB | 25.25-25.78 GiB | neutral (bounds slab +90 KB/job) |

e2e ON best 14.03 s @2^25 (wave-8 record 15.62; same-window OFF 15.08 —
code effect is the pair, not the absolute).

## 2^27 model (device-bound regime, E8 anatomy)

Tier-1 family 7.63 CB-s × −30.0% ≈ −2.29 CB-s. Measured wall-per-CB
transfer @2^25 was 0.92/1.07 ≈ 0.86 (the freed device time also relieves
send_wait backpressure and Miller co-scheduling). **Modeled st0 @2^27 ≈
−2.0 s (conservative −1.5, ceiling −2.3)** — clears the ≥0.8 s bar ~2.5×.
Post-cut st0 model ≈ 8.5-9 s with the GPU lane no longer pacing alone;
tier-2 Miller (3.50 CB-s) and the driver chain (~5.7 s) move up the ranks.

## Soundness / parity

- Sort + out_slot: same adds, same intra-segment order, same output slots ⇒
  out array bitwise identical by construction. Width/cap: value-exact by
  segmentation algebra; row partials normalize to affine (unique repr)
  before anything transcript-visible.
- Oracles: `assert_equivalent` (serial-Jacobian vs XYZZ vs cap variants,
  production rows) green in every g1 bench invocation;
  `metal_commit_matches_optimized` (full commitments + Dory hints == host
  optimized build) green; `seg_sums_match_arkworks`,
  `signed_seg_sums_match_arkworks`, `seg_sum_edge_cases` green (now
  exercising sorted bounds + slot mapping through `g1_seg_sums`).
- Gates: metal suites 406/406 (re-run at final default), byte-diff 20/20
  (re-run at final default), clippy host + host,zk `-D warnings`, fmt.

## Doors priced / parked

| door | verdict |
|---|---|
| memory-system anything (layout, prefetch, footprint) | **DEAD** — +2.3% worst case at 16× footprint; gather-only is 0.19 ms |
| per-thread ILP (2 accumulators, pipelined loads) | DEAD — saturated roof is occupancy-fed; ILP-4 mulroof flat; w3's +21% register finding stands |
| cap 64 | measured invert (+2.3% vs 128 in-pipeline) |
| TG width 32 | parity with 64 isolated (−1%), not worth the co-run risk; 64 shipped |
| batched-affine tree (Montgomery batch inversion, ~6 vs 10 muls/add) | **PARKED with mechanism**: only remaining >20% in-kernel door; needs TG-memory staging that w3's cap-32/TG-reduction data says costs occupancy — price only if a wave needs tier-1 below ~5 CB-s @2^27 |
| finer-grained thread assignment (multi-thread/segment) | DEAD (w3: +32%) |

## Discipline

- Timed evidence: ABBA pairs per decision (width, cap×2, full stack ×2 at
  e2e + commit) with 40-50 s cooldowns; isolated g1x sweeps are attribution,
  not retention evidence; the one isolated-vs-in-pipeline disagreement
  (w256 @2^25 shape) was decided in-pipeline per the kill-list rule — the
  isolated hint was a suite-position/thermal confound.
- No 2^27 runs. FrBind 250 µs probe before certs. All cargo under the
  wave-3 lock; all timed GPU under the GPU lockf + in-process gpu_lock.
- commitment.rs 2490 lines (over the 1000 soft flag — pre-existing wave-6
  debt item, +~40 here).
- Worktree `.worktrees/metal-w9-xyzz` @ `1ffe9388c`, 3 commits, not pushed,
  scratch untouched. Bench rig commit `eea04d67e` flagged for the PR-handoff
  audit.
