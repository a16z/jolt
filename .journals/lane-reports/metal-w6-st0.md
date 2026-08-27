# Metal wave 6 — lane S0: st0 driver host path

## Verdict

**Fused single-pass driver RETAIN, default-on (`b4ad086c0` on
`lane/metal-w6-st0`).** The extract-then-rebuild driver (bundle vec →
40 per-column re-walks → block-parallel count/scatter → serial inc staging)
is now one subchunk-parallel pass over the raw trace rows plus
geometry-flat scatters. Driver-span sum (extract+builds), paired
same-window sha2-chain: **2^24 ABBA 1.444 → 1.059 s (−26.7%)**; 2^25
hoist-off pair **2.313 → 1.870 s (−19.2%)**. Proof bytes identical
(byte-diff 20/20; `metal_commit_matches_optimized` exact on commitments
AND hints). Peak RSS @2^25 neutral (±0.1 GiB across pairs). Bar (≥1.5 s
modeled st0 @2^27) met: **modeled driver-span cut −2.9…−4.4 s**, st0 wall
−2.5…−3.3 s (floors at the GPU queue), remainder banks until tier-2
shrinks.

## Why this pays exactly at 2^27 (the geometry argument)

Trunk's count/scatter parallelism is per-(column, window) block:
`windows_per_sc` = 8 @2^24 → 4 @2^25 → **2 @2^27** (row_width 2^16), so
inc passes collapse to 4 parallel blocks and the whole build side lands on
the driver's critical path (R: build_gpu 4.06 + build_inc 4.26 = 8.32 s =
8.1 ms/superchunk contended). The fused driver's unit of parallelism is a
1024-row subchunk (128/superchunk at every scale): builds measured
**geometry-flat ~3.2-3.8 ms/superchunk** at 2^24 AND 2^25. Trunk gets
cheaper toward 2^24 (which is why 2^24 overstates the win) and collapses
toward 2^27 (which is why 2^25 understates it).

## What landed (one commit)

1. **Extract→bucket fusion.** `MetalColumns` is no longer a
   `StreamConsumer`; `commit_streaming_metal` drives `source.visit_chunks`
   directly and each superchunk takes ONE parallel pass: per row,
   `CommittedColumnsWitness::from_row` once → u16 hot addresses (staged
   column-major per subchunk) + bucket counts + i128 increment scalars.
   Kills: the 10.5 MB/superchunk bundle vec, ~40 full re-walks of it
   (~420 MB/superchunk of re-read traffic), the 84 MB/superchunk
   `Vec<Vec<Option<usize>>>` hot staging, and the fully serial inc
   staging. Driver DRAM traffic per superchunk drops ~750 MB → ~60 MB.
2. **Bucket-major/subchunk-minor cursor layout (`BucketLayout`).** Count/
   cursor matrices are stored in exactly the flat gather-array order
   `(column, window, bucket, subchunk)`, so the counts→cursors layout
   prefix is ONE sequential dependent-add sweep and bucket boundaries fall
   out at `spw` strides. The first cut kept per-subchunk-major counts and
   did a strided prefix sweep: **44 ns/element at 2^25 geometry vs
   ~1 ns sequential** (0.93 s of pure prefix at 2^25, `oh_prefix` span) —
   the strided dependent chain is a real cliff, avoid it anywhere else.
   Scatters write through single-owner raw cursors (every `(bucket, s)`
   cursor has one owning subchunk); each bucket's contents keep ascending
   cycle order, so jobs are bit-identical to trunk's.
3. **Inc build parallelized + de-serialized.** Digit count + scatter now
   subchunk-parallel (was 2-16 blocks); staging extraction fused into the
   main pass (was serial per column). build_inc @2^25 paired: 0.98 →
   0.62 s (−37%); @2^27 model: 4.26 → ~1.5-2 s.
4. **Job-slab recycling (`SlabPool`).** The GPU lane returns gather/
   segment slabs after CB completion; the driver re-fills warm pages
   instead of `posix_memalign`+zero+first-touch-fault per job (slabs
   travel full-length; segment tables reference only the written prefix,
   so stale tails are never read — byte-free). Measured on/off @2^24:
   build_gpu 0.325 → 0.233 s. Kill switch `JOLT_METAL_JOB_SLAB_REUSE=0`.
   Sys-time drops ~1.2 s e2e @2^24 (fewer kernel page ops).

## Numbers (paired same-window, sha2-chain)

2^24 ABBA (T-C-C-T, 30 s cooldowns):

| driver span | trunk mean | fused mean | Δ |
|---|---:|---:|---:|
| extract (`stream_extract`→`extract_bucket`) | 0.605 | 0.580 | −4% |
| build_gpu_job | 0.624 | 0.240 | **−62%** |
| build_inc_job | 0.216 | 0.239 | +11% |
| **sum** | **1.444** | **1.059** | **−26.7%** |
| send_wait (absorbs the cut; GPU-bound here) | 1.047 | 1.457 | +0.41 |
| TraceRecord::collect | 0.913 | 0.889 | −2.6% |
| st0 wall | 2.622 | 2.640 | +0.7% (floor) |

2^25 hoist-off pair (u16 hot, C-first, 60 s cooldowns): extract 1.047 vs
0.963 (**+8.7% — the staging residue**), build_gpu 0.404 vs 0.638,
build_inc 0.419 vs 0.712, **sum 1.870 vs 2.313 (−19.2%)**; st0 wall flat
(4.47 vs 4.41, send_wait 2.52 vs 2.01). RSS: T 26.7-29.1 GiB, C
26.6-29.4 GiB across four pairs — window jitter dominates, arm-neutral.

## 2^27 model (orchestrator certifies)

Trunk @2^27 (lane R, contended): extract 8.67 + builds 8.33 = 17.09 s
driver. Fused: extract 8.67 × [1.0…1.19 measured C/T ratio] ≈ 8.7-10.3;
builds 3.2-3.8 ms/superchunk × 1024 × [1.3…1.54 contention] ≈ 4.2-6.0 →
**driver ≈ 12.7-14.2 s (−2.9…−4.4 s)**. Wall: GPU queue under the window
carries ~14.3 CB-s (tier-1 8.8 + tier-2 Miller 5.5, one device queue), so
st0 ≈ max(driver, GPU-queue) + drain ≈ **~14.5-15 s vs 17.8 (−2.5…−3.3 s
wall)**; the rest of the driver cut is banked against tier-2 shrinkage
(B's cap-32 door, fly-Miller work). Biggest model unknown: the fused
extract's contention sensitivity (from_row's pc_map random access) — watch
`extract_bucket` vs R's `stream_extract` 8.67 at cert.

## Collect contention (mandate lever 2) — measured, no code change needed

- Cooled paired runs show **collect UNCHANGED** under the fused driver
  (2^24 ABBA 0.913 → 0.889; 2^25 hoist-on ABBA 2.06 → 2.13 ≈ position
  noise). The driver's traffic cut shrinks the bandwidth-coupling surface
  (W3D's pinned mechanism); no scheduling change shipped, nothing to
  kill-switch.
- Knob re-probe on the fused driver @2^24: `JOLT_RECORD_BACKGROUND_THREADS=6`
  collect 1.98 (worse — longer walk), `JOLT_RECORD_QOS=utility` 1.43 (nil),
  both 1.50 (nil) — matches W3D's negative result; defaults stay.
- **Measurement trap (cost me ~6 runs): back-to-back T-C-T-C with no
  cooldown produced a fake "+60% collect dilation" on whichever arm ran
  second-in-pair.** With 30-60 s cooldowns it vanished entirely. Same-window
  interleaving is NOT sufficient at 2^24/2^25 — pairs also need cooldowns
  and position-balancing (ABBA/BAAB), or collect/tier2 numbers are
  position artifacts.

## Re-wrap tax (mandate lever 3) — mis-attributed to st0, measured dead

st0's `gpu_wrap` span: **7-10 ms TOTAL @2^24** (65-80 µs/CB) — S5's
~13 ms/CB wall tax belongs to st5's multi-GiB scanner wraps, not st0's
10-25 MB job slabs (wrap cost scales with region size). No MTLBuffer cache
built; the slab pool (above) addresses the real st0 buffer cost, which was
driver-side alloc/zero/fault churn, not the wrap call.

## Parity + gates

- Byte-identity mechanism: same per-row facts, same derivations, buckets
  keep exact cycle order (subchunk-ascending bases = original row order),
  same segment split points, same segs tables → identical GPU inputs →
  identical commitments/hints/proof bytes. Slab stale tails are outside
  every segment range.
- `cargo nextest -p jolt-kernels -p jolt-dory -p jolt-eval` (metal):
  **405/405**. `jolt-prover --features prover-fixtures`: **20/20**
  (byte-diff vs legacy; 1 known-class leaky flag). muldiv covered by the
  fixtures suite; clippy `--all --features host -D warnings` clean; fmt
  clean. `metal_commit_matches_optimized` exercises whole-trace /
  single-window / cpu-inc / all-device / all-CPU Miller arms.
- `inc_rows_match_direct_msm` (arkworks MSM oracle) retained against the
  new `build_inc_job`; `G1SegBenchFixture` now stages through the shared
  layout + `build_one_hot_job` (no shadow builder kept).
- Span schema: `stream_extract` no longer appears in st0 (the metal path
  bypasses `deliver`); its replacement is `MetalCommit::extract_bucket`,
  plus new sub-spans `oh_prefix`/`oh_segwalk`/`oh_scatter` under
  build_gpu_job. Driver-sum comparisons across trunk/lane =
  {stream_extract | extract_bucket} + build_gpu_job + build_inc_job.
- New env switches: `JOLT_METAL_JOB_SLAB_REUSE=0` (slab ablation). The
  restructure itself has no toggle (host-structural, W3D-F2 precedent) —
  A/B via trunk binary.
- jolt-kernels gains a direct `jolt-program` dependency (TraceRow; was
  already transitive via jolt-witness).

## Discipline

- Timed 2^27 runs: **0**. Iteration at 2^24/2^25 driver spans; decision
  pairs interleaved same-window with cooldowns; FrBind 255.96 µs (<350
  gate) at the final pairs. All cargo under the wave-3 cargo lock, all
  runs under the GPU lock.
- Probe scaffolding audited out (sub-spans retained deliberately as
  attribution instrumentation, R-lane precedent; slab switch retained as
  the ablation arm).
- Not pushed; `scratch/metal-saturation` untouched. Worktree
  `.worktrees/metal-w6-st0` (branch `lane/metal-w6-st0` @ `b4ad086c0`)
  ready for merge + cleanup after the wave gate.
