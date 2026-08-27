# Metal wave 5 — lane R: st0/st5 re-attribution

## Verdict

**st5: the wave-3 scan kernels ARE engaged at 2^27 — the +5.4 s vs the
wave-3 model is a model error, not a regression and not window skew.** The
phase+suffix scans cost 11.80 s of GPU command-buffer wall at 2^27 (69% of
st5); the wave-3 fixture-calibrated model priced them at ~4.1 s. The
production-vs-microbench per-row gap is a constant **2.9× (phase) / 2.1×
(suffix) at every scale** (2^22, 2^25), i.e. a dispatch-context effect, not
a scale or engagement effect.

**st0: the 16.2–17.1 s `stream_witnesses` window is HOST-bound at 2^27**
(opposite of 2^25): the driver spends 8.7 s extracting witness bundles +
8.3 s building GPU jobs while `TraceRecord::collect` (12.8 s of CPU) runs
concurrently and dilates both; the Dory-commit GPU work (tier-1 8.8 s +
tier-2 Miller 5.5 s CB-wall) fits underneath with **7.9 s of GPU-lane
starvation**. More GPU offload does not shrink st0; cheaper host
extract/build or record-collect does.

Instrumentation commit: `b44ea7f95` on `lane/metal-w5-reattr`
(attribution-only spans; clippy clean; no kernel/schedule/protocol change;
no existing span renamed).

## Evidence base

- 2^27 instrumented profile, record-grade window at start (FrBind 256 µs;
  2^22 probe 3.36 s ≤ 3.40 gate): 70.82 s wall, RSS 72.3 GiB. Walls ran
  ~5–10% above the Aug-24 record (63.88/64.56); the inflation concentrates
  in uninstrumented host-heavy stages (st1 6.93 vs 5.03, st6b 7.49 vs 5.29)
  → ambient CPU load, shares unaffected.
- 2^25 spans run (17.66 s) + 2^25 `JOLT_METAL_CB_TRACE=1
  JOLT_IRR_SPLIT_TRACE=1` run (per-CB GPU timestamps).
- 2^22 spans smoke (3.36 s certified window).

## st5 @2^27 — 17.11 s stage (record window: 16.34)

| component | s | note |
|---|---:|---|
| IrrPhaseScan+reduce + IrrSuffixScan+reduce (16 CBs, device) | **11.80** | `IrrScanner::phase_run` ≈ 100% of `IrrKernel::phase_scan_device`; split ≈ 8.0 phase / 3.8 suffix (2^25 CB ratio 68/32) |
| RegistersValEvaluation::prove_round (host) | 3.01 | untouched by wave 3 |
| IRR prepare (shared-rows reclaim) | 1.84 | + RegVal prepare 0.35 |
| cycle_adopt + device cycle init | 0.53 | adopted; 13 device rounds (wait 0.22) + 14 host tail (0.01) |
| address_message ×128, binds, output_claims | 0.05 | noise |

**Engagement receipts @2^27:** `IrrKernel::phase_scan_device` ×16,
`IrrKernel::phase_scan_cpu` ×0 (the 6 GiB rows + 4 GiB u_evals nocopy wraps
succeed); `IrrKernel::cycle_adopt` succeeded (factors = 9 ≤ 16); optimized
kernels selected (no `*_LEGACY` env). The wave-3 collision-only SIMD scatter
and 2048-group suffix schedule are live.

**Gap mechanism (model vs measured):** wave-3 modeled st5 ≈ 10.87 s by
share-calibrating the microbench kernel times (2^24 fixture: phase
18.9 ms, suffix 12.1 ms per scan). Production per-phase CB GPU time @2^25:
**109 ms phase / 51 ms suffix** vs fixture-scaled 37.8 / 24.2 → 2.9× / 2.1×.
Same ratio @2^22 (e2e 23.6 ms/phase vs fixture 8.2 ms). Scaling 2^22→2^25
e2e is linear (7.5× for 8× rows) → per-row cost is scale-flat; the gap is
the e2e dispatch context (single cold CB per phase with host gaps between,
vs the fixture's back-to-back warm dispatch loop — clock/residency state),
NOT engagement, NOT sustained-load window skew (fresh certified window
reproduces 17.1 s).

**Window-skew answer:** none for st5 — the record trace's 16.34 was genuine
(fresh-window 2^25 scales to it: 3.75 × 4 + log terms; fresh 2^27 measures
17.1).

## st0 @2^27 — 17.75 s stage; stream_witnesses 17.10 (record: 16.86/16.21)

Driver thread (= the critical path; sums to 17.09 ≈ the window):

| driver | s | share |
|---|---:|---:|
| stream_extract (bundle materialization, rayon) | 8.67 | 50.7% |
| build_gpu_job (hot-addr count-sort + scatter) | 4.06 | 23.8% |
| build_inc_job (i128 extract + digit recode) | 4.26 | 24.9% |
| send_wait (GPU backpressure) | **0.08** | 0.5% |

Lanes under it (off critical path):

| lane | busy | waiting |
|---|---:|---:|
| GPU lane | gpu_run 8.82 (tier-1 G1SegSum CB-wall), readback 0.18 | **recv_wait 7.86 — starved by the driver** |
| tier-2 lane | decode 2.27 + cpu_absorb 3.10 + miller_fold 2.20 + reduce_inc 1.76 | miller_wait 5.47, recv_wait 2.16 |
| overlapped host | TraceRecord::collect 12.84 (background pool, whole window) | — |
| prepare (before stream) | prepare_tier2 0.47 ∥ base_affine_cache 0.47 | — |
| after stream | lane_drain 0.05 + finish_columns 0.05 | — |

- **Dory-commit share separated from trace-collect:** Dory GPU = 8.8 (tier-1)
  + 5.5 (tier-2 Miller) CB-s; Dory host = 8.3 (job build) + 9.3 (tier-2
  decode/absorb/fold); witness extract = 8.7; TraceRecord::collect = 12.8
  overlapped (not in the stream span, but stealing cores from it).
- **Regime flip vs 2^25:** at 2^25 the driver blocks on send (1.75 s,
  GPU-bound); at 2^27 send_wait ≈ 0 and the GPU starves 7.9 s (host-bound).
  Driver work dilates ~1.6× over clean 4× scaling (extract 1.35→8.67, jobs
  1.75→8.33) while collect grows 2.35→12.84 — CPU contention between the
  two is ~5 s of the st0 wall.
- `scalar_affine_base_fill` (streaming.rs:464): **never fires** at 2^27 —
  `commit_inc` rides the device (no CPU `feed_i128_rows`); the only base
  conversion is `base_affine_cache` 0.47 s, fully hidden under
  `prepare_tier2` in the rayon join.

## Ranked remaining targets @2^27

| # | target | size | mechanism / lever |
|---|---|---:|---|
| 1 | st5 phase+suffix scan dispatch efficiency | 11.8 s wall, ~7.7 s headroom at fixture rate | constant 2.9× e2e-vs-microbench per-row gap; reproduce in isolation (cold single-CB dispatch with host gaps vs warm loop), then chase clock/residency |
| 2 | st0 driver host path | 17.1 s wall; ~5 s is contention-dilation, more from fusing | extract then build re-walks the same superchunk (bundle staging + hot-addr re-derivation = two passes); single-pass extract→bucket, or cheapen/reschedule TraceRecord::collect (12.8 s CPU) |
| 3 | st0 tier-2 Miller on device | 5.5 s CB-wall + 2.2 s host fold | fits under the window today but floors any driver win (GPU demand 14.3 CB-s); B-lane fold work applies |
| 4 | st5 RegistersValEvaluation | 3.0 s host | untouched by wave 3; ordinary prove_round mass |
| 5 | st5 prepare + cycle init | 1.8 + 1.1 s | shared-rows reclaim copy; device init CBs |

st0 mass explained: 17.09/17.10 driver-side (100%); st5: 16.7/17.1 (98%).

## Discipline

- Timed 2^27 runs used: 1 of 2 (the instrumented profile). 2^25/2^22 runs
  were engagement/CB diagnostics, not certification claims.
- Battery not run (attribution-only lane; clippy on touched crates clean:
  `cargo clippy -p jolt-kernels -p jolt-witness -p jolt-dory --features
  jolt-kernels/metal --all-targets -- -D warnings`).
- Not pushed; `scratch/metal-saturation` untouched.
