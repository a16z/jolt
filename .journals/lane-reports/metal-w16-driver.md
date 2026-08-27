# Metal wave 16 — lane D16: st0 driver unbank (builder-lane overlap)

## Verdict

**RETAIN, two bundled cuts (`8001d8331` on `lane/metal-w16-driver`, base
8afbfe83f): (1) builder-lane overlap — job builds (prefix, segment walks,
both scatters, GPU-queue send) move off the driver thread onto a dedicated
scoped lane fed by recycled double-buffered `StagedChunk` staging sets, so
superchunk N+1 extracts while N builds; (2) `MILLER_CPU_FRACTION_DEFAULT`
0.05 → 0.0 — the 1.94 s tier-2 `cpu_absorb` moves onto device slack
(+~0.11 CB-s), because tier-2 (6.76 s busy @2^27) is the next-binding lane
once the driver unblocks.** Same-window @2^25 pair: **11.01 vs 11.39 s
(−0.38 s)** — ON is a record-class indication (standing 2^25 record 11.85;
lane window, not a cert claim). **Modeled @2^27: st0 8.39 → ~6.8–7.1
(−1.3..−1.6 s wall; conservative floor −1.1) — bar (≥1.0) passed on the
model; the orchestrator's 2^27 kill-switch ABBA + RSS capture decides**
(`JOLT_METAL_DRIVER_OVERLAP=0` + `JOLT_METAL_MILLER_CPU_FRACTION=0.05`
restores the serialized single-thread schedule and the old split).
Byte parity: metal suites 414/414; proof-byte ratchet **20/20 plain AND
20/20 metal-armed**; clippy host + kernels metal,bench-utils clean.

## Regime verification (mission item 1) — ZERO GPU runs spent

The w15 gate's traced 2^27 run (38.98 s, Aug 27 04:22, trunk acb7cb95d)
survives in the scratch checkout's `benchmark-runs/perfetto_traces/` —
parsed it instead of burning the sanctioned profile. **Post-tiling the
driver IS the binding lane; R12's precondition confirmed exactly:**

| lane @2^27 (w15 trace, st0 8.389 / stream 8.307) | busy | wait |
|---|---:|---:|
| driver | extract 3.179 + build_inc 2.217 + build_gpu 2.090 = **7.486** | send_wait **0.813** (w12: 2.05) |
| GPU | gpu_run 6.307 (w12: 7.66 — tiling −1.35, transferred) + readback 0.282 = 6.59 | recv_wait 1.14 |
| tier-2 | absorb 4.207 (decode 1.80 + cpu_absorb 1.94 + tile 0.27) + reduce_inc 2.39 + fold 0.15 = **6.76** | recv_wait 1.68 |

Driver 7.49 + 0.81 = 8.30 ≈ the whole window (99.9% explained). Steady
state after collect ends: consume 13.91 ms/sc vs gpu_run 11.52 — the
driver paces the device even without collect. While collect lives
(first 3.75 s): build_inc dilates 2.10×, build_gpu 1.50×, extract 1.31×
≈ **1.18 s of consume tax**, and gpu_run itself runs 1.19× (13.74 vs
11.52 ms/sc — host memory-traffic coupling, not scheduling).

## Mechanism

`consume_rows` kept extract → build_gpu → build_inc → send serialized on
one thread; the two host phases (extract 3.18 s, builds+send 5.12 s) sum
into the window while the device holds 1.7 s of slack. Now the driver
extracts into a recycled `StagedChunk` (scratch + inc scalars + geometry)
and ships it; the **builder lane** owns the `SlabPool` and both build
functions (bit-identical inputs, FIFO order → identical jobs → identical
commitments/hints/proof bytes) plus the GPU-queue send and its
backpressure. Two staging sets bound the overlap at one superchunk;
`JOLT_METAL_DRIVER_OVERLAP=0` drops to one set = the serialized schedule
at unchanged residency (same code path — the switch ablates only the
scheduling). Below-gate increment columns still feed on the driver
(before shipping). New spans: `stage_wait` (driver-side backpressure),
`builder_recv_wait`; `build_gpu_job`/`build_inc_job`/`send_wait` keep
their names on the builder thread — driver-sum comparisons by name stay
valid.

Miller share: `cpu_len = round(len × share)` at decode is partition-
invariant (documented, and the all-device / all-CPU arms are pinned by
`metal_commit_matches_optimized`); the 0.05 default dated from the fly
era when the device was the constraint. Tiled table absorbs a pair ~20×
cheaper than the CPU path (1.03 vs ~21 µs/pair), so 5% CPU share bought
~0.09 device-CB-s of relief for 1.94 s of tier-2 lane time — inverted
economics once tier-2 is next-binding.

## Numbers

- **@2^25 timed pair (same-window A-B, 50 s cooldowns, untraced, FrBind
  300 µs < 350 gate):** ON **11.01** vs OFF (OVERLAP=0 + FRACTION=0.05)
  **11.39** = **−0.38 s**. OFF beat the standing 11.85 record → record-
  grade window; deltas are the evidence, not the absolutes.
- **@2^24 chrome pair (untimed diagnostics):** st0 1.662 ON vs 1.803 OFF.
  ON driver = extract 0.596 + stage_wait 0.933; builder = 0.611 + 0.226 +
  send 0.693; `tier2_cpu_absorb` 0.300 → <0.05 (share knob verified).
  Both arms device-bound — @2^24 the win is absorbed by waits, as S0/E8
  history predicts.
- **@2^25 chrome (untimed, ON):** gpu_run 2.206 = **97% of stream 2.274**
  — the remaining @2^25 floor is pure device; every host lane slack
  (extract 0.77 + stage_wait 1.50; builder 1.21 + send 1.07; tier-2 busy
  1.23, recv 1.05).

## Modeled @2^27 (NOT a wall claim — gate ABBA decides)

From measured w15-trace lane busies: GPU 6.59 + 0.11 (share-0 Miller
growth: device pairs ×1/0.95 at the tiled in-pipeline rate) ≈ **6.70 =
new floor**; tier-2 6.76 − 1.94 ≈ 4.8; builder 4.31 + collect-era
dilation ≈ 4.6–5.2; extract lane 3.2–4.7 dilated — all under the GPU
lane. st0 ≈ 6.70 + fill/drain tails (~0.15) + contention slop ≈
**6.8–7.1 s vs 8.39 = −1.3..−1.6 s** (−1.1 with heavy slop). The @2^25
pair reproduces the model's small-scale prediction (−0.2..−0.4 where the
device floor caps the win). Risks the ABBA must exclude: rayon-pool
contention from extract∥builds∥collect co-run in the first ~3.7 s
(instantaneous CPU demand can exceed 18 cores; lanes have 1.5–2.4 s of
slack each to absorb it), and the share-0 device growth landing on a
window where the device is tighter than traced.

## Residency accounting (scale-transfer rule)

The only addition is the second staging set — **+29.6 MB @2^27 geometry
(0.04% of 71.9 GiB), deterministic, no row-scaled resident data**:
hot 20×2^18×2 B = 10.49 MB + oh_bases 20×256×256×4 B = 5.24 MB +
inc_bases 2×2176×256×4 B = 4.46 MB + inc_vals 2×2^18×16 B = 8.39 MB +
starts/bounds ≈ 1 MB (n_one_hot 20, n_inc 2, one_hot_k 256, superchunk
2^18, s_count 256 — from the w15 trace grid: 22 columns, total_vars 35).
Formula: `n·(2·n_oh + 16·n_inc) + 4·s_count·(n_oh·k + n_inc·2176)` bytes
per set. `PIPELINE_DEPTH` (2) and job-slab counts unchanged — queue
residency bound did not grow. Kill switch drops back to one set.
@2^25 in-run footprints were arm-neutral (ON stage-0 window 9.09→12.99
GiB vs OFF 7.84→11.71 — position/window jitter dominates; the ~15 MB
@2^25 set is invisible). RSS decision belongs to the orchestrator's 2^27
capture per the w13 rule.

## Doors closed with receipts (mission items b, c)

- **(b) build fusion residuals (inc digit second recode, oh_scatter
  staging):** CLOSED for this regime — post-overlap the builder lane
  carries 4.31 s against a 6.70 s GPU floor (2.4 s slack); cutting its
  CPU moves no wall. The ~5 CPU-s (count-pass walk + duplicate recode)
  remain a CPU-pressure bank if a future lane needs cores during st0.
- **(c) collect reschedule/thin:** CLOSED — measured tax on trunk is
  1.18 s of consume dilation (build_inc 2.10×, build_gpu 1.50×, extract
  1.31× while live) but post-overlap those phases sit on slack lanes;
  the residual harm is gpu_run's own 1.19× while-collect dilation, which
  no thread-pool knob addresses (S0/W3D knob probes were negative;
  E8 already cut collect to ~17 CPU-s). No change shipped.
- **The sanctioned 2^27 instrumented profile was NOT spent** (M15
  precedent): the regime question was answered free from the w15 gate
  trace; staging bytes are bounded by construction (no open full-scale
  parameter); the wall question belongs to the gate ABBA. Also solo-
  machine requirement was not satisfiable from this lane (sibling
  st6b lane live).

## What landed (one commit, 8001d8331)

1. `StagedChunk` + free-list recycling channels; `staging_sets()`
   (`JOLT_METAL_DRIVER_OVERLAP=0` → 1).
2. Builder lane in `commit_streaming_metal`'s scope: owns `SlabPool` +
   both builds + job send; returns send-failure for the honesty check;
   drain order driver → builder → GPU → tier-2.
3. `MetalColumns` slimmed to extract + below-gate inc feed + ship;
   module-doc pipeline diagram now four lanes.
4. `MILLER_CPU_FRACTION_DEFAULT = 0.0` (env override unchanged).
5. Deleted: `[w8-st0]` consume eprintln + `telemetry::process_cpu_s`
   (its drvcpu reading meant extract+builds on one thread — no longer a
   coherent quantity; ParPhase telemetry + spans carry attribution).

## Discipline

- **Timed GPU runs: 2** (the @2^25 decision pair; agreement, no 3rd).
  Untimed diagnostics: chrome @2^22 ×1, @2^24 ×2, @2^25 ×1;
  metal_microbench window probe ×1 (FrBind 2^20 = 300 µs). **Timed 2^27:
  0; 2^27 instrumented profile: 0** (see above). All cargo under the
  wave-3 cargo lock; every GPU run under the GPU lock, one at a time;
  cooldowns 40–50 s between runs.
- Gates: metal suites **414/414**; ratchet **20/20 first pass** in BOTH
  plain and metal-armed forms; `clippy --all --features host -D
  warnings` + `-p jolt-kernels --features metal,bench-utils
  --all-targets` clean; fmt clean; pre-commit hooks green.
- KernelId::ALL **89 unchanged**. commitment.rs 2519 → **2587** (+68
  net: builder lane + staging plumbing +99, funded −31 by the eprintln/
  process_cpu_s deletion; nothing else deletable without attribution
  loss — flagged per the soft cap).
- Sibling lane metal-w16-st6b respected: locks honored, no scratch or
  sibling worktree touched (read-only parse of the scratch checkout's
  benchmark-runs trace artifact), not pushed.
