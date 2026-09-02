# Metal wave 12 — lane R12: st0 re-attribution (post-S0/D2/E8/X9 anatomy)

## Verdict

**st0 @2^27 is DEVICE-PACED but near co-bound — the regime flipped again
since w5.** In a record-class window (traced e2e 49.47 s, FrBind 250 µs)
st0 = 9.34 s: device busy ≈ 7.9–8.0 s (85% of the window, GPU starved only
1.10 s vs w5's 7.86), driver host work 7.15 s + 2.05 s send_wait (was 0.08
in w5). Driver and device are within ~0.8 s of each other — a cut on either
side alone caps at ~1.2–2.2 s wall. **The sharpest new finding: tier-2
Miller device time is effectively ADDITIVE, not hidden** — the 2^25 CB
trace shows co-running G1SegSum CBs stretch 0.21 → 5.14 µs/segment at
identical segment counts (24×), and union device busy ≈ serial sum (2.69
vs 2.81 s). The w5 "tier-2 fits underneath tier-1" framing is dead; Miller
≈ 3.5 CB-s @2^27 genuinely floors the window inside gpu_run's 7.66 s wall.

Zero code diff — the w5–w9 span vocabulary is complete on trunk
(driver/lanes/waits, oh_* sub-spans, `JOLT_ST0_TELEMETRY` CPU telemetry);
no spans added or renamed. Attribution-only lane; the only commit is this
report.

## Evidence base (entire GPU budget: 2 runs)

- FrBind probe **250.47 µs** (<350 gate) → 2^27 instrumented profile
  (chrome + `JOLT_ST0_TELEMETRY=1`): **e2e 49.47 s traced** (record 48.92),
  peak RSS 69.18 GiB, machine solo, GPU-locked. st0 9.341 / stream_witnesses
  9.234 — matches the wave-11 gate vector (9.55/9.45).
- One 2^25 `JOLT_METAL_CB_TRACE=1` run: **14.38 s e2e** (record 14.70),
  544 CBs with device timestamps — the tier-1/Miller co-scheduling split.

## st0 @2^27 — 9.34 s stage; stream_witnesses 9.23

Driver thread (= consume 9.232 s; components sum 9.199 = **99.6% explained**):

| driver | s | share | note |
|---|---:|---:|---|
| extract_bucket | 2.97 | 32% | 28.9 CPU-s (E8's model said 27.8 — exact); sampled from_row ≈ 36% of task CPU — post-E8 the cost is staging, not the row walk |
| build_inc_job | 2.14 | 23% | inc_count 4.7 + inc_scatter 5.8 CPU-s |
| build_gpu_job | 2.04 | 22% | oh_scatter 1.48 wall / 12.0 CPU-s; oh_prefix 0.37; oh_segwalk 0.05 |
| send_wait (GPU backpressure) | **2.05** | 22% | was 0.08 in w5, 2.34 in w8 — the driver now waits on the device |
| residue | 0.03 | — | |

Lanes under it:

| lane | busy | waiting |
|---|---:|---:|
| GPU lane | gpu_run 7.66 (CB-wall, tier-1 + Miller co-run dilation) + readback 0.27 + wrap 0.04 = 7.97 | recv_wait **1.10** (w5: 7.86; w8: 0.86) |
| tier-2 lane | absorb 3.85 (decode 1.61 + cpu_absorb 1.83 + settle) + reduce_inc 2.23 + fold 0.34 = 6.42 | recv_wait 3.14, miller_wait 0.02 (D2 holds) |
| overlapped host | TraceRecord::collect 3.37 wall / 17.4 CPU-s (first 36% of window) | — |
| prepare | prepare_tier2 + base_affine_cache ≈ 0.002 (T2 setup-owned tables hold) | — |
| after stream | lane_drain 0.05 + finish_columns 0.05 | — |

Window CPU: ~111 CPU-s / 9.23 s = **12.1 of 18 cores** — no longer
CPU-demand-bound (w8 old-code: 14.1/18). Waits are spread evenly across the
window (send_wait thirds 0.62/0.76/0.67) — steady-state mixed regime, not a
phase split.

## Mandate answers

**(a) Is the driver still the critical path post-S0/E8?** NO — but barely.
Driver host work is 7.15 s vs device busy ~7.9–8.0 s; the driver blocks
2.05 s on the device and the device starves only 1.10 s on the driver.
Device-only cuts floor st0 at ~7.3 s (driver-bound); driver-only cuts floor
at ~8.1 s (device-bound). E8's extract model verified exactly (28.9 vs 27.8
CPU-s predicted); builds run slightly heavy (4.18 wall vs 3.4 modeled).

**(b) How much of D2's tier-2 device work floors the window?** ~3.4–3.5
CB-s of Miller @2^27 (w7's 2.0 µs/pair rate × 1.73 M device pairs — 2^25
re-measures it exactly: 1.734 s / 14 CBs). The 2^25 CB timeline proves this
mass is effectively serial with tier-1: G1 CBs co-running with a Miller CB
run at 5.14 µs/seg vs 0.21 solo (same ~20.6 k seg shapes), and union busy
(2.685) ≈ serial sum (2.81) — co-scheduling buys ~0.13 s, not hiding.
Miller is ~40–45% of st0 device busy. The tier-2 HOST lane, by contrast,
has 3.14 s of slack and binds nothing.

**(c) Does X9's tier-1 cut show as wall or absorbed slack?** WALL, mostly:
vs the w8 clean-window anatomy, gpu_run 9.44 → 7.66 (−1.78) and st0
10.54 → 9.34 (−1.20); ~0.6 s went into slack-shift (send_wait 2.34 → 2.05,
recv_wait 0.86 → 1.10). Implied pure tier-1 ≈ 7.66 − ~3.4 Miller-dilation
≈ **4.2–4.4 s**, consistent with X9's 5.34 CB-s model once co-run
accounting is applied.

**Collect contention share:** small now. Collect is 3.37 s wall / 17.4
CPU-s (E8 modeled ~10 CPU-s — runs heavy but finishes by t≈3.4 s). While
live it dilates extract ~28% (per-superchunk wall 6.72 vs 5.26 ms) ≈
**0.2–0.3 s of driver tax**, vs w5's ~5 s. Not a door on its own.

## Ranked remaining targets @2^27 (wave 13+)

| # | target | size | mechanism / lever |
|---|---|---:|---|
| 1 | Miller fly commit device mass | 3.4–3.5 CB-s, effectively serial | occupancy is maxed (flush 65536, 2.00 µs/pair) — the door is per-pair kernel compute (the Miller loop itself) and/or scheduling: a resident Miller CB stretches tier-1 24×. Halving it ≈ −1.5 s device + send_wait relief |
| 2 | tier-1 G1SegSum pure | ≈ 4.2–4.4 s device | X9's parked batched-affine tree (~6 vs 10 muls/add, gated on TG-memory occupancy) — the only remaining >20% in-kernel door; ~−1.5 s device |
| 3 | driver builds (build_gpu + build_inc) | 4.18 s wall / ~22.5 CPU-s | E8's "parked (banked)" residue UNBANKS: the moment device drops ~1 s, builds+extract become the binding floor. Scatter staging traffic (oh_scatter 12 CPU-s, inc pair 10.5 CPU-s) |
| 4 | extract_bucket staging | 2.97 s / 28.9 CPU-s | post-E8 it is staging-bound (from_row ≈ 36% of task CPU, ~215 ns/row total across 18 threads) — memory-traffic shaping, smaller |
| 5 | collect | 3.37 s / 17.4 CPU-s overlapped | ~0.2–0.3 s driver tax; fold into a driver lane only |

Floor math: #1+#2 without #3 → st0 ≈ 7.3; #3 alone → ≈ 8.1. A paired
device+driver wave (#1 or #2, plus #3) models st0 ≈ **7.0–7.5 s**; sub-7
needs all three. send_wait (2.05) is not a target — it evaporates 1:1 with
any device cut.

Mass explained: 9.199/9.232 driver-side = 99.6%; device side fully
decomposed via the 2^25 CB receipts + w7/w9 rate anchors.

## Discipline

- GPU runs used: **2 of 2** (2^27 instrumented profile; 2^25 CB trace).
  FrBind gate 250.47 µs before the 2^27 run. Both under
  `/usr/bin/lockf -k /tmp/jolt-metal-gpu.lock`; all cargo under the wave-3
  cargo lock. No timed certification claims made.
- Zero code diff → clippy N/A (no touched crates); byte-parity trivially
  holds. No span renames, no behavior edits.
- Sibling lane metal-w12-st5scan respected: locks honored, no scratch or
  sibling worktree touched, not pushed.
