# Metal W2-st0: stage-0 walk↔commit contention — mechanism pinned, fix space killed

Lane: `scratch/metal-w2-st0`. Charter: isolate the st0 trace-walk vs Metal
commitment shared-fabric door (good ~12 s / bad 19–21 s @2^27, the ±5 s
bimodal variance), identify the adverse-mode mechanism without full proves,
and retain only a measured scheduling/dataflow fix.

## Verdict

**KILL the fix; RETAIN the harness.** The adverse mode is ambient
device-power/clock state — it reproduces on the **commit alone, with no
co-runner** (same-window solo commits 3.8 s → 22.5 s wall at constant
~22 s utime). No in-process scheduling change can remove a mode that exists
without the walk. Every parked-door candidate (start order/stagger, pool
width, QoS, and both soak controls) was measured: null, regressive, or
fail-unsafe. The one arm that beat the default in stable windows
(background-QoS × 12 threads) has an unbounded starvation tail in exactly
the degraded ambient the door is about — fail-unsafe, not retainable.

## Deliverable: the isolated objective

`jolt-eval` bin `st0-contention` (feature `metal`, macOS) — no proving:

```
/usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo run --release \
  -p jolt-eval --bin st0-contention --features metal -- \
  --scale 24 --iters 5 --legs commit,walk,corun
```

- Legs: `commit` (production stage-0 Metal commit slot, real grid/ids),
  `walk` (production `spawn_shared_record_collect` + stage-1 join),
  `corun` (production shape), `corun-bg12` (E-cluster arm),
  `soak-stream`/`soak-fault` (bandwidth-only vs page-fault-only controls),
  interleavable in one invocation (`--legs commit,corun,commit,...`) to
  kill ordering/thermal confounds.
- Per-iteration JSON: wall, commit_s, join_s, **walk span inside the
  co-run** (tracing-layer capture of `TraceRecord::collect`), minor/major
  faults, utime/stime. `--dirty-gb N` = page-residency ballast probe.
- New production **bench knobs** (default-off, env-gated, zero default-path
  change) beside the existing `JOLT_RECORD_HOIST/BACKGROUND_THREADS`:
  `JOLT_RECORD_HOIST_DELAY_MS` (spawn stagger) and
  `JOLT_RECORD_QOS=background|utility` (walk pool scheduler class), plus a
  pub seam `join_shared_record_for_bench`.

## Measurements (sha2-chain, M5 Max 18-core; AC unless noted)

Isolated distributions (medians; full JSONL in `/tmp/st0-bench*.jsonl`):

| leg @scale | solo | co-run | tax |
|---|---:|---:|---|
| commit @2^22 (battery) | 1.172 s | 1.270 s | **+8.4 %** |
| walk @2^22 (battery) | 0.277 s | ≤ window (join 0) | — |
| commit @2^24 | 2.95 s | 4.22 s | **+43 %** |
| walk @2^24 | 1.05 s | 1.7–2.3 s | **+65–118 %** |

- The good-mode tax is the known W3D bandwidth tax at 2^22 and grows
  super-linearly with the walk's window share; at 2^24 the co-run median
  already ties/loses to serialization (4.22 vs 4.04 s) — the overlap's net
  sign is geometry- and ambient-dependent, consistent with the 2^27
  bimodality being worth ±5 s.
- Co-run burns ~+1.0–1.3 extra CPU-seconds (utime 7.9→8.9 @2^22): genuine
  contention, not just core sharing.

Mechanism discrimination:

1. **Adverse mode reproduced with no co-runner.** Same-invocation solo
   commit anchors: 3.9/4.4/6.5 → 4.4/4.3/3.8 → **22.5/13.2/5.4 s** with
   utime flat (~22 s) and stime *dropping* — the device side stalls; the
   CPU work is unchanged. A power-state flip (battery→AC mid-run) moved the
   whole distribution. This is device power/clock ambient, the same
   idle-time-correlated signature as the 2^27 flagship (good after 30-min
   idle; bad on 4-min gaps).
2. **Commit inflation tracks co-runner residency, not intensity.**
   2-thread walk (minimal bandwidth, 9–10 s residency) inflated the commit
   to 8.6–18.5 s; 8-thread walk (4× the traffic, 2 s residency) cost far
   less. The commit stays derated for as long as *any* sustained CPU
   co-runner keeps package draw up.
3. **Fault-vs-bandwidth controls tie.** 2-thread fault-soak (mmap/touch/
   munmap, ~0.5–3 GB/s kernel zero-fill) and 2-thread stream-soak both cost
   ~+27 % — no page-fault/VM-lock special effect; the residency/power story
   stands, with a real but second-order fabric-bandwidth component
   (8-thread stream-soak: +85 % median, wide spread).
4. **Page residency probe:** 40 GiB touch-and-free ballast per iteration
   did not produce a distinct mode beyond the ambient drift (elevated
   window, stable within-variant) — free-list depletion is at most an
   indirect contributor (it costs power/time, not a distinct cliff).

Fix-candidate matrix (co-run @2^24 medians, same-window solo anchors):

| candidate | result | verdict |
|---|---|---|
| stagger/serialize (`DELAY_MS`) | tax → +2.7 % but total +25 % @2^22; serial ≈ bad mode @2^27 arithmetic | dead |
| width 16 / 4 / 2 | 16 ≈ default; 4 ≈ default commit at +47 % residency; 2 catastrophic (12.9 s median) | dead — never throttle the walk |
| QoS utility | walk speed unchanged solo; no tax relief | dead knob |
| QoS background @8 | walk 2.8× slower solo | dead (residency) |
| **background × 12 threads** | stable window: co-run 3.7–4.5 s vs default 4.7–8.5 s (commit tax 3–4× smaller, tight spread) | **fail-unsafe**: degraded window sample walk_dur **26.3 s** (background hard-throttle) — unbounded tail exactly when it matters |

## Why this closes the door (for now)

- The ±5 s @2^27 st0 variance is the package power/clock state at proof
  start plus the walk's residency amplifying exposure. The trigger is
  outside the process; the amplifier (the walk) is already the right
  dataflow — hoist-off is worse in *both* modes (flagship r4, and my
  serialization probe agrees at small scale).
- Anything that lengthens walk residency to be "gentler" inverts the sign
  (residency dominates intensity). Anything that derates the walk's
  scheduler class risks unbounded starvation under the degraded ambient.
- Remaining honest lever if the door reopens: an E-cluster walk **with a
  starvation guard** (escalate pool QoS when the commit window closes or a
  residency deadline passes) — bg12's stable-window −21 % total suggests
  real headroom, but the guard is beyond a smallest-fix charter and needs
  2^27 certification through the orchestrator.
- Operational note: the 2^27 flagship protocol already encodes the real
  mitigation — cool, idle-settled windows. The bimodality is not a code
  regression; pre-hoist trunk shows the same ambient door (W3D same-window
  control 87.2 s).

## Gates & artifacts

- clippy `-D warnings`: jolt-eval (metal, all-targets), jolt-kernels
  (metal,parallel + parallel), fmt clean.
- `jolt-kernels trace_record` 2/2 (background/inline equality — knobs
  default-off leave the spawn byte-identical); `jolt-eval` 94/94.
- No default-path production behavior change; no protocol surface touched.
- Raw runs: `/tmp/st0-bench1-2to22.jsonl` + transcript tables above.
  Bench count: 2 per decision (reproduce: b1+b2; mechanism: 3a+3b with the
  sanctioned disagreement third 3c; fix: interleaved pair A/B).
- Merge note: `jolt-eval/Cargo.toml` defines `[features] metal` as a
  superset of sibling `scratch/metal-w2-harness`'s — union-merge cleanly.
- Ambient caveat: parts of the matrix ran on battery / mid-charge (logged
  per-run); all A/B claims are same-window interleaved.
