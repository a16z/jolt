# W17-st8: Dory-open re-attribution + two doors KILLED (table, fold pipeline) — no cut ships

Lane branch `lane/metal-w17-st8` @ `51fc968cf` (base `a76d08859`). Verdict:
**attribution delivered (zero 2^27 GPU spend, wave-16 gate trace); both
mandated doors are measured NO-GOs with mechanisms** — reduce-shape Miller
fly→table (D1 half included) and the w5-B parked fold→message chunked
pipeline. Split-fly at production dispatch sizes re-confirmed NO-GO for
free. **Branch tip is byte-identical to trunk (diff 0)** — the two
implementations live in receipt commits `1625610f7` (table) and `9644219a8`
(pipeline) for any future reopening. st8 is near its floor under all known
doors; residual levers identified below sum ~0.2-0.3 s.

## 1. Re-attribution @2^27 (wave-16 gate trace, Aug 27 06:43, 36.02 s wall)

Parsed `metal-saturation/benchmark-runs/perfetto_traces/modular_sha2_chain_27_metal.json`
(D16 precedent — the sanctioned instrumented profile was NOT spent).

**st8 4.795 s** =
- `combine_hints` **0.546** (no sub-spans; the device `jk_g1_combine_rows`
  hook is installed and its gate clears at this size — the span is the
  served wall; device-vs-host split inside it unattributed)
- `DoryScheme::open` **4.196**:
  - **preamble 0.518** = 0.257 untraced host (`compute_evaluation_vectors` +
    `vector_matrix_product` before the VMV) + 0.157 three HOST G1 MSMs
    (t_vec·v 2^18, Γ₁-prefix·v 2^18, e1) + 0.092 G2 fixed-base sweep
    (w5-B's table kernel — was 0.478 at w5)
  - **first messages 1.533** (18 rounds; r0 0.342 with the D2-MSM shortcut
    live, r1 0.537 — full width returns, r2 0.279, halving on)
  - **second messages 1.288** (r0 0.566, r1 0.295, …)
  - **folds 0.856** = apply_first 0.539 + apply_second 0.317
  - final message + verify ≈ 0.001
- stage-8 misc 0.053

Miller mass inside the messages: `miller_fly_device_batch` **2.71 s** over
~1.31 M pairs (9 device rounds, σ=18: n=2^18 → handoff 512) ≈ **2.07
µs/pair** average incl. small-round floors. vs w5 anatomy (open 5.11):
messages −0.53 (w5-B's r0-D2 shortcut + fixed-base table), folds/hints
unchanged — st8's post-w5 improvement came entirely from those two w5 cuts.

## 2. KILLED: reduce-shape Miller fly→table (mandate door 1)

**Design tried** (commit `1625610f7`): D1's G2 side is `g2_vec[..n/2]` — a
CONSTANT setup prefix (only v1/v2 fold; the hooks' g1/g2 never do), covered
by the W5-T2 setup-owned `PreparedG2Table` at every scale (consumed_rows =
2^⌊tv/2⌋ ≥ 2^(σ−1) in both parities). Thread-local publication around
`open()` → one step-major flatten at r0 width (`n_rows` is only a stride,
so every round reads prefix rows through one identity `row_idx`) → mixed
CB: `jk_miller_table` for D1 + fly for D2/C±. Parity was pinned
(table = fly = CPU, identity plants, both m1 shapes).

**Why it dies (three receipts):**
1. **Latency floor.** CB trace @2^24: the table kernel is flat **80-104 ms
   at 128-17 346 threads** — its per-thread serial f-chain (ppt=4: four
   dependent line-folds per shared squaring) is ~4-5× the fly kernel's
   (fly floors at ~20 ms). The commit-shape rates (1.03-1.17 µs/pair,
   M13/M15) transfer only above ~16k concurrent threads — @2^27 that is
   rounds r0-r2 only (60% of miller mass), @2^25 r0-r1, @2^24 nothing.
2. **The fly baseline at reduce geometry is better than modeled.** The
   priors (fly 1.93-1.98 µs/pair) are commit-shape numbers; the reduce's
   big merged dispatches run ~1.3-1.5 µs/pair (r0 m1 @2^27: 262k pairs in
   342 ms with MSM co-run). Thread-gated A/B @2^25: r0 −45 ms, r1 −17 ms,
   r2+ zero — total −59 ms st8, not the −0.4..0.5 s modeled from M13 rates.
3. **The flatten sits on the open critical path**: 0.155 s @2^25 (2^16
   rows; flatten + device copy), ~0.31 s @2^27 — bigger than the wins it
   enables. Off-path variants (async build, skip-r0) shrink the prize to
   ≤ ~0.07 s and add +2.19 GiB transient @2^27. First attempt also
   measured the per-CB wiring trap: re-wrapping the host Vec per round =
   ~100 ms/GiB flat per dispatch (the W13 residency mechanism in per-CB
   form) — fixed by a persistent device buffer, verdict unchanged.

Numbers: @2^24 A/B m1 0.554 vs 0.207 (ungated), @2^25 thread-gated wall
11.63 vs 11.49 (open 2.443 vs 2.334 — flatten-dominated). **@2^27 modeled
net ≈ 0 to negative.** M13's fence ("st8 reduce-shape is a different
regime — no reusable prepared table") now extends to the D1 half with
measurements. The v2 sides (D2, C±) stay fenced by W4's split receipts:
device G2-prep (`jk_miller_fly_lines` 0.77 µs/pair) + table fold ≈ 1.8
µs/pair ≈ the fused fly — a wash with a new-kernel cost.

## 3. KILLED: chunked fold→message pipeline (mandate door 2, w5-B parked 0.5-0.7 s)

**Design tried** (commit `9644219a8`): `apply()` splits big folds into 4
detached per-chunk CBs over paired (lo, hi) output ranges (shaders gained a
byte-neutral `out_offset` param); the next message waits chunk j,
host-normalizes it, dispatches its miller chunk while fold chunk j+1 runs;
cross-MSMs stay hazard-ordered behind the fold writes automatically.
No new kernels; parity pinned (`reduce_rounds_pipelined_match_sync`:
chunked = sync on both messages AND final vectors, two full rounds).

**Why it dies: GPU∥GPU overlap conserves work.** The premise ("fold
latency CAN hide under the following miller") assumed the fold wall was
sync/latency overhead. It is real device ALU (GLV window ladders), and the
miller is device ALU in the same spill band — co-running them on one GPU
cannot shrink the sum, and measured it GROWS: @2^25 apply spans collapse to
~0 but the messages absorb MORE than the folds released —
m2 r0 482 ms pipelined vs 426 ms serial (fold 125 + miller 301); the ratio
worsens as rounds shrink (r3: 172 vs 97+21) because each of the 4 miller
chunk CBs pays the fly latency floor. **Open 2.753 vs 2.334 s (+0.42),
wall 11.95 vs 11.49.** One timed A/B — the per-round structure is the
mechanism, not noise. The only hideable component is the ~1-2 ms/pass CB
sync (w5 §5's ≤50 ms fold-CB fusion residual, already priced dead).
Corollary: any future st8 "overlap" door must pair GPU work with a
NON-GPU resource (host, IO); there is no GPU-side slack inside open.

## 4. Re-confirmed NO-GO (free): split-fly at production dispatch sizes

`JOLT_MILLER_FLY_SPLIT=1` @2^25: m1+m2 1.634 vs 1.567 s fused, wall 11.59
vs 11.49 — W4's 8192-pair NO-GO (+12.7%) holds at 131k-pair dispatches
(+4%). The lines+fold split has no big-dispatch redemption.

## 5. Residual st8 doors (unranked prizes, none reached this lane)

- **Preamble host G1 MSMs 0.157 s @2^27** — three host `JoltG1Routines::msm`
  calls (2^18-scale) inside vendor `create_evaluation_proof`; the device
  SortedMsm machinery does this size in ~15-25 ms. Needs an MSM entry in
  jolt-dory's RoutineHooks (seam exists, fn missing). ~−0.10..0.12 s,
  low risk, small diff.
- **Preamble untraced 0.257 s** — `compute_evaluation_vectors` +
  `vector_matrix_product` host work before the VMV MSMs; structure
  unattributed (may partly be the joint-opening fold engine's host share).
- **combine_hints 0.546 s** — device hook serves, but the span has no
  internal attribution; host flatten/normalize share unknown. A CB-trace
  pass would split it.
- Messages 2.82 s: at the fly band's floor — M13 (no in-kernel headroom),
  W17 (no reusable G2 structure, no overlap slack). Needs a protocol-shape
  or new-algorithm idea, not scheduling.

## 6. Discipline

- **Timed 2^27: 0** (attribution from the existing gate trace). GPU e2e
  runs @2^24/@2^25: 6 chrome-instrumented span diagnostics (2 per decision
  max: table ungated, table gated, pipeline — plus one split-fly toggle and
  one baseline reused across decisions) + 1 warmup @2^22, all under the GPU
  lock, 20-45 s cooldowns, FrBind-class window gate 230-260 µs (<350).
  All cargo under the wave-3 cargo lock. Sibling worktree and
  scratch/metal-saturation untouched; not pushed.
- Parity gates ran green on both receipt commits: metal dory-reduce suites
  (new tests: `reduce_first_message_table_matches_fly` @1625610f7,
  `reduce_rounds_pipelined_match_sync` @9644219a8) + clippy host+metal.
  **Branch tip = trunk byte-identical (`git diff a76d08859` empty)** ⇒
  final-state gates are the wave-16 gate certification by identity; the
  full metal suites / proof-byte ratchet were not re-run on a zero diff.
- KernelId::ALL unchanged (89); no kernels added anywhere on the branch tip.
- Commits: `1625610f7` (table receipt), `9644219a8` (pipeline receipt),
  `51fc968cf` (revert to trunk), + this report.
