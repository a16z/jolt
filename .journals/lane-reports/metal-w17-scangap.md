# Metal wave 17 — lane G17: st5 scan-gap repricing + cut

## Verdict

**PARTIAL — measured −0.13 s wall @2^25 (st5 span −0.16), modeled
−0.5..−0.7 s @2^27; bar (≥1.0 s) missed. The scan branch/emission
machinery is repriced to component level and CLOSED as a campaign door:
after this lane's cut, every remaining term is either at a hardware roof
or killed with receipts — the branches ARE the algorithm at production
entropy.** Commit on `lane/metal-w17-scangap` (base a76d08859). Shipped,
all byte-identical (proof hashes equal across arms @2^21
`bd8587f7…5386` and @2^22 `a21b6d14…fb35`, both verify):

1. **Run-offset early-exit segmented scan (phase flush + suffix), RETAIN**
   — one ballot of run starts replaces the per-step key shuffles and
   bounds the scan by the longest equal-key run; skipped steps performed
   no adds, so the add tree is unchanged (byte-identical a fortiori).
   Kill switch `JOLT_IRR_SCAN_FIXED_STEPS=1` (both w12 fixed-step bodies
   as kernel arms).
2. **Suffix tile pre-gather + per-tile run hoisting, RETAIN** — the tile
   sorts once (chunk keys are suffix-invariant); the sorted-slot suffix
   bits + weight are pulled through the mapping once (3 uint4 shuffles)
   instead of one 2-uint4 Fr gather per suffix, and the run structure +
   tail flags are computed once per tile instead of per suffix. Suffix
   descriptors hoisted out of the row loop. Same values per sorted slot,
   same adds.
3. **Prepare fold (door d), RETAIN, `JOLT_IRR_PREPARE_FOLD=0` kills** —
   `u_evals = eq(r,·)` builds ON DEVICE as dispatch 0 of the phase-0 CB
   (`jk_irr_eq_outer`: hi/lo half-table outer product, exact by
   distributivity ⇒ byte-identical table), and the whole phase-0 CB
   commits DETACHED at prepare; the first round entry collects. The
   stage's remaining prepares (RamRA + RegVal ≈ 0.46 s @2^27) run on the
   host while the CB executes. @2^25 receipts: `InstructionReadRaf::prepare`
   **195.7 → 16.7 ms**; `phase0_wait` < 5 ms (CB fully hidden);
   `IrrScanner::phase_run` 16 → 15 sync CBs.

`KernelId::ALL` 89 → **92** (`jk_irr_phase_scan_fixed`,
`jk_irr_suffix_scan_fixed`, `jk_irr_eq_outer`).

## Machinery repricing (mandate item 1) — the component table

@2^24 REAL sha2-chain rows (`/tmp/irr_rows_sha2_24.bin`, S12's dump),
production shapes, 4096 sgs, gpu-locked, FrBind 256.6 µs. Method: probe
ladder sharing the production body with the flush swapped per rung
(permanent `mach` cell in `irr_roof`; all rungs report maxTG 1024 ⇒ no
register cliff at rest). Numbers are P1 / P12 (ms):

| component (cumulative rung − previous) | P1 | P12 |
|---|---:|---:|
| quiet floor (loads 2.3 + field ALU ≈ chain roof) | 6.60 | 6.60 |
| held-state detect/branch (`none` − quiet) | +0.21 | +0.09 |
| bitonic key sort, 15 steps | +0.18 | +0.19 |
| 3×Fr sorted-slot gathers | +0.41 | +0.41 |
| segmented scan (early-exit / w12 fixed-step) | +1.85 / +2.00 | +1.18 / +1.44 |
| tail cell RMWs | +0.88 | +0.96 |
| = production scan-only (ladder sums 10.13 / 9.43) | 10.11 | 9.39 |

Suffix twin: 5.94 = gathers-bound loads floor 3.59 + MLE/product ALU
0.14 + emission machinery 2.21 (was 2.39). Phase machinery total
2.8–3.5 ms/CB (~33%), suffix 2.2 (~37%) — the @2^27 4.12 s scan mass
carries ≈1.2–1.3 s of machinery, matching STATUS's ~1.4 s gap.

**Why the dominant scan term doesn't compress further:** the hardware
already skips all-lanes-false predicated add steps (branch-on-none), so
the fixed 5-step loop's dead steps cost only their 6 uint4 shuffle_ups —
measured exactly: fixed − early-exit = 0.15/0.26 ms (P1/P12), the
dead-step shuffle mass. The live steps are the Fr adds the reduction
requires at d≈8–18 distinct keys/tile with max-run p90 = 32 (host stats:
18.9% uniform tiles force full-depth scans; sorted mean steps 2.8–4.3).

## Measured (final kernel state, same-window A/B, n=6 each)

| cell | new | fixed arm | Δ |
|---|---:|---:|---:|
| P1 scan-only | 10.111 | 10.236 | −1.2% |
| P8 | 9.725 | 9.883 | −1.6% |
| P12 | 9.407 | 9.623 | −2.2% |
| P15 | 9.673 | 9.855 | −1.8% |
| suffix scan-only | 5.936 | 6.139 | −3.3% |

E2e @2^25 sha2-chain, position-balanced kill-switch ABBA (OFF =
`JOLT_IRR_PREPARE_FOLD=0 JOLT_IRR_SCAN_FIXED_STEPS=1`), 40 s cooldowns,
warmup run discarded: **ON 11.38/11.31 vs OFF 11.54/11.42 ⇒ −0.13 s**
(pairs −0.16/−0.11). Traced pair: Stage5 1320.8 vs 1480.4 (−0.16);
per-CB phase_run 56.9 vs 59.6 ms (−4.5%).

## Modeled @2^27

Components: eq host fill off the wall (P15's 0.33) + phase-0 CB hidden
(0.26 scan + ~0.02 eq + ~0.09 u_evals wire, window RamRA 0.117 + RegVal
0.345 = 0.46 covers it) + kernel cut (−4.5%/CB on the 4.12 s ≈ −0.19)
− the @2^25-observed prepare-contention payback (~0.1, other prepares'
GPU work co-runs with the hidden CB) ⇒ **−0.6..−0.75; measured-anchored
floor −0.13×4 = −0.52. Stated: −0.5..−0.7 s.** st5 model 6.35 → ~5.7.

## Doors closed (receipts)

| door | verdict |
|---|---|
| (a) phase-specialized function-constant variants | **CLOSED, desk + ladder**: dispatch-uniform branches (do_condense/canonical/shifts) live in the held+body rung = 0.09–0.21 ms/CB; the quiet floor they'd attack is at the fr-chain ALU roof (4.3 ms ≈ 50 M muls / 11.6 G). All remaining branches are data-dependent (per-row flag, per-tile votes, per-lane merges) — constants can't remove them; P1 vs P8-14 differ in DATA (run lengths), which the early-exit already adapts to per flush, strictly better than per-dispatch specialization |
| (b) two-pass detect→emit split | **CLOSED, arithmetic**: runs are length 1–2 (59.5% unique indices, repeat 25–42%) so descriptors are per-row — (key, 3×Fr) ≈ 100 B × 16.7 M = 1.67 GB written + read back = 4.3 ms @770 GB/s @2^24, exceeding the entire 2.8–3.5 ms machinery before the emit pass does anything |
| 2-slot held-state LRU (fewer flushes) | **CLOSED, host stats**: 2-slot hit rate 0.26–0.51 vs 1-slot 0.25–0.43 (+1–8 pp); flush events/tile 0.80 → 0.80 UNCHANGED — `simd_any` couples 32 lanes, some lane misses every iteration |
| tail-RMW load prefetch (hide the 0.9 ms RMW term) | **TRIED, REVERTED**: +1.0/+1.35 ms (P1/P12) — holding the 3×Fr old-cell triple across the scan costs more in register pressure/issue than the SLC round trip it hides (`flush=tailpre` rung 11.11/10.78 vs `tail` 10.12/9.43). The w12 "RMWs are free" held only while the 32-turn scatter supplied latency cover |
| uniform-flush butterfly shortcut (~17-19% of events) | **TRIED, REVERTED**: neutral +0.06/+0.07 — saved sort+gathers exactly pay the 2 extra votes per flush + unconditional butterfly adds |

**Floor statement:** post-w17 the scan-CB mass decomposes to loads
(roofed, 770 GB/s) + field ALU (roofed, 11.6 Gmul/s chain) + sort 0.19 +
gathers 0.41 + live-step scan adds + tail RMWs — with presort (P15),
grouped-butterfly/vec4/RMW-batch/width/sgs (S12), and this lane's five
doors all killed with receipts. Remaining in-shape headroom ≲ 0.2 s
@2^27. st5's residual mass is elsewhere: cycle exec + RegVal co-run
(~0.8, at ALU roofs per P15) and the reduce tax (~0.03).

## Gates

- metal suites **414/414** · byte-diff ratchet **20/20 first pass** ·
  `clippy --all --features host --all-targets -D warnings` clean ·
  clippy jolt-kernels/jolt-eval metal+bench-utils clean · fmt applied.
- Scanner parity 5/5 (device vs CPU per-round byte-equal through the
  deferred phase-0 + eq-outer path; handoff fallbacks exercised).
- Fixture oracles: all **5 phase + 5 suffix kernel arms** vs CPU sums on
  random AND skewed keys @2^22 + @2^24 (the `_fixed` arms added to both
  oracle lists).
- Proof bytes identical ON/OFF arms @2^21 + @2^22 (hashes above), all
  runs verify. Fallback contract intact: launch declines ⇒ host eq +
  sync scan (trunk behavior); mid-flight failure ⇒ `Corrupt` ⇒
  `rebuild_u_evals` host rebuild (existing path).

## Discipline

- 2^27: **0 timed, 0 instrumented** (parsed P15's surviving attribution).
- 2^25: 4 ABBA walls + 1 discarded warmup + 2 traced runs (span
  receipts). 2^21/2^22: byte-parity ×2 each. Kernel iteration @2^24 on
  REAL rows via `irr_roof` (gpu+cargo locks throughout; FrBind 256.6 µs;
  2^22 window probe 2.28 s).
- Host row-stats (2-slot LRU, run-length distribution) computed from the
  dump — zero GPU.
- Diff audited: `modular_benchmark` proof-dump probe REVERTED; dead
  probe rungs (tailpre/uni) removed from the rig; the `mach`
  decomposition ladder + `FIXED` A/B lines retained in `irr_roof`
  (bench-utils-gated attribution rig, S12 precedent — flag for PR-handoff
  audit alongside the existing rig).
- Not pushed; `scratch/metal-saturation` and sibling worktree untouched.
  Worktree `.worktrees/metal-w17-scangap` ready for merge + cleanup after
  the wave gate.

## Suggested gate measurements (orchestrator)

- Kill-switch ABBA @2^27 (one binary): default vs
  `JOLT_IRR_PREPARE_FOLD=0 JOLT_IRR_SCAN_FIXED_STEPS=1`. Expect
  −0.5..−0.7 s; st5 6.35 → ~5.7.
- The u_evals wire moved from host page-faults (inside the old eq fill)
  to the detached CB's schedule — RSS-neutral, but the phase-0 CB now
  wires 4.3 GiB at schedule; if the wave-gate profile shows the window
  not covering it, `JOLT_IRR_PREPARE_FOLD=0` restores without touching
  the kernel wins.
