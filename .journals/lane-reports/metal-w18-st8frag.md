# W18-st8frag: banked residuals SHIPPED — combine_hints split + NAF, preamble MSMs→device, RLC fold parallelized

Lane branch `lane/metal-w18-st8frag` (base `c334d07de`). Verdict: **GO — all
three mandate doors cut, modeled ≈ −0.44..0.52 s st8 @2^27 (bar 0.3 ✓)**;
byte parity pinned (416/416 metal incl. 5 new/extended parity tests, 20/20
byte-diff ratchet), zero 2^27 runs. ABBA @2^25 on the kill-switchable half
alone: **−0.145 s mean** (pairs −0.16/−0.13).

## 1. Door 1 — combine_hints 0.546 attributed + cut (−0.17..0.21 modeled)

New spans split the span for the first time (@2^24 chrome):
prep ~0 · flatten 0.0055 + normalize 0.0157 (host) · buffers ~0 ·
**kernel 0.1452 (≈ 88-96%)** · readback 0.0003. The `jk_g1_combine_rows`
kernel is ALU-bound (~6.6 Gmul/s — the fold-kernel band), not host-bound.

Two cuts, both landed:
- **Host prep rewrite**: the flatten→nonzero→normalize→points serial passes
  (3 full-size intermediates) became one parallel pass — per-hint disjoint
  output segments, 4096-point chunked `normalize_batch` straight into the
  affine stream. 21 → 5 ms @2^24, 9 ms @2^25; kills two ~100 MB-class
  transients (peak RSS @2^24 12.98 → 12.51 GiB). No switch needed: affine
  coordinates are exact quotients — batching structure cannot change a value.
- **NAF signed digits** (default on): the kernel's shared high-to-low sweep
  now consumes w=2 wNAF digits (density 1/3 vs the canonical ladder's 1/2;
  negation = one `fq_sub` on y before the parity-tested `g1_madd`). Kernel
  **0.1452 → 0.1037 @2^24 (−28.6%)**, right at the −33% add-count model;
  0.138 @2^25. Same group elements (parity test asserts normalized-coordinate
  identity vs CPU). Kill switch `JOLT_METAL_COMBINE_NAF=0` restores the
  canonical ladder (own parity test).

@2^27 model: 0.546 → ~0.34-0.38 (kernel ×0.714 + ~0.03 prep).

**Residual closed with mechanism:** post-NAF the span is ~96% kernel at the
ALU band. Windowed per-row buckets die on bucket-overhead ≥ savings at
≤42-term rows; GLV halves only the amortized doublings (~9% of kernel ops)
against an extra φ-mul per add — both sub-bar. Floor ≈ 0.33-0.38 @2^27.

## 2. Door 2 — preamble host G1 MSMs → device (−0.10..0.12 modeled)

`RoutineHooks` gained the missing `g1_msm` entry (T17's seam);
`JoltG1Routines::msm` consults it, `host_msm_g1` (jolt-kernels) serves it as
one `SortedMsmJob` pass over a zero-copy-wrapped host base buffer. The three
`create_evaluation_proof` MSMs (t_vec·v, Γ₁-prefix·v, e1) are the only
callers that clear the gate in a metal proof (hooks install at stage-8
prepare — after every commit; reduce-tail MSMs are ≤2^12 and decline).

- @2^24 (len 2^14): host 28.9 ms → device 10.7 ms (**2.7×**), fires by
  default (gate = len·2^3 ≥ min-terms, floor `MSM_SORT_MIN` 2^13).
- @2^25 (len 2^15): device 36.5 ms for all three.
- @2^27 (len 2^18/2^17): T17's 0.157 host → ~0.04-0.06 device.

Group-equal by the sorted-MSM window sum; e1 serializes via normalization,
t_vec·v/D2 feed pairings — proof bytes unchanged (same class as the D2
shortcut precedent). New parity test plants identity bases (the nu < sigma
padding shape) + zero/−1 scalars, and pins the undersized decline. Kill
switch `JOLT_METAL_MIN_TERMS_DORY_HOST_MSM=1000000000000` (or
`JOLT_METAL_DISABLE=1`). `SortedMsmJob` bases relaxed to `&'a
DeviceBuffer<'a>` (covariance keeps resident callers unchanged).

## 3. Door 3 — untraced preamble 0.257 attributed + cut (−0.17..0.19 modeled)

Spanned end-to-end: `compute_evaluation_vectors` 0.7 ms @2^24 (tensor
expansion — noise), `vector_matrix_product` = fused device fold CB
(0.027 @2^24 / 0.036 @2^25) + per-slot serves/clones (~1-3 ms) + **the RLC
accumulation** — 42 serial `result[j] += γ_p·fold_p[j]` passes,
~11M Montgomery muls @2^27 ≈ 0.20-0.22 s = the bulk of T17's 0.257.
`RlcSource::fold_rows` now accumulates with a rayon column partition
(per-column op order identical ⇒ bit-exact; no switch): 0.0074 s @2^24,
0.0078 @2^25 across all constituents. Remainder (fold CB + readback +
serves ≈ 0.05-0.10 @2^27) is device ALU + memcpy fragments — sub-bar,
closed.

## 4. Numbers

| | st8 @2^24 (chrome, in-span) | combine | preamble MSMs | vmp |
|---|---|---|---|---|
| lane start* | 0.832 | 0.174 | 0.0289 host | 0.039 |
| final | **0.711** | **0.110** | **0.0103 device** | 0.036 |

*first trace already carried the RLC-parallel; trunk adds ~+0.01-0.02.

Lane ABBA @2^25 (plain, same window, 75 s cooldowns, FrBind 255→248 µs
brackets): kill-switch arm 10.50 / 10.45 vs default 10.34 / 10.32 →
**−0.145 s** (isolates NAF+MSM only; RLC/prep are byte-exact host changes
live in both arms). Walls 10.3-10.5 vs the 11.35 record are window class,
not a claim — the orchestrator gates at 2^27.

**Modeled @2^27 st8: 4.795 → ≈ 4.28-4.36** (−0.44..0.52: door 1
−0.17..0.21 · door 2 −0.10..0.12 · door 3 −0.17..0.19).

## 5. Suggested orchestrator gate measurement

One traced 2^27: expect st8 ≈ 4.3±0.1; check `combine_hints_kernel`
(~0.33-0.38), `dory_host_msm_device` n=3 (~0.04-0.06, confirms the gate
engages at 2^17/2^18), `rlc_accumulate` (~0.03), `combine_hints_normalize`
(~0.03). The spans are permanent — future st8 lanes read the split for free.

## 6. Discipline

- Timed 2^27: **0**. GPU e2e: 3 chrome @2^24 + 1 chrome @2^25 (attribution)
  + 4 plain @2^25 (lane ABBA), all under the GPU lock, 45-90 s cooldowns,
  FrBind gates 255/248 µs. All cargo under the wave-3 cargo lock. Sibling
  lane and scratch/metal-saturation untouched; nothing pushed.
- Parity: 416/416 (`jolt-kernels -p jolt-dory -p jolt-eval` metal; includes
  new `host_msm_matches_cpu_with_identity_bases`,
  `combine_rows_canonical_ladder_matches_cpu`, extended
  `routine_hooks_scope_and_fall_through`) + 20/20 `jolt-prover
  prover-fixtures` byte-diff (covers the CPU-path RLC change). Clippy clean:
  `--all --features host` and the metal/bench-utils targets.
- KernelId::ALL unchanged (92) — one kernel modified (NAF branch), none
  added.
- Kill switches: `JOLT_METAL_COMBINE_NAF=0` ·
  `JOLT_METAL_MIN_TERMS_DORY_HOST_MSM=1000000000000` · `JOLT_METAL_DISABLE`.
- New spans (attribution infrastructure, deliberate):
  `combine_hints_{prep,normalize,buffers,kernel,readback}`,
  `opening_fold_{serve,buffers,kernel,readback}`, `RlcSource::fold_rows`,
  `rlc_constituent_fold`, `rlc_accumulate`, `dory_host_msm_device`,
  `DorySourceAdapter::{compute_evaluation_vectors,vector_matrix_product}`.
