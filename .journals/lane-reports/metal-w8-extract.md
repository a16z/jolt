# Metal wave 8 — lane E8: st0 extract_bucket × collect contention

## Verdict

**One RETAINED cut (`d75bcb948` on `lane/metal-w8-extract`): flatten
`BytecodePCMapper` to packed one-word slots.** `get_pc` — two dependent
heap hops through `Vec<Vec<(u16, usize)>>` — was **~220 ns/row and ~95% of
per-row extraction cost on BOTH row walks** (driver `from_row` AND the
record walk's `bundle_at`). Measured on real sha2-chain rows: standalone
`get_pc` 220.5 → 10.5 ns/row, full `from_row` **221.6 → 3.4 ns/row**.
In-pipeline @2^25 chrome pair: **extract_bucket 1.35 → 0.64 s (−53%),
TraceRecord::collect 2.00 → 0.48 s (−76%)**, driver sum −28%, `gpu_run`
identical (3.97/3.96 — device work untouched). Byte-diff 20/20, metal
406/406, jolt-program 38/38 (incl. 4 mapper-semantics oracles), clippy
host + host,zk clean. RSS @2^25 pairs neutral (T 25.25/25.52 vs C
25.60/25.03 GiB). No kill switch possible (representation swap,
value-identical; S0/W3D-F2 untoggleable-restructure precedent — A/B needs
the trunk binary).

## 2^27 attribution FIRST (the early profile) — and a premise correction

One valid 2^27 span profile (chrome + new `JOLT_ST0_TELEMETRY=1`
CPU/wall telemetry), commit 61cb71792 = trunk + telemetry, **record-grade
window** (FrBind 220 µs, machine solo): traced e2e **51.15 s** — 10 s
below the wave-7 record, confirming window grade dominates comparisons.

**st0 = 10.54 s, NOT 17.33.** The wave-7 profile's driver-bound picture
was substantially ambient-window skew (st0 inflates far more than other
stages under sustained-load windows — the known wave-2 "+6 s st0 ambient
penalty" phenomenon; tonight st0 −39% vs st4/st5 −12% against the same
D2 trace):

| st0 component (clean window, old code) | wall | task CPU | eff. threads |
|---|---:|---:|---:|
| extract_bucket | 4.66 | **55.0** | 11.9 |
| build_gpu_job (incl. oh_prefix 0.28, oh_scatter 1.33) | 1.65 | ~13 | 7.1 |
| build_inc_job (inc_count + inc_scatter) | 1.77 | 8.1 | 5.5 |
| send_wait (GPU backpressure) | 2.34 | — | — |
| TraceRecord::collect (co-run, 8-thread pool) | 7.84 | **47.2** | 6.0 |
| GPU lane: gpu_run 9.44 CB-wall, recv_wait 0.86 | | | |
| tier-2 lane: recv_wait 5.07 slack (D2's cuts hold) | | | |

Process CPU during st0: **147.7 CPU-s in a 10.45 s window = 14.1 of 18
cores** — the clean-window st0 is CPU-DEMAND-bound, not scheduling-bound.
Per-row telemetry: extract 452 ns/row CPU (real-row region), sampled
`from_row` = 75% of task time; collect `bundle_at` = 87%, 429 ns/row.
Both walks re-derive per-row facts; a 5-field bundle costing the same as
a 23-field bundle pointed at one shared fixed cost.

**Isolated attribution (extract_microbench, single-thread, real rows):
`MappedPc` = 220.5 of `from_row`'s 221.6 ns/row.** Everything else — the
u128 interleave LookupIndex 2.7, increments+remap 1.7, record-side
operands/output ~4 — is noise. The mapper's `Vec<Vec<_>>`: outer index →
inner Vec header → inner data = two dependent random misses per call, per
row, on both walks, at every stage-0 window.

## What landed (one commit)

Each validated address maps a single descending inline-vsr run at
consecutive bytecode indices (the pairwise decrement rule `try_new`
already enforced), so the whole per-address entry list packs into
`(first_vsr: u16, last_vsr: u16, first_pc: u32)` in one u64;
`get_pc(address, vsr)` = one flat-array load + range check + add.
Identical values for every (address, vsr) by construction; same
`InvalidInlineSequence` errors at build, plus a new
`NonContiguousInlineSequence` reject for interleaved runs the packed form
cannot represent (the entry-list form accepted them silently and no real
bytecode emits them — expander output is contiguous). Bytecode length
guarded at u32 (`BytecodeTooLarge`). Digest/parity unaffected: the
modular mapper's representation feeds no Fiat-Shamir input (the
preprocessing digest hashes legacy-side types) — byte-diff 20/20 is the
empirical receipt.

## Numbers

Microbench (real sha2-chain rows @2^22, single-thread min-of-3, identical
value sinks before/after):

| extractor | trunk | lane |
|---|---:|---:|
| full `CommittedColumnsWitness::from_row` | 221.6 | **3.4** |
| MappedPc (get_pc) standalone | 220.5 | 10.5 |
| LookupIndex (u128 interleave) | 2.7 | 2.8 |
| RdInc+RamInc+RemappedRam | 1.7 | 1.6 |

In-pipeline (telemetry task-CPU, @2^25 / @2^22):

| walk | trunk CPU | lane CPU | per-row |
|---|---:|---:|---:|
| extract_bucket @2^25 | ~13.8 (scaled from 2^27 anchor) | **6.94 s** | 452 → ~210 ns |
| collect @2^25 | ~11.8 (scaled) | **2.54 s** | 429 → 76 ns |
| collect wall @2^25 (8 threads) | 2.00 s | **0.48 s** | |

@2^25 e2e ABBA (two-binary T-C-C-T, 45 s cooldowns, untraced): trunk
15.23/15.53 vs lane 15.32/15.21 — **noise-level, as briefed** (2^25 st0
is GPU-bound; send_wait absorbs the driver cut: +0.62 in the span pair).

## 2^27 model + transfer argument

Transfer is argued from CPU-seconds (window-independent) anchored by the
old-code clean-window 2^27 profile, not from small-scale walls:

- extract CPU: 55.0 → **27.8** (= 6.94 @2^25 × 4; per-row cost is
  geometry-flat — same 1024-row subchunk grain, same per-superchunk count
  tables; the removed component was a per-row pointer chase with no scale
  dependence, and old-code per-row CPU was already scale-flat 452 vs
  ~480 ns across 2^22→2^27).
- collect CPU: 47.2 → **10.2**; collect wall 7.84 → ~1.6-2.0 s.
- st0 CPU demand: ~148 → ~85-90 CPU-s ⇒ the CPU floor (18 cores) drops
  ~8.2 → ~4.9 s and stage-0's known ambient hypersensitivity (the +6 s
  degraded-window st0 penalty = CPU oversubscription under derated
  clocks) loses its mechanism.
- Driver wall: extract ≈ 27.8/12 ≈ 2.3 + builds ≈ 3.4 ⇒ ~5.7 s, well
  under the device mass.
- **st0 @2^27 ≈ max(GPU queue ~9.4-9.7, driver chain) ≈ 9.5-9.9 s.**

Two frames, stated honestly:
- **vs the wave-8 mandate baseline (D2's 17.33 s degraded-window st0):
  modeled −6..−7 s** — that window class is where the mandate's evidence
  lived, and the cut lands 1:1 there (driver 16.6 → ~6).
- **vs the same-night clean window (st0 10.54): modeled −0.6..−1.0 s
  wall**, floored by tier-1+Miller device mass (gpu_run 9.44 CB-wall,
  recv_wait 0.86 — the GPU lane is no longer starved). Gate ABBA decides;
  in record-grade windows the wall cut is the small frame, the CPU cut
  banks against any device-side wins.

## Per-lever verdicts (mandate list)

| lever | verdict | evidence |
|---|---|---|
| extract_bucket work reduction | **RETAIN — get_pc flatten** (the "geometry collapse" was actually a per-row pointer chase on both walks) | tables above |
| contention shaping (collect co-run) | **MOOT post-fix** — collect drops to ~10 CPU-s / ~2 s wall @2^27; the 13.7 s core-stealing co-run no longer exists. Knob re-probe (pool width/QoS) skipped: nothing left to shape (S0's negative result stands for the old shape) |
| host→GPU offload repricing (~9 s idle) | **NO-GO, reframed** — the "52% idle device" was the starved-driver symptom of the degraded window; clean-window device busy ≈ 90% (recv_wait 0.86 s). Miller host-absorb costs ~21 µs/pair vs 2.0 device (D2's own fraction sweep agrees) — no host↔device rebalance pays |
| build_inc_job residue | **PARKED (banked)** — builds ≈ 3.4 s wall / ~21 CPU-s @2^27, under the device floor; cutting it moves nothing until tier-1/Miller shrink |

## Doors this opens (ranked for wave 9)

1. **st0 is device-bound in record-grade windows: tier-1 `G1SegSum` 7.63
   CB-s @2^27** (≈1.9 G gather EC adds; the parked "st0 XYZZ headroom:
   2.52 vs 11.30 Gmul/s roof" door is now the pacing item) + Miller 3.50
   CB-s co-scheduled. Kernel lane, single-kernel discipline.
2. The freed ~60 CPU-s @2^27 banks against ANY device-side st0 win and
   should also derisk degraded-window certs campaign-wide.
3. `RECORD_BACKGROUND_THREADS=8` is now oversized for a ~10 CPU-s walk —
   only worth touching if a future lane needs the cores during st0's
   first ~2 s.

## Discipline

- 2^27 span profiles: **one valid measurement** (61cb71792 + telemetry,
  FrBind 220 µs, solo). A first attempt ~04:57 was **VOIDED — exogenous
  contamination**: the sibling URS-hygiene lane ran `byte_diff` suites
  (two ~830%-CPU processes, load-avg 75) inside my window; e2e 106.6 s,
  st0 41 s. Only structural facts (geometry: row_width 2^18, 512
  superchunks × 256 tasks @2^27; CPU-second shares) were taken from it,
  all cross-checked against the clean run. No timed 2^27 certs.
- Timed 2^25: ABBA ×4 (two-binary, 45 s cooldowns) + chrome pair ×2
  (span attribution, not wall evidence). 2^22: telemetry smoke + sanity.
  FrBind probes: 220 µs / 210 µs (<350 gate). All cargo under the wave-3
  lock; every GPU run under the GPU lock.
- New attribution instrumentation retained (R-lane precedent), all
  env-gated off: `JOLT_ST0_TELEMETRY=1` per-superchunk `[w8-st0]` lines
  (task CPU vs wall vs threads for extract/scatters/inc, sampled
  from_row share, record-walk leaf CPU + bundle share, rusage timeline)
  — commit 61cb71792. `extract_microbench` example (+ jolt-prover libc
  dev-dep) kept for gate re-runs; flagged for deletion at PR handoff.
- KernelId::ALL unchanged (82). commitment.rs 2413 → 2476 (+63,
  telemetry only). No protocol change; host restructure, byte-identical.
- Not pushed; `scratch/metal-saturation` untouched. Worktree
  `.worktrees/metal-w8-extract` (branch `lane/metal-w8-extract` @
  `5d62762bb`; the cut itself is `d75bcb948`) ready for merge + cleanup
  after the wave gate.
