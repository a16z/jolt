# Metal wave 5 — lane S5: st5 scan dispatch-context gap

## Verdict

**The 2.9×/2.1× production-vs-fixture scan gap is a DATA-DISTRIBUTION
effect, not dispatch context.** Real traces have ~4.2 distinct scatter keys
per 32-lane tile in the top 8 phases (and 85% per-lane key repetition at
stride 32); with so few keys nearly every lane collides and the wave-3
collision-only SIMD scatter runs its full 32-source shuffle-reduce on
~every tile. The fixture's uniformly-random keys made collisions rare —
the microbench measured the kernel's best case. Host-gap (clock decay) and
buffer-freshness (residency) axes measured **nil in CB GPU windows**;
re-wrapping the nocopy `MTLBuffer`s costs ~13 ms *wall* per phase CB
(blocked−gpu, host-side only).

Fix shipped (`51b18a977` on `lane/metal-w5-st5`): run-length register
accumulation in the phase scan + hoisted/grouped emission in the suffix
scan. **e2e 2^25 paired: st5 scan CB total 2490 → 1876 ms (−24.7%)**;
proofs byte-identical; modeled st5 cut @2^27 ≈ **−2.5 s** (conservative)
to −3.8 s. Bar (≥1.5 s) met.

## Mechanism receipts (isolation curves)

Harness: `jolt-eval` bench `irr_dispatch_context` (permanent; plain-main
sweep, one production-shaped CB per dispatch — scan+its-reduce — timed by
`GPUEndTime − GPUStartTime`). Command:

```
/usr/bin/lockf -k /tmp/jolt-metal-gpu.lock env IRR_CTX_LOG_T=24 \
  cargo bench -p jolt-eval --features metal --bench irr_dispatch_context
# cells: IRR_CTX_CELLS=shape,entropy,gap,fresh,suffix,real
# real rows: JOLT_IRR_DUMP_ROWS=<path> on any e2e run, then IRR_CTX_ROWS_FILE=<path>
```

Phase scan @2^24, wave-3 kernel, CB GPU windows (FrBind probe 339 µs):

| axis | cell | ms | vs 18.2 base |
|---|---|---:|---:|
| shape | s=64 condense=0 | 16.1 | — |
| shape | condense=1 | 18.2 | 1.0× (base) |
| shape | s=120 wide suffix | +0.1 | nil |
| **entropy** | k=1 (tile-uniform path) | 21.5 | 1.18× |
| **entropy** | **k=2 distinct keys** | **72.2** | **4.0×** |
| **entropy** | k=4 / k=8 / k=16 / k=32 / k=64 | 71.4 / 64.1 / 51.3 / 37.8 / 27.9 | 3.9–1.5× |
| gap | idle 10 ms | 16.5 | nil |
| gap | idle/cpu-load 50–400 ms | 28–42, min 17 max 62 | noisy 1.5–2.3×, HIGH variance |
| fresh | re-wrap same pages (gpu / wall) | 16.4 / 29.2 | nil gpu; +13 ms wall |
| fresh | fresh host-written pages (gpu) | 18.0 | nil |

Production's per-phase profile is ±0.5%-tight (lane R trace: P0-7 plateau
127-130 ms @2^25, step down exactly at the s=64 boundary, decline to 83) —
incompatible with the erratic gap effect, exactly matching the entropy
curve. Real-rows stats (fib 2^20 dump): P0-6 d=4.21/tile, 17.5% uniform
tiles, 85.0% lane-repeat; P7 d=5.0; P8-14 d=11.6, repeat 25%; P15 d=15.7.
Real rows in the harness reproduce the production plateau within 10%
(4.42 ms @2^20 = 141 @2^25-equiv vs 128 measured) = 3.9× the random
fixture. Gap fully accounted: entropy × condense.

## Fixes (verdict · numbers · commit · mechanism)

1. **Phase scan run-length accumulation: RETAIN, default-on.**
   `jk_irr_phase_scan` holds per-lane `(key, 3×Fr)` accumulators; equal-key
   runs accumulate in registers, the collision reduce fires only on key
   changes (+ tail flush). Exact field-add regrouping ⇒ bit-identical
   cells. e2e 2^25 fib paired A/B: **phase CBs 2052 → 1466 ms (−28.6%)**;
   per-group P0-6 −52%, P7 −36%, P8-14 −5.7%, P15 0%. Fixture: k=2 −11%,
   pure-random +5.3% (real data always beats it — 25%+ repeat everywhere).
   `51b18a977`. Mechanism: 85% stride-32 key repetition → ~6.7× fewer
   scatter flushes on the plateau.
2. **Suffix scan hoisted+grouped emission: RETAIN, default-on.**
   `jk_irr_suffix_scan` detects the chunk-group structure once per tile
   (s-invariant since zero-value lanes join their group; adding 0 is
   exact) behind a xor-neighbor entropy probe; distinct ≤ 8 → per-group
   masked sums, else the wave-3 emit-trimmed scatter. e2e 2^25 paired:
   **suffix CBs 438 → 410 ms (−6.4%)** — this trace's suffix top phases are
   tile-uniform (10 ms, already fast); the grouped path bit on P7 only
   (43.2 → 17.4, −60%). Real-rows fixture cell: **−48%** (0.90 vs 1.72 ms);
   k=2 −65%. Lane R's 2^27 profile shows 52-65 ms non-uniform suffix
   plateau CBs → expect the larger cut there. High-entropy regression
   +1.5-3.4%. Same commit. Mechanism: detection hoisted out of the
   ≤8-iteration suffix loop + d×masked-sums ≪ 32-source reduce at d≈4.
3. **NOT fixed (mechanism receipts, doors closed):** idle-gap clock decay —
   real but erratic and small at production CB lengths (100-500 ms CBs keep
   clocks up; production plateau tightness excludes it as the driver).
   Buffer freshness/residency — zero GPU-window effect at any axis;
   NOT a door. Re-wrap wall overhead (~13 ms/phase CB @2^25, ~0.6 s
   @2^27 hidden in host-blocked time) — parked: wrap-once-at-build is a
   small host lever, entangled with the scanner borrow structure.
4. **P8-15 phase residual (117 ms @2^25) — parked with mechanism:**
   d≈11.6 + repeat 25% leaves the collision reduce ~24-source; grouped
   masked-sums lose above d≈8 (3d×40 shuffles > colliding×24); no cheap
   lever found. Remaining headroom ≈ 0.7 s @2^27 phase-side.

## Parity + gate

- Proofs **byte-identical @2^21 AND @2^22** (fib, metal backend): new
  kernels ↔ `JOLT_IRR_PHASE_SCAN_EAGER=1 JOLT_IRR_SUFFIX_SCAN_EAGER=1`
  (wave-3 bodies) ↔ CPU scan (`JOLT_METAL_MIN_TERMS_INSTRUCTION_READ_RAF`
  huge); all verify. Kill-switch engagement receipt: CB trace names
  `IrrPhaseScanEager/IrrSuffixScanEager` ×32.
- Fixture oracles assert all three kernel arms (new/eager/legacy) on
  random AND skewed (k=2) keys at 2^22+2^24; slot round-loop scanner tests
  in the metal suite.
- `cargo nextest run -p jolt-kernels -p jolt-dory -p jolt-eval --features
  jolt-kernels/metal,jolt-eval/metal` — **404/404**; clippy host `--all`
  + metal/bench-utils `-D warnings` clean; fmt clean.
- Wave-3 kernel structures intact: collision-only scatter + 2048-group
  suffix schedule retained (eager arms are the wave-3 bodies verbatim);
  legacy (wave-2) arms untouched. `KernelId::ALL` re-counted **79 → 81**
  (trunk was already 79, not 77).

## Modeled st5 @2^27

Against lane R's certified split (phase 8.0 s / suffix 3.8 s CB wall of
st5 17.1 s): phase −28.6% ≈ **−2.29 s**; suffix −6.4% ≈ −0.24 s
conservative (my 2^25 fib's suffix mix is uniform-heavy; lane R's traces
show collision-bound suffix plateaus → up to ~−1.5 s). **Modeled total
−2.5 s conservative / −3.8 s if R's suffix profile takes the grouped
path.** Orchestrator certifies at the wave gate.

## Discipline

- Timed 2^27 runs: **0**. 2^25 e2e: 2 (the paired A/B receipt). Kernel
  iteration at 2^22/2^24 via the harness under both locks; FrBind probe
  339 µs (<350 gate; all decisions on CB GPU windows, paired same-window).
- Diff audited: proof-dump probe reverted; `JOLT_IRR_DUMP_ROWS` retained
  as an env-gated diagnostic feeding the permanent harness (documented
  above, free when unset).
- Not pushed; `scratch/metal-saturation` untouched. Worktree
  `.worktrees/metal-w5-st5` (branch `lane/metal-w5-st5`) ready for merge +
  cleanup after the wave gate.
