# Metal M5 saturation campaign — live journal

Opened 2026-08-04 from `feat/metal` / `88b063db3`. Orchestrator dade2763
(wave 2+). Journal style (user directive 02:10 UTC): lane entries = verdict
+ numbers + commit + one-line mechanism. Waves-1–2 history + wave-1
reporting consolidation: `archive/metal-saturation-waves1-2.md`. Canonical
wave-1 attribution report: `.journals/metal-m5-saturation-report.html`.

## Standing rules

- Byte parity lifted; gate = e2e accept + tamper reject + full battery +
  written soundness argument for protocol changes. Naive shared-challenge
  two-round fusion BANNED (diagonal kernel `Δ=γX(X−Y)`, d≥2).
- Velocity v3: iterate 2^22–2^24, ≤2 timed runs per decision (3rd on
  disagreement), full battery + 2^25/2^27 certification once per wave gate.
- Timed A/B must be same-window interleaved (ambient device power/clock
  moves whole distributions ±5 s — st0 lane proof).
- Single-kernel discipline: harness one kernel via jolt-eval, optimize to
  max, then next. No e2e for kernel iteration.
- Night mandate (01:49): waves run continuously; prepare functions → GPU;
  every-kernel sweep. Kernel-adding merges: re-count `KernelId::ALL`
  (currently `[Self; 72]`).
- All cargo under `/usr/bin/lockf -k /tmp/jolt-metal-wave3-cargo.lock`;
  `gpu_lock()` for timed GPU. No pushing without parent's word.

## Flagship ledger

| point | 2^27 | 2^25 | commit |
|---|---:|---:|---|
| campaign baseline | 71.77 s / 1.870 MHz / RSS 76.87 GiB | 19.67 s / 27.42 GiB | 88b063db3 |
| **wave-2 gate (current)** | **69.63 s / 1.928 MHz / 76.77 GiB** | **19.01 s / 25.16 GiB** | 3830f4da8 |

Wave-2 record stage vector @2^27: st0 17.98 · st1 5.45 · st2 2.98 ·
st3 2.30 · st4 9.52 · st5 14.81 · st6a 0.19 · st6b 7.02 · st7 1.33 ·
st8 7.99. Both fresh 2^27 runs carried ~+6 s st0 ambient penalty vs the
baseline's window; baseline-window model ≈ 64 s.

## Kill list (permanent, with mechanism)

- Global address-major Dory flip: >68× measured @2^22 + sharding inversion.
- st4 cycle-prefix radix-4 on unchanged CSR: measured prefix is
  cycle-domain (wrong variables).
- **st4 address-first restructuring, BOTH arms** (Gate-1): address phase
  6.30 s binary / 7.10 s radix-4 @2^24 = 42× over the 0.15 s kill line;
  radix-4 loses to binary. State algebra sound, arithmetic dominant.
- st0 walk↔commit scheduling fixes: bimodality is ambient device
  power/clock state (reproduced solo); full fix matrix dead/fail-unsafe.
- W2B round-loop rewrites: +52 GiB footprint or in-place rounds +46.8%.
  (Its device PREPARE build is salvage: −64.6% isolated.)
- Generic radix-4/round-pairing outside st4: slots already fuse bind+eval;
  packed challenges are rank-2 weights, illegal in Dory opening points.
- `malloc_zone_pressure_relief` on freed huge regions: no-op.

## Parked doors

- Typed-Dory quaternary factor (st6b/st7 packing): oracle GO w/ conditions
  (ec0b50d07d63); re-price against post-adoption-fix st6b anatomy.
- st0 bg12 E-cluster commit + starvation guard: fail-unsafe without guard;
  needs explicit mandate + 2^27 cert.
- st6b gather residual (7.0 s: jk_ra_gather branch tables, waits, tail
  occupancy) — next lane slot.
- Radix-4 packed round oracle-SOUND (3dbb9c10e48a Q1/Q2/Q4) if a PCS-clean
  cheap-state site appears; Val temporal convention pinned in
  metal-w2-r4gate1.md.
- st1 packing: legal but round loop already fused — prize small.

## Wave 3 (open) — lanes

Doctrine: single-kernel + prepare→GPU. Off `3830f4da8`+.

| lane | task | scope | bar |
|---|---|---|---|
| st4 | 6add005e (sol-xhigh) | W2B prepare salvage (device CSR build, byte-identical, ≤+1 GiB) THEN round-loop wait/scan fusion | prep ≥50% slice; loop ≥0.5 s stage |
| st5 | eb10eb25 (sol-xhigh) | attribute 14.8 s → dominant kernel; ALU/occupancy/layout to max; achieved vs 11.30 Gmont-mul/s roof | ≥12% kernel / ≥0.4 s stage |
| st8 | c32db501 (fable-max) | Miller/pairing dispatch-geometry root-cause vs 4k-8k exposure; occupancy/packing cut; exact pairing parity | ≥10% kernel / ≥0.4 s stage |
| prepsweep | 9972a64a (sol-xhigh) | cross-stage prepare inventory (st4 excluded) + cut biggest (expect st6b IncCR::prepare 1.79 s) | ≥40% slice / ≥0.4 s per cut |

## Results log (terse)

<!-- verdict · numbers · commit · one-line mechanism -->
- **st5 IRR phase-scan: GO, merged.** 2^24 41.12→18.91 ms (−54.0%), modeled st5 −2.68 s @2^27 (14.81→~12.13). `36485d52e` — collision-only SIMD scatter replaces full-width scan. Parity 10/10+10/10.
