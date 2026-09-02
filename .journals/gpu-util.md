# Metal M5 Max GPU-Utilization Campaign — CLOSED (2026-08-04)

Mandate: eliminate the 0%-GPU windows on the Metal backend at 2^27.
**Status: CLOSED, mandate met** — no zero-GPU windows >1 s remain; every
>1 s GPU-idle host mass eliminated or evidence-closed. Full narrative:
`archive/gpu-util-campaign-full.md`. Lane reports: `lane-reports/w*.md`.
Successor campaign: `metal-saturation.md` (opened off this trunk).

## Final ledger (vs open: 2^27 = 77.168 s / 1.69 MHz; 2^25 = 19.822 s)

- Best certified close: **71.43–71.46 s = 1.88 MHz @2^27** (−7.4%);
  2^25 ≈ 18.0–19.2 s by regime.
- Bigger than the mean: the ±9 s ambient lottery (corpse-pile compressor
  storms) is structurally dead — worst back-to-back mode 78.6 s ≈ the old
  LUCKY case.
- Stage walls open→close: st1 8.0→4.5 · st2 4.3→2.7 · st4 10.4→8.3 ·
  st5 14.7→12.9-13.7 · st6a 2.3→0.15 · st7 2.1→1.2-1.8; st0 hosts the
  hoisted walk (10.8→12.3); st6b storm-proofed.

## Wave verdicts (one line each; details in archive + lane-reports)

- W1A killed (device st6a/st7 ports: SIMD shape inflates work with T).
- W1B retained (BytecodeReadRafCycle port, −0.91 s @2^25; 2^27 cap → later
  uncapped by W3B: the cliff was the storm, not the kernels).
- W1D null + arena deletion (pressure-tier theory falsified with receipts).
- W2A retained — sole protocol change: BooleanityAnchor V1 (stage-1
  anchor), fail-closed axis, load-bearing-tested; st6a −90%.
- W2B rejected (st4 CSR rewrite: fast+fat vs lean+slow; prepare salvage
  later landed in metal-saturation W3).
- W3A retained (MmapVec munmap-backed lifetimes; peak 96.9→79.3 GiB,
  storm impossible; −10.5 s matched-pair @2^27).
- W3B uncap (BRRC device exonerated; ordered-queue waits were storm-inflated).
- W3C retained-at-certification (parallel registers prepare, net −2.97 s
  @2^27 — orchestrator override of the per-stage clause, reasoning journaled).
- W3D retained (TraceRecord::collect hoisted into st0 commit window + eq
  materialization kills; st1+st2 −43%; net −9.5 s in adverse same-window pair).
- W3E no-go (InstrInput q0 device promotion ties the host pipeline).

## Negative-results index (durable — do not re-derive)

1. MADV_FREE_REUSABLE: silent no-op on any range ever wrapped by a no-copy
   MTLBuffer (munmap works, incl. live-wrapped — ordering!).
2. malloc_zone_pressure_relief: no-op on freed huge Vecs.
3. libmalloc never returns multi-GiB frees mid-proof; drop-site ≠ reclaim.
4. Round-pairing: dead — slots already fuse; the one unfused slot's fix was
   fusion, and post-fusion residual sync is noise.
5. Cycle-major booleanity sumcheck: mathematically vacuous (ra²=ra).
6. Per-round stateless gather for one-hot address phases: 10× ALU.
7. Device pushforward scatter (all shapes tried): loses to 12-core CPU.
8. Device q0 for InstrInput: best shape TIES host exact-integer pipeline.
9. Device IncCR prepare: no better than parallel CPU (DRAM-bound). (Note:
   metal-saturation W3 later landed a DIFFERENT IncCR prepare→GPU shape.)
10. "The 2^27 pressure tier" as OS pressure: false on trunk — it was
    working-set shape + (pre-W3A) the corpse-pile storm.

## Parked doors (inherited by metal-saturation)

1. st0 walk↔commit contention — dominant variance (±5 s, bimodal,
   idle-correlated). [metal-saturation later pinned this as ambient
   device power/clock state; scheduling fixes dead — see its kill list.]
2. st4 round-loop fusion under a memory-viable representation (the middle
   between W2B's fast+fat and lean+slow is unexplored).
3. st6b bandwidth tier (device but DRAM-bound; SLC tiling).
4. SpartanShift γ-split (≤0.3 s); st1 claimed_inputs device port (~0.6 s).
5. Co-issue probe, NTZ small-space (inherited from M5 close, still parked).

## Index

- `archive/gpu-util-campaign-full.md` — full journal: mandate + directive
  (byte-parity lifted 2026-08-04), baseline attribution tables, slot
  registry, lane cuts/kill gates, per-wave narratives, certification logs,
  velocity directives, flagship close-out.
- `lane-reports/` — w1a, w1b, w1d(+rootcause), w2a, w3a-rootcause, w3b,
  w3c, w3d(+analysis), w15-roundpair-scope.
- Traces: /tmp/gpu-util-trace-2to27-{wave2,wave3,final}-20260804.json.gz.
- Standing velocity rules (user, 2026-08-04): full battery once per wave;
  ≤2 timed benches per decision (3rd on disagreement); small scales for
  iteration, big for certification. [Carried into metal-saturation rules.]
