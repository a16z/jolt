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
  (currently `[Self; 77]`).
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
- Typed-Dory quaternary packing (st6b/st7): PRICED NO-GO — likely gain
  ~0.08 s @2^27 (best +0.40, worst −0.45; st7 negligible) vs 18-25
  lane-days protocol/PCS blast radius; shrinking w1/w2/w4/w8 rounds leave
  nothing to halve. Oracle soundness GO stands if geometry ever changes.
- st6b gather residual kernels: measured NO-GO — width-1/2/4 lazy gathers
  are the mass (75/43/31 ms @2^24); row-batching and SIMD-reduction
  prototypes both overlapped baseline. Attribution harness retained.

## Parked doors

- st0 bg12 E-cluster commit + starvation guard: fail-unsafe without guard;
  needs explicit mandate + 2^27 cert.
- Radix-4 packed round oracle-SOUND (3dbb9c10e48a Q1/Q2/Q4) if a PCS-clean
  cheap-state site appears; Val temporal convention pinned in
  metal-w2-r4gate1.md.
- st1 packing: legal but round loop already fused — prize small.

## Wave 4 (open) — lanes

| lane | task | scope | bar |
|---|---|---|---|
| fly | 5d0bb622 (fable-max) | MillerFly persistent-state spill quantification + restructure (planned spills / TG staging / two-kernel split) | ≥15% fly family / ≥0.6 s stage; kill at <8% first A/B |
| dorypack | 441841bc (sol-xhigh; kimi-k3 unavailable — no OPENROUTER_API_KEY) | typed-Dory packing re-pricing vs measured st6b/st7 anatomy — GO/NO-GO for wave 5 | docs deliverable |
| watchdog | 73a18dfb (fable-low) | fresh-window probe (2^22 e2e ≤3.9 s) → wake for absolute record run | — |

## Wave 3 (CLOSED) — lanes

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
- **st4 GPU CSR prepare (W2B salvage): RETAIN, merged.** 2^24 985.11→136.46 ms (−86.1%), modeled ≥1.58 s @2^27, alloc Δ0. Loop fusion NO-GO (−5.7%, JOLT_REGRW_FUSED=1 opt-in probe). `c3222b7ce` — exact legacy CSR built on device. Parity 4/4.
- **st6b IncCR prepare → GPU: RETAIN, merged.** 2^24 465-490→80-83 ms (−83%), modeled −1.48 s @2^27. st5 bucket attempt reverted (0.07 s, below bar). `1a6c6ff58` — jk_inc_prepare split-eq paired weights, detached fill overlapped with host column materialization. Oracle+transcript parity green. Inventory: st3 SpartanShift 0.66 s below bar; sweep complete.
- **st6b gathers: NO-GO, merged (docs+harness).** Lazy 178-185 ms of 258-265 ms deferred-RAV @2^24; both prototypes ≈ baseline. `01751baaa` — gather is already ALU/latency-optimal at these widths. Parity 14/14.
- **st8 Miller partial fold: RETAIN, merged.** Fold 282→15 ms (18.8×), hook @2^17 642→375 ms (−41.6%), modeled st8 −0.6..−1.05 s @2^27. `c05b48ddf` — Rayon-parallel exact associative Fq12 partial product replaces single-thread fold. Doors closed w/ mechanism: CPU co-execution (ark 4-pair re-chunk = 3-4× nondeterministic tax, 20% share REGRESSES), schoolbook fq6_mul (fly pressure is persistent state). Geometry map: thread exposure NOT the bottleneck; max_threads=1024 spills. Priced residual: fly persistent-state restructure (research, 3.4× spill tax on 5.5 s). 297/297.
- **st5 IRR suffix-scan: GO, merged.** 2^24 22.67→14.67 ms (−35.3%) + 2048-group schedule −17.4%; CB mass 92.4→48.3 ms (−47.7%); modeled st5 −1.26 s @2^27 (→~10.87). `52c985059` — suffix scan restructure + group scheduling. Parity 10/10+10/10.
- **st2 RAM-RW: NO-GO, merged (env-gated, default-off).** Prepare only 0.501 s @2^27; GPU CSR −58..−68% but models −0.22..−0.26 s (below −0.3 gate); CB fusion 23→12 flat. `bdc6d5717` — stage too small for the pattern. Parity+pins green.
- **st0 G1 XYZZ mixed-add: RETAIN, merged.** jk_g1_seg_sum 2^22 −51.4%; full commitment @2^24 18.10→16.18 s (−10.6% interleaved); modeled st0 −1.90 s @2^27. `44b0c8174` — 11→10 Fq products per mixed add; achieved 2.52 vs 11.30 Gmul/s roof (headroom noted). Oracle exact, parity 4/4.
- **Gate interlude:** first 2^25 gate run 86.02 s — uniform 3-5× on device stages only; FrBind microbench 1.34 ms vs 255 µs healthy ⇒ degraded device power window after ~3 h sustained GPU load (st0 lane's ambient mode, not a regression; st7/st6a normal). New rule: **cooldown + FrBind health check (<350 µs) before certification runs.**
- **Gate blocked on device window:** wave-2 certified commit (3830f4da8) also measures FrBind 1.34 ms in the current window — code exonerated, GPU pinned in degraded power state after sustained load; 12-min idle did not recover. Certification deferred until health probe <350 µs.

## Wave-3 gate (battery CLOSED green; certification DEFERRED)

Battery on final trunk `95511fa07`+: clippy host+zk clean · muldiv 3/3+3/3 ·
prover-legacy 444/480/445 (default/zk/akita) · verifier 85 · dory 47 ·
tracer 127 · witness 34 · metal suites 353/353.

Wave-3 shipped (modeled @2^27, vs 69.63 s wave-2 flagship):
st5 −3.94 (phase+suffix scans) · st0 −1.90 (XYZZ) · st4 −1.58 (GPU CSR
prepare) · st6b −1.48 (IncCR prepare) · st8 −0.6..−1.05 (parallel fold)
⇒ **−9.5..−10.0 s modeled → ~60 s expected**. NO-GOs with mechanism:
st6b gathers, st2 (both env-gated default-off), st5 buckets, st4 loop
fusion (opt-in probe).

**Certification blocked:** GPU pinned in degraded power window (FrBind
1.33 ms vs 255 µs; uniform ~5× on all device work; wave-2 code equally
slow ⇒ code exonerated; no zombie process, Renderer 1%, recoveryCount 0;
~100 min idle no recovery). Needs reboot/driver reset or spontaneous
window flip. Watchdog continues 30-min probes; certification fires
automatically at the next healthy window (<0.40 ms).

## Wave-3 certification (same-window paired A/B, 08:02–08:40 UTC)

Post-recovery window: FrBind 153 µs (faster than the 255 µs reference) but
sustained e2e much slower than the wave-2 record window and in the LOW-RSS
memory mode — absolute numbers not comparable across windows (identical
wave-2 code: 95.11 s tonight vs its own 69.63 s record). Ambient axis is
worth ±25 s; only same-window pairs are evidence (campaign rule upheld).

**Paired A/B @2^27, same window, back-to-back:**

| tree | prove | padded | RSS |
|---|---:|---:|---:|
| wave-2 certified (3830f4da8) | 95.11 s | 1.411 MHz | 76.93 GiB |
| **wave-3 trunk** | **78.46 s** | **1.711 MHz** | **72.22 GiB** |

**Wave-3 delta: −16.65 s / −17.5% code effect, RSS −4.7 GiB** — exceeds
the −9.5..−10 model (window nonlinearity compounds the wins). st5 legacy-
knob arm (78.67 s) ≈ new arm ⇒ the apparent st5 stage elevation is window
skew, not the new kernels. 2^25 in-window: 20.05 s (st4 −0.47, st6b −0.39
vs baseline; st5/st0 window-skewed).

Wave-2-window-equivalent extrapolation: 69.63 × (78.46/95.11) ≈ **57.4 s
(~2.34 MHz)** — labeled cross-window model; absolute record attempt
deferred to the next fresh window (post-reboot). 69.63 s remains the best
absolute measured. Wave 3 CLOSED.
- **Typed-Dory packing: NO-GO, door closed.** Likely +0.08 s @2^27 vs 18-25 lane-days. `3f96921a3` — shrinking lazy widths leave no round mass to halve; descriptor cost eats the remainder.
- **st8 fly restructure: door PRICED SHUT, merged (harness+lever, defaults unchanged).** Spill map: cliff below one Fq6 (fq6 sqr 2.8× off CIOS roof at 48 u32 live; fly peak ≈430 u32). Split ladder +12.7..18.2% (killed). NEW lever: TG cap 64 → −8.6% @2^13 hook but −2.8% @2^17 ⇒ ~0.19 s, below bar, env-gated JOLT_METAL_PAIRING_TG_CAP. `a18bfb890`. Handoff: jk_miller_table −24% at cap 32 on commit shape. Bundle door: cap-64 + dispatch-merge ≈ 0.4 s. 253/253+47/47.
- **Record attempt in probe-passed window: 77.44 s / 1.733 MHz / RSS 72.24 GiB.** Best absolute tonight (78.46→77.44) but window still ~11% off record-grade (2^22 probe 3.89 s vs ~3.0-3.3 healthy on this code; threshold was too loose). Three windows tonight agree at 77-79 s on wave-3 code vs 95 s wave-2 code — paired A/B certification unchanged. Absolute record deferred to a genuinely fresh window (tighter probe ≤3.4 s; likely post-reboot).
- **st8 dispatch-merge bundle: RETAIN, merged (default-on).** Reduce-message multi-pairs merged 4→1/2→1 into one gated jk_miller_fly dispatch + fly-only cap 64: r0 −10.3%, r5-r8 −45..−87%; measured −0.61 s / modeled −0.9 s st8 @2^27 (w3 had priced 0.2 s). Kill switches JOLT_MILLER_MERGE_DISPATCH=0 / JOLT_METAL_PAIRING_TG_CAP=0. `2c88f6dae`. st0 cap NO-SHIP: in-pipeline inversion (+2% wall — freed occupancy eaten by co-scheduled seg-sum). 255/255+47/47+20/20 byte-diff.

## Wave-4 gate (10:45 UTC)

Battery: kernels+dory+eval 404/404 (incl. new bundle parity + 20/20
byte-diff ratchet); clippy/fmt green in lanes. Shipped: st8 dispatch-merge
bundle (isolated-certified −0.61 s measured / −0.9 s modeled; kill
switches). Doors closed with pricing: fly restructure (spill cliff below
one Fq6), typed-Dory packing (+0.08 s vs 18-25 days), st0 TG cap
(in-pipeline inversion). Bundle-on 2^27 completes cleanly (exit 0; earlier
missing output = transient). E2e delta certification deferred — window
degrades under sustained load (78→95→102 s across the morning on
unchanged code); bundle's e2e effect will show in the next record-window
stage vector (default-on). Wave 4 CLOSED. Record watchdog 6e88f481
probing hourly (2^22 ≤ 3.40 s).
- **Post-daemon-restart (14:29 UTC):** no machine reboot (uptime 23 d). Window probe 3.62 s @2^22 — recovering with idle (3.89-4.20 this morning) but sub-record. Watchdog re-armed as b1ad9731 (hourly, ≤3.40 s).
