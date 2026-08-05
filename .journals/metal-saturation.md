# Metal M5 saturation campaign — live journal

Opened 2026-08-04 from `feat/metal` / `88b063db3`. Journal style (user
directive): lane entries = verdict + numbers + commit + one-line mechanism.

## STATUS: PAUSED (user, 2026-08-05)

- Waves 1–4 CLOSED; all retained work merged on `scratch/metal-saturation`
  and pushed to PR **a16z/jolt#1733** head `feat/metal` @ `61e5be763`
  (2026-08-05; push gates green: metal release build + kernels/dory/eval
  metal suite 404/404).
- **Absolute-record 2^27 run still PENDING a clean GPU window** (record-
  grade probe 2^22 ≤ 3.40 s; best probe since = 3.62 s, box uptime 23 d —
  likely needs a reboot). Watchdog b1ad9731 (hourly probe) was armed at
  pause time.
- Best absolute measured: **69.63 s @2^27** (wave-2 window). Wave-3 code
  certified **−17.5% paired A/B**; wave-4 bundle default-on ⇒ window-
  equivalent model **~56-58 s**.

## Current state (flagship ledger)

| point | 2^27 | 2^25 | commit |
|---|---:|---:|---|
| campaign baseline | 71.77 s / 1.870 MHz / RSS 76.87 GiB | 19.67 s / 27.42 GiB | 88b063db3 |
| wave-2 gate (best absolute) | 69.63 s / 1.928 MHz / 76.77 GiB | 19.01 s / 25.16 GiB | 3830f4da8 |
| wave-3 trunk, paired A/B | −17.5% vs wave-2 code same-window (78.46 vs 95.11) · RSS −4.7 GiB | 20.05 s in-window | 95511fa07+ |
| **wave-4 trunk (current)** | **~57.4 s window-equivalent model** (+ st8 bundle −0.61 s measured isolated) | — | 61e5be763 |

Wave-2 record stage vector @2^27: st0 17.98 · st1 5.45 · st2 2.98 ·
st3 2.30 · st4 9.52 · st5 14.81 · st6a 0.19 · st6b 7.02 · st7 1.33 ·
st8 7.99 (both fresh runs carried ~+6 s st0 ambient penalty).

## Standing rules

- Byte parity lifted; gate = e2e accept + tamper reject + full battery +
  written soundness argument for protocol changes. Naive shared-challenge
  two-round fusion BANNED (diagonal kernel `Δ=γX(X−Y)`, d≥2).
- Velocity v3: iterate 2^22–2^24, ≤2 timed runs per decision (3rd on
  disagreement), full battery + 2^25/2^27 certification once per wave gate.
- Timed A/B must be same-window interleaved (ambient device power/clock
  moves whole distributions ±5..25 s). Only same-window pairs are evidence.
- Cooldown + FrBind health check (<350 µs; healthy ref 255 µs @2^20)
  before certification runs; record-grade window probe: 2^22 e2e ≤3.40 s.
- Single-kernel discipline: harness one kernel via jolt-eval, optimize to
  max, then next. No e2e for kernel iteration.
- Kernel-adding merges: re-count `KernelId::ALL` (currently `[Self; 77]`).
- All cargo under `/usr/bin/lockf -k /tmp/jolt-metal-wave3-cargo.lock`;
  `gpu_lock()` for timed GPU. No pushing without parent's word.
- Gate battery: clippy host+zk `-D warnings` · muldiv host+zk · prover-
  legacy default/zk/akita · verifier · dory · tracer · witness · metal
  suites (`cargo nextest run -p jolt-kernels -p jolt-dory -p jolt-eval
  --features jolt-kernels/metal,jolt-eval/metal`). Build:
  `cargo build --release -p jolt-prover --example modular_benchmark
  --features prover-fixtures,metal`.

## Kill list (permanent, with mechanism)

- Global address-major Dory flip: >68× measured @2^22 + sharding inversion.
- st4 cycle-prefix radix-4 on unchanged CSR: measured prefix is
  cycle-domain (wrong variables).
- st4 address-first restructuring, BOTH arms (Gate-1): address phase
  6.30 s binary / 7.10 s radix-4 @2^24 = 42× over the 0.15 s kill line;
  radix-4 loses to binary. State algebra sound, arithmetic dominant.
- st0 walk↔commit scheduling fixes: bimodality is ambient device
  power/clock state (reproduced solo); full fix matrix dead/fail-unsafe.
- W2B round-loop rewrites: +52 GiB footprint or in-place rounds +46.8%.
  (Its device PREPARE build was salvaged in wave 3: −86.1%.)
- Generic radix-4/round-pairing outside st4: slots already fuse bind+eval;
  packed challenges are rank-2 weights, illegal in Dory opening points.
- `malloc_zone_pressure_relief` on freed huge regions: no-op.
- Typed-Dory quaternary packing (st6b/st7): PRICED NO-GO — ~0.08 s @2^27
  vs 18-25 lane-days blast radius. Oracle soundness GO stands if geometry
  ever changes.
- st6b gather residual kernels: measured NO-GO — width-1/2/4 lazy gathers
  are the mass; row-batching and SIMD-reduction prototypes both overlapped
  baseline. Attribution harness retained.
- st8 fly persistent-state restructure: PRICED SHUT — spill cliff below
  one Fq6 (fly peak ≈430 u32 live); split ladder +12.7..18.2%.
- st0 TG cap: NO-SHIP — in-pipeline inversion (+2% wall).
- st4 round-loop fusion: NO-GO on top of GPU CSR prepare (−5.7%;
  JOLT_REGRW_FUSED=1 opt-in probe kept).
- st2 RAM-RW device port: below bar (env-gated, default-off).

## Parked doors

- **Absolute record run** — armed, waiting on record-grade window (probe
  ≤3.40 s @2^22; likely post-reboot). First action on resume.
- st0 bg12 E-cluster commit + starvation guard: fail-unsafe without guard;
  needs explicit mandate + 2^27 cert.
- Radix-4 packed round oracle-SOUND (3dbb9c10e48a Q1/Q2/Q4) if a PCS-clean
  cheap-state site appears; Val temporal convention pinned in
  metal-w2-r4gate1.md.
- st1 packing: legal but round loop already fused — prize small.
- st8 jk_miller_table −24% at TG cap 32 on commit shape (fly-lane handoff).
- st0 XYZZ headroom: achieved 2.52 vs 11.30 Gmul/s roof.
- Predecessor campaign's parked doors: see `gpu-util.md` §Parked doors.

## Wave index (verdict one-liners; narrative in archive/)

- **Wave 1** (kernel attribution + first ports): see
  `archive/metal-saturation-waves1-2.md`; canonical attribution report
  `metal-m5-saturation-report.html`.
- **Wave 2** (st6b deferred+fused adoption, st7, st3): record 69.63 s
  @2^27 (−2.14 s vs baseline; st6b 16.3→7.0). Battery green @3830f4da8.
- **Wave 3** (single-kernel: st5 scans, st4 CSR prepare, st6b IncCR
  prepare, st8 parallel fold, st0 XYZZ): certified paired A/B **−16.65 s /
  −17.5%**, RSS −4.7 GiB; battery 353/353 metal. NO-GOs journaled with
  mechanism. `archive/metal-saturation-waves3-4.md`.
- **Wave 4** (st8 dispatch-merge bundle default-on; fly + typed-Dory + st0
  cap doors priced shut): battery 404/404; isolated −0.61 s measured.
  Same archive file.
- Lane detail: `lane-reports/metal-w{2,3,4}-*.md`, `briefs/`.

## Predecessor

GPU-utilization campaign (CLOSED 2026-08-04, mandate met): `gpu-util.md`
(compact) + `archive/gpu-util-campaign-full.md`. Its negative-results
index and parked doors remain binding context for this campaign.
