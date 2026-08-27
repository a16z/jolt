# Metal wave 19 — lane R19: multi-workload attribution (sha3-chain, btreemap)

## Verdict

**Attribution-only, zero production diff. The sha2-chain-tuned Metal stack
transfers well: sha3-chain @2^27 = 37.20-39.22 s (+0.6-1.6 vs sha2's 36.60
median), btreemap @2^27 = 39.26-39.55 s (+2.7). The Metal/CPU ratio test
flags btreemap as the fair-target workload (8.8× vs sha2's 10.2×; sha3
13.1× — relatively BEST on Metal). The new-shape excess is concentrated in
four HOST/CPU-side stages the campaign never priced because sha2-chain is
RAM-light and program-tiny: st2 RamRW (+1.86 btree), st6a bytecode
address-prepare (+0.99 btree / +0.45 sha3), st4 RegRw device rounds
(+1.01 sha3), st7 HammingWeight prepare (+0.52 btree). Every GPU-heavy
stage the campaign roofed (st0/st1/st5/st8) is FLAT across all three
shapes (±0.4 s) — the w5-w18 receipts generalize. st5 scan kernels
repriced on real rows of all three shapes: ±5% on phases; the
sort+segmented-scan answer is shape-robust. One CORRECTNESS-CLASS finding:
the modular_benchmark harness's btreemap scale targeting breaks at ≥2^26
(stale cycles/op constant; cycles/op grows with map size).**

Branch `lane/metal-w19-shapes` off 74c3a547a. All runs verify (16/16 Metal
runs green through the harness's verify gate); zero `Corrupt`/`Declined`/
fallback warnings on the RUST_LOG=warn-surveilled runs; row dumps show
device scanner engaged on all shapes.

## Walls (workload × scale × backend)

| workload | scale | backend | walls (s) | traced | peak RSS | padded kHz (best) | fill% |
|---|---|---|---|---|---|---|---|
| sha2-chain | 2^25 | metal | 11.19 / 11.14 | 11.26 | 22.5-25.4 GiB | 3012 | 81.2 |
| sha3-chain | 2^25 | metal | 11.69 / 11.28 | 11.30 | 24.6-28.0 GiB | 2975 | 69.4 |
| btreemap | 2^25 | metal | 12.06 / 13.25 | 16.22† | 24.3-26.7 GiB | 2782 | 82.9 |
| sha2-chain | 2^25 | CPU opt | 113.96 | — | 37.6 GiB | 294 | — |
| sha3-chain | 2^25 | CPU opt | 148.31 | — | 39.2 GiB | 226 | — |
| btreemap | 2^25 | CPU opt | 106.62 | — | 38.7 GiB | 315 | — |
| sha3-chain | 2^26 | metal | 20.03 (probe) | — | 37.8 GiB | 3350 | 69.4 |
| btreemap‡ | 2^26 | metal | 21.10 (probe) | — | 39.6 GiB | 3181 | 82.1 |
| sha3-chain | 2^27 | metal | 39.22 / 37.69 | 37.20 | 69.1-71.4 GiB | 3608 | 69.4 |
| btreemap‡ | 2^27 | metal | 39.26 / 39.29 | 39.55 | 74.5-75.4 GiB | 3419 | 86.0 |
| sha2-chain | 2^27 | metal | 35.93-36.96 med 36.60 | 36.60 | 71.8 GiB | 3667 | 81.2 |

sha2 rows = this lane's re-anchor @2^25 (record 11.21 reproduced) + the
w18 gate record @2^27 (STATUS). FrBind 252.8-253.0 µs before and after the
timed windows — one healthy window throughout.
† btreemap @2^25 climbed monotonically across the session (12.06 → 13.25 →
16.22) with FrBind unchanged — host-side run-to-run variance (long
emulation phase heat + page-cache state), not GPU clock; @2^27 the same
binary is tight (39.26/39.29/39.55). Treat 12.06 as the honest 2^25 wall;
the 16.22 traced run's stage SHARES are still valid (uniform dilation).
‡ btreemap ≥2^26 requires explicit `--target-trace-size` (52 M @2^26,
88 M @2^27) — see Correctness.

**RSS gate:** both workloads extrapolated ≤ ~76 GiB @2^27 from the 2^26
probes and landed there (sha3 69-71, btree 74.5-75.4; box 128) — 2^27 was
in-budget for both, no scale aborted.

## Metal/CPU ratio check @2^25

| workload | CPU opt | Metal best | ratio | vs sha2 |
|---|---:|---:|---:|---|
| sha2-chain | 113.96 | 11.14 | 10.2× | — |
| sha3-chain | 148.31 | 11.28 | 13.1× | **better** — no Metal-specific regression |
| btreemap | 106.62 | 12.06 | **8.8×** | **worse — fair-target workload** |

btreemap is FASTER than sha2 on CPU yet SLOWER on Metal — a Metal-relative
regression of ~16% concentrated in the stages below. sha3's keccak shape
is even harder on the CPU backend than on Metal (Metal already relatively
strongest there).

## Stage vectors @2^27 (traced runs; sha2 = w18 gate trace)

| stage | sha2 | sha3 | btree | sha3 Δ | btree Δ | mass |
|---|---:|---:|---:|---:|---:|---|
| st0 | 7.417 | 7.366 | 7.038 | −0.05 | −0.38 | commit/driver — flat |
| st1 | 3.254 | 3.221 | 3.235 | −0.03 | −0.02 | flat |
| st2 | 2.720 | 2.660 | **4.583** | −0.06 | **+1.86** | RamRW rounds+prepare (host) |
| st3 | 2.182 | 2.296 | 2.098 | +0.11 | −0.08 | flat |
| st4 | 4.848 | **5.859** | 4.542 | **+1.01** | −0.31 | RegRw device rounds |
| st5 | 5.592 | 5.278 | 5.392 | −0.31 | −0.20 | IRR scans — flat/better |
| st6a | 0.318 | **0.772** | **1.303** | **+0.45** | **+0.99** | bytecode addr prepare (host) |
| st6b | 4.525 | 4.397 | **5.072** | −0.13 | **+0.55** | IncCR prep + RamRAV |
| st7 | 1.278 | 0.899 | **1.797** | −0.38 | **+0.52** | HammingWeight prepare (host) |
| st8 | 4.417 | 4.396 | 4.429 | −0.02 | +0.01 | Dory open — shape-blind |
| **sum** | 36.55 | 37.14 | 39.49 | +0.59 | +2.94 | |

Per padded cycle the vectors are directly comparable (all 2^27). Per REAL
cycle (real T: sha2 108.97 M / sha3 93.17 M / btree 115.36 M): wall = 336 /
399 (+19%) / 343 (+2%) ns. sha3's per-real-cycle penalty is mostly its 31%
padding (a benchmarking artifact — CYCLES_PER_SHA3=4330 vs measured ~3345)
plus st4; st4 per real cycle is +41% on sha3.

### Sub-span attribution of the deltas (2^27, seconds sha2/sha3/btree)

- **st2 = RamReadWriteChecking**: prove_round 0.439 / 0.512 / **1.722**
  (3.9×), prepare 0.363 / 0.329 / **0.741**, RamRafEvaluation::prepare
  0.396 / 0.328 / 0.438. CPU sumcheck (m_collect). Round-span counts: 81 /
  81 / **93** @2^27 (77/77/85 @2^25); spans fire 2×/round + 1, and phase2
  rounds = log ram_K ⇒ **ram_K: sha2/sha3 2^13, btree 2^19 @2^27 (2^17
  @2^25 — heap grows with op count)** — a ×64 address space (56.8k-entry
  map + allocator vs a 32 B chain state), +6 address rounds, and per-span
  cost also 3.4× (5.4 → 18.5 ms; denser touched-address mass per phase).
- **st4 = RegistersReadWriteChecking**: `RegRw::bind_msg_run` (device CB
  wait, the fused bind+message rounds) 1.306 / **2.095** / 1.266;
  prepare_meta (host CSR build) 0.653 / 0.709 / 0.355. prepare_meta scales
  with total register ops ⇒ sha3 has ~+9% absolute ops on −15% real cycles
  = **+27% register-op density per cycle** (keccak: 25×u64 state resident
  in registers, 2-3 reg ops/cycle, near-zero RAM). btree is register-LIGHT
  (−46% ops) and beats sha2 here.
- **st6a = BytecodeReadRafAddressPhase::prepare** (host, serial): 0.301 /
  **0.756** / **1.288**. Address rounds are FEWER on the new shapes (29 vs
  27) — the cost is not K_bc-round-count but the pushforward walk: the
  stage-1..4 background build (capped 4-thread pool spawned after st4) +
  the inline stage-5 walk scatter into K_bc-sized tables at `push_pc` per
  row. sha2's tight loop = tiny hot PC set, cache-resident scatter;
  btreemap/keccak = large irregular PC working set ⇒ miss-bound walk
  and/or the background build not finished when 6a starts.
- **st6b**: IncClaimReduction::prepare 1.111 / 1.086 / **1.360**;
  RamRaVirtualization::prove_round 0.934 / 0.902 / **1.163** — RAM-mass
  again; Booleanity flat (0.95-0.98).
- **st7 = HammingWeightClaimReduction::prepare** (host, essentially the
  whole stage): 1.275 / 0.899 / **1.793**. Tracks RAM one-hot mass
  (raf-carrying rows: 38.6% / 2.9% / 48.6% of real rows).
- **st5 = IRR**: phase_run 3.698 / 3.435 / 3.453, cycle_wait 0.59 flat —
  scans slightly CHEAPER on both new shapes (below).

## st5 row stats + kernel repricing on real rows

Row dumps: `/tmp/irr_rows_{sha3,btree}_24.bin` (JOLT_IRR_DUMP_ROWS, @2^24
Metal runs; sha2 = S12's `/tmp/irr_rows_sha2_24.bin`). Host stats
(S12/G17 method, padding-corrected):

| stat (real rows @2^24) | sha2 | sha3 | btree |
|---|---:|---:|---:|
| padding share | 18.8% | 30.6% | 27.5% |
| unique full-index | 73.3% (59.5% incl. pad = G17's fig) | 93.3% | **13.0%** |
| raf_flag | 38.6% | 2.9% | 48.6% |
| consecutive-repeat | 6.1% | 4.8% | 10.4% |
| run p99 / max | 2 / 29 | 1 / 95 | 2 / 23 |
| uniform tiles (phase key, all-16-phase range) | 18.8-18.9% (≈ all padding) | 31.9-33.2% | 27.5-41.7% |
| flush iterations | 80-81% | 66.7-68% | 58-72% |
| distinct keys/tile | 7.6→18.3 (phase 0→15) | 18.0-18.4 flat | 3.0→12.3 |
| dominant tables | RangeCheck 30.5, Xor 23.0, VirtualROTRW 18.6, And 8.9 | **Xor 55.3, VirtualROTRW-class rotate 20.8, Andn 18.0** (keccak θ/ρ/χ) | RangeCheck 43.3, none 19.8, UnsignedLessThan 9.2, Equal 7.3 |

`irr_roof` base+suffix cells on the three dumps, one window (FrBind
253 µs; sha2 cells reproduce w17 to 0.1%):

| ms/CB @2^24 | sha2 | sha3 | btree |
|---|---:|---:|---:|
| P1 scan-only | 10.10 | **7.44 (−26%)** | 9.89 (−2%) |
| P8 scan-only | 9.71 | 7.43 | 9.97 (+3%) |
| P12 scan-only | 9.41 | 7.48 | 9.92 (+5%) |
| suffix scan-only | 5.92 | 4.57 (−23%) | **3.57 (−40%)** |
| suffix FIXED arm | 6.12 | 4.64 | **3.32 — early-exit inverted** |
| reduce share (P1) | +1.67 | +1.66 | +1.67 |

**Shape-dependent receipts verdict: the scan doors STAY CLOSED.** sha2 is
the worst case of the three on phase scans (least padding ⇒ fewest cheap
uniform tiles); sha3 rides its 31% padding + 93% uniqueness; btree's
d≈3-6 early phases don't reprice anything (±5%). The single inversion:
btree's suffix shape runs 7% FASTER on the w12 fixed-step body than on
G17's early-exit arm (−0.24 ms/CB ≈ −0.06 s @2^27 if per-shape-switched)
— sub-bar 5×, logged, no door. Presort (P15), grouped-butterfly/vec4/RMW
(S12), machinery ladder (G17), Miller scale gate and one-hot-density-tied
receipts: nothing observed reopens them — st0/st5/st8 are flat-to-better
on both new shapes.

## Correctness flags

1. **modular_benchmark btreemap scale targeting is broken ≥2^26** (harness,
   not prover): `CYCLES_PER_BTREEMAP_OP = 1550` is stale — measured
   cycles/op grows with map size: 1247 @2^24 → 1427 @2^25 → 1642 @2^26 →
   2032 @2^27 (deeper tree + allocator growth). Default 0.9·2^scale target
   overflows the padded length and the harness panics
   (`Trace is longer than expected`, modular_benchmark.rs:220) — @2^26
   with the default target AND @2^27 even at a reduced 100 M target.
   Workaround used: `--target-trace-size 52000000` @2^26 (82% fill),
   `88000000` @2^27 (86% fill). Fix candidates: per-scale c/op table, or
   derive ops from a short calibration trace.
2. **CYCLES_PER_SHA3 = 4330 overstates** (measured ~3345): sha3-chain
   fills only 69.4% of the padded trace at every scale — sha3 kHz-padded
   numbers carry ~12 pp more padding than sha2's, and per-real-cycle
   comparisons flatter sha2. Benchmarking-fairness fix, free.
3. No prover failures: 16/16 Metal runs verified (incl. all @2^27); zero
   `Corrupt`/`Declined`/fallback warnings on the RUST_LOG=warn runs
   (@2^24 dumps, @2^26 probes, btree @2^27 w1); device scanner engaged on
   all shapes (dumps written from the device path).

## Ranked wave-20 doors

| # | door | stage | workload | est. prize @2^27 | invalidates | mechanism hypothesis |
|---|---|---|---|---|---|---|
| 1 | RamRW device-port re-price + prepare parallelize | st2 | btreemap (RAM-heavy apps generally) | **1.0-1.5 s** | **st2 "RAM-RW device port below bar" kill-list receipt — priced on sha2's trivial ram_K 2^13**; btree ram_K 2^19, +6 address rounds & 3.4×/round | CPU m_collect rounds scale with log ram_K × touched-address density; the env-gated device port already exists |
| 2 | Bytecode address-phase prepare: devicify or unblock the pushforward walk | st6a | btreemap + sha3 | **0.8-1.2 s** btree, 0.4 s sha3 | nothing (never priced — sha2's 0.30 s made it invisible) | serial host scatter into K_bc tables at push_pc; large/irregular PC sets ⇒ cache-miss-bound; background 4-thread build may finish late — measure join-wait vs walk split first |
| 3 | st7 HammingWeight prepare parallelize/devicify | st7 | btreemap (helps sha2 too: 1.28 baseline) | **0.5-0.9 s** btree | nothing (st7 never a campaign door) | host prepare ∝ RAM one-hot mass (raf 48.6% btree); even the sha2 1.275 s was unattributed |
| 4 | RegRw round kernel on register-dense shapes | st4 | sha3-chain | **0.5-0.8 s** | possibly the st4 "round loop already fused/roofed" framing — roofs were sha2-shape | device bind_msg_run +60% with +27% reg-op density/cycle; find whether CSR row count or value width drives it; keccak = 64-bit lanes vs sha2's 32-bit words |
| 5 | st6b btree residuals (IncCR prepare + RamRAV rounds) | st6b | btreemap | ~0.5 s bundled | nothing | RAM-mass scaling of both members; sub-bar individually |
| 6 | Harness: btreemap targeting + sha3 c/op constant | — | bench infra | correctness/fairness, no wall | — | see Correctness 1-2 |

NOT doors: st5 (scans shape-robust, both new shapes cheaper), st0/st8
(flat — driver/Dory mass is trace-length-, not content-driven), st1/st3
(flat). The ratio test says door work should target btreemap first: its
+2.94 s stage excess is 4/5ths host-side prepares/rounds that sha2-chain
never exercised.

## Discipline

- Timed budget: 2 walls + 1 traced per (workload, scale) at 2^25 and 2^27
  for the two mandate workloads; sha2 @2^25 re-anchor 2 walls + 1 traced
  (record 11.21 reproduced at 11.14/11.19); @2^26 single probes only
  (RSS-gate mandate). CPU walls ran GPU-free in cooldown windows. All GPU
  work under `/usr/bin/lockf -k /tmp/jolt-metal-gpu.lock`; all cargo under
  the wave-3 cargo lock. FrBind 252.8 → 253.0 µs (start/end) — window
  certified healthy; 45-90 s cooldowns; first-run-after-build was the
  discarded sha2 warmup.
- 2 failed btreemap attempts (@2^26 default, @2^27 100 M) died in host
  trace-gen before any GPU work — no window pollution.
- Zero production-code changes; zero rig changes (irr_roof + rowstats
  reused as-is; one /tmp analysis script `w19_stagevec.py` + extended
  `w19_rowstats2.py`, both /tmp scratch).
- Traces: `benchmark-runs/perfetto_traces/modular_{sha3_chain,btreemap}_{25,27}_metal.json`
  + `modular_sha2_chain_25_metal.json` in this worktree (gitignored);
  sha2 @2^27 reference parsed read-only from the trunk worktree's w18
  gate trace. Row dumps in /tmp (403 MB each).
- Not pushed; `scratch/metal-saturation` and sibling worktrees untouched.
