# Akita Metal T=2^28: what 10 MHz requires

Date: 2026-09-02. Status: assessment, not a campaign contract.

Machine: MacBook Pro (Mac16,6), Apple M4 Max, 40-core GPU, 128 GB unified memory.
Source: Jolt `feat/akita-metal` @ `8a5f238f4`, Akita fork @ `0e52ebf` (the Cargo pin).
Evaluator: `modular_benchmark --backend metal --scale 28 [--format chrome]`, BTreeMap
with `--target-trace-size 150000000`, release build made from the clean tree, no other
load, runs sequential with 20 s gaps in the order SHA-2, Fibonacci, BTreeMap, SHA-2.
The score is `jolt_prover::prove` wall with `PROOF_VERIFIED ... value=true`.

## 1. Where the prover stands

| Workload (populated rows) | prove | MHz | commit (GPU cmd) | PIOP | eval proof | peak RSS | vs frozen CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTreeMap (253.8M) | 35.79 s | 7.50 | 13.02 (12.44) | 16.76 | 5.99 | 88.4 GiB | 4.65x (166.5 s) |
| Fibonacci (201.3M) | 38.87 s | 6.91 | 16.44 (15.96) | 16.55 | 5.84 | 81.7 GiB | 5.54x (215.2 s) |
| SHA-2 chain (217.8M) | failed at HEAD; 43.21 s after the fix below | 6.21 | 18.51 (18.00) | 18.09 | 6.58 | — | 4.95x (213.7 s) |

SHA-2 after the fix: Stage 1 5.24, Stage 5 3.15, Stage 6a 2.66 (the log_K=14 bytecode CPU
island), Stage 6b 4.09; commit reads 2.14 TB for 3.59 B hot entries.

10 MHz at T=2^28 is 26.84 s. Required cuts: BTreeMap 9.0 s (1.33x), Fibonacci 12.0 s
(1.45x), SHA-2 16.4 s (1.61x). Against the frozen CPU references
that is 6.2x–8.0x. "5x over CPU" is 6.2–8 MHz depending on workload, so the two targets
are not the same: 5x is within reach of the current design; 10 MHz is not.

Every earlier T=2^28 trace in `benchmark-runs/perfetto_traces` from Aug 30 21:50–22:05
overlapped a release build and is contaminated (BTreeMap Stage 1 at 11.9 s versus 4.7 s
clean, Fibonacci 46.2 s versus 38.9 s clean, the CPU Fibonacci run at 328 s versus a
215 s reference). Do not use those numbers.

### SHA-2 at HEAD did not prove (fixed in section 5a)

Three independent runs panicked at the same point with byte-identical claim values:

```
modular Akita prove: Verifier(StageClaimSumcheckFailed { stage: "Stage6a",
  reason: "prover final claim Fp128([12815858001508052155, 12105152042941471717])
  disagrees with the expected output fold Fp128([6809849655387688973, 15578148094295021073])" })
```

SHA-2 at T=2^25 verifies on the same binary, so the defect is in the T=2^28-only route.
The shape is the `log_K = 14` bytecode address hybrid (`MetalBytecodeReadRafAddress::route
fallback_reason="address_domain"`, realized on CPU). The Aug 28 SHA-2 run at `62320b9f0`
verified, so the regression entered with `8a5f238f4` (PIOP kernel integration) or the
Akita pin move `8291c2d -> 0e52ebf`. Nothing about SHA-2 performance can be claimed until
this is bisected and fixed; the deterministic values make it a plain bisect.

## 2. Commit: measured model

Three facts fix the commit cost at T=2^28.

**Density sweep on the packed D512 root kernel** (Akita bench `packed_onehot_commit`,
T=2^28, 30 live columns, 253.8M populated rows, uniform density, metal-only, one warm
plus one sample; note the bench geometry is 524,288 positions per block versus the
production 262,144):

| density | hot entries | GPU ms | matrix read |
|---:|---:|---:|---:|
| 13% | 0.990 B | 7,702 | 2.00 TB |
| 27% | 2.056 B | 12,694 | 2.00 TB |
| 54% | 4.111 B | 21,011 | 2.00 TB |

GPU time = 3.5 s + 4.26 ns × hot entries (residual under 4%). The intercept is the
matrix-tile streaming term (2.0 TB at roughly 575 GB/s, served from SLC because concurrent
streams read the same tiles); the slope is the accumulate loop. Production BTreeMap has
6.15 ns per hot entry all-in; the uniform bench has 6.17 ns at the same hot count.
Column skew therefore costs nothing: the earlier hypothesis that dense bytecode columns
stall barrier-coupled SIMD groups is falsified, and the reverted Aug 19 "schedule by
column" experiment was right to be reverted.

**Per-column census of committed hot entries** (instrumented streaming commit, reverted
after measurement; layout is 16 instruction chunks, 8 increment chunks, increment MSB,
2 bytecode chunks, 2–3 RAM chunks):

| Workload | hot entries | per populated row | instruction | increment (+MSB) | bytecode | RAM |
|---|---:|---:|---:|---:|---:|---:|
| BTreeMap | 2.023 B | 8.03 | 47.0% (3.78/row) | 19.9% (1.60) | 24.9% (2.00) | 8.2% (0.66) |
| Fibonacci | 3.264 B | 16.21 | 50.9% (8.26/row) | 36.7% (5.96) | 12.3% (2.00) | 0% |
| SHA-2 chain | 3.594 B | 16.50 | 56.5% (9.33/row) | 31.0% (5.11) | 12.1% (2.00) | 0.4% |

The model predicts the production commits: BTreeMap 3.5 + 8.6 = 12.1 s (measured 12.44),
Fibonacci 2.8 + 13.9 = 16.7 s (15.96), SHA-2 3.0 + 15.3 = 18.3 s (18.6). The Aug 23
analysis used 2.23 B hot entries for SHA-2; the real count is 3.59 B, which is why SHA-2's
commit is the largest of the three.

**Per-add cost.** Each hot entry adds one negacyclically shifted 512-coefficient fp128
matrix row. Per 32-bit word the kernel spends six scalar ops (two adds, two compares, an
or, a select) plus one threadgroup gather; per coefficient that is roughly 35–40 ops and
4 gathers, so roughly 600 SIMD instructions per hot entry across the two coefficient
bands. At the M4 Max's issue rate that is about 2.7 ns; the measured 4.26 ns is about 65%
of issue peak, and the 8 KB of matrix gathered per hot entry (16.5 TB per BTreeMap commit)
puts the on-chip traffic floor at roughly 2.3–4.6 s by itself. The accumulate loop is
within 1.3–1.5x of what this formulation can do on this GPU; the ledger's closed
micro-variants (radix-26, RNS, carry-save, sign quadrants, two tasks per SIMD group) are
consistent with that.

Protocol-preserving commit floor, this design: roughly BTreeMap 9.5 s, Fibonacci 12.5 s,
SHA-2 14.5 s (hide most of the streaming term, widen the CPU hybrid share beyond its
8-block cap). The only large lever is fewer committed nonzero bytes per row, and 82–88% of
those bytes are values encoded one-hot: instruction operand chunks and increment digits.

## 3. PIOP and eval proof

Stage walls (clean runs, seconds):

| Stage | BTreeMap | Fibonacci | dominant spans |
|---|---:|---:|---|
| 1 | 4.67 | 3.75 | uniskip handoff 1.35–1.68 wall vs 0.63–0.72 GPU; OuterRemainder prepare 0.89–1.33 (17 GB init); output claims 0.80–0.84 wall vs 0.26–0.33 GPU |
| 2 | 1.49 | 0.57 | |
| 3 | 0.88 | 0.77 | |
| 4 | 0.87 | 0.70 | |
| 5 | 3.28 | 2.99 | scatter prefetch join 0.88–0.90; 156 rounds 2.05–2.24 |
| 6a | 0.61 | 0.56 | |
| 6b | 4.52 | 6.84 | prepares 0.46 / 2.58 (Fibonacci: bytecode plan 0.87, RAM RA 0.69, instruction RA 1.02); accelerator-lane rounds 0–5 ≈ 3.4–3.5 |
| 7 | 0.44 | 0.36 | |

Preparation, handoff and join spans (not sumcheck rounds) total about 6.1 s on BTreeMap
and 7.3 s on Fibonacci, 37–44% of PIOP. The 17 GB Outer initialization at 0.9–1.3 s is
page-fault bound (about 1 µs per 16 KiB page), not bandwidth bound (40 ms at 412 GiB/s);
the same mechanism drives the other prepares. The K001–K010 kernel campaign made every
scored kernel at least 5x faster than its CPU twin, so the round work is no longer the
main PIOP gap; residency and first-touch are.

Eval proof is 5.84–5.99 s: opening command 4.28–4.53 s wall against 2.83–2.85 s GPU
active, opening-index build 0.85–0.89 s, seven serialized ring relations. The
protocol-preserving floor is about 4.5 s.

## 4. Budget arithmetic

Required total is 26.84 s. Current and protocol-preserving floors:

| Workload | commit now / floor | PIOP now / floor | eval now / floor | sum of floors | MHz at floors |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 13.0 / 9.5 | 16.8 / 10 | 6.0 / 4.5 | 24.0 | 11.2 |
| Fibonacci | 16.4 / 12.5 | 16.5 / 10 | 5.8 / 4.5 | 27.0 | 9.9 |
| SHA-2 chain | 18.6 / 14.5 | ~19 / 10 | ~6 / 4.5 | 29.0 | 9.3 |

The PIOP floor of 10 s assumes every preparation span is removed and rounds stay as they
are; it is the optimistic end. Reaching 10 MHz under the current protocol requires
hitting every floor at once on BTreeMap and is arithmetically out of reach on Fibonacci
and SHA-2. That matches the ledger's own conclusion that the protocol-preserving queue no
longer crosses even 5x on BTreeMap without a new multi-second mechanism.

## 5. Attack strategy

Tier 0, prerequisites for any claim:

- Bisect and fix the SHA-2 T=2^28 Stage 6a failure (`62320b9f0` passes, `8a5f238f4` fails).
- Score only interleaved, order-reversed pairs with the laptop cool; the ledger recorded
  70–80% per-work-unit swings by run order on this machine. A 10 MHz headline should be
  taken on a desktop part or with recorded GPU frequency.
- Keep RSS under 90 GiB; BTreeMap is at 88.4 GiB and any new owner must retire another.

Tier 1, protocol-preserving, worth about 4–6 s per workload (BTreeMap to roughly 30 s,
9 MHz; Fibonacci roughly 33 s; SHA-2 roughly 38 s once fixed):

1. Commit tile streaming overlap. The 3.5–4.5 s streaming term is serialized with the
   accumulate loop by the two barriers per tile. Stage the next tile in registers during
   the accumulate loop (threadgroup memory is already at the 32 KiB cap, so a second
   resident tile is not possible). Falsifier: the bench intercept must fall below 1.5 s at
   unchanged slope, with exact parity against the CPU commit; reject if the register cost
   drops occupancy enough to raise the slope. Not in the ledger's closed list.
2. Widen the CPU hybrid share of the root (cap is 8 blocks of 385–477; CPU is 8–12% of
   GPU throughput) only where the CPU is otherwise idle during commit; expected 0.5–1 s.
3. PIOP residency plan: one prefaulted arena sized to the T=2^28 lifetime plan, handed
   out by stage so no owner pays first touch, plus early retirement of the 17.6 GB opening
   index. This is a whole-lifetime change, not the per-buffer primers and storage-mode
   flips the ledger closed. Target the six prepare spans above; expected 3–4 s.
4. Stage 6b Booleanity accelerator lane (rounds 0–2 are 2.6–2.7 s): this member was not
   in the K001–K010 campaign. Apply the same matched-kernel process; expected about 1 s.
5. Eval proof host gaps (1.4–1.7 s of command wall beyond GPU active) and index build:
   pipeline the seven ring relations' host work under the previous relation's GPU work;
   expected 1–1.5 s.

Tier 2, protocol changes that move the bound (required for Fibonacci and SHA-2):

6. Fewer committed nonzero bytes per row. Instruction chunks are 47–57% of hot entries and
   increment digits 20–37%; both encode values one-hot at 512 fp128 adds per nonzero byte.
   Halving them (16-bit chunks, or digit-valued increments consumed linearly by the fused
   increment relation) saves 5–7 s of commit on Fibonacci and SHA-2 but breaks the K=256
   packed layout and the 2^41 opening domain the eval proof depends on. The bounded
   radix-4 study rejected the all-members dense-root version because the dense NTT root
   measured 11.2 G Montgomery products/s; a per-family hybrid (dense NTT only for the
   increment digits, one-hot elsewhere) needs that kernel at 4x its measured rate to break
   even at Fibonacci's density. Write the floor first; do not reopen without it.
7. Batch the seven eval-proof ring relations under one transcript challenge; about 1.5–2 s.

Tier 3, hardware: commit and the GPU-active PIOP fraction scale with GPU cores and
bandwidth. On an Ultra-class part with twice the GPU, Fibonacci lands near 27–29 s
before any software change; that is the cheapest route to a 10 MHz number, and it does
not change the software strategy above.

## 5a. Follow-up measurements (2026-09-02, later the same day)

Commit tile-streaming overlap (Tier 1 item 1) is falsified. Two software-pipelined
variants of `akita_packed_onehot_commit_fp128_d512_panels` were measured on the bench at
T=2^28, 30 columns, 253.8M rows, one warm plus one sample, GPU command time:

| variant | 13% density | 27% density | T25 CPU/Metal parity |
|---|---:|---:|---|
| baseline kernel | 7,702 ms | 12,694 ms | — |
| register-staged next tile plus next lane byte | 8,827 ms (+15%) | 13,606 ms (+7%) | exact |
| next lane byte only | 8,364 ms (+9%) | 13,236 ms (+4%) | — |

Both are slower at both densities; the pre-registered bar (intercept below 1.5 s at an
unchanged slope) is missed and the shader is restored. The streaming term is not a
latency gap that prefetch can hide. The commit kernel is at the floor of its formulation
on this GPU; only fewer hot entries or a different accumulate representation move it.

SHA-2 T=2^28 localization so far (all on HEAD, same binary family):

- The optimized CPU backend verifies SHA-2 at T=2^28. The defect is Metal-backend specific.
- Forcing the Booleanity address phase, or the bytecode read-RAF address phase, to its CPU
  route inside the Metal backend leaves the failing final claim and expected fold
  byte-identical. Neither Stage 6a member kernel is the cause; the corruption is in data
  or claims produced upstream by Metal stages.
- The failure tracks populated row count, not padded T: `--target-trace-size 134000000`
  (about 121M rows) and 148M (about 133M rows) verify; 150M (about 135M rows) and 200M
  fail; T=2^27 (108.9M rows) verifies.

**Root cause (found with a per-stage claim dump of both backends at the 150M shape).**
Stage 4 is self-consistent on both backends, but the Metal backend exports different
`rs1_ra` and `rs2_ra` opening values than the CPU. They come from the Stage-4 register
read-write member's device operand-claim reduction. `rs2_ra` is derived as
`(combined - gamma * rs1) / gamma^2`, so the stage's own fold, which only sees the
gamma-combination, still passes; the wrong pair then poisons Stage 6a's bytecode
input claim. The registers evaluator reproduces it in isolation: CPU checksum
`ea64db14ba7e7aad`, Metal with the packed source `ea64db14ba7e7aad`, Metal with the
Stage-1 source `3074c3ad589937d4`, round polynomials identical. A read-back of the
device compact rs1 index plane shows it holds the same raw register indices as the host
plane, but `solinas_registers_read_write_compact_rs1_claim` then applied `register_map`
(raw to compact dense-state slot) when `stage1_source && remap_registers`. The opening
`rs1_ra = sum_t eq(r_cycle, t) eq(r_address, rs1(t))` ranges over the real register
domain, so the remap must not enter it. `remap_registers` turns on only when the trace
touches a register index of 64 or more (`active_register_mask >> 64 != 0`); SHA-2 first
touches such a virtual register after about 2^27 cycles, Fibonacci and BTreeMap never
do, which is why only SHA-2 above 2^27 rows failed and why the "2^27" boundary is a
property of the SHA-2 trace, not of the kernel geometry. The compact-claim route is
T=2^28-only (`COMPACT_RS1_SOURCE_LOG_T_MIN = 28`), which is why T=2^27 passed.

**Fix.** Drop the remap from the compact rs1 claim kernel and its `register_map`
binding, and remove the now-unused `remap_indices` parameter. Regression check: the
registers evaluator at the failing production shape must match the CPU checksum,

```bash
./target/release/examples/metal_registers_read_write_cpu_eval --name sha2-chain \
  --scale 28 --target-trace-size 150000000 --samples 1 --arm metal --metal-source stage1
```

A small unit test is not yet possible: the synthetic `structured_fixture` (which does
touch registers 100 and 127) is rejected by the Stage-1 producer with
`InvalidBooleanityRow`, so a Stage-1-source fixture that uses virtual registers at a
small `log_t` is a follow-up.

## 6. Recommended order

1. Fix SHA-2 (Tier 0). 2. Commit tile-streaming overlap on the bench, then integrate
(item 1). 3. Residency plan (item 3), then Booleanity (item 4) and eval gaps (item 5).
4. Re-measure the matrix in interleaved pairs; expect BTreeMap near 9 MHz and 5x cleared
on all three. 5. Decide on Tier 2 with a written floor for the hybrid dense-increment root;
without it, 10 MHz on Fibonacci and SHA-2 on the M4 Max is not a software target.

## 7. Plan to 10 MHz

### 7.1 Why the commit is the decisive block

Commit GPU time is `S + 4.26 ns * H`, with `S` the matrix-tile streaming term
(2.0–2.6 TB per commit, 3.5–4.5 s) and `H` the hot entries (2.02 B / 3.26 B / 3.59 B).
Both terms scale with the same product: `S` with `domain * n_a * D * log q`, the
accumulate with `H * n_a * D * log q / 32` u32 operations. Two conclusions follow.

- Wider one-hot chunks (K = 2^16) are dead in this packing. They halve `H` but
  multiply the domain by 256, and `S` is proportional to the domain
  (about 2^47 coefficients would stream roughly 200 TB). This is why K stays at 256.
- The only knob that shrinks both terms at once is `n_a * D * log q`, the accumulator
  bit-volume per hot entry, currently 512 coefficients of 128 bits. The planner's
  D64/128/256 rows kept that volume (or the traffic) constant and correctly found no
  gain. The volume is set by the SIS instance's coefficient norm bound, which in the
  bench setup is `coeff_linf_bound = 65,535` for the inner (root) matrix, derived from
  the fold/challenge schedule, not from the honest 0/1 witness. Whether that bound can
  be tightened (shallower fold, smaller challenge norms, exact per-level range proofs)
  is the analysis that decides 10 MHz on Fibonacci and SHA-2.

### 7.2 Budget arithmetic

Per-workload savings required and the priced levers (seconds; commit levers scale with
each workload's `H`):

| lever | class | BTreeMap | Fibonacci | SHA-2 | evidence |
|---|---|---:|---:|---:|---|
| PIOP prefaulted arena (six prepare spans) | preserving | 3–4 | 3–4 | 3–4 | prepares total 6.1/7.3 s; page-fault bound |
| Stage-6b Booleanity kernel campaign | preserving | ~1 | ~1 | ~1 | rounds 0–2 = 2.6–2.7 s, not in K001–K010 |
| eval-proof host gaps + relation batching | minor protocol | 1.5–2 | 1.5–2 | 1.5–2 | 1.4–1.7 s gaps; seven serialized relations |
| SHA-2 log_K=14 address phase on GPU | preserving | 0 | 0 | 1.3 | Stage 6a island |
| increment as dense radix-8 digits (successor path) | protocol | 0.7 | 2 | 1.7 | inc = 20/37/31% of `H`; D64 digit rows run 132 G pairs/s |
| halve `n_a * D * log q` (norm/schedule) | params + soundness | 4.3 | 7 | 7.7 | accumulate term 8.6/13.9/15.3 s |
| **sum without the norm lever** | | 6.7–7.7 | 7.5–9 | 9.5–11 | leaves 28–29 / 30–31 / 32–34 s (9.2 / 8.9 / 7.7 MHz) |
| **sum with the norm lever** | | 11–12 | 14.5–16 | 17–19 | reaches 24–25 / 23–24 / 24–26 s |

Required: BTreeMap 9.0 s, Fibonacci 10–12 s, SHA-2 16.4 s. Without the norm lever
BTreeMap lands near 9 MHz and the others below; with it, all three clear 26.8 s with
1–3 s of margin, which is thin against the 46–59 s same-binary spread the ledger saw on
this laptop. A 2x-GPU desktop part is the alternative that makes the margin
comfortable without the soundness work.

### 7.3 Analyses to run before code (order matters)

1. **SIS/schedule sizing study (Akita, 1–2 weeks, decides everything).** Using
   `akita-sis-estimator` and the planner on the T=2^28 one-hot instance (2^41
   positions, 30 live columns): tabulate the required `(n_a, D, log q)` for inner
   coefficient bounds {65,535, 4,095, 255, 1} and fold depths {3, 2, 1}, with the
   extractor slack written out per level. Output: attainable `n_a * D * log q`, the
   implied commit time via the measured model, the eval-proof relation count, proof
   bytes, and verifier cost. Gate: a bound that halves the volume with a written
   soundness argument; otherwise record that 10 MHz on Fibonacci/SHA-2 is hardware-only.
2. **Increment encoding.** Cost the two increment polynomials as dense radix-8 digit
   vectors committed through the existing D64 successor machinery versus the nine
   one-hot members, including the digit range argument and the change to the fused
   increment relation (`delta = sum 8^j d_j`), transcript, and verifier. Gate: at least
   1.5 s on Fibonacci with no new soundness assumption.
3. **T28 residency plan.** Extend `akita-metal-t28-memory-lifetime.md` into an
   allocation-lifetime table and design one prefaulted arena with per-stage bump
   allocation. Falsifier: the six prepare spans total under 1.5 s at T28 with no
   displacement into later stages and RSS at most 90 GiB.
4. **Eval-proof batching.** Soundness accounting for folding the seven ring relations
   under one transcript challenge plus pipelining host preparation under the previous
   relation's GPU work.
5. **Booleanity address/cycle kernel.** Run the K-campaign method (matched CPU service,
   floor first) on the Stage-6b accelerator lane.

### 7.3a Study results (2026-09-02, `benchmark-runs/akita-10mhz-studies/analysis.md`)

- **S1 (SIS/schedule sizing) closes the norm lever.** The planner's honest fold policy for
  the T=2^28 one-hot root selects 4 response digits (bound 1,048,575); at that bound the
  audited tables give the same ~65 Kbit of accumulation per hot entry for every admissible
  geometry (Q128 D512 r1, Q64 D512 r2, Q64 D1024 r1; D256 needs rank 2). D256 rank 1 would
  need a 2-digit response, i.e. on the order of 1000x more fold response windows. Dense
  digit encodings inherit the same bound. Row 6 of the table above is worth 0 s.
- **S2 (page faults, measured):** first GPU touch 41 GB/s vs 411 GB/s refill (24 ms/GB),
  private buffers identical, CPU first touch 43 ms/GB, prefaulted host arena still 9 ms/GB
  on first GPU touch. Arena lever is 1.5-2.5 s, not 3-4.
- **S3 (eval, trace):** one root fold (2.0-2.4 s) plus 14 small per-level sumchecks (1.5 s)
  and ring switches (0.6 s); batching the levels bounds the saving at 1.5-2 s.
- **S4 (PIOP rounds, trace):** Stage 6b is a flat 143 ms/round (4.0 s, unscored lane);
  Stage 5's 156 rounds cost 13-14 ms each with almost no GPU time (CPU tail lever,
  0.5-0.8 s); prepares and rounds split about evenly.

Revised software total: 5-8 s (btree, fib), 6.3-9.3 s (sha2), landing at 8.7-9.6 / 7.9-8.7 /
7.3-7.9 MHz. 10 MHz on all three at T=2^28 is not reachable in software on this machine
with the current commitment construction.

### 7.4 Implementation order

Arena (3) and Booleanity (5) first: preserving, independently measurable, and they
create the margin the protocol levers need. Then eval batching (4) and the increment
encoding (2). The norm/schedule change (1) last, since it changes proof bytes and the
verifier and needs cryptographic review; if its study fails the gate, stop the
software campaign at roughly 9 MHz and move the 10 MHz target to a 2x-GPU part.

### 7.5 Campaign result (2026-09-03)

The protocol-preserving campaign reached the hard 5x target on the final exact binary in
both workload orders. Each workload had an unscored warm-up; every scored proof verified,
peak RSS stayed at or below 88.72 GiB, `/usr/bin/time -l` reported zero process swaps, and
system swap did not grow.

| workload | frozen CPU | order A | order B | speedup range | throughput range |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 166.548 s | 32.06 s | 32.03 s | 5.1949-5.1998x | 8.373-8.381 MHz |
| Fibonacci | 215.177 s | 33.52 s | 33.28 s | 6.4194-6.4657x | 8.008-8.066 MHz |
| SHA-2 chain | 213.703 s | 36.50 s | 35.73 s | 5.8549-5.9811x | 7.354-7.513 MHz |

Order A was BTreeMap, Fibonacci, SHA-2; order B reversed it. BTreeMap used the frozen
`--target-trace-size 150000000` workload in both samples. Its best result, 32.03 s, misses
the 29.8262 s / 9 MHz stretch wall, so the stretch target is not claimed.

Retained: L1a concurrent accelerator members, L1c's Booleanity leading-term simplification
and dead-table deletion, and L6a's SHA-2 `log_K=14` resident-radix bytecode address route.
Their paired mean complete-proof changes were -0.27 s, -0.56 s, and -1.735 s respectively;
L6a also reduced its charged SHA-2 bytecode prepare from 2.203 s to 0.562 s. L1c's local
round-0 timing did not confirm its proposed mechanism, so its retention rests on the two
complete-proof pairs rather than an attributed round-level saving.
Rejected after measurement: L1b queues, L1d width-8 Booleanity materialization, L2a's RAF
threadgroup change, L2b's cache-local dual scan, L4a's deferred shift projection, L4b's
retained heap, and L4c's detached transient heaps. L3's apparent per-round host overhead was
localized to Metal scheduling/page wiring and is closed with the exhausted L4 lifetime
family. L5 eval-proof batching was not attempted because it changes the transcript and
prover/verifier protocol; it remains a separate human-reviewed study.

The final binary is
`f0b8a6d759fca01c812d16f40a4481b37a595d88bbc48ef45fc8941fae26a9b6`; its retained
source diff is `8c9b9d86c8180a5f9f28f2a572103abd608a93f76e17abc9dd74f7e56bc51839`.
With the SIS sizing lever closed by S1 and every scoped preserving lever retained or
falsified, the measured software ceiling has been reached without widening into protocol
redesign.

**CPU references re-measured (2026-09-03).** The Aug 23 references above were single samples on
older CPU-tier code (`optimized/spartan_outer.rs` has changed by about 900 lines since). Re-run
on the final binary with the frozen commands, idle machine, sequential, all verified:

| workload | CPU 2026-09-03 | CPU MHz | superseded Aug 23 | other samples | final Metal speedup |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 180.29 s | 1.489 | 166.548 s | 179.68 s (Aug 30) | 5.62-5.63x |
| Fibonacci | 196.76 s | 1.364 | 215.177 s | 328 s (Aug 30, contaminated) | 5.87-5.91x |
| SHA-2 chain | 211.18 s | 1.271 | 213.703 s | 175.66 s (Sep 2) | 5.79-5.91x |

The 5x gate still passes in both orders against the new references (walls 36.06 / 39.35 /
42.24 s). The CPU tier has never measured above 1.7 MHz at any scale in `benchmark-runs/results`
(BTreeMap T26-T27); at T=2^28 its samples span 1.25-1.53 MHz. Traces for the new runs replaced
`akita_{btreemap,fibonacci,sha2_chain}_28_optimized.json`; the Aug 23 SHA-2 trace is kept as
`akita_sha2_chain_28_optimized_frozen_20260823.json` and the BTreeMap one as
`akita_btreemap_28_optimized_pre_k003_4d4c7e9a.json`.

**CPU references re-measured again after the accumulator fix (2026-09-03, afternoon).** With
`perf(akita): route D512 K=256 CPU commits through the deferred accumulator` (09e649061) the CPU
tier is 15-20% faster and the acceptance math changes:

| workload | CPU, fixed binary | CPU MHz | stage 0 before / after | final Metal speedup | 5x wall |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 155.11 s | 1.731 | 78.2 / 44.8 s | 4.84x | 31.02 s |
| Fibonacci | 165.05 s | 1.626 | 101.8 / 67.4 s | 4.92-4.96x | 33.01 s |
| SHA-2 chain | 170.32 s | 1.576 | 109.7 / 69.3 s | 4.67-4.77x | 34.06 s |

Against the fixed CPU the final Metal matrix (32.0 / 33.3-33.5 / 35.7-36.5 s) no longer clears
5x on any workload; the shortfall is 1.0 / 0.3-0.5 / 1.7-2.4 s. The Metal walls themselves are
unchanged. Section 9 gives the levers that close it.

## 8. Goal-mode prompt (software campaign to the measured ceiling)

```text
Create a persistent goal with this objective:

Execute the software campaign in specs/akita-metal-10mhz-attack-strategy.md sections 7.3a
and 7.4: on Jolt feat/akita-metal (worktree /Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt)
with Akita worktree /Users/mgeorghiades/worktrees/akita-metal-eval-proof at 0e52ebf, raise the
verified Metal prover at T=2^28 as far as the priced protocol-preserving levers allow, with a
hard acceptance of at least 5x over the frozen optimized-CPU references for BTreeMap,
Fibonacci, and SHA-2 chain, and a stretch of 9 MHz on BTreeMap. Do not target 10 MHz: study
S1 in benchmark-runs/akita-10mhz-studies/analysis.md shows the commit is at the audited SIS
table floor for the current fold design, and section 7.3a bounds the software total at
5-9 s per workload.

Read first: the whole specification, benchmark-runs/akita-10mhz-studies/analysis.md,
specs/akita-metal-protocol-preserving-5x-ledger.md (closed candidates), both repositories'
AGENTS.md, and the CLAUDE.md lint and test rules. Treat the specification and the study
ledger as canonical over live workspace state.

Step 0, before any optimization: the compact-rs1 operand-claim fix is uncommitted in
crates/jolt-kernels/src/metal/solinas/registers_read_write/{fused_sequence.metal,sequence.rs}.
Confirm the registers evaluator parity
(./target/release/examples/metal_registers_read_write_cpu_eval --name sha2-chain --scale 28
--target-trace-size 150000000 --samples 1 --arm metal --metal-source stage1 must print
checksum ea64db14ba7e7aad, equal to --arm cpu), then commit it alone with a message that
names the root cause (register remap applied to the raw compact plane when a trace touches
a register index >= 64). Do not push.

Fixed evaluator. Build once per source change:
  cargo build --release -p jolt-prover --example modular_benchmark --features prover-fixtures,metal
Score the reported jolt_prover::prove wall with PROOF_VERIFIED backend=metal value=true:
  ./target/release/examples/modular_benchmark --name fibonacci --scale 28 --backend metal
  ./target/release/examples/modular_benchmark --name sha2-chain --scale 28 --backend metal
  ./target/release/examples/modular_benchmark --name btreemap --scale 28 --target-trace-size 150000000 --backend metal
Frozen CPU references (re-measured 2026-09-03 with the CPU accumulator fix): BTreeMap 155.11 s, Fibonacci 165.05 s, SHA-2 170.32 s. Do not rerun
them. Frozen Metal parents from 2026-09-02: BTreeMap 35.79 s, Fibonacci 36.67-38.87 s, SHA-2
43.21 s. Never run a scored proof while a build or another proof is running; the Aug 30
traces show how badly overlap corrupts results. Repeat a scored run only when a result lies
within 0.3 s of its gate. Peak RSS must stay at or below 90 GiB with no swap growth.

Candidate rules. One lever at a time, in this order unless a measured result changes the
ranking: (1) Stage-6b Booleanity accelerator lane (4.0 s flat, 143 ms/round) using the
K001-K010 matched-service method with a written floor before code; (2) Stage-5 instruction
read-RAF address rounds CPU tail for small suffix tables (129 rounds at 13-14 ms), predicted
0.5-0.8 s; (3) per-round host overhead in Stages 1, 3 and 4 (20-32 ms/round with little
recorded GPU time), profile per round before changing anything; (4) cross-proof prefaulted
arena for the six prepare spans, admitted only with an allocation-lifetime table and a
falsifier of prepares under 3 s total without displacement into later stages, predicted
1.5-2.5 s; (5) eval-proof batching of the seven levels' stage-1/stage-2 sumchecks and ring
switches under one transcript challenge, predicted 1.5-2 s, only with a written soundness
delta and prover/verifier changed together, recorded in specs/akita-metal-protocol-changes.md;
(6) SHA-2 log_K=14 bytecode address phase on the GPU without the fused Stage-1 topology,
predicted 1.3 s on SHA-2 only.

For every candidate, before code: one mechanism, the exact charged boundary, a lower bound,
a predicted complete-proof saving, and one numerical falsifier. Then the smallest red
parity or route test, one scoped edit, focused tests, one T=2^25 sentinel, and one T=2^28
treatment on the affected workload. Retain only a verified complete-proof improvement of at
least 0.20 s with exact CPU/Metal parity and no fallback; otherwise revert exactly and log
the negative result. Log every candidate append-only in
benchmark-runs/akita-10mhz-studies/events.jsonl and analysis.md with parent and candidate
digests, command, result, and keep | discard | inconclusive.

Closed, do not reopen: commit ring dimension, modulus profile, dense-digit or radix-4
encodings, tile prefetch or software pipelining in the D512 panels kernel, column-major
task scheduling, root carry/sign/RNS/radix micro-variants, the CPU hybrid commit share,
wider one-hot chunks, page primers or storage-mode flips as standalone candidates, and
every candidate the protocol-preserving ledger marks closed.

Correctness and hygiene: cargo nextest, never cargo test; cargo fmt; both clippy modes from
CLAUDE.md for touched crates. cargo clippy -p jolt-kernels --features metal,test-utils
--all-targets has two pre-existing errors at sequence.rs near the test module and
optimized/spartan_outer.rs:3855; separate those from candidate diagnostics and fix them if
the touched file is in scope. Preserve unrelated changes and untracked study tools in the
Akita worktree. Do not push.

Completion: all three workloads verified at or above 5x in two order-reversed pairs each,
the retained levers documented with their measured deltas, the strategy spec's section 7
updated with the final matrix, and a closing note stating the measured ceiling reached
and which levers were rejected. If the campaign stalls with all levers closed, stop and
report; do not widen scope into protocol redesign, which is a separate human-reviewed
study.
```

## 9. Fundamental analysis: what can still move the Metal wall (2026-09-03)

Sources: the three T28 traces per backend (section 7.5 matrix, the 2026-09-03 CPU references),
the Metal D512 panels kernel (`akita-metal/src/kernels/onehot.metal:4100-4370`,
`packed_onehot_fp128_d512.rs`, `runtime.rs:6022-6240`), the planner re-query in
`benchmark-runs/akita-10mhz-studies/analysis.md` ("CPU speed investigation" and the geometry
re-query), and the Akita specs on subring packing, selective-L2 sizing and the SIS tables.

### 9.1 Where the wall is and what it is made of

Final Metal walls 32.0 / 33.3 / 35.7-36.5 s (BTreeMap / Fibonacci / SHA-2) against a 26.84 s
target. Commit 41-50%, PIOP 34-42%, eval proof 16-17%. Metal is 5.6-5.9x the re-measured CPU
(before the accumulator fix in 9.3; after it the CPU commit drops about 35% and the ratio
falls toward 4.7-5x). The CPU runs 15-16 of 16 cores; its speed does not enter the Metal wall.

The commit is the serialized sum of two measured terms:

- streaming `S`: matrix tiles filled into 32 KiB threadgroup memory, 2.06 TB per commit at
  the 575 GB/s SLC-served rate = 3.5 s. Derivation from the kernel: every threadgroup
  (one 256-coefficient output band of 32 tasks) must stream the full `n_a * D * 16` byte row
  of every position in its block because a negacyclic rotation mixes all input coefficients
  into every output band. Threadgroups = `C * T * K / (D * P * 32) * (n_a * D / 256)`, bytes
  each = `P * n_a * D * 16`, so `S = C * T * K * n_a^2 * D * 16 / 8192` = 2.06 TB at
  D512 rank 1, independent of positions per block `P`.
- accumulate `A = 4.26 ns * H`: 510-600 SIMD instructions and an 8 KiB threadgroup gather per
  hot entry, calibrated at 131 G coefficient-adds/s, 58-68% of issue peak. Both floors
  (issue and threadgroup-memory port) are the same order, so the kernel is within 1.3-1.5x of
  what this formulation can do; the closed micro-variants (radix-26, RNS, carry-save, sign
  quadrants, matrix reuse, tile prefetch) all measured <= 2% or regressed.

Overlap of `S` and `A` is measured-falsified (register-staged prefetch +7 to +15%); the
consistent mechanism is that tile fills and gathers share the core's L1 SRAM port.

### 9.2 The currency is accumulate volume, and the planner never priced it

Every term above scales with the accumulator volume per hot entry, `n_a * D * log q`
(65,536 bits at D512 rank 1). Study S1 concluded the volume was invariant across geometries;
that conclusion was an artefact of the S1 planner query constraining `inner_output_rank` to
{1, 2}. Re-running `find_schedule_with_root_constraint` at the ranks the SIS table demands
(planner rev 0e52ebf17, nv = 41) shows two admissible geometries at 49,152 bits, 25% less:

| root | positions/block | rank | bound | fold digits | volume (bits) | payload | admissible |
|---|---:|---:|---:|---:|---:|---:|---|
| D512 (production) | 2^18 | 1 | 1,048,575 | 4 | 65,536 | 76,138 B | yes |
| D128 | 2^19 or 2^20 | 3 | 1,048,575 | 4 | 49,152 | 76,138 B | yes |
| D64 | 2^20 | 6 | 1,048,575 | 4 | 49,152 | 76,138 B | yes |
| D256 | any | 1 | needs <= 16,383 | 2 | 32,768 | | no: cap needs ~64x more response windows |
| D64 / D128 at lower rank | | | | | 40,960 | | no schedule |

Production is D512 rank 1 for two reasons unrelated to security: Jolt's catalog generator
hard-codes `METAL_ROOT_DIMS = (512, 64, 64)` and `inner_output_rank: Some(1)`
(`crates/jolt-akita/src/schedules/mod.rs:164-199`, added with the Aug 19-22 Metal series), and
Akita's own planner objective is setup size then payload, never commit volume
(`specs/subring-coefficient-packing.md`, Decision). The Aug 12 CPU configuration (D64, rank 7,
2^20 positions) was the pre-packing D64 catalog and is still table-admissible.

Same SIS policy, same buckets, same proof bytes: this is a public-parameter change (new
schedule row and digests, CPU and Metal catalogs regenerated together), not a new assumption.

### 9.3 What a 49,152-bit root does to each backend

CPU: no streaming term; per hot entry it gathers `n_a * D * 16` bytes into accumulators.
The Aug 12 D64 x7 trace measured 21 vs 30 ns per hot entry, but the cause was not the
geometry: `jolt-akita/src/trace_onehot.rs:1249` gated the fast `DeferredFp128Ring` path
(18 B per coefficient, L1-resident) to D <= 256, so D512 fell onto the generic
`WideCyclotomicRing<Fp128x8i32, 512>` accumulator (32 B per coefficient, a 16 KB
read-modify-write per hot entry on an L2-resident 464 KB set plus a 16 KB limb conversion per
visited position). A `deferred_rows_per_ring_k256` branch through the existing deferred ring,
in column groups of 8 so eight 9.2 KB accumulators stay in L1, measures 16.6-19.1 ns per hot
entry (Fibonacci T25 stage0 11.1-12.5 -> 7.3-8.1 s, T26 25.4 -> 17.1 s, proofs verified,
canonical-oracle test added; worktree `../cpu-accumulate`, branch `perf/cpu-d512-accumulate`).
At T28 the fixed binary proves Fibonacci in 165.05 s (1.626 MHz, stage0 67.4 s vs 101.8 s), verified.
The CPU therefore does not need the geometry change; a D128 x3 catalog would still cut its
gather bytes by 25%. Note the 5x acceptance math moves with it: Fibonacci Metal 33.3-33.5 s is
4.9-5.0x the fixed CPU, so once the fix lands the CPU references must be re-frozen.

Metal, current kernel: `S` scales with `n_a^2 * D`, so D128 rank 3 streams 2.25x the bytes
(7.9 s) while the accumulate falls to 0.75x; net +0.6 to +2.3 s. D64 rank 6 streams 4.5x.
On the D512 panels kernel rank > 1 loses.

Metal, restructured kernel: the rotation only mixes coefficients within one ring element, so
a band of ring element `i` needs only that element's `D * 16` bytes per position. Tiling
per ring element gives `S = C * T * K * n_a * D * 16 / 8192`, proportional to volume like
`A`. Then both terms scale 0.75x:

| | BTreeMap | Fibonacci | SHA-2 |
|---|---:|---:|---:|
| commit now (model / measured) | 12.1 / 12.9 s | 17.4 / 16.4 s | 18.8 / 18.2 s |
| commit at 49,152 bits, per-element tiles | 9.1 s | 13.0 s | 14.1 s |
| saving | 3.0 s | 4.3 s | 4.7 s |

Costs: a new packed kernel for D128 rank 3 (128-coefficient rows are half a SIMD-group band,
so the task-to-lane mapping changes; per-coefficient efficiency is unmeasured on GPU), setup
+1.4-3.6 GB RSS against the 90 GiB cap, and the catalog regeneration. Falsifiers to register
before code: bench slope <= 3.2 ns per hot entry and intercept <= 2.7 s at T28 on the new
kernel; planner row accepted by the verifier; RSS <= 90 GiB.

### 9.4 The other levers, priced

| lever | class | Metal saving B / F / S (s) | status |
|---|---|---:|---|
| D128 rank-3 root with per-element tiles (9.3) | public parameter + new kernel | 3.0 / 4.3 / 4.7 | open; largest |
| dense root for the increment family | protocol (new relation, digest) | 0 at measured rates | closed: 22 radix-8 digits per row (the largest L-inf inside the 1,048,575 bucket) = 11.53 M D512 rings = 162 G Montgomery products = 14.5 s at the measured 11.19 G/s; break-even needs 2.9-8.5x. The 1.1 s estimate omitted the NTT butterflies and used inadmissible radix-256 digits. Derivation in `analysis.md` |
| release raw trace rows after async prepare (19 GiB) | preserving | ~1.7 each | untested; L4a analogue regressed |
| Stage 6b RA lazy rounds + Booleanity floor | preserving | 0.5-1 each | no written floor yet |
| eval host packing on GPU or under Stage 7 | preserving | 0.5-0.8 each | untested |
| Stage 5 address installs (sort-and-segment) | preserving | 0.4-0.5 each | L2a/L2b failed |
| eval level restructuring (L5) | protocol | <= 1.0 each | levels are sequentially dependent; not an RLC |
| wider hybrid CPU tail | scheduling | <= 0.4 | closed at 8 blocks |
| 2x GPU part | hardware | ~7 / ~9 / ~10 | not this machine |
| D256 rank 1, K = 2^16 chunks, committed address symbols, Q64 D1024, RNS / radix / carry variants, tile prefetch, matrix reuse, column scheduling | | 0 or regression | closed by measurement or by the response cap |

### 9.5 Landing zones

Preserving levers alone: 3-4 s, BTreeMap ~28-29 s (9.4 MHz), Fibonacci ~29.5-30.5 s,
SHA-2 ~32-33 s. With the D128 rank-3 root: BTreeMap ~25-26 s (10.3-10.7 MHz), Fibonacci
~25.5-26.5 s (10.1-10.5 MHz), SHA-2 ~27.5-28.5 s (9.4-9.8 MHz). SHA-2 does not reach 10 MHz on this
machine with the levers that survive pricing; the increment dense root did not. Every number here is model
arithmetic on measured terms; the D128 kernel's efficiency is the one unmeasured factor, and
it decides the result.

### 9.6 Corrections to earlier sections

- 7.1 "The planner's D64/128/256 rows kept that volume (or the traffic) constant": traffic on
  the current kernel scales with `n_a^2 * D`; volume at D128 rank 3 and D64 rank 6 is 0.75x.
- 7.3a S1 verdict "the norm/schedule lever is closed": closed at rank <= 2 only. At the
  table's ranks the lever is open as a public-parameter change gated on a new kernel.
- 2 "Protocol-preserving commit floor ... hide most of the streaming term": overlap is
  measured-falsified; the floor is the sum of the two terms.

## Not verified

- Single clean runs per workload; the ledger's run-order spread (46–59 s on one binary)
  means repeats can move any number here by several seconds.
- The bench used 524,288 positions per block; production uses 262,144 and reads 2.59 TB
  rather than 2.0 TB, so the production streaming term is closer to 4.5 s.
- The commit floors in section 4 are model estimates, not measured kernels.
- The SHA-2 regression was not bisected; only the passing (`62320b9f0`, Aug 28) and failing
  (`8a5f238f4` + Akita `0e52ebf`, three runs) endpoints are established.
- The census instrumentation was reverted; the built `target/release/examples/modular_benchmark`
  still contains the env-gated census code until the next build.
