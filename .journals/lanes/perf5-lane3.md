# PERF-5 lane 3 — typed T2 rows and preparation replay

Date: 2026-09-03. Original base after lane 1: `1996e1dc2`. Final base after
lane 2: `2914be614`. Rebased code commit: `6297538ca`. Machine: Mac mini M4,
10 Rayon threads. Fixture: `fibonacci_2_18_blake3.bin`, `k = 32`, `N = 2^23`.

## Result

- Canonical honest online wall after lane 2: **29.802 s -> 27.792 s**.
- Lane-3-only wall on the lane-1 base: **37.523 s -> 34.328 s**.
- T2 adaptation: **1.426 s -> 0.640 s**.
- T2 finish plus member setup: **~1.289 s -> 0.493 s**.
- T2 stage-A row member: **2.854 s -> 1.910 s**.
- Proof stays **7,392 B payload / 7,530 B bincode / 352 B statement** after
  lane 2; verifier cost stays **4,868,177 gas**.

The adaptation and finish/setup targets passed. Preparation was **0.454 s**
against the 0.350 s target. T2 stage A was **1.910 s** against the 1.400 s
target.

## Idle gate after lane-2 rebase

The final run acquired the measurement mutex at command-start load
`3.10 / 6.82 / 10.49`, with no competing Cargo process. The honest interval
reported `8.08 / 7.56 / 10.41` at start and `9.10 / 7.84 / 10.42` at end.

| clock | lane 2 | lane 2 + lane 3 | delta |
|---|---:|---:|---:|
| honest online wall | 29.802 s | 27.792 s | **-2.010 s** |
| online phase sum | 29.796 s | 27.786 s | -2.010 s |
| process CPU | 242.390 s | 219.040 s | -23.350 s |
| CPU / wall | 8.133 | 7.881 | -0.252 |

### Printed phases

| phase | lane 2 ms | lane 2 + lane 3 ms | delta ms |
|---|---:|---:|---:|
| wrapper preparation | 564 | 454 | -110 |
| T1/R stream adaptation | 72 | 73 | +1 |
| T2 adaptation | 1,426 | 640 | **-786** |
| phase 1a commitment | 770 | 778 | +8 |
| T2 phase 1b commitment | 1,043 | 966 | -77 |
| T2 phase 2a commitment | 7,198 | 7,140 | -58 |
| T2 phase 2b commitment | 99 | 116 | +17 |
| CopyLink helpers | 34 | 37 | +3 |
| T2 phase 2c + helpers | 383 | 372 | -11 |
| T2 finish | 598 | 457 | **-141** |
| all member constructors | 1,440 | 785 | **-655** |
| proof stages/opening | 16,169 | 15,968 | -201 |

### T2 member split

| work | before ms | after ms |
|---|---:|---:|
| T2 finish | 598 | 457 |
| T2 member setup | ~691 | **36** |
| finish + setup | ~1,289 | **493** |
| T2 stage-A row member | 2,854 | **1,910** |

The pre-change T2 member setup is inferred from the 655 ms drop in the full
constructor bucket plus the new 36 ms T2-only timer; lane 2 had no T2-only
constructor timer. The stage-A before value is the measured planning split;
lane 2 did not change T2 rounds.

## Changes

1. `Columns::generate` decomposes each row value into 96-bit limbs once,
   derives the integer slot sum from the same position coefficients, and
   skips expensive chunk work on un-emitted rows. The isolated probe moved
   column materialization from **1,106 ms to 365 ms**; program evaluation
   remained **162 ms**.
2. `RowMatrix` consumes the typed stream columns once into row-major `u32`
   and `Fr` storage. Bits, `u16`, and `u32` values stay small through round 0
   and promote during the first bind.
3. Each later bind computes the next round in the same traversal. The running
   claim supplies `p(1)` and a direct coefficient-form degree-5 calculation
   supplies the leading coefficient, leaving four full relation evaluations.
4. Relation witness generation uses the hash-table recording transcript. T1
   consumes that same verifier run instead of invoking the native verifier a
   second time.

A full coefficient-form range evaluation measured **3.192 s** for the T2 row
member and was removed. Reusing the entire reference T2 layout failed the
scalar link because selected operands and window rows depend on `theta`; that
attempt was removed. The retained layout build remained about **78 ms**.

## Proof and verifier

| item | lane 2 | lane 2 + lane 3 |
|---|---:|---:|
| payload | 7,392 B | 7,392 B |
| bincode | 7,530 B | 7,530 B |
| statement | 352 B | 352 B |
| ecMul / ecAdd | 227 / 226 | 227 / 226 |
| pairing pairs | 8 | 8 |
| Fr mul / inversions | 121,705 / 10 | 121,705 / 10 |
| Keccak | 846 | 846 |
| N4 gas | 4,868,177 | 4,868,177 |

## Gates

| command | result |
|---|---|
| `cargo check -p jolt-wrapper --all-targets --features prover-fixtures` | pass before commit and after rebase |
| `cargo fmt -q` | pass |
| wrapper clippy, all targets, warnings denied | pass |
| `cargo nextest run -p jolt-wrapper --cargo-quiet` | 64/64 pass before and after rebase |
| locked feature-enabled `real_wrapper --no-capture` | 1/1 pass; every tamper rejects |
