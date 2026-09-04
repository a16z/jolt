# PERF-5 lane 5a — T2 LogUp grouping

Date: 2026-09-03. Base: `80ac4bc7d`. Machine: Mac mini M4, 10 Rayon
threads. Fixture: `fibonacci_2_18_blake3.bin`, `k = 32`, `N = 2^23`.

## Decision

Keep **`s = 4`**. It reduces the range helpers from 22 to 17 and phase 2a
from 67 to 62 full-Fr columns. The phase fits two 32-column groups: 62 live
slots and two padding slots, so neither group is padding-only.

The locked real gate measured **27.792 s -> 26.624 s** honest online wall,
**7.140 s -> 6.227 s** phase-2a commitment, and **1.910 s -> 2.700 s** T2
stage A. Payload stays 7,392 B; bincode changes 7,530 B -> 7,529 B. The
verifier gas model rises **4,868,177 -> 4,890,645** because the fifth final
factor adds 1,524 observed Fr multiplications despite one fewer commitment.

`s = 6` misses the byte cap at +64 B; `s = 9` misses it at +160 B and its
row check is slower. Both passed the same relation/term correctness check as
`s = 4` before the constant was restored.

## Sweep

`M` = measured; `E` = estimated.

| s | helpers | phase-2a columns / groups | row degree / final factors | T2 terms | phase 2a | T2 stage A | online wall | payload / bincode | gas | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 17 | 62 / 2 | 6 / 5 | 167 | **6.227 s M** | **2.700 s M** | **26.624 s M** | **7,392 / 7,529 B M** | **4,890,645 M** | chosen |
| 6 | 11 | 56 / 2 | 8 / 7 | 155 | 5.624 s E | 3.621 s E | 26.942 s E | 7,456 / 7,593 B E | 4,953,029 E | +64 B |
| 9 | 8 | 53 / 2 | 11 / 10 | 149 | 5.323 s E | 5.040 s E | 28.060 s E | 7,552 / 7,689 B E | 5,046,605 E | +160 B, slower |

Timing model for `s = 6/9`: phase 2a scales the locked `s = 4` time by
active full-Fr columns (`56/62`, `53/62`). T2 stage A scales the locked
`s = 4` time by the same-harness dense-row ratios: `3.215834/2.397830` and
`4.475847/2.397830`. Online wall substitutes those two components into the
locked `s = 4` wall and holds every other phase fixed. The dense-row
correctness timings were unlocked and are used only as ratios.

The byte estimates follow the proof shape: all candidates remove one 32 B
phase-2a commitment; final-factor evaluations add 32 B each. The gas estimate
adds 31,192 gas per factor after `s = 4`: 32 B of calldata plus the measured
`s = 3 -> 4` term-reduction rate of about 1,534 Fr multiplications. Statement
size stays 352 B.

## Locked phase comparison

The before column is the final lane-3 gate. The after gate acquired the lock
after a 10-minute gap and a prebuild, at command-start load 2.77; the honest
clock load was 3.95 -> 6.83.

| phase | s=3 | s=4 | delta |
|---|---:|---:|---:|
| wrapper preparation | 454 ms | 452 ms | -2 ms |
| T1/R stream adaptation | 73 ms | 70 ms | -3 ms |
| T2 adaptation | 640 ms | 644 ms | +4 ms |
| phase 1a commitment | 778 ms | 763 ms | -15 ms |
| T2 phase 1b commitment | 966 ms | 1,089 ms | +123 ms |
| T2 phase 2a commitment | 7,140 ms | 6,227 ms | **-913 ms** |
| T2 phase 2b commitment | 116 ms | 100 ms | -16 ms |
| CopyLink helpers | 37 ms | 36 ms | -1 ms |
| T2 phase 2c + helpers | 372 ms | 351 ms | -21 ms |
| T2 finish | 457 ms | 281 ms | -176 ms |
| all member constructors | 785 ms | 791 ms | +6 ms |
| proof stages/opening | 15,968 ms | 15,814 ms | -154 ms |
| T2 stage-A row member | 1,910 ms | 2,700 ms | **+790 ms** |
| **honest online wall** | **27,792 ms** | **26,624 ms** | **-1,168 ms** |
| process CPU | 219.040 s | 215.430 s | -3.610 s |
| CPU / wall | 7.881 | 8.092 | +0.211 |

## Wire and verifier deltas

| item | s=3 | s=4 |
|---|---:|---:|
| phase-2a commitment bytes | 96 | 64 |
| final-factor bytes | 128 | 160 |
| payload | 7,392 | 7,392 |
| bincode | 7,530 | 7,529 |
| statement | 352 | 352 |
| proof wire / key / full groups | 20 / 7 / 27 | 19 / 7 / 26 |
| T2 1b / 2a / 2b / 2c groups | 3 / 3 / 1 / 2 | 3 / 2 / 1 / 2 |
| total terms / term rounds | 510 / 9 | 500 / 9 |
| ecMul / ecAdd | 227 / 226 | 226 / 225 |
| pairing pairs | 8 | 8 |
| Fr multiplications / inversions | 121,705 / 10 | 123,229 / 10 |
| Keccak | 846 | 848 |
| N4 gas | 4,868,177 | 4,890,645 |

## Implementation

- `columns::GROUP_SIZE` owns `s`; helper count uses ceiling division and
  `range_group` owns the partial final group.
- Witness helpers, row evaluation, final terms, key member degree, and the
  gate geometry derive their shape from that owner.
- The fused row prover reconstructs any declared degree from `d` evaluations
  plus its leading coefficient; the previous interpolator was fixed at five.
- The out-of-range chunk tamper recomputes the helper over the canonical group
  and still rejects.

## Gates

| command | result |
|---|---|
| `cargo fmt -q --message-format=short` | pass |
| feature-enabled all-target `cargo check` | pass |
| feature-enabled all-target clippy, warnings denied | pass |
| `cargo nextest run -p jolt-wrapper --cargo-quiet` | 64/64 pass |
| locked feature-enabled `real_wrapper --no-capture` | 1/1 pass; every tamper rejects |
