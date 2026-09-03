# Wrapper draft PR — final measured tables

Source: `wrap_real_t1_r::real_t1_relation_table_round_trip_and_tampers`, cached fibonacci
`2^18` proof, code `c41d96826` on the T2 fix-3 stack through `1ffd0088c`. Machine:
Mac mini M4, 10 Rayon threads. Default `k=32`; `WRAP_K=16` selects the retained comparison.

## Proof bytes

| section | k=32 | k=16 |
|---|---:|---:|
| phase 1a wire commitments | 384 | 672 |
| T2 phase 1b wire commitments | 96 | 160 |
| T2 phase 2a wire commitments | 96 | 160 |
| T2 phase 2b wire commitments | 32 | 32 |
| T2 phase 2c + R/Copy helper wire commitments | 64 | 64 |
| stage A, 18 committed rounds | 1,184 | 1,184 |
| term stage, 9 committed rounds | 608 | 608 |
| shared BDFG/degree-shift proof | 96 | 96 |
| four factor evaluations | 128 | 128 |
| stage B, 10 clear degree-2 rounds | 640 | 640 |
| reduced claim | 32 | 32 |
| HyperKZG opening | 2,240 | 2,144 |
| **proof payload** | **5,600** | **5,920** |
| **bincode** | **5,706** | **6,038** |

Statement bytes are separate: 0 B challenge words (CopyLink-bound), 224 B for seven known
field elements. Payload plus statement: **5,824 B at k=32**, **6,144 B at k=16**. Five VK
commitments are key data and absent from the proof wire.

## Commitment groups

| table / phase | k=32 wire | k=32 VK | k=16 wire | k=16 VK |
|---|---:|---:|---:|---:|
| T1 phase 1a | 11 | 2 | 20 | 2 |
| R fixed + wires, phase 1a | 1 | 1 | 1 | 1 |
| T1↔R CopyLink fixed, phase 1a | 0 | 1 | 0 | 1 |
| T2 phase 1b | 3 | 0 | 5 | 0 |
| T2 phase 2a | 3 | 0 | 5 | 0 |
| T2 phase 2b | 1 | 0 | 1 | 0 |
| T2 phase 2c | 1 | 1 | 1 | 1 |
| R + CopyLink helpers, final phase | 1 | 0 | 1 | 0 |
| **total** | **21** | **5** | **34** | **5** |

T1: 227 committed bits, 64 wired bits, 16 u32 words, six VK columns. T2: phase 1b
71 columns, phase 2a 67, phase 2b two, phase 2c three, VK six. R: nine fixed, three wire,
two helper. CopyLink: 12 fixed, two helper. T2's VK suffix is physically part of phase 2c and
included in `commitment_phases`: k=32 `[3, 3, 1, 2]`, k=16 `[5, 5, 1, 2]`.

## Relations and stages

All members use the common `2^18 = 262,144` row domain.

| table | used rows / items | logical columns | members | degree | terms |
|---|---:|---:|---:|---:|---:|
| T1 | 219,784 active rows | 313 | 2 | 3, 3 | 232 |
| T2 | 201,319 used rows at N=42 | 149 | row + digit side | 5, 2 | 175 |
| R | 40,960 allocated rows | 14 | row + scalar side | 5, 2 | 15 + 1 |
| T1↔R CopyLink | 376 links | 14 | 1 | 5 | 10 |
| **batched total** |  | **490** | **6** | max 5 | **433** |

The linked member proves T2's occurrence-weighted digit claim minus R's scalar claim, with
public input `W[K] + W[K+1]·theta`; the 173 named scalars use `FlattenedCheck::wires()` order.

| stage | rounds | degree | wire |
|---|---:|---:|---:|
| ordered commitments: 1a → 1b → 2a → 2b → 2c | 0 | — | group rows above |
| A: six members | 18 | 5 | `18 × (G1 + Fr) + S_A(0)` |
| term compression, T=433 | 9 | 6 | `9 × (G1 + Fr) + S_T(0)` |
| shared committed-round opening | 0 | — | 3 G1 |
| factor evaluations | 0 | — | 4 Fr |
| B: weighted packed-column reduction | 10 | 2 | 20 Fr |
| reduced claim | 0 | — | 1 Fr |
| HyperKZG | 23 vars (k=32), 22 vars (k=16) | — | rows above |

## Timing and verifier cost

| phase (ms) | k=32 | k=16 |
|---|---:|---:|
| deterministic SRS setup | 8,044 | 38,781 |
| trusted T1 key | 189 | 1,577 |
| wrapper preparation | 649 | 6,383 |
| T1/R/Copy adaptation | 458 | 4,690 |
| T2 adaptation | 1,429 | 6,325 |
| offline VK commitments | 1,129 | 6,457 |
| phase 1a commitments | 2,180 | 9,312 |
| T2 phase 1b commitments | 1,846 | 5,792 |
| T2 phase 2a commitments | 57,954 | 133,419 |
| T2 phase 2b commitments | 92 | 138 |
| R/Copy helpers | 281 | 501 |
| T2 phase 2c + VK + R helpers | 400 | 537 |
| stage A + term + B + opening | 15,399 | 22,559 |
| verifier | 24 | 73 |

k=32 load average at start: 8.76 / 7.98 / 7.27. k=16: 26.01 / 12.18 / 8.89.

| verifier operation | k=32 | k=16 |
|---|---:|---:|
| ecMul | 171 | 183 |
| ecAdd | 170 | 182 |
| pairing pairs | 8 | 8 |
| Fr multiplications | 29,491 | 29,511 |
| Fr inversions | 8 | 8 |
| Keccak | 471 | 481 |
| **N4 gas model** | **2,520,261** | **2,625,325** |

Observed Fr work includes T1 statement claims, T2's 9,875-term exporter path,
occurrence-weighted R scalar evaluation, and stream reductions. N4 units: transaction 21,000;
nonzero calldata 16 gas/B; BN254 ecMul 7,700; Fr multiplication 20; Keccak 100; batched
inversions use one EIP-2565 modexp plus `3(n-1)` Fr multiplications; fixed pairing charges are
two 2-pair checks at 114,700 and one 4-pair HyperKZG check at 183,400.

## Tamper matrix

| mutation | result |
|---|---|
| proof shape differs from stored key | reject |
| T1 phase-1a commitment | reject |
| theta-dependent T2 phase-1b commitment | reject |
| T2 phase-2b fingerprint commitment | reject |
| T2 phase-2c helper commitment | reject |
| T2 VK pin in verifier key | reject |
| sign-flag row | reject |
| psi-chain input row | reject |
| digit-link occurrence row | reject |
| stage-A `S(0)` | reject |
| term-round commitment | reject |
| factor evaluation | reject |
| reduced claim | reject |
| HyperKZG value | reject |
| R/T2 scalar order or membership differs | contract assertion |

The permanent scalar contract pins all 173 names in order and excludes `Chi(sigma)`, `S1Acc`,
and `S2Acc`.

Full crate gate: `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet` —
73/73 passed, 8 skipped; nextest 250.365 s, wall 270.55 s. Crate-wide all-target clippy with
warnings denied passed.
