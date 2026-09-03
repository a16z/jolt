# Wrapper draft PR — measured tables

Source: `wrap_real_t1_r::real_t1_relation_table_round_trip_and_tampers`, real cached fibonacci
`2^18` proof, code `8133720a3` on top of T2 integration `7b5bb611f`. Machine: Mac mini M4,
10 Rayon threads. Default is `k=32`; set `WRAP_K=16` for the retained comparison gate.

## Proof bytes

| section | k=32 | k=16 |
|---|---:|---:|
| phase-1 wire commitments | 480 | 832 |
| phase-2 wire commitments | 128 | 192 |
| stage A, 18 committed rounds | 1,184 | 1,184 |
| term stage, 9 committed rounds | 608 | 608 |
| shared BDFG/degree-shift proof | 96 | 96 |
| four factor evaluations | 128 | 128 |
| stage B, 10 clear degree-2 rounds | 640 | 640 |
| reduced claim | 32 | 32 |
| HyperKZG opening | 2,240 | 2,144 |
| **proof payload** | **5,536** | **5,856** |
| **bincode** | **5,640** | **5,972** |

Statement bytes are separate: 0 B challenge-output words (CopyLink-bound), 224 B for the seven
known public field elements. Payload plus public statement: **5,760 B at k=32**, **6,080 B at
k=16**. The five VK commitments are key data and absent from both proof-wire totals.

## Commitment groups

| table / phase | k=32 sent | k=32 VK | k=16 sent | k=16 VK |
|---|---:|---:|---:|---:|
| T1 phase 1 | 11 | 2 | 20 | 2 |
| R fixed + wires, phase 1 | 1 | 1 | 1 | 1 |
| T1↔R CopyLink fixed, phase 1 | 0 | 1 | 0 | 1 |
| T2 phase 1 | 3 | 1 | 5 | 1 |
| R + CopyLink helpers, phase 2 | 1 | 0 | 1 | 0 |
| T2 phase 2 | 3 | 0 | 5 | 0 |
| **phase 1 full / wire** | **20 / 15** | **5** | **31 / 26** | **5** |
| **phase 2 wire** | **4** | **0** | **6** | **0** |

T1 logical columns: 227 committed bits, 64 wired bits, 16 u32 words, six VK columns. T2:
70 phase-1 (61 u16 chunks, five bits, four field/small columns), 72 phase-2 field columns, five
VK columns. R: nine fixed, three wire, two helper. CopyLink: 12 fixed and two helper. T2 mixed
groups keep the five digit bits and four non-chunk columns in the last phase-1 groups. T1's native
adapter keeps its final bit/u32 groups separate; changing that would also change its owned physical
column IDs and key geometry.

## Row relations

All members use the common `2^18 = 262,144` row domain.

| table | used rows / items | logical columns | stage-A members | rounds | member degree | exported terms | max factor count |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 219,784 active rows | 313 | 2 | 18 | 3, 3 | 232 | 2 |
| T2 | 189,586 used rows | 147 | 2 (row + linked digit side) | 18 | 5, 2 | 132 | 4 |
| R | 40,960 used rows | 14 | 2 (row + linked scalar side) | 18 | 5, 2 | 16 | 4 |
| T1↔R CopyLink | 376 linked items | 14 | 1 | 18 | 5 | 10 | 4 |
| **batched total** |  | 488 | 6 | 18 | max 5 | **390** | **4** |

The T2 digit member and R scalar member are one difference member with public input `rho^172`;
their two final terms have opposite signs. T2's low-to-high row binding is represented by
bit-reversed stream columns at the common high-to-low opening point.

## Protocol stages

| stage | rounds | degree bound | wire |
|---|---:|---:|---:|
| phase 1 commitments + challenge draw | 0 | — | group rows above |
| phase 2 commitments + challenge draw | 0 | — | group rows above |
| A: six row/copy members | 18 | 5 | `18 × (G1 + Fr) + S_A(0)` |
| term compression, T=390 | 9 | 6 | `9 × (G1 + Fr) + S_T(0)` |
| shared committed-round opening | 0 | — | 3 G1 |
| final factor evaluations | 0 | — | 4 Fr |
| B: weighted packed-column reduction | 10 | 2 | 20 Fr |
| final reduced claim | 0 | — | 1 Fr |
| one HyperKZG opening | 23 vars (k=32) / 22 (k=16) | — | rows above |

## Prover and verifier timing

| phase (ms) | k=32 | k=16 |
|---|---:|---:|
| trusted T1 key | 90 | 71 |
| wrapper preparation | 480 | 433 |
| deterministic SRS setup | 6,560 | 3,228 |
| real T1/T2/R adapters | 1,417 | 1,418 |
| offline VK group commitments | 1,215 | 1,092 |
| online phase-1 commitments | 3,658 | 3,101 |
| T2/R/CopyLink helpers | 1,676 | 1,147 |
| online phase-2 commitments | 65,437 | 96,310 |
| stage A + term + B + opening | 14,827 | 10,951 |
| verifier | 15 | 14 |

k=32 load average at start: 3.08 / 3.58 / 5.72. k=16: 8.11 / 5.25 / 6.13.

| verifier operation | k=32 | k=16 |
|---|---:|---:|
| ecMul | 169 | 181 |
| ecAdd | 168 | 180 |
| pairing pairs | 8 | 8 |
| Fr multiplications | 28,516 | 28,536 |
| Fr inversions | 8 | 8 |
| Keccak | 468 | 478 |
| **N4 gas model** | **2,483,013** | **2,588,077** |

N4 model units used by the gate: transaction base 21,000; all-nonzero calldata 16 gas/B;
BN254 ecMul 7,700; Fr multiplication 20; Keccak step 100; eight field inversions batched as one
EIP-2565 modexp (200-gas minimum) plus `3(n-1)` field multiplications; two fixed 2-pair checks
at 114,700 each; one fixed 4-pair HyperKZG check at 183,400. `ecAdd` is recorded but folded into
the measured N4 ecMul/MSM unit rather than charged separately by this model.

## Tamper matrix

| mutation | gate | result |
|---|---|---|
| proof profile differs from stored key | real wrapper | reject |
| T1 phase-1 commitment | real wrapper | reject |
| T2 phase-1 commitment | real wrapper | reject |
| phase-2 helper commitment | real wrapper | reject |
| stage-A `S(0)` | real wrapper | reject |
| term-round commitment | real wrapper | reject |
| final factor evaluation | real wrapper | reject |
| reduced claim | real wrapper | reject |
| HyperKZG value | real wrapper | reject |
| proof supplies a key-owned commitment | `assembly` | reject |
| key commitment differs | `assembly` | reject |
| T2 consumed scalar lacks an R link | `t2_consumed_scalars_match_the_relation_links` | reject/assert |
| T2 chunk, digit, copy operand, lookup operand | `limb_table_e2e` | reject |

The scalar-name contract pins 172 T2 inputs and exactly three R-only omissions:
`Chi(sigma)`, `S1Acc`, `S2Acc`.
