# Wrapper draft PR — R1CS + Spartan final measurements

Source: non-ignored `wrap_real_t1_r::real_wrapper_round_trip_and_tampers`, cached
fibonacci `2^18` proof, Mac mini M4, 10 Rayon threads. Default `k=32`;
`WRAP_K=16` selects the comparison.

The k=32 timing column was rerun after PERF-5 lane 1. The k=16 timing and
verifier-cost columns remain the pre-review-3 measurements; its commitment
geometry and proof bytes are unchanged.

## Link coverage

| source | destination | count | binding |
|---|---|---:|---|
| T1 squeeze outputs | Spartan W challenge cells | 376 | CopyLink |
| T1 pre-final-squeeze Fr words | Spartan W proof-value cells | 1,200 | CopyLink |
| T1 element bytes | T2 input chunks/flags | 45,152 B / 1,526 rows | CopyLink |
| seven statement fields + one | R1CS public segment | 8 Fr | Spartan `z` assignment |
| T1 state/tail | wrapper statement suffix | 54 B / 4 Fr | transcript input |
| Spartan W Dory cells | T2 scalar input | 173 occurrences | occurrence-weighted link |
| program/profile digest | wrapper key | 32 B | verifier-key check |

Ten CopyLinks contribute 100 terms, 120 pinned columns, and 20 sparse helper columns. The
element links include compressed G1/G2 sign flags, zero high bits, GT limb order, and the
profile-derived commitment permutation. The last 23 absorbed Fr values follow the final native
squeeze and do not affect a Fiat-Shamir challenge. `Chi(sigma)`, `S1Acc`, and `S2Acc` stay internal
to R and do not enter the 173-scalar link.

## Fiat-Shamir challenge schedule

| committed phase | challenges drawn after commitment | count |
|---|---|---:|
| phase 1a: T1 + W | 38 T1 randomizers, `theta` | 39 |
| T2 phase 1b | `xi`, `alpha`, ten CopyLink `(beta, gamma)` pairs, scalar-link `rho` | 23 |
| T2 phase 2a | `fp_root` | 1 |
| T2 phase 2b | `beta`, `fp_combine`, `copy_root` | 3 |
| T2 phase 2c + CopyLink helpers | T2 row/member challenges; ten CopyLink points and weights | 232 |

## Proof bytes

| section | k=32 | k=16 |
|---|---:|---:|
| phase 1a wire commitments | 384 | 672 |
| T2 phase 1b wire commitments | 96 | 160 |
| T2 phase 2a wire commitments | 96 | 160 |
| T2 phase 2b wire commitments | 32 | 32 |
| T2 phase 2c + CopyLink helpers | 64 | 96 |
| Spartan outer, 13 committed rounds | 864 | 864 |
| Spartan inner, 13 clear rounds | 832 | 832 |
| stage A, 18 committed rounds | 1,184 | 1,184 |
| term stage, 9 committed rounds | 608 | 608 |
| shared BDFG/degree-shift proof | 96 | 96 |
| four factor evaluations | 128 | 128 |
| stage B clear rounds | 704 | 640 |
| reduced claims (opening + Az/Bz/Cz/W) | 160 | 160 |
| HyperKZG opening | 2,240 | 2,144 |
| **proof payload** | **7,488** | **7,776** |
| **bincode proof** | **7,628** | **7,928** |
| statement, 11 Fr | 352 | 352 |
| **payload + statement** | **7,840** | **8,128** |
| **bincode + statement** | **7,980** | **8,280** |

## Geometry

| item | value |
|---|---:|
| R1CS constraints / variables | 5,323 / 6,831 |
| public / private variables | 7 / 6,823 |
| outer / inner rounds | 13 / 13 |
| common row rounds | 18 |
| matrix nonzeros | 35,346 |
| native matrix-evaluation Fr multiplications | 87,081 |
| T2 rows | 201,575 |
| total terms / term rounds | 510 / 9 |
| T1 / CopyLink / T2 / scalar / carry terms | 232 / 100 / 176 / 1 / 1 |

| groups | k=32 | k=16 |
|---|---:|---:|
| proof wire / key / full | 21 / 13 / 34 | 35 / 13 / 48 |
| T1 sent / VK | 11 / 2 | 20 / 2 |
| Spartan W | 1 | 1 |
| CopyLink VK | 10 | 10 |
| T2 1b / 2a / 2b / 2c | 3 / 3 / 1 / 2 | 5 / 5 / 1 / 2 |
| final helper groups | 1 | 2 |

## Timing

| phase (ms) | k=32 | k=16 |
|---|---:|---:|
| deterministic SRS setup (offline) | 8,161 | 3,773 |
| key/profile (offline) | 160 | 355 |
| offline key commitments | 973 | 517 |
| wrapper preparation | 550 | 602 |
| T1/R stream adaptation | 270 | — |
| T2 adaptation | 1,261 | 1,432 |
| phase 1a commitment | 1,945 | 790 |
| T2 phase 1b commitment | 1,050 | 979 |
| T2 phase 2a commitment | 7,271 | 7,481 |
| T2 phase 2b commitment | 101 | 78 |
| CopyLink helpers | 2,960 | 2,709 |
| T2 phase 2c + helpers | 344 | 206 |
| T2 finish | 457 | — |
| member construction | 1,983 | — |
| proof stages/opening | 19,326 | 13,428 |
| **honest online total** | **37,523** | — |
| verifier (outside online clock) | 25 | 27 |

k=32 command-start load: `3.95 / 11.31 / 20.64`; honest-clock start/end:
`8.72 / 10.49 / 18.98` -> `9.94 / 10.57 / 18.66`. Process CPU was
274.380 s over 37.523 s wall. k=16 old start load: `8.80 / 9.87 / 8.27`.

## Verifier cost

| operation | k=32 | k=16 |
|---|---:|---:|
| ecMul | 234 | 247 |
| ecAdd | 233 | 246 |
| pairing pairs | 8 | 8 |
| Fr multiplications | 127,884 | 172,364 |
| Fr inversions | 10 | 10 |
| Keccak | 857 | 864 |
| **N4 gas model** | **5,048,805** | **6,050,469** |

The same observer counts transcript replay, native sparse-matrix evaluation, sumchecks, links,
term reduction, and the final opening. The k=32 native sparse-matrix block accounts for 87,081
Fr multiplications over 35,346 nonzeros, down from the prior 136,946 test formula. Total k=32
cost moved from 179,547 to 127,884 Fr multiplications and 6,082,065 to 5,048,805 gas.

## Tamper matrix

The real gate mutates every serialized field independently and requires rejection:

- every wire commitment, including W and all T2 phases;
- Spartan outer commitments/claims/`S(0)`, inner clear coefficients, and Az/Bz/Cz/W claims;
- every stage-A/term committed round and every stage-B clear coefficient;
- shared BDFG shifted commitment, quotient, and evaluation witness;
- every factor evaluation and final HyperKZG fold commitment/evaluation/quotient field;
- direct T2 window/sign/psi/digit/input-row mutations, an absorbed-Fr W row, T2 VK pin,
  statement mismatch, a fixed-challenge T1 initial-state claim change, and program/profile
  mismatch.

The permanent scalar contract pins the 173-wire order and occurrence-weight formula. Feature-enabled
all-target clippy passed with warnings denied. The combined HyperKZG/wrapper suite passed 89/89;
the feature-enabled real gate passed 1/1 in 55.562 s.
