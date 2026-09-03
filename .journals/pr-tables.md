# Wrapper draft PR — final measured tables

Source: non-ignored `wrap_real_t1_r::real_t1_relation_table_round_trip_and_tampers`, cached
fibonacci `2^18` proof, code `b39157273`. Mac mini M4, 10 Rayon threads. Default
`k=32`; `WRAP_K=16` selects the comparison.

## Link coverage

| source | destination | count | binding |
|---|---|---:|---|
| T1 squeeze outputs | R challenge cells | 376 | CopyLink |
| T1 pre-final-squeeze Fr words | R proof-value cells | 1,199 | CopyLink |
| T1 element bytes | T2 input chunks/flags | 45,152 B / 1,526 rows | CopyLink |
| statement fields | R public cells | 7 Fr | public CopyLink |
| T1 state/tail | key-checked statement suffix | 54 B / 4 Fr | verifier-key check |
| R Dory wires | T2 scalar input | 173 scalars | occurrence-weighted link |
| program/profile digest | wrapper key | 32 B | verifier-key check |

The final 23 R proof-value occurrences are the opening point and evaluation absorbed after the
last native squeeze; no Fiat–Shamir value depends on those bytes. Eleven CopyLinks contribute 110
terms, 132 key columns, and 22 sparse helper columns. The element links include compressed G1/G2
sign flags, zero high bits, GT limb order, and the commitment permutation between T1 and T2.

## Proof bytes

| section | k=32 | k=16 |
|---|---:|---:|
| phase 1a wire commitments | 384 | 672 |
| T2 phase 1b wire commitments | 96 | 160 |
| T2 phase 2a wire commitments | 96 | 160 |
| T2 phase 2b wire commitments | 32 | 32 |
| T2 phase 2c + R/Copy helpers | 64 | 96 |
| stage A, 18 committed rounds | 1,184 | 1,184 |
| term stage, 10 committed rounds | 672 | 672 |
| shared BDFG/degree-shift proof | 96 | 96 |
| four factor evaluations | 128 | 128 |
| stage B clear rounds | 704 | 640 |
| reduced claim | 32 | 32 |
| HyperKZG opening | 2,240 | 2,144 |
| **proof payload** | **5,728** | **6,016** |
| **bincode proof** | **5,836** | **6,136** |
| statement, 11 Fr | 352 | 352 |
| **payload + statement** | **6,080** | **6,368** |
| **bincode + statement** | **6,188** | **6,488** |

The soundness links add 128 proof bytes at k=32 and 96 at k=16 versus the fix-3 baseline. Ten new
CopyLink VK groups are key data and add no proof bytes.

## Geometry and phases

| item | k=32 | k=16 |
|---|---:|---:|
| proof wire groups | 21 | 35 |
| key groups | 15 | 15 |
| full groups | 36 | 50 |
| T1 sent / VK groups | 11 / 2 | 20 / 2 |
| CopyLink VK groups | 11 | 11 |
| T2 phases 1b / 2a / 2b / 2c | 3 / 3 / 1 / 2 | 5 / 5 / 1 / 2 |
| final R/Copy helper groups | 1 | 2 |
| stage B rounds | 11 | 10 |

All members use the `2^18` row domain. T1 contributes 232 terms; R 15; eleven CopyLinks 110; T2
177; the R→T2 scalar input one: **T=535**, ten term rounds. T2 uses 201,575 rows with N=42 and
the 256 unique-recoding window rows. The T2 verifier path performs **9,973 Fr multiplications**;
T1 statement evaluation adds **705**. Exporter metadata fixes the term degree at key construction:
T2's maximum four factors plus the coefficient MLE gives degree 5. Stage A is also degree 5.

## Timing

| phase (ms) | k=32 | k=16 |
|---|---:|---:|
| deterministic SRS setup | 7,745 | 3,758 |
| key/profile | 394 | 360 |
| wrapper preparation | 642 | 601 |
| R adaptation | 2,835 | 2,394 |
| T2 adaptation | 1,360 | 1,425 |
| offline key commitments | 1,204 | 773 |
| phase 1a commitments | 1,539 | 993 |
| T2 phase 1b commitments | 1,160 | 1,056 |
| T2 phase 2a commitments | 7,370 | 7,434 |
| T2 phase 2b commitments | 119 | 79 |
| R/Copy helpers | 3,148 | 2,895 |
| T2 phase 2c + helpers | 404 | 438 |
| proof stages/opening | 19,654 | 13,807 |
| verifier | 21 | 21 |

Start load averages: k=32 25.09 / 26.25 / 16.57; k=16 13.30 / 22.67 / 15.88. PERF-4 reduced
phase 2a from roughly 58 s to 7.3 s at k=32.

## Verifier cost

| operation | k=32 | k=16 |
|---|---:|---:|
| ecMul | 185 | 198 |
| ecAdd | 184 | 197 |
| pairing pairs | 8 | 8 |
| Fr multiplications | 40,722 | 33,539 |
| of which T1 statement / T2 | 705 / 9,973 | 705 / 9,973 |
| remaining R + CopyLink + stream + opening | 30,044 | 22,861 |
| Fr inversions | 8 | 8 |
| Keccak | 755 | 762 |
| **N4 gas model** | **2,883,641** | **2,852,045** |

The same counting transcript executes key/statement replay and proof verification. No detached
Keccak adjustment remains.

## Tamper matrix

The real gate mutates every serialized field independently and requires rejection:

- every phase/key/helper commitment, including theta, fingerprint, window, sign, psi, digit-link,
  T2 input-chunk, T2 VK-pin, and R absorbed-word cases;
- every clear and committed sumcheck round coefficient, commitment, claim, and `S(0)`;
- every shared/stage BDFG shifted commitment, quotient, and evaluation witness;
- every stage claim, term evaluation, and reduced claim;
- every HyperKZG fold commitment, both evaluation rows, witness, and `P0(r^2)`;
- public-field mismatch and program/profile-digest mismatch.

The 173-scalar contract test pins wire order and excludes internal `Chi(sigma)`, `S1Acc`, and
`S2Acc`. Full crate gate: 69/69 passed, nextest/wall **190.155/190.67 s**. All-target clippy with
warnings denied passed.
