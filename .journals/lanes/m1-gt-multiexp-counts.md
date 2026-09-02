# Lane M1 — deferred Dory final-check operation counts

Date: 2026-09-02. Source: `dory-pcs 0.4.2`, arkworks revision `76bb3a45`, Jolt branch `wrap/spartan-hyperkzg`.

## Result

**Cheapest measured RHS: 4-dimensional GT Frobenius decomposition + variable-time Pippenger, `c=7`.** At `sigma=11, N=41`: **6,204 Fq12 multiplications + 63 cyclotomic squarings = 336,150 Fq products = 75,204 output-coefficient rows** (4.470 products/row). At `sigma=12, N=41`: **6,521 + 63 = 353,268 Fq products = 79,008 rows** (4.471 products/row). With `N=42` at `sigma=11`: **6,240 + 63 = 338,094 Fq products = 75,636 rows**.

Four dimensions win because Frobenius produces the `X, X^q, X^(q^2), X^(q^3)` bases with constant-coordinate maps, while the reduced lattice bounds every mini-scalar to 64 bits in this sample. Relative to 2D/`c=6`, 4D/`c=7` removes 63 cyclotomic squarings and 568 Fq12 multiplications at `sigma=11`; its larger base set costs less than the saved window reductions. `c=8` reduces squarings by seven but adds 259 multiplications. The algorithm choice is unchanged at `sigma=12` and for `N=42`.

## Measurement convention

- `sigma=11`: actual Fiat–Shamir challenges replayed from a seeded random `2^22`-evaluation polynomial proof. The 41 nonidentity commitment bases are GT offsets whose `rho`-weighted product equals the proof commitment.
- `sigma=12`: analytic count over the same closed form using seeded synthetic field challenges; no `2^24` polynomial allocation. Counts depend on public challenge digits.
- Pippenger skips zero digits, treats the first write to an empty bucket/accumulator as an assignment, and counts every later group operation. A relation may choose this public schedule before committing its operation rows.
- Naive counts 254 generic squares per base, every set-bit multiply, and `B-1` final combines. Pippenger uses cyclotomic squaring.
- Frobenius maps and GT conjugations are linear/constant-coordinate maps and carry zero bilinear rows here.

## GT multi-exponentiation

`S` is generic for naive and cyclotomic for every Pippenger row. `Fq` uses `54M + 36S_generic + 18S_cyclotomic`; rows use `12(M+S)`.

| Algorithm | c | bits | sigma=11 M | sigma=11 S | Fq products | rows | products/row | sigma=12 M | sigma=12 S | Fq products | rows | products/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | — | 254 | 16,940 | 36,576 | 2,231,496 | 642,192 | 3.475 | 19,288 | 38,862 | 2,440,584 | 697,800 | 3.498 |
| Pippenger | 4 | 254 | 8,734 | 252 | 476,172 | 107,832 | 4.416 | 9,923 | 252 | 540,378 | 122,100 | 4.426 |
| Pippenger | 6 | 254 | 8,107 | 252 | 442,314 | 100,308 | 4.410 | 8,953 | 252 | 487,998 | 110,460 | 4.418 |
| Pippenger | 7 | 254 | 9,284 | 252 | 505,872 | 114,432 | 4.421 | 10,021 | 252 | 545,670 | 123,276 | 4.426 |
| Pippenger | 8 | 254 | 12,064 | 248 | 655,920 | 147,744 | 4.440 | 12,694 | 248 | 689,940 | 155,304 | 4.443 |
| 2D GLV | 4 | 127 | 8,267 | 124 | 448,650 | 100,692 | 4.456 | 9,438 | 124 | 511,884 | 114,744 | 4.461 |
| 2D GLV | 6 | 127 | 6,772 | 126 | 367,956 | 82,776 | 4.445 | 7,617 | 126 | 413,586 | 92,916 | 4.451 |
| 2D GLV | 7 | 127 | 6,998 | 126 | 380,160 | 85,488 | 4.447 | 7,730 | 126 | 419,688 | 94,272 | 4.452 |
| 2D GLV | 8 | 127 | 8,099 | 120 | 439,506 | 98,628 | 4.456 | 8,729 | 120 | 473,526 | 106,188 | 4.459 |
| 4D GLV | 4 | 64 | 8,596 | 60 | 465,264 | 103,872 | 4.479 | 9,165 | 60 | 495,990 | 110,700 | 4.480 |
| 4D GLV | 6 | 64 | 6,631 | 60 | 359,154 | 80,292 | 4.473 | 7,038 | 60 | 381,132 | 85,176 | 4.475 |
| **4D GLV** | **7** | **64** | **6,204** | **63** | **336,150** | **75,204** | **4.470** | **6,521** | **63** | **353,268** | **79,008** | **4.471** |
| 4D GLV | 8 | 64 | 6,463 | 56 | 350,010 | 78,228 | 4.474 | 6,765 | 56 | 366,318 | 81,852 | 4.475 |

The 2D decomposition is exact integer radix in `lambda = q mod r = 6u^2`; the 4D decomposition uses a reduced kernel lattice for `[1, lambda, lambda^2, lambda^3] mod r`. Every decomposed scalar is recomposed in `Fr` before counting. Jolt's `jolt-crypto` 2D constants do **not** apply: they target the G1 eigenvalue `q^4 mod r`. Its 4D table targets the needed Frobenius powers, but the bench keeps a local four-row lattice so production modules remain unchanged.

## G1 and G2 multi-exponentiations

`c=4` is cheapest for both sizes. Terms are the full-width nonunit scalar multiplications needed to form the four pairing inputs: `3sigma+4 = 37/40`. Bucket writes use mixed affine/projective addition; reductions use projective/projective addition.

| sigma | group | c | mixed adds | projective adds | doublings | Fq products | rows | products/row |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | G1 | 4 | 820 | 1,581 | 252 | 34,499 | 34,499 | 1.000 |
| 11 | G2 | 4 | 1,085 | 1,677 | 252 | 105,591 | 77,708 | 1.359 |
| 12 | G1 | 4 | 1,487 | 1,762 | 252 | 44,551 | 44,551 | 1.000 |
| 12 | G2 | 4 | 1,493 | 1,768 | 252 | 121,562 | 89,414 | 1.360 |

Ark's `a=0` Jacobian formulas give G1 mixed add `8M+3S=11`, projective add `11M+4S=15`, double `3M+4S=7`. Over Fq2, using `M=3` and `S=2` Fq products gives G2 costs `30/41/17`; their row counts are `22/30/14`. Projective-to-affine input normalization is outside these counts because the committed-limb relation can take affine inputs.

## Four-pair multi-pairing

Ark's BN254 signed ate loop has 64 doubling lines, 21 signed-add lines, and two terminal Frobenius lines **per pair**. Four pairs share the 63 accumulator squarings but perform `4*(64+21+2) = 348` sparse accumulator multiplications.

| Component | Exact events | Fq products | rows | products/row |
|---|---|---:|---:|---:|
| G2 line preparation, four pairs | 256 doubles, 84 signed adds, 8 terminal Frobenius adds | 10,108 | 8,056 | 1.255 |
| Miller accumulator squares | 63 generic Fq12 squares | 2,268 | 756 | 3.000 |
| Evaluated sparse lines | 348 `mul_by_034` events | 14,964 | 5,568 | 2.688 |
| **Miller loop** | above | **27,340** | **14,380** | **1.901** |
| FE hard part | 82 Fq12 multiplies, 189 cyclotomic squares, 3 Frobenius maps | 7,830 | 3,252 | 2.408 |
| FE easy part | 2 Fq12 multiplies, 1 Fq12 inversion, 1 Frobenius map | 108 | 24 | 4.500 |
| FE inverse relation | witness inverse checked by 1 Fq12 multiply | 54 | 12 | 4.500 |
| **Final exponentiation relation** | native inversion replaced by its product check | **7,992** | **3,288** | **2.431** |

Line-preparation formulas counted directly from arkworks: double = 26 Fq products/22 rows; add = 37/26; two Frobenius input maps = 12/8 per pair. Each line evaluation has two Fq2-by-Fq scales (4 products/4 rows), then `mul_by_034` has 13 Fq2 multiplications (39 products/12 final-output rows). The FE hard part has three NAF exponentiations by `u`; `u` has 63 NAF digits and weight 24, hence `3*62+3=189` cyclotomic squares and `3*24+10=82` multiplies.

## Full relation sizing

These totals combine the best GT RHS, best G1/G2 schedules, Miller loop, FE, and the FE inverse product check. Public scalar arithmetic and linear Frobenius/constant maps add no bilinear rows.

| sigma / N | Fq products | output-coefficient rows | mean products/row |
|---|---:|---:|---:|
| 11 / 41 | 511,572 | 205,079 | 2.495 |
| 12 / 41 | 554,713 | 230,641 | 2.405 |

Tower formulas: Fq2 multiplication = 3 Fq products; Fq6 multiplication = 6 Fq2 = 18; Fq12 multiplication = 3 Fq6 = 54; generic Fq12 square = 36; Granger–Scott cyclotomic square = 6 Fq2 = 18. An Fq12 operation emits 12 coefficient rows, Fq6 emits 6, and Fq2 emits 2. Each row is `z = sum_i x_i*y_i mod q`, reduced once.

## Flattened RHS bases and scalars

Let `u_j = beta_(j+1)^-1`, `v_j = beta_(j+1)` for `j < sigma-1`; `u_(sigma-1)=d^-1`, `v_(sigma-1)=d`. Setup index is `k=sigma-j`.

| Bases | Count | Scalars |
|---|---:|---|
| `C_init` | 1 | `1` |
| `C_i` | N | `beta_0^-1 rho^i` |
| `D2_init` | 1 | `beta_0 + d^2` |
| `C+_j`, `C-_j` | `2sigma` | `alpha_j`, `alpha_j^-1` |
| `D1L_j`, `D1R_j` | `2sigma` | `u_j alpha_j`, `u_j` |
| `D2L_j`, `D2R_j` | `2sigma` | `v_j alpha_j^-1`, `v_j` |
| `Delta1R[k]`, `Delta2R[k]` | `2sigma` | `u_j beta_j`, `v_j beta_j^-1` |
| `chi[t]` | `sigma+1` | `1 + u_j alpha_j beta_j + v_j alpha_j^-1 beta_j^-1` for `t=sigma-j-1`; unit contribution at every `t` |
| `HT` | 1 | `s1_acc s2_acc` |

Total: `9sigma + N + 4` = 144 (`sigma=11,N=41`) or 153 (`sigma=12,N=41`). Proof GT bases: `6sigma+2`; commitment bases: `N`; setup GT bases: `3sigma+2`. This matches `.journals/lanes/dory-offload-study.md` section 1.4 with no base/scalar discrepancy. One count clarification: the study's 87 Miller line multiplications is per pair; the four-pair call has 348 and shares its squarings.

Pairing-input scalar-multiplication multiset used for EC counts:

- G1: each round's `beta_j, alpha_j, alpha_j^-1`; then `-gamma^-1`, `-gamma^-1 d s2_acc`, `d`, `d^2`.
- G2: `evaluation`; each round's `beta_j^-1, alpha_j, alpha_j^-1`; then `-gamma`, `-gamma d^-1 s1_acc`, `d^-1`.

## Executable check and wall time

File: `crates/jolt-dory/benches/deferred_check_counts.rs`.

```bash
CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/wrap-spartan-hyperkzg RAYON_NUM_THREADS=1 RUST_MIN_STACK=67108864 cargo bench -p jolt-dory --bench deferred_check_counts -- --nocapture
```

The bench locally replicates the private Jolt-to-Dory transcript adapter, following `verify_evaluation_proof` byte-for-byte. It asserts the production verifier accepts, the flattened four-pair/one-GT-multiexp equation accepts, 2D and 4D decompositions reproduce the plain result, every one of 144 individually changed `X_k` bases fails, and production verification rejects a changed commitment.

Latest release run, one Rayon worker: setup 1.256 s, commit 27.238 s, open 3.151 s, production `dory::verify` **49.664 ms**, best deferred check **19.114 ms**. The deferred timer covers transcript replay, scalar/base formation, pairing-input accumulation, one four-pair call, 4D decomposition, and the `c=7` GT multi-exponentiation; commitment splitting and mutation checks are outside it.
