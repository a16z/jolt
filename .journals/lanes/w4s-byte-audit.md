# W4-S byte audit — implemented `2^17` G-shape

Date: 2026-09-02. The ignored release gate executes `jolt-wrapper` production code for `k = 8`
and `k = 16`. Payload counts BN254 G1 and Fr values as 32 B and exclude serde framing.

## Exact payloads

| item | `k = 8` | bytes | `k = 16` | bytes |
|---|---:|---:|---:|---:|
| packed commitments | 30 G1 | 960 | 15 G1 | 480 |
| stage A committed rounds | 17 G1 + 34 Fr + 3 G1 | 1,728 | same | 1,728 |
| stage B column batch | 8 rounds × 2 Fr | 512 | same | 512 |
| stage C opening reduction | 20 rounds × 2 Fr | 1,280 | 21 rounds × 2 Fr | 1,344 |
| A→B factor claims | 5 Fr | 160 | 5 Fr | 160 |
| reduced packed claim | 1 Fr | 32 | 1 Fr | 32 |
| multilinear HyperKZG | 20 G1 + 60 Fr | 2,560 | 21 G1 + 63 Fr | 2,688 |
| **payload** | | **7,232** | | **6,944** |
| standard-bincode framing | | 115 | | 102 |
| **standard bincode** | | **7,347** | | **7,046** |

The previous `k = 8` proof was 10,304 B payload / 10,445 B bincode. Implemented payload saving:
**3,072 B**. `k = 16` saves another **288 B**.

## Packed commitments

The fixture has 163 bit, 54 u16, 19 helper-Fr, and one witness-Fr column: 237 total.

```text
k=8:  ceil(237/8)  = 30 emitted groups; group domain pads to 32
k=16: ceil(237/16) = 15 emitted groups; group domain pads to 16
```

The column domain has eight variables in both cases: five group + three slot bits at `k = 8`,
four group + four slot bits at `k = 16`. Missing slots/groups are canonical zeros and emit no
commitments.

## Stage A: degree-bounded variable-point KZG

Rows are `2^17`, hence 17 rounds. Each round sends:

```text
C_i = commit(s_i)             1 G1
s_i(0), claim_{i+1}=s_i(r_i)  2 Fr
                              ----
                              96 B
```

The verifier derives `s_i(1) = claim_i - s_i(0)`. After all rounds, the proof adds:

```text
C_shift                       1 G1   batched degree-5 check
W, W'                         2 G1   BDFG20 §4 variable-point batch
```

Total: `17 * 96 + 3 * 32 = 1,728 B`. The old compressed stream used 2,688 B, so the
implemented saving is **960 B**.

The two BDFG elements are both required for different point sets. `W` commits to the quotient by
the global vanishing polynomial; after challenge `z`, `W'` opens the derived polynomial at `z`.
Collapsing them to one G1 would require extra G2 powers/commitments beyond this protocol.

### Degree soundness

Let `L` be the G1 SRS length, `D = 5`, and `rho` be sampled after every round commitment. The
prover sends

```text
C_shift = commit(X^(L-1-D) * sum_i rho^i s_i).
```

The verifier setup contains `[beta^(L-1-D)]_2` and checks

```text
e(C_shift, [1]_2) = e(sum_i rho^i C_i, [beta^(L-1-D)]_2).
```

Any coefficient above degree `D` shifts beyond the available G1 SRS, except with the random-RLC
cancellation probability. A production Powers-of-Tau ceremony supplies the matching G2 power;
`setup_from_secret` derives it directly for tests. KZG binding plus this degree check fixes a
degree-at-most-five `s_i`; the BDFG openings then bind its values at `0`, `1`, and `r_i`, which
checks both sumcheck recurrence equations. Three evaluations alone would not determine a generic
degree-five polynomial.

## Stage B: one shared column point

Stage A exposes the five factor evaluations at its final row point. The verifier checks their
product against A's derived output. Five degree-two eq-weighted column reductions then share one
eight-round `prove_batch` stream:

```text
8 rounds * 2 compressed Fr * 32 = 512 B
```

The stage's five transcript RLC coefficients are verifier-derived. At the common final point `s`,
all members use the same `T(s)`; the verifier reconstructs each final member value as
`eq(column_i, s) * T(s)`. Thus one reduced `T(s)` claim (32 B) replaces five claims (160 B).

The five A→B factor values cost 160 B for the gate's one degree-five tensor term. They are needed:
A's scalar output fixes their product, not each factor. The prior A/B/C singleton output claims
are all derived and no longer serialized (−96 B).

## Stage C and multilinear opening

Stage C has `log2(rows*k)` degree-two rounds: 20 at `k = 8`, 21 at `k = 16`. It reduces the one
`T(s)` claim to the eq-weighted commitment RLC. HyperKZG then opens the resulting packed
polynomial at the same stage-C point:

```text
ell G1 + 3*ell Fr = 4*ell*32
ell=20: 2,560 B
ell=21: 2,688 B
```

## Measured release gate

16 GiB M4 mini, one heavy binary, isolated target/worktree:

| `k` | setup | commit | proof after commit | commit + proof | verify | payload |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 14.712 s | 2.056 s | 8.220 s | 10.276 s | 0.122 s | 7,232 B |
| 16 | 28.291 s | 1.454 s | 11.614 s | 13.068 s | 0.042 s | 6,944 B |

The same run built both SRS sizes sequentially. The prior quieter baseline was 1.013 s commit +
4.948 s proof at `k = 8`; host load in this run also roughly doubled SRS generation and commit
time, so these absolute deltas are not an isolated protocol microbenchmark.
