# W4-S byte audit — implemented `2^17` G-shape

Date: 2026-09-02. The ignored release gate executes `jolt-wrapper` production code for `k = 8`
and `k = 16`. Payload counts BN254 G1 and Fr values as 32 B and exclude serde framing.

## Exact payloads

| item | `k = 8` | bytes | `k = 16` | bytes |
|---|---:|---:|---:|---:|
| packed commitments | 30 G1 | 960 | 15 G1 | 480 |
| stage A committed rounds | 17 G1 + 34 Fr + 3 G1 | 1,728 | same | 1,728 |
| stage B column batch | 8 rounds × 2 Fr | 512 | same | 512 |
| stage C opening reduction | deleted | 0 | deleted | 0 |
| A→B factor claims | 5 Fr | 160 | 5 Fr | 160 |
| reduced packed claim | 1 Fr | 32 | 1 Fr | 32 |
| multilinear HyperKZG | 20 G1 + 41 Fr | 1,952 | 21 G1 + 43 Fr | 2,048 |
| **payload** | | **5,344** | | **4,960** |
| standard-bincode framing | | 93 | | 79 |
| **standard bincode** | | **5,437** | | **5,039** |

The original `k = 8` proof was 10,304 B payload / 10,445 B bincode. Implemented payload saving:
**4,960 B**. `k = 16` saves another **384 B**.

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

## Direct multilinear opening

The shared B batch leaves one claim at one packed point:

```text
T(s) = sum_g eq(s_group,g) P_g(r_A,s_slot).
```

The verifier directly forms that eq-weighted commitment RLC and HyperKZG opens it at
`(r_A,s_slot)`. The former C sumcheck changed one point into another point and cost 1,280 B at
`k=8` / 1,344 B at `k=16`; it is deleted. HyperKZG transmits both `P_i(r)`/`P_i(-r)`
rows and only `P_0(r^2)`. The verifier reconstructs `P_1(r^2)..P_{ell-1}(r^2)` from the Gemini
fold identities before the unchanged cubic KZG check:

```text
ell G1 + (2*ell+1) Fr
ell=20: 1,952 B
ell=21: 2,048 B
```

Lane Z's single-point Zeromorph target is `ell+2` G1: 704 B at `ell=20`, 736 B at `ell=21`.
Replacing only the final PCS would therefore yield **4,096 B** (`k=8`) or **3,648 B** (`k=16`).
A three-point Zeromorph opening adds `3(ell+1)+1` G1 and is larger than the deleted C stream.

## EVM verifier operations

The ignored gate calls `verify_stream_with_cost`. Its observer sits on the executed verifier
operations: the two two-pair committed-round checks, the four-pair HyperKZG check, every G1-side
MSM/divisor operation, every chained-digest append/squeeze, and each verifier Fr multiplication.
Field arithmetic uses a multiplication shim at each operation site, including batch padding,
compressed-round evaluation, interpolation, vanishing-polynomial construction, Gemini folds, and
the cubic KZG check; there is no aggregate Fr formula.

| `k` | ecMul | ecAdd | pairing pairs | Fr mul | Keccak | N4 gas estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 109 | 108 | 8 | 6,127 | 282 | 1,545,184 |
| 16 | 95 | 94 | 8 | 6,123 | 270 | 1,422,792 |

Gas applies the N4 constants: 21k base, 16 gas/calldata byte after expanding each G1 to 64 bytes,
7.7k per paired ecMul+ecAdd MSM term, 114.7k per two-pair call, 183.4k for four pairs, and 100 gas
per chained Keccak event. The 20 gas/Fr multiplication term is the EVM-plan estimate, not an N4
measurement. The total excludes contract-code/data access and G1 decompression.

The permanent `2^12` synthetic test has an independent 3,072-multiplication trace: 90 for
compressed A, 120 for the public tensor, 2,658 for batched B, 7 for group weights, and 197 for the
final HyperKZG opening. It asserts that the execution observer returns the same total.

## Packed public challenges

Spartan proof bytes contain a canonical 16-byte decoder preimage, not a field integer. For the
125-bit decoder the prover transmits the post-mask word; the verifier rejects any word with
`bytes[15] & 0xe0 != 0` before calling the production decoder. Thus the seven alternate words that
the decoder would otherwise mask to the same field value are invalid.
The statement records a decoder for each slot:

```text
Challenge125: Fr::from_challenge_bytes(raw)        # v * 2^-128, 125-bit squeeze
Scalar128:    Fr::from_scalar_challenge_bytes(raw) # 128-bit big-endian scalar
```

Packing recovers the canonical word using the inverse of that production decoder and rejects any
value that does not round-trip. Verification calls the same `jolt-field` decoder selected by the
statement. The 28 challenge slots remain 448 B. A real `RecordingTranscript` fixture covers both
decoder kinds; tests reject a low-bit change and all seven noncanonical high-bit aliases.

## Measured release gate

16 GiB M4 mini, one heavy binary, isolated target/worktree:

| `k` | setup | commit | proof after commit | commit + proof | verify | payload |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7.254 s | 1.156 s | 1.669 s | 2.825 s | 0.007 s | 5,344 B |
| 16 | 15.750 s | 1.434 s | 2.810 s | 4.244 s | 0.006 s | 4,960 B |

The same run built both SRS sizes sequentially. The setup/prove spread across runs tracks concurrent
host load; the byte and verifier-operation counts are deterministic.
