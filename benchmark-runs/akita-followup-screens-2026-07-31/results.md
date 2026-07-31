# Akita large-trace follow-up results

Date: 2026-07-31
Machine: Apple M4 Max

## Result

Four follow-up candidates were screened. None passed the promotion gate, so
there is no retained production-code change in this batch:

1. D256 reduces setup capacity but increases the dominant commitment
   coefficient traffic by 33.3%.
2. A fused fp128 affine bind is fast in isolation, but its measured
   whole-proof ceiling is below 0.3 seconds at `T = 2^28`.
3. Replacing the Booleanity product accumulator with the multi-lane Solinas
   accumulator is flat.
4. Virtualizing the nine fused-increment lane columns is sound and removes a
   named 2.2 GiB owner at `T = 2^28`, but both tested encodings slow Stage 6b
   enough to fail the CPU-neutral memory guard.

All candidate code was reverted. The traces are retained as negative
controls.

## D256 planner audit

The exact packed layout is 41 variables and one physical polynomial. The
current D128 schedule and the useful D256 Pareto points are:

| D | log basis | rank | `rank * D` | positions | live blocks | payload | setup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6 | 3 | **384** | `2^21` | `2^13` | 130,056 B | 12.0 GiB |
| 256 | 4 | 2 | 512 | `2^19` | `2^14` | 173,736 B | 4.0 GiB |
| 256 | 6 | 2 | 512 | `2^18` | `2^15` | 179,624 B | 5.5 GiB |

No D256 schedule reaches rank 1. In the K256 root loop, each trace row has one
active ring contribution. The streamed shift-accumulate work is therefore
proportional to `T * rank * D`, not just rank. D256 changes 384 live
coefficients per row to 512, a 33.3% increase. It is a possible
setup-capacity trade, but not a no-performance-regression optimization.
Implementing a new D256 backend was not warranted.

## Fp128 affine-bind screen

An exact parallel microbenchmark compared the current
`a + r * (b - a)` bind with the existing fp128 fused multiply-add path over
`2^22` inputs:

| Layout | Current | Fused | Ratio |
|---|---:|---:|---:|
| Adjacent pairs | 728.875 us | 611.208 us | 0.8386 |
| Split halves | 841.125 us | 687.958 us | 0.8179 |

The primitive signal is real, but the accepted `T = 2^28` trace contains only
0.314 seconds in the affected dense top-zero bind spans. Even a perfect
replacement cannot materially move the 160-second prover. Compact and RA
binding spans dominate the aggregate `MultilinearPolynomial::bind_parallel`
name and do not use this kernel. No production patch was retained.

## Booleanity accumulator screen

The lattice Booleanity inner fold was switched from the raw fp128 product
accumulator to the existing multi-lane Solinas accumulator. A focused
proof-equivalence test passed before timing.

| Variant | Prover | Stage 6b | Booleanity compute |
|---|---:|---:|---:|
| Raw control | 4.892803 s | 0.325285 s | 0.098718 s |
| Multi-lane A | 5.055007 s | 0.341927 s | 0.101518 s |
| Multi-lane B | 4.957738 s | 0.323107 s | 0.096300 s |
| Multi-lane mean | 5.006373 s | 0.332517 s | 0.098909 s |

The directly affected Booleanity mean is 0.19% slower than control, within the
noise floor and without a positive signal. The change was reverted.

## Fused-increment lane virtualization

### Structural opportunity

The current Stage 6/7 state holds:

```text
packed signed fused delta     8.125 T bytes
nine materialized hot lanes  9.000 T bytes
-------------------------------------------
total                        17.125 T bytes
```

The hot lanes are a deterministic view of the signed delta. Two sound virtual
representations were tested:

- direct sign-magnitude decoding, which would remove exactly `9T`;
- eight balanced lanes packed in one `u64` plus a two-bit carry tag, which
  would replace both owners with `8.25T` and remove `8.875T`.

The second form reconstructs the signed delta in constant time. If `x` packs
the centered radix digits and `H` selects each digit's sign bit, their signed
low-limb value is `x - 2(x & H)`; the two-bit tag supplies the `-1`, `0`, or
`1` carry above bit 63. Boundary values and the full Booleanity round loop
matched the materialized-column implementation.

At `T = 2^28`, `8.875T` is 2.21875 GiB. At `T = 2^26`, it is
0.5546875 GiB.

### Performance falsifier

| Variant | Prover | Stage 6a | Stage 6b | Stage 7 | Lane build | Affected total |
|---|---:|---:|---:|---:|---:|---:|
| Materialized control | 4.892803 s | 0.092610 s | 0.325285 s | 0.044948 s | 0.010840 s | **0.473683 s** |
| Direct virtual | 5.075326 s | 0.098809 s | 0.401556 s | 0.053394 s | 0 | 0.553759 s |
| Packed-balanced virtual | 5.071006 s | 0.097381 s | 0.390664 s | 0.044194 s | 0 | 0.532239 s |

The packed-balanced retry reduces the naive penalty, but its affected total
is still 12.4% slower and Stage 6b is 20.1% slower than control. The repeated
lazy-RA reads need a fused multi-lane state machine to share source loads and
binding work; changing the scalar encoding alone is insufficient. Per the
contract, the candidate did not advance to `T = 2^26` and all code was
reverted.

## Retained traces

| Trace | Purpose | SHA-256 |
|---|---|---|
| `akita_22_booleanity_raw.json` | raw-accumulator control | `2da35c33ae82739da4b1e0cfa7353c14b39f2970c53f369ab1f6f6bf473f180a` |
| `akita_22_booleanity_solinas.json` | multi-lane accumulator A | `7ff025889a0b0b3ef1979ececa6b21459148b37a6ada8a481f881ea2a84e77d7` |
| `akita_22_booleanity_solinas_repeat.json` | multi-lane accumulator B | `aaed71535ebdc625e9aca1a9e9a0bdc978f2d74ff5a6d18af27a394015981a5c` |
| `akita_22_fused_inc_virtual_raw.json` | direct sign-magnitude virtualization | `b7101bd1d1c9cf59af2f93f8e3f9cc19c03f300ba930d9b441935586218faff6` |
| `akita_22_fused_inc_virtual_packed.json` | packed-balanced virtualization | `aee2ff997a45c457cb0bbbef7277c2aabffcfa4dadb3401fc690ed11ce48527e` |

The traces live in `benchmark-runs/perfetto_traces/`.
