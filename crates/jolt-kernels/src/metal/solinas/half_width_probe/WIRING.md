# Half-width Solinas primitive contract

Status: executable screening primitive. All 15 entry points are registered,
the independent-oracle parity suite passes, and Criterion covers the chain
matrix. Downstream promotion remains blocked on compiler occupancy/spill
evidence and matched alternating confirmation.

## Layer boundaries

| File | Responsibility |
|---|---|
| `abi.rs` | Stable two-word operand ABI and the 15 probe entry-point names |
| `oracle.rs` | Independent 32-bit-limb arithmetic and Akita-field reference outputs |
| `model.rs` | Checked geometry, source-shape budget, occupancy floor, and fail-closed gates |
| `shader.metal` | The candidate primitive and pointwise/chain probe kernels |
| `runtime.rs` | Checked Metal preparation, zero-allocation dispatch, timestamps, and readback |
| `WIRING.md` | Evidence protocol and downstream promotion policy |

Keep these boundaries when the probe is integrated. In particular, GPU output
must be checked against `reference_outputs`; shader code is not its own oracle.

## Arithmetic contract

The promoted primitive is specialized to Akita's field

```text
c = 0xffff_a7f7 = 2^32 - 22537
p = 2^128 - c
```

and accepts a canonical coefficient `0 <= a < p`. Its operand domains are:

| Domain | Meaning | Valid encoding |
|---|---|---|
| Unsigned | `a * m` | `(m, 0)`, `0 <= m <= u64::MAX` |
| Signed magnitude | `a * (-1)^s * m` | `(m, s)`, `s in {0,1}`, no negative zero |
| Unsigned delta | `a * (x - y)` | `(x, y)`, both full-range `u64` |

The delta domain never narrows through `i64`; `|x-y|` can be `u64::MAX`.

For `a * m = L + H * 2^128`, `H < 2^64` and `2^128 = c (mod p)`, so the first
fold is `L + Hc`. Since `Hc < 2^96`, this fold carries at most one bit past
128. If it does carry, the wrapped residue is below `2^96`; folding that one
bit as another `c` cannot overflow 128 bits. One final add-`c`-and-select is
therefore sufficient to canonicalize the result. The shader deliberately has
no second-carry correction path. `oracle.rs` records both carries and contains
fixtures that reach the first-carry and canonical-correction paths.

This proof assumes a canonical 128-bit coefficient and a 64-bit magnitude.
Arbitrary field operands, wider integers, and already-bound multilinear state
must use the full-width path.

## Compiler-visible shape

The source has explicit limbs rather than a dynamically indexed schoolbook
loop. Its multiplication-expression budget is:

| Primitive | Coefficient products | High-fold products | Carry-fold products | Total |
|---|---:|---:|---:|---:|
| 128-by-64 | 8 | 2 | 1 | 11 |
| 128-by-128 control | 16 | 4 | 1 | 21 |

The source-level ratio is `21 / 11 = 1.909x`. The 1.60x Metal-over-Metal gate
therefore asks for about 84% of the ideal product-expression reduction before
accounting for carry, selection, and scheduling overhead. It is not a measured
instruction ratio.

Before promotion, inspect emitted AIR/ISA and reject a candidate if any of the
following appears in the hot helper:

- a call or loop backedge around the eight coefficient products;
- generic 64-by-64 multiply emulation introduced by the `ulong` casts;
- thread-local or stack loads/stores;
- divergent sign or canonical-correction branches;
- more than eight variable coefficient-product sequences or three
  offset-product sequences, unless the compiler artifact explains an
  equivalent strength reduction.

The chain kernels use explicitly named lanes, not arrays indexed in the hot
loop. This makes ILP 8 inspectable, but does not prove that ILP 8 avoids spills.

## Register and occupancy risk

`HalfWidthRegisterFloor` is a structural liveness floor in 32-bit-word
equivalents. It is not a physical register count. With a conservative 12-word
shared helper scratch floor, the chain estimates are:

| Domain | Persistent words/chain | ILP 1 | ILP 2 | ILP 4 | ILP 8 |
|---|---:|---:|---:|---:|---:|
| Unsigned | 6 | 18 | 24 | 36 | 60 |
| Signed magnitude | 7 | 19 | 26 | 40 | 68 |
| Delta chain after one-time magnitude extraction | 7 | 19 | 26 | 40 | 68 |

The pointwise delta kernel still consumes both endpoints. The chain extracts
magnitude and sign once, matching the full-width control's one-time factor
preparation and keeping endpoint conversion out of the repeated multiply roof.

Pipeline limits expose thread execution width, maximum threads, and static
threadgroup memory, but not register allocation, resident SIMD groups, or
spills. Capture those from compiler artifacts and Instruments. Promotion
requires no spills and at least two resident threadgroups per core; four is the
target. Tune ILP and threadgroup width together rather than assuming maximum
ILP wins.

## Traffic assumptions

All domains deliberately share a 16-byte operand allocation. Per element, the
probe allocates/addresses 16 bytes of coefficient, 16 bytes of operand, and 16
bytes of output. The minimum meaningful one-pass traffic differs:

| Domain | Semantic bytes/element | `2^20` bytes | Floor at 451.701710520 GB/s |
|---|---:|---:|---:|
| Unsigned | 40 | 40 MiB | 92,856 ns |
| Signed magnitude | 41 | 41 MiB | 95,178 ns |
| Unsigned delta | 48 | 48 MiB | 111,427 ns |

The allocated working set is 48 MiB plus parameters at `2^20`. A one-pass
pointwise kernel is traffic-bound at no more than roughly 11.29, 11.02, or 9.41
billion operations/s for the three rows above. It cannot satisfy the arithmetic
roof gate and must be reported as an integration/traffic control.

The padded operand is a probe ABI for apples-to-apples controls, not a required
production layout. A fused downstream kernel should consume an existing `u64`
column directly (8 bytes/element), retain resident values, or use a separate
packed sign plane. It must not materialize a 16-byte operand solely to call this
helper.

The 512-iteration chain loads each coefficient and factor once, performs
536,870,912 useful products, and stores once. Its roughly 0.1 ms one-pass
traffic floor is negligible beside the 20.435099 ms arithmetic target. Timings
must exclude allocation, upload, compilation, command submission latency, and
readback; use GPU-active timestamps.

## Parity and throughput gates

1. Validate the ABI and canonical coefficient range on the host.
2. Compare every pointwise and chain entry point with `reference_outputs`.
   Cover `0`, `1`, `p-1`, every 32/64-bit limb boundary, both signs, both
   endpoint orders, `u64::MAX` magnitude, deterministic random vectors, and
   chain iteration counts 1, 3, and 7.
3. Use a full-width chain with the same explicit-lane topology as the arithmetic
   control. Its rhs buffer is also 16 bytes/element. The existing full-width
   chain is usable only if emitted code proves its lane arrays are unrolled and
   spill-free; otherwise add an explicit-lane control before timing. For signed
   and delta controls, encode the equivalent canonical field factor on the
   host. This favors the control by omitting sign conversion, so a passing
   half-width ratio is conservative.
4. At `2^18`, independently tune ILP 1/2/4/8 and threadgroup width for each
   half-width domain and for the full-width control. Freeze each winner before
   confirmation.
5. Interleave control/candidate samples at `2^16`, `2^18`, and `2^20`; use
   `2^18` and `2^20` as saturated-size gates. Report medians and relative MAD.
6. Require all of the following at both saturated sizes:
   - exact parity;
   - the emitted-code shape above for the candidate and equivalent unrolling
     for the control;
   - no spills in either path and at least two resident threadgroups/core;
   - at least 26.272 billion useful products/s;
   - at least 1.60x the fastest same-run full-width control;
   - relative MAD at most 3%.

At `2^20 * 512`, the absolute floor is exactly 20,435,099 ns GPU-active after
rounding up. The planning constant of 16.42 billion full-width products/s does
not replace a same-run control. If the artifact or measurements show clear
headroom beyond 1.60x, continue optimizing rather than stopping at the floor.

This primitive gate is not the project's 5x optimized-CPU PIOP gate. Every
downstream fused kernel must still demonstrate its own parity and at least 5x
end-to-end PIOP improvement, including transfers and the chosen CPU/Metal
switchover.

## Observed screening result

The 2026-08-06 M4 Max screen used 512 dependent products per element, ILP 1,
and 256 threads per threadgroup. Median GPU-active throughput was:

| Elements | Unsigned | Signed magnitude | Unsigned delta | Full-width control |
|---:|---:|---:|---:|---:|
| `2^18` | 84.305 G/s | 68.480 G/s | 69.808 G/s | 44.650 G/s |
| `2^20` | 86.012 G/s | 69.998 G/s | 71.367 G/s | 45.435 G/s |
| `2^21` | 86.592 G/s | 70.417 G/s | 71.855 G/s | 45.709 G/s |

ILP 1 beat ILP 2/4/8 for every domain at `2^20`. At `2^21`, 128 and 256
threads were effectively tied; 512 was slightly slower. Every half-width
domain clears the 26.272-G/s absolute gate. Unsigned clears the standalone
1.60x relative gate. Signed magnitude and delta do not clear 1.60x in
isolation against the deliberately favorable full-width control.

The Registers RW raw-linear census contains 216,828,872 unsigned products and
197,132,288 delta products. At the `2^21` rates, that mixture sustains 78.887
G/s, or 1.726x the full-width control. Including its 73,449,472 full products
gives a 6.854-ms arithmetic projection; the existing 20.952-ms traffic floor
therefore dominates. This is a design-screen result, not a complete-kernel
speedup claim.

## Downstream promotion policy

| Candidate | Policy after primitive promotion | Required fallback or qualification |
|---|---|---|
| Spartan shift native values | May use | Exact `u64` operand path |
| Increment claim reduction | May use | Use signed magnitude; reject noncanonical sign encodings |
| RAM output check | May use | Only weight-by-`u64` products |
| Full address suffix | May use | Only suffixes statically bounded to 64 bits |
| Instruction claim reduction | Hybrid only | Lookup/left `u64` columns may use it; right lookup/input `u128`/`i128` columns remain full width |
| Address RAF/direct | Hybrid only | Compact suffix products may use it; wider identity values remain full width |
| Product remainder | Hybrid only | Raw `u64` columns may use it; signed-128 and bound terms remain full width |
| Register read/write | Hybrid only | Raw first-round `u64` terms may use it; later bound state is arbitrary field data |
| Bytecode raw increment first message | Hybrid only | Endpoint products may use signed magnitude; a leading expression requiring 65 bits stays full width unless a separately benchmarked two-product decomposition wins |
| Register claim reduction | Retain deferred accumulator | Its 224-bit `u64` accumulation performs one final reduction; per-product reduction is expected to regress |
| Bound multilinear state | Full width required | Binding turns native values into arbitrary field elements |
| Genuine signed/unsigned 128-bit values | Full width required | The half-width range proof does not apply |

Promotion is per operation, not per kernel. Mixed-width kernels must keep the
full-width fallback even when one column adopts the primitive.

## Integration checklist

Steps 1--5 are implemented. The GPU parity suite and screening benchmark from
step 6 pass, but the final alternating confirmation is still pending. Step 7
is not complete: emitted-code, spill, and residency captures are missing.
Scoped formatting, focused nextest, and the benchmark clippy target pass; the
two repository-wide clippy modes remain for integration validation.
