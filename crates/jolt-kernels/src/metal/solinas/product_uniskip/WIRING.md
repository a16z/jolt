# Spartan product uni-skip wiring contract

This directory is an isolated implementation slice for the first round of
`SpartanProductUniskip`. It fixes the two-node algebra, the reusable row ABI,
the shader entry points, checked storage accounting, and a CPU oracle. It is
not yet connected to the shared Metal source library or prover backend.

## Two-node first round

The uni-skip domain is `{-1, 0, 1}`. For one trace row, let

```text
u = (left_instruction_input, lookup_output, should_jump)
v = (right_instruction_input, should_branch, 1 - next_is_noop).
```

Stage 1 already supplies the product evaluations at the three domain nodes:

```text
t1(-1) = product
t1( 0) = should_branch
t1( 1) = should_jump.
```

The degree-four product polynomial therefore needs only the two remaining
values. The device evaluates the Lagrange coefficient rows

```text
-2: [3, -3, 1]
 2: [1, -3, 3]
```

and returns `[t1(-2), t1(2)]`. The host assembles the centered evaluation array
`[t1(-2), t1(-1), t1(0), t1(1), t1(2)]`, interpolates `t1`, multiplies it by
the degree-two `K_tau` factor, and continues through the existing uni-skip
proof path. Fiat-Shamir remains entirely on the host.

For each row, this schedule performs two relation products and two `E_in`
weight products. The block epilogue adds two `E_out` products per block. It
does not materialize either relation polynomial or any five-node row table.

## Reused row ABI

Buffer 0 is exactly `product_remainder::ProductRemainderRow`; product uni-skip
must retain and reuse that allocation rather than translate or copy it. Its
40-byte, eight-byte-aligned layout is:

| Word | Meaning |
| ---: | --- |
| 0 | left instruction input as `u64` |
| 1, 2 | little-endian magnitude of the signed right instruction input |
| 3 | lookup output as `u64` |
| 4, bit 0 | right input is nonnegative |
| 4, bit 1 | should jump |
| 4, bit 3 | should branch |
| 4, bit 4 | next instruction is noop |

The shader intentionally consumes the struct and flag helpers declared by
`product_remainder/shader.metal`. The unused product-remainder flag bits remain
part of the shared ABI.

## Source and pipeline registration

When promoted, concatenate sources in this order:

1. `fp128.metal`
2. `simd_reduce.metal`
3. `product_remainder/shader.metal`
4. `product_uniskip/shader.metal`

Register these entry points:

| Entry point | Purpose |
| --- | --- |
| `solinas_product_uniskip_extended_blocks2` | Compute the `-2` and `2` block partials in one native-row scan |
| `solinas_product_uniskip_reduce2` | Recursively reduce two column-major partial arrays |

The block pipeline buffer order is:

```text
0 shared ProductRemainderRow allocation
1 E_in
2 E_out
3 two-column partial output
4 ProductUniskipBlockParams
```

Dispatch exactly `e_out_length` threadgroups. Threads per threadgroup must be a
nonzero multiple of 32. Dynamic threadgroup memory is
`2 * (threads_per_threadgroup / 32) * sizeof(Fp128)`.

The reduction pipeline buffer order is:

```text
0 two-column input
1 two-column output
2 ProductUniskipReductionParams
```

For each reduction level, set `output_count = ceil(input_count / 32)`, dispatch
`output_count * 32` threads, and swap the two scratch buffers. At the target
`e_out_length = 8192`, the levels are `8192 -> 256 -> 8 -> 1`. Both columns can
remain in one command buffer. The final device-to-host result is two field
elements, or 32 bytes.

## Host sequence and ownership

For `T = 2^26`, use the exact split `E_in.len() = E_out.len() = 8192`; the Rust
shape constructor requires `E_in.len() * E_out.len() = T`. The row allocation
is 2.5 GiB. Equality tables and both two-column partial buffers use 0.75 MiB in
total. Retain the row allocation after this kernel because ProductRemainder
consumes the same rows.

Kernel preparation may upload the rows and equality tables before the prover
reaches the first-round barrier. The host draws `tau_high`, waits for the two
endpoint results, assembles the five `t1` values, constructs the full
degree-six round polynomial, and runs the existing clear or ZK uni-skip proof.
The transcript absorbs the same polynomial as the CPU path; this is a prover
implementation change, not a protocol change.

## Fair CPU rebaseline

The current optimized CPU implementation also computes all five extended-node
sums even though stage 1 already supplied three of them. Comparing a two-node
Metal kernel against that avoidable work would overstate the backend gain.
Before accepting a result, make the same algebraic reduction on CPU:

1. Change `UniskipKernel::first_round_poly` in `uniskip.rs` to accept known
   base-domain evaluations.
2. In `optimized/spartan_product.rs`, replace the five-node
   `extended_t1_values` output with the two endpoints and carry only those two
   accumulators.
3. Have the stage-2 call in `stage2.rs` pass the three stage-1 claims. Keep the
   stage-1 caller on the existing no-known-nodes behavior.
4. Assemble all five values immediately before interpolation and preserve the
   reference implementation as a parity oracle.
5. Measure a new same-machine CPU median before setting the final speedup gate.

The frozen pre-optimization CPU median is 293.429 ms. Linear work scaling
projects the two-node CPU path near 117.387 ms, but that number is not evidence
and must not become the denominator. The final hard gate is
`metal_hybrid <= measured_reoptimized_cpu / 5`. The old frozen 5x gate is
58.686 ms. Until the fair CPU rebaseline exists, use 20.435 ms as the active
implementation bar. The projected fair-CPU 5x and 8x limits are 23.477 ms and
14.673 ms, respectively. If measurements show that substantially more than 5x
is feasible, continue toward the measured roof rather than stopping at 5x.

## Roofline at `T = 2^26`

The compulsory native-row scan reads 2.5 GiB. At the measured 420.68 GiB/s
device bandwidth, that traffic takes 5.943 ms. The main loop performs
`4T = 268,435,456` full Solinas products, plus 16,384 block-weight products.
At 32.33 billion products/s, its arithmetic floor is 8.303 ms; at the
conservative 16.42 billion products/s rate, it is 16.348 ms. The first pass is
therefore expected to be compute-bound under the conservative roof. Reduction,
submission, synchronization, 32-byte readback, host interpolation, host
Fiat-Shamir, and any row preparation owned by this seam remain additive costs.

Benchmark accounting must include every cost unique to selecting Metal:
native-row production or conversion, uploads, command encoding, GPU work,
synchronization, endpoint readback, host polynomial construction, and host
Fiat-Shamir. Exclude witness generation and stage-1 claims shared by both
implementations. Report GPU-active time separately, but enforce the 5x gate on
the complete PIOP seam.

## Deliberately unfinished promotion work

- shared `solinas/mod.rs`, `source.rs`, and kernel-registry wiring;
- runtime pipeline creation, checked dispatch, and command-buffer ownership;
- asynchronous preparation and ProductRemainder allocation reuse;
- the uni-skip API change and fair two-node CPU rebaseline;
- clear and ZK CPU/Metal parity integration tests;
- Criterion stage benchmarks and hybrid switchover measurement;
- occupancy, command-latency, and `2^26` GPU validation.

Those omissions are explicit promotion stages. This isolated slice has not
been compiled or run.
