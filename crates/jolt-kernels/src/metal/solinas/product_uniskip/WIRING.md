# Spartan product uni-skip wiring contract

This directory implements the low-level first round of
`SpartanProductUniskip`. It fixes the two-node algebra, the reusable row ABI,
the shader entry points, checked storage accounting, an optimized CPU mirror,
and an independent CPU oracle. It is connected to the shared Metal source
library and Criterion harness, but not yet to the prover backend.

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

The shared source library concatenates these sources in order:

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
nonzero multiple of 32. The measured default is 64 threads. Dynamic
threadgroup memory is
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

`prepare_product_remainder_rows` uploads and validates the native rows once.
Both `prepare_product_uniskip` and
`prepare_product_remainder_sequence_with_rows` retain that same Metal buffer;
the parity test compares its allocation identity at both consumers. The host
draws `tau_high`, waits for the two endpoint results, assembles the five `t1`
values, constructs the full degree-six round polynomial, and runs the existing
clear or ZK uni-skip proof. The transcript absorbs the same polynomial as the
CPU path; this is a prover implementation change, not a protocol change.

## Fair CPU rebaseline

The production optimized CPU implementation still computes all five
extended-node sums even though stage 1 already supplied three of them.
Comparing against that avoidable work would overstate the backend gain. The
benchmark therefore uses `evaluate_product_uniskip_extensions_cpu`, a parallel
two-node integer-accumulator implementation of the same work as Metal, and
checks it against the independent field oracle before timing. Prover promotion
still requires the same algebraic reduction in the production CPU path:

1. Change `UniskipKernel::first_round_poly` in `uniskip.rs` to accept known
   base-domain evaluations.
2. In `optimized/spartan_product.rs`, replace the five-node
   `extended_t1_values` output with the two endpoints and carry only those two
   accumulators.
3. Have the stage-2 call in `stage2.rs` pass the three stage-1 claims. Keep the
   stage-1 caller on the existing no-known-nodes behavior.
4. Assemble all five values immediately before interpolation and preserve the
   reference implementation as a parity oracle.
5. Confirm the production member against the low-level two-node CPU median.

The frozen five-node production CPU median is 293.429 ms. At `T = 2^26`, the
measured two-node CPU median is 161.46 ms, giving a fair 5x wall cap of 32.292
ms and an 8x cap of 20.183 ms. The retained CPU-first Metal wall median is
11.080 ms, or 14.57x. A short Metal-only screen measured 9.537 ms at 64
threads. That standalone number selects the launch shape; the CPU-first result
is the conservative ratio.

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

## Log-26 observation

The native row allocation is 2,684,354,560 bytes. Equality and reduction
scratch bring the retained set to 2,685,140,992 bytes. The retained CPU-first
Criterion run measured 24.229 billion useful products/s from the 11.080 ms
wall median. The
short launch screen was:

| Threads | Resident wall median |
| ---: | ---: |
| 32 | 9.647 ms |
| 64 | 9.537 ms |
| 256 | 10.018 ms |
| 512 | 9.692 ms |

At 64 threads, dynamic threadgroup storage is only 64 bytes, static
threadgroup storage is zero, the device SIMD width is 32, and 8,192 independent
threadgroups expose ample grid parallelism. Threadgroup memory and grid size
therefore do not limit theoretical residency. Metal does not expose the final
register allocation through this harness, so register-limited occupancy still
requires a captured pipeline artifact before calling occupancy proven.

The measured host-to-Metal row upload was 185.084 ms and is not included in the
resident ratio. Paying it inside this member would make the path slower than
the fair CPU kernel. Promotion therefore depends on the upstream producer
filling the shared resident row allocation, not on hiding or amortizing a copy
inside this benchmark.

## Deliberately unfinished promotion work

- kernel-registry and high-level backend wiring;
- producer-owned construction of the shared ProductUniskip/ProductRemainder
  row allocation;
- the production uni-skip API change to accept the known three base nodes;
- clear and ZK CPU/Metal parity integration tests;
- host polynomial/Fiat-Shamir timing and complete-member hybrid measurement;
- register capture, hybrid switchover, and log-27 transfer validation.

Those omissions are explicit promotion stages. The low-level Rust, MSL,
independent oracle, optimized CPU mirror, source assembly, direct allocation
handoff, edge-case parity, and log-26 Criterion measurement are complete.
