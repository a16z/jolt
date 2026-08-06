# Spartan product-remainder Metal contract

This directory owns the dense rounds and final eight openings of
`ProductRemainder`. Product uni-skip, transcript operations, round-polynomial
completion, and the CPU tail remain outside this boundary.

## Exact boundary

The device receives one native 40-byte row per trace cycle, the three
uni-skip Lagrange weights, and the current split-equality tables. It owns two
resident left/right state planes, two equality-table buffers, and reusable
column-major reduction scratch. A round returns two field elements:
`q(0)` and the quadratic coefficient `q_infinity`. These are not a complete
round polynomial. The host must call
`GruenSplitEqPolynomial::gruen_poly_deg_3(q_zero, q_infinity, previous_claim)`
and derive Fiat-Shamir challenges exactly as the CPU prover does.

The first command materializes the two relation tables and emits the first
dense-round endpoints. Each later command binds both tables by one host
challenge and emits the next endpoints in the same pass. After the final
challenge, the device evaluates the eight raw witness columns at
`EqPolynomial::evals(challenges.iter().rev(), None)`. The high-level wrapper
must enforce that reversed opening-point order.

The target workload is Fibonacci at `T = 2^26`. The frozen optimized-CPU
complete-member median is 433.565832 ms, so the hard 5x wall cap is
86.713166 ms. This comparison includes every Metal-only row conversion or
upload, allocation, command submission, synchronization, readback, CPU tail,
and host Fiat-Shamir operation. GPU-active timing is diagnostic only.

## Layout and traffic floor

`ProductRemainderRow` is five little-endian `u64` words with a 40-byte stride.
At `T = 2^26`, the row plane is 2.5 GiB. State A holds two `T`-element field
tables (2 GiB); state B holds their first bound result (1 GiB). With two
8192-field equality tables and two eight-column partial buffers, the complete
resident allocation is about 5.5022 GiB. It is allocated before the first
dispatch and no round allocates a device buffer.

With a hybrid cutoff of `2^16`, compulsory traffic is approximately:

| Phase | Compulsory traffic | Floor at 420.68 GiB/s |
| --- | ---: | ---: |
| Materialize + first message | `72T` bytes = 4.5 GiB | 10.70 ms |
| Dense transitions | `48 * sum(N)` bytes, about 6.0 GiB | 14.25 ms |
| Eight openings | `40T` bytes = 2.5 GiB | 5.94 ms |
| Total | about 12.994 GiB | 30.89 ms |

This is cache-optimistic for the small equality tables and omits negligible
final-result traffic. Re-reading or reformatting the 40-byte row plane inside
this member must be charged separately.

## Compute floor and falsification bars

The materialization pass performs `5T + 2H` useful full-field products:
three relation products per row and four weighted endpoint products per row
pair. At target scale this is 335,560,704 products. Dense transitions perform
roughly 268.34 million products through the `2^16` cutoff. The opening scan
performs `3T + 8H = 201,392,128` products. Against the measured 18.1
Gproduct/s distributed arithmetic control, the compute floors are about
18.54, 14.83, and 11.13 ms. Compute therefore binds materialization and
openings; transitions sit near the compute/traffic crossover.

The pre-registered 80%-of-roof active-time gates are:

| Phase | GPU-active gate |
| --- | ---: |
| Materialize + first message | <= 23.17 ms |
| All dense transitions to `2^16` | <= 18.53 ms |
| Eight openings | <= 13.91 ms |
| Total GPU-active | <= 55.61 ms |

Promotion additionally requires complete-member wall time <= 86.713166 ms.
If the complete member has clear headroom beyond 5x, optimization continues
toward the measured roof rather than treating 5x as a stopping cap.

## Adjustment candidates

1. Reuse the row allocation produced for product uni-skip. A second 2.5-GiB
   conversion or upload is unlikely to fit the complete-member budget.
2. Keep relation materialization fused with the first message and every bind
   fused with its successor message; separate passes add at least one full
   state read/write.
3. Keep split equality tables and Fiat-Shamir on the host. Their uploads and
   32-byte round readbacks are small, while moving transcript state to Metal
   would enlarge the protocol boundary without reducing the dominant scans.
4. If the eight-opening kernel spills, split it into two four-column scans
   only after codegen or timing proves that the extra 2.5-GiB row read wins.
5. Select the CPU switchover by paired complete-member measurements. Small
   late rounds cannot amortize command latency even though the resident state
   avoids allocation.

## Promotion evidence

Before backend wiring, require exact Akita parity for materialization, every
A-to-B-to-A transition, and all eight openings at even and odd log sizes.
Vectors include `i128::MIN`, `i128::MAX`, zero, mixed flags, and challenges
`0`, `1`, `p-1`, and random nontrivial values. Report at least three scales,
including `2^26`, with identical warmup/repetition and wall boundaries for CPU
and Metal. Retain raw phase observations, pipeline limits, device identity,
zero-round-allocation evidence, and generated-code register/spill evidence.
