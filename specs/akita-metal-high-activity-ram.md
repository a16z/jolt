# Address-segmented Metal RAM read/write checking

## Decision

High-activity RAM read/write checking will keep the existing sumcheck relation,
variable order, transcript, claims, and verifier. The Metal route will bucket RAM
accesses once by address, then bind each address's cycle frontier independently and
in place. A separate tiled cycle frontier supplies the `inc * hamming` term. This
replaces the current global `(cycle block, address)` host merge without changing the
proved polynomial.

The route is selected from public geometry and the measured access count. Tiny traces
retain the existing host-sparse path; unsupported certificates or capacity failures
fall back before round zero. A qualified route must fail closed after it begins.

## Contract and invariants

The input is the in-prove RAM projection over `T` cycles: address `u32`, pre/post
values `u64`, the committed increment column, and `K` final-memory words. The output
is byte-identical Stage-2 round polynomials and the existing `val`, `ra`, and `inc`
opening claims. Preparation, allocation, bucketing, command submission, waits,
readback, and the CPU address tail remain charged to proving.

For each address, records are stable-sorted by cycle. After `r` low-to-high cycle
binds, its frontier contains exactly one entry for every `2^r`-cycle block touching
that address, ordered by block. Binding adjacent low/high blocks preserves the raw
pre/post boundary checkpoints and interpolates `val` and `ra` exactly as
`CycleMajorEntry::bind`. Each address owns a fixed-capacity span, so a single owner
may compact its output toward the span's start without a global prefix scan or race.

The relation is evaluated as

`(1 + gamma) * eq * sum_k(ra_k * val_k) + gamma * eq * inc * h`,

where `h = sum_k ra_k`. This is only distributivity of the current polynomial; it
does not replace `inc * h` by `inc`. The latter would need an independently enforced
support constraint and is out of scope.

## Measured denominator and roofline

The frozen BTreeMap `T=2^28` Metal run spends 8.86 s in Stage-2 RAM read/write
checking; a later traced A0 diagnostic measured 9.42 s. A0 observed
`R=65,195,206`, `K=2^19`, exact Hamming, RAM-RA compatibility, and increment
compatibility. The current fallback's 47 rounds spend effectively all round time in
the 28 cycle rounds; its 19 address rounds total about 25 ms.

On the same M4 Max, sentinel-checked calibrations measured 412.5 GiB/s for a 20 GiB
Metal read/write stream and 16.36 billion full fp128 multiplications/s with four
independent chains and eight live fp128 operands. These are observed microkernel
rates, not application predictions.

Let `N_r` be the number of live per-address entries after `r` cycle binds. A
random-uniform occupancy model gives `sum(N_0..N_27)=1.443 billion`; skew can only
reduce distinct `(block,address)` pairs for fixed access count, although the model is
not a proof of the actual trace distribution. A fused in-place schedule moves
approximately `56 * (2*sum(N_0..N_27) + N_28) = 150.6 GiB`, plus at most 10.4 GiB
for counting and direct scatter. Its traffic floor is therefore about 0.39 s.

Address-major evaluation deliberately pays two extra head-weight products per live
output to avoid the global merge. Bind plus message needs at most
`6*sum(N_1..N_28) = 8.27 billion` fp128 multiplications under the same occupancy
model, a 0.51 s compute floor. The cycle-only frontier has an estimated 231 million
live-block rounds and adds about 0.06 s of fp128 work. The modeled bottom is thus
compute-bound near 0.57 s before launch, reduction, and readback latency. One
56-byte address plane is 3.40 GiB; one 40-byte cycle plane is 2.43 GiB. Replacing the
fallback's 3.40 GiB matrix projects A0's 79.88 GiB peak near 82.4 GiB, below the
90 GiB guard without a second plane.

## Mechanism and ownership

Jolt owns stable bucketing, RAM certificates, routing, transcript round trips, and
the CPU address tail. The first implementation uses direct shared-buffer scatter and
one in-place worker per address; it records maximum and percentile segment lengths so
hot-address skew is visible. A fixed 4096-cycle tile owns the `inc`/Hamming frontier;
after 12 binds its at-most-65,536 roots move to the CPU tail. Both kernels fuse the
previous bind with the next message and reduce to constant-size round output on the
GPU. After all cycle variables bind, a gather returns at most `K` address entries to
the existing checked address-major code.

Generic fp128 arithmetic and reductions stay in Akita only if a reusable primitive
is actually needed there. The Jolt RAM layout and schedule remain in `jolt-kernels`.

## Falsification and validation

Before promotion, the address/cycle kernels must measure at most 0.75 s GPU-active at
BTreeMap T28 (80% roofline efficiency against the 0.57 s modeled bottom), and the
complete Stage-2 boundary must be at most 2.5 s. Reject or redesign if direct
bucketing exceeds 1.25 s, peak RSS exceeds 90 GiB, a hot segment creates more than
0.25 s of GPU tail latency, or one warm T28 treatment saves under 0.5 s end to end.

Focused parity must compare every round polynomial, terminal claim, and derived
table against the optimized kernel for empty, singleton, repeated-address,
alternating-address, read-only, write, and maximum-local-span fixtures. A full proof
must verify, report the qualified route with no fallback, and retain the unchanged
Fibonacci sparse route. The first BTreeMap treatment is one warm candidate-only run
under the frozen evaluator; SHA-2 is tested only after BTreeMap promotes.

Unverified before implementation: the actual BTreeMap `N_r` sequence, address-count
skew, direct-scatter cost, and Metal compiler occupancy. The candidate must emit
these counters; a missed bar updates or kills the mechanism rather than silently
weakening it.
