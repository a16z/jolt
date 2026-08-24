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
the CPU address tail. Direct shared-buffer scatter records maximum and percentile
segment lengths. Initial spans of at most 4,096 entries use one in-place worker per
address; larger spans use one 256-thread cooperative group. The hot group uses SIMD
prefix counts to compact each chunk stably in place, then reduces its message across
the group. Ownership is fixed by initial capacity so a shrinking hot span never
falls into the scalar kernel. A fixed 4096-cycle tile owns the `inc`/Hamming frontier;
after 12 binds its at-most-65,536 roots move to the CPU tail. Both kernels fuse the
previous bind with the next message and reduce to constant-size round output on the
GPU. After all cycle variables bind, a gather returns at most `K` address entries to
the existing checked address-major code.

Generic fp128 arithmetic and reductions stay in Akita only if a reusable primitive
is actually needed there. The Jolt RAM layout and schedule remain in `jolt-kernels`.

## Fixed-chunk hot-message candidate

The next candidate changes only message production for hot address spans. The
verified one-group-per-address kernel continues to own in-place compaction. After it
finishes, a read-only kernel assigns one 256-thread group to each fixed 4,096-entry
chunk of every initially hot span. A chunk recognizes leaders against the preceding
global entry, so a pair crossing a chunk boundary is evaluated exactly once. Each
chunk writes two field-element partials; the existing column reducer sums all hot
chunks directly because the protocol observes only the sum across addresses. Cold
address partials remain a disjoint reduction. Empty fixed chunks must overwrite their
partials with zero after the live span shrinks.

At T28, `R = 65,195,206` and there are 2,392 hot addresses. The fixed descriptor
count is bounded by `R / 4096 + 2392 <= 18,310`. An 8-byte descriptor plus two
double-buffered columns of 16-byte partials costs at most `72 * 18,310 = 1,318,320`
bytes (1.26 MiB), so this candidate does not threaten the 90 GiB guard. The fixed
worklist also adds under 1 MiB of descriptor and partial traffic per round.

A message leader performs at most five full fp128 multiplications. Using the existing
uniform-frontier upper model of 1.443 billion live entries gives at most 7.22 billion
multiplications, or 0.44 s at the measured 16.36 billion multiplications/s. A
conservative 108 bytes read per leader gives 145.1 GiB, or 0.35 s at the measured
412.5 GiB/s. Compute therefore remains the lower-bound term. Chunking does not change
that work; it removes the longest address as a serial scheduling unit. Because bind
compaction still has one group per hot address, this candidate is predicted to reduce
the measured 2.97 s RAM sequence to 1.2--1.8 s, saving roughly 1.0--1.8 s from the
complete BTreeMap prover. It is an intermediate schedule, not a claim that the
0.57 s full-sequence floor has been reached.

Exact parity must cover an evaluated pair split across the 4,096-entry message
boundary as well as the existing 256-entry compaction boundary. Admit one T28 run
only if the verified T25 sentinel does not exceed the accepted 0.279 s RAM GPU time.
Retain the candidate only if T28 saves at least 0.5 s end to end with RSS below
90 GiB; treat RAM GPU time above 1.8 s as evidence that hot compaction, rather than
message scheduling, is the next bound. No protocol, variable order, claim, or
verifier change is permitted in this candidate.

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

The first T25 measurement found p99 segment length 609 but maximum length 914,071;
this falsified the original scalar-per-address schedule despite its favorable
aggregate roofline. The cooperative hot schedule reduced total RAM GPU-active time
from roughly 7.7s to 0.279s, with a 37.2ms worst round.

The first T28 attempt exposed an in-place chunk-boundary hazard: a chunk could
overwrite its final source block before the next chunk classified a pair against
that block. Carrying the original boundary parent through threadgroup memory fixed
the cause. An optimized-CPU shadow then matched every T27 round and terminal claim,
and the complete proof verified. The corrected T28 treatment verified in 52.20s,
down from the 56.34s accepted Metal parent, at 80.08GiB peak RSS. Bucketing and
setup took 1.36s and the RAM kernels used 2.97s GPU-active. The candidate is a
material improvement but misses the 0.75s kernel bar, so hot-message parallelism
must be redesigned rather than treating the current kernel as finished.
