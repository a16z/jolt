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

The fixed-chunk candidate verified twice at T28. RAM GPU-active time reproduced at
2.145--2.148 s, 0.82 s below the 2.966 s parent. Complete proving measured 52.50 s
and 50.54 s; their 51.52 s mean clears the 0.5 s provisional materiality bar by
0.68 s, but the spread is not final validation evidence. Peak RSS remained
80.08 GiB. The candidate is retained at Jolt `7acb4be74`, while its failure to reach
the 1.8 s internal bar identifies hot compaction as the next bound.

Before adding a second state plane, the next candidate widens only the compaction
threadgroup from 256 threads to the largest SIMD-aligned width supported by the
pipeline, capped at 1,024. It performs the same reads, field operations, stable
prefixes, and writes. On the M4 Max, a 1,024-thread group reduces the hottest
segment's sequential chunk count by four; the scalar prefix work per input stays
constant because there are four times as many SIMD-group counts in one quarter as
many chunks. The cost is up to four times as many inactive lanes after spans shrink
below 1,024 entries and potentially one resident group per GPU core.

The width candidate predicts 1.3--1.7 s T28 RAM GPU-active and 0.4--0.9 s complete
prover saving. Its T25 sentinel must not exceed the accepted 0.246 s RAM time. Retain
it only if a T28 treatment saves at least 0.5 s from the 51.52 s provisional parent
mean with RSS below 90 GiB. A pipeline limit below 512 threads, a T25 regression, or
T28 RAM time above 1.7 s rejects the width change and admits the modeled hot-only
out-of-place count/prefix/scatter design.

The M4 Max admitted 1,024 threads. Focused parity remained exact; T25 verified in
6.39 s with 0.230 s RAM GPU-active. T28 verified in 50.46 s at 80.10 GiB with
1.823 s RAM GPU-active. This saves 1.06 s from the prior provisional mean and
0.325 s from the RAM sequence, so `e3bd59d3b` is retained as an intermediate. It
misses the 1.7 s terminal RAM bar. The out-of-place design remains eligible, but its
roughly 1.25 s absolute ceiling now ranks below the 2--3.5 s deferred-opening
opportunity shared by all three workloads.

## Hot-only out-of-place compaction candidate

The next candidate replaces only the one-group-per-hot-address bind. Cold segments,
the cycle frontier, equality weights, reductions, address-tail handoff, transcript,
and verifier stay unchanged. Each bind will count parent leaders in fixed 4,096-entry
hot chunks, prefix those counts once per hot address, and scatter bound records into
the other state plane. The existing hot-message kernel then reads the completed
destination frontier. The protocol observes the same two message coefficients and
terminal address roots; no work moves outside proving.

Every hot segment keeps its primary offset and receives an auxiliary offset. One
round-parity bit, derived from the public number of completed binds, selects the
source plane for all hot segments; the destination is the other plane. A prefix
kernel saves the old length, writes deterministic exclusive chunk offsets, and then
publishes the new segment length. Scatter reads the saved length, so it cannot race
that metadata update. A chunk's first entry compares its parent with the preceding
global source entry. Thus a pair crossing a chunk boundary is emitted once by the
earlier chunk, while prefix offsets keep all outputs in cycle order. After scatter,
message evaluation reads the destination plane. Odd `log_T` leaves roots in the
auxiliary plane and must be handled explicitly at the CPU handoff.

The accepted denominator is 1.823 s GPU-active for the complete address/cycle
sequence. A later run reproduced 1.832 s: 1.565 s occurred in rounds 0--11 and
0.267 s in rounds 12--27. These command intervals include cycle work, messages, and
reductions, so 1.25 s is only an absolute ceiling (`1.823 - 0.57`), not a measured
compaction subtotal.

The prior occupancy model gives
`S_in = sum(N_0..N_27) = 1.443 billion` input records and
`S_out = sum(N_1..N_28) = 1.378 billion` bound records. Compaction must read each
input block, value, and `ra`, read two boundary words per output, and write the
52-byte SoA output record. Message evaluation must at least reread each current
block, value, and `ra`. This is

`(36*S_in + 68*S_out) + 36*S_in = 184.1 GiB`.

At the measured 412.5 GiB/s, the compulsory traffic floor is 0.446 s. Charging the
new four-byte count read to every input raises it by only 0.013 s; cold entries do
not pay that pass. Bind plus message requires at least
`6*S_out = 8.27 billion` full fp128 multiplications, or 0.506 s at the measured
16.36 billion multiplications/s. The unchanged cycle frontier adds about 0.06 s, so
the bottom remains mildly compute-bound near 0.57 s. Its address operational
intensity is about 0.0419 multiplication/byte; the measured bandwidth/compute rates
cross at 0.0369. Count, prefix, and scatter use 256-thread groups and less than
1 KiB static threadgroup memory, avoiding the one-group-per-core occupancy failure
seen in the rejected coefficient-fusion experiment.

The auxiliary allocation is five SoA buffers: block `u32`, previous/next `u64`, and
two 16-byte fp128 fields, exactly 52 bytes per initially hot entry. Even charging all
65,195,206 accesses gives 3,390,150,712 bytes (3.157 GiB). Counts, offsets, saved
lengths, and hot descriptors add under 0.2 MiB at the conservative 18,310-chunk
bound. Added to the accepted 82.03 GiB peak, the worst projection is 85.19 GiB,
leaving 4.81 GiB under the guard. Capacity rejection remains legal before round zero;
once selected, the route fails closed.

Prediction: RAM GPU-active time falls to 0.75--1.15 s and complete BTreeMap T28
saves 0.6--1.0 s. Exact lockstep parity must cover multiple hot chunks, a pair split
at the 4,096-entry boundary, odd live lengths, cold and hot segments together, and an
odd `log_T` final auxiliary root. Pipeline telemetry must show the multi-group route,
256 threads, and at most 1 KiB static threadgroup memory. Reject before T28 if the
verified T25 sentinel exceeds 0.260 s RAM GPU-active. At T28, reject or revert if the
complete prover exceeds 48.92 s against the accepted 49.42 s mean, RAM GPU-active
exceeds 1.25 s, RSS exceeds 90 GiB, or any fallback or exactness check fails. A
retained intermediate is not a finished kernel: closing this component still
requires at most 0.71 s, 80% efficiency against the 0.57 s floor.

Atomic output reservation was rejected because threadgroup arrival order would
scramble the sorted frontier. Fixed padded output slices avoid a prefix but turn one
contiguous segment into a fragmented tree and carry empty capacity through later
rounds. A global segmented scan touches cold entries and adds a larger scheduling
surface. The per-address prefix is the smallest design that gives deterministic,
disjoint multi-group output ownership.

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
