# RAM output-check successor design

Status: static design packet. Nothing in this directory is registered,
compiled, or measured. The shader is an experiment sketch, not performance
evidence or a production candidate.

## Decision

There is no standalone Metal kernel that can fairly beat the current optimized
CPU member by 5x at the production address size. The address domain is only
`2^13`, and the retained one-command Metal wall time is already larger than the
entire CPU member. Another standalone command is rejected before tuning.

The only GPU experiment admitted by this packet is a marginal dispatch inside a
batch-owned command/wait that would happen without `RamOutputCheck`. It folds a
producer-owned native `u64` final-memory table to eight field values after the
ten certified zero rounds. The host reduces the 64 partials and proves the
three-round tail. The current stage-2 driver does not expose such a command
owner, so integration must retain a resident CPU implementation until that
boundary exists.

This distinction is part of the speed claim:

- a new command, commit, wait, or allocation is a standalone path and is
  rejected at `K = 8192`;
- a coalesced dispatch may claim only its measured incremental GPU-active time
  and incremental host time;
- the parent command's service may be excluded only when an alternating
  ablation shows the same command/wait in the control;
- a CPU fold selected by the Metal hybrid is reported as CPU work, not as a
  Metal kernel win.

No protocol change is proposed. A more favorable relation would require a new
soundness argument because this sumcheck is what ties `RamValFinal` at the RAM
read/write address point to public I/O.

## Exact relation and transcript boundary

For address `k`, the modular relation is

```text
s(k) = eq(output_address, k)
     * io_mask(k)
     * (val_final(k) - val_io(k)).
```

It has degree three, a constant-zero input claim, 13 low-to-high address
rounds, and one output opening `RamValFinal`. The verifier also derives
`EqAddress`, `IoMask`, and `ValIo` at the output point. Stage 4 requires this
opening point to equal the address prefix of the stage-2 RAM read/write point.

The stage-2 batch draws `output_address` before its round loop, using one raw
`challenge()` per address variable after the two batch gammas. Those 13 values
are not the 13 sumcheck challenges. For each round, the host batch absorbs the
combined polynomial and draws the next shared challenge. The device never
absorbs, hashes, or draws a challenge.

At the production layout,

```text
K                    = 8192 = 2^13
[mask_start, end)    = [1024, 4096)
deferred low rounds  = 10
low block length     = 1024
high-tail elements   = 8
high-tail rounds     = 3.
```

The first ten messages are exactly zero. Every 1024-address block is wholly
inside or outside the mask; outside the mask the mask factor is zero, while
inside it `val_final = val_io` at every Boolean address. This shortcut is
admitted only when the producer certifies that the public-I/O cells came from
the checked `JoltDevice`. Failing certification selects the exact optimized
CPU kernel without scanning the table a second time.

After challenges `r_0, ..., r_9`, the only large operation is

```text
V[h] = sum(i = 0..1023) val_final[1024*h + i] * w[i],
w[i] = product(j = 0..9) (bit_j(i) ? r_j : 1-r_j),
h    = 0..7.
```

The high mask is `[0, 1, 1, 1, 0, 0, 0, 0]`. Certification implies that the
folded `ValIo` endpoints are `[0, V[1], V[2], V[3], 0, 0, 0, 0]`; no dense mask
or public-I/O table is needed. The host combines those endpoints with the
remaining equality factor and proves the last three messages exactly.

## Frozen screening denominator

The current optimized deferred-prefix Criterion observation is frozen in
`autoresearch/evidence/ram_output_check_cpu_deferred_log13_observed_821665b4b.json`
(SHA-256
`ebf03a8f4ea5acadca7f3ae8e32e8469d9de14f4f34127ec913b1dfb6dc5bb3f`).
Its 95% interval is `[274,207.645, 276,099.966, 278,486.098] ns`; the rounded
point estimate used for screening is `276,100 ns`.

```text
5x screening cap = 55,220 ns
8x screening cap = 34,512 ns (rounded down)
```

This is a complete deferred CPU helper at `K = 8192`, not a new alternating
production comparison. The older five-pair PIOP artifact predates the CPU
deferral and must not be used as the successor denominator. Promotion requires
five alternating current-revision pairs over the real member seam. Until then,
all ratios in this directory are screening ratios.

The predecessor Metal artifact is
`autoresearch/evidence/ram_output_check_log13_rejected_5da0a82d6.json`
(SHA-256
`f5747dc502706d0b0a89ef5989507bbe5b6733154dfbe385f5ca588ad47047ac`).
Its favorable one-shot values were 31.458 us GPU-active and 524.667 us wall;
Criterion medians were 52.321 us active and 830.27 us wall. The same resident
CPU block fold measured 22.750 us one-shot and 30.939 us by Criterion.

Even the favorable 524.667-us Metal wall control is 1.90x the whole current
CPU member and 9.50x its 5x cap. The earlier 155.334-us command-service control
is also 2.81x the cap. This is the analytical reason a standalone command is
not an experiment candidate.

## Producer and ownership contract

The input must be a producer-owned `StorageModeShared` range containing exactly
8192 little-endian `u64` words. The owner records:

- Metal device registry identity;
- allocation identity, byte offset, and byte length;
- address count and word stride;
- certification that public-I/O cells equal the checked public memory;
- a host-readable flag, because the required fallback operates on the same
  allocation.

The successor borrows that range. It performs no `RamValFinal` conversion,
upload, full-table copy, or allocation in a round. The 64-field partial range,
ten-field challenge range, optional 1024-field weight range, eight-field output
range, and status word are allocated before the PIOP or borrowed from a
batch-owned arena. Their byte ranges must not overlap the source or each other.

The current optimized RAM read/write prepare already requests `RamValFinal` to
reconstruct `ValInit`; it is the natural producer boundary. It currently owns a
host field vector, not the required native Metal range, so the handoff is not
implemented. Producer construction is charged once in the backend-witness
preparation diagnostic and never hidden in this member.

## Candidate schedules

### A. Incremental host weights, one partial dispatch, host reduction

After each certified zero-round challenge, the host expands the shared weight
table from `2^j` to `2^(j+1)` entries. Across all ten rounds this is 1023 field
products and 16 KiB of final weight storage. The work is distributed across
the existing ten member calls; it is still included in complete-member time.

At the active boundary, 64 threadgroups of 128 threads each cover eight
128-word chunks in each of eight blocks. Every thread performs one promoted
128-by-64 Solinas product, then four SIMD groups reduce to one partial. The
host reads 64 fields, reduces eight groups of eight, and runs the serial tail.
There is no device reduction dispatch.

This is the selected GPU experiment because it retains 64 threadgroups, removes
one dispatch from the predecessor, and avoids a nine-product weight dependency
chain in every lane.

### B. Device weights

The same 64 threadgroups read ten challenges and construct each coefficient
from its low index. Starting with the first factor requires nine full field
products per address, followed by one 128-by-64 product. It removes the 16-KiB
weight table and all host weight products but adds a dependent multiply chain.
It is an ablation, not the default.

### C. Eight block threadgroups

One 1024-thread group per high block can write the final eight fields in one
dispatch. It exposes the same 256 SIMD groups as schedule A but only eight
threadgroups, which may leave most M4 Max GPU cores idle because a threadgroup
cannot migrate between cores. It is admitted only if compiled limits allow
1024 threads and a counter capture disproves the core-utilization concern.

### D. Resident CPU fallback

The same native range and incremental weights feed a CPU dot product and the
same serial tail. The fair CPU candidate should use a wide unreduced
accumulator and one Solinas reduction per block rather than reducing every
128-by-64 product. It is mandatory as the target-shape fallback and as the
GPU selector control. It is outside this shader packet because root owns the
integration and benchmark boundary.

## Exact work and traffic

For schedules A and B, the partial shader has 64 threadgroups, 8192 threads,
and 256 SIMD groups. Its dynamic threadgroup memory is four field elements,
64 bytes. It writes 64 fields (1024 bytes). The optional device reducer reads
those 64 fields and writes eight fields (128 bytes).

| Quantity | Host weights | Device weights |
| --- | ---: | ---: |
| useful native contributions | 8,192 | 8,192 |
| host full-field products | 1,023 | 0 |
| device full-field products | 0 | 73,728 |
| device 128-by-64 products | 8,192 | 8,192 |
| partial additions before host tail | 8,128 | 8,128 |
| dispatches with host reduction | 1 | 1 |
| dispatches with device reduction | 2 | 2 |
| output readback with host reduction | 1,024 bytes | 1,024 bytes |
| output readback with device reduction | 128 bytes | 128 bytes |

The retained measured roofs are 451.701710520 GB/s copy, 45.709 billion
full-field products/s, and 86.592 billion 128-by-64 products/s. They give:

```text
host-weight device arithmetic floor       = 0.095 us
device-weight device arithmetic floor     = 1.708 us
```

For the two-dispatch device-reduction form, semantic traffic including host
writes and final reads is:

| Traffic | Host weights | Device weights |
| --- | ---: | ---: |
| perfect-cache bytes | 100,608 | 68,160 |
| shader-requested bytes if the reused table misses | 215,296 | 1,378,720 |
| perfect-cache copy floor | 0.223 us | 0.151 us |

For the selected host-reduction form the corresponding totals are 100,352 and
67,904 bytes. The host-weight requested total is 215,040 bytes if every reused
weight load misses. Device challenges are constant-sized and should cache, but
the requested 1.379-MiB figure is retained rather than assuming that result.

The arithmetic and traffic roofs are not the practical bound at this size.
Dispatch service, underfilled cores, the host/device join, and the serial tail
bind. Compiled register allocation, spills, and resident threadgroups are not
available for this unregistered source. The structural source floor is one
field accumulator, one field coefficient, reduction temporaries, and the
half-width helper scratch; promotion requires emitted-code and Instruments
evidence rather than converting that source count into a physical occupancy
claim.

## Pre-registered experiment bars

The first experiment is an auxiliary-only control encoded into an otherwise
identical batch-owned command. It must use the same parent command, commit,
wait, and neighboring dispatches in both arms. Measure the auxiliary dispatch
with GPU counter sample boundaries; whole-command timestamps are insufficient
when neighboring work exists.

The 8x pursuit budget is:

| Component | Maximum |
| --- | ---: |
| ten incremental host weight updates | 6,000 ns |
| one half-width partial dispatch, incremental active | 18,000 ns |
| 64-field host reduction plus three-round tail and outputs | 8,000 ns |
| complete incremental member | 32,000 ns |

`32,000 ns` is 8.63x against the frozen screening point. The hard 8x limit is
34,512 ns. If the candidate misses 8x but remains at or below 55,220 ns, it may
advance only when the counter trace and the retained CPU control show no clear
route to 8x. The user goal is a floor, not a stopping point.

Before applying those bars, measure an empty one-dispatch auxiliary with the
same encoder position and counter boundaries. The partial shader must be no
more than 1.25x that measured service floor plus the larger of its arithmetic
and traffic floors. A bar that the dispatch mechanism itself cannot reach is
revised before implementation, never after seeing a slow result.

Standalone execution has a separate falsification rule: any new commit or wait
at `K = 8192` selects CPU without a target run. The existing wall controls have
already falsified that topology by a wide margin.

## Correctness and parity gates

The Rust oracle in this directory is independent of the shader. Integration
must compare every one of the 13 messages, every absorbed coefficient, every
challenge, the final `RamValFinal` opening, and all three derived values with
the optimized CPU kernel. Required cases include:

- all-zero, all-one, `u64::MAX`, limb-boundary, and deterministic random final
  values;
- zero, one, `p - 1`, and random output-address and round challenges;
- public segments with zero padding and all three production high blocks;
- a mutation inside the mask, which must disable deferral or fail parity;
- mutations outside the mask, which must change later messages and the output
  opening;
- source, scratch, and status range alias attempts;
- wrong allocation/device identity and an uncertified producer;
- host-weight and device-weight agreement before either is timed.

The source ABI validates exact geometry, buffer sizes, alignment, nonoverlap,
threadgroup count, thread width, cleared status, and reserved words on the
host. The shader repeats geometry and thread-width checks and sets an atomic
status bit on failure. A nonzero status aborts before any polynomial is
absorbed.

## Integration blockers and cutover

1. Stage 2 has no batch-owned command aggregator at the first active output
   round. Adding one is a scheduling change across kernel slots and belongs to
   the root integrator.
2. No producer currently publishes the checked native `RamValFinal` range.
3. The shared 128-by-64 primitive has screening throughput but still needs the
   retained compiler spill/residency evidence for downstream promotion.
4. The current CPU evidence is Criterion-only. A fresh current-revision
   alternating production denominator is required.
5. The eight-element tail needs a no-allocation adapter that reuses the exact
   batch polynomial interpolation and derived-table checks.

The runtime selector is therefore:

```text
if !certified_resident_native_source:
    optimized CPU
else if !batch_owned_command_with_incremental_counters:
    resident CPU successor
else if measured_complete_incremental_gpu <= measured_resident_cpu - noise:
    coalesced Metal partials + CPU tail
else:
    resident CPU successor
```

The selector is frozen per machine and geometry from alternating measurements.
At geometries without an aligned certified zero prefix, use the optimized CPU
kernel; the successor does not generalize by silently emitting nonzero rounds
as zero.
