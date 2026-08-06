# Spartan outer remainder on Metal

`OuterRemainder` is the Metal phase after Hamming-weight claim reduction. In the
five-pair `2^26` Fibonacci production run at revision `55c909600`, the
optimized-CPU member took 905.872 ms and the then-CPU member in the Metal process
took 912.167 ms, or 12.39% of the 7.363-s Metal PIOP. That portfolio profile
predates the current Outer integration and cannot rank the current HEAD.

The closed isolated evaluator for the integrated implementation measured 881.996
ms on optimized CPU and 219.487 ms on Metal, or 4.01495x. It is exact, but it is
below the v2 5x floor and has no log-27 transfer result. The next evidence must
come from a fresh v2 run; no schema-1 snapshot is a resumable parent.

The member is a better next target than the slightly larger registers read/write
member because its input rows already exist on the device. Stage 1's Metal uni-skip
uses a 48-byte `InstructionInputRow` and a 112-byte residual row for every cycle.
The remainder follows immediately, has a dense fixed shape, and needs the same
canonical values. Retaining those two allocations changes ownership only; it does
not upload or reformat another row plane.

## Exact boundary

The protocol and transcript stay unchanged. The device computes only deterministic
field arithmetic. The host still constructs every degree-three round polynomial,
absorbs it, samples the challenge, and checks the running claim.

At `T = 2^26`, the relation has `2T = 2^27` `(cycle, stream)` cells and 27 rounds.
The optimized member currently has three material components:

| Component | Median in the Metal production arm |
|---|---:|
| Prepare and first message | 534.417 ms |
| Round sequence | 176.538 ms |
| 35 output openings | 200.190 ms |

The fixed evaluator's 881.996-ms optimized-CPU denominator gives a 176.399-ms
budget for 5x. Keeping the roughly 200-ms opening walk on the CPU would make that
floor unreachable. The Metal design therefore owns preparation, the large round
prefix, and all 35 output openings. Five times is the minimum; the calibrated
80%-efficiency bar below is tighter at 170.5 ms.

The first target-scale resident candidate produced the same proof and verified in
both arms. Its single timed pair measured 889.474 ms on optimized CPU and 217.138 ms
on Metal, or 4.096x. Full initialization of its 4,300,079,856-byte scratch set took
77.959 ms before the PIOP; charging that one phase gives a 295.097-ms, 3.014x cold
diagnostic. This pair validates the mechanism but is not promotion evidence; the
five-pair fixed evaluator remains authoritative.

The CPU-first and Metal-first strata measured 4.01495x and 4.10172x. Every
exactness and lifecycle guard passed, but neither stratum clears the v2 floor. The
76.601-ms median scratch preparation remains outside the primary member; charging
it produces a 298.111-ms median cold-inclusive Metal diagnostic and a 2.94269x
median paired speedup.

## Resident lifetime

Backend witness preparation already creates the split stage-1 row plane. The
uni-skip invocation currently takes and drops the residual owner after its command
completes. The remainder design instead applies this ownership sequence:

```text
backend witness prepare
    -> allocate and fully pre-touch reusable remainder storage
    -> stage-1 uni-skip use
    -> retain the same compact + residual handles
    -> OuterRemainder materialization and rounds
    -> OuterRemainder opening scan
    -> release residual rows
    -> keep the shared compact handle for InstructionInput
```

Allocation identity, row count, and Metal device registry must match at every
handoff. All nine scratch identities established during pre-touch must also match
the active sequence. There is no row upload or device-buffer allocation inside the
timed remainder member.
The CPU `SpartanOuterCarry` remains available until the adapter has made its
pre-submit admission decision, so an ineligible trace or capacity rejection can
select the optimized kernel. Capacity, initialization-command, and initialization
timestamp failures are recoverable before protocol state changes; invalid geometry,
configuration, pipeline, and state errors remain fatal. The fallback reason is
recorded in the trace. Any error after command submission aborts the proof; the
adapter never retries from mutated state.

`with_metal_compute` installs the uni-skip producer and remainder consumer as one
residency family. Replacing only one slot is legal at the type level but may retain
an unused residual allocation until the proof session drops.

## Device schedule

### Materialize and emit the first message

One thread evaluates one cycle at a time. It folds the ten first-stream and nine
second-stream rows in uniform loops, stores both `Bz` values, and accumulates its two
message contributions. The SIMD group then reduces across cycles. This replaces
the initial row-per-lane mapping, whose divergent row switch serialized nearly all
19 paths and left only lane zero doing the final products. The remap reduced
target-scale first-message GPU-active time from roughly 791 ms to 84.6 ms.

The dispatch stores only `(Bz(0), Bz(1))` for each cycle. `Az` depends only on the
compact flag word, so keeping both stream values would spend another 2 GiB on state
that is cheaper to reconstruct once. Both `Az` values remain live long enough to
reduce the first round's canonical `q(0)` and `q(infinity)` endpoints. The host turns
those endpoints and the running claim into the same Gruen polynomial used by
`OptimizedOuterRemainder`, then performs Fiat--Shamir.

### Fuse binding with the next message

The first transition is specialized for the stream challenge. It reads the stored
`Bz` pair and only the compact flag word, reconstructs the challenge-blended `Az`,
binds `Bz`, writes one interleaved `(Az, Bz)` cell per cycle to the other 2-GiB
buffer, and computes the next message before the values leave registers.

Later transitions read adjacent interleaved pairs, bind both fields with the host
challenge, write the half-sized state to the other buffer, and compute the next
message endpoints from the bound pairs before they leave registers. This avoids a
separate message scan. The two fixed 2-GiB allocations ping-pong; obsolete initial
`Bz` storage becomes the next output buffer.

The checked-in v2 baseline uses Metal while the current table is larger than
`2^16` cells. The specialized stream transition reduces the initial `2^27`
relation cells to `2^26` cycle cells; ten dense transitions reach the cutoff. The
shared buffer is then synchronized once and optimized host arithmetic finishes the
tail.
The cutoff is a measured parameter, not a protocol constant; neighboring powers of
two must be tested. Host split-equality state advances with every challenge and is
the source for both the device prefix and CPU tail.

### Evaluate the 35 openings

After the final cycle point is known, one more resident-row scan computes the 35
canonical R1CS-input evaluations. A threadgroup tile loads 64 packed rows and their
`E_in` weights once into roughly 11 KiB of shared memory. Each SIMD group owns a
uniform subset of columns while its lanes walk tile rows, avoiding the baseline's
35-way divergent column switch. Eighteen boolean columns conditionally add the
weight. Thirteen `u64` and four signed or unsigned `u128` columns use the same
generic wide product and block-local reduction. A fixed-256 experiment accumulated
the narrow products exactly in seven limbs and reduced once per dot product. That
removed nearly all per-product canonical reductions without adding a row scan, but
opening GPU-active time improved by only 2.217 ms. The next opening experiment must
target row extraction, threadgroup traffic, or work ownership rather than scalar
field arithmetic. Each block result is scaled by one `E_out` value.

At the baseline cap, the first dispatch writes `35 * 8192` partial field sums, or
4.375 MiB. A second dispatch reduces by column, and the host reads exactly 35
canonical fields. Output IDs, the common reversed cycle point, derived-weight
validation, final relation checks, and transcript absorption use the existing host
path.

## Traffic, arithmetic, and occupancy ceilings

The traffic model counts shader-visible values at the checked-in `2^16` cutoff.
It is not a hardware-counter measurement. In particular, the first stream bind
requests only the 8-byte flag word from each 48-byte compact row, but its stride
can charge the entire cache-line span. The current-layout range is therefore 4.5
GiB of logical flag/state payload to 7 GiB of conservative row-span traffic.

| Item at `T = 2^26` | Bytes | GiB |
|---|---:|---:|
| Compact resident rows, 48 B/cycle | 3,221,225,472 | 3.000 |
| Residual resident rows, 112 B/cycle | 7,516,192,768 | 7.000 |
| Initial `(Bz(0), Bz(1))` state | 2,147,483,648 | 2.000 |
| Second ping-pong allocation | 2,147,483,648 | 2.000 |
| Materialization traffic | 12,884,901,888 | 12.000 |
| First stream bind, conservative 112 B/cycle | 7,516,192,768 | 7.000 |
| Dense prefix to `2^16` | 6,436,159,488 | 5.994 |
| Opening scan and minimum partial traffic | about 10,746,593,280 | about 10.009 |
| Current-layout member total | about 37,583,847,424 | about 35.0 |

A packed or coalesced flag plane reduces first-bind payload to 4.5 GiB and makes
the transition traffic floor 24.95 ms at the retained 420.68-GiB/s copy control.
The conservative current layout instead puts that phase near a 30.9-ms traffic
floor. Hardware counters are required before calling either figure measured DRAM
traffic.

The useful optimized product counts exclude removable small-scalar `A` products:

| Phase | `2^26` products | `2^27` products |
|---|---:|---:|
| Materialize | 1,543,503,872 | 3,087,007,744 |
| Transitions | 536,608,768 | 1,073,479,680 |
| Openings | 1,141,137,408 | 2,281,988,096 |

The checked-in shader still executes additional data-dependent full products in
the `A` fold. These are optimized-useful counts, not current instruction counts.

At the measured 24.08-Gproduct/s control, combined with attainable coalesced flag
traffic, the provisional calibrated floors are:

| Scale | Materialize | Prefix | Opening | Total floor | 80%-efficient latency |
|---|---:|---:|---:|---:|---:|
| `2^26` | 64.10 ms | 24.95 ms | 47.39 ms | 136.43 ms | 170.54 ms |
| `2^27` | 128.20 ms | 49.91 ms | 94.77 ms | 272.87 ms | 341.09 ms |

The v2 gate is pre-registered as exact, at least 5x, and at most 170.5 ms at
`2^26`. At `2^27` it is exact and at most `min(341.1 ms, fresh CPU median / 5)`.
These are calibrated ceilings, not hardware-theoretical limits; fresh phase
controls or ISA/Instruments evidence may revise them before candidate testing.

Against those floors, the historical implementation reached about 74.1% in
materialization, 61.2% in the prefix, 75.1% in openings, 71.7% for GPU-active
work, and 62.2% for the complete member. Occupancy is not yet established. The
opening dispatch requests 15,184 bytes of threadgroup memory, including 3,920
bytes of unused shard scratch. If the observed 32-KiB per-threadgroup limit is
also the per-core shared pool, no more than two groups can reside; that conclusion
remains conditional until Instruments or ISA evidence reports active SIMD groups,
register pressure, and spills.

The sequence owns 4 GiB of ping-pong state beyond the existing 10-GiB row plane;
the opening partials add 4.375 MiB. Its largest allocation is 2 GiB, below this
machine's measured 80.64-GiB per-buffer limit. Admission uses the live whole-proof
allocation count because other Metal stages may already be resident.

## Fixed evaluator

The authoritative isolated evaluator belongs in `jolt-prover`, because it needs the
real generated stage-1 driver and a production Fibonacci witness. A `jolt-kernels`
fixture would either introduce a dependency cycle or silently substitute a
different member boundary. The harness constructs and pads one real `2^26` trace
once. Each proof replay then repeats production backend preparation outside the
member; the Metal arm creates the split row plane and fully pre-touches reusable
scratch there. Both arms use the same immutable fixture and produce an exact full
proof.

The timed member starts before remainder preparation and ends after all output
claims and recorder work. It includes:

- preparation and the first message;
- 27 round calls and all 27 host Fiat--Shamir squeezes;
- every Metal command, completion wait, handoff, CPU tail, and readback;
- the final bind, 35 openings, derived-table validation, final-relation check, and
  transcript absorption.

It excludes trace construction, shader compilation, backend witness preparation,
and uni-skip from both member arms. The evaluator separately reports the Metal
scratch-preparation wall and a conservative `member + scratch preparation` cold
diagnostic; neither replaces the resident PIOP metric.

One excluded warmup precedes five alternating CPU/Metal pairs with Rayon fixed at
16 threads. A candidate can become a search parent only by improving beyond the
fixed noise threshold. Production promotion additionally requires at least 5x in
both order strata, no more than 170.5 ms at `2^26`, exact component reconciliation,
and equality of every round polynomial, host challenge, running claim, final claim,
all 35 openings, derived value, and transcript digest. Resource guards cover row
and scratch identities, full initialization outside the member, zero member
allocations/uploads, command and dispatch counts, per-round table lengths, one
prefix-to-tail transition, and one 35-field readback. Evaluator schema
`outer_remainder_v3` additionally verifies active post-attach scratch identities
and logical ownership released after the opening.

The production holdout is five fresh alternating full-PIOP pairs at Fibonacci
`2^26`, with both proofs verified and the same lifecycle topology. It cannot be
used for tuning. Transfer repeats five pairs at `2^27`; its local latency must be
at most `min(341.1 ms, fresh CPU median / 5)`. A local winner is only an accepted
search parent until fresh revalidation, holdout, and transfer all pass.

## Fresh v2 experiment order

The shader cannot change until the fresh v2 baseline confirms the historical
phase attribution and either preserves or revises the calibrated floor. The first
candidate retains both `A` and `B` stream state: materialization writes `A` into
the otherwise unused second ping-pong allocation, and the stream bind reads and
overwrites it in place. This adds no allocation and removes the second full `A`
fold. Its failure mode is turning compute-bound materialization into a bandwidth
bottleneck without enough first-bind savings.

If retained `A` loses, materialization emits a packed flag plane. That changes the
later 48-byte-stride compact-row walk into an 8-byte coalesced read and removes
roughly 2 GiB of conservative traffic at `2^26`. Next, opening ownership removes
the unused 3,920-byte shard scratch and tests direct or cached weight access, with
47.39 ms as the phase target. Only then should materialization transpose or unroll
the 19 `B` folds to reuse extraction and common subexpressions. Cutoff and launch
geometry are last because late underfilled rounds offer less headroom.

Before promotion, Instruments or ISA evidence must report threadgroup memory,
register pressure, spills, active SIMD groups, achieved occupancy, and dispatch
utilization. Row-release deferral is admissible only when an adjacent production
consumer proves direct ownership reuse; moving work outside this member is not an
optimization.

The historical resident remap is below the v2 5x floor. Its dominant GPU-active
phases are the 86.5-ms first message, 25.5-ms first bind, 15.2-ms dense prefix, and
63.1-ms opening scan. Two analytical candidates were rejected on exact
`2^26` pairs. Rewriting the flag-only `Az` fold as affine additions produced a
218.015-ms member versus the retained 217.138-ms parent; its 85.367-ms first message
also missed the parent's 84.641 ms. Reducing the opening accumulator array from nine
to five slots increased opening GPU-active time from 62.075 ms to 85.088 ms and the
member to 248.279 ms. Both changes were fully reverted.

That mixed-width candidate was also exact but did not improve the target kernel:
opening GPU-active time was 62.187 ms and member wall was 220.625 ms, versus 62.075
and 217.138 ms for the retained parent. A direct 1,048,576-element Criterion control
showed why: the custom 4-by-2-limb product took 221.01 us while the generic 4-by-4
product took 183.02 us on this compiler and GPU. Removing source-level multiplies
introduced a longer dependency/code-generation path. The helper, probe, tests, and
integration were fully reverted. The next opening design must reduce the number of
canonical reductions or change work ownership; another eager mixed-width product
is pruned.

A seven-limb deferred-dot candidate then removed per-product reduction for all 13
true `u64` columns while preserving the one-scan schedule. It passed the real-device
field oracle and every exact five-pair proof and lifecycle guard. Opening GPU-active
median improved from 63.104 to 60.887 ms, but complete output time moved from 82.165
to 82.452 ms and member median from 219.487 to 219.774 ms. Its primary paired ratio
was 4.02564x versus the 4.01495x parent, a 0.27% change below the fixed 3% promotion
threshold, so the controller restored the parent. The result rejects canonical
reduction as the dominant opening bottleneck. Further opening work starts with a
reusable resident-dispatch benchmark and targets threadgroup access or work
ownership. The controller should continue toward 5x when measured headroom remains
plausible.
