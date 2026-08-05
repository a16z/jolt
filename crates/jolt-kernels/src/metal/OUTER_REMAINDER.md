# Spartan outer remainder on Metal

`OuterRemainder` is the next Metal phase after Hamming-weight claim reduction. In
the five-pair `2^26` Fibonacci production run at revision `55c909600`, the
optimized-CPU member took 905.872 ms and the unchanged member in the Metal process
took 912.167 ms. The latter is 12.39% of the 7.363-s Metal PIOP. Reaching the 4x
portfolio target still requires removing roughly 2.57 s, so this port is necessary
but cannot finish the portfolio by itself.

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

The complete optimized-CPU denominator gives a 226.5-ms budget for 4x and a
181.2-ms target for 5x. Keeping the 200.2-ms opening walk on the CPU would cap an
otherwise free implementation at about 4.5x. The first Metal design therefore owns
preparation, the large round prefix, and all 35 output openings. Four times is the
falsification bar, not the stopping point.

The first target-scale resident candidate produced the same proof and verified in
both arms. Its single timed pair measured 889.474 ms on optimized CPU and 217.138 ms
on Metal, or 4.096x. Full initialization of its 4,300,079,856-byte scratch set took
77.959 ms before the PIOP; charging that one phase gives a 295.097-ms, 3.014x cold
diagnostic. This pair validates the mechanism but is not promotion evidence; the
five-pair fixed evaluator remains authoritative.

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

The baseline uses Metal while the current table is larger than `2^18` cells. Nine
fused transitions take the initial `2^27` cells to that cutoff. The shared buffer is
then synchronized once and the optimized host arithmetic finishes the small tail.
The cutoff is a measured parameter, not a protocol constant; neighboring powers of
two must be tested. Host split-equality state advances with every challenge and is
the source for both the device prefix and CPU tail.

### Evaluate the 35 openings

After the final cycle point is known, one more resident-row scan computes the 35
canonical R1CS-input evaluations. A threadgroup tile loads 64 packed rows and their
`E_in` weights once into roughly 11 KiB of shared memory. Each SIMD group owns a
uniform subset of columns while its lanes walk tile rows, avoiding the baseline's
35-way divergent column switch. Eighteen boolean columns conditionally add the
weight. Thirteen `u64` and four signed or unsigned `u128` columns currently use the
same generic wide product and block-local reduction. Specializing the 13 narrow
columns without another row scan is the next arithmetic candidate. Each block
result is scaled by one `E_out` value.

At the baseline cap, the first dispatch writes `35 * 8192` partial field sums, or
4.375 MiB. A second dispatch reduces by column, and the host reads exactly 35
canonical fields. Output IDs, the common reversed cycle point, derived-weight
validation, final relation checks, and transcript absorption use the existing host
path.

## Traffic and capacity model

The model counts shader-visible values. It is an optimistic roof, not a hardware
counter claim.

| Item at `T = 2^26` | Bytes | GiB |
|---|---:|---:|
| Compact resident rows, 48 B/cycle | 3,221,225,472 | 3.000 |
| Residual resident rows, 112 B/cycle | 7,516,192,768 | 7.000 |
| Initial `(Bz(0), Bz(1))` state | 2,147,483,648 | 2.000 |
| Second ping-pong allocation | 2,147,483,648 | 2.000 |
| Materialization traffic | 12,884,901,888 | 12.000 |
| Stream bind and fused cycle prefix to `2^18` | 11,249,123,328 | 10.477 |
| Opening scan plus partial write/read | 10,746,593,280 | 10.009 |
| Total modeled member traffic | 34,880,618,496 | 32.485 |

At the retained 420.68-GiB/s copy control, the three phases have traffic floors of
28.52, 24.91, and 23.79 ms, or 77.22 ms total. Equality tables and reduction
partials are small relative to the row and state planes but remain charged by the
measured implementation.

The sequence owns 4 GiB of ping-pong state beyond the existing 10-GiB row plane;
the opening partials add 4.375 MiB. Its largest allocation is 2 GiB, below this
machine's measured 80.64-GiB per-buffer limit.
Admission uses the live whole-proof allocation count, not these local sizes alone,
because InstructionInput and Instruction-RA storage can already be resident.
Because remainder storage now overlaps the temporary uni-skip invocation, aggregate
admission sums those footprints rather than taking their maximum.

The arithmetic roof is less certain than the traffic floor. Preparation performs
about `23T = 1.544` billion signed wide or full field products plus cheap guard
weight additions. The specialized stream bind and fused GPU prefix perform about
469 million field products. The opening scan performs `17T = 1.141` billion
field-by-word products; its 18 boolean columns need no multiplication. At the
measured 16.42-Gproduct/s compute control these phases sum to roughly 192 ms before
command and reduction overhead, while the faster 24.08-Gproduct/s fused-transition
rate gives a 131-ms arithmetic projection. Specialized scalar products and
deferred reduction are therefore required. Treating all three phases as generic
pointwise multiplication is a conservative model, not the implementation plan.

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
16 threads. Promotion requires at least 4x in both order strata, a gain above the
fixed noise threshold, exact component reconciliation, and equality of every round
polynomial, host challenge, running claim, final claim, all 35 openings, derived
value, and transcript digest. Resource guards cover row and scratch identities,
full initialization outside the member, zero member allocations/uploads, command
and dispatch counts, per-round table lengths, one prefix-to-tail transition, and
one 35-field readback. Evaluator schema `outer_remainder_v3` additionally verifies
the active post-attach scratch identities and logical ownership released after the
opening. It does not claim that Metal immediately returns cached allocations to the
operating system.

The production holdout remains five fresh alternating full-PIOP pairs at Fibonacci
`2^26`, with both proofs verified and the same lifecycle topology. A local winner is
only an accepted search parent until that gate passes.

## Initial experiment

The first implementation holds the algorithm fixed and varies only:

- materialization, transition, and opening threadgroup widths;
- the capped transition/partial grid size;
- the power-of-two CPU-tail cutoff.

The baseline target is at most 200 ms (about 4.5x against the current optimized
control), with 226 ms as the 4x floor and 181 ms as the 5x stretch target. If a
correct run misses 4x because the opening scan or materialization cannot reach its
modeled roof, the phase records that result and revisits the dataflow rather than
tuning protocol-visible behavior.

The resident remap is now below the 4x floor but above the original 200-ms working
target. Its dominant GPU-active phases are the 84.6-ms first message, 25.0-ms first
bind, and 62.1-ms opening scan. Two analytical candidates were rejected on exact
`2^26` pairs. Rewriting the flag-only `Az` fold as affine additions produced a
218.015-ms member versus the retained 217.138-ms parent; its 85.367-ms first message
also missed the parent's 84.641 ms. Reducing the opening accumulator array from nine
to five slots increased opening GPU-active time from 62.075 ms to 85.088 ms and the
member to 248.279 ms. Both changes were fully reverted. The next search halves the
schoolbook limb products for the 13 true `u64` opening columns while preserving the
single resident-row scan; the controller should continue toward 5x when measured
headroom remains plausible.
