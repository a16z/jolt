# RAM output-check deferred-prefix design

## Decision

Do not port the existing 13-round dense walk to Metal. At the production
`log_T = 26`, `log_K = 13` shape, the output check is an address-only member
with just 8192 coefficients. Thirteen host-dependent submissions would spend
the entire CPU denominator on command latency.

The only Metal candidate evaluated here is a one-command hybrid:

1. Emit the first ten local round polynomials as exact zeros while the stage-2
   batch owns Fiat--Shamir on the host.
2. After challenge 9, fold a producer-owned, Metal-resident native `u64`
   `RamValFinal` table through all ten challenges in one command, from 8192
   words to eight fields.
3. Read back 128 bytes and prove rounds 10 through 12, the terminal bind, and
   the output claim serially on the CPU.

The low-level source, runtime, parity test, and Criterion probe are registered.
The measured selector rejects Metal at the target shape: the resident CPU block
fold is substantially faster even before command latency is charged. The
production backend should implement the same ten-round deferral and select the
specialized CPU fold. The high-level proof adapter and shared producer are not
yet wired.

## Frozen denominator

The comparison artifact is
`benchmark-runs/metal-piop-eval/20260806-133709-697013`. The five complete
optimized-CPU member samples, in pair order, are:

```text
1.683916 ms, 1.937875 ms, 1.734583 ms, 1.592332 ms, 1.970918 ms
```

Pair 3 is the median, exactly `1,734,583 ns`. The hard 5x cap is therefore
`346,916 ns` after rounding down to an integral nanosecond. Its trace gives:

| Component | Time |
| --- | ---: |
| prepare | 280.042 us |
| rounds 0 through 9 | 1,322.625 us |
| rounds 10 through 12 | 130.958 us |
| terminal bind | 0.250 us |
| output claim | 0.708 us |
| complete member | 1,734.583 us |

`RamValFinal` materialization is 246.208 us inside prepare. Host Fiat--Shamir
spans are outside the member spans and are not part of this denominator.

The producer-neutral relation denominator, excluding all prepare time, is
`1,454.541 us`; its separate 5x cap is `290.908 us`. Promotion must report both
the complete-member ratio and this relation-only ratio.

## Protocol boundary

For address `k`, the relation is

```text
s(k) = eq(output_address, k)
     * io_mask(k)
     * (val_final(k) - val_io(k)).
```

The round polynomial has degree three. The member has no input opening and its
input claim is the constant zero. It produces one `RamValFinal` opening. The
verifier derives and checks `EqAddress`, `IoMask`, and `ValIo`; none is an
additional opening.

The production read-write split has phase 1 equal to all 26 cycle variables
and phase 2 equal to all 13 address variables. The stage-2 batch leader has 39
rounds, while this member is tail-aligned at global round 26 and owns 13 local
low-to-high address rounds. There is no internal cycle gap.

The output-address reference point is drawn once before the batch, one raw
`challenge()` per address variable, after the two batch gammas. It is distinct
from the 13 round challenges. For every active round, the member returns its
unbatched cubic; the batch driver combines all active members, absorbs the
combined polynomial, and draws the shared challenge. No shader hashes,
absorbs, or draws a challenge.

The final opening point is the 13 address challenges extracted from the stage-2
sumcheck point and reversed into the verifier's normal address order. Stage 4
requires this point to equal the address prefix of the RAM read-write opening
point before it constructs `RamValCheck`. No transcript, relation, round order,
claim, or opening-point change is proposed.

## Why ten messages are free

The default memory policy reserves 4096 bytes each for trusted advice,
untrusted advice, input, and output, plus panic and termination words. The
whole region pads to 4096 words. In the remapped address domain, the output
mask is exactly

```text
[io_mask_start, io_mask_end) = [1024, 4096).
```

Both boundaries are divisible by `2^10`. During low-to-high rounds 0 through
9, every paired block is wholly inside or wholly outside the mask. Outside the
mask the product is zero. Inside it, `val_final = val_io` at every Boolean
address, including zero padding, so the difference is zero. Hence all four
evaluations of each of those round polynomials are zero, not merely their
Boolean-point sum.

This uses the production witness invariant that the public-I/O cells in the
final-state oracle are synthesized from the same checked `JoltDevice`. A custom
witness plane that cannot certify that invariant must use the exact dense CPU
path; this member may not buy the shortcut with a new full-table validation
scan. The verifier still checks the final random-point value against public
I/O, so the optimization changes neither the relation nor the proof statement.

The ten binds cannot be discarded, but binds of distinct multilinear
variables commute. With challenges `r_0, ..., r_9`, build the 1024 low-bit
weights

```text
w[i] = product over j=0..9 of (bit_j(i) ? r_j : 1-r_j).
```

Then each surviving value is one block dot product:

```text
folded[h] = sum over i=0..1023 val_final[1024*h + i] * w[i],
            h in 0..7.
```

The host obtains the eight mask values directly as `[0,1,1,1,0,0,0,0]` for
the target layout. It folds `ValIo` by visiting only the public sparse segments
and multiplying each word by its low-bit weight. No dense mask or public-I/O
table is allocated. The host still advances the tiny Gruen equality state for
each zero-round challenge. The serial tail evaluates the cubic directly at
`0,1,2,3`, so it inherits the current implementation's behavior even when
either Gruen endpoint is zero.

The independent Rust oracle in this directory materializes all four dense
factors, checks the zero messages, binds directly, and exposes the block-dot
fold. It does not use the proposed shader reduction.

## One-command Metal work

The default fold uses 128 threads per threadgroup and eight chunks per
1024-element block. Its first dispatch has 64 threadgroups and 256 SIMD groups;
each lane performs one full field product. A second dispatch reduces the eight
partials for each of the eight output blocks. Both dispatches are encoded in
one command buffer, followed by one completion wait and one 128-byte readback.

| Quantity | Target value |
| --- | ---: |
| device full-field products | 8,192 |
| host products to construct weights | 1,023 |
| command buffers / waits | 1 / 1 |
| dispatches | 2 |
| first-pass threadgroups / SIMD groups | 64 / 256 |
| cache-optimistic compulsory traffic | 84,096 bytes |
| traffic if low weights miss for every high block | 198,784 bytes |

At the retained 18.10-Gproduct/s six-accumulator control, the device arithmetic
floor is `0.453 us`. At 420.68 GiB/s, the two traffic floors are `0.186 us` and
`0.440 us`. Neither explains wall time; fixed command service dominates. The
shader embeds each native `u64` into the field before multiplication; a future
128-by-64 multiply is an implementation ablation, not an assumption in these
numbers.

The first pass exposes only 64 threadgroups. Dynamic threadgroup memory is one
field per SIMD group, 64 bytes at width 128, and the live state is one field
accumulator plus multiply temporaries. Threadgroup memory and buffer capacity
therefore do not gate occupancy. The launch is too small to claim theoretical
device saturation: promotion requires compiled register/spill data and an
Instruments capture, but tuning occupancy cannot remove the command-latency
floor.

## Storage and ownership

| Allocation | Bytes |
| --- | ---: |
| borrowed native `RamValFinal` words | 65,536 |
| low-bit weights | 16,384 |
| 64 partials | 1,024 |
| eight folded outputs | 128 |
| private scratch | 17,536 |
| total including borrowed input | 83,072 |
| maximum single buffer | 65,536 |

The native final RAM state is already constructed by the earlier stage-2 RAM
read-write member before that member embeds values into the field to reconstruct
`ValInit`. The shared owner should therefore be created there, carry the device
registry, allocation identity, address count, and certified public-I/O
construction invariant, and remain live until this output-check fold completes.
This member may attach it in constant time. It must not create a private field
vector, upload 64 KiB, or validate the table again inside its measured span.
Any producer-side mapped-buffer fill is reported once in the PIOP total. Stage
4 consumes only the scalar opening produced here, not the full table, so the
resident allocation can be released after this member.

The current output-check prepare's 246.208-us duplicate materialization is a
reusable producer saving, not shader speed. Report it separately. Conversely,
the 8192-product fold, command service, readback, zero-message handling, sparse
public fold, and CPU tail are all member-local. The stage-2 RAM RA address plane
is unrelated: this member consumes the `K = 8192` final-value table, not the
`T = 2^26` address plane.

If the resident owner does not exist, the Metal path is rejected before
dispatch. A diagnostic may charge conversion and upload, but it cannot be
reported as the resident member.

## Latency gate and CPU switchover

The retained native-primer samples used 64 source elements and two dispatches,
which is the nearest favorable command control. Their median submit and active
times are `47.834 us` and `78.750 us`; median post-completion join is
`28.750 us`. The resulting `155.334 us` command-service control leaves only
`191.582 us` under the complete-member cap.

Combining that control with the current, unspecialized three-round CPU tail
gives `286.292 us`, leaving `60.624 us` for resident attachment, ten zero
messages, weight construction, sparse public folding, and output assembly.
That is tight but credible because the tail state has only eight fields and
must be rewritten as a serial path rather than invoking the current parallel
dense machinery.

Before integration, register these complete warm-path bars:

| Component | Maximum |
| --- | ---: |
| resident attach, zero rounds, weights, and public fold | 45 us |
| one Metal fold command including wait and readback | 165 us |
| serial three-round tail, terminal bind, output | 125 us |
| complete member | 335 us |

The component bars sum to 335 us, or 5.18x against the frozen member. The hard
acceptance limit remains 346.916 us. The producer-neutral view must additionally
be at most 290.908 us after excluding equivalent prepare work from both sides.
If the measured implementation supports materially more than 5x, continue
optimizing rather than stopping at the floor.

Benchmark the resident CPU block fold and the one-command Metal fold from the
same input allocation. Metal is selected only when:

```text
metal_fold_wall + max(5% of cpu_fold_wall, 10 us) < cpu_fold_wall,
complete_member <= 346.916 us,
producer_neutral_relation <= 290.908 us.
```

The measured command crossover is approximately 8192 / 155.334 us =
52.73 Mproduct/s before common tail work. A dedicated CPU dot product that
exceeds that rate should remain on CPU at `log_K = 13`. This is why the design
is CPU-first despite admitting a Metal probe.

## Observed target-shape crossover

The integrated probe uses the same 65,536-byte shared allocation for both
paths, precomputes the same 1,024 low-bit weights, and excludes the diagnostic
host upload and one-time pipeline setup from both timed folds. Exact Metal
output matched an independent dense ten-bind oracle with zero, one, random,
zero-word, and `u64::MAX` inputs.

At 128 threads, a focused Criterion run measured:

| Path | Median wall or active time |
| --- | ---: |
| Resident specialized CPU fold | 30.939 us |
| Resident Metal fold, wall | 830.27 us |
| Resident Metal fold, GPU active | 52.321 us |
| Complete optimized CPU deferred member | 276.10 us |

The distributions were noisy, but the decision is not marginal. Metal wall is
26.84x slower than the resident CPU fold and 2.39x above the entire
346.916-us complete-member cap. Even GPU-active time alone is slower than the
CPU competitor. A warm one-shot control gave 22.75 us CPU, 524.667 us Metal
wall, and 31.458 us GPU active, leading to the same decision.

No threadgroup-width search can satisfy the registered selector: the device
arithmetic itself does not beat the CPU fold, and changing occupancy cannot
remove the command wait. The Metal path is retained only as a falsified
diagnostic and parity guard.

The optimized CPU kernel now implements the certified deferral and switches
the eight-element tail to a serial message evaluator. With the frozen
`RAYON_NUM_THREADS=16` policy, its focused Criterion interval is
`[274.21, 276.10, 278.49] us`, or 6.28x against the old 1.734583-ms complete
member. It clears the 346.916-us cap by 70.816 us. A production-domain
`K=8192` lockstep test matches the reference kernel at every round and output
claim; the existing smaller fixtures cover the all-deferred case.

This is a common-path CPU improvement, not a Metal-over-current-CPU speedup:
the optimized CPU backend and the Metal backend's retained CPU slot both use
it. The slot therefore remains CPU-owned and contributes roughly equal time
to both PIOP arms until a future fused producer can remove that common cost.

## Rejected schedules and reopen conditions

A literal all-device port performs 33,786 message products plus 24,573 bind
products, or 58,359 useful products total. It moves at least 1,572,576 bytes of
three-table state before the terminal bind. Those small roofs are irrelevant
because it needs 13 host-dependent command buffers.
Thirteen median submit-plus-active controls total 1.645592 ms, already 4.74x
the entire 5x cap. It is permanently rejected under the current host-FS batch
contract.

Keeping the last three rounds on Metal reduces the schedule to three commands,
but three median submit-plus-active controls total 379.752 us, also above the
cap before preparation or readback. Only the single-command fold plus CPU tail
is admissible.

Reopen a multi-command design only if a batch-level fused submission or
host/device synchronization primitive demonstrates a measured marginal command
cost satisfying

```text
commands * command_service
  + resident_prepare
  + readback
  + host_tail
  <= 346.916 us.
```

Evaluate such work as a fused stage-2 boundary; do not attribute another
member's already-paid submission as standalone `RamOutputCheck` speedup.

## Integration still required

The source registry, both diagnostic pipelines, typed resident owner, exact
prefix parity test, CPU-versus-Metal probe, and optimized deferred
`SumcheckKernel` are present. The kernel certifies the public-I/O equality
region before deferring; a failed certification falls back to the exact dense
round path. The Metal hybrid intentionally retains this optimized CPU slot.
The remaining production gate is a paired proof-stage profile that confirms
the focused complete-member result and derived-table checks under the retained
Fibonacci fixture. Reusing the earlier RAM producer's native final-state
allocation remains an optional preparation saving.
