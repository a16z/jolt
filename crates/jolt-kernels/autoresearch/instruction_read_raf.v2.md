# InstructionReadRaf Metal v2 design packet

Status: design-only. This file does not register a backend, assemble a shader,
or claim a measured successor result. It fixes the protocol boundary, records
the latest exact diagnostic, derives the relevant roofs, and splits the work
into independently killable implementation slices.

## Decision

The current Metal member is `3.562x` faster than the optimized CPU member at
log 26, but it misses the required `5x` bar by `304.880 ms` and the `7x`
target by `520.626 ms`. The shortfall is architectural:

- stage-5 preparation constructs a 40-byte host row, a full equality table,
  table buckets, a second 40-byte resident row, and then another table-major
  address representation;
- each of the 16 address phases scans the large lookup/weight state twice,
  once for RAF statistics and once for lookup-table suffix statistics;
- the first cycle message derives all four virtual-RA factors, and the next
  command derives them again before writing dense Product5 tables;
- the first cycle bind and following message are separate commands, so the
  newly written tables are immediately read back by the GPU.

The v2 architecture has three pieces:

1. A producer-owned, cycle-order instruction-facts carrier and a one-scan,
   six-lane grouped address engine. This is the bounded fallback and the first
   implementation slice.
2. An optional exact-key address-atom representation. It replaces repeated
   rows by one mutable mass per distinct raw address/selector key. It is
   admitted only from an exact target-tape census and an end-to-end roof model.
3. A cycle engine in which the first message writes the four virtual-RA bases,
   the first bind consumes that cache and computes the following message in
   the same dispatch, and later Product5 transitions use bounded-size tiles.

Address work and cycle work have disjoint shaders and state after their typed
handoff. They can be implemented by separate agents. Only the root evaluator
may build or run the serialized CPU/GPU comparisons.

## Fixed evidence and target

The latest record used here is the single-pair diagnostic at
`benchmark-runs/metal-piop-eval/20260807-103715-208977/result.json`:

| property | value |
| --- | ---: |
| result SHA-256 | `32169c799351400ab5f559c8d23a4ff47a07ebdf35bdda1df7bbb0baf6d2f0c2` |
| Metal trace SHA-256 | `f5631125ce234ae93051323631894570c20e18022675c6bee6a3873bb33f74ec` |
| optimized trace SHA-256 | `a5a23a6eeda0059d6a8e50c32df8aef658bf08ae38bc3d835c263e90aac44039` |
| revision | `2ed9ce265f00ca06120a7d4a46fb979ee07919b8` |
| workload | Fibonacci, `T = 2^26 = 67,108,864` |
| machine class | Apple M4 Max, 40-core GPU campaign machine |
| Rayon threads | 16 |
| run class | diagnostic, one optimized-first pair, dirty worktree |
| optimized CPU member | `3775.559408 ms` |
| current Metal member | `1059.991539 ms` |
| current member speedup | `3.561877x` |
| mandatory `5x` cap | `755.111882 ms` |
| `7x` target | `539.365630 ms` |

Both proof-verification guards passed in the enclosing artifact. One pair is
useful attribution, not acceptance evidence. Promotion still requires the
schema-2 paired policy in `HARNESS.md`.

If this member alone reached `5x`, the same diagnostic's Metal PIOP would fall
from `5494.278291 ms` to `5189.398634 ms`, raising the paired PIOP ratio from
`3.692x` to about `3.909x`. At a `7x` member it would fall to
`4973.652382 ms`, or about `4.079x`. These are disjoint-share projections, not
new measurements.

## Protocol contract

Let `j` be a cycle, `k_j` its raw 128-bit lookup address, `t_j` its optional
lookup-table selector, `f_j` its RAF flag, and

```text
u_j = eq(r_reduction, j).
```

The member has 154 variables: 128 address variables bound high-to-low, then 26
cycle variables bound low-to-high. The four virtual-RA factors partition the
128 address coordinates into four 32-coordinate chunks. The input claim is

```text
lookup_output(r_reduction)
  + gamma * left_lookup_operand(r_reduction)
  + gamma^2 * right_lookup_operand(r_reduction).
```

At the terminal point, write `F_t` for the cycle opening of lookup-table flag
`t`, `F_raf` for the RAF-flag opening, `RA_i` for the four virtual-RA openings,
and `U` for the leading-half all-ones polynomial used by the fp128 canonical
address check. The verifier expects

```text
eq(r_reduction, r_cycle) * product_i RA_i * (
    sum_t table_t(r_address) * F_t
  + gamma * left(r_address)
  + gamma^2 * right(r_address)
  + (
        gamma^2 * identity(r_address)
      - gamma * left(r_address)
      - gamma^2 * right(r_address)
      + gamma^3 * U(r_address)
    ) * F_raf
).
```

The `gamma^3 * U` term is present when
`CANONICAL_INSTRUCTION_ADDRESS` is true. It is load-bearing for Akita fp128:
the address topology must compare and retain raw `u128` keys, not keys reduced
modulo the field.

Grouping cycles with the exact same `(t_j, f_j, k_j)` is sound because all
address-side functions and every later condensation multiplier are identical
within that key. Replacing the individual `u_j` values by their exact field sum
is distributivity, not a protocol change. Cycle order is not groupable: the
cycle rounds and final flag openings remain in original cycle order.

### Fiat-Shamir and parity boundary

Fiat-Shamir stays on the host. The device may run only up to the next data
needed by the host transcript.

1. Stage 5 draws the InstructionReadRaf `gamma` before the RAM member's gamma.
2. In clear mode, the batch consumes member input claims in declaration order.
3. The first local `prove_round` receives no bind and returns address round 0.
4. Local calls 1 through 127 first bind the prior address challenge and then
   return the next address polynomial. After every eighth address bind, the
   next 256-entry phase state may be generated.
5. Local call 128 receives the challenge drawn from address round 127, performs
   the final address bind, and only then returns the first cycle polynomial.
6. Every cycle polynomial is combined with the other active stage-5 members,
   absorbed by the host transcript, and followed by one challenge. A later
   cycle command cannot be submitted before that challenge exists.
7. The challenge drawn from cycle round 25 is delivered through
   `finish_rounds`. Output claims cannot be opened before that bind.
8. `r_cycle` is the reverse of the 26 low-to-high cycle challenges.
9. Output claims remain in canonical order: 40 lookup-table flags, four
   virtual-RA values, then the RAF flag.

For a fixed input tape and challenge tape, parity means equality with
`optimized/instruction_read_raf.rs` of the input claim, all 154 member round
polynomials, the terminal member claim, and all 45 output claims. A digest of
only the final proof is not a sufficient local oracle. Address-phase tests must
also compare all six RAF arrays and all 88 declared suffix arrays after every
phase; this catches a wrong table-to-lane map before polynomial interpolation
hides its source.

## Current implementation and measured wall

The complete member wall is exactly the sum of one prepare span, 154 round
spans, `finish_rounds`, and `output_claims`.

| current component | Metal wall |
| --- | ---: |
| prepare total | `530.861000 ms` |
| prepare outside named Metal spans | `251.002292 ms` |
| `sequence_prepare` | `175.352375 ms` |
| initial address phase | `104.506333 ms` |
| 15 later address phase boundaries | `277.008377 ms` |
| 113 ordinary address calls | `29.485584 ms` |
| final address bind and CPU cycle-state initialization | `10.626791 ms` |
| first resident cycle message | `59.158667 ms` |
| resident first bind/handoff | `79.322292 ms` |
| nine resident dense transitions | `56.844794 ms` |
| remaining round wrapper/CPU-tail work | `6.691159 ms` |
| resident readback | `0.799375 ms` |
| output claims | `9.193250 ms` |
| finish | `0.000250 ms` |
| **complete** | **`1059.991539 ms`** |

The optimized CPU control spends `208.898458 ms` in prepare,
`3557.119117 ms` in rounds, and `9.541583 ms` in output claims.

The later address phase boundaries are not launch-noise-sized. Phases 1--7
sum to `155.029625 ms` (`22.147 ms` average); phases 8--15 sum to
`121.978752 ms` (`15.247 ms` average). The `104.506 ms` phase-0 wall is much
larger than any warm phase even though it omits condensation. Because the
buffers use shared unified memory and are initialized by the CPU immediately
before first GPU use, cold residency/coherence is a plausible contributor,
but the artifact has no DRAM or page-fault counters. It must not be labeled as
measured physical traffic.

The nine dense transition walls are, in order,

```text
37.135, 6.468, 3.738, 2.414, 1.886,
1.428, 0.976, 1.367, 1.434 ms.
```

The last rounds are latency dominated; the first transition owns most of the
recoverable dense-cycle wall.

## Current work, traffic, occupancy, and waits

### Preparation and resident representations

Preparation first materializes `InstructionCycleRow[T]` at 40 bytes per row.
The current Metal adapter also materializes a 40-byte `BooleanityRows` device
allocation for later stage-5/6 consumers. `AddressPhaseSequence` then creates
table-major arrays for a one-byte selector, a 16-byte raw lookup, a 16-byte
field weight, and a four-byte cycle-to-table-major inverse. Merely writing
those address arrays is

```text
(1 + 16 + 16 + 4) T = 37 T bytes = 2.3125 GiB.
```

That count excludes reading the 40-byte source rows, reading the 16-byte CPU
equality table, building the buckets, and the separate 40-byte resident-row
copy. The trace does not subdivide the `251.002 ms` unnamed preparation wall,
so those costs need explicit spans before any preparation claim is promoted.

### Address phases

Let `S` be the number of cycles with a lookup-table selector and
`beta = S / T`. For large row state, excluding partial reductions and small
cache-resident tables, the current 16 phases issue

```text
phase 0 RAF       : 33 T       // packed + lookup + weight reads
phase 0 suffix    : 32 S       // lookup + weight reads
phases 1..15 RAF : 15 * 49 T  // plus in-place weight write
phases 1..15 suffix: 15 * 32 S

total = 768 T + 512 S bytes
      = (48 + 32 beta) GiB at log 26.
```

The 15 previous-phase-table loads add `15 GiB` of logical load requests, but
each source table is only 4 KiB; they are reported as cache-requested bytes,
not assumed DRAM bytes.

The RAF partial array is 24 MiB per phase. Its write and finalizer read account
for another 768 MiB across 16 phases. If `J_p` is the suffix-job count for
phase `p`, suffix partial writes add `16 KiB * sum_p J_p`; finalizer reads add
`4 KiB * sum_p sum_t(J_{p,t} * suffix_count_t)`. Address outputs contain
`(6 + 88) * 256` fields. Their GPU write plus host read is 11.75 MiB across
all phases.

The retained M4 Max copy control is `451,701,710,520 B/s` (`420.68 GiB/s`).
It is a roof anchor from the campaign, not a measurement of these shaders.
The large-row traffic above alone has a `114--190 ms` copy floor as `beta`
ranges from zero to one. The observed address-phase wall is `381.515 ms`,
before the `175.352 ms` table-major preparation.

Useful full-field products depend on the actual selector/suffix tape. For a
one-scan implementation they are

```text
T                       split-equality products
+ 15 T                  phase condensation products
+ R                     nontrivial RAF scalar products
+ Q                     nontrivial suffix scalar products
+ 335,872               final flag outer products.
```

The production-table bounds are `1,074,077,696` through
`6,308,569,088` products. At the conservative retained
`16.42 Gproduct/s` control, the arithmetic floor is therefore
`65.413--384.200 ms`. These products do not account for deferred atomic
addition. Each accumulated field term currently issues five 32-bit
threadgroup atomic adds, so a same-binary atomic issue control and an exact
nonzero-contribution census are mandatory.

The useful-product intensity of the grouped one-scan design ranges from about
`0.021` to `0.124 product/byte` before partial traffic. The conservative
machine ridge point is `16.42 / 451.702 = 0.0364 product/byte`; a real tape
with more than roughly 27.5 products per row is compute-side rather than
copy-side under these controls.

The current RAF tile requests 30,720 bytes of dynamic threadgroup memory and
1,024 threads, exactly 32 SIMD groups at the observed SIMD width of 32. The
suffix tile requests 20,480 bytes and the same thread count. At log 26 the RAF
grid has 1,024 threadgroups, or 25.6 groups per GPU core. This is enough grid
parallelism, but exact resident occupancy cannot be recovered from the trace:

```text
resident_groups <= min(
    resident_thread_budget / 1024,
    threadgroup_memory_budget / dynamic_threadgroup_bytes,
    register_budget / registers_per_group,
    architectural_group_limit
).
```

The current telemetry records neither the per-thread register allocation nor
the device's resident budgets. A successor may claim only the 32-SIMD-group
launch shape until compiler/resource evidence is captured. Width tuning
without that evidence is not an occupancy result.

### Cycle phase

The first raw cycle message reads 21 large bytes per row: table-major lookup,
packed selector, and inverse. It derives four RA factors with 12 products per
row, folds `E_in` with one product per row, and evaluates the five-factor grid
with ten products per row. Its exact count is `23 T + 40,960 =
1,543,544,832` useful products.

The current handoff re-derives those 12 RA products per row, writes five
half-domain tables, and then launches a separate Product5 message that rereads
them. Its large-state traffic is approximately `101 T` bytes. Every later
dense transition reads five source fields and writes half as many bound
fields, or 120 bytes per source element.

The raw message uses 128 threads (four SIMD groups) and has at least 25 live
fp128 values in its obvious source-level arrays (`lanes`, `lo`, `hi`, `evals`,
and `steps`) before inlined factor temporaries. Dense transitions use 64
threads (two SIMD groups) with at least 15 array-held fp128 values plus bind
temporaries. Dynamic threadgroup memory is small; registers, instruction
latency, and accumulator lifetime are the likely occupancy constraints. That
is an inference from the source shape, not a register-spill measurement.

### Synchronization

Each address phase uses one command buffer containing four dispatches and then
waits so the host can use the 94 by 256 field output. Sixteen such waits are
protocol-minimal for an eight-address-round phase design. The first raw cycle
message uses one wait, the current handoff uses two, and the nine dense GPU
rounds use nine. Thus the current GPU-served path has 28 command-buffer waits.
Host Fiat-Shamir prevents speculative submission of the next cycle message,
but it does not require the handoff's bind and following message to be separate
commands.

## Producer and carrier contract

The authoritative cycle-order allocation is the existing 40-byte
`BooleanityRows`/`InstructionCycleRow` representation produced in stage 5. V2
adds a typed carrier receipt rather than another anonymous copy:

```text
InstructionFactsCarrier {
    rows: T,
    device_registry_id,
    source_allocation_id,        // existing BooleanityRows
    lookup_cycle_order: u128[T], // optional compact SoA view
    claim_cycle_order: u8[T],    // table_plus_one | raf_flag << 7
    topology_receipt,
}
```

The compact views are transcript-independent. They must be co-produced once,
outside the PIOP timing boundary defined by `piop_goal.v2.json`, or projected
by one producer dispatch from the existing rows. They are consumed by the
address topology, the first cycle message, final flag openings, and the later
InstructionRaVirtualization backend. Every consumer checks length, device
registry, allocation identity, and source provenance. No member-local upload
or standalone CPU repack is allowed.

The diagnostic metric still charges transcript-independent backend
representation materialization. Moving work outside `jolt_prover::piop` is
not permission to hide it: evidence reports producer wall, GPU-active time,
bytes, allocations, and the consumers among which it is amortized.

Two topology receipts are permitted:

- `GroupedAddressTopology`: 82 stable segments in
  `2 * table_plus_one + raf_flag` order, segment ranges, and a
  group-major-to-cycle index. It is an O(T) counting layout and is the required
  fallback.
- `AtomAddressTopology`: CSR cycles grouped by the exact
  `(table_plus_one, raf_flag, raw_lookup_u128)` key, plus raw atom keys and
  segment ranges. It is optional. Its construction must be producer-owned and
  separately charged; a hash or radix sort may not appear silently inside
  `InstructionReadRaf::prepare`.

The current table-major lookup plus cycle inverse may remain temporarily for
the old InstructionRaVirtualization path. It is not part of the v2 canonical
carrier, and the v2 virtualization consumer should use the cycle-order lookup
plane directly.

## V2 address engine

### Grouped one-scan fallback

The group descriptor supplies the table and RAF selectors. Phase zero reads
the group-major cycle index and raw lookup, derives the exact split-equality
weight, writes that weight once, and emits all needed statistics. Later phases
read one raw lookup and one weight, multiply by the preceding 256-entry phase
table, overwrite the weight, and emit the next statistics.

One segment needs six 256-bin accumulator lanes:

- lanes 0--2 hold the three RAF statistics for that segment's RAF branch;
- a declared `Suffixes::One`, wherever it appears, aliases lane 0;
- the other declared suffixes use lanes 3--5;
- a checked 160-byte table/slot map restores the declared suffix order.

The plan fails closed unless there are 40 tables, 88 total declared suffixes,
at most four suffixes per table, and at most three non-`One` suffixes per
table. This avoids the false assumption that `One` is always suffix slot 0.

With at most 65,536 rows per job, 82 segments produce 1,024--1,105 jobs at log
26. The one-scan issued-byte upper bound, including partials and address-output
write/read, is `48.138245 GiB`. Its copy floor is `114.430 ms`; the 80%-of-roof
cap is `143.037 ms`. Its useful-product floor remains tape-dependent, so the
actual product and deferred-atomic roofs can supersede the copy cap.

The grouped path uses one command buffer per phase. Phase commands contain one
fused tile dispatch and the minimum finalization dispatches; they do not launch
independent RAF and suffix row scans. The host retains the current 256-entry
prefix/suffix binding and eight address messages per phase.

### Exact-key atom path

Let `U` be the number of exact address atoms. After `r_reduction` is known,
phase zero computes

```text
mass[a] = sum_{j in cycles(a)} eq(r_reduction, j)
```

and immediately emits the atom's first-phase contribution. Phases 1--15 scan
`U` atoms, multiply each mass by the preceding phase table at the atom's raw
lookup byte, overwrite the mass, and emit the same six-lane output as the
grouped oracle.

Phase-zero jobs are balanced by cycle count. An atom larger than the fixed
cycle budget is split into mass partials and finalized once; a giant repeated
atom must not serialize onto one SIMD group. Later phases lower their
atoms-per-job setting as needed to retain at least four waves on the 40-core
GPU. At `U/T = 1/8`, 65,536 atoms per job gives only 128 base jobs, so the job
size must be reduced to reach at least 160 jobs.

Let `M` be phase-zero mass jobs and `P` split-mass partials. Before six-lane
partial traffic, the logical issued large-row bytes are

```text
4 T + 736 U + 32 M + 32 P.
```

Useful products are

```text
T + 15 U + R(U, P) + Q(U, P) + 335,872,
```

bounded by `[T + 15 U, T + 93 U + 5 P] + 335,872`. The following optimistic
rows use `M = U`, `P = 0`, and exclude six-lane partial traffic and atomics:

| `U/T` | large-row traffic | copy floor | upper useful products | compute floor | 80%-roof compute cap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.125 | `6.250 GiB` | `14.857 ms` | `847,585,280` | `51.619 ms` | `64.524 ms` |
| 0.250 | `12.250 GiB` | `29.120 ms` | `1,627,725,824` | `99.131 ms` | `123.914 ms` |
| 0.500 | `24.250 GiB` | `57.645 ms` | `3,188,006,912` | `194.154 ms` | `242.693 ms` |

`U/T <= 0.25` is therefore a useful worst-case screening threshold, not an
admission rule. The exact tape may pass above it if suffix products are sparse,
or fail below it because atom construction, split jobs, or atomics dominate.
Admission uses

```text
1.25 * max(
    issued_bytes / measured_copy_rate,
    useful_products / matched_product_rate,
    atomic_word_adds / matched_atomic_rate
) + validated_command_floor.
```

The atom path is rejected if that model does not beat the grouped path by at
least 20% after charging the topology producer diagnostic.

## V2 cycle engine

The cycle side consumes the cycle-order compact lookup and claim planes. It
does not gather through a table-major inverse.

1. The first raw message derives all four virtual-RA factors once. While those
   values are live, it evaluates the five-factor message and writes the four
   RA bases to a factor-major cache.
2. The first cycle challenge dispatch consumes four raw cycles at a time. It
   binds the four cached RA factors and the claim-derived value factor, writes
   the five half-domain factors, and evaluates the next message from the bound
   values before they leave registers.
3. Later dense transitions use fixed 64-pair tiles. Tiles shorten accumulator
   lifetime and make the partial-traffic cost explicit. A reduction over tile
   partials produces the five host-visible values.
4. The sequence switches to the CPU at a measured cutoff. `2^16` remains the
   initial control, not a protocol constant.

At log 26, using the retained matched controls, the primary cycle model is:

| phase | useful products | large-state bytes | traffic floor | compute floor | 80%-roof cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| first message plus RA-cache write | `1,543,544,832` | `5,435,817,984` | `12.034 ms` | `47.218 ms` at 32.69 G/s | `59.022 ms` |
| cached first bind plus next message | `536,911,872` | `7,046,430,720` | `15.600 ms` | `29.664 ms` at 18.10 G/s | `37.080 ms` |
| nine dense transitions | `536,190,976` | `8,037,335,040` | `17.793 ms` | `29.624 ms` at 18.10 G/s | `37.030 ms` |

The corresponding intensities are `0.284`, `0.076`, and `0.0667`
product/byte, all above the `0.0401 product/byte` ridge point of the
18.10-G/s register-pressure control. These are compute-side kernels unless
register spilling inflates physical traffic.

Adding the retained `0.141 ms` command-wall floor for eleven GPU messages gives
a `134.682 ms` cycle target. The current three measured spans sum to
`195.326 ms`; v2 must recover at least 45 ms net to justify its cache and
partial storage.

The first-message cache adds four full-domain field arrays and the dense
sequence owns five half-domain arrays. Their incremental peak is `104 T = 6.5
GiB` before ping-pong slack and partials. Working-set admission and allocation
identity telemetry are hard gates; a cache that spills the process or forces
another carrier copy is not a win.

## Bounded implementation slices

Each slice gets its own schema-2 phase. Scope cannot expand after its baseline.
The two address candidates and the cycle candidate use separate shader entry
points and can be written in parallel, but representative evaluation remains
single-owner and serialized.

### Slice A: carrier attachment and grouped address fusion

Scope:

- typed borrow of the producer-owned cycle facts and grouped topology;
- split-equality materialization inside phase zero;
- one six-lane row scan per address phase;
- scalar oracle for all 94 phase outputs and all 128 address polynomials;
- exact spans, counts, traffic census, pipeline limits, and allocation IDs.

Pre-registered variants: 1,024-thread fused tile, then one 512-thread variant
only if resource telemetry shows a residency gain. No other width sweep.

Kill gates:

- any phase output or address polynomial differs from the optimized oracle;
- a member-local row upload/repack or a second large row scan appears;
- the exact product/atomic model puts the phase sequence above its 80%-roof
  cap before compilation;
- transcript-dependent address layout + 16 phase walls + ordinary address
  messages exceeds `260 ms` at log 26, or the projected complete member does
  not fit the `755.112 ms` 5x cap.

If the one-shot checkpoint misses, do not spend the phase tuning threadgroup
widths. Preserve the grouped path as a parity oracle and proceed only to the
pre-registered atom phase.

### Slice B: first-message RA cache and fused cycle handoff

Scope:

- cycle-order compact facts;
- first-message-owned four-factor RA cache;
- fused cached bind plus following message;
- 64-pair tiled dense transition and reduction;
- current CPU-tail export and exact final-state oracle.

Kill gates:

- first message exceeds `62 ms`, cached handoff exceeds `39 ms`, or all nine
  dense transitions exceed `39 ms` in the one-shot representative checkpoint;
- total cycle wall exceeds `140 ms` or saves less than `45 ms` against the
  exact `195.326 ms` current spans;
- register/resource evidence shows spills or lower occupancy that invalidate
  the matched-rate model;
- incremental working-set admission fails.

One pre-registered 32-pair tile is allowed only if 64-pair telemetry shows
register pressure rather than arithmetic throughput. Otherwise the phase is
exhausted.

### Slice C: exact-key atom address path

This slice starts only after Slice A records a target-tape topology census.
It owns the producer-side CSR ABI, mass jobs, split-atom finalizer, and atom
phase shader. It does not edit the cycle engine.

Kill gates:

- topology does not use the raw 128-bit key or crosses a table/RAF segment;
- `M != U - split_atoms + P`, a giant atom is assigned to one serial job, or
  later phases have fewer than four scheduled waves;
- the exact traffic/product/atomic model predicts less than 20% gain over the
  grouped path;
- topology materialization makes the diagnostic PIOP-plus-backend-prepare
  result worse;
- atom address wall exceeds `125 ms`, or the integrated complete member does
  not fit the `539.366 ms` 7x target.

The grouped path remains the correctness and high-`U/T` fallback. A failed atom
path does not weaken the 5x requirement.

## Evaluator and required telemetry

The cheap evaluator consumes a frozen production-derived carrier, topology,
`r_reduction`, `gamma`, and challenge tape. It runs the optimized and candidate
members on the same data and reports separate address, cycle, and complete
digests. A smaller scale is a ranking proxy only after its ordering is
calibrated against log 26; topology ratio, job geometry, or occupancy drift
disables the proxy.

Every candidate record must include:

- `T`, `S`, `U`, atoms per segment, cycle-count histogram, `M`, `P`, split
  atoms, jobs per phase, and waves per 40-core GPU;
- exact useful products, nonzero accumulator contributions, five-word atomic
  adds, large issued bytes, cache-requested bytes, compulsory bytes, and peak
  resident allocation;
- per-pipeline execution width, maximum threads, static and dynamic
  threadgroup memory, compiled register/spill evidence when available, and the
  requested/effective threadgroup width;
- per-phase GPU-active and wall time, command buffers, dispatches, waits,
  output bytes, readbacks, and round allocations;
- producer allocation IDs and bytes, carrier consumer IDs, row uploads, and
  whether producer time is inside the PIOP or diagnostic boundary;
- address phase outputs, every member polynomial, challenge tape, terminal
  factor state, output claims, and proof-verification guards.

Logical traffic comes from the checked model. Physical DRAM traffic may be
reported only when a supported counter capture exists; otherwise evidence
reports a compulsory lower bound and issued upper bound. Likewise, 1,024
threads or 32 SIMD groups is a dispatch shape, not an occupancy percentage.

## Code organization after promotion

The final production family should be self-contained:

```text
src/metal/solinas/instruction_read_raf/
    mod.rs          runtime owner and public sequence API
    abi.rs          repr(C) params, receipts, and checked lane map
    carrier.rs      producer borrow and provenance validation
    address.rs      grouped/atom address sequence
    cycle.rs        RA cache, Product5 sequence, and CPU handoff
    model.rs        pure checked work/traffic/roof arithmetic
    oracle.rs       scalar phase and cycle oracle
    tests.rs        ABI, parity, lifecycle, and model tests
    shader.metal    family-specific entry points only
```

Shared fp128 arithmetic remains at the Solinas root. The family must not add
InstructionReadRaf-specific branches to the generic `product5` module, and it
must not leave competing production shaders in successor directories. Isolated
rejected prototypes retain only their evidence and design record.

## Falsifiers and unresolved measurements

The first measurements should try to disprove the design, not tune it:

1. Record the exact `U/T`, atom-size distribution, table density `beta`, and
   scalar/atomic census on the log-26 production tape.
2. Separate producer, cold phase-0, warm address phases, host messages, raw
   cycle message, cached handoff, dense rounds, and output openings with GPU
   timestamps and complete wall spans.
3. Capture pipeline resource limits and register/spill evidence. If the
   six-lane address tile or tiled Product5 kernel is register-limited, the
   arithmetic roof is inapplicable.
4. Charge the carrier and topology producer in the diagnostic metric. A fast
   member backed by a slower one-off representation is not an end-to-end win.

Until those four items exist, the grouped engine is the implementation target,
the atom table is conditional headroom, and `7x` is a target rather than a
performance claim.
