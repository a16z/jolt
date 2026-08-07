# RAM RA claim-reduction Metal contract

This directory contains executable first-principles Q and H-prime device
passes for `RamRaClaimReduction`. Their shaders are registered in the shared
Metal library, the Rust runtime is checked, and Criterion measures the resident
compact layouts. It is not yet a production backend or a complete sumcheck
member.

## Observed resident Q slice

The original dense scan was analytically wrong for sparse RAM traces. At the
target density, almost every 32-lane SIMDgroup enters the multiply branch, so
masked lanes consume nearly `3T` product slots even though only `3A` products
are useful. Replacing local accumulator arrays did not fix that mechanism: ten
same-binary pairs measured only a 1.011x active improvement.

The retained candidate stores accessed rows in low-index-major order. Each
32-bit entry packs `(hi, address)`, and an 8,193-element offset table identifies
each low index's contiguous range. Eight count-balanced partitions make every
issued lane perform useful products and preserve output ownership without field
atomics. On the Apple M4 Max at log 26 with `A = 22,000,000`, ten warmed,
same-binary alternating pairs measured:

```text
                         compact access     dense explicit    speedup
GPU active median:          1.799875 ms        5.604917 ms      3.114x
resident wall median:       2.915791 ms        6.645541 ms      2.279x
useful full products:      66,000,000
```

Criterion independently measured `1.8002 ms` active, or `36.663` billion
useful products/s, and `2.6441 ms` resident wall. Partition screens retained
eight partitions: one partition was clearly slower, four was slightly slower,
and eight and sixteen had indistinguishable wall time while sixteen doubled
partial storage. The compact Q slice uses 22.2% of the complete CPU member's
8.1015-ms 5x budget. Full raw samples and limitations are frozen in
`screening_evidence.json`.

The standalone fixture spends about 196 ms constructing and uploading both
layouts; that time is excluded from the resident slice. This exclusion is valid
only if the shared RAM producer emits the compact layout during its existing
trace walk. A production PIOP comparison must charge any separate transpose or
upload. The complete member still needs the host rounds/transcript bridge and
alternating end-to-end validation.

The matched compact roof explains the observation. The full-width multiply
probe sustains `45.709` billion products/s, so 66 million useful products have
a `1.444-ms` arithmetic floor and a `1.805-ms` 80%-of-roof pursuit cap. The
measured `1.8002 ms` is 80.2% of that probe ceiling. With perfect lookup-table
reuse, the compact producer/reducer has `95,241,748` compulsory bytes, only a
`0.211-ms` traffic floor. Its `1,502,717,460` shader-requested bytes would take
`3.328 ms` with no lookup-cache reuse; the measurement therefore also confirms
that the 128-KiB address table and 384-KiB high-equality tables are being served
effectively from cache.

## Observed resident H-prime slice

The high-index-major compact gather performs the remaining `A = 22,000,000`
useful products in `0.57147 ms` active at log 26 after a 200-ms residency
warmup, or `38.498` billion useful products/s. Its Criterion interval was
`0.57009--0.57570 ms`. This is 84.2% of the matched one-chain full-width
probe's `45.709 Gproduct/s` ceiling, so no further shader variants are
justified. The resident-wall estimate was `1.8601 ms`, with a wide
`1.3907--2.2337 ms` interval; active time is the stable kernel metric and the
complete-member evaluator must resolve controller noise.

Together, the two accepted device passes take `2.37167 ms` active for 88
million useful products. Their matched arithmetic floor is `1.925223 ms`, so
the pair reaches 81.2% of the relevant product ceiling. The independent
resident-wall estimates sum to `4.5042 ms`, leaving comfortable room under the
`8.1015-ms` 5x member cap but not enough evidence by themselves for a complete
member claim. The raw samples and limitations for the second pass are frozen
in `gather_screening_evidence.json`.

## Frozen evaluator and hard target

The fixed denominator is the optimized-CPU member in the Fibonacci log-26
capture at
`benchmark-runs/metal-piop-eval/20260806-133709-697013`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`. The five complete-member samples
are:

```text
42.226419, 41.593040, 39.237041, 35.865838, 40.507503 ms
```

The median is `40.507503 ms`, so the exact 5x cap is `8.1015006 ms`; integer
acceptance uses `5 * metal_ns <= 40_507_503`. Component medians are
`35.368166 ms` for prepare and `5.620082 ms` for all 26 rounds. The phase-switch
gather occurs in message 13 (zero-based) and has a `4.685375 ms` median; the
remaining round work is about `0.935 ms`. Complete-member time, not a sum of
component medians, is authoritative.

The current Metal arm still runs the CPU kernel. Its five complete samples have
a `35.838878 ms` median and are not GPU evidence. The frozen member spans omit
generic batch-owned host Fiat-Shamir. A production comparison must charge that
work to both arms or neither; the intended PIOP evaluator charges it to both.

The retained M4 Max controls are `420.68 GiB/s` (`451,701,710,520 B/s`)
streaming copy bandwidth and `45.709 Gproduct/s` for the matched one-chain
full-width Solinas probe. They are reusable primitive measurements, not either
shader's result. Both compact kernels call `solinas_mul_wide` on canonical
equality-table values and perform zero half-width products.

## Exact relation and ordering

Stage 5 draws the instruction batching challenge, then this relation's `gamma`,
before proving a three-member batch in declaration order:

1. `InstructionReadRaf`;
2. `RamRaClaimReduction`;
3. `RegistersValEvaluation`.

This relation consumes three `RamRa` openings:

- `raf` from stage-2 `RamRafEvaluation`;
- `read_write` from stage-2 `RamReadWriteChecking`; and
- `val_check` from stage-4 `RamValCheck`.

Each point has `log_K + log_T` coordinates. The verifier rejects them unless
their first `log_K` address coordinates agree. At the target,
`log_K = 13`, `log_T = 26`, and the degree-two sumcheck has 26 low-to-high
cycle rounds. For remapped word address `a(j)`, with `u32::MAX` denoting no
access, define

```text
H(j) = 0                                      if a(j) is no-access
     = eq(r_address, a(j))                    otherwise.

S(j) = H(j) * (E_0(j) + gamma E_1(j) + gamma^2 E_2(j)),
E_x(j) = eq(r_cycle_x, j).
```

The input claim is

```text
raf + gamma * read_write + gamma^2 * val_check = sum_j S(j).
```

After all challenges, the relation returns one `RamRa` value at

```text
[common_address || reverse(sumcheck_challenges)].
```

It also checks `EqCycleRaf`, `EqCycleReadWrite`, and `EqCycleValCheck` between
the three fixed input cycle points and that reversed output point. The output
is consumed by stage-6b `RamRaVirtualization`. The host batch driver owns every
absorb and challenge; neither shader hashes or advances transcript state.

## Chosen rank-three split

Let

```text
p = floor(log_T / 2), q = log_T - p,
I = 2^p, O = 2^q,
j = hi * I + lo.
```

At log 26, `p = q = 13` and `I = O = 8,192`. Split each cycle equality as

```text
P_x(lo) = eq(r_cycle_x low half, lo),
E_hi_x(hi) = eq(r_cycle_x high half, hi),
Q_x(lo) = sum_hi H(hi, lo) E_hi_x(hi).
```

Cycle points are stored in big-endian multilinear order while the prover binds
low-to-high. Therefore `P_x` is built from `r_cycle_x[q..]`, `E_hi_x` from
`r_cycle_x[..q]`, and the row address is exactly `hi * I + lo`. Reversing the
first `p` challenges before constructing `E_prefix` is required; omitting that
reversal or swapping the point slices changes the round messages.

The first 13 messages run on the host over the six 8,192-field `P/Q` tables.
After host challenge 12, form

```text
E_prefix(lo) = eq(reverse(prefix challenges), lo),
H'(hi) = sum_lo H(hi, lo) E_prefix(lo),
scale_x = eq(r_cycle_x low half, reverse(prefix challenges)).
```

The final 13 messages run on the host over `H'` and the three `E_hi` tables.
This is algebraically identical to the optimized CPU algorithm. It uses exactly
`3A` useful full products to build `Q` and `A` to build `H'`, where `A` is the
number of accessed cycles. The rank-three `Q` state is necessary because the
three low-half factors differ; multiplying `H` by one gamma-batched high factor
would lose the independent `P_x` factors required by later prefix messages.

A dense device sumcheck, a retained full `H` table, and producer-side retention
of stage-4 `(inc, address)` rows are rejected. They add repeated protocol waits
or 1-GiB field/row state without removing the `4A` products. The balanced split
minimizes small-table storage while leaving enough workgroups in both device
passes.

## Two-wait hybrid schedule

The preferred device work uses compact access views for both scans:

1. **Q build.** The shared producer groups packed `(hi, address)` entries by
   `lo` and provides `I + 1` offsets. Dispatch `8I` threads in one-SIMDgroup,
   32-thread groups. Thread `(partition, lo)` owns a count-balanced eighth of
   that low index's contiguous entries, keeps three field accumulators, and
   writes term-major, partition-major partials. A product-free second dispatch
   sums eight partials into each `Q_x(lo)`. Both dispatches occupy one command
   buffer and have no output atomics. One SIMD-aggregated counter update per
   producer group audits the exact compact entry count.
2. Read back `Q` (`384 KiB` at the target), execute prefix messages 0 through
   12 on the host, and draw challenge 12 on the host.
3. **H-prime gather.** A second compact view groups packed `(lo, address)`
   entries by `hi`. One 32-thread SIMDgroup per high index walks only its
   accessed entries, reduces in-register partials, and writes `H'(hi)` without
   field atomics. This pass is implemented and benchmarked.
4. Read back `H'` (`128 KiB`), execute messages 13 through 25 and the terminal
   bind on the host, then return `H'[0]` and validate the three derived values.

There are three dispatches, exactly two command buffers and waits, and 512 KiB
of mapped field readback. No round allocates device memory. Command buffer 1
first clears the 16-byte counter buffer, then encodes Q production and Q
reduction. After its wait, the host rejects a nonzero unsupported or Q-invalid
counter and requires `q_accessed_rows == validated_A` before consuming Q.
Command buffer 2 encodes only H gather; after its wait, the host also requires a
zero gather-invalid counter while preserving the first audit. `Q` is independent
of `gamma`; once stage 4 has produced the third input point, it may be submitted
early and overlapped with other stage-5 preparation. Independent member
accounting still charges its full service interval.

## Implemented device slices

The code in this directory implements both compact device passes. The Q
producer and product-free reducer use one command buffer and one wait; the
H-prime gather is a separately prepared command because the host must draw the
prefix challenges between them. Both can be checked independently of
transcript state. The host transition and production handoff are still absent,
so this does not yet claim to accelerate the complete relation.

`RamRaClaimQPlan` freezes the config, ABI parameters, dispatch geometry, and
allocation sizes. At log 26 the benchmark allocation contract is:

| Q-slice storage | Bytes |
| --- | ---: |
| Borrowed resident address plane | 268,435,456 |
| Borrowed compact low-major entries | 88,000,000 |
| Borrowed compact low-major offsets | 32,772 |
| Host-uploaded `eq_address` | 131,072 |
| Host-uploaded three-term `E_hi` | 393,216 |
| GPU-private eight-way Q partials | 3,145,728 |
| Shared final Q output | 393,216 |
| Shared audit counters | 16 |
| Sequence-owned total | 4,063,248 |
| Candidate-plus-control working set | 360,531,476 |
| Mapped readback | 393,232 |

The production path is expected to attach an immutable address allocation
whose metadata was created by a crate-internal producer after content
validation. The metadata constructor is not public: callers cannot forge a
low access count to enter the Metal path. Preparation rechecks byte length,
device and allocation identity, exact shape, trace cutoff, and density before
allocating or compiling the Q invocation. Dispatch rechecks them again, clears
the counters, executes both kernels, waits once, and rejects unsupported
geometry, an invalid address, or an accessed-row count different from the
validated count before reading Q.

The standalone `prepare_ram_ra_claim_addresses` helper validates and uploads a
private fixture for parity tests. Its upload is not a resident benchmark result.
`prepare_ram_ra_claim_q` builds equality tables, uploads them, allocates scratch
and output, and compiles both pipelines. `execute_timed` performs zero device
allocations. Its `resident_wall` begins before validation and includes counter
clear, both dispatches, the wait, counter audit, canonical Q conversion, and
checksum; it excludes address upload, equality generation, allocation, and
pipeline compilation. A complete PIOP comparison must report preparation
separately and charge any work that is neither reused nor overlapped.

Independent host tests compare the eight partition-major partial tables and
their reduction with a direct Q construction. Full-relation fixtures compare
the split algorithm with an unfactored dense relation for even and odd splits,
including no-access, one-access, domain-edge, alternating, dense, random sparse,
zero/one/minus-one/random gamma, an invalid address, checksum ordering, and
resident metadata/counter failures. The registered dense-array, dense-explicit,
and compact kernels all pass the actual GPU parity test against the independent
direct oracle.

## Ownership and resident handoffs

The `u32` address plane is shared RAM-family state. The optimized CPU lifecycle
creates it at the first stage-2 RAM consumer, reuses it in stages 4 and 5, and
takes the final session reference in stage-6b `RamRaVirtualization`. A Metal
producer must carry row count, byte length, address limit, exact accessed-row
count, device registry ID, storage identity, and validated provenance. The
buffer remains immutable from validation through the second command-buffer
completion; detached metadata cannot establish that invariant.

The shared owner, rather than this sequence, survives the stage-5 return.
Stage 6b takes that owner, holds it through RAM-RA lazy materialization command
completion, and then releases it. A stage-5 error or CPU fallback must return
without consuming the owner, so stage 6b observes the same storage identity.

This member only borrows the allocation. It neither uploads nor repacks it and
must preserve it for stage 6b. The accessed-row count is part of correctness of
the performance admission: if it is absent, stale, or disagrees with the Q
shader's audit counter, the Metal path is rejected. The checked host API routes
execution through `execution_for_validated_plane` and validates final counters
with `validate_completed_dispatches`. A standalone benchmark may measure a
private upload, but its upload and validation must be reported and charged; it
is not a resident result.

Stage-4 `RamValCheck` already needs the same `eq(r_address, .)` table. Passing
that 128-KiB allocation or its canonical host contents forward is worthwhile,
but it saves no `4A` product work. Keeping a 1-GiB dense `H(j)` table or the
stage-4 16-byte rows is not.

At log 26, target storage is:

| Storage | Bytes |
| --- | ---: |
| Borrowed address plane | 268,435,456 (256 MiB) |
| Device `eq_address` | 131,072 |
| Device three `E_hi` tables | 393,216 |
| Device eight-way three-term `Q` partials | 3,145,728 (3 MiB) |
| Device three final `Q` outputs | 393,216 |
| Device `E_prefix` | 131,072 |
| Device `H'` output | 131,072 |
| Sequence-owned device fields | 4,325,376 (4.125 MiB) |
| Host `eq_address`, three `P/Q/E_hi`, `E_prefix`, `H'` | 1,572,864 (1.5 MiB) |

Buffers are allocated once per sequence. The `Q` and `H'` outputs can reuse
non-overlapping storage during integration, but the conservative plan does not
depend on that alias.

## Shader ABI

Concatenate sources in this order:

1. `fp128.metal` with `SOLINAS_OFFSET = 0xffff_a7f7`;
2. `simd_reduce.metal`;
3. this directory's `shader.metal`.

Register only:

| Entry point | Purpose |
| --- | --- |
| `solinas_ram_ra_claim_build_q_partials` | one coalesced address scan, eight-way partials |
| `solinas_ram_ra_claim_build_q_partials_compact` | low-major compact Q scan |
| `solinas_ram_ra_claim_reduce_q` | sum partials into three final `Q` tables |
| `solinas_ram_ra_claim_gather_h` | one coalesced address scan, one `H'` output |
| `solinas_ram_ra_claim_gather_h_compact` | high-major compact H-prime scan |

The producer buffers are:

```text
0 borrowed resident u32 address plane
1 eq_address, K canonical fields
2 term-major E_hi, 3 * O canonical fields
3 term-major, partition-major Q partials, 3 * 8 * I canonical fields
4 cleared RamRaClaimCounters
5 RamRaClaimParams
```

The compact bindings are exact:

```text
Q reducer: 0 Q partials, 1 final Q, 2 counters, 3 params
Q compact producer: 0 entries, 1 offsets, 2 eq_address, 3 E_hi,
                    4 partials, 5 counters, 6 params
H compact gather:   0 entries, 1 offsets, 2 eq_address, 3 E_prefix,
                    4 H', 5 counters, 6 params
```

Use `dispatchThreadgroups`, never a rounded `dispatchThreads`: producer
`(I / 32) * 8` groups, reducer `I / 32` groups, and gather `O` groups, all with
`(32, 1, 1)` threads. The shader checks the actual
`threads_per_threadgroup`, not only `params.threads`. The runtime must also
require pipeline `threadExecutionWidth == 32` before submission. `Q` and `H'`
are host-visible shared buffers; Q partials are GPU-only sequence scratch; the
counter buffer is shared and cleared once before command buffer 1. Any other
partition count, SIMD width, group width, term count, shape product, sentinel,
or address limit is unsupported.

## Roofline, density admission, and phase bars

Each compact pass reads one four-byte entry per access plus one offset table.
The exact compulsory bytes and source-level shader-requested lookup bytes are:

```text
B_Q_compulsory = 4A + 4(I + 1) + 16K + 3(16O)
                 + 2(3 * 8 * I * 16) + 3I * 16 + 16
B_Q_lookup = 4A * 16

B_H_compulsory = 4A + 4(O + 1) + 16K + 16I + 16O + 16
B_H_lookup = 2 * 16A.
```

Q requests one address-equality value and three high-equality values per
access. H requests one address-equality value and one prefix-equality value.
At `T = 2^26` and `A = 22,000,000`:

| Phase | Compulsory bytes | Logical lookup bytes | Total shader-requested | Copy floor, compulsory | Copy floor, all requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q | 95,241,748 B | 1,408,000,000 B | 1,502,717,460 B | 0.210851 ms | 3.326792 ms |
| H' | 88,426,004 B | 704,000,000 B | 792,163,860 B | 0.195762 ms | 1.753733 ms |

The perfect-cache column is the optimistic off-chip floor. The no-cache column
charges every requested 16-byte field exactly once; it is a sensitivity bound,
not a strict hardware upper bound, because cache-line overfetch can add bytes.
Actual off-chip traffic should fall between those columns only when line
amplification is negligible. The unique lookup working sets are small: 128 KiB
for `eq_address`, 384 KiB for all `E_hi`, and 128 KiB for `E_prefix`. Promotion
nonetheless requires hardware counters rather than assuming those repeated
requests hit.

The table excludes constant-parameter fetches and cache-line amplification; it
includes the 16 logical counter bytes. At
the target, Q performs between `ceil(A / 32768)` and `min(A, 2048)`
SIMD-aggregated accessed-counter atomics depending on row distribution; H
performs at most one per high index. Those atomics are required telemetry but are too small and too
cache-dependent to disguise as streaming bytes.

Useful arithmetic is exactly `3A` full-width products in Q and `A` full-width
products in H'. There are no half-width products. At the registered gate
`A <= 22,000,000` (`32.7826%` of log-26 rows), the matched one-chain probe gives:

| Phase | Full-width floor | 80%-of-probe pursuit | Measured active |
| --- | ---: | ---: | ---: |
| Q build (`3A`) | 1.443917 ms | 1.804897 ms | 1.8002 ms |
| H-prime gather (`A`) | 0.481306 ms | 0.601633 ms | 0.57147 ms |
| Both device passes | 1.925223 ms | 2.406530 ms | 2.37167 ms |

With perfect lookup caching, the model reserves `1.500 ms` for the host phases,
submissions/waits, readback, checks, and Fiat-Shamir. Its conditional complete
projection is `3.906530 ms`, or `10.37x`. Charging every source-level lookup
request at streaming bandwidth gives `7.850657 ms`, or `5.16x`. These remain
projections until the host bridge and resident producer are measured together.

At dense `A = T`, the matched arithmetic floor is `5.872705 ms`; its
80%-of-probe pursuit plus the fixed envelope is `8.840882 ms`, or 4.58x. Dense
inputs therefore remain on optimized CPU. The current `A = 22,000,000` gate is
also the largest frozen point whose pessimistic all-lookup traffic model clears
5x with useful margin. The access census for the frozen Fibonacci artifact was
not retained, so eligibility of that exact workload is unknown until a producer
counter is measured.

Promotion bars are:

| Phase | Bar |
| --- | ---: |
| Resident handoff | zero upload; exact validated `A <= 22,000,000` |
| Q build | accepted at 1.8002 ms active |
| Host prefix through challenge 12 | <= 0.65 ms |
| H-prime gather | accepted at 0.57147 ms active |
| All non-device work, including both host phases and waits | <= 1.50 ms |
| Complete hybrid member | hard `5 * metal_ns <= 40_507_503`; pursue <= 5.06 ms when 8x is clear |

If measured density or cache behavior makes substantially more than 5x
available, the pursuit target is tightened instead of stopping at the minimum.
The default trace-length cutoff is `2^26`, the only retained scale with a frozen
CPU denominator and the first power-of-two scale that clears 5x under a linear
same-density extrapolation of the current variable and fixed costs. That
extrapolation predicts a continuous 5x crossover near `2^25.568`; it is not a
measurement. Paired multi-scale measurements may lower the cutoff. Density
rejection applies at every trace length.

## Occupancy, measurements, and parity

The eight-way Q producer has 2,048 independent one-SIMDgroup threadgroups at
log 26, or 51.2 group-waves per core on the measured 40-core M4 Max. That is
enough launch supply to test a 32-resident-SIMDgroup/core target, but it does not
establish occupancy. The actual compiled register limit remains a measurement
gate. The short product-free reducer has 256 groups and is not on the arithmetic
roof. The H launch has 8,192 groups and only one dependent accumulator per lane;
Q has three. Neither has the six independent chains of the retained throughput
control. Promotion requires:

- `threadExecutionWidth == 32` and an admitted 32-thread group size;
- compiled register counts and spill counters for both pipelines;
- enough simultaneously resident one-SIMDgroup groups to hide multiplication
  latency, with no persistently idle core in either pass;
- GPU active time, wall time, submissions, waits, dispatches, allocations,
  upload bytes, readback bytes, storage identities, `A`, invalid rows, and
  achieved useful products/s per pass;
- L2/system-memory counters proving the small lookup tables are resident, or a
  revised traffic roof that charges their misses; and
- separate projected and measured columns. A projection is never promoted as
  a benchmark result.

Required deterministic parity vectors include all no-access, one accessed row,
all one address, alternating no-access/access, address-domain edges, invalid
addresses, random sparse traces below and above the density gate, zero/one
cycle coordinates, zero/one `gamma`, odd `log_T`, and the retained Fibonacci
workload. For each vector, compare direct dense relation values, split `Q`,
the sum of all eight shader partials for every final `Q` entry, `H'`, all 26
`[s(0), s(2)]` messages under identical challenges, the final `RamRa` value,
output point order, and all three derived cycle equalities.

## Kill and redesign rules

Do not integrate this sketch if any of the following holds:

- the resident producer cannot provide a validated exact accessed-row count;
- the target workload exceeds the density gate;
- either kernel spills enough to invalidate the product roof;
- measured lookup misses push the complete projection or observation below 5x;
- complete paired service time misses 5x after two focused shader variants; or
- parity requires changing transcript order, challenge order, or output-point
  derivation.

If density is too high, redesign the protocol or upstream representation before
writing more variants. A useful redesign must reduce the generic rank-three
`4A` product count, not merely move it to another stage or omit its ownership
cost from the evaluator.

## Remaining integration sequence and blockers

The module, shared source registration, independent GPU parity tests, and both
Criterion device microbenchmarks are complete. Continue in this order:

1. make the shared RAM trace producer emit both compact orderings during its
   existing walk, with typed provenance and no standalone transpose;
2. add host prefix/suffix transitions and the transcript bridge, then promote a
   complete hybrid member benchmark; and
3. retain the shared address owner through stage 6b before enabling the PIOP
   selector.

The genuine blockers to a complete-member claim are the host prefix/suffix and
transcript bridge, the real stage-owned resident address handoff, and a measured
access census for the frozen workload. Until those land, the device
measurements are component evidence only; complete speedups remain conditional
projections.
