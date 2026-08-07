# RAM RAF-evaluation successor design

This directory is an isolated design packet. It is not in the source registry,
backend, or benchmark harness. The Rust ABI, checked arithmetic, independent
oracles, and MSL entry points are implementation sketches for the next
integration pass; no file in this directory has been compiled or run.

The design conclusion is adaptive rather than GPU-only. The retained Fibonacci
trace has 190 accesses, so its best first-principles path is a compact host
pushforward. A one-command direct Metal scatter covers medium support. A
producer-bucketed Metal histogram covers high support without six scans of a
dense address plane. The current dense tiled kernel remains a checked fallback
until compact producer ownership is integrated and measured.

## Frozen boundary and denominator

The optimized-CPU control is
`benchmark-runs/metal-piop-eval/20260806-133709-697013` at revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`. The artifact SHA-256 is
`587e00a65bde003a7c3481f58b1ea047ed2c908b0e3d9808bbc7eec6f894b2df`.
Its five complete
`RamRafEvaluation` samples at `T = 2^26`, `K = 2^13`, and 16 Rayon threads are:

```text
76.520166, 76.746208, 73.944962, 73.501876, 74.870252 ms
```

The median is `74.870252 ms`. The hard 5x cap is `14.974050 ms` after integer
rounding. The 8x pursuit cap is `9.358781 ms`. The retained member spans include
preparation after `tau` is available, the direct handoff, the selected
pushforward, and all 13 host rounds. Generic stage-batch Fiat-Shamir lies
outside both member arms. The production PIOP charges that host work to both
backends. GPU-active time is diagnostic.

The primary PIOP evaluator starts with a transcript-independent backend witness
representation ready. The mandatory secondary metric adds that representation's
construction and upload. A compact projection may not move work across this
boundary without appearing in the secondary metric.

## Existing implementation audit

The optimized CPU path in
`src/optimized/ram_raf_evaluation.rs` constructs the full `T`-field equality
table, scans all `T` dense addresses into a `K`-field output, constructs the
affine `UnmapAddress` table, then runs the generic two-table address sumcheck.
This is topology-independent `O(T)` work even when almost every cycle is a
no-access row.

The current Metal path in `src/metal/solinas/ram_raf_evaluation` splits the
equality table into 32,768 low weights and 2,048 high weights. Six
`(outer, address-tile)` groups scan every dense address row. Each group
histograms one 1,376-address tile in 27,520 bytes of threadgroup memory, then
weights every nonzero subtotal once and scatters it into a global deferred sum.
It is robust and exact, but it always reads the 256 MiB address plane six times.

Retained random-topology evidence at revision `9de144572` reports:

| Quantity | Observation |
| --- | ---: |
| Warm GPU active | 6.3445 ms |
| Warm pushforward wall | 6.540292 ms |
| Specialized host tail, no FS | 0.177958 ms |
| Setup plus Criterion service | 6.996225 ms |
| Frozen-CPU ratio | 10.70x |
| Standalone dense-plane upload | 30.475833 ms |

The source is
`autoresearch/evidence/ram_raf_evaluation_log26_observed_9de144572.json`,
SHA-256 `9a26cec509ab11b8e7f33963ee157fc8cc4b00e153f5dab162fc17a6bb0ea6ff`.

The production Fibonacci diagnostic at the same revision reports only 190
accesses and 76 algebraically nonzero `(outer, address)` subtotals. It did not
record structural occupancy or nonempty descriptors. The current command was
active for 8.540917 ms but overlapped 667.808042 ms of other stage-2 work;
only 24.708 us of its ticket lifecycle remained at join. The enclosing join
span was 65.208 us. The retained diagnostic SHA-256 is
`9c38dc2b06261e70a47a1fb206bd6aa12c828f19b697f9c36420c70c1fbcd69d`.
This is useful scheduling evidence, but it means reducing
this command cannot improve that PIOP trace unless the command leaves the
overlap window or its representation cost is also reduced.

The opening paragraph of the current low-level `WIRING.md` says the proof
adapter is absent. The code and the retained `9de144572` evidence supersede
that status: the adapter, asynchronous submission, exact lockstep test, and
production diagnostic exist. The successor must preserve them.

## Exact relation and transcript contract

Only the default read-write split is admitted:

```text
phase1_num_rounds = log_T
phase3_cycle_rounds = 0
raf_evaluation_rounds = log_K = 13.
```

For cycle point `tau`, remapped address `a(j)`, and no-access sentinel rows
omitted from the compact stream:

```text
R(k) = sum_{j: a(j) = k} eq(tau, j)
U(k) = lowest_address + 8k
sum_k U(k) R(k) = ram_address_spartan.
```

The output opening is `RamRa([r_address || tau])`, where
`r_address = reverse(c_0..c_12)` for the low-to-high address challenges.
The derived output is

```text
UnmapAddress = Identity(r_address) * 8 + lowest_address.
```

`R` and `U` bind low-to-high. For each pair, the round polynomial is the sum of

```text
(u0 + x * (u1 - u0)) * (r0 + x * (r1 - r0)).
```

It is degree two. The host sends the same three coefficients as the optimized
CPU member, the stage batch absorbs them in the same order, and the stage batch
draws the challenge. The device never hashes, absorbs, or derives a challenge.
There is no relation-specific challenge and no protocol change.

The symbolic input expression includes a `2^phase3_cycle_rounds` coefficient.
The optimized CPU and current Metal specializations reject a nondefault phase
split. This successor does the same before transcript mutation; it must not
pretend that the coefficient is one for a general split.

## Equality factorization

For `I = 2^15`, `O = T / I`, `outer = j / I`, and `inner = j mod I`, split the
big-endian point before its last 15 coordinates:

```text
E_lo(inner) = eq(tau_suffix, inner)
E_hi(outer) = eq(tau_prefix, outer)
eq(tau, j) = E_hi(outer) * E_lo(inner).
```

At log 26, `E_lo` is 512 KiB and `E_hi` is 32 KiB. No `T`-field equality table
is allowed in a successor lane.

The current `EqPolynomial::evals` construction uses one field product for each
new pair, so the two host tables cost exactly

```text
(2^15 - 1) + (2^11 - 1) = 34,814 field products.
```

These are host setup products. The device-product tables below do not include
them.

## Producer representation

The common compact input is the eight-byte record already emerging in the RAM
kernel designs:

```text
(cycle: u32, address: u32)
```

There is one record for every non-sentinel address row and no record for any
other row. Records are strictly increasing by cycle. The producer validates
`cycle < T`, `address < K`, no duplicate cycle, no omitted non-sentinel row,
and no invented row while it still has the authoritative witness columns.
`RamRafAccessRecord` repeats the ABI inside this isolated packet; integration
must move the type to the common producer rather than retain sibling copies.

The direct device lane borrows this record stream unchanged. The tiny host lane
borrows the host copy.

The high-support bucket lane uses an additional member projection. For each
nonempty `(outer, tile)` bucket it stores one 16-byte descriptor:

```text
(first_record, record_count, outer, tile).
```

Each access in that bucket becomes one packed `u32`:

```text
bits  0..14  inner cycle index
bits 15..25  address offset inside the 1,376-entry tile
bits 26..31  zero
```

Descriptors are strictly ordered by `(outer, tile)` and cover the packed record
array exactly once. Empty buckets have no descriptor and launch no threadgroup.
This removes the current no-access floor, which still launches all 12,288
groups and clears every tile.

`RamRafTopology::from_bucket_projection` derives `A`, `Q`, `B`, and `S` from
the validated packed view, where `Q` is structural `(outer,address)` occupancy.
Challenge-dependent benchmark instrumentation supplies `Z`, the number of
subtotals that are actually nonzero after weighting; production admission uses
the checked `Z=Q` structural upper bound. Cost-model callers use these censuses
rather than supplying a useful-product count from metadata.

The projection is justified only for the bucket lane. It is created during the
same transcript-independent witness traversal or not at all. Constructing it
inside the timed member, deriving it with an uncharged GPU count/scan/scatter,
or calling it shared when no other consumer uses it is disallowed. The
secondary metric charges its bytes and producer wall.

The packet builder consumes the cycle-ordered stream once. It reuses six tile
vectors for one outer block, flushes them in tile order, and never keeps more
than `I=32,768` packed entries live outside the final projection. It therefore
does not allocate an `O*K` histogram on the host.

The common eight-byte stream is not always smaller than the dense four-byte
plane: it is twice as large at full support. The owner therefore chooses
representations from the real census. Compact records are a clear win for the
190-access target and for sparse RAM kernels, but they are not a blanket
replacement for every dense consumer.

## Adaptive execution lanes

### Tiny host-sparse lane

Build the two small equality tables, initialize a zeroed `K`-field `R`, and for
each compact record execute

```text
R[address] += E_lo[inner] * E_hi[outer].
```

Then use the affine host tail with the incremental-`U` loop described below.
At 190 records the working set is about
1.5 KiB of records, 544 KiB of equality weights, and 128 KiB of output. A GPU
submission cannot create useful occupancy at this support. The provisional
screen chooses host sparse for `A <= 2^15`; an alternating crossover benchmark
must replace that screen before production.

The successor affine continuation performs exactly 24,599 host field products:
`3 * (K - 1) + 2 * log_K`. It advances `u0` by the addition `u0 += 2*step`
instead of recomputing `step * (2*pair_index)` for every pair. The retained
177.958 us micro-observation used the current per-pair multiply and also
computed an 8,192-product input-claim control that the production adapter
receives from upstream, so it is conservative for the successor continuation.
The target host-sparse path has 59,603 full field products in total: 34,814 for
equality tables, 190 for the pushforward, and 24,599 for the continuation.

This lane also exposes a fairness issue: the same compact algorithm is a valid
optimized-CPU improvement. If it lands in the CPU backend, the frozen 74.87 ms
denominator is obsolete for sparse topologies and must be remeasured. A Metal
speedup claim may not hide that stronger CPU control. The portfolio may still
benefit from removing the current dense address upload in the secondary metric.

### Direct compact Metal lane

One thread reads one eight-byte record, loads `E_lo` and `E_hi`, performs one
full Solinas product, and adds the result to the address's five-word deferred
sum. The finalizer canonicalizes all 8,192 outputs. It is one command buffer,
one scatter dispatch, one finalizer dispatch, one wait, and a 128 KiB readback.

This lane is useful when there are enough records to fill the device but too
few to justify a second producer projection. It performs one global five-word
add per record, so it is not selected for a high-support hot address. The
provisional screen caps it at `2^20` records and requires `Q >= A/8`; measured
direct-versus-host and direct-versus-bucket crossovers replace both values.

### Producer-bucketed Metal lane

One 1,024-thread group owns one nonempty descriptor. It clears the same
1,376-by-five local accumulator as the retained kernel, reads each packed
record in its bucket exactly once, adds `E_lo(inner)` locally, multiplies each
nonzero address subtotal by `E_hi(outer)` once, then adds it to the global
five-word output. The finalizer is unchanged.

This preserves the retained kernel's tested local/global carry scheme and
atomic topology while removing five of six record scans. It also skips all
empty buckets. For dense random support its threadgroup atomic work is exactly
the retained kernel's work; only external input traffic falls.

### Retained dense and optimized CPU fallbacks

Use the current dense tiled kernel when its validated resident plane exists and
the compact producer or projection does not. Use optimized CPU for unsupported
geometry, capacity failure before any message is observed, nondefault phase
splits, or a producer provenance failure. Once a member message is absorbed,
any Metal error returns `SumcheckError::ComputeBackend`; it cannot restart on a
different state.

## ABI and command sequence

Concatenate Akita `fp128.metal` before `shader.metal`. Registration must reject
any offset other than `0xffff_a7f7`.

| Type | Bytes | Alignment |
| --- | ---: | ---: |
| `RamRafAccessRecord` | 8 | 4 |
| `RamRafBucketRecord` | 4 | 4 |
| `RamRafBucketDescriptor` | 16 | 4 |
| `RamRafDirectParams` | 32 | 4 |
| `RamRafBucketedParams` | 48 | 4 |
| `RamRafFinalizeParams` | 16 | 4 |
| `RamRafStatus` | 16 | 4 |

Entry points and buffers are:

```text
direct:
  0 access records, 1 E_lo, 2 E_hi, 3 deferred output,
  4 cleared status, 5 direct params

bucketed:
  0 packed bucket records, 1 descriptors, 2 E_lo, 3 E_hi,
  4 deferred output, 5 cleared status, 6 bucket params,
  threadgroup(0) = 27,520 bytes

finalize:
  0 deferred output, 1 canonical R, 2 status, 3 finalize params
```

The host clears the deferred and status buffers with a blit, dispatches the
selected producer and finalizer in the same command buffer, submits once, and
waits once at the first active RAM RAF round. No round allocates. Empty support
uses the host zero table and submits no command.

The blit, producer, and finalizer use three ordered encoders. In particular,
the producer compute encoder ends before the finalizer encoder begins; the
integration must not collapse them into one encoder without an equivalent
device-memory barrier.

The direct grid has `ceil(A / 256)` groups of 256 threads. The bucket grid has
exactly `B` groups of 1,024 threads. The finalizer has 32 groups of 256 threads
for `K=8,192`. Thus every nonempty device lane has two compute dispatches, two
in-command buffer clears, one submission, and one wait; it never dispatches a
group for a missing bucket.

The owner metadata carries `log_T`, `K`, access count, generation, device
registry ID, allocation IDs, byte lengths, content-validation result, and the
authoritative source identity. Bucket metadata additionally carries descriptor
count, source generation, and packed projection identity. The pending ticket
captures all borrowed and owned allocation IDs and waits on drop.

## Exact storage

At log 26 the sequence-owned device state is independent of support:

| Storage | Bytes |
| --- | ---: |
| `E_lo` | 524,288 |
| `E_hi` | 32,768 |
| Global five-word deferred output | 163,840 |
| Canonical `R` | 131,072 |
| Status | 16 |
| Sequence-owned total | 851,984 |

The direct lane borrows `8A` record bytes. The bucket lane keeps the common
record owner alive and additionally borrows `4A + 16B` bytes, where `A` is
access count and `B` is nonempty bucket count. Its dynamic threadgroup
allocation is 27,520 bytes.

Examples:

| Topology | Common records | Bucket projection | Total bucket residency |
| --- | ---: | ---: | ---: |
| Fibonacci, `A=190`, structural maxima `B=190` | 1,520 B | 3,800 B | 857,304 B |
| `A=22,000,000`, all 12,288 buckets | 176,000,000 B | 88,196,608 B | 265,048,592 B |
| Full support, all 12,288 buckets | 536,870,912 B | 268,632,064 B | 806,354,960 B |

The full-support figure demonstrates why producer admission must consider the
whole PIOP working set and why the dense plane remains available.

## Exact work and traffic

Let:

```text
A = access records
Q = structurally occupied (outer, address) subtotals
Z = algebraically nonzero subtotals after E_lo weighting
B = nonempty (outer, tile) buckets
S = sum of active address capacities for those buckets
C_g = global deferred additions whose low 128 bits wrap
C_t = threadgroup deferred additions whose low 128 bits wrap.
```

Every deferred add updates four words. It updates the fifth word only on a
wrap. Thus `0 <= Z <= Q <= A`, `0 <= C_g <= A` for direct,
`0 <= C_g <= Z` for bucketed, and `0 <= C_t <= A`. A retained input can
compute these carry counts; the screening
table below uses their conservative maxima. `RoofProjection::exact_external_bytes`
and `exact_threadgroup_internal_bytes` reject out-of-range carry censuses and
return the exact byte counts for a retained input.

The direct lane performs:

```text
field products                         A
global 32-bit atomic adds        4A + C_g
exact device bytes       589,856 + 8A + 32A + 8C_g
equality logical bytes               32A
```

The bucket lane performs:

```text
field products                         Z
threadgroup 32-bit atomic adds   4A + C_t
global 32-bit atomic adds        4Z + C_g
exact device bytes       589,856 + 4A + 16B + 32Z + 8C_g
equality logical bytes               16(A + Z)
threadgroup internal bytes       40S + 32A + 8C_t.
```

The fixed 589,856-byte service term is one deferred clear, one deferred read by
the finalizer, one canonical write, one canonical readback, one status clear,
and one status read. It does not include setup allocations or equality-table
construction. Each atomic RMW is priced as one four-byte read and one
four-byte write.

The fifth word cannot overflow under the admitted ABI. A local cell receives
at most `I=32,768` additions, a bucketed global cell at most `O=2,048`, and a
direct global cell at most `T<=u32::MAX`; the correction
`overflow * 0xffffa7f7` also fits in 64 bits.

The retained measured roofs are 451.701710520 GB/s copy bandwidth and 18.1
G full-field products/s. Their regime-flip intensity is

```text
18.1 / 451.701710520 = 0.04007 products/byte.
```

Both proposed device lanes remain below this external traffic threshold. The
threadgroup and device atomic issue ceilings are separate structural bounds;
Metal counters have not measured them, so the arithmetic below cannot promote
a kernel by itself.

For the Fibonacci census, the direct zero-carry external floor is 0.001323 ms
and its all-carry floor is 0.001327 ms. Under the conservative structural
maxima `B=Q=190` and observed `Z=76`, the bucket equivalents are 0.001320 ms
and 0.001321 ms. A measured input uses exact `B`, `Z`, and `C_g`;
neither endpoint is silently presented as the observed traffic.

### Log-26 controls

The 22m uniform control uses
`Q = round(O*K*(1-(1-1/K)^(A/O))) = 12,256,639` and screens with `Z=Q`.
A real run replaces the occupancy expectation and cancellation-free assumption
with projection- and kernel-derived censuses.

| Lane/topology | Products | Max device bytes | Compute floor | Cached max-traffic floor | Conservative 80% screen | Known no-FS terms | Known-term CPU ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct, Fibonacci `A=190` | 190 | 598,976 | 0.000011 ms | 0.001327 ms | 0.001659 ms | 0.320617 ms | 233.52x |
| Bucket, `A=22m`, expected `Q=Z=12,256,639` | 12,256,639 | 579,052,024 | 0.677163 ms | 1.281935 ms | 1.602419 ms | 1.921377 ms | 38.97x |
| Direct, full support | 67,108,864 | 3,221,815,328 | 3.707673 ms | 7.132617 ms | 8.915772 ms | 9.234730 ms | 8.11x |
| Bucket, retained dense-random `Z=16,470,998` | 16,470,998 | 928,061,840 | 0.910000 ms | 2.054591 ms | 2.568239 ms | 2.887197 ms | 25.93x |

`Known no-FS terms` adds the 141 us retained command-wall control and the
177.958 us observed affine tail to the cached 80%-roof bar. It excludes
atomic issue, equality construction, allocations, host Fiat-Shamir, and
producer wall. It is a ceiling-screening number, not a latency prediction.
The final column divides the frozen 74.870252 ms CPU median by that incomplete
known-term sum; it is an opportunity screen, not a speedup claim.

If every equality load missed cache, the 22m bucket traffic bar becomes
3.119200 ms and the full-support bucket bar becomes 6.268904 ms. `E_lo` and
`E_hi` have only a 544 KiB unique footprint, but promotion still requires
counters or a sensitivity control rather than assuming cache residency.

The direct full-support path has only 0.124 ms of margin under the 8x cap
before setup, hashing, and atomic service. It is therefore rejected as the
dense pursuit path even though it clears 5x analytically. The bucket lane has
substantial 8x margin.

With every fifth-word carry, full support has 3,355,443,200 bytes of
threadgroup-internal traffic, the same local accumulation work as the retained
dense kernel. The corresponding 22m maximum is 1,551,088,640 bytes. Exact
values subtract `8(A-C_t)`. These terms need a measured threadgroup-atomic
issue rate; copy bandwidth does not price them.

## Occupancy and latency

The direct sketch uses 256 threads, eight SIMD32 groups, and no dynamic
threadgroup memory. Its full multiply structurally carries two four-limb
inputs, an eight-limb product, a four-limb reduction, and carry temporaries.
The compiler's register count and spills are unknown. With only 190 records it
launches one group for the whole device, so no register tuning can make it an
occupancy kernel; command latency and underfill bind. This is why the host lane
screens first.

The bucket sketch uses 1,024 threads, 32 SIMD32 groups, and 27,520 bytes of
dynamic threadgroup memory. Against the observed 32 KiB limit it leaves 5,248
bytes for pipeline-static memory and admits at most one such group per core.
The retained compiled pipeline reports execution width 32, maximum 1,024
threads, and zero static threadgroup bytes. Those facts justify the launch as
a candidate, not an occupancy claim for this uncompiled successor.

Promotion needs an Instruments or equivalent capture with registers, spills,
resident SIMD groups, inactive lanes, threadgroup-memory residency, external
bytes, and atomic stalls. Reject a compiled bucket pipeline that cannot launch
1,024 threads, uses more than 5,248 static threadgroup bytes, spills more than
10% of modeled external traffic, or leaves a persistent tile underfilled on a
high-support control.

## Pre-registered falsification bars

The fixed complete-member gates are:

- hard 5x: at most `14.974050 ms`;
- 8x pursuit: at most `9.358781 ms`;
- exact round polynomials, challenges, final `RamRa`, derived
  `UnmapAddress`, and proof verification.

The successor-specific bars are:

- target sparse direct service, including finalization/readback: at most
  `0.50 ms`; otherwise retain host sparse;
- target sparse complete no-FS member: at most `0.75 ms`;
- 22m bucket GPU active: at most `3.0 ms`;
- dense-random bucket GPU active: at most `4.0 ms` and statistically below
  the retained 6.3445 ms control;
- bucket projection producer plus upload is reported separately and cannot
  exceed the improvement it enables on the producer-inclusive metric; and
- any newly measured atomic ceiling replaces, rather than supplements, the
  provisional bars when it raises the bottomed-out floor.

The dense bucket bars intentionally allow more than the 2.57 ms external
80%-roof number because threadgroup atomics are an unmeasured structural term.
Missing that bar starts an atomic/layout redesign; clearing the easy 5x member
cap is not enough.

## Host affine tail

The device returns only `R`. It never materializes `U`. The host retains

```text
U(y) = base + step * y.
```

For `R(2y)=r0`, `R(2y+1)=r1`, `dr=r1-r0`, and
`u0=base+2*step*y`:

```text
q(0)       = sum u0 * r0
q(1)       = previous_claim - q(0)
leading(q) = step * sum dr
q(2)       = 2q(1) - q(0) + 2*leading(q).
```

After challenge `c`:

```text
R(y) = r0 + c * dr
base = base + step * c
step = 2 * step.
```

Within a round, start `u0=base` and add `2*step` after each pair. That makes
the message cost two full products per pair plus one for `leading`; binding
costs one per pair plus one for `base`. It also avoids materializing `U`.

All 13 rounds stay on the host. Moving them to Metal adds 13 dependent waits
to at most 8,192 fields and is rejected.

## Claim-to-code map

| Claim or invariant | Upstream authority | Successor unit |
| --- | --- | --- |
| Input and output expressions, degree two | `jolt-claims/.../ram/raf_evaluation.rs` | `WIRING.md`, affine oracle |
| Output point and `UnmapAddress` orientation | `jolt-verifier/.../ram_raf_evaluation.rs` | `prove_affine_address_rounds` |
| Dense cycle fold | `reference/ram_raf_evaluation.rs` | `dense_pushforward_oracle` |
| Current optimized sparse-equivalent fold | `optimized/ram_raf_evaluation.rs`, `optimized/ram_trace.rs` | `compact_pushforward_oracle` |
| Compact record equivalence and ordering | common RAM producer, not yet integrated | `validate_access_records` |
| Bucket projection equivalence | no upstream implementation yet | `build_bucket_projection`, `validate_bucket_projection`, bucket oracle |
| Direct device scatter | design in this packet | `solinas_ram_raf_successor_direct` |
| Bucketed local histogram | retained Metal algorithm with compact input | `solinas_ram_raf_successor_bucketed` |
| Deferred canonicalization | retained Metal five-word scheme | successor finalizer |
| Batch-owned transcript | stage-2 generated driver | no shader entry point |

The dense oracle evaluates `eq(tau, j)` coordinate by coordinate and scans the
authoritative plane. The compact oracle scans ordered records. The bucket
oracle reconstructs global cycle and address indices from the packed ABI and
again evaluates unsplit equality coordinates. None calls the shader's split
factorization or local histogram helper.

## Required parity and performance experiments

Before registration, root should run these in serial:

1. Compile the isolated source with Akita `fp128.metal`; record pipeline limits
   and reject ABI/source mismatch.
2. Compare dense, compact, bucket, direct-Metal, and bucket-Metal pushforwards
   on no access, one access, one hot address, random addresses, boundary
   addresses, inner-block boundaries, and maximum-carry fields.
3. Run lockstep member parity against optimized CPU for every message,
   challenge, output claim, derived value, and proof.
4. Benchmark host sparse and direct Metal across access counts around the
   provisional `2^15` cutoff. Alternate order in the same binary.
5. Benchmark direct, bucket, and retained dense around `2^20` and at 22m/full
   support. Include one-hot and random address entropy.
6. Run the target Fibonacci member and PIOP with the command submitted at the
   current stage-2 seam. Report marginal join wall as well as service time.
7. Report backend representation construction/upload in the secondary metric,
   then run the five alternating log-26 pairs and log-27 transfer check.

The checked Rust tests in this directory cover ABI sizes, producer ordering,
projection reconstruction, dense/compact/bucket oracle equality, the affine
terminal relation, exact target arithmetic, and provisional lane selection.
They have not been run because this packet is deliberately unregistered.

## Rejected alternatives

- A full `T`-field equality table writes and rereads 1 GiB at log 26.
- Six scans of the eight-byte compact record stream still read `48A` bytes and
  leave most of the current dense architecture intact.
- A full `O * K` subtotal table owns 256 MiB and needs another reduction.
- Direct global scatter at high hot support performs `5A` contended device
  atomics; the bucket path reduces this to `5U` after local aggregation.
- Building the bucket projection between stage-2 transcript rounds moves
  producer work into the critical path and breaks the one-command overlap.
- Running address rounds on the device adds 13 waits for a 128 KiB table.
- Treating the 190-access trace as a dense-GPU occupancy problem optimizes an
  8.54 ms command already hidden from the PIOP while ignoring a tiny host path.

## Ambiguity and blockers

1. The compact common producer is designed in sibling packets but is not a
   registered, provenance-carrying backend representation.
2. The bucket projection's real producer wall and peak host allocation are
   unmeasured. They must be charged in the secondary metric.
3. The optimized CPU compact path has not been benchmarked. Until it is, the
   frozen denominator is valid only as a historical control, not a claim about
   the best sparse CPU algorithm.
4. Threadgroup and device atomic issue rates are absent from the retained roof
   controls. They block a bottomed-out latency claim.
5. The successor shaders have not been compiled; register count, spills,
   occupancy, and MSL ABI compatibility are unknown.
6. Host Fiat-Shamir cost for this member has not been isolated on the exact
   complete-member boundary.
7. The 190-access count comes from one retained Fibonacci diagnostic. Other
   workloads need their own census and may select a different lane.
8. Nondefault read-write phase splits remain CPU-only.

Until these are resolved, the production backend should retain its current
exact Metal path and optimized-CPU fallbacks. The successor packet is ready
for a root-controlled compile/parity pass, not promotion.
