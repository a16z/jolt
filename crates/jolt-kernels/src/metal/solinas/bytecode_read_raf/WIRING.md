# Bytecode read/RAF address successor packet

This directory is an isolated successor for the packed Akita
`BytecodeReadRafAddressPhase`. It does not replace the existing stage-6b cycle
kernel. Nothing here is registered yet.

## Evidence status

The optimized-CPU denominator is durable production evidence. The test-utils
evaluator now executes both a prebuilt long-worker slice and the complete
CSR-to-indirect-worker device path; neither is registered with the production
backend. `screening_evidence.json` records exact log-26 screens for one and 26
active addresses per outer block. The host-shell and complete-member numbers
remain unmeasured until the precomputed pushforwards enter the real relation.

The log-26 CPU evidence is
`benchmark-runs/metal-piop-eval/20260806-133709-697013`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, on an M4 Max with 16 Rayon
threads. The five complete address-member samples are:

```text
172.796544, 198.165708, 181.211502, 190.915958, 198.945292 ms
```

The median is 190,915,958 ns. The strict integer caps are 38,183,191 ns for
5x and 23,864,494 ns for 8x. The rounded values sometimes shown in reports are
one nanosecond higher and are not acceptance limits. Median CPU `prepare` was
182.930333 ms. The 7,918,251-ns retained component is the total of the 13
`prove_round` calls only. It is not a host residual: it excludes
`finish_rounds`, output construction, Fiat-Shamir, equality-table construction
and transfer, the six value tables, `Int`, entry tables, shell construction,
command latency, readback, and status validation. There is no measured
complete-host-shell value to add to a device roof. Component medians are not
additive evidence, so the complete member remains the acceptance gate.

The executable CSR path is exact at log 15 against the independent direct
oracle with 26 active addresses. At log 26, its exact one-address fixture has
11.4365 ms CSR-only, 20.283833 ms GPU-active, and 20.941375 ms complete-slice
medians. The 26-address fixture has 53,248 long runs, a maximum run of 1,261,
11.419042 ms CSR-only, 28.561625 ms GPU-active, and 29.109917 ms
complete-slice medians. It leaves 9.073274 ms under the 5x cap but already
exceeds the 8x cap by 5.245423 ms before host work. Adding the retained
7.918251-ms `prove_round` component would leave 1.155023 ms under 5x, but that
sum is only a scheduling screen for the reasons above and is not a speedup
claim. The next architecture change is to reuse producer-side address counts,
removing one full 2.68-GB row scan from CSR construction.

The existing cycle member has a 1,004.692916-ms CPU median and a
160.876418-ms Metal median. Their ratio is 6.245x; this is not the median of
the paired speedups. The paired-run median of CPU address
plus cycle is 1,203.638208 ms. Its strict 8x cap is 150.454776 ms, already
below the current cycle Metal median. Therefore an 8x combined
BytecodeReadRaf result also requires a cycle improvement. An address result at
its 5x cap projects to 199.059609 ms for the two members, about 6.047x.

## Frozen relation

The frozen evaluator uses `log_T = 26`. The shader shape accepts larger
power-of-two traces with the same `log_K = 13` and `2^15` inner split. The
protocol has five base stages, four fused-increment stages, and 13 low-to-high
address rounds. Let `r_s` be a stage cycle point, `pc(j)` the pushed program
counter, and `inc(j)` the signed fused increment:

```text
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j)                   s < 5
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j) * inc(j)          s >= 5
```

An absent mapped PC pushes to address zero. This matches the optimized address
phase and intentionally differs from the stage-6b cold-cycle convention.

The nine value sources, in stage order, are:

```text
T0, T1, T2, T3, T4, T5, T5, 1 - T5, 1 - T5
```

For batching challenge `gamma`, stage weights are `gamma^0` through
`gamma^8`. The within-stage RAF weights are `gamma^9` on stage 0,
`gamma^8` on stage 2, and zero otherwise. Because the outer stage weights are
also applied, the two resulting RAF terms have overall weights `gamma^9` and
`gamma^10`. The entry term has weight `gamma^11`.

The address summand is

```text
sum_s gamma^s F_s(k) * (Val_s(k) + raf_s * Int(k))
    + gamma^11 * EntryTrace(k) * EntryExpected(k).
```

Its round degree is two. For each round the host computes only `q(0)` and
`q(2)`, reconstructs `q(1) = previous_claim - q(0)`, and builds the hinted
quadratic. `prove_round` binds the previous challenge before computing the
next message. `finish_rounds` binds the thirteenth challenge. Fiat-Shamir,
polynomial absorption, and challenge derivation stay on the host.

The verifier opens every address result at the reverse of encounter order:

```text
r_address = sumcheck_challenges.iter().rev()
```

After all binds, the output is

```text
intermediate = gamma^11 * EntryTrace(r) * EntryExpected(r)
             + sum_s gamma^s * F_s(r)
                 * (Val_s(r) + raf_s * Int(r)).
```

Committed-program mode also returns the six raw bound `T0..T5` values in
table order. Full-program mode returns no `val_stages`. Complemented stages
must not replace the raw `T5` output claim.

The packet contains an independent canonical 128-by-64 arithmetic model,
equality-table oracle, direct pushforward oracle, CSR-form oracle,
round-message oracle, final-output oracle, and canonical FNV-1a checksum.
These do not call the optimized CPU implementation.

The source anchors for integration are
`optimized/bytecode_read_raf.rs` for `OptimizedBytecodeReadRafAddress` and its
private `AddressKernel`,
`jolt-verifier/src/stages/stage6a/bytecode_read_raf.rs` for reversed output
points, `metal/solinas/booleanity/mod.rs` for the row ABI, and
`metal/instruction_read_raf.rs` for stage-5 production and admission.

## Producer-owned row ABI and lifetime

The input is the existing 40-byte `BooleanityRows` allocation produced by the
stage-5 Metal instruction read/RAF kernel. Its five `u64` words are:

```text
0 lookup_lo
1 lookup_hi
2 ram_address_plus_one
3 fused_inc_magnitude
4 packed_pc_and_flags
```

The low 56 bits of word 4 encode `mapped_pc + 1`; zero means absent. The real
stage-5 producer also stores `table_index + 1` in bits 56--61 and the RAF flag
in bit 62. Bit 63 is the fused-increment sign. The bytecode address kernel
masks the low 56 bits for PC and ignores bits 56--62 without clearing them.
The magnitude is exact through `u64::MAX`. The Rust tests pass a row through
`BooleanityRow::from_words` with all producer metadata classes populated.

The stage-6a adapter must clone or borrow `session.state::<BooleanityRows>()`,
verify length and device registry identity, and use its buffer directly. It
must not derive CPU `InstructionCycleRow`s, pack a new row vector, allocate a
replacement buffer, or upload rows. It must leave the session handle alive
for the Booleanity address member, stage 6b, and any admitted stage-7 consumer.

The durable run records the same allocation identity across the stage-5,
stage-6a, and stage-6b Booleanity row lifecycle, with zero allocations and
uploads in the later consumers. The existing bytecode cycle source also clones
the same session handle, but its retained attribution row does not independently
prove that identity. Bytecode address integration must add its own identity
field and extend stage 5's `resident_rows_requested` admission condition. Zero
marginal row cost is valid only when another admitted consumer already caused
stage 5 to produce the allocation. If this successor is the sole cause, its
comparison must charge the stage-5 allocation and 2,684,354,560-byte log-26
upload.

There is no bytecode-address-specific CPU row-source span in the retained
artifact. The raw CPU member denominator includes whatever its `prepare`
performed, while the Metal design assumes producer reuse. Microbenchmarks
must report this asymmetry. Promotion requires alternating whole-PIOP runs so
shared production is charged once on both routes. Allocation identity and
producer/consumer upload counters are evaluator fields, not debug-only logs.

## Device algorithm

Split every big-endian cycle point at `I = 2^15`:

```text
O = T / I
j = (j_hi << 15) | j_lo
eq(r_s, j) = E_hi_s[j_hi] * E_lo_s[j_lo].
```

A direct row kernel would perform nine full equality products per trace row
and contend on address outputs. The proposed block-local CSR converts those
products to nine full products per nonempty `(outer, address)` run.

One 1,024-thread producer group owns one 32,768-row outer block and exactly
32 KiB of dynamic threadgroup memory:

1. Clear 8,192 atomic count bins.
2. Scan the block and count pushed PCs.
3. Retain eight counts per thread and prefix-scan the 1,024 thread totals.
4. Reserve short and long run ranges with one global atomic per class.
5. Emit one 16-byte descriptor for each nonempty address and replace each bin
   with its scatter cursor.
6. Rescan the block and scatter one `u32` row index per occurrence.

Runs of at most 128 occurrences grow from the front of one arena. Longer runs
grow from the back. The total run count never exceeds the arena capacity, so
the regions cannot overlap. One thread processes a short run; one SIMDgroup
processes a long run. Each worker retains nine field accumulators. The four
fused stages use exact signed 128-by-64 multiplication, then all nine stage
sums receive one full-width multiply by `E_hi`. Results atomically accumulate
into five `u32` words per `(stage, address)` and a final kernel reduces them to
canonical fields.

The full-width fused path is a parity and performance control. Exact-u64 is
selectable only after its isolated control clears 26.272 Gproduct/s and it
wins the complete member.

### First executable worker slice

`BytecodeReadRafLongWorkerSlicePlan` defines a harness-only direct dispatch.
At log 26 it uses one 32,768-row run per outer block, so
`U = U_long = O = 2,048`. The fixture requires every row in one outer block to
push to the same address. Its occurrence array is the identity permutation,
and descriptor `i` is written to `run_capacity - 1 - i`, matching the
long-worker tail arena. The ceiling fixture assigns a distinct address to each
outer block; an all-to-one-address control measures output contention.

The caller borrows the stage-5 `BooleanityRows` allocation and checks both its
length and device registry identity. It owns occurrences, the full run arena,
`E_lo`, `E_hi`, deferred sums, and canonical output. All allocations and input
uploads happen before the active worker measurement and stay resident between
samples. The direct counter buffer sets only `long_runs`; its CSR completion
and occurrence fields stay zero, so it cannot pass production status
validation.

The long worker binds shared rows, occurrences, the run arena, direct counters,
`E_lo`, `E_hi`, deferred sums, and pushforward params at slots 0 through 7 in
that order. `finalize` binds deferred sums, canonical output, and the same
params at slots 0 through 2.

The first command sequence is:

1. Clear the five-word deferred output.
2. Direct-dispatch `long_runs_u64` or `long_runs_full` with the plan's grid and
   exactly the recorded threadgroup width.
3. End the worker encoder before dispatching `finalize`.
4. Complete the command buffer, then compare all 73,728 canonical fields with
   `direct_pushforward_oracle` and its canonical checksum.

The two-outer host fixture freezes checksum `0x7a78b91f9539b12c`. It covers an
absent PC, address 8,191, full-limb equality values, both signs of `u64::MAX`,
zero, and small increments.

At minimum topology, the worker has 268,435,456 useful signed products and
18,432 useful full products. The packet charges 2,953,560,064 logical run
bytes. Its conservative issued-product screen is 10.260 ms, compared with a
6.539-ms run-traffic screen. This slice isolates arithmetic, SIMD reduction,
and output atomics. It excludes CSR, equality-table construction and upload,
host sumcheck rounds, readback attribution, and producer cost, so it cannot
support a complete-member speedup claim. The next controls are an all-short
topology and then the registered CSR-to-indirect-worker path.

## Shader ABI and command dependencies

Define `SOLINAS_OFFSET` as `0xffff_a7f7`, then concatenate sources in this
order:

1. `fp128.metal`
2. `simd_reduce.metal`
3. `bytecode_read_raf/shader.metal`

Register all seven entry points named by the constants in `mod.rs`. The CSR
buffers are resident rows, occurrences, bidirectional run arena, the 32-byte
`BytecodeReadRafStatus`, `BytecodeReadRafCsrParams`, and the 80-byte
`BytecodeReadRafDiagnostics`, plus 8,192 dynamic threadgroup `atomic_uint`s.
Run buffers are rows, occurrences, runs, status, stage-major
`E_lo`, stage-major `E_hi`, five-word deferred output, and
`BytecodeReadRafPushforwardParams`.

The status fields are short runs, long runs, invalid rows or group invariants,
completed outer groups, and accounted occurrence rows. The diagnostics contain
short and long occurrence totals, maximum run length, and 16 floor-log2
run-length buckets. Rust freezes the size, alignment, and every field offset of
all shader-visible structs. The diagnostic buffer is required for benchmark
admission and topology-aware projection; it is not used to construct a proof
claim.

The encoded short and long threadgroup widths must equal the corresponding
parameter fields used by `write_dispatch`. Initial values are 256 and 256;
both are nonzero multiples of SIMD width 32 and at most the pipeline limit.

One invocation has this required order:

1. Blit-clear the status, diagnostics, deferred output, and indirect grids.
2. End the blit encoder before CSR reads those clears.
3. Dispatch one 1,024-thread CSR group per outer block with exactly 32 KiB of
   dynamic threadgroup memory.
4. End that compute encoder, then encode `write_dispatch`; an encoder boundary
   or explicit resource barrier must make status and run descriptors visible.
5. End that encoder before the two indirect run dispatches.
6. End the run encoder before `finalize` reads deferred sums.
7. Complete the command buffer before reading status, diagnostics, or canonical
outputs.

`write_dispatch` emits zero indirect grids unless `invalid_rows == 0`, every
outer group completed, accounted occurrences equal `rows`, and the run total
is within `[outer_length, run_capacity]`. The host repeats those checks after
completion. It also validates that diagnostic short plus long occurrences
equal `rows`, histogram counts equal total runs, the maximum run is valid, and
the short/long partition agrees with the threshold. Every short run contributes
between one and `threshold` occurrences; every long run contributes between
`threshold + 1` and `inner_length`, and the reported maximum belongs to the
reported class. No output claim is constructed after a failed check. Counter
overflow, incomplete dispatch, and invalid-PC cases fail closed.

Buffers are allocated once in `prepare`; no sumcheck round allocates device
storage. The specialization requires `maxThreadgroupMemoryLength >= 32 KiB`,
1,024 admitted threads, no unexpected static threadgroup storage, each buffer
within `maxBufferLength`, and total storage within the recommended working
set. Failed admission falls back to the complete optimized CPU phase. It does
not tile the address domain or partially dispatch.

## Log-26 and log-28 capacity

At log 26, `O = 2,048` and run capacity is
`min(T, O*K) = 16,777,216`:

| Storage | Bytes |
| --- | ---: |
| Shared resident rows | 2,684,354,560 |
| Occurrence indices | 268,435,456 |
| Maximum run arena | 268,435,456 |
| Nine `E_lo` tables | 4,718,592 |
| Nine `E_hi` tables | 294,912 |
| Deferred sums | 1,474,560 |
| Canonical outputs | 1,179,648 |
| Status, diagnostics, indirect grids | 144 |
| Successor-owned total | 544,538,768 |
| Total with shared rows | 3,228,893,328 |

At log 28, occurrence indices and the run arena are 1,073,741,824 bytes each,
successor-owned storage is 2,156,036,240 bytes, shared rows are
10,737,418,240 bytes, and the aggregate is 12,893,454,480 bytes.

## Checked roofline

Let `N = T`, `U` be the number of nonempty `(outer, address)` runs, and
`U_long` the number longer than the selected threshold. A valid log-26
topology has `I = 32,768`, `O = 2,048`, threshold `t = 128`,
`ceil(I / t) = 256`, and `floor(I / (t + 1)) = 254`. It satisfies

```text
2,048 <= U <= 16,777,216.
0 <= U_long <= U
U + 128 * U_long <= N                         threshold = 128
N <= 128 * (U - U_long) + 32,768 * U_long
U_long <= 2,048 * 254 = 520,192
U >= U_long + 256 * (2,048 - min(U_long, 2,048)).
```

The last inequality accounts for outer blocks with no long run: each needs at
least 256 short runs to cover 32,768 rows. The checked work model requires both
`U` and `U_long` and rejects violations of these aggregate and per-outer
limits. Useful work, issued lane work, and traffic are:

```text
P_fused = 4N signed 128-by-64 products
P_outer = 9U full-width products
P_fused_issued <= 4N + 124U_long lane products
P_outer_issued = 9U + 279U_long lane products
A_local = 9N field additions
A_local_issued <= 9N + 279U_long lane additions
A_long_reduce_useful = 279 * U_long field additions
A_long_reduce_issued = 1,440 * U_long lane additions
X_csr_atomic = 2N + 4O + 3U atomic operations
X_output = U nine-accumulator output updates
B_csr_logical = 84N + 40U + 32O
B_csr_cached_second_pass = 44N + 40U + 32O
B_runs_cache_optimistic = 44N + 376U
B_E_lo_shader_logical = 144N.
```

`B_csr_logical` counts two 40-byte row scans, one occurrence write, one
descriptor per run, three diagnostic atomic read-modify-writes per run, and
four status atomic read-modify-writes per outer block. The cached alternative
is reported separately and is not used for acceptance until hardware counters
support it. `B_runs` counts
occurrence and gathered-row reads, descriptors, and the maximum five-word
atomic traffic for nine outputs. The nine `E_lo` loads per row are 9 GiB at
log 26 but touch only a 4.5-MiB unique table set. `E_hi` adds `144U`
shader-logical bytes from a 288-KiB unique set at log 26. Treating both
equality-table sets as cache resident is a falsifiable assumption.

One `X_output` update means all nine field accumulators for one run, including
their four mandatory and possible fifth carry atomics. Its matched control is
reported in run updates per second, not raw atomic instructions per second.

The long loop rounds instruction issue to a SIMD width. Its nine `E_hi`
products execute with only lane zero active, and its nine SIMD reductions
issue five shuffle-add stages across all 32 lanes. The projection charges all
of this issued work, not only useful products.

The 451,701,710,520-B/s copy anchor comes from the log-28 active copy
measurement documented in `src/metal/SPEC.md`: 8 GiB of logical copy traffic
in 19.017 ms on the M4 Max. This packet does not retain a separate raw copy
artifact. The 18.10-Gproduct/s anchor is copied from the six-accumulator
full-field control documented in
`solinas/registers_claim_reduction/WIRING.md`; it is a register-pressure
screen, not a matched result for this nine-accumulator shader. The
26.272-Gproduct/s signed-u64 value is an unmeasured promotion floor inherited
from the half-width probe, not an observed rate.

The model refuses to produce a projection until the selected product path and
the same binary/device record supply all of these matched rates:

- full-field products in the nine-accumulator shader;
- signed-u64 products when that path is selected;
- field additions in the nine-accumulator loop;
- SIMD reduction-lane additions;
- CSR atomics with the selected topology builder;
- nine-accumulator output updates under the observed address contention.

At the attainable maximum threshold-128 long-run count,
`U = U_long = 2,048 * floor(32,768 / 129) = 520,192`. The charged
issued-product upper bounds are 332,939,264 fused and 149,815,296 outer
products. Using the unmeasured signed-u64 floor plus the retained full-product
screen gives a product-only floor of 20,949,869 ns. Padding contributes
2,455,231 ns of signed-u64 work and 8,018,430 ns of full-width work, or
13,092,074 ns at the 80% line. This topology also issues 749,113,344 local-add
lanes, 749,076,480 reduction-add lanes, 135,786,496 CSR atomics, and 520,192
nine-accumulator updates. Its conservative CSR and run traffic are
5,658,017,792 and 3,148,382,208 bytes. The product-and-traffic-only 80%-roof
cap is 41,844,844 ns, already above the complete 5x cap before host work. The
matched addition and atomic rates are currently missing, so this is not a
complete feasibility result.

At the minimum run count, `U = U_long = 2,048`; one run covers each outer
block. Conservative CSR traffic is 5,637,292,032 bytes and run traffic is
2,953,560,064 bytes. The generic issued upper bounds are 268,689,408 fused
and 589,824 outer products, giving a 10,259,803-ns product-only screen. CSR
and run traffic screen at 12,480,122 and 6,538,741 ns. Selecting the larger
run screen and applying the 80% phase caps yields 28,424,907 ns before command
wall, host work, and every missing matched control. The former
product-and-traffic-only calculations that yielded `U` cutoffs of 5,546,617,
12,961,199, and 2,843,481 ignored `U_long`, local and reduction adds, atomics,
and the unmeasured complete host boundary. They are optimistic necessary
screens only and cannot admit a topology. There is no current checked 5x or 8x
feasibility claim. Promotion depends on the complete-member benchmark, not an
analytical sum using the 7,918,251-ns `prove_round` total.

## Occupancy floor

The CSR producer requests 1,024 threads and exactly 32 KiB of dynamic
threadgroup memory. One group per core is the expected memory limit, exposing
32 SIMDgroups if registers do not reduce residency. Each thread holds eight
counts and prefixes plus scan state.

A run worker structurally needs nine field accumulators, 36 scalar words,
plus row data, loop state, an equality value, and a six-word u64 or eight-word
full product. About 50 scalar words before compiler temporaries is a source
floor, not occupancy evidence. The `local_counts[8]` and `sums[9]` arrays are
specific spill risks.

Promotion needs a capture for every selected entry point: compiler register
count, spills/local-memory traffic, resident SIMDgroups, execution width 32,
and the actual limiter. The CSR kernel must admit one 1,024-thread group
without spills. Run kernels must sustain enough active SIMDgroups to reach
the retained arithmetic controls.

## Hybrid boundary and promotion gates

Metal produces all nine pushforwards once. The optimized host shell performs
all 13 address rounds and Fiat-Shamir. A mid-round device handoff has no useful
work because preparation leaves only nine 8,192-element tables. The initial
trace cutoff is `2^20`; paired `2^19`, `2^20`, and `2^21` runs select the real
crossover.

Promotion order is fixed:

1. Compile the concatenated source and record shader hash, binary hash,
   compiler options, pipeline limits, register counts, and static/dynamic
   threadgroup memory.
2. Run checked ABI, model, long-worker plan, and independent-oracle tests.
3. Run the prebuilt minimum-topology long-worker slice with both product paths.
   Prove exact parity for absent PC, PC zero, maximum PC, zero and both signs
   of maximum increment. Record canonical checksums and active worker timing.
4. Compare direct and CSR oracle topology, then require GPU parity for all
   nine pushforwards, all 13 round polynomials and challenges,
   `intermediate`, and committed `val_stages`. Finish with proof verification.
5. Capture `U`, short/long counts, short/long occurrence counts, maximum run,
   and a run-length histogram. Reject any nonzero invalid counter.
6. Measure the matched signed-u64, field-add, SIMD-reduction, CSR-atomic, and
   nine-accumulator controls in the same binary/device record. Only then build
   the topology-aware 80%-of-roof caps. Use hardware counters to accept or
   reject the cached-row and cached-equality traffic models; the complete 5x
   cap remains fixed.
7. Freeze threshold 128 unless 64 or 256 wins by at least 3% with relative MAD
   at most 3%. Compare full and exact-u64 paths; retain neither based on a
   single sample.
8. Run alternating complete CPU/Metal log-26 members with equal transcript and
   synchronization charges. Require parity, median at most 38,183,191 ns,
   relative MAD at most 3%, no first-sample winner, and no capacity fallback.
9. Run alternating whole-PIOP validation with row-production ownership
   attributed once. Only then report end-to-end speedup.
10. Recheck log-28 admission and scaling after log-26 promotion.

If measured evidence shows a materially better ceiling than 5x, continue
toward it. The current architecture does not justify a standalone 8x claim;
upstream CSR reuse is the first design that can change that conclusion.

## Integration work left to root

- Register the seven entry points in the production backend instead of the
  test-utils source constructor.
- Add bytecode address config and stage-5 resident-row admission.
- Carry producer-side address counts into this successor and remove the first
  CSR row scan; retain the two-pass builder as the exact fallback/control.
- Expose a narrow optimized host-shell constructor that accepts precomputed
  pushforwards without duplicating relation logic.
- Add the all-short topology control, Criterion microbenchmarks, paired
  complete-member evaluator rows, GPU parity tests, and proof integration.

The current runtime remains test-utils-only and does not change backend
selection or the protocol.
