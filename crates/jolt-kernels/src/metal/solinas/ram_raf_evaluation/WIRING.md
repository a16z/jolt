# RAM RAF-evaluation Metal contract

This directory contains the low-level Metal implementation for
`RamRafEvaluation`. The source registry, resident address-plane API,
one-command sequence, independent parity test, and Criterion microbenchmark are
wired. The proof-stage adapter and shared stage-2 address producer are not.

## Frozen evaluator and hard target

The log-26 optimized-CPU denominator comes from
`benchmark-runs/metal-piop-eval/20260806-133709-697013` at revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`. Its five complete-member samples
are:

```text
76.520166, 76.746208, 73.944962, 73.501876, 74.870252 ms
```

The median is `74.870252 ms`; the hard 5x cap is therefore `14.974050 ms`.
The component medians are `71.039375 ms` for `prepare` and `3.860836 ms` for
all 13 rounds. Finish and output claims are negligible. Component medians need
not belong to the same run and are planning inputs only. Promotion uses the
complete-member wall time.

The current Metal arm still executes the CPU implementation. Its five samples
have a `75.926000 ms` median, so they are a parity baseline, not GPU evidence.
The ceilings below were registered before implementation from the retained M4
Max controls: `420.68 GiB/s` streaming copy bandwidth and `18.10 Gproduct/s`
full-width Solinas multiplication. Measured results appear after the phase
budgets.

The frozen artifact places generic stage-batch Fiat-Shamir outside the member
spans. A production PIOP comparison must charge host Fiat-Shamir to both CPU
and Metal arms or to neither. The intended evaluator charges it to both.

## Exact algebra

Only the default read-write split is admitted:

```text
phase1_num_rounds = log_T
phase3_cycle_rounds = 0
raf_evaluation_rounds = log_K = 13.
```

For the stage-1 cycle point `tau` and remapped word address `a(j)`, where
`u32::MAX` means no RAM access, the preparation result is

```text
R(k) = sum_{j: a(j) = k} eq(tau, j),
U(k) = lowest_address + 8k.
```

The address sumcheck proves

```text
sum_k U(k) R(k) = ram_address_spartan
```

and returns the `RamRa` opening at `[r_address || tau]`. The table is bound
low-to-high, so the first challenge binds adjacent indices. There are no
relation-specific challenges and no protocol change.

Split `tau` after its first `log_T - 15` big-endian coordinates. With
`I = 2^15`, `O = T / I`, `outer = j / I`, and `inner = j mod I`,

```text
eq(tau, j) = E_hi(outer) E_lo(inner).
```

The only useful full-field products in preparation are therefore one
`E_hi(outer)` multiplication per nonzero `(outer, address)` subtotal, not one
per cycle. Let `U_live` denote that topology count; at log 26,
`U_live <= O K = 2^24`.

## Chosen resident pushforward

One 1,024-thread threadgroup owns an `(outer, address-tile)` pair. The default
tile has 1,376 addresses, so six tiles cover `K = 8,192`. Each group:

1. clears `1,376 * 5` deferred `u32` words in threadgroup memory;
2. scans its 32,768-address outer block;
3. adds `E_lo(inner)` to the matching tile-local field accumulator;
4. canonicalizes each nonzero subtotal;
5. multiplies it once by `E_hi(outer)`; and
6. atomically adds the result to the global five-word accumulator for that
   address.

All six tiles are dispatched in one two-dimensional grid. A final kernel folds
the five-word sums to the canonical `K`-field `R` table. Both kernels and the
optional upload are encoded in one command buffer, followed by one wait and a
128-KiB readback.

The dynamic threadgroup allocation is exactly `27,520 B` (`26.875 KiB`). The
odd tile width is deliberate: 1,376 is the smallest SIMD-aligned width that
keeps the scan at six passes, and it leaves 5.125 KiB below a 32-KiB limit for
pipeline-static memory. Smaller aligned tiles require a seventh full address
scan and are not equivalent tuning points.

No-access rows contribute zero. A non-sentinel address at or above `K` is a
hard error. Tile zero counts invalid rows while doing its normal scan, so a
resident producer need not add another full validation pass.

## Host Fiat-Shamir and affine address tail

The device stops after constructing `R`. It never absorbs a message or draws a
challenge. The host executes all 13 address rounds because the live table is
only 8,192 fields and each round depends on the previous host challenge.

`U` is not materialized. At any round it has the form

```text
U(y) = base + step * y.
```

For a pair `R(2y) = r0`, `R(2y + 1) = r1`, let `dr = r1 - r0` and
`u0 = base + 2 * step * y`. The quadratic message is reconstructed from

```text
q(0)        = sum_y u0 * r0
leading(q)  = step * sum_y dr
q(1)        = previous_claim - q(0)
q(2)        = 2q(1) - q(0) + 2 leading(q).
```

After host challenge `r`, bind `R` in place and update

```text
R(y) = r0 + r * dr
base = base + step * r
step = 2 * step.
```

This emits exactly the same degree-two polynomial as the generic CPU prover,
but removes the `K`-field `U` table and its binds. The conservative plan still
budgets the measured `3.860836 ms` generic CPU tail until the specialized host
tail is benchmarked.

## Ownership and lifetime

The address plane is shared RAM-family state, not sequence-owned scratch. The
optimized CPU lifecycle creates it at the earliest stage-2 RAM consumer,
reuses it in stages 4 and 5, and releases it after stage 6b. The Metal owner
must carry row count, byte length, `K`, device registry ID, storage identity,
and validated provenance. `RamRafEvaluation` borrows that allocation and does
not repack it.

No shared Metal RAM-address producer is integrated yet. Until one lands, a
standalone benchmark must include the 256-MiB `u32` upload and must report it as
such. If another stage-2 RAM member creates the plane first, the producer cost
is charged exactly once to that first consumer and this member verifies the
handoff by allocation identity.

At log 26 the storage contract is:

| Storage | Bytes |
| --- | ---: |
| Borrowed resident address plane | 268,435,456 (256 MiB) |
| `E_lo` | 524,288 (512 KiB) |
| `E_hi` | 32,768 (32 KiB) |
| Global deferred sums | 163,840 (160 KiB) |
| Canonical `R` | 131,072 (128 KiB) |
| Sequence-owned total | 851,968 (0.8125 MiB) |
| Dynamic memory per threadgroup | 27,520 (26.875 KiB) |

No `T`-field equality table, `O * K` partial table, occurrence arena, CSR run
arena, or per-round device allocation is permitted.

## Shader ABI

Define `SOLINAS_OFFSET` as `0xffff_a7f7`, concatenate `fp128.metal` before
this directory's `shader.metal`, and reject any runtime compiled for a
different field offset. Register:

| Entry point | Purpose |
| --- | --- |
| `solinas_ram_raf_fold_tiles` | six-pass block-local address histogram and outer weighting |
| `solinas_ram_raf_finalize` | reduce five-word global sums to canonical `R` |

The fold buffers are:

```text
0 borrowed resident u32 address plane
1 E_lo, 32768 canonical fields
2 E_hi, O canonical fields
3 K * 5 cleared device atomic u32 output words
4 cleared [nonzero_subtotals, invalid_rows, accessed_rows, unsupported] counters
5 RamRafFoldParams
threadgroup(0) tile_addresses * 5 atomic u32 words
```

Dispatch the fold as `(O, 6, 1)` threadgroups. The finalizer receives the
deferred sums at buffer 0, canonical output at buffer 1, and the same params at
buffer 2; dispatch `ceil(K / finalize_threads)` groups. The host must clear the
deferred and counter buffers in the same command buffer before dispatch. The
per-row profiling counters are SIMD-aggregated before their device atomics and
therefore do not add one contended global atomic per cycle.

## Roofline and phase budgets

Six scans read

```text
B_address = 6 * 4T = 1,610,612,736 B = 1.5 GiB.
```

The matching tile reads one `E_lo` field per accessed row, but the unique
table is only 512 KiB. Its shader-logical traffic is at most 1 GiB; the primary
DRAM roof treats it as cache-resident and reports it separately. `E_hi` has a
32-KiB unique working set.

Each nonzero subtotal performs one full-field product and at most five 32-bit
global atomic read-modify-writes:

```text
P = U_live
B_global <= 40 U_live.
```

At the dense worst case `U_live = 2^24`, the primary DRAM charge is 2.125 GiB.
Its copy-roof floor is `5.051 ms`; the 80%-of-roof cap is `6.314 ms`. The
product floor is `0.927 ms`, so retained-bandwidth arithmetic predicts a
traffic-bound kernel. A standalone 256-MiB upload adds a `0.594-ms` floor and
a `0.743-ms` 80%-of-roof cap.

The projection using the worst-case tile cap, standalone upload cap, measured
generic CPU tail, and a provisional 0.5-ms envelope for setup, readback, and
host Fiat-Shamir is `11.418 ms`, or about `6.56x`. Without a private upload it
is `10.675 ms`, or about `7.01x`. These are analytical projections, not
measurements. Threadgroup-atomic issue rate is the largest unmodeled term.

The phase gates are:

| Phase | Promotion bar |
| --- | ---: |
| Resident handoff or charged upload | provenance-valid; upload at most 0.9 ms |
| Equality weights plus transfer | at most 0.25 ms |
| Tiled fold plus finalization | pursue 6.5 ms; redesign above 8.5 ms |
| Readback, affine host tail, and host FS | at most 4.25 ms initially |
| Complete hybrid member | hard cap 14.974050 ms; pursue 11.5 ms or lower |

The complete member bar overrides every component bar. If a measured topology
makes substantially more than 5x clearly possible, the 11.5-ms pursuit target
is tightened rather than stopping at 5x.

## Log-26 observations

The retained kernel uses 1,024 threads and the registered 1,376-address tile.
The compiled fold pipeline reports a 32-thread execution width, a 1,024-thread
maximum, and zero static threadgroup bytes. Exact parity passed across two
outer blocks, including no-access rows, randomized addresses, and a hot address
that stresses both levels of deferred carry propagation.

For a dense uniform-random synthetic address plane, one warm observation was:

| Component | Time |
| --- | ---: |
| Fold/finalize/readback wall | 6.978 ms |
| GPU active | 6.640 ms |
| Specialized 13-round affine CPU tail without Fiat-Shamir | 0.507 ms |
| Hybrid without Fiat-Shamir | 7.485 ms |

The device reported 67,108,864 accessed rows and 16,470,998 nonzero outer/bin
subtotals, close to the dense `2^24` topology cap. A focused Criterion run over
220 iterations measured the hybrid at `[7.381, 7.842, 8.547] ms`, or 9.55x at
the median against the frozen 74.870252-ms CPU member. The result clears the
11.5-ms pursuit target as well as the 14.974050-ms hard cap, but is not yet the
promoted member metric because host Fiat-Shamir and resident-producer ownership
are outside this low-level evaluator.

The topology controls were:

| Address topology | Warm active | Hybrid without Fiat-Shamir |
| --- | ---: | ---: |
| No access | 4.860 ms | 5.252 ms |
| Dense random | 7.013 ms | 7.720 ms |
| One hot address | 7.206 ms | 8.066 ms |

The one-hot/random active ratio is 1.028x, well below the registered 2x atomic
contention rejection. A repeated 512-thread control regressed to a 16.192-ms
Criterion median (4.62x), so 1,024 threads remains required.

Standalone creation of the 256-MiB address plane took 33--43 ms and cannot be
charged to this member while retaining 5x. The production path must borrow the
allocation created by the shared RAM witness producer. Sequence setup took
0.7--1.4 ms separately; its equality-table construction and allocation reuse
must be charged in the paired proof-stage measurement.

## Occupancy and counters

The 1,024-thread default supplies 32 SIMD groups to the one threadgroup that
can fit per core under the 26.875-KiB allocation. Promotion requires all of the
following evidence from the compiled pipeline and a captured run:

- `threadExecutionWidth == 32` and at least 1,024 threads per threadgroup;
- pipeline-static plus dynamic threadgroup memory fits the device limit;
- no register spill traffic large enough to invalidate the traffic model;
- one resident 1,024-thread group per core, with no persistently idle tile;
- `U_live`, maximum subtotal occupancy, invalid-row count, device bytes,
  dispatch count, readback bytes, and allocation identities;
- uniform-no-access, one-hot-address, uniform-random, and retained-Fibonacci
  profiles, because local atomic contention depends on topology; and
- counters showing whether `E_lo` is cache-resident. If not, its logical
  1-GiB traffic is added to the admission roof.

The default is an occupancy candidate, not an occupancy measurement. A
512-thread variant is only a diagnostic control: with the same dynamic memory
it cannot add a second resident group and therefore halves resident lanes.

## Rejected layouts

- A full `eq(tau, j)` table owns 1 GiB and writes then rereads it before the
  scatter. It adds at least 2 GiB of avoidable traffic.
- Direct equality evaluation from all 26 coordinates costs roughly
  `26T = 1.74 billion` field products.
- One global atomic per cycle first multiplies `E_lo` by `E_hi` `T` times and
  concentrates device atomics on hot addresses.
- An `O * K` partial field table owns 256 MiB and requires another reduction.
- CSR occurrences plus a maximal run arena own about 512 MiB for a one-stage,
  one-field pushforward. That representation is justified for the nine-stage
  bytecode member, not here.
- Running the 13 transcript-dependent address rounds on Metal adds 13 waits to
  work over at most 8,192 fields. The host tail is the intended architecture.

## Kill and redesign rules

Reject this tiled-atomic representation if any of these survives one focused
occupancy/tile-width tuning pass:

- the fold plus finalization exceeds 8.5 ms at log 26;
- local-atomic serialization makes the one-hot control slower than the random
  control by more than 2x;
- register spills add more than 10% to modeled DRAM traffic;
- the pipeline cannot launch 1,024 threads with the requested dynamic memory;
- the complete hybrid misses 14.974050 ms; or
- parity requires materializing a full cycle equality table or per-outer field
  table.

The first successor is SIMD-group key aggregation before the threadgroup
atomic, preserving the same storage and global algorithm. If that cannot clear
the cap, use a count-only topology census to choose between a hot-bin direct
reduction and this tiled path per outer block. Do not fall back to an unbounded
partial table or silently exclude producer costs.

## Parity and integration work

The low-level pushforward test and benchmark validation compare the canonical
`R` table against dense and split independent oracles. They cover no-access,
hot, randomized, and boundary-crossing rows, check allocation identity, replay
the same sequence, and validate all device counters. The shader compiles and
the runtime rejects invalid non-sentinel addresses before submission.

The proof-stage adapter still must compare every round polynomial, challenge,
final `RamRa`, and derived `UnmapAddress` value against the optimized CPU member
for:

- all no-access rows;
- one hot address;
- every address touched once;
- repeated addresses straddling inner-block boundaries;
- addresses `0` and `K - 1` plus the no-access sentinel;
- one invalid non-sentinel address, which must fail loudly;
- deterministic equality points containing zero and one;
- random traces at small powers of two; and
- the retained log-26 Fibonacci fixture.

The integration pass must create the shared resident-address owner, wire the
affine host tail into the stage-2 batch driver, preserve batch-owned
Fiat-Shamir, add CPU fallback below the measured crossover, and add paired proof
parity. All 245 `jolt-kernels` Metal tests passed after the target-scale runs.
