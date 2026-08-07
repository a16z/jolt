# Booleanity-address packed-hot successor

Status: isolated executable slice, registered in the Solinas source library but
default-off in the backend. The checked runtime owns one command buffer and
exposes the all-hot lease only after successful completion. CPU ABI/oracle/model
tests and a log-15 exact Metal mass-parity test pass. Backend wiring, proof
parity, target-size timing, and promotion remain outstanding.

The old candidate in this directory was correctly rejected: assigning one
SIMD group to a selector and one lane to each eight-bin stripe left only the
lane owning a row's hot bucket useful. This replacement never assigns buckets
to lanes. Every accumulation lane owns a distinct cycle row and contributes to
five or six selector histograms.

## Frozen relation and output

For production selector `i` and address bucket `k`, the device must return

```text
G_i[k] = sum_j eq(reference_cycle, j) * [hot_i(row_j) = k].
```

The 29 tables remain in the current ABI order:

```text
0..15   lookup bytes, shifts 120, 112, ..., 0
16..17  mapped-PC bytes, shifts 8, 0; absent cycles contribute nothing
18..19  remapped-RAM bytes, shifts 8, 0; absent cycles contribute nothing
20..27  recentered fused-increment bytes, shifts 0, 8, ..., 56
28      signed fused-increment carry in buckets 255, 0, or 1
```

The result is still `29 * 256` Akita fields. Integration must pass it unchanged
to `BooleanityAddressMetalPlan::finish`, which retains the existing eight host
address rounds, output claim, challenge reversal, and transcript operations.
No Fiat--Shamir operation moves to the device and no protocol message changes.

`oracle::unfactored_pushforward` is the primary scalar oracle. It decodes all
29 selectors directly from each original 40-byte row and forms the full
`E_out[x_out] * E_in[x_in]` weight per cycle. It does not read the packed
buffer, reuse its selector decoder, or regroup by outer block. The separate
packed/factorized oracle models the proposed dispatches and must match it.

## Dominant slice

The accepted kernel scans the 40-byte row allocation once for each of five
selector tiles. This slice pays for a transient selector projection but makes
only one original-row pass:

1. `pack_and_first` assigns one 512-thread group to each `x_out`. Every lane
   streams distinct rows, accumulates lookup selectors 0--5 in the accepted
   six-table five-word threadgroup histogram, and writes all 29 hot bytes to a
   device-private plane-major lease plus one invocation-private validity byte.
2. `packed_tiles` assigns four independent 512-thread groups to each `x_out`.
   They consume selector tiles 6--11, 12--17, 18--23, and 24--28. All lanes
   stream rows. The final tile retains the accepted exact common-high-byte and
   signed-carry local aggregation.
3. `finalize` assigns one 1,024-thread group to each selector and reduces its
   2,048 outer partials with four lanes per bucket.

The two producer/consumer boundaries must be separate compute encoders in one
command buffer. At log 26 the fixed topology is:

| Phase | Threadgroups | Threads | Dynamic threadgroup bytes |
|---|---:|---:|---:|
| pack + selectors 0--5 | 2,048 | 512 | 30,720 |
| four packed selector tiles | 8,192 | 512 | 30,720 |
| 29-selector finalize | 29 | 1,024 | 16,384 |

The complete member has one command buffer, three encoders, three dispatches,
one completion wait, and one `118,784`-byte readback. A buffer fence or extra
command completion between phases is not part of this design. The integration
test must prove that encoder ordering provides the required write/read hazard
on the target runtime.

The MSL uses 64-bit packed and partial offsets. Row counts remain 32-bit, which
covers the log-26, log-27, and log-28 targets without the `29 * T` hot-plane
offset wrapping at log 28.

## Producer ABI and lifetime

The input is the existing resident `BooleanityRows` allocation:

```text
lookup_lo:               u64
lookup_hi:               u64
ram_address_plus_one:    u64
fused_inc_magnitude:     u64
packed_pc_and_flags:     u64
```

Its allocation identity, device registry id, row count, and stage-5 producer
event must be identical at stage 6a and stage 6b. There is no row upload,
second 40-byte row plane, CPU projection, or repack.

The projection is not free producer work. It is created and completely
overwritten inside the timed first dispatch. Stage 6a consumes both buffers,
then releases validity and retains the hot lease through stage 7. Its
plane-major layout is:

```text
hot planes 0..28: selector 0..28 hot byte
validity[cycle]: bit 0 mapped-PC present, bit 1 remapped-RAM present
hot index:       selector * rows + cycle
```

Writing zero to absent hot slots is required even though the flag suppresses
their contribution. This gives a complete overwrite on reuse and prevents a
cold row from aliasing a present bucket-zero row.

At log 26, owned storage is:

| Allocation | Bytes |
|---|---:|
| retained 29-byte hot projection | 1,946,157,056 |
| invocation-private validity | 67,108,864 |
| all-selector outer partials | 243,269,632 |
| `E_in` | 524,288 |
| `E_out` | 32,768 |
| output | 118,784 |
| total | 2,257,211,392 |

Admission must use the live proof allocation census, not the machine's
nominal unified-memory size. Allocation and first touch belong to the member
unless a proof-scoped arena lease with a prior producer event and complete
overwrite is demonstrated.

## Traffic and work at log 26

Let `T = 2^26`, `I = 2^15`, `O = 2^11`, `P = 29`, and `K = 256`.
Cache-optimistic traffic counts the 512-KiB `E_in` and 32-KiB `E_out` tables
once. Fully issued traffic charges all five logical `E_in` scans.

| Term | Formula | Bytes |
|---|---:|---:|
| original resident-row read | `40T` | 2,684,354,560 |
| hot + validity write | `30T` | 2,013,265,920 |
| packed selector/flag reads | `25T` | 1,677,721,600 |
| partial write + read | `2 * 16PKO` | 486,539,264 |
| output write + host read | `2 * 16PK` | 237,568 |
| cache-unique weights | `16I + 16O` | 557,056 |
| cache-optimistic total | | 6,862,675,968 |
| fully issued total | | 12,230,991,872 |

The accepted five-row-scan topology has 13,909,106,688 cache-optimistic bytes
under the same accounting. This slice reduces that term by 2.027x. At the
retained 420.68-GiB/s copy control, it gives a 15.193-ms traffic floor and an
18.992-ms 80%-of-copy cap. Those are not complete latency predictions.

The exact selector-row opportunity count is `29T = 1,946,157,056`. Present
contributions are `25T + 2B + 2R`, where `B` and `R` are the observed
mapped-PC and remapped-RAM present-row counts. For dense optional columns and
no common high bytes, the shader issues 7,541,358,592 mandatory four-limb
threadgroup atomic word additions. If every row hits the three shared
high-increment zero buckets, the count is 6,736,052,224, including the six
per-worker flushes. Fifth-word overflow atomics are additional and
data-dependent. The post-bucket field products remain
`PKO = 15,204,352`, a 0.926-ms floor at the retained 16.42-Gproduct/s control.

`model::CalibratedRoof` deliberately does not hide these operations behind the
aggregate traffic number. It computes a lower bound for each sequential phase
as the maximum of that phase's DRAM, cached-weight issue, threadgroup-atomic,
and bucket-product floors, then sums the three phase bounds. Before the first
target run, measure cached 512-KiB field-read bandwidth and the five-word
threadgroup atomic stream in the same binary and freeze their rates. The
candidate must reach at least 80% of the resulting phase-summed roof before
geometry tuning or promotion.

## Speed gates

The frozen equal-input CPU samples from the clean log-26 production holdout
are:

```text
972037919, 929139914, 899211126, 907191128, 948932957 ns
median = 929139914 ns
```

They set complete-member caps of 185.828 ms at 5x, 116.142 ms at 8x, and
92.914 ms at 10x. Five times is the hard project minimum in every execution-
order stratum. It is not a sufficient reason to replace the accepted kernel:
that kernel already has a 111.635-ms production median and 8.453x paired
speedup. Because this slice halves its dominant modeled traffic, 10x is the
pre-registered stretch target. Search must continue past 5x when the roof and
paired samples retain that headroom.

A candidate is promotable only if all of the following hold:

- every one of the 7,424 masses, all eight host round polynomials and
  challenges, the final claim, transcript state, proof bytes, and verifier
  result match the optimized CPU member;
- five alternating pairs at log 26 and five at log 27 each clear 5x in both
  order strata, with a sealed holdout after tuning;
- the paired successor is statistically faster than the accepted Metal arm;
- the 10x target is pursued unless the calibrated roof or held-out samples
  falsify it;
- exactly one original-row scan, 30 projection writes per row, 25 packed reads per
  row, three dispatches, one completion wait, and one output readback are
  observed;
- original-row allocation identity is stable through stages 5, 6a, and 6b;
- buffer allocation, first touch, encoder creation, dispatch, wait, readback,
  host rounds, and any arena acquisition delay reconcile to member wall time;
- compiled limits admit both 30,720-byte accumulator kernels at 512 threads
  and the 16,384-byte finalizer at 1,024 threads, with no spills or private
  memory traffic.

## Root integration order

1. Add an explicit `Accepted`/`PackedHotSuccessor` A/B selector and pass the
   7,424-field readback to `BooleanityAddressMetalPlan::finish`.
2. Park the completion-gated hot lease in `ProofSession`; keep validity
   invocation-private and release it after stage 6a.
3. Add GPU parity tests for cold bucket zero, all 256 byte values, both signed carries,
   adversarial fifth-word overflow, log-28 offset arithmetic, allocation
   identity, reset/reuse, and command failure.
4. Keep
   `Accepted` as the default until the paired promotion campaign completes.

## Still unverified in this lane

- clippy and full-workspace tests;
- register allocation, occupancy, private memory, or spills;
- allocation success or peak proof residency at log 26, 27, or 28;
- cached-weight bandwidth and deferred-atomic issue controls;
- transcript parity, proof parity, or verifier parity;
- any latency, throughput, speedup, or incumbent comparison for this successor.

These are promotion work, not evidence that the topology is blocked. The next
go/no-go is proof-session integration followed by one target-size screen.
