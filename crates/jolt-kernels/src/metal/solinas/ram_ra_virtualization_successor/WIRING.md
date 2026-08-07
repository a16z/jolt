# RAM-RA virtualization successor

Status: analytical packet only. It is not registered, compiled, or benchmarked.

## Decision

Use a 16-cycle compact microtile for messages 0--4, materialize two dense
factor tables at `T / 16`, continue on Metal through message 16, and run the
remaining nine messages on the optimized CPU. The microtile replaces, rather
than supplements, `RamRaClaimReduction`'s high-major compact view.

This keeps the protocol unchanged. The all-sparse event graph in
`ram_ra_virtualization/` is not the first implementation: at the measured
`A = 22,000,000` access count, support becomes nearly dense after a few binds,
and building and retaining an `O(A log T)` graph is the wrong producer cost.
The existing dense-address P0 remains the control.

## Frozen boundary

The optimized-CPU control is Fibonacci `T = 2^26`, `log_K = 13`, two committed
eight-bit factors, and 16 Rayon threads from
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`.

Complete-member samples are:

```text
278.459584, 332.764663, 270.177247, 274.665791, 270.797830 ms
```

The median is `274.665791 ms`; the 5x cap is `54.933158 ms` and the 8x cap is
`34.333223 ms`. The median CPU continuation after message 16 is `0.748206 ms`.

The resident Metal boundary begins with validated producer allocations and
compiled pipelines. It includes all message dispatches, reductions, waits,
scalar reads, the dense cutoff readback, and the CPU continuation. It excludes
the shared producer only in the member-incremental view. The joint RAM-family
view charges that producer once. Allocation, attachment, and equality-table
construction remain part of complete-member preparation and must fit inside
the `25 ms` pursuit bar.

Fiat--Shamir remains on the host. Stage 6b hashes the combined batch message
and draws one shared challenge after all members return. Local member timings
charge generic hash work to neither arm; the PIOP evaluator charges it to both.

## Algebra

For cycle `j`, with no-access rows mapped to zero:

```text
f_i(j) = eq(r_chunk_i, chunk_i(address[j]))
S(j)   = eq(r_cycle, j) * f_0(j) * f_1(j).
```

The 26 cycle variables bind low-to-high. Each message is the same cubic as the
optimized kernel. Outputs are the two factor values in committed-factor order
at `[r_chunk_i || reverse(c_0..c_25)]`; the bound cycle equality is checked
against `EqCycle`. The shader may return an endpoint plus the inner quadratic
coefficient, but the host must reconstruct the canonical four evaluations and
check the running claim before absorbing them.

## Shared producer and typed owner

The authoritative RAM walk publishes one owner with:

```text
generation, rows, address_limit, access_count
device_registry_id
dense_address_storage_id
low_major_storage_id, low_major_offset_storage_id
microtile_mask_storage_id, microtile_offset_storage_id,
microtile_address_storage_id
byte lengths and a content-validation seal
```

The microtile representation is:

```text
mask[tile]       : u16, one bit per cycle in a 16-cycle tile
offset[tile]     : u32 start in compact_addresses
offset[tile + 1] : u32 end
compact_addresses: u16 address values, set-bit order
```

Every address is below `2^13`. Offsets are monotone, their final value equals
`access_count`, and each range length equals `popcount(mask)`. These invariants
are checked while the authoritative dense plane is available; detached counts
are not trusted.

`RamRaClaimReduction` Q borrows the low-major view. Its H-prime gather borrows
the microtile view, grouping the fixed 512 tiles for each high index. Stage 5
releases the low-major allocation after Q and passes the same microtile owner
through the session. Stage 6b takes it, verifies generation/device/storage
identity, keeps it through message 4 completion, then releases it after dense
materialization. Error and CPU-fallback paths leave or release the owner exactly
once.

The former high-major view was `4A + 4(8192 + 1) = 88,032,772` bytes. The new
microtile is `2(T/16) + 4(T/16 + 1) + 2A = 69,165,828` bytes, so it removes
`18,866,944` producer-write bytes while serving both consumers.

At log 26 the complete co-materialized producer outputs are:

| output | bytes |
|---|---:|
| dense address plane | 268,435,456 |
| low-major claim view | 88,032,772 |
| shared microtile | 69,165,828 |
| total, charged once | 425,634,056 |

The output-only copy floor is `0.942291 ms`. A forbidden late conversion reads
the dense plane and writes the microtile, moving `337,601,284` bytes with a
`0.747399 ms` copy floor before launch overhead. The first production probe
must co-materialize it; a late conversion is only a diagnostic control.

## Kernel schedule

Messages 0--4 scan the microtiles and the small challenge-scaled branch tables.
Empty subblocks emit no arithmetic. The exact live-block census controls useful
products; the model uses the placement-independent worst case

```text
E_1..E_5 = 22,000,000, 16,777,216, 8,388,608, 4,194,304, 2,097,152.
```

Message 4 writes two dense tables of length `T / 16`. Messages 5--16 use the
ordinary fused bind-and-message dense transition. The host reads 1,024 values
per factor (`32,768` bytes), binds challenge 16 exactly once on CPU, and emits
messages 17--25 plus the final outputs.

There are 17 command buffers and waits in the standalone accounting. A later
stage-level scheduler may co-submit other stage-6b members and pay one wait per
round, but this kernel receives no analytical credit for that overlap.

Cutoffs after messages 10, 16, and 19 are the required alternating controls.
Message 16 is selected because six additional dense rounds add only 306,176
products while replacing about 2.879 ms of CPU continuation with six waits.
The message-19 control determines whether the final launches are worthwhile.

## Exact roof at log 26

The worst-case census and message-16 cutoff produce:

| quantity | value |
|---|---:|
| prefix products, including branch scaling | 213,926,400 |
| dense-transition products | 16,919,552 |
| total useful full products | 230,845,952 |
| perfect bytes | 893,151,988 |
| logical bytes | 977,038,048 |
| shader-requested bytes | 4,496,784,096 |
| sequence-owned storage | 202,432,496 |
| resident storage including borrowed microtile | 271,598,324 |

Perfect bytes count each offset once per pass and cache-sized branch tables
once. Logical bytes count the two offset requests per tile. Requested bytes
also count every 16-byte factor-table lookup; the tables are at most 128 KiB,
so this is deliberately pessimistic about cache reuse.

Using the retained M4 Max controls of `18.1 Gproduct/s` and
`451.701710520 GB/s` gives:

| floor | latency |
|---|---:|
| arithmetic | 12.753920 ms |
| perfect traffic | 1.977305 ms |
| logical traffic | 2.163017 ms |
| requested traffic | 9.955208 ms |
| 80%-of-roof active bar | 15.942400 ms |

The requested-byte ridge is 180,189,249 products. The worst-case plan is
compute-bound at 230,845,952 products; a clustered census can become
traffic-bound without changing the requested bytes. Seventeen 141-us command
floors, a 141-us setup floor, and the 0.748206-ms CPU tail give a
`19.228606-ms` resident pursuit projection. Charging the entire shared
producer floor once gives `20.170897 ms`. Both leave preparation slack below
the explicit `25-ms` pursuit bar and the `34.333223-ms` 8x cap.

## Occupancy boundary

The prefix shader needs two field accumulators and at most four live factor
values; the reduction scratch is `2 * (threads / 32) * 16` bytes, or 128 bytes
at 128 threads. The dense transition has the same two-column reduction shape.
Static threadgroup memory is expected to be zero.

No numerical occupancy claim is made before compilation. Public Metal APIs do
not expose register allocation or resident SIMD-group capacity on this device.
The first compiled artifact must report execution width, legal threadgroup
maximum, static/dynamic threadgroup memory, spills, and an Instruments
occupancy capture. Widths 64, 128, and 256 are screened in one binary. A width
is rejected if it lowers occupancy or useful product throughput without a
traffic reduction.

## Independent correctness oracle

`oracle.rs` uses a small prime field and evaluates the unfactored dense
relation at all four round points. It independently encodes and decodes the
microtile, checks the two committed chunk equalities multiply to the full
13-bit address equality, and shows the microtile H-prime gather equals a dense
scan. Production parity must repeat these checks with `jolt-field` Akita,
including empty, single-access, clustered, scattered, dense, repeated-address,
domain-edge, zero/one/random challenge, malformed-offset, and storage-identity
fixtures.

## One first probe and kill gate

Implement only the shared microtile H-prime gather first, using the same
allocation as an unregistered round-0 virtualization probe. Compare it in one
binary with the accepted high-major H gather and dense-address P0 round 0.
This single probe decides both layout reuse and prefix viability.

Proceed only if all of the following hold at the real log-26 access census:

1. exact Akita outputs match the independent dense oracle;
2. microtile H-prime active time is no more than 1.05x the accepted high-major
   gather;
3. round-0 useful-product throughput projects the five-round prefix to at most
   `14.773923 ms` active, the prefix's 80%-roof bar; and
4. producer delta is nonpositive when the high-major view is replaced, with
   zero extra full-domain scan or upload.

Any miss kills the microtile phase immediately. Keep the dense-address P0 as
the next candidate; do not repair the evaluator or add a sparse event graph in
place. If the probe passes, implement messages 0--4, then the dense ladder,
then run the 10/16/19 cutoff comparison. Promotion requires five alternating
exact complete-member pairs, proof verification, the 5x interval, and either
the 8x interval or evidence that the measured active path already reaches the
pre-registered roof bar.

Absorbing this relation into stage-5 claim reduction remains a protocol-level
option, but it is not pursued while the unchanged protocol has credible 8x
headroom.
