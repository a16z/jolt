# Booleanity-address v2: validity-free retained-hot producer

Status: source-only architectural packet. It deliberately has no root module
declaration, shader source, runtime adapter, or backend selector. The packet
freezes the relation, ABI, ownership receipts, independent oracle, roof model,
and campaign gates before another shader iteration begins.

## Exact boundary

For selector `i` and address bucket `k`, the device-owned prefix returns

```text
G_i[k] = sum_j eq(reference_cycle, j) * [hot_i(row_j) = k].
```

The output remains 29 plane-major tables of 256 Akita fields in canonical
order:

```text
0..7    lookup high bytes, most-significant first
8..15   lookup low bytes, most-significant first
16..17  mapped-PC bytes (shifts 8, 0), absent contributes nothing
18..19  remapped-RAM bytes (shifts 8, 0), absent contributes nothing
20..27  recentered fused-increment bytes (shifts 0 through 56)
28      signed fused-increment carry (-1, 0, 1 -> 255, 0, 1)
```

The host passes the 7,424 fields unchanged to
`BooleanityAddressMetalPlan::finish`. All eight address rounds, every
Fiat--Shamir append and challenge, the output claim, transcript state, proof
bytes, and verification remain on the host. This packet does not change the
protocol.

The input is the existing resident 40-byte stage-5 row allocation. The first
dispatch scans it once, writes the 29-byte plane-major hot projection needed
by Hamming, and accumulates six selectors. Four are the optional selectors:

```text
first/raw selector ids:       [16, 17, 18, 19, 0, 1]
remaining packed tile ids:    [2..7], [8..13], [14,15,20..23], [24..28]
```

Optional presence is known while decoding the original row, so selectors
16--19 can skip absent rows directly. The remaining 23 selectors are always
present. Consequently no validity plane is written or reread. Absent optional
hot bytes are still overwritten with zero for deterministic lease reuse;
stage 7 removes bucket zero, so this encoding remains exact for Hamming.

The final packed tile keeps selectors 24--28 together. It therefore preserves
the incumbent's exact local aggregation for the three common high increment
zero buckets and the signed carry. The selector reordering changes only which
dispatch owns canonical output slots; it does not reorder the output ABI.

## ABI, topology, and lifetime

`BooleanityAddressV2Params` is a checked 40-byte, 8-byte-aligned host/shader
record. The selector schedule is compile-time ABI version 2; the runtime must
reject any other version. Buffer indices are frozen in `abi.rs` and contain no
validity slot.

One member owns exactly one command buffer, three encoders/dispatches, one
completion wait, and one 118,784-byte output readback. Encoder ordering is the
only pack-to-consume and consume-to-finalize dependency. No intermediate
completion, CPU projection, row upload, or private projection dispatch is
allowed.

The hot allocation is completely overwritten by the first dispatch and then
leased to Hamming. A lease is usable only after command completion and binds
the source allocation identity, hot allocation identity, device registry,
proof generation, row count, byte count, and selector-order version. It must
report zero row upload and zero private projection dispatches. The validity
allocation and validity traffic are both exactly zero.

Complete-member accounting includes allocation/arena acquisition,
first-touch, equality-weight preparation, encode/submit/wait, output readback,
and the eight host rounds. The lifecycle receipt requires those disjoint terms
plus an explicit unattributed term to sum exactly to member wall time. GPU
active time is nested evidence and is not added a second time.

At log 26 the incumbent `inner_log2 = 15` geometry is retained because its
sealed Booleanity median is already 6.7541x. At log 27 and above v2 uses
`inner_log2 = 17`. This quarters outer partials and bucket products while
retaining 1,024 outer blocks at log 27, or 5,120 accumulator threadgroups over
the raw and four packed passes. The 512-thread accumulator and 1,024-thread
finalizer shapes remain unchanged.

## Log-27 lower bounds

Let `T = 2^27`, `P = 29`, `K = 256`. The retained control uses `I = 2^15`,
`O = 4096`; v2 uses `I = 2^17`, `O = 1024`. Cache-optimistic traffic counts
each split-equality table once. Fully issued traffic charges all five logical
`E_in` scans.

| Term | retained packed-hot | v2 validity-free |
| --- | ---: | ---: |
| resident row read | 5,368,709,120 | 5,368,709,120 |
| hot/validity write | 4,026,531,840 | 3,892,314,112 |
| packed selector/validity read | 3,355,443,200 | 3,087,007,744 |
| cache-unique `E_in + E_out` | 589,824 | 2,113,536 |
| partial write + read | 973,078,528 | 243,269,632 |
| output write + host read | 237,568 | 237,568 |
| cache-optimistic total | 13,724,590,080 | 12,593,651,712 |
| fully issued total | 24,461,746,176 | 23,329,038,336 |
| owned bytes | 4,513,779,712 | 4,016,181,248 |

V2 removes 1,130,938,368 cache-optimistic bytes and 497,598,464 owned bytes.
At the retained M4 Max 420.68-GiB/s copy control, the cache-optimistic floors
are 30.384189 ms and 27.880461 ms. The v2 80%-of-copy cap is 34.850576 ms.
These are traffic controls, not complete latency forecasts.

Useful field products are only `P*K*O`: 30,408,704 for the retained geometry
and 7,602,176 for v2. At the retained 16.42-Gproduct/s control, the v2 product
floor is about 0.463 ms. Its product intensity is 0.000604 products per
cache-optimistic byte, so field multiplication cannot bind this slice.
Present histogram contributions remain exactly `25T + 2B + 2R`, where `B`
and `R` are mapped-PC and remapped-RAM present-row counts. The incumbent local
aggregation is retained, so an atomic-service control must still be measured
and included phase-by-phase before roof promotion.

The only exact log-27 production artifact retained for this member is the
one-pair speedup, 4.969700993x; raw component timings are unavailable. This is
not enough evidence to claim a deterministic scaling regression. It does show
the required margin: 5.3x needs 6.646% more throughput. The cache-optimistic
traffic ratio is 1.089802, projecting 5.415991x if complete time scales with
that traffic alone. The larger split also quarters product/finalizer work and
reduces allocation/first-touch pressure. Those are hypotheses to falsify, not
measured gains.

## Adjustment and falsification

This is one architectural packet, not a shader search:

1. Reorder optional selectors into the raw-row tile.
2. Delete the validity plane from allocation, bindings, writes, and reads.
3. Select `inner_log2 = 17` at log 27+, while retaining the proven log-26
   geometry and implementation.
4. Preserve the existing deferred-atomic accumulator and common-increment
   local aggregation. Do not introduce a new atomic scheme in this campaign.

Before target execution, compile and inspect the pipelines. Kill the candidate
if it spills, uses private memory in the accumulator, cannot admit 512 threads
with 30,720 dynamic threadgroup bytes, or cannot admit the 1,024-thread
finalizer with 16,384 bytes. Admission must include the live proof allocation
census, not nominal machine memory.

The first runtime screen is one exact log-27 pair in the same binary. It must
show all of the following:

- equality of all 7,424 masses, all host round polynomials/challenges, final
  claim, transcript state, proof bytes, and verifier result;
- exact receipt topology and complete lifecycle reconciliation;
- no validity allocation or traffic, no row upload, and one original-row scan;
- at least 5.3x complete-member CPU speedup;
- at least 3% complete-member improvement over retained packed-hot in the same
  binary; and
- no regression to the retained Hamming consumer or the combined family.

Any miss kills this packet; do not tune it in place. A passing screen advances
to five alternating CPU/v2 pairs at log 27, followed by a sealed holdout. Every
pair must exceed 5x, while the overall median, both order-stratum medians, and
sealed holdout must each reach 5.3x. Continue beyond 5.3x if the calibrated
phase-summed roof and held-out samples show clear headroom.

## Root wiring and validation

1. Declare `booleanity_address_v2` in `solinas/mod.rs` for the CPU-only
   ABI/model/oracle tests.
2. Add a separate v2 shader fragment and pipelines only after source review;
   do not mutate the retained successor shader.
3. Add a runtime adapter mapping the checked receipts onto `BooleanityRows`,
   `HammingHotRows`, and `BooleanityAddressMetalPlan::finish`.
4. Add the backend selector only for the pre-registered log-27 screen. Keep
   the retained implementation at log 26 and as fallback before submission.
5. Exercise absence/present-zero parity, every byte and carry bucket,
   adversarial fifth-word overflow, log-28 offsets, allocation reuse,
   command failure, transcript/proof parity, and verifier parity.

Root validation commands (not run in this source-only lane):

```bash
cargo fmt -q
cargo clippy -p jolt-kernels --features metal,test-utils --lib -- -D warnings
cargo clippy -p jolt-kernels --features metal,test-utils --tests -- -D warnings
cargo nextest run -p jolt-kernels booleanity_address_v2 \
  --features metal,test-utils --cargo-quiet
```

## Could not verify in this lane

- compilation, formatting, clippy, or tests;
- MSL register allocation, occupancy, spills, or private-memory traffic;
- target-size allocation or execution;
- a same-binary raw retained/v2 log-27 timing;
- cached-weight or deferred-atomic issue controls;
- backend, transcript, proof, or verifier parity.
