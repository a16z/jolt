# Hamming-weight retained-projection successor

Status: the default-off backend candidate has exact low-scale parity, a sealed
five-pair log-26 production result, and an exact log-27 diagnostic holdout.
Hamming clears 8x at log 26 and 7x at log 27; the fused family clears 5x at
both scales. Defaults remain unchanged because the one-pair log-27
Booleanity-address member measured 4.9697x, just below its fixed 5x gate.

## Decision

Keep the accepted six-selector deferred-atomic histogram topology. Replace its
five scans of the 40-byte `BooleanityRows` allocation with five scans of a
29-byte all-hot projection produced during stage 6a and retained through stage
7. This is a producer/consumer fusion, not a new stage-7 preprocessing pass.

The earlier fixed-29 sketch assigns a SIMD group to each selector and eight
buckets to each lane. Only the lane owning a row's hot bucket is useful, and
the 928-thread group carries eight field accumulators per thread. The retained
projection avoids both utilization and spill risks: every current histogram
lane still owns a cycle row and performs five or six useful selector updates.

## Protocol boundary

For selector `i`, cycle `j`, and hot address `h_i(j)`, preparation returns

```text
G_i(k) = sum_{j: h_i(j) = k} eq(r_cycle, j).
```

Akita's existing `HammingWeightPreparePlan::finish_flat` sets `G_i(0)=0`,
constructs the 29 weight tables and default-lane baseline, and runs all eight
address rounds on the host. The output order remains 16 instruction lookup
bytes, two bytecode-PC bytes, two RAM-address bytes, eight fused-increment
bytes, and fused-increment carry. Output points remain
`[reverse(address challenges) || r_cycle]`.

All Fiat--Shamir operations stay on the host. The projection contains witness
bytes only; it is independent of `r_cycle`, Hamming's `gamma`, and every
address-round challenge.

## Producer contract

Extend the stage-6a `BooleanityAddress` packed producer to write this logical
plane-major lease while it already holds each original row:

```text
planes  0..28: selector 0..28 hot byte
plane       29: bit 0 mapped-PC present, bit 1 remapped-RAM present
logical index:   plane * rows + cycle
```

The physical ABI uses one `29T` hot buffer and one `T` validity buffer. This
lets stage 6a release validity after its own packed consumers while retaining
only data stage 7 reads. The frozen device reports an 86,586,540,032-byte
maximum buffer length; the hot buffer is 1,946,157,056 bytes at log 26 and
7,784,628,224 bytes at log 28. Offsets and checked lengths remain 64-bit.

This is the 24-plane layout in
`solinas/booleanity_address_successor` with selectors 0--5 added. Its
`pack_and_first` kernel already decodes those six selectors for its first
histogram tile, so the extension is six byte stores per row, not another row
read or dispatch. The exact log-26 allocation is `30 * 2^26 = 2,013,265,920`
bytes; relative to that packet's 24-plane projection, the increment is
`402,653,184` bytes.

The lease records device registry id, source row storage id and generation,
row count, buffer identities, selector-order version, byte lengths,
producer command completion, and complete-overwrite generation. Stage 6a
releases the 64-MiB validity buffer after its own packed consumers complete and
parks the hot buffer in the proof session. Stage 6b continues to use the
original rows. After stage 6b, the raw 2.5-GiB rows can be released and only
the 1.8125-GiB hot projection survives until stage 7. Stage 7 consumes it
terminally.

Absent PC/RAM values are stored as hot byte zero and distinguished by plane 29
for stage 6a. Stage 7 does not read that flag: absent and present-at-zero are
equivalent only after Hamming's mandatory `G_i(0)=0` recentering. The stage-7
shader must skip every hot-zero update, and the host must still validate that
all returned bucket-zero fields are zero.

A private projection dispatch is outside this topology. If the packed stage-6a
producer is not selected, Hamming falls back to the accepted path; it does not
hide a sixth full-domain scan in another member.

## Stage-7 consumer

Use the frozen split `I=2^15`, `O=2^11`. The five tile widths remain
`[6, 6, 6, 6, 5]`, with 512 threads and the accepted five-word deferred
threadgroup accumulators. Each tile reads only its selector planes, the shared
`E_in[x_in]`, and `E_out[x_out]`; it performs no row decode. Retain the final
tile's common high-increment-zero and signed-carry aggregation.

The first probe deliberately keeps the current five tile dispatches and five
finalizers in one command buffer, followed by one completion and the exact
118,784-byte mass readback. This changes one variable: input representation.
It does not add an all-selector partial allocation, a new command completion,
or a device Fiat--Shamir boundary.

Compiled admission is expected to resemble the incumbent because the
histogram scratch remains at most `6 * 256 * 5 * 4 = 30,720` bytes. Promotion
still requires SIMD width 32, 512-thread tile admission, 1,024-thread finalizer
admission, no private-memory spill, and stable occupancy across all five packed
entry points.

## Frozen target and roof

The holdout is
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, SHA-256
`587e00a65bde003a7c3481f58b1ea047ed2c908b0e3d9808bbc7eec6f894b2df`,
revision `5f520c21e338632aa0bf5936ceb02be6c22fa40f`, M4 Max 40-core GPU, log 26,
16 Rayon threads, and five alternating pairs.

```text
equal-input CPU samples (ms): 545.613583 554.614169 525.892210
                              548.702500 555.909956
Metal member samples (ms):    112.953333 113.150665 111.646165
                              110.735835 110.867875
Metal GPU-active samples (ms): 84.675875 85.104750 84.896875
                               84.738625 84.998875
```

The paired median is `4.901554657x`; the metal-first and optimized-first
strata are `4.928306396x` and `4.830433671x`. The equal-input CPU median gives
complete-member caps of `109.740500 ms` at 5x and `68.587812 ms` at 8x. The
median `member - GPU-active` remainder is `26.749290 ms`, leaving active
budgets of `82.991210 ms` and `41.838522 ms` respectively. The first-probe
target is `<=40 ms` GPU-active, which leaves about 1.84 ms of complete-member
headroom at 8x if the frozen non-GPU remainder holds.

At log 26, using the incumbent partial geometry:

| traffic term | accepted | retained projection |
|---|---:|---:|
| histogram input | `5 * 40T = 13,421,772,800` | `29T = 1,946,157,056` |
| cache-unique `E_in + E_out` | 557,056 | 557,056 |
| partial write + read | 486,539,264 | 486,539,264 |
| output write + read | 237,568 | 237,568 |
| cache-optimistic total | 13,909,106,688 | 2,433,490,944 |
| fully issued total | 19,277,422,592 | 7,801,806,848 |

The cache-optimistic reduction is `5.716x`. At the retained
`451,701,710,520 B/s` copy control, the successor total has a `5.387 ms` copy
floor and `6.734 ms` 80%-of-copy cap. Charging the complete 30-byte producer
write once gives `4,446,756,864` producer-plus-consumer bytes, still `3.128x`
below the accepted stage-7 cache-optimistic traffic. This accounting is only
valid when the row read is shared with stage 6a.

The frozen census has `1,588,505,707` retained nonzero additions. The fastest
structurally matching standalone observation is
`benchmark-runs/metal-autoresearch/hamming-weight-v1/logs/trial-003-05.stdout`:
its 1,024-thread parameter trial recorded a `33.509 ms` median GPU-active interval, or
`47.405 Gadd/s`. That trial was rejected as noisy and is not promotion
evidence. The incumbent 512-thread baseline's median across its five
per-evaluator active medians was `35.452 ms`; both observations are below the
8x active budget. Treat them as directional service controls, not isolated
atomic ceilings. Reaching that budget needs
`37.968 Gadd/s`, or `80.092%` of that control; the 40-ms target needs
`39.713 Gadd/s`, or `83.773%`. Thus retained atomics, not projected DRAM, are
the likely first bound. The packed candidate preserves the proven atomic
topology and removes row decode and most source traffic, so 8x is credible
enough to pursue but is not assumed.

## One probe, bounded iteration, and kill gate

The first implementation probe is the all-hot producer plus packed stage-7
reader at the unchanged `[6,6,6,6,5]` geometry. Validate a small exact fixture,
then record one log-26 GPU-active sample with compiled resources and counters
before running an alternating campaign.

Stop immediately on a mass, round polynomial, challenge, final claim, output,
transcript, proof-byte, or verifier mismatch; a stale lease/generation; an
extra row upload or projection dispatch; an unadmitted allocation; paging; a
spill; or a stage-6a regression caused by the six added stores that outweighs
the measured stage-7 saving.

For performance:

- `GPU-active <= 41.838522 ms`: pursue the 8x complete-member gate, with
  `<=40 ms` as the preferred screen.
- `41.838522 ms < GPU-active < 84.896875 ms`: allow exactly one
  counter-directed geometry or zero-bucket-aggregation iteration, and only if
  a measured traffic/atomic roof predicts `<=41.838522 ms`.
- `GPU-active >= 84.896875 ms`, or no measured roof path to the 8x budget:
  kill this topology and keep the incumbent.

Regardless of the screen, every alternating order stratum must clear the hard
5x complete-member cap. Promotion additionally requires a positive combined
stage-6a-plus-stage-7 PIOP delta, five sealed alternating pairs, log-27
transfer, and live-proof memory admission. If counters expose headroom beyond
8x, continue rather than stopping at the target.

## First log-26 diagnostic

Revision `9f9c5575a` ran the candidate and accepted topology on the same clean
release binary, in that order, with 16 Rayon threads. Both Akita proofs
verified. The traces are retained under
`benchmark-runs/metal-retained-hamming-screen/9f9c5575a/` and sealed in
`screening_evidence.json`.

| measurement | accepted | packed + retained |
|---|---:|---:|
| Booleanity-address member | 120.065 ms | 153.262 ms |
| Booleanity-address GPU-active | 99.430 ms | 93.195 ms |
| Hamming member | 107.445 ms | 63.315 ms |
| Hamming GPU-active | 85.019 ms | 27.410 ms |
| combined member | 227.510 ms | 216.577 ms |
| combined GPU-active | 184.449 ms | 120.605 ms |

Against the frozen equal-input CPU medians, the candidate is 6.047x for
Booleanity address, 8.666x for Hamming, and 6.840x for their combined members.
It reduces same-revision combined member wall by 4.81% and combined GPU-active
time by 34.61%. The unbalanced single PIOP pair regressed by 115.800 ms because
unrelated CPU-heavy stages varied by more than the local delta; it is not
promotion evidence. This screen authorizes the five-pair alternating campaign
without another geometry iteration.

## Integration order

1. Version the all-hot projection lease, split hot bytes from validity, and
   extend the stage-6a packed producer from 24 to 30 logical planes; leave all
   defaults unchanged.
2. Add CPU-only ABI/model/oracle tests, then compile and inspect all packed
   tile pipelines before any target-size run.
3. Add an explicit Hamming `Accepted`/`RetainedAllHot` selector. Reject the new
   path unless the exact producer lease is parked and completed.
4. Add low-scale GPU mass parity, clear/ZK proof parity, allocation identity,
   reset/reuse, absent-vs-zero, signed-carry, and command-failure tests.
5. Attribute the six-plane producer increment, retention interval, stage-7
   active time, and release in the fixed production evaluator. Do not change
   the CPU denominator or host-Fiat--Shamir boundary.

## Production and holdout evidence

Revision `b4ccdaea680c2c2c141f2d8ee635bd615ebc4301` and release binary
`33e834bbe5e93868ab80efd84ec83b1a9fa4236f99deabbd4e4ce2d4c2c7786c`
were built from a clean worktree. The full records and hashes are frozen in
`production_evidence.json`.

The log-26 production evaluator ran five alternating pairs. CPU and Metal
proofs verified in every arm, and the trace proved exact dispatch/readback
counts, zero row upload, source-allocation reuse, working-set admission, and
terminal retained-buffer release.

| log 26 scope | paired median | paired samples |
|---|---:|---|
| Booleanity address | 6.7541x | 6.2075, 6.9941, 6.7541, 6.4292, 6.9424 |
| Hamming weight | 8.4015x | 8.4015, 8.2227, 8.9059, 14.1162, 7.8633 |
| fused family | 7.3855x | 6.9101, 7.3855, 7.4343, 8.7417, 7.2316 |
| full PIOP | 3.7533x | 3.8640, 3.7533, 3.7062, 3.8818, 3.4922 |

The exact one-pair log-27 diagnostic measured 4.9697x Booleanity address,
7.2183x Hamming, 5.6533x fused family, and 3.2977x full PIOP. All correctness
and lifecycle guards passed. This establishes scale and memory transfer for
the retained Hamming topology but is not sufficient to waive the fixed
Booleanity-address gate or the alternating-pair requirement.

## Still unverified

Register allocation, occupancy, cache-line behavior, target-size atomic issue
rate, and alternating-order stability at log 27 remain unverified. Promotion
also needs a sealed log-27 campaign in which Booleanity address clears 5x in
both order strata.
