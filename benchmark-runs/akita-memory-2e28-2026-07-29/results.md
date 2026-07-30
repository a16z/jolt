# Akita K256 memory campaign

Date: 2026-07-29 EDT

## Outcome

This pass kept K256 and the proof protocol fixed. Nine independently committed
changes reduced retained or phase-local prover memory without a reproducible
prover slowdown:

| Commit | Change | Storage removed |
|---|---|---:|
| `839ab0a5e` | Stop retaining `R1CSCycleInputs` | 208 B/cycle during Stage 1 |
| `7eb3d7a03` | Store packed one-hot lanes as native bytes | 29 B/cycle throughout the packed commitment/opening lifetime |
| `e3ada4ee0` | Release `RaIndices` after Stage 7 | 54 B/cycle before reconstruction/opening |
| `ca6fb8e52` | Materialize fused-inc lane columns at Stage 6 | 18 B/cycle during commitment and Stages 1–5 |
| `a1edad11d` | Materialize signed fused deltas at Stage 6 | 16 B/cycle during commitment and Stages 1–5 |
| `937319abb` | Retain compact proof rows and stream trace conversion | 32 B/cycle throughout proving |
| `0be326e83` | Read instruction RA from the row-major index source | 32 B/cycle during the first three Stage-6b rounds |
| `39bc6ce38` | Read RAM RA from the row-major index source | `2 * ram_d` B/cycle during the first three Stage-6b rounds |
| `1f3652bc4` | Store fused-increment one-hot lanes as dense bytes | 9 B/cycle during Stages 6–7 |

At `2^26`, the lowest measured maximum RSS is 44.244 GB with zero process
swaps, down from 50.49 GB after the R1CS-row policy change and 50.72 GB before
it. The latest dense-lane run measured 44.310 GB; another phase determines
that peak, so the 576 MiB Stage 6/7 reduction is not visible in headline RSS.
The R1CS change has a much larger phase-local effect than the headline maximum:
Stage 1 drops by the exact 13 GiB row allocation.

The effective maximum-RSS slope between the measured `2^22` and `2^26`
points falls from approximately 566 to 469 B/cycle. A linear extrapolation of
the final measurements is approximately 129 GiB at `2^28`. Different phases
can become the maximum at different sizes, so this is not a capacity forecast,
but it is sufficient to reject the idea that the current stack is ready for a
low-swap `2^28` run.

## Measurements

All full runs forced
`PERF_LOG_K_CHUNK=8` and
`PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32`. The harness printed
`OneHotTrace one-hot K: 256`, verified the proof, and reported zero swaps.
RSS values from `/usr/bin/time -l` are decimal bytes; phase samples are
reported in GiB.

### R1CS rows

| Variant | Prove | Stage 1 | Maximum RSS |
|---|---:|---:|---:|
| Retained rows | 65.27 s | 6.98 s | 50.72 GB |
| Direct trace access | 65.18 s | 7.96 s | 50.49 GB |

Direct access adds approximately 0.98 s to Stage 1 at `2^26`, while removing
13 GiB from that phase. The whole-prover pair did not regress, but no speedup
is claimed. The memory-minimal policy is unconditional at every trace size.

### Native byte lanes

The rejected first implementation stored bytes but widened every cache read
back to `u16`; its affected aggregate regressed by approximately 5.6%. The
accepted implementation changes the internal Akita trace-one-hot API to
consume `u8` directly. Byte zero denotes the already-virtualized logical lane
zero/no committed coefficient; lanes `1..=255` are unchanged.

At `2^26`, normal warm proofs were 64.35 s and 65.45 s versus the 65.18 s
control. Maximum RSS was 48.75 GB, approximately 1.74 GB below the control.
The exact retained saving is 29 B/cycle: 1.8125 GiB at `2^26` and 7.25 GiB at
`2^28`.

### Release `RaIndices`

`RaIndices` is needed through Stage 7, but the packed opening source needs only
the byte lanes. Splitting ownership and dropping the final
`Arc<Vec<RaIndices>>` before reconstruction removes 54 B/cycle from the late
working set: 3.375 GiB at `2^26` and 13.5 GiB at `2^28`.

This does not lower every `2^26` headline maximum because Stage 3/4 can peak
before the drop. In the phase-sampled run:

| Phase | Maximum RSS |
|---|---:|
| Stage 3 | 42.17 GiB |
| After Stage 7 | 27.26 GiB |
| Packed opening | 38.43 GiB |

The warm proof was 63.69 s. The change adds no passes or conversions.

### Deferred fused-inc columns

Nine `Option<u8>` columns (18 B/cycle) were previously allocated before the
commitment and retained through Stage 5, although those phases never read
them. The packed row-major cache now derives its lanes directly; the
column-major `RaPolynomial` inputs are materialized immediately before Stage 6.

Maximum RSS fell to 47.33–47.50 GB. The warm `2^26` proof was 64.78 s, within
the existing normal-run distribution. The logical early-phase saving is
1.125 GiB at `2^26` and 4.5 GiB at `2^28`.

### Deferred fused deltas

The 16-byte signed delta vector is also unused before Stage 6. Its first
derivation is now fused into the existing row-cache pass, and the signed
vector is materialized at Stage 6.

The final measured maximum RSS is 46.33–46.46 GB. A temporary test-only switch
ran eager and deferred schedules in the same compiled binary:

| Size | Eager | Deferred | Difference |
|---|---:|---:|---:|
| `2^22` | 6.319 s | 6.338 s | +0.019 s |
| `2^26` | 66.877 s | 65.898 s | −0.980 s |

The target pair does not establish a speedup: unchanged phases account for the
movement. The directly added target-scale work is 52.6 ms in cache
construction plus 73.7 ms in the deferred delta pass, below the 0.48 s
promotion threshold. The retained early-phase saving is 1 GiB at `2^26` and
4 GiB at `2^28`; avoiding the old temporary allocation also lowers observed
resident pressure during cache construction.

### Compact proof trace rows

The prover now retains the existing 64-byte `JoltTraceRow` instead of the
96-byte tracer `Cycle`. A bounded sink converts `2^18` cycles at a time, so a
trace-sized raw vector is never allocated. The compatibility adapter confirmed
why this matters: dropping a 5.230 GB raw allocation before proving did not
return its pages to the OS, and maximum RSS remained 46.321 GB.

At `2^26`, the bounded path measured 44.244 GB, 2.213 GB below the 46.457 GB
control and close to the exact 2 GiB retained-byte prediction. Trace generation
plus conversion took 7.293 s versus 7.273 s for the full-vector adapter.

This representation change also reduced the proof from 67.184 s to 54.95 s.
Cached flags, logical values, and bytecode indexes remove repeated decoding in
Stages 1, 3, and 6a, while witness commitment scans the compact rows instead of
replaying the lazy emulator trace. This does not change proof messages or
verifier inputs.

### Row-major instruction RA

At K256, the instruction RA virtualization prover previously transposed the
retained RA index rows into 16 `Option<u8>` columns. It now reads those 16
indices directly from the shared row-major source for the first three rounds,
then materializes the same field polynomials as before.

This removes exactly 32 B/cycle from the early Stage-6b working set: 2 GiB at
`2^26` and 8 GiB at `2^28`. It also improves locality. At `2^26`, Stage 6b
falls from 5.569 to 5.129 seconds (-7.9%), while instruction initialization
falls from 74.4 to 0.19 ms. The proof measured 52.63 seconds.

Maximum RSS was 44.325 GB versus the 44.244 GB compact-row control, so another
phase still determines the global peak at this size. The result is accepted as
a phase-local allocation and runtime improvement, not claimed as a new
headline RSS low.

### Row-major RAM RA

The RAM virtualization prover now reads optional RAM chunk indices directly
from `RaIndices` for three rounds and then materializes the same field
polynomials at `T / 8`. This removes one `Option<u8>` column per RAM chunk, or
`2 * ram_d` B/cycle, from the early Stage-6b working set.

At `2^26`, RAM initialization, message generation, and binding changed from a
218.1 ms aggregate to 201.0 ms. Total Stage 6b was effectively identical:
5.1289 versus 5.1330 seconds. Maximum RSS was also effectively unchanged
(44.325 versus 44.304 GB) because another phase determines the process peak.

The full proof measured 53.48 seconds versus 52.63 seconds, but unchanged
commitment and packed-opening spans account for 0.81 seconds of the 0.85-second
difference. The change is accepted as a performance-neutral phase-local
allocation cut, not a claimed speedup.

### Dense fused-increment lanes

The eight K256 increment chunks and carry column are present on every cycle,
but were stored as `Option<u8>`. They now retain the same contiguous
column-major access pattern in nine dense byte vectors, with an all-present
source added to the existing lazy `RaPolynomial` state machine.

This removes 9 B/cycle while the columns are live: 576 MiB at `2^26` and
2.25 GiB at `2^28`. At `2^26`, the Stage 6a+6b+7 aggregate improves from
6.6795 to 6.6424 seconds (-0.56%). Maximum RSS is effectively unchanged
(44.304 versus 44.310 GB) because another phase sets the process peak. The
proof measured 53.84 seconds versus 53.48 seconds; an unchanged commitment
span accounts for 0.260 seconds of the 0.36-second movement.

### Rejected: dense bytecode RA

The bytecode transpose was changed from `Option<u8>` to `u8` while preserving
its blocked column-major layout. For this workload's two bytecode chunks, that
would remove 128 MiB at `2^26` and 512 MiB at `2^28`.

The gather improved from 29.2 to 17.3–17.5 ms at `2^26`, but bytecode message
generation regressed from 581.8 to 597.8 and 612.6 ms. Total Stage 6b moved
from 5.102 seconds to 5.197 and 5.246 seconds (+1.87%, +2.82%). The candidate
was reverted; the saving does not pass the no-regression gate.

### Rejected: row-major bytecode RA

The analogous bytecode port removed its transpose but made the sparse
coefficient-read kernel slower in both `2^22` screens: +11.7% and +2.9%.
Initialization and binding savings kept Stage 6 neutral, but the direct
row-major access shape is not the right primitive for this low-density family.
The candidate code was reverted. A dense `u8` column can retain contiguous
access while halving bytecode's current `Option<u8>` storage.

### Rejected: shared packed RA source

The next experiment replaced the retained 54-byte `RaIndices` row with views
over the packed byte lanes. The final variant used one RAM-validity byte per
cycle and removed the predicted 53 B/cycle: its `2^26` maximum RSS was
42.911 GB, 3.546 GB below the 46.457 GB control versus 3.557 GB predicted.

It did not pass the performance gate. The `2^22` focused Stage 6/7 aggregate
improved by 3.04%, but at `2^26` the same aggregate regressed from 8.543 s to
9.206 s (+7.77%). The full proof remained within ordinary phase noise, but that
does not erase a repeatable hot-path regression. All candidate code was
reverted; the experiment log and named traces are retained as negative
evidence.

## Trace inventory

Primary Perfetto traces are in `benchmark-runs/perfetto_traces/`.

| Purpose | Trace |
|---|---|
| R1CS retained control | `mem-r1cs-parent-2e26.json` |
| R1CS direct-access control | `mem-r1cs-nocache-2e26.json` |
| Native byte lanes, warm | `mem-u8native-2e26-b.json`, `mem-u8native-2e26-c.json` |
| Release RA indices, phase sampled | `mem-drop-ra-2e26-b.json` |
| Deferred inc columns, phase sampled | `mem-defer-inc-2e26.json` |
| Deferred inc columns, warm | `mem-defer-inc-2e26-b.json` |
| Deferred deltas, phase sampled | `mem-defer-delta-2e26.json` |
| Deferred deltas, warm | `mem-defer-delta-2e26-b.json` |
| Same-binary eager/deferred pair | `mem-delta-pair-eager-2e26.json`, `mem-delta-pair-deferred-2e26.json` |
| Packed RA source v1 screens | `mem-ra-source-2e22-b.json`, `mem-ra-source-2e22-c.json` |
| Packed RA source v2 screen/target | `mem-ra-source-v2-2e22-b.json`, `mem-ra-source-2e26.json` |
| Packed RA source v3 screen/target | `mem-ra-source-v3-2e22.json`, `mem-ra-source-v3-2e26.json` |
| Compact proof rows | `mem-trace-row-2e22.json`, `mem-trace-row-2e26.json` |
| Full-vector row adapter | `mem-trace-row-adapter-2e22.json`, `mem-trace-row-adapter-2e26.json` |
| Row-major instruction RA | `mem-ra-row-2e22.json`, `mem-ra-row-2e22-b.json`, `mem-ra-row-2e26.json` |
| Row-major bytecode RA rejection | `mem-ra-bytecode-2e22.json`, `mem-ra-bytecode-2e22-b.json` |
| Row-major RAM RA | `mem-ra-ram-2e22.json`, `mem-ra-ram-2e22-b.json`, `mem-ra-ram-2e26.json` |
| Dense fused-increment lanes | `mem-dense-inc-2e22.json`, `mem-dense-inc-2e22-b.json`, `mem-dense-inc-2e26.json` |
| Dense bytecode RA rejection | `mem-dense-bytecode-2e22.json`, `mem-dense-bytecode-2e22-b.json`, `mem-dense-bytecode-2e26.json`, `mem-dense-bytecode-2e26-b.json` |

The matching `.log` and `.rss` files for phase-sampled runs are under
`benchmark-runs/akita-memory-2e28-2026-07-29/logs/`.

## `JoltTraceRow` layout

`JoltTraceRow` is statically checked to be 64 bytes. Its 32-byte value area
aliases columns that are equal or mutually exclusive on final rows:

| Row class | Four physical value slots |
|---|---|
| Non-memory | `rs1`, `rs2`, `rd_pre`, `rd_write` |
| Load | `rs1`, RAM address, `rd_pre`, loaded value |
| Store | `rs1`, `rs2`/RAM write, RAM read, RAM address |

For a load, the final loaded value is simultaneously RAM read, RAM write, and
register write. For a store, `rs2` is the RAM write value and no register is
written. The conversion rejects a raw cycle if these contracts do not hold.
The other 32 bytes cache source PC, compact bytecode PC, immediate, flags,
instruction tag, and three register ids.

Replacing `Cycle` with this row saves 32 B/cycle: 2 GiB at `2^26` and 8 GiB at
`2^28`, while cutting trace bandwidth by one third. The real-trace parity test
compares all R1CS inputs, logical values, flags, lookup indexes, table routing,
operand presence, and canonical padding against the former `Cycle` path.

## Next memory targets

The next targets are ordered by retained bytes and likelihood of remaining
performance-neutral:

1. **Compact signed deltas in Stage 6.** A magnitude vector plus sign bitset
   can reduce the 16-byte `i128` stream to about 8.125 B/cycle, saving roughly
   2 GiB at `2^28`. The representation must preserve the bytecode read-RAF
   conversion and the first bind of the fused-increment MLE without adding a
   hot-loop regression.
2. **Audit field-vector and setup lifetimes at the Stage 3/4 and opening
   peaks.** The remaining structural cuts do not by themselves reach
   the 95 GiB objective under the current slope, so phase-local `Fp128`
   vectors and setup matrices must be counted and streamed/reused.

K16 is not part of this campaign. K256 remains the fixed performance choice.
