# Akita K256 memory campaign

Date: 2026-07-29 EDT

## Outcome

The source-derived capacity ledger and ordered attack plan are in
[`analytical-memory-model.md`](analytical-memory-model.md). The first four
structural targets have landed; their derivation and measurements are in
[`structural-cuts-results.md`](structural-cuts-results.md).

At `2^28`, the commit projection is now 67.00 GiB, the Stage-6b transition is
71.18 GiB, and the evaluation proof is about 33.3 GiB. The structural ceiling
is Stage 5 at 75.18164 GiB, or 300.7266 B/cycle.
That leaves 14.81836 GiB below the 90 GiB working target for background
destruction, allocator residency, and unmodelled state.

The full forced-K256 `2^28` proof then passed and verified in 236.72 seconds.
Maximum RSS was 80.655 GiB (322.62 B/cycle), leaving 9.345 GiB below the
90 GiB working target. The process reported zero swaps, and the system
swapout counter did not increase.

This pass kept K256 and the proof protocol fixed. Eighteen independently committed
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
| `1355fab03` | Pack signed fused-increment deltas | 7.875 B/cycle until the first Stage-6 cycle bind |
| `2d08372ec` | Delay Fp128 expansion of instruction-input columns | 3 GiB of Stage-3 field state at `2^26` |
| `7afb90166` | Release compact trace rows after Stage 7 | 64 B/cycle before reconstruction/opening |
| `5d1ff81a1` | Materialize RA rows after Stage 5 | 53 B/cycle during commitment and Stages 1–5 |
| `cffef8618` | Release packed lane rows after the accepted root fold | 29 B/cycle during the recursive opening tail |
| `8232e5828` | Cache only the negacyclic packed-row transform | 23.515625 GiB from the `2^28` commit peak |
| `a6c5ed811` | Release the compact trace at its final reader | 64 B/cycle from Stage 6b onward |
| `095ae7eb5` | Stream capacity-safe root quotient chunks | 47.03125 GiB from the `2^28` evaluation proof |
| `720e1a7d1` | Reuse read-RAF transition storage | `(16 + 8lambda)` B/cycle from the Stage-5 transition |

At `2^26`, the lowest measured maximum RSS is now 36.264 GB with zero process
swaps, down from the 44.157 GB packed-delta control, 50.49 GB after the
R1CS-row policy change, and 50.72 GB before it. The three-round
instruction-input change removes 4.128 GB (-9.35%) from the process maximum.
Releasing the trace then removes 4.01 GiB at the end of opening and moves the
sampled global peak to Stage 6b, although the earlier `/usr/bin/time` maximum
moves by only 0.106 GB. Deferring RA rows removes another 3.3125 GiB of
retained state before Stage 6 and lowers the process maximum by 0.999 GB. The
opening lifecycle hook then releases 1.8125 GiB of packed rows after the
accepted root fold. It lowers the sampled opening tail but leaves the
Stage-6b headline peak effectively unchanged.
The compact trace's earlier final-reader release removes another exact 4 GiB
before Stage 6b and lowers maximum RSS by 2.56 GB. Streamed root quotient
chunks then remove the later 10 GiB fallback cache and improve packed opening
from 10.977 to 10.547 seconds; the process maximum remains in the earlier
PIOP window.
Reusing the read-RAF `u_evals` allocation and releasing lookup state at its
final reader lowers the modeled Stage-5 transition by 4–6 GiB at `2^28`.
The affected `2^26` span was 4.97 seconds versus 5.00 seconds in the
immediate control.
Earlier phase-local cuts are not always fully visible in headline RSS. The
R1CS change, for example, drops Stage 1 by the exact 13 GiB row allocation.

The latest `2^26` maximum is 539.67 B/cycle. It must not be
scaled linearly to `2^28`: setup ranks, matrix rounding, program state,
allocator arenas, and thread stacks do not scale as `4T`. The source-derived
post-cut ceiling is 300.7266 B/cycle. The remaining capacity question was
whether logically dead and allocator-resident pages would stay inside the
14.81836 GiB working reserve.

At `2^28`, the sampled PIOP maximum is Stage 6b at 77.95 GiB, followed by
Stage 4 at 74.08 GiB and Stage 5 at 67.82 GiB. The source model successfully
bounded capacity, while its 14.82 GiB reserve covered allocator residency and
short-lived Stage-6b scratch. macOS compressed an additional 14.49 GiB at the
sampled pressure peak, so further memory work can still improve operating
margin even though the no-swap target now passes.

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

### Packed fused-increment deltas

The signed fused-increment stream now uses one `u64` magnitude plus one sign
bit per cycle instead of one `i128`. This removes 7.875 B/cycle until the first
Stage-6 cycle bind: 504 MiB at `2^26` and 1.96875 GiB at `2^28`.

A naive first bind using two field-by-`u64` multiplies regressed 5.0% in the
microbenchmark and was rejected. The accepted word-aligned kernel reconstructs
signed pairs and preserves the existing one-multiply interpolation. It
measured 442.09 µs per `2^20` inputs versus 523.94 µs for the generic current
path.

At `2^26`, the proof measured 53.35 seconds and 44.157 GB maximum RSS with
zero swaps. The Stage 6a+6b+7 aggregate was 6.697 seconds versus 6.642 seconds
for the control (+0.83%), while the directly changed or adjacent instrumented
spans improved by 37.6 ms in aggregate. The whole-proof improvement is not
claimed because the unchanged commitment span moved more than the headline.
The result is accepted as a performance-neutral 504 MiB phase-local cut.

### Delayed instruction-input expansion

Eight compact Stage-3 columns previously expanded to Fp128 at the first bind,
allocating 4 GiB of field coefficients at `2^26`. They now remain as small
`bool`, `u64`, and `i128` values for three rounds and materialize the same
bound polynomials directly at `T / 8`, where they occupy 1 GiB.

The optimization is prover-local and enabled only for 16-byte fields, so
sumcheck messages and the Dory comparison baseline remain unchanged.

At `2^26`, maximum RSS fell from 44.157 to 40.029 GB (-4.128 GB, -9.35%).
Stage 3 improved from 1.127052 to 1.099730 seconds. On-demand message reads
became 187.971 ms slower, but challenge binding became 221.715 ms faster, so
the directly affected aggregate improved by 33.744 ms (-4.23%). The whole
proof moved from 53.35 to 53.59 seconds; an unchanged commitment span accounts
for 278 ms, so no prover regression is attributed to the change.

Although the polynomials are dead after Stage 3, avoiding their large
intermediate field allocations lowered the Stage 4 baseline by 4.79 GB. The
sampled global peak is now 36.29 GiB inside packed opening. Full measurements,
the binding-equivalence argument, and validation are in
`stage3-svo-experiment.md`.

### Release compact trace before opening

The 64-byte `JoltTraceRow` vector is no longer read after Stage 7. Akita
opening reads the independently owned packed byte-lane cache in its hint, so
the final 4 GiB trace owner at `2^26` can be dropped without a rebuild or
conversion.

At `2^26`, proving measured 53.66 seconds versus the 53.59-second control, and
packed opening measured 11.11 versus 11.04 seconds. The opening-end RSS sample
fell from 23.95 to 19.94 GiB, matching the exact allocation. Opening maximum
fell by 3.11 GiB, and the whole-proof sampled maximum fell by 2.13 GiB. The
headline maximum moved only from 40.029 to 39.923 GB because an earlier
Stage-3/4 transient now determines it.

The same trace is 16 GiB at `2^28`, making this a material capacity cut even
though it cannot change phases that precede Stage 7. Full ownership analysis,
measurements, and validation are in `opening-drop-trace-experiment.md`.

### Deferred RA-row materialization

The initial one-hot cache pass previously retained both the 29-byte packed
lane row used by commitment/opening and the 54-byte `RaIndices` row first used
in Stage 6. It now retains the lanes plus one RAM-validity byte and constructs
the same RA rows immediately after Stage 5.

This removes 53 B/cycle during commitment and Stages 1–5: 3.3125 GiB at
`2^26` and 13.25 GiB at `2^28`. At `2^26`, the retained commitment plateau
fell by approximately 3.50 GiB and maximum RSS fell from 39.923 to 38.924 GB.
Proving measured 53.54 seconds versus the 53.66-second control; Stage 6b
measured 5.19 versus 5.27 seconds. No speedup is claimed.

The Stage-6 representation and kernels are unchanged, so this is an
early-lifetime capacity cut rather than a solution to the current Stage-6b
peak. Full measurements and validation are in
`deferred-ra-materialization-experiment.md`.

### Release packed opening rows after the root fold

The trace-backed packed lane cache is read by Akita's root evaluation and root
decomposition, but not by the recursive fold tail. Akita now calls a generic
release hook after `prepare_root` has finished all nonce retries and produced
the accepted root-fold witness. Jolt drops its cache at that boundary.

This releases exactly 29 B/cycle: 1.8125 GiB at `2^26` and 7.25 GiB at
`2^28`. Two target runs measured 54.21 and 53.70 seconds versus the
53.54-second control; the affected packed-opening span measured 11.25 and
11.12 seconds versus 11.14 seconds. The repeat and small-scale pairs support
performance neutrality.

Opening-end RSS fell from 20.42 GiB to 18.62 and 18.09 GiB. The lowest process
maximum was 38.876 GB versus 38.924 GB, an effectively unchanged headline
because Stage 6b precedes the release. The proof and transcript are unchanged.
Full ownership analysis, measurements, and validation are in
`opening-hint-lifetime-experiment.md`.

### Rejected: drop lazy replay snapshot

`JoltCpuProver` retains a `LazyTraceIterator` clone of the initial emulator,
although neither current prover reads it after compact trace rows are built.
Dropping it at construction reduced every `2^22` phase baseline by only
0.04 GB. Proving measured 5.69 seconds versus the 5.72–5.77-second controls,
with the same 2.64-second opening.

The saving is fixed rather than per-cycle and is too small to justify removing
a public prover field. The candidate was reverted. Its trace is retained as
`mem-drop-lazy-2e22.json`.

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

### Rejected: fourth delayed RA round

Akita's shared and per-polynomial RA sources were kept indexed for one extra
sumcheck round, moving field materialization from `T / 8` to `T / 16`. At
`2^22`, the reported Stage-6b plateau fell from 5.52 to 5.30 GB, but Stage 6b
regressed from 324.133 to 401.518 ms (+23.9%).

The generic fourth-round coefficient path performs eight indexed table reads
where the control reads one contiguous Fp128 coefficient. The candidate was
reverted at the small-trace gate, before a `2^26` run. Full measurements and
the retained trace are in `stage6-ra-round4-experiment.md`.

### Rejected: row-level RAM validity

`RaIndices` repeats the same `Option<u8>` presence tag in all eight RAM
slots. A row-level validity flag shrank the row from 54 to 47 bytes, removing
7 B/cycle. At `2^26`, two runs lowered maximum RSS from 38.924 GB to 38.475
and 38.415 GB.

The directly affected `compute_all_G + SharedRaRound3::bind` aggregate
regressed by 3.3–3.4%. Padding the row to a 48-byte, eight-byte-aligned stride
recovered the `2^22` signal but regressed the target-scale aggregate by 6.5%.
Both variants were reverted under the no-performance-regression policy.
Measurements and all retained traces are in
`ram-validity-layout-experiment.md`.

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
| Packed fused deltas, accepted | `mem-packed-delta-2e22-c.json`, `mem-packed-delta-2e22-d.json`, `mem-packed-delta-2e26-b.json` |
| Packed fused deltas, untuned | `mem-packed-delta-2e22.json`, `mem-packed-delta-2e22-b.json`, `mem-packed-delta-2e26.json` |
| Packed fused deltas, K16 side data | `mem-packed-delta-k16-2e22.json`, `mem-packed-delta-k16-2e22-b.json` |
| Instruction-input three-round screens | `mem-svo3-2e22.json`, `mem-svo3-2e22-b.json` |
| Instruction-input three-round target | `mem-svo3-2e26.json` |
| Late trace release screen | `mem-drop-trace-2e22.json` |
| Late trace release target | `mem-drop-trace-2e26.json` |
| Deferred RA rows screens | `mem-defer-ra-2e22.json`, `mem-defer-ra-2e22-b.json` |
| Deferred RA rows target | `mem-defer-ra-2e26.json` |
| Opening-row release screens | `mem-opening-release-2e22.json`, `mem-opening-release-2e22-b.json` |
| Opening-row release targets | `mem-opening-release-2e26.json`, `mem-opening-release-2e26-b.json` |
| Lazy replay snapshot rejection | `mem-drop-lazy-2e22.json` |
| Fourth delayed RA round rejection | `mem-ra-round4-2e22.json` |
| RAM-validity 47-byte rejection | `mem-ram-valid-2e22.json`, `mem-ram-valid-2e22-b.json`, `mem-ram-valid-2e26.json`, `mem-ram-valid-2e26-b.json` |
| RAM-validity 48-byte rejection | `mem-ram-valid-a8-2e22.json`, `mem-ram-valid-a8-2e22-b.json`, `mem-ram-valid-a8-2e22-c.json`, `mem-ram-valid-a8-2e26.json` |
| Negacyclic-only cache | `mem-neg-ntt-2e22.json`, `mem-neg-ntt-2e26.json` |
| Trace final-reader release | `mem-trace-early-2e22.json`, `mem-trace-early-2e26.json` |
| Streamed root quotient | `mem-stream-t-2e22.json`, `mem-stream-t-2e26.json` |
| Stage-5 read-RAF reuse | `mem-stage5-reuse-2e22.json`, `mem-stage5-reuse-2e26.json` |
| Full K256 `2^28` fit | `akita_28.json` |

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

Stage 6b now sets the sampled `2^26` peak at 34.16 GiB, with the earlier
Stage-3/4 transient setting `/usr/bin/time`'s maximum. The next target is the
Stage-6b field-vector overlap through allocation ownership or scheduling;
another generic indexed RA round and a RAM-tag-only row compaction are now
rejected. The opening-hint audit is complete: its remaining packed row cache
now ends after the accepted root fold. Releasing the setup matrix remains
rejected under the current implementation because its late reconstruction
regresses both opening and verifier performance.

K16 is not part of this campaign. K256 remains the fixed performance choice.
