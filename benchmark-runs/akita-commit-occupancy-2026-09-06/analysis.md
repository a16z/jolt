# Working model: root commit only

Source audit: ../akita-commit-eval-boost-2026-09-05/commit-eval-kernel-audit-2026-09-06.md,
sections2 and6. Source and public counter availability are verified; occupancy,
physical register count, spills and physical DRAM bytes remain unknown.

## Exact boundary

Root panel computes rank3 negacyclic shifted sums at D128, P524288, K256,
29 live/32 capacity columns,1024 blocks/column,16 position partials. It emits
canonical fp128 partial coefficients for the existing final reduction.
Production code, matrix footprint, selector loop, arithmetic,1024-thread
launch and32KiB shared tile are unchanged in initial diagnostic D0.
Full traced H3263846381 gives U=H*384=1253317010304 updates;
panel12.319893832s, device envelope12.485318084s. Production commands contain
8 streams *3rank *16partials=384 threadgroups, not the full16752 simultaneously.

## Bounds and unknowns

Matrix requests349 *3GiB=1047GiB. At measured copy440.544806GiB/s, the
all-requests-hit-DRAM proxy is2.376602653s; this is conditional, not an
irreducible traffic floor because caches reuse matrix data. Compulsory A,
allocated selector and hint bytes alone give(3+7.25+0.1875)/440.544806=
0.023692s, ignoring shared gathers, partials and overfetch. Neither is an
achievable prediction. Logical shared gathers16U=20.053TB.

Direct source tally~36 scalar operations/update =>45.119412371T operations.
Compute proxy is45.119412371/R seconds for effective issue rate R Tops/s.
R and actual ISA count are not measured; fitting R to12.32s is circular.
No defensible numerical compute floor or dominant limiter is established.
The initial work is therefore identification, not code promotion against an
invented optimum. Exact occupancy/optimality certification is unavailable.

## Ranked mechanisms and costs

1. Resource/concurrency pressure:32KiB shared plus40 persistent source words
per lane,32SIMDs/group. D0 tests fixed production PSO saturation at48..768
groups (production384); controlled shared-reservation follows only if useful.
2. Barrier imbalance: two task-specific ballots/loops per SIMD followed by
group synchronization. Count selected iterations per SIMD on real inputs
before changing mapping. Source-level binomial estimate is not idle time.
3. Live-state relief: price matrix sweep growth, instructions and barriers.
R4 already showed arithmetic-only throughput did not transfer; reshuffling
accumulators remains closed until a new causal observation justifies it.

## D0 preregistration: unchanged production shader saturation

Question: does the current384-group dispatch lie below the throughput plateau
for the complete production kernel body at its P19/3GiB matrix footprint?
Host-only harness loads the accepted unmodified MSL and compiles with safe
math/default production pipeline policy. No kernel edits in this diagnostic.
Use deterministic synthetic selectors with~56% hot entries, native stride29;
clearly not a Fibonacci input or an end-to-end prediction. Preserve total
matrix footprint and per-group2048tile work. Allocate real output geometry;
only initialize/touch the source prefix needed for the largest dispatch.

First exact reduced-shape full-output CPU oracle covers zero bytes, selected
zero mask, rotations/sign, tail tasks, rank/position partials, padded columns.
At P19, independent sampled oracle at first/last/random output coordinates
and exact H of dispatched tasks. Report GPU ms, updates/s, group count,
pipeline limits, input hash/seed, command status and telemetry if available.
One bounded warmup; initial order streams1,8,2,16,4,8. The repeated8 control
detects thermal/order drift, not a license for repeated optimization samples.

Expected: saturation by production384 groups. Falsifier:768 groups gives
>=10% higher useful updates/s than384 with <=3% control drift. That unlocks
a separately priced batching/layout test, not an occupancy claim. If plateau
is already reached, deprioritize grid-size tuning and investigate shared
resource/barrier mechanisms. >3% control drift or mismatched oracle =>invalid;
at most one targeted ordering repeat after explanation. Budget: <=15min
implementation/compile, <=5min guarded run including120s initial cooling.

## D0 result / model update,05:04 UTC

Full small-shape oracle passes checksum8efa303caf4f4675;65 independent
coefficient samples pass every target observation. P19 throughput Gupdates/s:
48groups71.554914;96groups91.062989;192groups109.811770;
384groups112.337670 and110.904011;768groups111.085341.
384control drift-1.2762%;768versus384mean-0.4797%, below the10% unlock.
Deprioritize larger-dispatch tuning. This is an effective throughput plateau,
not measured shader occupancy. Synthetic target remains~10% faster than the
recorded real input; do not interpret that cross-session difference causally.

macmon6919d7781b6c55a6e3bedff83a210435837e1dfe built locally without sudo.
Read-only JSON telemetry succeeds; target sampling windows~1572–1578MHz
after warmup. The first cold warmup includes338→1578MHz ramp. Coarse100ms
samples straddle command edges; no per-instruction frequency/cycle claim.
Peak process-family RSS6.31GiB, swaps0, no watchdog. Raw:
runs/d0-saturation.out SHA256a710f7e48b17bf698f870be3b8196b6bd7547549dfb10975556e9e0a21970e3a.
41telemetry samples; runs/d0-production.bin archives accepted compiled PSO.

## D1 preregistration: shared reservation versus useful tile work

Question: does shared-memory reservation materially restrict throughput of
the same full commit body, and would a smaller useful tile survive its extra
synchronization cost? Keep D0 inputs, H, output ownership,128coefficients,
two tasks/SIMD,1024threads,384groups and forty source accumulator words/lane.

Diagnostic-only MSL appended to accepted source uses its arithmetic/store
helpers. Replace the static shared array by a dynamic threadgroup argument;
specialize useful tile positions16 or8. Plane strides and row-loop geometry
follow useful tile size. For tile8, only reservation changes16/24/32KiB;
same PSO/instructions/read addresses/barriers in those three observations.
Metal runtime allocation is explicit; unused reserved space is not a proven
resident L1 allocation on dynamic-caching hardware. A null reservation result
does not rule out pressure from the useful working set or registers.

Price: tile8 keeps matrix requests, source selector loads and coefficient
updates unchanged. It doubles tile iterations, ballots and TG barriers
(2048→4096tiles,4096→8192barriers/TG). It does not double matrix sweeps.
The conditional matrix/compute arithmetic floors above are unchanged, while
the unmeasured control/synchronization floor rises. Tile16/dynamic32 is the
bridge control against the original static32 PSO; a >3% bridge difference
requires explanation before attributing results to tile size.

Gate: full reduced CPU oracle for original and both tile specializations
(selected-zero,sign,partial/tail/padding);65 target samples each and identical
sample checksum for all variants. One warmup original; order original,
tile16/reserve32,tile8/reserve16,tile8/reserve32,tile8/reserve24,original.
Before measured tile8 observations run one short unreported-to-score warmup
to avoid cold-PSO first-use effects. Falsifier for reservation pressure:
tile8/reserve32 no >5% slowdown againstreserve16; this only rejects a
reservation-sensitive effect at this shape. >=10% tile8 whole-panel saving
unlocks a real-input diagnostic, >=20% unlocks production candidate pricing.
<=3% original drift required; otherwise one justified ordering retry max.
Budget<=20min code/build,<=5min guarded observation with120s cooling.

No production kernel or protocol changes yet. The first20% improvement is a
milestone, not a stopping ceiling; stronger supported levers remain in scope.

## D1 result / model update,05:12 UTC

Full reduced oracle checksum8efa303caf4f4675 for all three pipelines;
every P19 sampled checksum34ba807bf9e37a5e. Original258.491250/262.292000ms;
dynamic32KiB/full tile259.004125ms (bridge difference-0.53% vs originalmean).
Half tile at16/32/24KiB reservation285.436500/286.463000/289.138625ms.
Original drift1.4704%; reserve32 vs16 penalty0.3596%; half tile16 vs
originalmean9.6182% slower. Frequency~1573–1578MHz after warmup.
Reject smaller-tile candidate; no meaningful reservation-sensitive effect.
Unused dynamic reservation may not consume resident L1, so this does not
rule out register or useful-working-set pressure. Extra tile/control work
is expensive enough to erase this resource-relief attempt.

Raw runs/d1-saturation.out SHA256
fe6233e4522c0c0eb6639ef41cda083623648b9d0cb5b92ddb3ab582d7240529.
Peak family RSS6.18GiB, swaps0, no watchdog. D0 oracle reused unchanged;
host harness only gained optional dynamic reservation and an include guard.
Further source inspection: production public A uses Private storage while
D0/D1 used Shared storage. These matched diagnostic controls remain valid
for their stated resource intervention, but production transfer additionally
requires matching storage mode. Do not promote their raw speed as real-input
commit speed. Exact occupancy/physical traffic remain unavailable.

## D2 preregistration: capture real selectors, count imbalance and repetition

Question: how much real Fibonacci root work is concentrated on the busiest
SIMD at each barrier, and how much comes from byte-identical selector blocks?
These decide between scheduling work more evenly and removing redundant work.

One isolated diagnostic build adds an opt-in host-only capture before the
existing root commit; no shader/protocol/arithmetic/evaluator changes. Capture
the canonical row-major selectors, selected-zero bitmap, shape/H/suffix metadata
to fresh files (~7.3GiB;200GiB disk free). No matrix readback needed for these
questions. Complete the full proof and require verification. Capture/IO time
invalidates this run as a performance baseline. Retain accepted binary.

Then a bounded offline CPU census computes per-column/block exact H and
per-eight-row selector counts. Reconstruct the actual two-task SIMD mapping,
including cross-block pairs and inactive tails. Report sum actual iterations
and sum32*max iterations per barrier, not a claimed GPU-idle percentage.
Also hash each column/block's canonical selector sequence, with selected-zero
distinct from absence; any hash match requires full equality comparison before
counting reusable work. Matrix indexing in the shader is independent of task
block/column, so equal full block sequences imply equal partials at every
rank/position. Retaining every original output slot leaves the commitment
and downstream eval unchanged. Hashes alone must never authorize reuse.

Falsifiers: if imbalance-only ideal gain<10%, deprioritize task balancing;
if exact duplicate elimination removes<10% updates, deprioritize deduplication.
Any production reuse candidate must additionally price fingerprinting/exact
comparison, matrix traffic, metadata construction and output replication.
No gain promised merely from a high duplicate count or imbalance ratio.

Budget<=25min implementation/build, one180s guarded full proof after120s
cooling,<=180s offline census. One production input capture is the explicit
exception allowed by the contract to replace the synthetic proxy. No other
workload matrix or CPU proof run. Inspect captured metadata/byte counts and
compare census H exactly to captured producer H before interpreting results.

## D2 result / model update,05:35 UTC

Capture binary65867b284f32fe746045892ae67ec3ef69387240d96e6617a1a9da03b39730d8,
isolated Jolt b2f89f9b9 / Akita cbd3d5b6f. Full Fibonacci proof verified,
85.44GiB RSS, swaps0, no watchdog. Its37.513090875s includes capture IO
and is not a parent performance observation. Metadata exactly matches the
prior real-input geometry/H. Captured lanes SHA256
a13d07771432eb1e21fcdef3196bdba9f0b61d9ad6649775c5777978fad5abb4;
zero bitmap e3ba30fece63b0f8a183e1d3238851cc10addb7339067c9df0dd4c2789027ca6.

CPU-only, count-only census completed11.18s immediately after capture; its
elapsed time is not a cooled production-preprocessing benchmark. Exact H
matches3263846381. Of22301tasks,1534 are zero and1528 are nonzero exact
duplicates. Unique hot work2863290349; removable fraction12.2725149%.
Duplicates are overwhelmingly bytecode columns25/26: their combined
402655160hot entries reduce to2099128 representative entries. This offers
~1.512s only under proportional panel scaling, before fingerprint/comparison
and replication costs; not enough alone to establish the20% milestone.

Barrier selected-work sum3263846381; sum32*max5838522912 =>ratio0.559019195.
99.7135% of tile barriers have a SIMD doing the maximum16 updates. Adjacent
dense columns repeatedly share a SIMD while columns0..6 are~10% dense and
columns27/28 nearly zero. This is actual input structure, not the prior
independent Bernoulli estimate. It strongly motivates pairing heavy with
light tasks while preserving each threadgroup's exact64-task set.

Raw runs/d2-census.out SHA256
7182e2d4a786b9b2aa9866f3205a458d5456bfab8b6c8b1b8eecb864077172f3.
The census is not GPU barrier-wait or occupancy measurement. Rank3 and16
position partials repeat the same selector-count structure; the ratio does
not imply44.1% wall-time savings because matrix/control work and concurrent
groups also contribute.

Preflight: fmt, taplo(outside sandbox), dependency boundaries, shared field
identity and line-cap self-tests pass; full schedule regeneration206.35s is
byte-identical. Modern Python3.13 with PATH fixed leaves only the inherited
recursive_commit error-owner failure (78/79pass). Inherited backend/runtime
line caps and two unused akita-metal dependencies still fail; typos unavailable.
Capture build4m34s; do not repeat unchanged repository-wide checks per variant.

## D3 preregistration: heavy/light task pairing within each existing group

Question: can the measured barrier imbalance be reduced without increasing
matrix sweeps, changing accumulator width or shrinking the shared tile?
First price one deterministic mapping offline on the frozen real capture.
Estimate per-column density from1024 SplitMix64-selected rows in the certified
live prefix (not regularly spaced rows, which can alias loop periodicity).
Rank each existing group's<=64tasks by sampled column count times valid rows
in its block; tie-break by original task index. Pair lightest with heaviest,
then next lightest with next heaviest. Keep an odd middle task last. No
benchmark-name or hard-coded column-family rules. Output a global task-id
permutation and recompute the exact barrier max envelope with captured counts.

Bar to implement: >=15% smaller sum32*max with unchanged exact total H,
same per-group task sets and no out-of-group tasks. If it misses, do not
write the GPU variant; investigate finer scheduling/repetition instead.
This is an analytical selection rule, not a promised speedup or an optimum.

If the mapping clears that bar, append a diagnostic shader using the original
full tile, original arithmetic/selector/store helpers,1024threads and40
source accumulator words/thread. Only global task0/1 identity becomes two
loads from a read-only mapping buffer. Each task retains its original local
row iteration and output address. U, output size, groups, matrix requests
and barriers are unchanged; task distribution before each barrier changes.
Added costs: two uniform task-index reads per SIMD per position/rank partial,
~90KiB map,1024sampled source rows and~22301*log2(64) host sort comparisons.
Source cache ordering may change even though each group's footprint does not.
No host trace-sized scan or materialization is allowed for production mapping.

Target test uses captured Fibonacci selectors and Private A/partial buffers,
matching production storage. Maintain independent reduced oracle and sampled
target oracle; compare original/permuted full output on a bounded command.
Use a predeclared interior8-stream region, not a favorable handpicked subset.
Whole-panel replay follows only if the bounded original/candidate observation
improves>=10%; repeat a surprising/near-threshold result at most once. Reject
any error, watchdog or missing parity. Keep20% as milestone, not ceiling.
This is epoch1 transaction4; checkpoint model/results before another epoch.

D3 pricing result05:44 UTC: paired max envelope4663050848 vs5838522912,
20.1330385% lower, exact H unchanged. Balance ratio0.699937978, map89204B.
Every per-group task set checked; map samples1024rows, no full-column counts
used to select the mapping. Clears15% implementation bar. Raw
runs/d3-pairing-price.out SHA256
d0ebaf9b15320b65b075f3773ac2240a65c2c0e1b35e1f5767a2c3f9cc5c315c.

GPU trial freezes first_stream=floor((349-8)/2)=170, task_offset10880,
512tasks,384groups. Use one warmup per pipeline followed by original,
mapped,mapped,original observations, no reruns unless the predeclared drift
or surprising-result rule applies. Both use the captured source/zero bitmap,
identical generated canonical A uploaded to Private storage, Private partials
and matched readback (readback excluded from GPU panel timing). Full active
output equality against the first original plus65 independent u128 samples
per observation. This remains a ranking diagnostic, not a full-proof gain.

## D3 GPU result / epoch1 checkpoint,05:58 UTC

Original262.752667/262.860375ms, mapped250.159833/248.686875ms.
Means262.806521→249.423354ms:5.0924% less time,5.3656% higher throughput;
original drift0.0410%, sampled GPU frequencies~1572–1578MHz. Full3145728
active coefficients agree and all independent samples pass checksum
8ef3d516f470806e; reduced selected-zero/tail oracle also passes.
No swaps/watchdog. Raw runs/d3-saturation.out SHA256
eb6b9081e6c13535111d7a9d2703b0531e0f78d66b8a53fc76d8ca9541c0a6ea.

Misses10% gate for full-panel replay. Park as a measured small scheduling
lever, not a production finalist. Proportional extrapolation~0.627s is not
measured full-panel/e2e saving. Broken assumption: the20.13% selector-max
envelope improvement is not the same as GPU-time improvement; other work
and/or latency hiding absorbs much of the predicted effect. No occupancy
percentage or register-pressure attribution is justified.

Epoch1's four transactions complete. No production optimization accepted;
accepted binary/pin unchanged, no pushes. One verified full proof was used
only to obtain the reusable capture. All diagnostic processes terminal.

## Epoch2 / D4 preregistration: repetition at existing position-partial granularity

Epoch2 checkpoint by07:30 UTC,<=3transactions,35min or two failed variants
per mechanism without new causal data. Same security/evaluator/serial guards;
no user authority expansion. Before a finalist, reserve the40min validation
epoch explicitly. Twenty percent remains the first milestone, not a cap.

The full-block census may hide repetition broken by rare rows. Existing
commit output already has16 position partials, each16384trace rows. Compare
exact canonical selector fragments across blocks/columns, but ONLY within
the same position-partial index: different partials address different public
A rows and cannot reuse their results merely because selectors match.

D4 is CPU-only exact counting on the existing capture. One unchanged source
pass, fragment fingerprints plus complete equality checks for every accepted
match; selected-zero remains distinct from absence. Each partial repeats
for all3rank elements. Count removable H, unique/duplicate/zero fragments,
and unique task groups ceil(unique_fragments_per_partial/64). Verify total
H exactly equals3263846381. Do not report the old full-block barrier metric
on the new fragment indexing: this pass prices repetition only.

Prediction: shorter fragments expose more exact reuse than the12.27%
full-block result. Falsifier/implementation gate: <20% removable updates or
<15% matrix-task-group reduction does not justify a standalone fragment-reuse
candidate. If it clears both, next price GPU fingerprinting, exact equality,
CPU scheduling and duplicate-output scatter BEFORE writing production code.
Never land the11s exhaustive CPU census as preprocessing. Target metadata
overhead must be subtracted from the saving; no extra trace-sized owner.

Correctness argument: original partial contribution depends on matrix rank,
position-partial and the canonical selector sequence, not original column
or block. Compute representatives once, reproduce every original partial
output slot, then keep the existing reduction and eval input unchanged.
Hash equality only filters comparisons; it cannot authorize reuse.
Budget<=10min tooling and<=180s count-only process; no GPU/proof rerun.

## D4 result,06:10 UTC / D5 preregistration: constant-template corrections

D4 exact H3263846381 conserved. Unique H2862798829:12.2875744% removable,
only0.015 percentage points beyond full-block reuse. Matrix groups per rank
5584→4816,13.7535817% lower. Both gates miss. Reject standalone fragment
reuse: shortening fragments did not uncover materially more exact equality.
Raw runs/d4-fragments.out SHA256
2dd653cb82f993d5f681ed079865d6333064a127140196a1adacb55cdd970084.
CPU count5.36s, zero swaps. Add a hand-counted domain-restriction fixture before
reusing the fragment tool; no GPU candidate was selected by this result.

D5 is a different algebraic mechanism, not another exact-repetition variant.
For task t and row r let f_r(v) be its signed negacyclic public-A contribution
for selected byte v, and f_r(absent)=0. For a fixed selected template b,
sum_r f_r(v_r) = B_b + sum_{r:v_r!=b}(f_r(v_r)-f_r(b)),
where B_b=sum_r f_r(b) depends on position partial/rank, not task block/column.
Compute B_b once, preserve every original partial and the existing reduction.
No protocol, field, security, transcript or verifier changes. Selected-zero
is a distinct symbol from absence throughout.

Price both sides: over N rows, original H updates; with n_b exact matches,
corrections need (N-n_b)+(H-n_b)=N+H-2*n_b updates. This only beats H when
n_b>N/2, excluding reusable baseline construction and lookup/control costs.
This transformation can help near-constant columns even when whole fragments
are all different. A baseline is indexed by actual selected byte and the same
public-A position domain. At most29 chosen templates; their partial storage
is29*16*3*128*16=2850816B, not a new trace-sized owner. Per-template streaming
may read3GiB of A; its measured copy proxy is~6.81ms, not a timing guarantee.
Baseline construction also costs one coefficient update per row per rank and
reduction work. Corrections retain original matrix sweeps/barriers unless a
separately registered layout intervention changes them.

First CPU-only census on the frozen capture: choose each column's template
using1024 deterministic SplitMix64 samples in the certified live prefix;
use a selected symbol only when its sampled frequency exceeds half, otherwise
choose absent/original. Exact full-column histogram then prices that frozen
choice, including the padded tail of the769 dispatched blocks. Also report
the oracle-best constant's bound separately, never silently select using it.
Conserve producer H and validate the cost identity on a small fixture with
absence, selected-zero, matching and differing selected bytes. No kernel code
before this census. Gate: >=15% aggregate correction-update reduction for a
bounded real-input shader trial; otherwise do not implement this mechanism.
The gate ranks an operation-count intervention, not a promised walltime gain.
Next shader trial must price baseline construction, extra selector branches,
negative updates, unchanged40-word state and full-output parity together.
Epoch2 transaction2,<=10min tooling/180s CPU-only observation, no GPU/proof.
