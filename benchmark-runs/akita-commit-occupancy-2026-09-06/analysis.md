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

Current checkpoint08:58UTC: D22 bounded radix26 is the next registered
mechanism; full derivation/cost/claim map is in radix26.md. It keeps five
source words/coefficient and349matrix sweeps while moving carry propagation
to a bounded16-contribution cadence. No implementation or timing yet.

D0/D1 resource-grid/reservation, D19 column grouping, D20 widened carry,
and D21 state-halving all failed their promotion gates. Reference reuse is
parked after D16/D17 metadata missed its cost gate. Do not relaunch those
mechanisms without new causal evidence. Small D3/D10 signals remain parked,
not accepted or additive savings. Physical occupancy and optimality remain
unknown. See each registered result below for the exact scope of rejection.

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

## D5 result,06:16 UTC / D6 preregistration: interleave independent task loops

Sampled and oracle-best constant-template choices both remove only6.1603563%
of updates:3263846381→3062781815, one selected template(byte4, column25).
The other dense columns are not dominated by one selected byte. This falsifies
the suggested near-constant-column mechanism at this input; no shader written.
Raw runs/d5-templates.out SHA256
2983a4b39109f886272b35aac1bdf8ca12f0da6465402a5443868f407698688c.
2.78s CPU census, zero swaps, producer H and independent correction-count
fixture pass. New fragment fixture also passes: identical symbols in16
different A domains retain16 distinct representatives, including selected-zero.

D6 returns to the original shader's dependency/control structure. Two fully
independent task accumulators already occupy40 persistent source words/lane,
but every tile runs all of task0's selected rows before task1's rows. Interleave
one selected row from each task inside a common while(mask0|mask1) loop. Keep
the original per-task update order, arithmetic helpers, coefficients, matrix
tile, task mapping, Private storage, number of groups and output addresses.
This is not the failed R4 widening: U, matrix sweeps and accumulator width do
not increase. It tests loop/control and instruction scheduling only.

Costs: retain two local selector values/masks instead of one and two active
predicates; two conditional update bodies per iteration. Loop iterations
drop from n0+n1 to max(n0,n1), but both update bodies still execute when both
are selected. Carry dependency within each coefficient is unchanged; there
is an opportunity to schedule independent task work, not a claim the compiler
will do so or that physical registers are unchanged. More temporary live state
can regress. Compulsory traffic, conditional2.377s matrix-copy proxy and U
are unchanged; effective integer issue/physical registers remain unmeasured,
so no bottomed-out/optimality percentage is available.

Use D3's frozen real-input interior8-stream command, independent small/full
oracle and65 target samples, full3145728 active-coefficient equality, Private
A/partials, one warmup per pipeline and original,candidate,candidate,original.
Prediction5–15% less bounded-command GPU time if serialized loop/control is
material. >=10% time saving with <=3% original drift unlocks full-panel replay;
otherwise park after this one variant. Any parity/watchdog/resource failure
rejects. No full proof or production changes at this stage. Epoch2 transaction3,
<=20min implement/build and<=5min guarded observation including120s cooling.
Afterward checkpoint the epoch/model before choosing another experiment.

## D6 result / epoch2 checkpoint,06:25 UTC

Original264.921833/266.010750ms; interleaved273.312250/288.793542ms.
Means265.466292→281.052896ms,5.87% slower; original drift0.411%.
All3145728 active coefficients and independent65 samples agree, selected-zero
small oracle passes. Candidate observations differ5.66%, but neither beats
either original, so there is no near-threshold winner to replicate. Reject
interleaving; no claim that added registers caused the regression. Raw
runs/d6-saturation.out SHA256
a7d79a8a51afd187939b10b86df49d770508db6799ee60c9746b943e0c58a396.
Zero swaps/watchdog, peak family6.15GiB. Epoch2's three transactions complete;
none clears its gate, and no production optimization has been accepted.

The count-only experiments cheaply rejected two algebraic mechanisms instead
of spending full-proof builds on them. Remaining evidence does not establish
an optimal kernel: exact integer issue rate, residency and cache traffic are
still unknown. Stop repeating accumulator or tile-size variants without a new
mechanism. Keep20% a milestone, not a ceiling or a promised result.

## Epoch3 / D7 preregistration: cached device gathers instead of shared staging

Epoch3 checkpoint by07:30 UTC,<=3transactions and35min or two failed variants
per mechanism. Same serial lock/cooling/parity/resource/security rules.

Exact boundary unchanged: each task accumulates the same selected negacyclic
public-A rows into the same canonical partials. Keep1024threads,64tasks/group,
four coefficients per lane/task,16position tiles,16position partials, Private
matrix/output and original task order. Replace only shared-memory staging and
gather helpers with direct readonly device-A gathers of the four selected
coefficients. Remove the cooperative matrix-copy loop and its first barrier.
Retain one execution-only threadgroup barrier at each tile end, keeping all
SIMDs within the same32KiB logical A window to preserve cache-reuse opportunity.
No mapping, wider arithmetic, protocol or host trace changes are combined.

Price: device logical coefficient-read requests rise from1047GiB cooperative
sweeps to16U=20.053TB direct gathers; explicit20.053TB shared gathers and1.124TB
cooperative loads/stores disappear. A fully uncached direct stream at measured
473.03GB/s would take~42.4s, worse than the12.32s panel. To make its memory-only
proxy fit10.267s requires>=75.8% byte reuse/hits (ignoring selector/partial
traffic); to retain the original2.377s matrix proxy requires~94.4% hits.
Ideal per-selected-position reuse is~17.8 tasks at this density, but actual
cache behavior is UNKNOWN. This is a cache-versus-explicit-staging experiment,
not an assumption that logical requests are physical DRAM traffic.

U and carry arithmetic unchanged.40 persistent source accumulator words remain;
32KiB explicit shared allocation disappears, but cached A competes for on-chip
space and addresses/temporaries change. Barriers halve4096→2048/group. The
compulsory traffic proxy is unchanged; the unknown compute floor cannot certify
optimality. A poor device cache path can outweigh both removed costs.

Use the frozen real-input interior512tasks and same private buffers/A, independent
small/full and65-sample oracles, full active-output equality, original,candidate,
candidate,original after one warmup each. >=10% time saving and<=3% parent drift
unlocks full-panel replay. Any parity/watchdog/resource failure rejects. No
automatic no-barrier variant: inspect this result before registering a second
variant. Add a9-task independent small oracle to cover the odd active-task tail
before a full-panel candidate is allowed. Budget<=20min implementation plus
<=5min guarded GPU diagnostic. Production files remain unchanged.

## D7 result,06:36 UTC / D8 preregistration: uniform coefficient-sign bands

Original263.195250/260.350500ms; cached304.064083/325.303833ms.
Means261.772875→314.683958ms,20.21% slower, original drift-1.081%.
Both candidate observations lose; no repeat. Full target equality and65
independent samples pass;10-task and new9-task odd-tail full oracles pass,
including untouched inactive output. Zero swaps/watchdog. Raw
runs/d7-saturation.out SHA256
c98e0d07c9c8fb4a3c40ed79600998038ac9191ca7917f6e3f0697e48a30657f.
Explicit shared staging remains the selected design. This does not measure
cache hits; it rejects replacing staging by this synchronized direct path.
No unsupported extrapolation to an unsynchronized variant is warranted.

D8 addresses sign/carry work without changing arithmetic width or traffic.
Each selected shift s=32q+r has q in0..3,r in0..31. For SIMD lane l and
coefficient band k in0..3, coefficient c=l+32k is positive iff
k>q or(k==q and l>=r). Exactly one band can have mixed signs across SIMD
lanes; the other three bands have uniform compile-time signs once q is known.
The current generic bool4 sign expression does not explicitly specialize q.
Use one uniform four-way branch on q, calling the unchanged existing gather/
arithmetic helper with three constant sign components and one lane predicate.
No inline assembly or guessed ISA deficit; ask the high-level compiler to
exploit a proved range property, then measure the complete shader.

Price: same U,40 persistent accumulator source words,32KiB shared tile,
matrix/shared requests, selector loop order, barriers, task schedule and
output stores. Positive bands can eliminate complement/select and first-limb
carry-in work; negative bands can eliminate dynamic sign choices. A source
count suggests roughly5–6 fewer scalar operations/update averaged across
shifts (~14–17% of naive36-op tally), NOT a measured ISA saving or time bound.
Cost: one uniform four-way branch per128-coefficient update and four copies
of the arithmetic body, increasing instruction-cache/code-size pressure.
The compiler may merge those bodies or otherwise erase the expected saving;
matrix/compute proxy floors remain as previously qualified, no optimum claim.

Gate: original,candidate,candidate,original at frozen real512-task window,
one warmup per pipeline, original drift<=3%, full3145728 coefficient equality,
65 independent samples, full10/9-task oracles with selected-zero and odd tail.
>=10% bounded GPU-time saving unlocks full-panel replay; otherwise park after
one variant. Budget<=15min implementation/compile and<=5min guarded run.
Epoch3 transaction2. No production source or protocol/security changes.

## D8 result,06:42 UTC / D9 preregistration: price deferred-sign metadata

Original265.400375/261.669958ms; sign-specialized280.053250/268.925750ms.
Means263.535167→274.489500ms,4.16% slower. Original drift-1.406%.
Correct outputs, all independent oracles/odd-tail checks pass; zero swaps or
watchdog. Reject; high-level constant signs did not yield a useful whole-body
gain. Raw runs/d8-saturation.out SHA256
adc55c74f25efedc5c7d84484c32fd4b2cbc16b8a23d4f4425f7cd6044ed9277.

D9 prices an exact arithmetic reformulation before changing the accumulator.
For a negative contribution -a, accumulate the unsigned128-bit complement
~a=2^128-1-a without the hot-loop +1. With m negative contributions and
p=2^128-OFFSET, the final result is the unsigned wide sum minus
m*(OFFSET-1) modulo p. Thus unsigned carries can accumulate normally with
initial carry0 and no per-update negative wrap decrement. The existing four
u32 limbs plus one wrap word suffice; no accumulator widening is required.
The correction is exact even for a=0. At16384rows/partial, unsigned wraps
fit i32 and m*(OFFSET-1)<2^46 fits u64. This changes only honest computation
of the same field sum, not commitment semantics, security or verification.

m for coefficient c equals count of selected shifts>c. Selected shift is
raw_byte&127; absence, selected-zero and byte128 all have shift0 and never
contribute to m. m depends on(column,block,position-partial,c), not rank or A.
A128-bin local histogram followed by a suffix sum computes all128 counts.
Store u16 counts (<=16384) for356816 fragments:91344896B (~87.1MiB), reused
across3ranks. This is bounded fragment metadata, not a trace-sized field table.
The final correction reads~274MB, and ~137M coefficient corrections are small
compared with1.253T hot coefficient updates. Original matrix traffic, tile
geometry and40 persistent accumulator source words would remain unchanged.

Both costs: proposed hot loop saves dynamic first-limb carry and signed-wrap
adjustment, approximately6/36 source ops per update before compiler effects.
New preprocessing reads201588736*29=5846073344 selector bytes, writes91.3MB,
and performs up to3.264B shared atomics plus356816 short prefix reductions.
Traffic-only copy proxy is~12.55ms at473GB/s; atomic issue/contention floor
is UNKNOWN and must be timed. It must not be hidden outside end-to-end timing.

D9 implements only this metadata producer as a diagnostic:128threads/fragment,
128 relaxed threadgroup atomic counters, inclusive SIMD prefix and4 SIMD
totals, independent suffix counts. Validate a reduced hand-counted fixture,
all target counts' range/monotonicity/last-zero, the full per-fragment identity
sum_c m_c=sum_rows(raw_byte&127), and129 direct count-by-comparison samples
including boundaries. Selection-zero does not require a special count because
its shift is0, but its nonzero commitment contribution remains in the main
kernel. No arithmetic/production shader changes before this price clears.

One warmup then two target GPU observations under existing120s initial cooling,
5s-command/180s-process/88GiB guards, same read-only capture. Gate<=150ms mean
including metadata dispatch, no sample>200ms, all checks pass. Otherwise
reject this preprocessing choice before implementing the accumulator. This is
an overhead gate, not a performance win. Epoch3 transaction3,<=20min tooling
and<=5min guarded observation. Checkpoint the epoch afterward.

## D9 result / epoch3 checkpoint,06:50 UTC

Full356816-fragment producer57.729333/58.075167ms GPU (mean57.902250),
58.838166/59.982583ms command wall. Clears150ms mean/200ms max gate.
Warmup66.633667ms GPU but548.099667ms wall: first-dispatch host mapping/setup
is material and must be charged once in the complete metadata+panel boundary,
not ignored or presumed to recur independently of the original panel's own
first-dispatch setup. No production cost conclusion from warm GPU time alone.
All356816 fragment moment/range/monotonicity invariants and129 direct samples
pass; reduced512-count full comparison passes; all repeated outputs identical.
91.3MB output, zero swaps/watchdog, peak RSS5.63GiB. Raw
runs/d9-saturation.out SHA256
d5d9f8bcd579ffa0010f4e45fa41c2d489bf7442ce589764b7d0cca2bd274aba.

Epoch3 complete: direct gathers and sign-band specialization rejected. The
negative-count preprocessing price clears its gate; it is NOT a speedup yet.
No production source changed or optimization accepted. The next question is
whether explicitly removing hot-loop correction operations pays on real data.

## Epoch4 / D10 preregistration: unsigned complements plus final correction

Checkpoint by07:30 UTC,<=3transactions;35min/two failed variants per mechanism.
Implement only D9's algebraic identity using the original shared-staged panel,
original task mapping and selected-row order. No D3/D6/D7/D8 combinations.

For each selected value, choose a or ~a according to its negacyclic sign.
First limb is unsigned sum with carry=(sum<old), then use the existing
carry helper for limbs1..3. Accumulate only unsigned overflow into the existing
wrap vector. At each original output slot, use the original unsigned-wide
reducer and subtract m*(OFFSET-1) with the original modular subtract helper.
Read m from D9's exact u16 table, indexed by(block*16+partial,column,coefficient).
Every original output slot and downstream reduction remains unchanged.

No extra hot-loop accumulator/counter state:40 source words per lane remain;
same U,matrix/shared traffic,32KiB tile,1024threads,64tasks/group,barriers and
stream ordering. Costs are D9 metadata plus final output correction/reads and
changed compiler scheduling. Explicit first carry and signed-wrap operations
are removed, rather than hoping constant-branch specialization removes them.
No claim of a known machine compute floor or20% speedup before measurement.

Bounded real-input ranking uses the frozen count artifact (never a promotion
timing with preprocessing hidden). Original,candidate,candidate,original,
one warmup each; independent65-sample and full3145728 active output equality.
Reduced10/9-task full oracles also include a separate matrix with0,1,p-1,p-2,
32/64/96-bit carry boundaries and2^127, plus selected-zero/inactive padding.
Small-shape counts are derived by direct row comparisons, independent of the
histogram producer. Admission in this diagnostic is <=16384 rows/partial.

Gate>=10% bounded panel GPU-time saving, <=3% parent drift, and proportional
12.3199s-panel saving minus measured59.41ms warm metadata wall>=1s. This is
only a ranking projection. Full-panel replay must include metadata dispatch,
first-use mapping/setup, all task windows, count storage and final correction
before any full proof or production candidate is authorized. Any parity,
watchdog or resource failure rejects. Budget<=20min tooling/compile and<=5min
guarded comparison. At most one justified ordering refresh for invalid drift.

## D10 result,06:58 UTC / D11 preregistration: interleaved shared limbs

Original264.639500/264.097500ms; deferred259.734000/242.256792ms.
Means264.368500→250.995396ms,5.06% less GPU time, original drift-0.205%.
Candidate spread is6.73%; both observations improve but neither clears10%.
No ordering refresh is authorized by the parent-drift rule. All independent
oracles and full target equality pass, including the extremal matrix checksum
7c0cd4ac28b19996. Zero swaps/watchdog. Raw
runs/d10-saturation.out SHA256
0f0a7ae4731e8232b2cd81b2775ed115c0df099025f2afff1275a2706a4dacaf.
Park below full-panel gate. Proportional panel saving~0.62s before preprocessing
is not a measured end-to-end saving. Simplifying the arithmetic did not yield
the source-operation-count percentage as elapsed time. No bottleneck fraction
can be inferred precisely from that intervention alone.

D11 isolates shared-memory layout, retaining the accepted arithmetic and all
other original scheduling. Current shared matrix has four u32 planes of2048
elements, and each coefficient update gathers one scalar from every plane.
Replace it by2048 interleaved AkitaFp128 records (same32768B), matching the
public matrix's existing AoS records. Cooperative copy stores a whole record;
each selected update reads four complete records and transposes their limb
components in registers before calling the unchanged arithmetic helper.

Costs: same U,40 persistent accumulator source words,1047GiB matrix requests,
20.053TB shared reads,32KiB tile,1024threads,64tasks/group,16partials,barriers
and original task ordering. The compiler may lower four-word records to wider
loads/stores, reducing memory instruction issue; actual ISA count is unknown.
Bank mapping changes from consecutive scalar words across SIMD lanes to
four-word strides with wide records. That can improve issue cost or introduce
bank serialization; explicitly do not assume a fourfold bandwidth gain.
Temporary live record values may also change register allocation. The compulsory
traffic and qualified copy proxies are unchanged; this is not a new measured
memory floor or an occupancy claim. No deferred-sign/cached-device combination.

Gate and order: frozen real512tasks; one warmup each, original,candidate,
candidate,original; >=10% GPU-time saving, <=3% original drift; independent
10/9-task full oracles and65 target samples plus full3145728 coefficient
equality. If below gate, park after one layout variant unless a distinct new
causal observation justifies another. <=15min tooling/compile,<=5min guarded
run. Epoch4 transaction2, no production or security changes.

## D11 invalid timing / drift reassessment,07:04 UTC

Original280.284708/262.957792ms (6.18% drift), shared-AoS256.585625/
256.960750ms. Timing verdict INCONCLUSIVE: parent exceeds3% guard; no gain
claim or promotion. All parity, odd-tail and target equality checks pass;
zero swaps/watchdog. Raw runs/d11-saturation.out SHA256
391d083c656b71ba4f8e0750fe1882c5d9097c6baae83b6317de271fad4b4ac5.

Read-only telemetry shows1572–1578MHz through measured windows, including
both parents. That <0.4% frequency range does not explain the6% time spread.
Temperature samples38.5→52.5C while measured frequency remains high; neither
thermal throttling nor background contention is established as the cause.
Current process snapshot is not historical GPU attribution. Do not kill
user/system processes or normalize timings by an invented occupancy metric.
Pause new GPU candidate timing; before resuming, register a parent-only
stability check with longer fixed-work timing aggregates. The~0.26s screen
has now shown a baseline guard failure and several4–7% candidate spreads.
Old observations stay immutable; full-proof evaluator/cooling gates unchanged.

## D12 preregistration: price row-varying column differences and a fixed block pivot

CPU-only, existing capture, one bounded transaction while GPU timing is paused.
Canonical column ownership is confirmed in jolt-claims lattice/strategy.rs:
0..15 instruction lookup selectors,16..23 balanced increment digits,24 carry,
25..26 bytecode,27..28 RAM for this shape. No workload/column IDs will be
hard-coded into selection. The data may have row-varying correlations even
though D5 found no useful constant for most dense columns.

Choose a predecessor i<j for column j only when1024 fixed SplitMix row samples
predict>=10% less work than treating j independently. For same row domain,
F_j=F_i+sum_r(f(v_jr)-f(v_ir)); skip equal canonical symbols. Correction cost
is H_j+H_i-2*equal_selected(i,j). Roots retain original H_j. Earlier-column
parents form an acyclic forest; final output reconstruction in column order
would preserve every original partial. A matching absence is not a saved hot
update; selected-zero stays distinct. Extra costs would be reference selector
reads, negative updates, predicates, and reading/writing the original partial
buffer during reconstruction. No new trace-sized field buffer is implied.

Freeze sample-selected parents before the full scan, then count exact H and
correction H. Gate>=15% aggregate update removal before any shader design.
Do not choose parents from full-data oracle counts after seeing the result.

In the same CPU pass, price a simpler preprocessing implementation of D2's
already measured full-block reuse: compare each column/block exactly to the
same column in fixed block1 (block0 if only one block). Stop comparisons after
the first mismatch, but count all task H. No hashes or O(number_of_blocks^2)
search. Record nonzero duplicate H/tasks, all-zero tasks and unique64-task
groups. This could avoid the complex histogram/dedup host path entirely; it
is not presumed to find every duplicate. D2's>=10% removed-update gate still
applies, whereas D4's>=20% gate concerned finer-fragment reuse specifically.
Any later candidate must price GPU equality flags, host mapping, clearing
unused partials and duplicate-output scatter before implementation/keep.

Conserve producer H3263846381; independent two-column/four-row fixture fixes
original H7, difference work5, fixed-pivot removable H2. <=10min tooling,
<=180s CPU-only process under lock. Epoch4 transaction3; checkpoint afterward.

## D12 result / epoch4 checkpoint,07:13 UTC

No sample-selected predecessor clears the10% per-column rule, so the frozen
forest has only roots: exact correction work equals original H3263846381.
Reject this column-difference selection mechanism, not a universal bound on
all algebraic relationships. Fixed block1 recovers919 duplicate tasks,
240910336H=7.3811788%, plus1534 zero tasks;349→311 groups (10.8883% lower).
It misses D2's10% removed-update gate. The all-block exact census had1528
duplicates and12.27% removedH, so one representative does not cover every
repeated sequence. A small fixed bank of reference blocks is a distinct
preprocessing option to price, not evidence that it already reaches that bound.
Raw runs/d12-dependencies.out SHA256
e5b9b01885ee33f6180167c1c781e551f536d14f6dcdeeb7fa1eb2066b7cf5f6.
7.59s CPU, producerH/fixture pass, zero swaps. No GPU run during drift audit.

Epoch4 complete: deferred correction has a~5% local observation below its
gate; AoS timing is invalid from parent drift; column/pivot census rejects
the registered choices. No production optimization accepted. Original source,
binary and user edits remain untouched; no pushes or watchdogs.

## Epoch5 / D13 preregistration: longer unchanged-kernel stability control

Checkpoint by08:00 UTC,<=3transactions, no candidate GPU timing until this
control resolves the short-window drift concern. Full-proof evaluator and
120s separate-process cooling remain frozen. This changes only diagnostic
timing granularity, not a candidate's acceptance threshold.

Use the unmodified production PSO and same captured512-task interior window.
Each observation encodes8 sequential compute encoders in one command buffer,
all using the same pipeline, params, buffers and384-group dispatch. Tracked
buffer hazards order the repeated writes; final output is independently checked.
No extra in-kernel loop or greater per-dispatch grid concurrency. Each batch
is expected~2.1s, below the5s command watchdog, followed by the existing matched
readback/CPU checks. GPU timestamps cover all8 dispatches; divide by8 only
when reporting per-dispatch latency, and count useful work8 times explicitly.

Two fixed warmup batches, then four measured batches of the SAME original PSO
(not two different shader bodies). Full small10/9-task parity,65 target samples
and full active-output equality remain. Gate:(max-min)/mean<=3% across four
per-dispatch means. If it fails, do not refresh candidates under this screen;
reassess full-panel/isolated observations and named external contention checks.
If it passes, future candidate screens may use this documented granularity
but old gains are not silently promoted or combined. Do not rerun until lucky.
One guarded process,120s initial cooling,<=180s and88GiB/zero-swap guards.
Budget<=10min host tooling/compile and<=5min observation. No kernel edits.

## D13 result,07:20 UTC / D14 preregistration: fixed eight-block reference bank

Unchanged per-dispatch means270.350906,268.475661,275.799630,281.571859ms.
Range/mean4.78%, above3%: reject the continuous-aggregate timing protocol.
All correctness/resource guards pass. Raw runs/d13-saturation.out SHA256
7ab7dd342c2a9b7958be2c76a82755be986557e9547913db9a0fd666ea0ea664.
The longer continuous workload added a demonstrated confound: temperature
climbed~37→77.5C, and later measured windows fell from1578MHz toward1500–1540.
This does not retrospectively explain D11's drift at near-constant frequency.
Do not normalize away the effect; separate cooled processes are the next
parent-only control. No candidate is promoted by these observations.

D14 is CPU-only pricing of an eight-reference version of D12. Fixed bank is
blocks1..8, clipped to available blocks; block0 only for a single-block shape.
Compare canonical selectors exactly, including selected-zero. For each task,
maintain a bitmask of references still exactly equal, dropping each bit on
its first mismatch. After all rows choose the smallest matching bank index;
a bank task that maps to itself remains a representative. References chosen
this way cannot form cycles: every reference maps to the least-index equal
bank member. All-zero tasks are classified separately. No hash can authorize
reuse and no input/workload-derived pivot search is performed.

This prices whether a small deterministic bank recovers D2's12.27% removed
work while avoiding full hash clustering. Both sides: up to8 reference checks
per row while masks remain live, more source loads and mask state; a GPU
version may early-finish a nonmatching nonzero task but must fully certify
zeros/matches. Later host mapping, partial zeroing/scatter still need pricing.
Gate>=10% exact removedH and>=10% unique64-task-group reduction before GPU
preprocessing design. No kernel during the current timing pause. Conserve
producerH; add a five-block/two-column hand fixture with H10 and removableH6
to test multiple reference equivalence classes and no self/cyclic reuse.
<=10min tooling,<=180s CPU under lock; epoch5 transaction2.

## D14 result,07:24 UTC / D15 preregistration: isolated cooled parent observations

Fixed eight-block bank recovers400556032H=12.2725149% and1528 nonzero
duplicate tasks, exactly D2's full-census removable work at this input.
1534 zero tasks;349→301 unique matrix groups,13.7535817% lower. Both gates
clear. This is exact work removal, not measured GPU-time savings. CPU13.01s,
producerH and multiple-class/self-reference fixture pass, zero swaps. Raw
runs/d14-bank.out SHA256
e9e4fb221564f24f3440c08b5d3a7afd8142a2242a6c60acfcc9dd7bf0f216f6.
Next reuse transaction must price GPU exact-comparison flags, CPU mapping,
partial clearing and scatter; no expensive CPU census is proposed on the
prover path. GPU candidate timing remains paused until stability control.

D15 changes the diagnostic control from continuous batches to FOUR separate
process observations, each under the machine lock after120s cooling. Each
process uses the unchanged production PSO, reduced10/9-task checks, two single
target dispatch warmups, then ONE eight-dispatch timing batch (~2.1s). This
keeps measured batch length but limits continuous target compute to~2.7s per
process, rather than D13's~13s heat ramp. Same real512-task window, Private
storage, exactH and output checks. No candidate, no evaluator/goal relaxation.

Retain all four observations in fixed order1..4; no retries. Gate range/mean
<=3% across per-dispatch times, with sampled clocks/temperatures inspected
for any residual systematic confound. No timing normalization. If it fails,
do not refresh candidates and reassess the blocked measurement assumption;
read-only analysis may continue. It is not proof of causality if it passes,
only evidence that this isolated screen is adequate to resume10%-gate ranking.
Fresh immutable artifact prefix per observation; resume only missing cells
after interruption, never overwrite/repeat completed observations. Guard each
process180s/5s-command/88GiB, zero swaps/watchdogs. Cohort<=12min including
four120s cooldowns, excluding<=10min host tooling. Epoch5 transaction3;
checkpoint afterward. Full-proof evaluator and final validation reserve unchanged.

## D15 result / epoch5 checkpoint,07:37 UTC

Isolated per-dispatch means270.975000,274.256844,268.951422,273.317609ms.
Mean271.875219ms; range/mean1.951418%, clearing3% gate.75 telemetry samples
strictly inside measured windows span1575–1578MHz and reach at most61.68C;
cold warmup samples are excluded only from this window-specific frequency
statement, not deleted from artifacts. All reduced/target oracles and full
output checks pass, zero swaps/watchdogs. Original PSO archive identical in
all four observations. Raw SHA256, observations1..4:

-025341126b0d0c685d40a4f1271b1066f9a3613997bffec4a367896bce33c384
-668562264167b119fcc3fab8dfe90a81b5418feb6bb886c76393da505def4006
-7cabb98f4ce74cc5d61c92342342c264368215f59c3b096da978dfaddf60eeeb
-da5fd412bad2b5e3c4260f36fa0ef208740d998be0491069ed7e59b584fb1ac4

Resume ranking only with separately cooled observations; do not reuse the
continuous D13 aggregate protocol. Old5% observations remain provisional
and below their gates. No optimization accepted, no production source changed.
Epoch5's three transactions complete; metadata producer for exact bank reuse
is the next priced lever. First20% throughput milestone remains unmet.

## Epoch6 / D16 preregistration: exact reference metadata producer

Checkpoint by08:30 UTC,<=3transactions,35min/two failed variants per mechanism.
D16 prices preprocessing only, before implementing a reuse panel/reducer.
One128-thread group owns one original(column,block) task. Threads scan disjoint
rows of the full block, count all canonical selected entries and AND an8-bit
mask of still-equal fixed references. A mismatch permanently clears a bit;
all rows are still counted so executed-work metrics can be conserved exactly.
Reference bank1..8 is clipped as D14; selected-zero != absence. SIMD reductions
plus4 threadgroup totals yield taskH and the least-index exactly matching
reference block, or the original block when no reference matches. No hashes.

Output two u32 words/task: true hot count and a valid representative block.
22301tasks imply178408B. A zero task retains a valid own-block representative
but is recognized as zero by its actual H, not an arbitrary sentinel. A
representative must itself be canonical/nonzero with the same H; host release
validation will reject invalid ranges/cycles/counts before any panel dispatch.
SumH must equal3263846381; unique nonzero representatives should equal19239,
executedH2863290349 on the frozen capture (independently established by D2/D14).

Costs: one additional full5.846GB selector scan, up to8 live-reference reads
per row, selected-zero bitmap reads, mask/count arithmetic and short reductions;
reference checks stop after mismatches but H counting does not. Traffic/issue
floors remain qualified because cache reuse and mask survival vary. At most
~46.8GB reference-byte requests plus source/bitmap/output if all masks survived;
this deliberately pessimistic count is not predicted DRAM traffic. Measure
GPU and complete command wall, including cold source mapping, not a fake
zero-cost preprocessing step. Routine warm overhead bar<=250ms GPU mean,
no measured sample>300ms; require all exact checks. Cold setup is recorded
separately and must be included in any complete-panel comparison.

CPU oracle uses canonical full-row comparisons (D14) and independently checks
every metadata pair, representative idempotence/same-column identity and H
conservation. Reduced fixtures cover multiple equality classes, self-reference,
no-match, all-zero, selected-zero versus absence, and a clipped bank. GPU
preprocessing is one warmup then two observations only; at this subsecond
scale the D13 sustained-load confound is not silently transferred into a panel
gain claim.120s initial cooling,5s command/180s process/88GiB, no swaps/watchdogs.

Integration design to price, not yet implement: compact only nonzero canonical
tasks, keeping original task order; original panel stores representative
partials at their original addresses. A reuse-aware final reducer reads the
representative partials and writes EACH original final output slot; zeros and
padding produce zero without reading unwritten scratch. This removes separate
partial clearing/scatter, at the cost of metadata lookup/addressing in the
~5ms reducer. It preserves the existing16-term summation order and every final
coefficient consumed by commit/eval. No protocol, arithmetic, parameters,
transcript or verifier changes. No whole-trace field allocation.

Source integration caveat from runtime.rs6379+: production currently creates
zero-copy selector slices per command and sets lane_row_offset. A metadata
pass needs a whole-source or explicitly bounded source view. Never silently
copy7.25GiB near the85.4GiB proof RSS: verify page alignment/zero-copy capability
and retain a safe existing path when a large zero-copy view is unavailable.
Charge all mapping and host compact-map costs in the complete boundary.
Budget<=20min tooling/compile and<=5min guarded preprocessing diagnostic.

D16 controller was deliberately stopped during cooldown at07:50:31 UTC:
the harness used a stale summary filename `zeros.u64le` instead of the actual
`active_zero_rows.u64le`. No shader compilation or GPU dispatch had started,
and no output/archive was created. Corrected only the input filename and
froze a new binary v2; fresh artifact prefix D16b, same kernel and all gates.

### D16b result / D17 preregistration,07:56 UTC

D16b passes every metadata pair against the serial CPU oracle and all reduced
fixtures. OriginalH3263846381,executedH2863290349,19239 unique nonzero tasks,
1534 zeros. Warm GPU678.294083/688.456500ms,mean683.375292ms: REJECT above
250ms gate. Setup7.772667ms; cold command1170.537375ms vs691.855208ms GPU.
CPU validation72.336s is diagnostic-only, never a proposed prover step.
75.23s process,5.49GiB RSS,zero swaps/watchdogs. Raw SHA256
5c4464699f1a0871243a3ec2d596ae54f6af0bbb24a05a14340ebedbed6afdad.

New causal evidence: zero columns27/28 keep all eight comparison masks alive
for the whole block despite having no commitment work. Column25's eight bank
references are exactly identical; all eight are compared for every row.
The kernel also continues full hot counting after all references mismatch.
These are source-level redundant work terms, not inferred occupancy counters.

D17 is the second/final preprocessing variant for this mechanism. Split into
two dependent encoders in one command buffer: first count trueH for each task
(same128-thread group scan/reduction, no reference comparison); then exact
reuse comparison. H0 writes own valid representative immediately. For H>0,
try bank references in increasing order, skipping references whose H differs.
Equality implies equalH, so this filter cannot discard a valid match. Compare
one reference at a time; a thread stops at its first mismatching row. Uniform
group reductions decide equality; after the first full match, stop trying
later references. Self-reference ends successfully without rereading input.
All group barriers remain uniform; no hash or sampled equality authorization.

Added costs: second dispatch,178408B metadata plus89204B hot-count buffer,
up to8 group mismatch reductions per task. Removed costs: all reference work
for zero tasks; repeated full scans of equivalent bank references; counting
inside the exact-comparison scan. Traffic floor remains at least5.846GB for
counting (~12.36ms at measured473GB/s, conditional streaming model), plus
exact comparison reads and bitmap traffic. One full duplicate comparison
costs2*262144 bytes/task, about801MB for1528duplicates, excluding bitmaps;
other comparisons can reject at first mismatch. No certified issue-rate floor.
This prices the added pass explicitly; it is not an occupancy improvement claim.

Same frozen capture, complete independent CPU metadata oracle and reduced
fixtures, one warmup/two overhead observations,250ms mean/300ms max GPU gate.
Every timing charges both encoders.120s cooling,5s GPUcommand/180s process/
88GiB/zero swaps/watchdogs. <=15min tooling+5min run; if gate fails again,
park reference reuse rather than extending the search. No reuse panel code
until this gate clears. Epoch6 transaction2, checkpoint remains08:30 UTC.

### D17 result / D18 preregistration,08:03 UTC

Both-pass GPU288.820708/289.832542ms,mean289.326625ms. Cold GPU293.791833ms,
command wall776.394166ms,setup8.061958ms. All metadata entries and reduced
fixtures match the independent oracle; exactH and unique counts unchanged.
This removes57.66% of D16 preprocessing time, but still FAILS the frozen250ms
gate. Park reference reuse after the second variant. No panel implementation
or production promotion; do not lower the gate to accommodate the result.

D18 is a different schedule mechanism with NO preprocessing requirement:
column-major original-task enumeration. Logical taskg maps to column
floor(g/full_blocks),block(g mod full_blocks), originalid=block*num_columns
+column. This is a bijection on all22301original tasks. No omitted zeros,
duplicates, changed summation, parameters or protocol. Same34964-task groups,
sameH3263846381, same16partials and public-A position domain per task.

Causal hypothesis: current row-major groups mix dense and sparse columns,
so per8-row tile barriers wait for their densest pair. D3 only repaired
pairing inside the SAME groups; D18 changes group membership globally.
Most new groups contain one column across different blocks; only column
boundaries mix columns. This may reduce maxima while preserving all useful
arithmetic and the1047GiB logical matrix requests. Price the exact tile-count
envelope using the captured input before a GPU implementation.

Added terms: global task mapping adds two integer divisions and a multiply
once per thread, not in the2048-tile loop. Selector requests span more widely
separated blocks, potentially hurting cache/TLB locality. Production would
need a validated whole-source zero-copy view (no7.25GiB copy) or bounded
range views. No extra source scan, hash, task metadata or field storage.
Measured copy-based conditional matrix floor remains2.377s; useful compute
count unchanged, actual issue/occupancy floor unknown. A smaller barrier
envelope is a ranking signal, not a predicted proportional wall gain.

D18 CPU-only census computes both original and mapped32*sum(tile_maxima),
checking H conservation, permutation coverage and identical group count.
Proceed to a separately registered GPU panel comparison only if envelope
decreases>=15%. Otherwise reject without kernel code. One deterministic
mapping, no search sweep. Existing real capture and zero semantics unchanged.
Diagnostic census may reuse its existing729MB tile counts, but no such
storage belongs on the prover path. <=10min tooling,180s CPU process under
machine lock,zero swaps, no GPU overlap. Epoch6 transaction3; checkpoint
after census, before any next-epoch GPU experiment.

## Epoch6 checkpoint / Epoch7 D19 preregistration,08:10 UTC

D18 exact envelope5838522912 ->3537163936 (39.4168013% reduction), conserved
H3263846381,349groups; balance proxy0.559019195 ->0.922729746. This is source
work accounting, NOT measured occupancy. Clears15% census gate.7.67s process,
zero swaps. Raw SHA25669ffc61bcb11d897c5e4545190e11d2cf34fe118d15137308ed5e48346279a9c.
D17 raw SHA256cc424cfdba06e07900c69545f9c843c6c00d7c8cf013a8c1075e5bfc36351013;
75.19s process,zero swaps. Reuse metadata remains parked; no accepted changes.
Epoch6 three transactions complete; source security/protocol still frozen.

Epoch7 checkpoint by09:00 UTC,<=3transactions. D19 compares the COMPLETE
P19root panel and original partial reducer, parent/candidate/candidate/parent
in four separately120s-cooled processes. No full-target warmup; reduced
independent arithmetic fixtures precede each observation. Same deterministic
public-A values as prior diagnostics,3GiB Private A and3GiB Private partials,
real captured selectors and selected-zero bits; all22301tasks,349streams,
44commands of up to8streams. Candidate changes only originaltask indexing
to the D18 column-major bijection. Arithmetic/staging/reducer unchanged.

Timer includes output/scratch allocation, selected-zero binding, selector
buffer creation, all44panel commands and reduction. Parent reproduces current
per-command zero-copy selector slices and lane_row_offset; candidate uses
one validated whole-source zero-copy view. No fallback copy. Public-A setup
and shader compilation are outside the root boundary for both. File-backed
capture differs from the prover's already-resident host input, so preserve
cold mapping/wall metrics and require later uninstrumented full-proof transfer.

Enqueue all commands as production does; wait in order with a fresh5s wait
deadline per command (earlier queued commands have finished before moving
to the next). This retains the per-command hang guard without imposing a
false5s limit on the entire~12s panel.180s process/88GiB/zero swaps/watchdogs.
Per-command timestamps, summed activeGPU, panel span, reducer and complete
wall recorded. No telemetry frequency normalization or post-hoc exclusions.

Correctness: unchanged independent fp128 oracle on reduced shapes including
odd15-task tail, selected zero, zero inputs; target65independent partial
samples. Compare every192MiB final coefficient byte against the first parent
artifact in the other three observations, including padding. Each original
task writes its own partial addresses, so there is no reconstruction change.

Gate: complete-boundary meanwall saving>=1.0s AND >=10% lower panelGPU mean,
parent wall and panelGPU drift each<=3%, all parity/resource gates. Otherwise
reject/inconclusive without a full proof. This diagnostic gate ranks a
production candidate; it does not itself promote it or certify optimality.
No repeats beyond the four preregistered observations. Cohort<=12min including
cooling, tooling<=25min. If complete-panel evidence clears, freeze a narrow
isolated-fork production candidate next; final12-proof reserve remains separate.

### D19 result / D20 preregistration,08:28 UTC

D19 GPU(P,C,C,P)11926.750875,11766.565250,11739.698333,11990.337042ms.
Complete wall11986.594916,12286.507750,12258.947625,12051.991792ms.
Parent means11958.5439585GPU/12019.293354wall; candidate means11753.1317915/
12272.7276875ms. GPU improves1.7177%; wall REGRESSES253.434334ms.
Parent GPU drift0.5317%,wall0.5441%, both clear3%. All192MiB final outputs
match SHA2560fce4fc89b37432779b9aa794aee2dc520f2c6215ea1a2805e190602904bb8d1;
all independent oracles,zero swaps/watchdogs. Reject column-major scheduling.
Raw SHA256 P,C,C,P:
e0cf9295499452cd967e95e6f1eec7d9be29260dcb217b8b2b3aa3e516e784f5
b9f398594e5e9cc84d0d06de3a5d124c7faa6cdeec8d38ffaef8b5676da2ff9b
6d2b801228ab9078b147073b96bc124405b346ab50fa56ea343949b681eaa2b8
692e489854cc4a7035c8c38ad4ab1052107aabd8217f25593b0d66d00a690e0d.

Model update: a39.4% smaller exact max-iteration envelope did not produce a
large GPU gain.32 SIMD groups/threadgroup are not32 independently issuing
physical units; scheduling overlaps their instructions. Total arithmetic,
shared gather/staging issue, and/or the changed selector locality can dominate.
The trial does not distinguish these explanations, nor prove the kernel
optimal. Whole-source cold binding adds about0.46s relative to sliced parent
binding in this replay. Do not transfer that file-backed cost unqualified to
resident-prover input; the GPU-only gain already misses the10% gate anyway.

D20 tests a different code-generation question: explicit widened limb sums
instead of manual u32 overflow detection. For each limb, form
wide=u64(lhs)+u64(rhs)+u64(carry), outputlow=u32(wide), carry=u32(wide>>32).
Since lhs,rhs<=2^32-1 and carry<=1, wide<=2^33-1 exactly. Preserve existing
complement/initial-carry sign handling, signed wrap count and final fp128
reduction. No signed16-bit wide accumulator (R4), no extra metadata (D10),
no schedule/grouping/layout changes. Both variants use original slice binding.

Price: usefulU1.253317T updates,16B shared requests/update,1047GiB logical
matrix requests,40persistent sourceu32words/thread stay unchanged. Original
limb helper spells two adds,two compares,OR,select. Widened form spells two
u64 adds and low/high extraction; Metal may lower it better, equivalently,
or worse. No claim of native64-bit issue or fewer machine instructions.
If even one source-equivalent op/limb were eliminated, that is4U=5.013T ops,
11.1% of the naive36op/update accounting; two would be22.2%. These are
conditional opportunity estimates, NOT measured floors or expected speedups.
Temporary widened state/compiler expansion could erase all benefit. The
traffic floor is unchanged and an actual issue-rate floor is still unknown.

Use the D19 complete-panel harness and frozen real input, four separately
cooled P,C,C,P observations, original44-command slice schedule for BOTH.
Gate unchanged: >=10% panelGPU mean reduction AND>=1s completewall saving,
parent drift<=3%, exact fullfinal output equality,zero swaps/watchdogs.
Add small independent extremal-field fixtures (0,1,p-1,p-2,word boundaries)
to the existing selected-zero/allzero/odd-tail oracles before target work.
No whole-target warmup, no timing normalization/retries. <=15min tooling+
12min cohort, percommand5s/process180s/88GiB. Epoch7 transaction2;
checkpoint by09:00UTC. No production implementation without a cleared gate.

### D20 result / D21 preregistration,08:44 UTC

D20 GPU(P,C,C,P)12006.517208,12715.773792,12671.314500,11983.575875ms;
wall12069.275375,12776.312375,12732.641167,12044.187917ms. Parent means
11995.046542GPU/12056.731646wall, candidate12693.544146/12754.476771ms.
GPU regresses5.8232%,wall5.7872%; parent drift0.1913%/0.2081%. Reject.
All output hashes equal D19, all six small oracles pass, zero swaps/watchdogs.
RawSHA256(P,C,C,P):
92b8c4c26c4bf0ddef7cfd7b4f673d530d11ced8394bb508e31ba5e691476c5a
201751b8802756a023a5717580794d222cd2d626402117c1379074481ac8fa7a
4e1bacc2669486ba74f6065e0f641aa1171a8c7c0312d3a08e3d73235513f123
44c906e90e5c278826779cec26ed623c3a5bb0254d9a9ea0068477cb606bcb90.

Model audit revisited existing r4-calibration-steady.out, not a new run:
one carry accumulator141.03/139.58/141.74Gupdates/s; two carry accumulators
113.89/114.00/113.82. Two wide accumulators already failed badly (~48.55),
so do NOT reopen wide64. The generic shift/xor/add issue probe~4.53 nominal
Tops/s would put36sourceops/update at~9.96s, but this is neither the same
instruction mix nor an ISA count and CANNOT certify a10s hardware floor.

D21 tests ONE carry task per SIMD, keeping four coefficients/lane. Persistent
source accumulator state40->20u32words/thread; actual registers/residency
remain unknown. Same1024threads,32KiB tile,fp128 arithmetic,16position partials,
input/output geometry and original task order.32rather than64tasks/stream;
still512tasks/command, hence16streams/768groups instead of8/384;44commands.
Each SIMD's second accumulator/loop/store is compile-time inactive. No matrix,
parameter, transcript, verifier or arithmetic change. Source binding stays
the parent's original per-command slices, including lane_row_offset.

Added cost is explicit:697matrix sweeps rather than349,2091GiB logical A
requests, twice cooperative loads/stores and tile barriers. The additional
all-requests-hit-DRAM proxy is2.370s at measured440.545GiB/s; actual additional
DRAM traffic may be lower through cache reuse, but cannot be assumed free.
UsefulH/U, shared coefficient gathers and final output writes are unchanged.
The calibration's~24% useful-rate opportunity corresponds to~2.3s at this
12s boundary IF it transferred; the added streaming/control work can erase
it entirely. The trial resolves this tradeoff in the production body rather
than inferring occupancy from register-shaped source code. No optimality claim.

Same full-panel real-input harness and exact parent-output artifact, small
odd/selected-zero/allzero/extremal oracles. All costs included. Acceptance
ranking remains>=10% panelGPU reduction AND>=1s completewall saving, final
parent drift<=3%, all parity/resource checks. No production promotion here.

Future-screen efficiency amendment, preregistered before D21 data: run cooled
P,C first. If C saves<3% GPU OR<0.3s wall, stop and reject without remaining
C,P. This stricter futility rule saves failed-trial time; it does NOT relax
acceptance or allow retries. Otherwise finish the fixed C,P observations
and score all four.120s cooling between processes,5s percommand/180s process/
88GiB/zero swaps/watchdogs. Tooling<=10min,cohort<=12min; checkpoint09:00UTC
or immediately afterward if the fourth fixed observation is still completing.
Epoch7 transaction3; checkpoint regardless of result. No additional candidate
until the epoch model/queue is updated.

### D21 result / epoch7 checkpoint,08:58 UTC

D21 GPU(P,C,C,P)12041.494542,11652.467458,11680.803042,11920.984083ms;
wall12103.277000,11714.857125,11741.449500,11982.840209ms. Parent means
11981.239313GPU/12043.058605wall; candidate11666.635250/11728.153313ms.
Saving314.604063ms GPU (2.6258%) and314.905292ms wall. Parent drift1.0058%/
1.0001%, below3%. First P,C narrowly cleared the strict futility gate, so
all four fixed observations ran. Final ranking gate fails; REJECT.
All final outputs match the original hash, all six reduced oracles pass,
zero swaps/watchdogs. RawSHA256(P,C,C,P):
a8e241fd21ae1b59ebcd2ab13084bc920c7aed6b9c75c15adbf2c92165d63b06
d384820c9eb4e22fdc66566a4d10bf35052bd2c4fc9225a4c559d288ace9ff20
af09e9f2f015563eda59629cef7c704209d3b7d89d237e953582b9b3104c51c9
738bfd9f9becde0ff8ceb31a6b72871221751c2563c6679c2ec711bfe40e5466.

Epoch7 three transactions complete. No processes or lock remain. No source
optimization accepted; target milestone unmet. Next epoch's D22 design is
radix26.md: full arithmetic invariant, i32 bound1141233817<2^31, existing
canonical reducer reuse, priced added normalization/extraction work, and
unchanged source-state/matrix traffic. It must pass independent arithmetic
checks before timing; its theoretical source-op savings are not measured
throughput. Epoch8 checkpoint09:45UTC,<=3transactions. User files and accepted
Jolt/Akita revisions remain untouched. Goal remains active beyond20% if an
actual validated larger gain can be obtained.

### D22 result and D23 preregistration,09:24UTC

D22 GPU(P,C,C,P)11956.137417,10888.848167,10953.412708,12038.310583ms;
wall12018.127209,10950.597250,11013.208584,12099.974250ms. GPU means
11997.224000->10921.130438ms:8.9695% time reduction,9.8533% throughput gain.
Complete boundary saves1077.147812ms. Parent drift0.685%GPU/0.679%wall.
Reject below fixed10% component gate; retain arithmetic evidence as a parent
mechanism for D23, not a production promotion. All512normalizer states,ten
P1024 field-oracle checks,65target samples and192MiB final bytes pass in all
four observations; zero swaps/watchdogs. RawSHA256(P,C,C,P):
e9dbf9dc10f901b963ec0c8b7ebbd3379328236f4ea28724386d4973a0c295dc
138bf622acb32a872a3c6001df528f5b5bc72f4a98ab2f5bf295e619dc762b09
c8386fb03f53c4a5623b94f0a68f907d5700948f3bb980fa0c39a41ab91db5e4
c07e645d51d6ffa7f2577d00243a8c898314646d5e8ea00fa6799dafa1928e90.

D23 cost, arithmetic reuse, layout, exact checks and frozen gates are now
preregistered in radix26.md before code. It stages decoded digits in20KiB,
removing16.56T naive source decode operations but adding25% shared traffic
and doubling tile barriers. Neither term is hidden. Epoch8 transaction2;
checkpoint09:45UTC (finish a cohort already in flight). No accepted gain.
