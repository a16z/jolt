# Akita Metal protocol-preserving 5x ledger

Status: active. This ledger supersedes the protocol-change direction in
`akita-metal-e2e-structural-5x-goal.md` for the current campaign. Major protocol
changes, including a permutation argument or a new committed-address
representation, are out of scope.

## Objective

Reach a verified complete-prover speedup of at least 5x over the frozen optimized
CPU prover for BTreeMap, Fibonacci, and SHA-2 chain at `T = 2^28`, while preserving
useful performance at lower trace sizes. Prover/verifier equations, transcript
order, proof layout, soundness assumptions, and the Jolt statement remain unchanged.

The score is `min(cpu_prove_s / metal_prove_s)` over the three workloads. The
complete `jolt_prover::prove` boundary is authoritative; component spans diagnose a
result but cannot reject an otherwise valid complete-proof win.

## Frozen references and evidence warning

The accepted runtime ancestors are Jolt
`6ec86d08a77d2210676c4f299d55cf7f0ab46892` and Akita
`8291c2dbcd75f413e9697b7cb7ff89942a0c9005`. Jolt HEAD
`5cd8417ddd24c5597853dd78ba10c436d7394cf8` adds documentation and an independent
benchmark-harness fix. Preserve the local Cargo path overrides and all unrelated
worktree changes.

P0 rebuilt the accepted source as binary
`17d35ffa4c78c7abb057aae5df43b8ca72498fd6bc609ced0015a2f562f94975`.
Its first BTreeMap run completed in 46.53 seconds at 82.12 GiB RSS. Subsequent
Fibonacci and SHA-2 observations were 41.31 and 47.60 seconds, but root command
time per Metal work unit increased by roughly 70--80% in run order. P0 therefore
established the evaluator and exposed an ordering confound; it did not freeze a
comparative three-workload reference.

The currently retained P1+C1a+C1b2 source is binary
`bf8f6b04e57a278c075cc618492151f51841e3b88db38c092c13b7b7c35ebf7b`.
Every observation below produced an exact verified proof. BTreeMap's 46.08-second
candidate result ran before the reversed matrix; the 58.78-second observation ran
third after SHA-2 and Fibonacci. The spread is part of the result, not a source
change.

| Workload | Frozen CPU | 5x ceiling | Retained-source Metal | Speedup | Remaining at best observation |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 166.548 s | 33.310 s | 46.08 s first / 58.78 s third | 3.614x / 2.833x | 12.770 s |
| Fibonacci | 215.177 s | 43.035 s | 38.27 s second | 5.623x | 4.765 s headroom |
| SHA-2 chain | 213.703 s | 42.741 s | 47.21 s first | 4.527x | 4.469 s |

This is a diagnostic reverse-order Metal matrix, not final acceptance. The source
is retained because its candidate-local comparisons passed their registered gates,
but the campaign cannot claim a stable speedup until paired CPU/Metal runs in both
orders separate code gains from GPU frequency, queue, temperature, and residency.

## Phase ledger at T28

Each budget is the corresponding optimized CPU phase divided by five. The measured
side below is the retained binary's SHA-2 -> Fibonacci -> BTreeMap diagnostic order.

| Workload | Commit / budget | PIOP / budget | Eval proof / budget |
|---|---:|---:|---:|
| BTreeMap | 21.792 / 13.153 s | 29.655 / 16.437 s | 7.304 / 3.347 s |
| Fibonacci | 16.607 / 22.464 s | 15.589 / 15.405 s | 6.055 / 4.770 s |
| SHA-2 chain | 19.531 / 21.565 s | 21.496 / 15.823 s | 6.136 / 4.975 s |

In the stressed BTreeMap run, phase excess is 8.639 seconds commitment, 13.218
seconds PIOP, and 3.957 seconds eval proof: approximately 34%, 51%, and 15% of the
phase-level miss. The earlier thermally favorable BTreeMap split was 14.149,
27.211, and 6.001 seconds, where PIOP represented about 75% of excess. Both views
matter: PIOP owns the reproducible algorithmic gap, while commitment and eval-proof
submission amplify badly under run-order pressure.

Fibonacci clears the complete 5x target because commitment has 5.857 seconds of
margin, despite PIOP exceeding its phase budget by 0.184 seconds and eval proof by
1.285 seconds. SHA-2's net phase miss is about 4.800 seconds: commitment is 2.034
seconds under budget, PIOP is 5.673 seconds over, and eval proof is 1.161 seconds
over. SHA-2 therefore needs PIOP work, not just more commitment tuning.

The current PIOP priorities are:

| Stage | BTreeMap | Fibonacci | SHA-2 | Decision |
|---|---:|---:|---:|---|
| Stage 1, outer remainder | 2.568 s | 2.353 s | 2.878 s | shared, but no longer the first target |
| Stage 2, Product/RAM | 7.122 s | 1.406 s | 3.048 s | P3 first for BTreeMap; preserve SHA transfer |
| Stage 4, register/RAM value | 6.749 s | 4.005 s | 5.649 s | largest shared PIOP attack surface |
| Stage 6b | 7.689 s | 3.874 s | 4.517 s | accelerator path is critical; BTreeMap host lane remains hidden |

Commit is almost entirely the packed root: inner commitment took 21.240 seconds
for BTreeMap, 16.195 for Fibonacci, and 19.107 for SHA-2. Metal command time was
21.191, 16.136, and 19.053 seconds respectively, so host setup is not the broad
limit. The three calls read about 2.586, 1.974, and 2.135 TB of setup-matrix data;
C2 must reduce arithmetic or traffic, not shave buffer construction.

Eval proof took 7.304, 6.055, and 6.136 seconds. Within it, opening-index build was
1.003, 0.881, and 1.008 seconds, while command submission-to-completion was 5.907,
4.464, and 4.636 seconds. E2 should target late compact index ownership and overlap
around that command; retuning the already-near-floor first fused fold remains
deprioritized.

## Evidence classes

- **Measured:** valid proof and complete-prover observation already exist.
- **Projected:** a smaller-scale or component measurement exists, but target-scale
  complete-prover transfer is unmeasured.
- **Modeled:** traffic, ownership, or dependency evidence identifies a candidate;
  no candidate implementation has been timed.
- **Accepted:** reconstructed on the current parent, revalidated under the evaluator
  below, and retained in source.

Estimates from overlapping ownership or scheduling changes are never added until a
composed treatment measures them.

## Fixed evaluator

### Correctness

- Exact finite-field parity against the independent CPU route on the smallest shape
  that exercises the changed arithmetic or lifecycle.
- `PROOF_VERIFIED backend=metal value=true` for every integrated sentinel.
- Qualified Metal routes must remain fail-closed; a fallback cannot count as a Metal
  result.
- No changes to protocol messages, claims, transcript, verifier, proof bytes, or
  soundness target.

### Performance and memory

- Candidate search uses a frozen rebuilt parent. Run one warm candidate observation
  on its worst relevant sentinel. Repeat only if the result lies within 0.25 seconds
  of its retention boundary or exposes a named noise/residency confound.
- Retain a valid complete-proof improvement of at least 0.20 seconds, or a
  performance-neutral simplification that frees at least 4 GiB needed by a later
  candidate. Charge setup, synchronization, CPU work, and displaced later-stage work.
- Peak RSS must remain at most 90 GiB, with zero process swaps and no system-swap
  growth.
- A global route must not regress the T25 guard by more than 3%. Size-dispatched
  routes may be workload-specific. At T20, select the faster complete backend rather
  than forcing a fixed Metal setup cost.
- Refresh the three-workload T28 matrix after two cumulative seconds are retained,
  after BTreeMap falls below 40 seconds, or when a route can change workload ordering.
- Final acceptance requires two order-reversed CPU/Metal pairs for all three T28
  workloads, relevant T20/T25 guards, exact verification, and memory guards.

### Logging

Record every candidate append-only in
`benchmark-runs/akita-metal-e2e-polish/events.jsonl` and its analysis in
`benchmark-runs/akita-metal-e2e-polish/analysis.md`. A line contains the parent and
candidate source digests, command, artifact digest, correctness, complete time,
component diagnostics, memory, and `keep | discard | inconclusive` verdict.

## Candidate queue

| ID | Component | Candidate | Evidence | Expected complete gain | Order |
|---|---|---|---|---:|---:|
| P0 | measurement | clean accepted rebuild, paired B/F/S references, command timestamps | required | none | 0 |
| P1 | PIOP lifecycle | dead Product/Instruction transition-state retirement | accepted | 1.07 s BTreeMap; -18 GiB live | 1 |
| P2a | PIOP storage | half-sized Outer state B | rejected | unsafe: state B physically needs `2T` fields | closed |
| P2b | PIOP storage | delete eager Outer zero-fill | rejected | -3.05 s BTreeMap; +4.76 GiB RSS | closed |
| C1a | commit | prewarm the largest root/successor matrix prefix once | accepted | 0.697 s local; 2.16 s observed complete | 3 |
| C1b | commit | remove repeated radix-8 work in the remaining successor | C1b2 accepted | 0.260 s local; 0.53 s complete | 4 |
| P3 | PIOP RAM | reconstruct compact RAM-record bootstrap S14 | rejected on paired proof gate | local preparation only | closed |
| C2 | commit | radix-26 C12 at T28 | rejected on both root gates | none retained | closed |
| P4 | PIOP bytecode | generalize the `log_K=13` Metal carrier to `log_K=14` | rejected: topology construction erased address win | closed | closed |
| P8 | PIOP outer remainder | in-place stream bind plus half-sized alternate state | accepted | 1.71--1.73 s outer member; 3.23--3.44 s Stage 1 | retained |
| P9 | PIOP Product/Instruction | in-place first bind plus strided second bind | rejected: slower GPU kernels and missed local wall gate | none retained; 3 GiB allocation reduction reverted | closed |
| P10 | PIOP Stage 4/5 scheduling | delay Instruction Read-RAF scatter until register round 4 | rejected by parent-candidate-parent bracket | no stable Stage 4+5 or complete gain | closed |
| P5 | PIOP rows | reconstruct compact terminal row layout S15 atop P1 | rejected: cold Stage-3/later residency persists | closed | closed |
| P6 | PIOP registers | producer-native, right-sized register RW frontier | closed: already retained; fresh Metal frontier rejected by bound | none beyond accepted source | closed |
| P7 | PIOP RAM | retain compact high-activity RAM owner through Stages 2/4/6 | closed: existing compact consumers reject T28 density; Stage-2 half already rejected | requires a new accelerator family | closed |
| E1 | eval proof | retire the 18.18-GiB coefficient index after commit | rejected: exact retirement regressed opening and complete proof | -18.18 GB allocation; no speed credit | closed |
| E2 | eval proof | late compact/private index plus pack/relation overlap | closed: both retained opportunities are already active; root rows depend on packing | no additional gain | closed |
| L1 | lower scale | C8 five-stream commitment at <=T25; delayed bytecode materialization | C8 accepted; bytecode previously rejected by S2 | 0.262 s retained root gain | 13 |

## Candidate cards

### P0: trustworthy baseline and scheduling observability

**Question.** What does the accepted source measure today, and which part of the
large wall-minus-GPU gaps is queue delay versus page commitment or host
post-processing?

**Minimal surface.** Rebuild the existing source. Add no optimization. If existing
Metal timestamps cannot distinguish submission, GPU start, GPU end, completion, and
host post-processing, add diagnostic fields at the shared command boundary before
P2/C1/P6.

**Decision.** Freeze the rebuilt artifacts and use them until source, evaluator,
compiler, machine, OS, or timing boundary changes. P0 is invalid if any qualified
route falls back or an artifact contains a rejected candidate.

### P1: dead transition-state retirement

**Exact boundary.** After Product copies its 4,096-element CPU tail and Instruction
reads its final two values, drop Product state A/B and Instruction state A/B. Retain
the original Stage-1 rows and opening workspaces. Product and Instruction openings,
claims, and all caller-visible values are unchanged.

**Measured evidence.** The prior valid S8 treatment retired exactly 18 GiB, reduced
peak RSS to 78.57 GiB, and moved traced BTreeMap from 47.388661 to 46.894453 seconds:
a 0.494208-second complete saving. Product opening remained source-residency-bound,
so P1 does not claim the original 1.921-second wall-minus-GPU gap.

**Current result.** The reconstructed candidate retired 12,884,901,888 Product
bytes and 6,442,450,944 Instruction bytes, verified at T25 and T28, and moved the
thermally comparable T28 BTreeMap sentinel from 46.53 to 45.46 seconds. Peak RSS
fell from 82.12 to 80.07 GiB and process swaps remained zero. The T25 guard was
6.60 seconds. P1 is retained; system-wide swap growth was not captured for this
observation and remains a final-validation check.

**Falsifiers.** A lifecycle test must reject transition access after retirement while
both openings retain exact CPU parity. The integrated proof must verify and report
the expected retired bytes. Discard if it adds more than 0.10 seconds on the rebuilt
parent; retain otherwise as a measured complete win and memory enabler.

### P2: Outer storage geometry and initialization — closed

P2a falsified the original geometry model. Materialization stores the `2T` entries
of the B polynomial in state A. After binding the stream variable there are `T`
logical rows, but each row contains both A and B fields, so state B still stores
`2T` physical fp128 fields. The deliberately red smaller-capacity test corrupted
the first transition and both independent field-oracle tests failed. Restoring the
`2T` allocation restored exact parity. There is no redundant 4-GiB state-B owner.

P2b switched only the production initialization policy from `Full` to `Lazy`; the
lazy path already had exact field-oracle coverage. It deleted the 17,185,247,408-byte
T28 blit and still produced an exact proof, but moved first-touch latency into the
protocol. Outer first-message wall time rose from 0.306469 to 1.050311 seconds,
first-bind wall time rose from 1.498082 to 1.780190 seconds, and Stage 1 rose from
4.049224 to 5.147635 seconds. The thermally ordered complete T28 sentinel regressed
from P1's 45.46 to 48.51 seconds and peak RSS rose from 80.07 to 84.83 GiB. T25
remained healthy at 6.33 seconds, but the target-scale regression closes the route.
Production eager initialization is restored.

### C1: commitment root-successor path

**Exact boundary.** The Akita root commitment output is unchanged. Replace the host
successor that reconstructs all fp128 outputs, decomposes them again, clones every
outer slice, and only then submits outer commitments. The caller observes identical
inner rows, outer commitments, evaluation hint, and proof.

**Measured split.** On the current full-initialization P1 parent, the T28 BTreeMap
root successor decomposes in 37.113 ms and materializes its four 360,710,144-byte
outer-slice inputs in 46.900 ms. The B-row backend boundary is 1.463000 seconds:
112.383 ms of preflight, 788.875 ms preparing a larger setup-matrix prefix, 20.933
ms of input/output buffer setup, 540.672 ms in the Metal command, and negligible
readback/reconstruction. At T25, matrix preparation is already a cache hit and costs
0.004 ms; the corresponding buffer and command costs are 8.780 and 112.629 ms.

The scale discontinuity is exact. T28 first prewarms the 2-GiB D512 root prefix, but
the D64 successor needs a larger prefix and lazily allocates a second resident
matrix. T25's root prefix already covers its D64 successor. C1a therefore prewarms
the larger of the two public prefixes once. It changes neither matrix coefficients
nor commitment arithmetic and should remove one preparation while reducing the
resident cache from two prefixes to one.

**Analysis gate before kernel code.** The measured 20.933-ms buffer setup falsifies
host zero-copy as the primary T28 target. Implement C1a first. Only then analyze C1b:
its remaining hard ceiling is the 540.672-ms command plus 112.383-ms preflight and
84.013 ms of decomposition/materialization. Any kernel change still requires a
traffic/compute floor and exact CPU parity before integration.

**C1a falsifier.** The regression test must prove that a larger D64 successor prefix
is resident after one prewarm and that the exact D64 Metal/CPU commitment oracle
still passes. At T28 the successor matrix-prepare span must fall below 1 ms, the
combined prewarm plus successor-matrix preparation must improve by at least 0.35
seconds, and the complete proof must improve by at least 0.20 seconds. T25 may not
regress by more than 3%. Otherwise restore the original D512-only prewarm.

**C1a result.** The new regression and exact D64 Metal/CPU oracle pass. At T28 the
successor matrix lookup fell from 788.875 ms to 4.291 us. Moving from the old
2-GiB root prefix plus larger successor prefix to one 2.6875-GiB prefix increased
the initial prewarm from 542.253 to 634.529 ms, so combined matrix preparation fell
from 1.331128 to 0.634533 seconds, a 0.696595-second local win. The complete B-row
boundary fell from 1.463000 to 0.660496 seconds. The proof verified in 46.61 seconds
at 80.77 GiB RSS versus the equivalently ordered 48.77-second diagnostic parent.
T25 also verified and moved from 6.35 to 6.21 seconds. Keep C1a; the observed 2.16
seconds is not all attributed to the 0.697-second mechanism because later GPU and
opening spans also varied.

**C1b1 gate: remove repeated small-digit work.** After C1a, the T28 root-successor
boundary is 660.496 ms: 112.402 ms of exact radix-8 range validation, 17.700 ms of
buffer setup, and 530.314 ms in the Metal command. The command evaluates exactly
46,171,922,432 digit/coefficient pairs. Its old inner loop calls the wide
accumulator `abs(digit)` times and repeats the work for wrapped quotient terms;
under the balanced radix-8 distribution this is about 138 billion accumulator
calls. A signed scaled accumulation computes the identical integer sum in roughly
60 billion vector scale-adds. Per-partial worst-case magnitude remains
`64 * 64 * 4 = 16,384`, so each 16-bit-limb accumulator stays below 2^30, within
the existing signed-32-bit bound.

The compulsory root-call traffic is about 12.6 GB: 10.75 GiB of logical matrix
reads across four vectors, 344 MiB of digits, and roughly 688 MiB of partial
writes plus reduction reads. Its favorable bandwidth floor is about 30 ms, far
below the observed 530 ms, so the optimization is compute-directed. The same
candidate builds each final slice once in parallel instead of cloning a reusable
buffer and retains exact range checking while scanning independent vectors in
parallel. Pre-register a root command at most 430 ms, exact negacyclic and quotient
parity, a complete T28 improvement of at least 0.20 seconds, and at most 3% T25
regression. Revert the C1b1 changes if any gate fails.

**C1b1 result.** Exact proof and component gates passed, but the complete gate
failed. T28 materialization fell from 49.191 to 21.615 ms, range validation from
112.402 to 29.818 ms, and the root command from 530.314 to 370.976 ms. The B-row
boundary improved by 237.303 ms and aggregate digit-row GPU time fell from 563.507
to 374.043 ms. Nevertheless, the complete proof regressed from 46.61 to 46.99
seconds. T25 moved only from 6.21 to 6.19 seconds while peak RSS rose from 16.78 to
19.31 GiB. Reject the bundled candidate and restore the exact C1a source; a local
kernel win is not sufficient to retain globally worse ownership/residency behavior.

**C1b2 gate.** Restore the original sequential slice materialization and keep only
the signed scale-add, parallel exact validation, and a 128-column partial. The new
red arithmetic test proves 128 is the maximal safe width: `128 * 64 * 4 * 65535`
fits signed 32-bit, while width 129 does not. Doubling the width halves partial
storage and reduction reads without changing the sum. Exact product/quotient and
C1a prewarm tests pass. Require T28 range validation at most 40 ms, root command at
most 390 ms, B-row boundary at most 430 ms, complete time at most 46.41 seconds,
and no more than 3% T25 regression. Otherwise restore C1a again.

**C1b2 result.** Every gate passed. At T28, exact validation took 29.888 ms,
the root command took 350.114 ms, and the B-row boundary took 400.875 ms. Relative
to C1a, the boundary improves by 259.621 ms and aggregate digit-row GPU time falls
from 563.507 to 366.317 ms. The proof verified in 46.08 seconds at 82.05 GiB RSS,
0.53 seconds faster than C1a. T25 verified in 6.12 seconds at 16.82 GiB, improving
from 6.21 seconds. Keep C1b2. Its partial-width change also reduces the reported
opening allocation by about 22 MiB at T28; it does not explain the run-to-run RSS
peak difference, which remains under the 90-GiB cap.

**Retained-source matrix.** The reverse-order SHA-2, Fibonacci, BTreeMap refresh
verified all three proofs at 47.21, 38.27, and 58.78 seconds with peak RSS of
82.97, 82.12, and 80.77 GiB. The source and release binary were unchanged between
runs. BTreeMap's packed-root command rose from 14.063 seconds of simultaneous CPU
work and 21.155 seconds of GPU work to a 21.191-second wall boundary; eval-proof
command wall was 5.907 seconds. Compared with the earlier 46.08-second BTreeMap
candidate observation, the 12.70-second complete spread is too large to interpret
as ordinary benchmark noise. It is a scheduling/thermal/residency falsifier for
single-order portfolio accounting, not a falsifier of C1b2's local parent/candidate
gate. Final scoring must use interleaved order-reversed CPU/Metal pairs and record
GPU frequency/temperature if the platform exposes them.

### C2: size-dispatched root arithmetic

C12 radix-26 reduced the T25 BTreeMap root from 1.473321 to 1.304687 seconds with an
exact verified proof, but missed its old 1.25-second component gate. Its proportional
T28 BTreeMap projection is about 11.08 seconds versus 12.51 seconds accepted; this is
not a T28 observation. Reconstruct it only after C1 so successor costs cannot mask
the root result. C8 five-stream batching is admitted only below T28 because it won at
T25 and lost at T28.

**C2 reconstruction gate.** Keep the accepted fp128 root for the middle geometries.
At root widths of at least 262,144 positions, use C12's exact five-digit balanced
radix-`2^26` representation. At widths of at most 65,536 positions, keep fp128
arithmetic but batch five 32-task streams per command as in C8. No other geometry
changes route. Radix conversion is setup-bound; canonical fp128 partials/output,
hybrid CPU ownership, the opening index, transcript, proof, and verifier are fixed.

Correctness must cover centered field round trips, normalization after 61/62/63/64
selected rows, both negacyclic signs, shifts 0/1/255/256/511, resident and streaming
sources, and the 512-block geometry against the independent CPU commitment. The
T28 BTreeMap treatment requires root GPU time <=11.4 seconds, >=0.8 seconds of root
saving against an adjacent frozen parent, paired complete-proof saving >=0.5 seconds,
exact verification, no fallback, RSS <=90 GiB, and no system swap growth. The T25
batching sentinel must save >=0.15 seconds in the root and keep the complete proof
within 3% of the parent. Use one T25 and one T28 treatment, with one T28 repeat only
for threshold ambiguity.

**C2 result.** The size dispatch and radix implementation passed 44/44 Akita Metal
tests and all three modular Akita muldiv variants. C8's T25 branch verified at 6.05
seconds and 16.82 GiB, with root GPU time 1.211820 seconds versus the fixed 1.473321
reference; swap stayed at 111.75 MiB. Retain this <=65,536-position branch.

The T28 radix branch verified at 46.74 seconds and 80.76 GiB versus its adjacent
47.48-second parent, but root GPU time fell only from 12.740696 to 12.440432 seconds.
It missed both the <=11.4-second absolute root gate and the >=0.8-second adjacent
root-saving gate. The complete saving cannot override those hard component
falsifiers. Reject without a repeat and remove radix conversion, cache, and shader
code; retain only C8's lower-scale command batching.

The selective rollback removed every radix symbol and restored the field and
prepared-setup modules exactly. The retained C1+C8 Akita diff is
`73b78a499dec5dcff1091f3266eae0ad687b5fe10c82254838c2b0ec09df17c0`.
The four focused batching/parity tests and the full 42-test Akita Metal suite pass;
`git diff --check` is clean. The release evaluator still contains the rejected
radix candidate and is non-scorable until rebuilt.

### P4: generalize the Bytecode Read-RAF address carrier to `log_K=14`

SHA-2's T28 Bytecode Read-RAF address member falls back because the carrier is
hard-coded to 8,192 addresses. Its CPU route takes 1.453351 seconds. The same
address-major route takes 0.205474 seconds for BTreeMap at 8,192 addresses. The
carrier shape, Metal parameters, and shader are already runtime-sized; the fixed
domain remains in host admission, topology sizing/construction, and provenance
checks. A `u16` address still represents every `log_K=14` index.

Support exactly `log_K in {13, 14}` and preserve the T28-only `2^26` trace cutoff.
Pass the selected address exponent through resident-row admission and both Stage-1
producer variants; size topology scratch, worklists, offsets, pushforward outputs,
and provenance from the receipt shape. Do not alter the relation, sumcheck rounds,
claims, transcript, proof, verifier, cycle member, or the accepted `log_K=13` route.

The red ratchet requires both domains to publish a fused Stage-1 topology; full
correctness requires independent optimized-CPU lockstep for both address domains,
including every round and output claim. At T25, SHA-2 must retain the CPU
`trace_cutoff` route and remain within 3% of the parent. At T28 require the fused
address-major route, 16,385 address offsets, zero additional source scans/uploads,
address-member wall time at most 0.45 seconds, Stage-1 witness preparation at most
7.50 seconds, at least 0.50 seconds adjacent complete-proof saving, exact
verification, RSS at most 90 GiB, and no swap growth. Use one T25 guard and one T28
treatment; add closing-parent measurement only if the candidate passes its local
gates.

**P4 result.** Dual-domain exactness and the lower-scale guard passed. The repaired
T28 candidate verified in 39.75 seconds at 80.81 GiB with no swap growth, versus
38.45 seconds and 78.32 GiB for the adjacent frozen parent. Generalizing the
address phase itself worked: it fell from 1.468624 to 0.374446 seconds and published
16,385 offsets with no additional source scan or upload. Constructing the doubled
Stage-1 topology did not: witness preparation rose from 7.363953 to 9.590259
seconds. The 2.226306-second construction regression exceeds the 1.094178-second
address saving, producing a 1.132128-second regression at the affected boundary and
a 1.30-second complete-proof regression. This fails both the Stage-1 and complete
gates without threshold ambiguity. Reject P4 and restore the `log_K=13`-only parent.
The reusable result is narrower: a `log_K=14` fused address consumer is fast enough,
but materializing its 97,836,573 descriptors and 12,232,248 work items during
Stage 1 is not.

### P3/P5: reconstruct valid whole-proof row-lifecycle wins

S14's compact RAM bootstrap produced two valid T28 BTreeMap proofs at 45.928440 and
45.521475 seconds. S15's compact terminal rows produced a valid 45.342650-second
proof, despite displacing 0.810 seconds into Stage 3. These are hypotheses for the
new parent, not accepted source. Reconstruct P3 first; rebuild P5 only atop P1 and
measure the complete proof so ownership overlap is charged once.

**P3 reconstruction gate.** Retain the Stage-1 producer's cycle-ordered 24-byte
access-record chunks and consume them directly, without S14's temporary contiguous
flatten. Stage 2 writes only address/cycle `block, previous, next` bootstrap
metadata. Initial values, read indicators, Hamming values, and increments are
synthesized from that metadata for round 0; the first bind materializes the existing
fp128 frontier ABI and retires the record chunks plus the two temporary cycle-u64
planes. Every later round and public observation is unchanged.

At the frozen 65,195,206-access BTreeMap shape, the producer input is 1.565 GB. The
minimal preparation writes 1.304 GB of address metadata and 1.304 GB of cycle
metadata, for 4.173 GB compulsory read/write traffic. At the measured 412.5 GiB/s
copy rate its favorable traffic floor is about 9.4 ms; preparation is therefore a
host census/allocation/first-touch problem, not a Metal arithmetic kernel. Round 0
performs the same field work as the accepted route and previously measured
68.5--70.9 ms GPU-active, so a reconstructed route above 75 ms GPU-active falsifies
the no-added-device-work claim. There is no protocol or soundness delta.

The red test must require a record-bootstrap route and compare every round and
output claim with optimized CPU, including hot segments and a chunk boundary. The
route must consume chunk slices directly and report no flat copy. At T25 require an
exact proof, preparation plus rounds 0--1 at most 0.15 seconds, and no more than 3%
complete regression. At T28 require exact verification, preparation at most 1.10
seconds, round-0 GPU-active at most 75 ms, at least 0.20 seconds paired complete
saving, RSS at most 90 GiB, and no swap growth. Because same-binary BTreeMap varied
by 12.70 seconds, preserve the current parent binary and use baseline-candidate-
baseline order when the first candidate observation is promising. The manual
experiment budget is one T25 guard and at most three T28 runs.

**P3 result.** The red route assertion failed on the parent and the three exact
CPU/Metal lockstep tests passed after implementation, including cycle handoff and a
hot-address test whose 4,093-record producer chunks cross the 4,096-record Metal
boundary. The production route reported chunk-native input and no flat copy. At
T25 it verified in 6.19 seconds at 16.69 GiB; preparation plus rounds 0--1 was
95.826 ms, below the 150-ms guard.

At T28 the final hybrid candidate retained only the increment fp128 plane on the
host and synthesized address values, read indicators, and Hamming values on Metal.
It reduced preparation from the surrounding parents' 1.239--1.393 seconds to
0.492 seconds, kept round-0 GPU-active at 74.477 ms, verified exactly, and used
82.78 GiB. The complete proof was 46.07 seconds. Its immediately following frozen
parent was 46.18 seconds, only a 0.11-second win; the 45.15/46.18-second surrounding
parent average was 45.665 seconds, a 0.405-second regression. This misses the
predeclared 0.20-second paired complete-proof gate. The earlier raw-cycle variants
also failed at least one hard gate: 47.03 seconds with 76.682-ms round 0, then
44.82 seconds with 77.672-ms round 0. Reject P3 and restore the P1+C1a+C1b2 source.
The 0.75--0.90-second local preparation win is reusable evidence, but not retained
code. System-wide swap growth was not measured.

**P5 reconstruction gate.** Rebuild S15's semantic 56/104-byte Stage-1 split on
the retained P1 parent: the existing 48-byte `InstructionInputRow`, one 8-byte
`left_lookup_operand` plane, and a 104-byte Outer-only residual. Total producer
traffic remains 160 bytes per row. Product scans only the compact row, derives its
seven available columns exactly, and recovers the omitted lookup-output evaluation
from its final left factor when the relevant Lagrange coefficient is nonzero.
Instruction scans only the left-lookup plane and recovers the right-lookup
evaluation from its final combined claim when `gamma` is nonzero. Either zero
denominator selects the unchanged full-opening path.

P1 must retire the Product and Instruction transition buffers before their terminal
openings. The reconstruction adds no heap, primer, copy, early Stage-3 write, or new
lifetime controller; the existing exact Outer-residual release receipt remains the
only handoff into the 24-GiB Stage-3 arena. This tests whether P1's 18-GiB live-state
reduction removes S15's former residency displacement.

Require independent row/source parity, both quotient identities, both zero-
denominator fallbacks, exact terminal output parity, and one verified lower-scale
proof. At T25 require no more than 3% complete regression. At T28 require Product
plus Instruction output at most 0.55 seconds, `MetalInstructionInput::first_bind`
at most 0.30 seconds, Stage 3 no more than 0.20 seconds slower than the adjacent
parent, at least 0.50 seconds paired complete-proof saving, RSS at most 90 GiB, no
swap growth, qualified Metal routes, and exact verification. Freeze the rebuilt
parent before editing; use parent-candidate-parent order only if the first candidate
passes its local gates. One T25 treatment and at most two T28 candidate observations
are the fixed budget.

**P5 result.** The composed source passed 7/7 focused Metal tests, all three
modular Akita proofs, clippy, and the lower-scale guard. T25 verified in 6.23
seconds versus the retained 6.05-second reference (+2.98%), with a 4.168-ms
compact/sparse terminal boundary and 11.786-ms Stage-3 first bind.

At T28 the adjacent parent/candidate pair verified at 45.582599/48.858924 seconds
and both used 80.78 GiB with no swap growth. P5 reduced Product plus Instruction
terminal output from 1.599638 to 0.611466 seconds, a 0.988172-second saving, but
missed the 0.55-second local ceiling. The former residency failure remained:
Instruction first bind rose from 0.091291 to 0.900390 seconds and Stage 3 rose from
0.908315 to 1.863434 seconds. Stage 4 and Stage 6b then regressed another 1.875392
and 1.756178 seconds. P1's 18-GiB transition-state retirement is real but does not
keep the unscanned residual destination resident. Reject without a repeat and
restore P1+C1a+C1b2. The terminal quotient algebra remains correct evidence; this
owner layout is closed unless the Stage-3 destination itself is eliminated.

### P6/P7: change producers, not storage flags

The prior private register route failed because it duplicated host construction,
faulted fresh pages, and contended with Stage 5. P6 must generate the final register
frontier directly, retain only current/next bind state, and schedule outside that
contention window. P7 must retain compact Stage-1 RAM records as the shared owner
rather than add another dense owner. Merely changing `StorageMode` or adding a page
primer is out of scope.

**P6 result.** Close without code. The retained register producer already emits
cycle-ordered, chunked 40-byte `IndexedSparseEntry` rows together with the
rs1/rs2/rd/increment planes in one witness pass. It keeps the chunks through the
early binds, allocates only each exact next frontier, and defers conversion to
field-valued entries until the frontier is small. There is no full concatenated
register owner left for a new producer-native route to delete.

The only remaining interpretation of P6 is the previously measured R1 fresh Metal
frontier. R1 increased preparation by 1.418987 seconds and exposed 3.447497 seconds
of wall time beyond reported GPU work. Granting the physically impossible case that
both costs vanish entirely still puts Stage 4+5 at about 6.04 seconds, only a
1.85-second saving against the then-accepted parent. That misses the registered
2.30-second Stage-4 admission gate by about 0.45 seconds before charging compulsory
frontier writes. The current retained T28 parent still spends 8.119659 seconds in
Stages 4+5, so no intervening retained change invalidates the mechanism-level
falsifier. Rebuilding R1 would repeat a closed experiment.

**P7 result.** Close the storage-only card without code. The retained T28 BTreeMap
trace has 65,195,206 RAM accesses. Its Stage-1 tape correctly discards compact
records above the 262,144-access cap, and every existing compact consumer would
still reject this shape if that cap were lifted: the RA-claim address term alone is
1,238,708,914 products, the first Hamming parent level has at least 32,597,603
nodes, and RA virtualization has at least one product per access. These are 32--1,239
times the one-million-product admission caps before later rounds are counted.

Retaining the 24-byte records plus only the leaf-cycle and first-merge topology
would add at least 2,086,246,592 bytes; most topology levels and member frontiers are
not included in that floor. P3 and S12d already tested the Stage-2 half of this
ownership change. S12d saved 3.121 seconds in Stage 2 but regressed Stage 4 by 0.285
seconds and Stage 6b by 0.489 seconds while the record/destination pages displaced
later owners; P3's later chunk-native reconstruction saved only 0.11 seconds against
its adjacent parent and failed its paired proof gate. Extending the same record
lifetime without replacing the later dense algorithms adds residency and selects no
new route. A viable successor is a new accelerated high-activity RAM RA/Hamming/value
kernel family, not the low-hanging shared-owner change registered as P7.

### E1/E2: eval-proof ownership and overlap

The 18.18-GiB coefficient index lives from commitment through PIOP and is then read
cold. The prior fused-deferred O2 route verified and freed it but saved only
0.09--0.25 seconds. Retain E1 under the memory-enabler rule if it does not regress the
rebuilt parent. E2 then constructs the compact index late and overlaps coefficient
packing with independent relation-vector and opening-row preparation. Do not retune
the first fused decompose/fold kernel: 1.694 seconds measured is already close to its
approximately 1.58-second modeled floor.

**E1 result.** The exact prior O2 patch replayed cleanly on P1+C1a+C1b2+C8. All 42
Akita Metal tests and all 17 modular Akita tests passed. At T25, where the retained
index route remains active, the candidate verified in 6.56 seconds versus an adjacent
6.48-second parent (+1.23%), with 16.82 versus 16.81 GiB and no swap growth.

At T28 the candidate removed exactly 18,182,307,840 opening-index bytes and reduced
peak RSS from 80.00 to 78.08 GiB. It nevertheless increased opening command wall
from 4.618329 to 5.237437 seconds and opening GPU time from 2.811195 to 3.522988
seconds. Complete proof time rose from 49.48 to 56.18 seconds. Unrelated commit and
PIOP variation made the complete delta unusually large, but the candidate-local
opening regression independently fails the memory-enabler rule. Reject without a
repeat and restore the exact C1/C8 Akita diff. E2 may not assume E1's fused consumer;
it must overlap the accepted indexed route or prove a different critical-path
deletion.

**E2 result.** The retained source already constructs the private coefficient index
late, inside root coefficient packing, and releases it after the one indexed
consumer. It also already runs Stage-2 opening-term preparation and the complete
static Metal relation session concurrently with Stage 1. In the adjacent E1 parent
trace, that worker ran for 0.496004 seconds underneath the 0.551343-second Stage-1
sumcheck and left only a 14-microsecond join.

No second independent root lane exists. Root coefficient packing took 1.247529
seconds. The work immediately after it was 0.002675 seconds of relation-input setup,
followed by 0.434936 seconds of opening-row work: 0.052710 seconds materializing the
packing D input, 0.381185 seconds computing `v = D * e_hat`, and 0.001010 seconds of
compression. Each material step consumes the packing output. The independent
0.367761-second NTT prewarm is already fully hidden under the 1.826492-second fold
grind, and the static E/T prefix also consumes `e_hat`. Later relation-weight
compilation depends on challenges sampled after the next-witness commitment.

The proposed 0.7--1.4-second overlap window therefore double-counted retained work
and violated the measured dependency graph. Close E2 without code. A successor must
delete a serialized packing or fold service; moving existing preparation between
these boundaries has no critical-path credit.

## Lower-scale policy

At T20 the current Metal backend is only about 1.08--1.11x faster than optimized CPU;
fixed setup and large-owner initialization dominate. The production backend already
uses relation-level trace cutoffs, and trace commitment/opening fail closed to CPU
outside their qualified shapes. At T25, retain C8's five-stream root batching.
Every T28-only memory-owner path must remain size-gated.

**L1 result.** C8 is the only retained lower-scale code change. Its exact T25 proof
completed in 6.05 seconds with a 1.211820-second root, saving 0.261501 seconds from
the fixed root reference. The proposed delayed Bytecode Read-RAF materialization is
S2, already measured and rejected: it moved the dense transition from width 2 to
width 8, raised the member from 1.244350 to 1.516519 seconds, raised Stage 6b from
4.994482 to 5.316562 seconds, and did not lower peak RSS. Do not reopen it without
eliminating factor arithmetic or physical first touches.

Fresh frozen-parent T16 sentinels verified for all three main workloads. Rounded
optimized/Metal prover times were 0.53/0.45 seconds for Fibonacci, 0.48/0.45 for
SHA-2, and 0.31/0.29 for BTreeMap (whose realized trace domain was T15). Trace
commitment and opening correctly reported `qualified=false` and used CPU. Combined
with the existing T20 pairs--1.73/1.60, 1.83/1.69, and 1.37/1.23 seconds--there is no
measured lower-scale regression to justify a new whole-backend cutoff. These tiny
sentinels cover the prover window, not one-shot Metal library initialization; a
caller optimizing startup latency should still select the optimized CPU backend.

The final accepted-source rebuild completed in 8 minutes 9 seconds and reproduced
the trusted parent bit-for-bit: `target/release/examples/modular_benchmark` and
`/private/tmp/modular_benchmark-p1-c1-c8-accepted-rebuild` both have SHA-256
`e3a793a9cf4d77530939a74d03891a67953813579032b7710019a415d3eea3f8`.
The previously stale executable containing rejected E1 code is no longer present.

## Portfolio accounting

P1, C1a, C1b2, and lower-scale C8 are composed and retained. P3, P5, C2, E1, E2,
and delayed Bytecode Read-RAF materialization
are closed and contribute no portfolio credit. Against BTreeMap's thermally
favorable 12.770-second gap, the remaining protocol-preserving queue no longer
crosses 5x on projected work alone; it needs a new multi-second mechanism in Stage
4, a new high-activity RAM accelerator family, or another serialized root service.
The 58.78-second reverse-order result adds a separate stability requirement that
cannot be paid down by summing candidate wins.

Fibonacci currently has 4.765 seconds of complete headroom. SHA-2 misses by 4.469
seconds, predominantly in PIOP, so P4 and shared Stage-4 work must create margin.
BTreeMap remains the deciding algorithmic workload; run-order stability is the
deciding measurement workload.

## Closed or deprioritized work

- No permutation argument, new memory argument, committed-address representation,
  transcript reordering, or other major protocol change.
- Do not retry root carry/sign/RNS variants already rejected without a new floor.
- Do not optimize BTreeMap's Stage-6b host lane; it is hidden behind the accelerator.
- Do not optimize root-row generation; it is overlapped.
- Do not flip `StorageModePrivate` or add page primers as standalone candidates.
- Do not grind the first eval-proof decompose/fold kernel without a revised floor.

## Completion

Completion is two order-reversed valid CPU/Metal T28 pairs with a worst-case speedup
of at least 5x for each workload, qualified routes, exact proof verification, peak RSS
at most 90 GiB, no swap growth, lower-scale crossover guards, focused tests, required
nextest suites, formatting, and both required clippy modes. Component wins alone do
not complete the campaign.
