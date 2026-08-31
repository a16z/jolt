# Akita PIOP kernel scoreboard

Date: 2026-08-26

## Fixed workload and measurement boundary

This scoreboard fixes the workload at BTreeMap with `T = 2^28` and the
150,000,000 target-trace-size input. The realized trace has 253,779,321 populated
rows, 65,195,206 RAM accesses, and `RAM log_K = 19`.

The comparison is one verified traced optimized-CPU reference against one verified
trace from the retained P1+C1+C8 Metal source. It is suitable for localization and
attack ordering, not a final performance claim: the runs were not interleaved or
repeated.

| Arm | Trace | SHA-256 | PIOP wall |
|---|---|---|---:|
| Optimized CPU | `benchmark-runs/perfetto_traces/akita_btreemap_28_optimized.json` | `4d4c7e9a13d8f52e3efb7884ff242883cb5b436ca72a7817a17c4f5e5cd51cb5` | 82.182811 s |
| Metal | `/private/tmp/akita-btreemap-28-metal-e1-parent-a.json` | `dfaffbcdc63b34ca4d88439f771e562cd5f791cc7024d31bae1937d68337a061` | 29.564634 s |

The PIOP is therefore **2.780x**. A componentwise 5x target allows 16.436562
seconds, leaving a 13.128072-second PIOP deficit.

## Current-source refresh after K001

The goal-mode refresh on 2026-08-27 uses the promoted K001 source and a verified
BTreeMap T28 Metal trace (`cc8b7e3a5c1af8e6`). Its proof completed in 46.29 s at
82.60 GiB peak RSS with no swap. The CPU column remains the frozen historical
optimized trace (`4d4c7e9a13d8f52e`) and is used only to choose the next adapter;
every promotion still requires fresh, isolated, current-source CPU and Metal arms.

| Logical kernel/member | CPU localization | Current Metal | Speedup | 5x gap |
|---|---:|---:|---:|---:|
| RamReadWriteChecking | 3.574 s | 5.138 s | 0.695x | +4.424 s |
| RegistersReadWriteChecking | 2.917 s | 4.861 s | 0.600x | +4.278 s |
| RamRaVirtualization | 1.941 s | 2.785 s | 0.697x | +2.396 s |
| ProductRemainder | 1.561 s | 2.023 s | 0.772x | +1.710 s |
| RamRafEvaluation | 0.827 s | 1.045 s | 0.792x | +0.879 s |
| RamValCheck | 1.006 s | 1.077 s | 0.934x | +0.876 s |
| BytecodeReadRafCycle | 4.472 s | 1.552 s | 2.883x | +0.657 s |
| RamHammingBooleanity | 0.591 s | 0.537 s | 1.100x | +0.419 s |
| SpartanShift | 0.491 s | 0.391 s | 1.255x | +0.293 s |
| RamRaClaimReduction | 0.098 s | 0.150 s | 0.650x | +0.131 s |
| HammingWeightClaimReduction | 2.316 s | 0.575 s | 4.029x | +0.112 s |
| RegistersClaimReduction | 0.439 s | 0.142 s | 3.098x | +0.054 s |
| RegistersValEvaluation | 1.386 s | 0.311 s | 4.459x | +0.034 s |
| BytecodeReadRafAddressPhase | 0.806 s | 0.160 s | 5.046x | -0.001 s |
| OuterRemainder | 3.760 s | 0.750 s | 5.012x | -0.002 s |
| SpartanProductUniskip | 0.452 s | 0.080 s | 5.647x | -0.010 s |
| Booleanity | 15.436 s | 3.051 s | 5.059x | -0.036 s |
| InstructionInput | 2.766 s | 0.384 s | 7.202x | -0.169 s |
| SpartanOuterUniskip | 8.955 s | 1.577 s | 5.679x | -0.214 s |
| InstructionClaimReduction | 1.401 s | 0.046 s | 30.150x | -0.234 s |
| BooleanityAddressPhase | 5.446 s | 0.509 s | 10.691x | -0.580 s |
| InstructionRaVirtualization | 7.759 s | 0.925 s | 8.388x | -0.627 s |
| InstructionReadRaf | 13.428 s | 2.045 s | 6.565x | -0.640 s |

RamReadWriteChecking is K002. Its 5.138-second Metal localization splits into
3.520 seconds of prepare and 1.619 seconds of rounds, versus 0.589 and 2.984
seconds on the historical optimized CPU trace. The matched evaluator must charge
both parts and resolve whether production residency inflates prepare before any
kernel candidate is admitted.

## Sequential stage walls

Positive `5x gap` means the Metal wall is above `CPU / 5`; negative means the
stage already has headroom. Stage walls are sequential and do not double-count.

| Stage | CPU | Metal | Speedup | 5x gap |
|---|---:|---:|---:|---:|
| **PIOP total** | **82.183 s** | **29.565 s** | **2.780x** | **+13.128 s** |
| Stage 1 | 12.873 s | 5.682 s | 2.266x | +3.107 s |
| Stage 2 | 8.006 s | 8.496 s | 0.942x | +6.894 s |
| Stage 3 | 3.697 s | 0.917 s | 4.033x | +0.177 s |
| Stage 4 | 3.923 s | 5.644 s | 0.695x | +4.859 s |
| Stage 5 | 14.914 s | 2.512 s | 5.937x | -0.471 s |
| Stage 6a | 6.252 s | 0.683 s | 9.152x | -0.567 s |
| Stage 6b | 30.201 s | 5.085 s | 5.939x | -0.955 s |
| Stage 7 | 2.317 s | 0.547 s | 4.238x | +0.083 s |

## Logical kernel/member scoreboard

A member time is the sum of its outermost `prepare`, `prove_round`,
`finish_rounds`, and `output_claims` spans. Uni-skip rows combine `prepare` and
`first_round_poly`. The CPU and Metal traces have identical member call counts and
round counts. Outermost-only accounting avoids the optimized backend's nested
same-name uni-skip spans.

These are inclusive logical-kernel times. Members can overlap, most materially in
Stage 6b, so neither member walls nor member `5x gap` values are additive. The
`Route` column records the selected Metal-backend implementation using the route
events and relation-specific spans; `CPU host` is an intentionally host-only
member, while `CPU fallback` is a relation whose accelerated or sparse route
resolved to the optimized CPU implementation for this shape.

Sorted by current Metal wall:

| Stage | Logical kernel/member | Route | CPU | Metal | Speedup | 5x gap |
|---|---|---|---:|---:|---:|---:|
| 4 | RegistersReadWriteChecking | CPU host | 2.917 s | 4.566 s | 0.639x | +3.983 s |
| 2 | RamReadWriteChecking | Metal | 3.574 s | 4.527 s | 0.789x | +3.812 s |
| 1 | SpartanOuterUniskip | Metal | 8.955 s | 3.080 s | 2.907x | +1.289 s |
| 6b | Booleanity | Metal | 15.436 s | 2.802 s | 5.508x | -0.285 s |
| 2 | ProductRemainder | Metal | 1.561 s | 2.637 s | 0.592x | +2.324 s |
| 1 | OuterRemainder | Metal | 3.760 s | 2.601 s | 1.446x | +1.849 s |
| 6b | RamRaVirtualization | CPU fallback | 1.941 s | 2.185 s | 0.889x | +1.797 s |
| 5 | InstructionReadRaf | Metal | 13.428 s | 2.062 s | 6.513x | -0.624 s |
| 6b | BytecodeReadRafCycle | Metal | 4.472 s | 1.290 s | 3.468x | +0.395 s |
| 2 | RamRafEvaluation | CPU fallback | 0.827 s | 1.190 s | 0.695x | +1.024 s |
| 4 | RamValCheck | CPU fallback | 1.006 s | 1.077 s | 0.934x | +0.876 s |
| 6b | InstructionRaVirtualization | Metal | 7.759 s | 0.904 s | 8.580x | -0.647 s |
| 6b | RamHammingBooleanity | CPU fallback | 0.591 s | 0.551 s | 1.071x | +0.433 s |
| 7 | HammingWeightClaimReduction | Metal | 2.316 s | 0.546 s | 4.243x | +0.083 s |
| 6a | BooleanityAddressPhase | Metal | 5.446 s | 0.506 s | 10.758x | -0.583 s |
| 3 | SpartanShift | Metal | 0.491 s | 0.446 s | 1.099x | +0.348 s |
| 5 | RegistersValEvaluation | Metal | 1.386 s | 0.311 s | 4.460x | +0.034 s |
| 3 | InstructionInput | Metal | 2.766 s | 0.311 s | 8.909x | -0.243 s |
| 6a | BytecodeReadRafAddressPhase | Metal | 0.806 s | 0.177 s | 4.561x | +0.016 s |
| 3 | RegistersClaimReduction | hybrid | 0.439 s | 0.159 s | 2.762x | +0.071 s |
| 5 | RamRaClaimReduction | CPU fallback | 0.098 s | 0.137 s | 0.715x | +0.117 s |
| 2 | SpartanProductUniskip | Metal | 0.452 s | 0.079 s | 5.692x | -0.011 s |
| 2 | InstructionClaimReduction | Metal | 1.401 s | 0.047 s | 29.895x | -0.233 s |
| 2 | RamOutputCheck | CPU host | 0.011 s | 0.014 s | 0.840x | +0.011 s |

The inactive optional members are included below for completeness. BTreeMap uses
neither advice nor a committed program in this run, so their sub-microsecond ratios
are instrumentation noise rather than optimization signals.

| Stage | Inactive member | CPU | Metal | Speedup |
|---|---|---:|---:|---:|
| 6b | BytecodeReductionCyclePhase | 0.376 us | 0.291 us | 1.292x |
| 6b | ProgramImageReductionCyclePhase | 0.459 us | 0.333 us | 1.378x |
| 6b | TrustedAdviceCyclePhase | 0.209 us | 0.250 us | 0.836x |
| 6b | UntrustedAdviceCyclePhase | 0.291 us | 0.417 us | 0.698x |
| 7 | BytecodeReductionAddressPhase | 0.500 us | 0.208 us | 2.404x |
| 7 | ProgramImageReductionAddressPhase | 0.375 us | 0.208 us | 1.803x |
| 7 | TrustedAdviceAddressPhase | 2.417 us | 0.250 us | 9.668x |
| 7 | UntrustedAdviceAddressPhase | 0.500 us | 0.292 us | 1.712x |

## Initial attack order

1. **Stage 2:** RamReadWriteChecking, ProductRemainder, then RamRafEvaluation.
   Together they occupy essentially the entire 8.50-second stage, while
   InstructionClaimReduction and SpartanProductUniskip already exceed 5x.
2. **Stage 4:** RegistersReadWriteChecking, then RamValCheck. Both still use the
   optimized CPU path, and the complete stage is slower than the CPU reference.
   Attribution must separate kernel work from the Stage-5 compatibility prefetch
   that overlaps this boundary.
3. **Stage 1:** SpartanOuterUniskip and OuterRemainder. Both use Metal but together
   deliver only 2.27x at the stage wall.
4. **Small residuals:** SpartanShift, HammingWeightClaimReduction, and the remaining
   BytecodeReadRaf address/cycle gap only after the first three stages move.

Do not rank Stage-6b host members by their inclusive walls alone. The stage's host
and accelerator lanes overlap and already produce 5.94x; a local member win has no
end-to-end value unless it shortens the 5.085-second stage critical path.

## Attack A1: overlap high-activity RAM sequence preparation

The first implementation attack is scheduling-only. At BTreeMap T28,
`RamReadWriteChecking::prepare` occupies 3.018553 seconds. Its existing internal
timer attributes 1.184465 seconds to construction of the exact
`RamReadWriteSequence`, application of final memory, and construction of
`Val_init`; 1.833964 seconds elapse before that timer. The RAM address/value
columns and activity certificate are already published by asynchronous witness
preparation before PIOP, so this is not a second witness scan.

The candidate starts the unchanged, transcript-independent sequence construction
immediately after `MetalSpartanStage1::source_primer_join`. In the fixed trace that
leaves 3.328558 seconds of Stage 1 before the Stage-2 boundary. The worker owns the
RAM value columns while it runs and returns them with the prepared sequence;
Stage-2 prepare joins it, restores the columns for their Stage-4 consumer, checks
the exact `(log_T, log_K)` provenance, and only then adds the challenge-dependent
Gruen table and gamma. A failed or stale prefetch fails closed or uses the existing
synchronous path. No relation, round, claim, transcript event, variable order, or
verifier code changes.

The worker performs the same compulsory traffic as the synchronous route. A lower
bound counts one 1.0-GiB address pass for bucket planning, one 5.0-GiB scan of the
`u32/u64/u64` source columns, and 8.196 GB (7.632 GiB) of published address/cycle
state writes: at least 13.632 GiB, or 0.033 seconds at the measured 412.5-GiB/s
M4 Max bandwidth before counts, conversion, allocation, and page-fault costs. The
candidate does not claim to reduce that 1.184-second measured service; its ideal
visible floor is zero because the 3.329-second overlap window is larger. It can
instead lose through memory-bandwidth contention or earlier residency pressure.
The sequence adds at most 7.632 GiB while Stage-1 state is live, projecting the
80.08-GiB parent peak to 87.72 GiB before allocator overhead, below the 90-GiB
guard but close enough that RSS is a hard gate.

Focused parity must compare every round polynomial, terminal claim, and derived
table for prefetched and synchronous construction, and must prove that the RAM
value columns are restored exactly once. The lower-scale T25 BTreeMap sentinel may
regress by at most 3%; T24 and below must not select the prefetch. At T28, retain
only if the worker is complete before the Stage-2 join, combined Stage 1+2 wall
falls by at least 0.50 seconds from 14.178196 seconds, complete proof improves by
at least 0.50 seconds against an adjacent retained-parent observation, peak RSS is
at most 90 GiB with no swap growth, and the proof verifies. Any route mismatch,
worker panic, downstream value-column loss, or displaced cost in Stages 3--7
rejects the candidate.

**Result: rejected and exactly reverted.** Focused lockstep parity passed. The
T25 BTreeMap proof verified: the worker completed before Stage 2, joined in 12.6
microseconds, reduced RAM prepare from 122.100 ms to 0.069 ms, left Stage 1 flat
(0.321 to 0.315 seconds), and reduced the adjacent complete proof from 6.561 to
6.187 seconds at 16.83 GiB RSS.

At T28, the unchanged worker took 1.552437 seconds and completed 3.506 seconds
before Stage-2 prepare; its join was only 0.126 ms. RAM prepare fell from
3.018553 seconds to 0.000571 seconds and Stage 2 fell from 8.495620 to 6.427092
seconds. The cost moved into the overlap window: Stage 1 rose from 5.681693 to
7.397246 seconds, including OuterRemainder prepare rising from 0.345130 to
1.735682 seconds and its terminal opening rising from 0.810183 to 0.964429
seconds. Combined Stage 1+2 improved by only 0.353 seconds, below the 0.50-second
gate. More decisively, PIOP rose from 29.564634 to 31.564824 seconds and the
verified complete proof rose from 49.484252 to 52.049944 seconds. Peak RSS was
74.54 GiB, so capacity was not the failure. Starting the 8.196-GB sequence while
Stage-1 state is live is closed: it trades serialized setup for residency and
bandwidth contention rather than removing work.

## Attack A2: in-place outer stream bind and half alternate state

The next attack changes only the storage schedule of the first
`OuterRemainder` bind. Materialization leaves the B-only table in state A. For
each pair, the stream-bind shader reads exactly four consecutive fields and
writes the four interleaved A/B fields back to those same indices; pairs are
disjoint across invocations. The candidate therefore performs this bind in
place in state A. The following dense bind remains out of place and writes only
half as many fields, so state B's capacity falls from `2T` to `T` fields. All
round polynomials, Fiat-Shamir challenges, claims, openings, variable order,
and verifier behavior remain unchanged.

At BTreeMap T28, state A remains 8 GiB while state B falls from 8 GiB to 4 GiB.
Planned outer-remainder storage falls from 19,534,844,080 to 15,239,876,784
bytes, and eager initialization falls from 17,185,247,408 to 12,890,280,112
bytes. This removes one 4-GiB allocation and one 4-GiB zero-fill pass. The
stream-bind kernel's logical traffic is unchanged: it still reads the compact
rows and B state and writes `2T` fields. Its fixed trace, however, spends 1.376
seconds of wall time for 0.075 seconds of recorded GPU activity while first
touching the separate 8-GiB destination. The next dense round needs only the
4-GiB alternate prefix. Thus the candidate deletes half of that destination's
resident extent instead of merely moving its first touch.

The correctness gate is exact field-oracle parity for the first message, every
round polynomial, the CPU handoff, and all openings, plus an allocation test
that proves state B is exactly `T` fields and remains the largest size used by
later dense rounds. T25 may regress by at most 3%. At T28, retain only if the
proof verifies, `OuterRemainder` rounds improve by at least 0.40 seconds,
Stage 1 improves by at least 0.40 seconds, complete proof improves by at least
0.40 seconds against the frozen parent, and the saving is not displaced into
storage preparation, terminal openings, or a later PIOP stage. Any Metal alias
hazard, changed output, or first dense write beyond the half-sized buffer rejects
the candidate.

**Result: accepted.** The allocation test was observed red against the old
two-full-buffer geometry, then passed with state B at exactly `T` fields. All 12
focused outer-remainder tests pass, including the independent field oracle,
adapter parity for every round and opening, and the GPU-to-CPU handoff. The
release benchmark binary was
`d26389a24797a17f960a8f716d674ad21cf0240b1f75b11f481e78ed74f20866`.

The verified T25 screen improved complete proving from 6.561067 to 6.065117
seconds. Stage 1 fell from 0.321132 to 0.257377 seconds, outer-remainder rounds
from 0.050690 to 0.028028 seconds, first bind from 0.018063 to 0.008793 seconds,
and storage initialization from 0.066515 to 0.040549 seconds. Peak RSS was
15.21 GiB. The trace is
`/private/tmp/akita-btreemap-25-metal-a2-candidate.json`, SHA-256
`42cce21c8fa7d1496d0ec71191d7fcfb802023b2eb6c09e49700249776da99e7`.

Both verified T28 observations cleared every gate:

| Metric | Frozen parent | A2 run 1 | A2 replication |
|---|---:|---:|---:|
| Complete proof | 49.484252 s | 45.147468 s | 47.705414 s |
| PIOP | 29.564634 s | 25.917121 s | 28.602209 s |
| Stage 1 | 5.681693 s | 2.243617 s | 2.450343 s |
| OuterRemainder member | 2.601543 s | 0.874906 s | 0.896019 s |
| OuterRemainder rounds | 1.445474 s | 0.231309 s | 0.272188 s |
| First bind | 1.375882 s | 0.066487 s | 0.068612 s |
| Later dense rounds | 0.064840 s | 0.160243 s | 0.198855 s |
| Storage initialization | 1.759345 s | 0.832300 s | 0.898027 s |
| Peak RSS | about 80 GiB | 80.77 GiB | 81.41 GiB |

The expected first-dense-round shift is present, but it costs only 0.095--0.134
seconds against a 1.307--1.309-second first-bind saving. The smaller resident
workspace also removes the old 2.353-second Stage-1 source-primer join in both
runs; it becomes effectively free in run 1 and remains outside the critical
path in the replication. The T28 traces are
`/private/tmp/akita-btreemap-28-metal-a2-candidate.json` and
`/private/tmp/akita-btreemap-28-metal-a2-replication.json`, SHA-256
`dfdd4fa456e4460eb7aa4189594e3eb74a9cb8b2b2c0b6d5ad259d9537a612fb` and
`c9d660c32fe823a11083fd2d44b3223225810e91b6c753fc21dff0aee0f77b34`.
No work appears in a later outer phase, and the source change is retained.

## Attack A3: strided in-place first Product/Instruction bind

The retained ProductRemainder and InstructionClaimReduction sequences share one
command for their early rounds. Their first bind currently reads 12 GiB of
materialized state and first-writes 6 GiB of alternate state at T28. Across the
two A2 traces that joint command records only 48--51 ms of GPU activity but
occupies 0.867--1.252 seconds of ProductRemainder round wall.

The candidate keeps the first bound value in the first slot of its original
pair. Each four-source group is disjoint, so round 1 can bind in place without a
cross-invocation hazard. Round 2 reads those values with stride two and writes
the existing compact representation; round 3 and every later round use the
unchanged dense transition. Product state B therefore falls from `T` to `T/2`
fields and instruction state B from `T/2` to `T/4` fields. At T28 this removes
2 GiB and 1 GiB respectively. It also removes 2 GiB from Product's eager
workspace prime; Instruction's alternate state is lazy today.

No logical round traffic or field arithmetic is claimed away: the first two
rounds read and write the same number of field values. The mechanism deletes
3 GiB of cold destination extent and moves the remaining first write to round
2, where only a 3-GiB compact destination is needed. Messages, challenges,
claims, transcript order, CPU-tail format, terminal openings, and verifier code
remain unchanged.

The storage assertions must first fail against the current `T`/`T/2` alternate
capacities, then pass at `T/2`/`T/4`. Independent Product, Instruction, and joint
round-service tests must compare every intermediate message and resident table;
the special strided representation must never reach the CPU-tail or opening
boundary. T25 may regress by at most 3%. At T28, retain only if the first joint
bind and round-2 compaction together beat their adjacent A2 parent by at least
0.25 seconds, ProductRemainder round wall improves by at least 0.25 seconds,
complete proof improves by at least 0.25 seconds in a parent/candidate pair,
the proof verifies, and no cost is displaced into InstructionClaimReduction,
Stage 3, or terminal output.

**Result: rejected and exactly reverted.** The deliberately tightened storage
assertions were observed red against the old capacities, then passed at `T/2`
and `T/4`. All 9 Product tests and all 26 Instruction/joint-service tests passed,
including exact intermediate-message and resident-state parity. The candidate
also passed clippy. Its release binary was
`673f6367bb7fb7864605d68fcba9c69f976a93f4acef9c52b3e585b459c71d47`.

The verified T25 guard improved complete proving from 6.038282 to 5.956734
seconds and PIOP from 2.697776 to 2.684823 seconds. At T28, workspace priming
fell from 0.395517 to 0.294710 seconds and Product round wall fell from 0.939264
to 0.761432 seconds. The first two transition walls together fell from 0.907632
to 0.713775 seconds, a 0.193857-second gain that missed the 0.25-second local
gate. More importantly, their GPU-active time rose from 67.913 to 101.322 ms:
the writable in-place source and strided second read are slower kernels even
though deleting 3 GiB of cold destination extent reduces first-touch wall time.

The adjacent complete proof was also invalid as a promotion result: it regressed
from 44.197186 to 52.033234 seconds while PIOP rose from 24.407700 to 31.457606
seconds. The slowdown was system-wide rather than isolated to A3: Stages 1, 2,
and 4 rose by 0.858, 2.514, and 2.612 seconds, and Product terminal openings rose
from 1.144260 to 1.644271 seconds despite only a 4.744-ms GPU-active increase.
Peak RSS was unchanged at 80.78--80.79 GiB. This noisy pair cannot quantify the
candidate's complete-proof effect, but the candidate already fails its local
wall and GPU-efficiency gates. The traces are
`/private/tmp/akita-btreemap-28-metal-a3-parent.json` and
`/private/tmp/akita-btreemap-28-metal-a3-candidate.json`, SHA-256
`19d0c9634fd6143f52527604679256c4741e6b2d5808cde2ea0826bf35a14dc5`
and `cf806468a2a3e5fbaa113996f5f06a3fe6f889c8ffdf7e7ffbef326a1cca1d8d`.
The dense A2 transition and its original alternate-state capacities are restored.

## Attack A4: delay the Stage-5 compatibility scatter past the wide register rounds

The Stage-5 Instruction Read-RAF compatibility worker is released immediately
after `RegistersReadWriteChecking::prepare`, then competes with the widest host
register rounds in Stage 4. In the current A2 parent, release is visible at about
50.015 seconds, compatibility construction runs until the address prefetch begins
at 51.498 seconds, and that prefetch completes at 51.590 seconds. The complete
service is therefore about 1.575 seconds. It overlaps register rounds 0--2; the
first five register rounds take 2.919985 seconds in the Metal run versus 2.058763
seconds in the fixed optimized-CPU trace.

The candidate changes only the release point. It prepares the unchanged host
register kernel and retains the worker token, then releases the unchanged
compatibility worker after register round 4 completes. In the parent timeline
that is 53.024 seconds, leaving 1.693 seconds before Stage 5 starts. This exceeds
the measured worker service by 0.118 seconds. Later register rounds have much
smaller frontiers, so the worker should consume the existing slack while avoiding
contention with the five widest rounds. Stage-5 prepare still joins the worker and
fails closed on stale input, failure, or panic.

This candidate deletes no arithmetic or traffic. The compatibility construction
still reads and writes about 19.9 GiB, creates the same four resident planes, and
computes the same initial address message. Its bandwidth floor is about 0.05
seconds at 412.5 GiB/s, plus roughly 0.02 seconds of measured-rate field work.
Messages, challenges, round order, output claims, proof bytes, verifier behavior,
storage mode, and lifetimes are unchanged. The only claim is that the current
launch point spends bandwidth and page-fault service on the wrong side of the
Stage-4 critical path.

The release gate must prove that rounds 0--3 do not signal the worker, round 4
signals it exactly once, and terminal cleanup still releases a pending worker.
The existing compatibility-scatter oracle and modular proof remain the arithmetic
correctness checks. T25 must verify and may regress by at most 3%; its T28-only
prefetch route must remain unselected. At T28, retain only if the release occurs
after round 4, the Stage-5 join is at most 0.10 seconds, the first five register
rounds improve by at least 0.30 seconds, combined Stages 4+5 improve by at least
0.30 seconds, complete proving improves by at least 0.20 seconds against an
adjacent A2 parent, and no cost moves into Stage 5 or Stage 6b. Do not sweep the
release round under this candidate.

**Result: rejected and exactly reverted.** The focused release gate passed and
clippy was clean. The verified T25 guard produced no release event and completed
in 6.156303 seconds versus the 6.038282-second A2 reference, a 1.95% regression
inside the 3% limit. The A4 release binary was
`226a3c82c99fa52caf4bb3d38e272444f0b7c6dc66a3d31c61abdb33b3264096`.

At T28 the worker was released after exactly round 4 and the Stage-5 join took
only 38.6 microseconds. Its address prefetch was complete 0.226 seconds before
the join, so the fixed release point had adequate slack. The first parent made
the candidate look promising: complete proof fell from 44.537682 to 43.818297
seconds and combined Stages 4+5 fell from 8.524142 to 8.046003 seconds. The first
five register rounds improved by only 0.211116 seconds, however, 0.088884 seconds
short of their gate and within the registered repeat band.

The closing unchanged A2 parent completed in 43.004490 seconds. Against the mean
of the two surrounding parents, A4 regressed complete proof by 0.047211 seconds
and PIOP by 0.367008 seconds. Combined Stages 4+5 improved by only 0.090762 seconds
and the first five register rounds by only 0.004389 seconds, both effectively
noise and below their 0.30-second gates. Stage 6b was 0.208262 seconds slower than
the surrounding-parent mean. All three proofs verified at 80.77 GiB peak RSS.
Delaying this worker does not remove the Stage-4 register cost; the apparent first
pair win was run-order variation.

The traces are `/private/tmp/akita-btreemap-28-metal-a4-parent.json`,
`/private/tmp/akita-btreemap-28-metal-a4-candidate.json`, and
`/private/tmp/akita-btreemap-28-metal-a4-parent-replication.json`, with SHA-256
`c6d1c7770c8bb3dc587ad484c038091d08df7cdb33d2f07ea6a718c6202d6457`,
`068a60bc250b2be25ee93b643e70c4a8cff7fcf003a2f47f8848a0f30e9003ca`, and
`d948e6c89a01ec72720e69b66c9eae746b524d460bd4fbecb48a3997c5ed9a5d`.
Immediate post-prepare release is restored.

## Current-source refresh after K003 (2026-08-28)

The selection trace was refreshed after the final K003 build using adjacent,
verified BTreeMap T28 proofs (`target_trace_size=150000000`, 16 Rayon threads).
The release binary is `f00002049f23b81de94fcbb72869d022b0dcb799c316084108f9ab069e1a641e`.
The optimized and Metal Chrome traces hash to `aa2500a9...` and `078ad4ce...`.
Logical member time is the sum of each member's outermost `prepare`,
`prove_round`, `finish_rounds`, and `output_claims` spans; nested Metal spans
and duplicate same-name nesting are excluded.

| Remaining member | CPU ms | Metal ms | CPU / Metal | Metal over 5x budget ms |
|---|---:|---:|---:|---:|
| RamRaVirtualization | 2457.399 | 2440.760 | 1.007x | 1949.281 |
| ProductRemainder | 1657.586 | 2025.375 | 0.818x | 1693.857 |
| RamRafEvaluation | 408.799 | 1049.786 | 0.389x | 968.027 |
| RamValCheck | 946.104 | 934.146 | 1.013x | 744.925 |
| BytecodeReadRafCycle | 4275.546 | 1361.315 | 3.141x | 506.206 |
| RamHammingBooleanity | 534.640 | 532.887 | 1.003x | 425.959 |
| SpartanShift | 569.469 | 427.915 | 1.331x | 314.021 |
| BooleanityCycle | 14049.111 | 2968.577 | 4.733x | 158.755 |
| HammingWeightClaimReduction | 2101.036 | 556.849 | 3.773x | 136.642 |
| BytecodeReadRafAddress | 741.817 | 240.875 | 3.080x | 92.512 |

K001 OuterRemainder, K002 RAM read/write, and K003 registers read/write are
excluded from reselection because their isolated matched evaluators have
already passed promotion; their broad E2E spans contain displaced or shared
work and are not comparable to those member boundaries. By the registered
largest-gap rule, K004 is **RamRaVirtualization**.

The current Metal slot does not run a device kernel at this workload. The
cycle-family owner rejects 65,195,206 accesses at the 262,144 retained-record
cap, after which RamRaVirtualization falls back to the optimized CPU. Merely
raising either the retained-record or one-million-product cap is not a valid
attack: the surviving implementation is host-sparse, and its construction and
round work scale with the same high-activity access set. K004 therefore starts
with a fixed real-witness member evaluator and a direct-device model.

The evaluator falsified reuse of the fixed RAF plane: BTreeMap T28 reaches
remapped address 514,386, outside that plane's 8,192-entry domain. The viable
source route is a zero-copy unified-memory alias of the existing shared RAM
address column, with its owner retained for the sequence lifetime. The matched
optimized-CPU p25 is 1,715.951 ms, so the fixed 5x and 10x walls are 343.190
and 171.595 ms. A traffic and arithmetic model predicts 120--160 ms, making
10x a clear priced route. C164 is preregistered to implement that sequence;
T25 must also clear 5x when routed to Metal, while T20 retains a no-regression
CPU crossover.

## K004 promotion: RAM RA virtualization (2026-08-28)

C165 replaces the high-activity CPU fallback with a zero-copy, lazy-prefix /
dense-suffix Metal sequence for both production geometries (two or three
committed 8-bit factors). The final conservative promotion ratios against the
matched optimized CPU implementation are:

| Workload | CPU p25 ms | Metal p75 ms | Speedup |
|---|---:|---:|---:|
| BTreeMap T28 | 1745.128 | 160.807 | 10.852x |
| SHA-2 chain T28 | 877.847 | 69.427 | 12.644x |
| Fibonacci T28 request (effective T27) | 436.758 | 40.349 | 10.824x |

Every sample has exact cross-arm parity and selects the direct Metal route.
BTreeMap T25 clears 6.189x; T20 remains on the CPU crossover at an 18.355-ms
median, below its 18.603-ms cap. A production T28 BTreeMap proof verifies in
43.90 seconds at 80.79-GiB peak RSS with zero swap growth. K004 is `done_10`
and is excluded from subsequent ordering.

## Current-source refresh after K004 (2026-08-28)

Fresh proof-verified BTreeMap T28 traces use binary `fe4152f3...`; optimized
and Metal hashes are `a50f96a4...` and `80d97521...`. After excluding K001
through K004, the leading gaps are:

| Remaining member | CPU ms | Metal ms | CPU / Metal | Metal over 5x budget ms |
|---|---:|---:|---:|---:|
| ProductRemainder | 1645.906 | 2189.285 | 0.752x | 1860.104 |
| RamRafEvaluation | 416.934 | 1116.014 | 0.374x | 1032.628 |
| RamValCheck | 975.209 | 1010.572 | 0.965x | 815.530 |
| RamHammingBooleanity | 548.334 | 523.674 | 1.047x | 414.008 |
| SpartanShift | 576.657 | 458.486 | 1.258x | 343.155 |
| RamRaClaimReduction | 133.763 | 133.375 | 1.003x | 106.623 |

K005 is ProductRemainder. The trace records 179.971 ms of bind-plus-opening GPU
activity inside 2,187 ms of corresponding wall, so a fully charged standalone
evaluator is the first experiment. Its 5x wall is 329.181 ms. The 164.591-ms
10x wall is already below those two phases' current GPU activity before initial
materialization, so no 10x route is registered yet.

## K005 promotion: Product remainder (2026-08-28)

C175 submits Product's 4-GiB state-B fill on a dedicated Metal command queue
while the main queue materializes state A, then fences and validates both
commands before round 1. The same lifecycle is used by standalone and joint
Product/Instruction materialization. It changes no relation, round, opening,
claim, transcript event, or verifier behavior.

| Workload | CPU p25 ms | Metal p75 ms | Speedup |
|---|---:|---:|---:|
| BTreeMap T28, seven pairs | 1631.271 | 280.180 | 5.822x |
| SHA-2 chain T28 | 1470.269 | 276.452 | 5.318x |
| Fibonacci T28 | 1623.977 | 283.133 | 5.736x |

All arms have exact round, terminal, and eight-opening parity. BTreeMap T28
prepare p75 is 127.154 ms and round-1 p75 is 30.086 ms, versus 109.662 and
119.522 ms before overlap; the combined critical path falls by about 72 ms.
The three T20 Metal guards are 5.702, 5.546, and 5.724 ms, each below its
pre-candidate observation. A production Metal BTreeMap T20 proof verifies in
1.24 seconds at 951.75 MiB peak RSS.

K005 is `done_5`. Ten times is not a priced route: the 147--164-ms 10x budgets
are below the observed mandatory materialization, transition, and terminal
opening service. After excluding K001--K005, K006 is **RamRafEvaluation**, the
largest remaining frozen 5x gap at 1,032.628 ms.

## Goal-mode closure after K010

The analysis-first campaign exhausted the refreshed positive-gap order at K010.
The final K010 promotion uses the complete member boundary and optimized CPU as the
numerator:

| Workload | Actual domain | Production route | CPU p25 | Candidate p75 | Conservative speedup |
|---|---:|---|---:|---:|---:|
| Fibonacci | 2^28 | host sparse | 39.847 ms | 0.256 ms | 155.425x |
| SHA-2 chain | 2^28 | compact-record Metal | 42.780 ms | 7.229 ms | 5.918x |
| BTreeMap | 2^28 | dense no-copy Metal | 114.415 ms | 16.682 ms | 6.859x |

At T20 the forced Metal implementation remains exactly equal to optimized CPU, but
production selects optimized CPU because launch and setup overhead dominate. Clear
modular acceptance passed 21/21 tests and ZK acceptance passed 14/14. With K010
excluded, no materially active below-5x kernel remains in the registered order.
