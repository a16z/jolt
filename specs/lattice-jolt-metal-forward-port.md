# Spec: Metal Acceleration for Optimized Lattice Jolt

| Field | Value |
|---|---|
| Author(s) | Codex, for Justin Thaler |
| Created | 2026-11-09 |
| Status | in progress — initial design reviewed |
| PR | |

## Summary

Port the useful algorithms from the existing Akita Metal branch onto current
optimized Jolt. CPU Jolt has gained substantial speed since that branch diverged,
including compact witnesses, shared preparation, cheaper sumcheck arithmetic, and
implicit trace padding. The new Metal backend must retain these reductions in
work while accelerating the remaining work. Implementation proceeds through
small, independently reviewable changes, using Jolt's existing kernel registry,
proof session, shared Metal runtime, and Akita source interfaces. The performance
target is at least 20 MHz at a padded trace size of 2^27 on the M5 Max, with
approximately 40 MHz as the stretch target, using the same workload suite and
measurement boundary as the previous Metal comparison.

## Intent

### Goal

Provide an opt-in Metal backend for current lattice Jolt that reaches the target
throughput without changing the proof protocol or discarding CPU optimizations.

The implementation starts from Jolt
`5bfb6b5a51e4ab8b422cb260818d2797ac6a0c26`. The principal donor is
`pr/akita-metal-upstream` at `7aa549cc9ffce0abe1578609b233d8133d6c14f8`.
The initial Akita dependency is main's pinned revision
`28602b086d65f12fa5b1b2784d7dbbe560e9a642` in `LayerZero-Labs/akita`.
These revisions define the initial comparison; later dependency updates must be
explicitly recorded and used by both CPU and Metal benchmark arms.

Implementation lives on a new branch from current main. Donor kernels are ported
selectively, with their algorithms adapted to the current interfaces. A wholesale
merge of the donor's prover, verifier, witness, dependency pins, or CPU kernels is
outside this plan.

### Invariants

1. **The current protocol remains authoritative.** `jolt-claims` and the current
   verifier own relations, dimensions, ordering, and opening points. Metal does
   not change schedules, the field, challenge distribution, grinding, security
   parameters, or serialized proof types to reach the target. The existing host
   transcript absorbs round messages and draws challenges in its existing order.
2. **The logical polynomial and its stored prefix are distinct.** Skipping
   padding changes computation and storage, not the logical domain or number of
   rounds. Sumcheck must preserve zero, constant, and two-periodic tails as
   applicable. In particular, `PrefixColumn::tail: [F; 2]` becomes constant only
   after its first low-bit bind. Odd prefix lengths require the correct missing
   endpoint; they cannot be handled by dropping the final pair.
3. **PCS zero suffixes do not alter proof geometry.**
   `TraceOneHotRows::nonzero_prefix_len()` bounds internal work. It must not
   replace the semantic polynomial domain in schedules, opening challenges,
   chunk ranges, or protocol live-block counts. An absent selector and a
   committed digit zero remain distinct through their mask semantics.
4. **Keep the optimized CPU implementation.** The default CPU backend retains
   its current compact representations, shared rows, arithmetic shortcuts, and
   padding handling. It remains the fallback for unsupported or unprofitable
   shapes. GPU implementations preserve reductions in mathematical work;
   CPU-specific scheduling and loop choices may have different GPU counterparts.
5. **Use one Metal arithmetic and runtime implementation.** New Jolt kernels use
   `jolt-metal` field headers and runtime. The donor's duplicate Solinas field
   types, reduction helpers, and `metal-rs` runtime are not imported as a second
   foundation. New accumulator operations belong in `jolt-metal` only when a
   production kernel needs them, with their capacity established there.
6. **Every large buffer has an owner.** The reusable backend owns device state
   and compiled pipelines. Witness-derived buffers belong to a proof or a
   narrower operation. Shared consumers retain the same owner; unrelated proofs
   cannot reuse its contents. Reusable setup buffers have the setup's lifetime.
   Host access and reuse obey the runtime's completion rules.
7. **Routing is explicit.** Unsupported shapes may select the CPU before GPU
   execution. A planned GPU-to-CPU tail transfer preserves the already-bound
   state. An execution fault fails that proof; it does not silently restart a
   member after transcript progress or turn a failed GPU benchmark into a CPU
   result.
8. **Metal stays outside the verifier.** CPU-only builds and non-macOS builds
   retain their current behavior. The optional GPU dependencies remain in the
   prover dependency graph, subject to the existing isolation checks.

Existing protocol, field, and verifier invariants remain unchanged. This work
does not create a parallel invariant framework: new coverage is added to the
owning crate only for behavior introduced by a port.

### Non-Goals

- Dory/BN254 GPU acceleration, CUDA, or a general cross-device compute framework.
- New proof parameters, weaker security settings, protocol changes, or guest
  instruction changes to improve the headline number.
- New ZK or field-inline GPU kernels in the initial ordinary-workload port.
  Shared changes must preserve those existing builds; unsupported features use
  their existing CPU implementations or are explicitly unavailable.
- Importing the donor's entire configuration surface, experiment journals,
  benchmark executables, validation machinery, or fixture collection.
- Predicting a speedup by multiplying the old GPU gain by the CPU improvement.
  Completion requires measured full proofs.

## Evaluation

### Acceptance Criteria

- [ ] A `metal` feature and backend selector run the current Akita prover with
      the current verifier and canonical setup selection.
- [ ] Each enabled GPU family has focused arithmetic/round coverage and has
      participated in a verified proof through the production backend.
- [ ] Padding remains implicit in the affected paths; boundary cases preserve
      round messages and commitments, including committed zero selectors and
      nonzero/two-periodic tails where those are part of the relation.
- [ ] The same CPU backend remains available in the Metal-enabled binary for
      matched measurements, and CPU-only builds do not gain a Metal dependency.
- [ ] The four-workload T27 comparison reaches at least 20 MHz under the
      aggregation rule below. Approximately 40 MHz is the stretch target.
      Every individual workload, proof result, actual cycle rate, and latency is
      reported; the aggregate never substitutes for the individual results.
- [ ] Final changes pass the affected crate checks, required repository checks,
      and applicable existing verifier-isolation checks. Evidence identifies the
      tested revisions and feature sets.
- [ ] Production dependencies are pinned and reproducible. No absolute local
      paths, environment-selected experimental kernels, or unreviewed setup
      changes remain in the deliverable.

### Testing Strategy

Use existing tests and fixtures first. A kernel family needs a small test that
compares its rounds/final evaluations against the current independent reference
tier or direct field evaluation at tractable sizes. Extend that case to the
family's actual failure boundaries: partial tiles, an odd live prefix, nonzero
tails, committed digit zero, or the CPU handoff, as applicable. Do not multiply
the cross-product of every shape, challenge, backend, and fixture.

Use the existing end-to-end Akita matrix for integration. Run an affected small
proof when a family is enabled; reuse the same guests across stages. Large
timing runs must verify their proofs. Full proof-byte equality is not a generic
requirement for randomized proving; exact kernel algebra, current transcript
semantics, and verification are.

Use `cargo nextest`, the affected `cargo clippy` targets, and formatting checks.
Run normal and shader-validation GPU checks when adding or changing unsafe
dispatch layouts; use the existing `jolt-metal` report when its runtime or
arithmetic changes. Leave the full standard/ZK feature matrix to the appropriate
repository checks and CI, with local focused checks for shared changes. Do not
rerun a broad matrix after every shader tuning change once the relevant checks
pass.

Temporary performance probes and old/new comparison scaffolding are development
artifacts, not permanent tests. Retain one production benchmark path and tests
that detect distinct errors. Avoid fixtures whose expected result is merely a
second copy of the implementation under test.

### Performance

**Workloads and measurement.** Preserve the donor suite: Fibonacci, SHA-2 chain,
BTreeMap, and BLAKE2b-512 chain. Record exact input counts and guest identities;
run CPU and Metal on identical inputs with the same current proof configuration.
Restore the donor's `blake2b_chain` guest entry in
`examples/hash-bench/guest/src/lib.rs` with the same semantics, plus its adapter
in the current profiler; both are currently absent. Reuse the current profiler
instead of resurrecting the old standalone benchmark. Pin the T27
inputs once for the campaign; do not select a favorable workload or alter inputs
between candidates. The donor's BLAKE2b T27 case uses 40,000 hashes from a
64-byte seed of `0x05`, with the complete 64-byte digest chained each time.
Retain its independent output check. Its actual trace occupies substantially less
than the padded domain, making the actual/padded distinction consequential.
Trace each baseline input with the current guest/compiler and confirm that its
derived padded domain is exactly 2^27. If a historical operation count no longer
produces that domain, record the change and determine the current T27 count once
before freezing the CPU/Metal inputs. A requested scale alone is not evidence of
the resulting proof size.

The primary timer covers trace-backed witness preparation performed by proving,
commitment, all sumcheck stages, and the complete grouped opening. This is the
current Akita `prove` interval. Guest compilation, execution/tracing, reusable
preprocessing/setup, shader/pipeline initialization, and verification are outside
it and reported separately. Any proof-specific GPU extraction, uploads, packing,
buffer preparation, or synchronization remains inside it; work must not move
outside the timer to manufacture a speedup. Also report an online time including
tracing for context.

The old suite's headline is the arithmetic mean of its four padded-cycle MHz
values. Preserve that definition and label it. Also report the actual-cycle MHz
of every workload and their mean. For workload `i`, compute its rate from its
median untraced prover time, then average the four rates. Do not pool cycles and
times or average inconsistent timing windows.

| Rate for one T27 proof | Corresponding prover time |
|---|---:|
| 20 MHz | 6.7108864 s |
| 40 MHz | 3.3554432 s |

These are per-workload latency equivalents, not a claim that an arithmetic-mean
suite target imposes the same bound on every workload. Report any workload below
20 MHz explicitly.

**Measurement effort.** Begin with one clean CPU sweep and phase profiles on
representative workloads. During development, measure the affected workload and
phase; do not run all T27 cases after every edit. Use small traces for bring-up.
For a final candidate, collect at least three matched untraced CPU/Metal
observations per workload, rotating order, after a warmup. Report individual
times, medians, and ranges. If noise prevents a conclusion, investigate the
contention and repeat a bounded comparison; retain failed/contended observations
with their explanation. Shader-validation and tracing runs are not headline
timings.

Record the Jolt/Akita revisions, lockfile, build flags, thread count, device and
power mode, input identities, actual/padded rows, selected backend, proof result,
and peak memory. Benchmark without competing proving or build jobs. Account for
CPU and GPU buffers together on the 128 GiB unified-memory machine. Integrate
Metal allocations with current memory budgeting rather than treating GPU memory
as an additional independent allowance.

**Optimization order.** Measure current stage times before assigning budgets.
For each workload, split the full time into remaining CPU preparation, GPU work,
synchronization/readback, and other prover work. The part that remains on CPU must
itself fit inside the target time; a fast kernel cannot compensate for a slower
untouched stage. Use those measurements to order the stages below and stop
optimizing a kernel when another part determines the full-proof time.

Existing November 1 CPU measurements give a useful starting point, not a new
baseline: two verified Fibonacci T27 proofs at `fec5f7de6` took 30.794 s and
29.241 s (4.359 and 4.590 padded MHz). Stages 0, 6b, 5, and 8 accounted for
approximately 73% of those times. Commitment, instruction lookup cycles, and
opening therefore deserve early attention. Even eliminating stages 0 and 8
entirely would leave roughly 18.5–19.5 s in those runs, so a PCS-only port cannot
reach the target. More recent integration measurements include contention and
are not a clean current-main suite baseline.

The November 9 Booleanity candidate at Jolt `51090b762`, with Akita `551159bb`,
verified one untraced Fibonacci T27 proof in 26.008432208 s
(5.160546661 padded MHz, 18 workers). Its build manifest is
`jolt-metal-forward-port-artifacts-20261109/booleanity-metal/manifest.json`.
This records the trace-commitment, instruction-RA, and Booleanity routing before
root-fold offload; it does not establish the four-workload target or a root-fold
speedup.

The general sparse root-fold candidate at Jolt `c5fe773eb`, with Akita
`aa58a7af`, verified the same Fibonacci input in 24.6202655 s
(5.451514241 padded MHz). Its separate Chrome run measured 1.3323 s for the
complete root-fold adapter, including 0.9311 s packing and 0.3345 s GPU work;
the earlier CPU accumulation took 2.4219 s. Both small BLAKE2b and T27 proofs
verified, and the T27 profile confirmed 32 GPU root-fold dispatches. Evidence
is in `jolt-metal-forward-port-artifacts-20261109/root-sparse-metal/`.
These are development observations, not the repeated four-workload acceptance
comparison. Ring classification found only 2.6% uniform rings, so an additional
uniform-only transformed key is deferred.

ReadRAF at Jolt `a743ff017` also verified the small and T27 proofs. Its cycle
phase fell from 2.1477 to 1.7571 s in the separate profiles, including a shared
row upload moved from a later consumer. Overall throughput remains provisional:
two matched untraced observations took 26.2297 and 26.0313 s, while the prior
root-fold binary took 25.2315 and 27.8937 s. That variation prevents a firm
full-proof speedup claim. The frozen binaries, manifests, and observations are
in `read-raf-metal/` and `read-raf-paired/` under the same artifact directory.

The isolated TensorOps candidate (`866ae6cdd`, Akita `52b079d8`) verified
three T27 proofs and matched every coefficient in all 32 source windows under
both execution orders. The paired component means were 5.2212 s for scalar
and 4.8131 s for TensorOps, including source packing and each backend's key
preparation, but excluding common output allocation, scatter, and RingVec
construction. GPU means were 4.1016 and 3.6021 s. These dual-execution proof
timers are not throughput results; TensorOps remains disabled in this branch.
Evidence is in `tensor-paired/`. Dispatch-boundary counters are unsupported on
this M5 Max. Encoder-boundary counters are supported, but independent encoders
can overlap: the corrected diagnostic reports interval union and overlap
separately. At Jolt `863256a27` / Akita `3493f130`, the corrected diagnostic
verified two proofs and matched all 32 windows. Source transformation and
packing accounted for 72.39% of summed encoder durations, dot products 24.98%,
inverse transforms 2.30%, and combination 0.33%. These are diagnostic duration
shares, not production GPU occupancy or throughput. Evidence is in
`tensor-stage-overlap-timestamps/`.

A subsequent isolated candidate (`369dd00bb` / Akita `9a8beb4c`) coalesces
Tensor source writes and uses the current runtime with a per-library Metal 4
selection. Its focused arithmetic oracle passed normally and under Metal
validation; the small and frozen T27 proofs verified. The untraced Fib proof
took 22.9884 s (5.8385 padded MHz). In the separate profile, commitment used
3.9071 s of GPU time and 4.7646 s inclusive, versus 4.0850 and 4.7800 s in the
saved NTT profile. Extra Tensor key preparation took 0.1158 s. These unmatched
observations establish neither a full-proof gain nor a benefit from coalescing;
other components also varied. Keep NTT as the default and this candidate
isolated. Evidence is in `tensor-coalesced-current-metal/`.

The current limb-NTT commitment substantially reuses the November 8 reference
`cdf52d02`. The donor also has a direct radix26 K16 panel algorithm, which fits
the current frozen workload's
39 columns, 65,536 positions, rank one and single inner digit. Current source
ownership and implicit padding require adapter changes, not replacement of its
arithmetic. The isolated port retains 32-task key sharing and consumes current
retained selector windows directly. It uses a 10 KiB one-position tile so Metal
validation fits the device limit; this changes barrier frequency relative to
the donor's two-position tile and must be measured. The audit also identified
omitted fused cooperative register rounds and a width-one Booleanity lookup
table. Source crosswalks
and exact donor references are in `metal-pcs-donor-provenance-crosswalk.json`
and `donor-sumcheck-crosswalk.json` in the task artifact directory.
The first panel port (`ec4f1d62f` / Akita `18448889`) passed the independent
ring oracle, Metal validation, and small/T27 proofs, but remained isolated:
the T27 proof took 25.163155083 s (5.333899011 padded MHz), and the separate
profile recorded 6.9160 s of commitment GPU work. The one-position tile doubles
barrier frequency relative to the donor. The two-position comparison
(`2949ebb24` / Akita `60399604`) also passed focused checks and full proofs;
all 32 profiled windows selected the two-position kernel. Its proof took
25.036913584 s (5.360793676 padded MHz), with 6.5117 s of commitment GPU work.
This modest improvement remains slower than the current NTT route, which stays
selected. These observations establish the current comparison, not an
optimization limit. Evidence is in `donor-panels-metal/` and
`donor-panels-two-position-metal/`.
A later configuration audit found that donor `7aa549cc9` selects 8-bit committed
and 32-bit virtual chunks at T27, whereas current Akita uses 4/16 bits. The two
K16 panel experiments therefore do not reproduce the donor large-workload path.
That path uses a different raw-limb K256 panel kernel. The frozen original-port
control uses unchanged kernels/configuration and published Akita `79f09e2`.
Its adapter (`f4ef0a628`) verified the same frozen Fibonacci guest, input and
output in 32.4982 s. It produced 134,217,861 proof rows versus current
134,217,720, crossing into a padded T28 domain. Both versions exclude canonical
trailing padding; instruction lowering has changed. This is a verified
workload observation, not an equal-domain MHz comparison or reconstruction of
the historical 5x result. The separate profile also verified; evidence is in
`original-donor-observation/`.

The forward port will evaluate K256 through an explicit Metal selection before
preprocessing, preserving CPU defaults. Frozen Fib requires exactly 20 native
columns at arity 35. The current JoltOneHotK256 planner and validated catalog
emitter generated that exact schedule: D512, rank one, one inner digit,
131,072 positions per block, 512 blocks per column, subring dimension 128,
and 31 signed-unit challenge terms. The generator passed focused Clippy.
Catalog hashes and provenance are in `k256-native-catalog-manifest.json`.
The isolated source candidate selects K256 before preprocessing and adapts
commit/root/packing kernels to this geometry. Do not import the donor capacity-32
catalog or pad native columns to fit it. K256 device sources need byte selectors
plus zero-selection masks, because 255 is a valid selector. Shared source
ownership and implicit suffixes remain unchanged. Instruction RA, Booleanity
and Bytecode already admit 8-bit selectors; ReadRAF cycle processing needs
branch-local two-table products and a five-factor round specialization. Plans
are in `k256-metal-config-forward-port-plan.json` and
`k256-sumcheck-admission-audit.json` in the task artifact directory.

The K256 candidate (`ed73e6b5b` / Akita `b177d86c`) passed the three existing
PCS oracles normally and under Metal validation, parallel/serial companion
Clippy, integrated prover Clippy, and small/full proofs. Frozen Fib retained
exactly 134,217,720 actual rows and 134,217,728 padded rows, and verified in
23.2004 s (5.7852 padded MHz). The separate profile completed 15 native windows
with retained-source reuse and the five-factor ReadRAF width-four handoff.
Commitment used 5.4101 s of GPU time; coefficient packing used 0.1193 s and root
folding 0.2777 s. Several sumcheck families shortened, but the whole proof was
slower than the saved K16 Registers observation (20.7476 s); K256 remains
isolated. This is not a paired comparison. Evidence and the repaired final
runner-status write are preserved in `k256-metal/`; no proof was rerun for
that postprocessing repair.

A subsequent same-proof diagnostic (`dd0f014e1` / Akita `a87f64c0`) compared
32-task and 16-task panels on identical keys and sources, alternating order.
Every coefficient agreed across all 15 windows, and the proof verified. GPU
time totaled 5.3690 s for 32 tasks and 4.8767 s for 16; dispatch through copied
readback totaled 5.4290 and 4.9093 s. The 16-task schedule won in all 14 full
windows under both orders; the eight-block tail was essentially unchanged.
This supports selecting 16 tasks for the next K256 candidate. The diagnostic
used release optimization with LTO disabled and executed both schedules, so its
whole-proof time is not throughput. Evidence is in `k256-panel-tasks/`.

The isolated Outer diagnostic (`6cef6ad8a`) also verified. Outer occupied
2.024 s in its separate profile. Sampled extraction accounted for 13.8%,
20.3%, and 26.9% of staged worker work across uniskip, remainder preparation
and final opening, respectively. These include allocation/stores and cannot
be multiplied by phase wall time to predict savings. The proposed 56-byte
lookup cache would add 7 GiB retained storage and 21 GiB traffic; defer it.
The measured allocation call itself took 38 microseconds, while backing-memory
writes remain inside materialization. Keep the diagnostic isolated; evidence
is in `outer-source-attribution/analysis.json`.

The isolated Booleanity width-one LUT and ReadRAF width-one message ports
(`ac6f4feda`) passed focused normal/Metal-validation oracles and small/T27
proofs. One untraced Fib proof took 29.1053 s (4.6114 padded MHz); the separate
profile slowed across several unchanged components as well. ReadRAF family
time fell from 2.7090 to 2.3722 s, while Booleanity rose from 1.8448 to 2.3620 s.
These saved-baseline observations do not establish a combined speedup or a
causal whole-proof regression. Both changes remain isolated pending selection.
Evidence is in `selector-messages-metal/`.

The reported donor 5x speedup was against the older CPU implementation, and
Markos believed roughly 10x was possible. Its measured throughput is therefore
a reference, not the target or an optimization ceiling. After each port, use
the current inclusive profile to improve it further; acceptance remains at
least 20 MHz, with approximately 40 MHz as the stretch target.

Reuse existing profiler stage labels and `jolt-eval` telemetry objectives for
prover/stage time where available. Add only measurements needed to identify GPU
dispatch time, host preparation, and waiting; keep end-to-end time authoritative.
No new objective framework or fleet-wide performance dashboard is needed.

## Design

### Architecture

| Owner | Responsibility |
|---|---|
| `jolt-claims`, current verifier | Protocol expressions, geometry, challenge/claim ordering, verification |
| `jolt-prover` | Existing orchestration, backend selection, profiling |
| `jolt-kernels` | GPU sumcheck implementations, CPU/GPU routing, proof-scoped shared witness buffers |
| `jolt-sumcheck` | Existing fused round loop and scheduler seam; any later device-neutral scheduling extension |
| `jolt-metal` | Canonical field arithmetic, compilation, typed device buffers, dispatch and completion |
| `jolt-akita` | Current semantic trace-column adapters and Jolt-specific PCS integration |
| Akita prover/Metal crates | Ring/NTT/commitment/opening computations behind current source and backend interfaces |

#### Kernel integration

Construct the Metal backend by starting with `JoltBackend::optimized()` and
replacing only qualified slots. Implement the current `PrepareKernel` and
`SumcheckKernel` contracts. Keep the current fused `ProveRounds::prove_round`
operation, which receives the preceding challenge and computes the next message;
retain that fused contract throughout the port.

The first production GPU family uses the current synchronous scheduler. Add
generic scheduling capabilities only when a second real consumer demonstrates
that shared submission or CPU/GPU overlap is valuable. A proposed extension must
identify its callers and replace a concrete duplication or serialization cost.
It must not put Metal types, relation formulas, or transcript logic into
`jolt-sumcheck`.

Instruction RA and Booleanity retain the CPU's lazy branch tables, binding
schedule, and output checks. Large prefixes use the existing GPU reductions
from width one onward; small prefixes retain the categorical CPU path. At the
canonical lazy-policy crossover, they
materialize directly on the GPU and keep shrinking field prefixes there until
the small CPU tail. Both use the same selector materialization and fold helper.
They share one proof-owned compact device row buffer. The session keeps a weak
reference, so materialization by the last consumer releases it. Independent
round oracles and Metal validation passed, and the combined candidate verified
the frozen Fibonacci T27 proof. Its measurement is recorded below.

ReadRAF keeps its canonical address-phase schedule and grouped first cycle
message. Qualified address scans retain coalesced representatives and weights
on the device; three-lane histograms handle RAF and suffix jobs separately,
within the device's shared-memory limit even under Metal validation. The
canonical CPU scalar suffix definitions remain the independent test oracle.
The combined full-proof profile exercised all sixteen address phases. The cycle path
offloads lazy width-two/four messages, materializes directly at the existing
lazy-policy crossover, and keeps subsequent shrinking prefixes on the device
until the small CPU tail. Separate claim and selector extents retain their
implicit tails. Its packed claim byte shares the existing 32-byte instruction
row, which a proof-owned carry keeps alive for later RA and Booleanity use.
Focused round and Metal validation checks pass; the full-proof development
measurements above remain provisional.

Bytecode address pushforwards use split-equality tables and bounded PC
histograms. A signed-magnitude increment sidecar stays separate from the shared
selector rows and survives the address-to-cycle handoff. The cycle kernel
consumes the canonical equality-term recipe, combines repeated PC tuples,
and materializes RA and increment prefixes at their existing lazy-policy
crossovers. It retains shrinking prefixes until the CPU tail. Focused reference
checks, Metal validation, and integrated lint checks pass. The combined full-proof
profile exercised native production and resident continuation.

The combined candidate `02eefba441` with Akita `aa58a7af` verified the frozen
Fibonacci T27 proof in 25.760409167 s (5.210232770 padded MHz), one untraced
observation. Its separate profile confirmed reuse of all 32 selector windows
without root repacking. Native Product spent 2.698438 s in three passes, including
1.135321 s packing and 0.896267 s upload; its total instrumented subtotal was
3.16634 s. That result justified restoring CPU Product before further integration.
The CPU subtotal omits its opening walk, so the subtotal difference is not an
exact end-to-end regression estimate. Evidence is in `resident-metal/`.

The next combined candidate adds uninitialized RA/Booleanity outputs, direct
ReadRAF address fills and row projection, InstructionInput native production,
shared CPU instruction rows from commitment through stage five, and retained-source
coefficient packing. The shared row carry uses exact immutable source identity,
keeps physical selector extent distinct from the padded domain, and leaves only
a weak reference in opening sources. Coefficient packing borrows the selector
cache; root folding consumes it later. Focused ownership and arithmetic checks
pass. At `71cb1b601` with Akita `982abab4`, the frozen Fibonacci workload
verified in 20.907483458 s: 134,217,720 actual cycles and 134,217,728 padded
cycles, or 6.419601875 actual / 6.419602257 padded MHz. This is one untraced
observation, not the final matched four-workload comparison. Packing, uploads,
witness preparation and first-use key preparation are inside the prover timer.

The separate profile confirmed all new GPU paths and continued reuse of all
32 selector windows. ReadRAF preparation fell from 1.113811 s to 0.105636 s,
with 0.186938 s now separately attributed to scan construction and projection;
shared CPU-row collection cost 0.153521 s in stage zero. The inclusive ReadRAF
family union fell by 1.028104 s. Input rounds fell from 0.869291 s to
0.675459 s. Coefficient packing took 0.575866 s, including 0.537854 s on the
GPU, compared with the previous CPU packing span of 0.748200 s. These profile
comparisons include run variation and are attribution evidence, not isolated
causal measurements. Peak RSS was 43.8073 GiB; the proof-exit RSS was 13.1803
GiB. Artifacts and the frozen binary are under `direct-source-metal/` and
`direct-source-metal-jolt-prover` in the task artifact directory.

At `b37f2a868` / Akita `978d7060`, direct selector-buffer production verified
the small proof and the same T27 Fibonacci proof. The untraced observation took
21.99135675 s, or 6.103203614 padded MHz. Its separate profile reduced source
packing from 0.831321 to 0.478342 s, while several unchanged CPU families were
slower. This establishes the packing improvement in those profiles, not a
full-prover speedup or regression. Evidence and the frozen comparison binary
are in `direct-selector-metal/` and `direct-selector-metal-jolt-prover`.

The sparse register prototype at `5bd928644` passed its independent oracle,
Metal validation, and full proofs, but its data transfer cost erased the GPU
benefit. Preparation rose from 0.2682 to 1.4401 s, including 1.1721 s packing
and upload, while round time fell only from 1.9889 to 1.9240 s. The optimized
CPU baseline was restored while the adapter was revised. The current port
restores GPU routing using a canonical producer that writes padding-free
24-byte seed entries directly into typed uninitialized Metal storage. Sparse
outputs become initialized only after their complete writer finishes. CPU
topology, coefficient-table policy and small-tail handoff remain canonical.

The donor cooperative indexed/transition/direct bind-message schedule is
adapted from `7aa549cc9:crates/jolt-kernels/src/metal/solinas/registers_read_write/fused_sequence.metal`.
It supports the current 128-column masks and odd/missing rows. Bind-only and
fused paths share transition, increment and message helpers. The LUT advances
once; indexed messages use the newly bound codebook, while the transition
resolves the old saturated codebook.

At measured Jolt `b13eba9d2` / Akita `978d7060`, the normal and instrumented
reference oracle, five CPU boundary checks and kernel/prover Clippy passed.
Small and T27 proofs verified. The untraced T27 observation took 20.747624417 s
(6.469064858 padded MHz), versus the saved 21.99135675 s baseline observation.
The separate profile reduced the register-family wall union by 0.636915 s.
Native seed filling took 0.236916 s; the profile confirmed one fused indexed,
one transition and ten direct dispatches. Peak RSS was 39.1484 GiB. This is
one development observation, not the final matched four-workload acceptance
comparison. The measured code is integrated; evidence and the frozen binary
are in `registers-current-metal/` and `registers-current-metal-jolt-prover`.
The earlier prototype remains preserved in `registers-metal/`.

Spartan outer remainder stays on the optimized CPU path. Its initial GPU port
was correct, but upload and buffer preparation alone exceeded the measured CPU
round cost. Reconsider it when an integrated GPU producer can supply its inputs
and the complete path improves prover time.

Introduce device-neutral compute-error propagation only where the first GPU
round requires it. Keep Metal types out of `SumcheckError` and the verifier's
dependency graph. Complete terminal GPU readback in `finish_rounds`, before
typed output extraction. Preserve the current generic field bounds when changing
the round API; do not narrow all consumers to the donor's Jolt-specific bounds.

When measured overlap or shared-submission needs arise, consider splitting
`Batch::commit_and_wait` into submission and
completion through an owned in-flight handle, while retaining the synchronous
convenience path. The handle must preserve buffer exclusion until completion is
known, including error and drop paths. Batch only independent work for the same
known challenges; the host still needs the completed round messages before
drawing the next challenge. Do not import the donor's per-round thread-spawning
policy without comparing it to the current CPU scheduler.

#### Shared witness storage and padding

The first bounded native-product candidate was removed after full-proof
measurement. Its three source passes spent 1.135 s packing and 0.896 s uploading,
against 0.240 s of GPU execution. The complete measured product family took
3.166 s, while the preceding CPU profile spent about 0.779 s in the named product
spans (excluding its uninstrumented opening walk). A further attempt must share
an existing canonical extraction pass and write directly to reusable device
storage; retaining the three-pass candidate alone is insufficient. The CPU
product implementation remains selected.

Use the current compact witness producers. Allocate a shared device source when
its first consumer needs it and release it after its last consumer. Port related
families together when the measured benefit depends on that sharing, notably
Spartan outer/product and instruction address/cycle processing. Avoid repeated
trace walks, a dense field vector per selector, and unrelated full-trace copies.

For prefix-backed tables, carry the stored prefix length and tail semantics into
the kernel. Dispatch over stored pairs, handle the boundary pair, and add the
tail contribution analytically using the current equality-polynomial semantics.
After binding, update both the prefix and tail before the next round. A small
GPU-friendly description of this existing representation can be shared by its
actual consumers; it must not introduce a new protocol-sizing law.

When columns have different prefix lengths, combined work covers the longest
required prefix and reads each column's own tail beyond its extent. CPU handoff
copies the current compact prefix and tail once, without rebuilding the padded
table.

CPU handoff decisions use actual remaining work, data already resident, transfer
cost, and the current CPU implementation. Start with measured per-family
thresholds. Avoid a broad tuning API or a public collection of experimental knobs.

#### Commitment and opening

Current `TraceOneHotColumn` views share a `TraceOneHotRows` owner and preserve
semantic column order. The donor instead integrated a physical prefix-packed
trace and column capacity. Preserve the current source descriptors and canonical
plans, and port the GPU computation behind their capability interfaces.

Commitment kernels consume compact selectors plus committed-zero masks in bounded
windows, skip certified zero work, and accumulate the same planned ring products.
Opening kernels support the current sparse representations, including chunked
sparse challenges. Both obey the plan's ring dimension and column ordering. A
shape unsupported by a GPU kernel retains the current CPU route before the
operation starts.

`commit_trace_one_hot` calls `TraceOneHotComputeKernel::bind_source` once for the
exact `TraceOneHotRows` owner, then shares that child kernel across its column
views and opening hint. Metal shares the setup, device, and transformed key by
`Arc`; the setup never retains a child or its trace buffers. The child checks
source identity and owns any retained selector windows under a mutex.
Decomposition constructs a
read-only `TraceOneHotFoldPlan` after the existing geometry, challenge, and
canonical-range checks. The Metal D512/K16 path accepts sparse ring challenges
and whole-block segments, then returns compressed centered `i32` coefficients
in position/chunk/coefficient order. Chunk ranges partition each column's blocks;
challenges remain claim-major. CPU and GPU share digit expansion and witness
construction, with no coefficient copy for one chunk and one digit. The fold
reuses bounded row packing and checks term-index and accumulation bounds before
dispatch. Base-field coefficient packing uses the canonical PCS position and
packing weights over the same retained selectors, in bounded block tiles. It
borrows those windows so root folding can consume them later. Unsupported
extensions, geometries, or unavailable caches retain the CPU implementation.

Stage 0 may retain its existing `DeviceK16Sources` windows for root folding.
For `P` physical prefix rows and `C` columns, their device payload is
`C * P + C * ceil(P / 32)` bytes: selectors plus ring classifications. Host masks
and packing buffers remain transient. The direct producer now uses at most
8192 rows of selector/mask scratch per active worker and writes normalized
selectors and ring classifications directly to their final uninitialized shared
buffers. Both source constructors use one encoder for absent/committed-zero
semantics and partial-ring classification. Successful completion of all host
writes precedes publication; errors drop the unpublished buffers. The implicit
suffix allocates nothing. Standalone arithmetic and Metal validation passed
against the published runtime pin; inclusive timing of this producer is pending.
Retention is admitted only when that payload fits the smaller of 8 GiB or
one-sixteenth of the device's recommended working set. This is an additional
payload cap, excluding allocator overhead and small per-window host metadata,
not a reservation or a bound on total prover memory; ordinary Metal allocation
checks still apply. Smaller memory budgets retain the bounded
streaming path without storing a partial cache.

Root folding takes the retained windows only when its rows-per-block geometry
matches commitment. It releases each window after its completed dispatch.
A mismatched geometry drops the cache and streams; an unopened source releases
all windows with its source/opening hint. No cache is indexed by dimensions or
kept in the setup. Device errors propagate rather than retrying GPU work.

The measured T27 Fibonacci geometry has 39 columns and at most 5,398,069,248
bytes (5.0273 GiB) of resident source payload. The preceding root-fold profile
spent 0.9311 s packing/uploading and 0.3345 s on GPU execution. The verified
`71cb1b601` profile reused all 32 windows without repacking; root folding took
0.3612 s inclusive, with 0.3343 s on the GPU. Retained payload, peak memory,
and proof-exit memory are recorded in `direct-source-metal/memory.json`. These
remain development observations; final matched full-proof comparison is pending.

Integer limb-NTT and deferred-accumulation kernels retain explicit reconstruction
and accumulation bounds for every admitted geometry. Changing tile sizes or
digit widths must preserve those bounds.

Use `fill_rows_with_masks` to pack directly into a compact row-major or tiled GPU
input. Avoid the experimental intermediate `Vec<Vec<Option<u8>>>`, its transpose,
and repeated validation scans. Cache transformed commitment keys with the exact
setup owner rather than uploading the full key on every proof. Retain uploaded
selectors for opening only when the saved preparation justifies their lifetime;
otherwise reuse bounded windows.

Port root opening coefficient packing and sparse/chunked decompose-fold as
separate operations. Preserve current `SubringCoefficientPackingPlan` tables and
response ordering. Profile the remaining recursive PCS work before adding GPU
NTTs, matrix products, or digit decomposition. A CPU/Metal backend split must
preserve current same-owner reuse of auxiliary commitments rather than forcing
them to be recommitted.

Initially, Akita's existing backend retains protocol orchestration and ownership
of commitment/opening handles. The trace-source operations delegate qualified
arithmetic to an explicitly owned Metal context. A full Metal PCS backend is
introduced only if a measured recursive operation requires it; that change must
preserve same-owner auxiliary commitment reuse.

Pass this context through the existing Akita setup/prover selection. If that
path cannot carry the needed device state,
extend its narrow owner in the same change as the first production caller. Do not
route through process-global mutable contexts or environment variables. Preserve
current compact auxiliary buffers and shared owners.

Local November 8 commitment/opening experiments are additional implementation
references. Their absolute dependency patches, environment switches, and
temporary transpositions are not the target design. Changes to Akita land as
separately reviewable dependencies; Jolt pins the resulting commit. All Akita
packages use one compatible revision, and the verifier remains GPU-independent.

### Alternatives Considered

**Merge main into the old Metal branch.** This combines the stale integration
with hundreds of subsequent changes and makes it difficult to tell which current
optimizations survived. A forward port keeps the current CPU behavior as the
starting point and makes every GPU addition reviewable.

**Rewrite every shader from scratch.** The donor contains useful arithmetic
algorithms, tiling choices, and residency strategies. Reuse those after checking
their assumptions against current representations and the shared runtime.

**Copy a working GPU backend first and clean up later.** This would introduce
duplicate field arithmetic, old witness layouts, and a large integration diff
before any part is maintainable. Each stage instead adopts current ownership and
arithmetic boundaries as it lands.

**Design a general accelerator API up front.** The kernel registry, proof
session, and scheduler already provide useful seams. Extend them only for an
actual production consumer, with the CPU path demonstrating that the abstraction
remains device-neutral.

## Documentation

Update `book/src/usage/profiling/zkvm_profiling.md` with the Metal feature/backend,
supported configurations, timing definition, and CPU comparison command. Add a
short developer description alongside the new kernels explaining buffer owners,
prefix/tail representation, routing, and the focused validation commands.
Update the `jolt-metal` runtime documentation only if its API changes. Keep the
final benchmark report separate from this proposed design.

## Execution

The following are implementation stages, not a request for one large PR per row.
Split a stage further whenever the interfaces and first caller can be reviewed
independently. Every code change must build, preserve the existing CPU route,
identify its production caller, and state its correctness/performance evidence.

| Stage | Deliverable | Exit evidence |
|---|---|---|
| 0. Reviewed design and baseline | Review this spec; restore the BLAKE2b guest entry and profiler adapter, pin inputs with a derived T27 domain, capture current CPU stage times and donor references | Matching inputs/timers, current proofs verify, exact revisions recorded |
| 1. First complete GPU path | Add opt-in backend selection and Spartan outer-remainder rounds on `jolt-metal`, initially retaining current CPU preparation; retain CPU for other work | Focused family check, small verified proof, affected-stage timing |
| 2. Trace commitment | Port current-plan selector packing and commitment computation with bounded buffers and implicit zero suffixes | Current CPU commitment agreement, verified proof, preparation included in timing |
| 3. Root opening | Port coefficient packing and sparse/chunked decomposition as separate changes; measure remaining PCS cost | Current-plan agreement, complete verified proof, stage-8 timing |
| 4. Instruction/bytecode families | Shared compact instruction source; read-RAF, RA virtualization, Booleanity and claim reductions, with current CPU fast paths retained | Address/cycle agreement, qualified handoff, stage and full-proof improvement |
| 5. Connected Spartan families | Outer/product preparation and rounds, then instruction/register reductions that reuse their state where profitable | Prefix/tail boundary coverage, full-proof verification, no duplicate trace-sized owner |
| 6. Remaining dominant work | Port expensive RAM/register families and recursive PCS operations selected by the refreshed profile | Current RAM/padding semantics, canonical PCS plans, sparse-case CPU routing, verified proof |
| 7. Overlap and thresholds | Add justified batch submission/residency reuse, tune CPU handoffs, remove remaining dominant host work | Same correctness checks, measured full-proof benefit, bounded peak memory |
| 8. Final comparison and cleanup | Run matched T27 suite, relevant nearby-size/padding checks, required checks; remove experiment-only code | Report all workload rates, target status, remaining bottlenecks, reproducible revisions |

Stage order after the baseline is driven by measured full-proof cost. In
particular, advance remaining recursive opening work if it limits throughput. Do not defer
all PCS work behind a complete sumcheck port, and do not require every donor
kernel to be ported if the optimized CPU path is already better.

The baseline and first GPU family must establish actual stage costs before
claiming the 20 MHz target is feasible. If the target remains unmet, report the
measured gap and continue with the largest justified optimization; do not mark
the performance objective complete on kernel-only results.

Implementation needs build space beyond the approximately 1 GiB free at initial
inspection. Identify disposable build outputs and obtain any required deletion
authorization before reclaiming them; preserve sources, benchmark evidence, and
other running work. This does not block spec review.

## References

- [Current kernel registry and proof session](../crates/jolt-kernels/src/backend.rs)
- [Current fused round loop and scheduler](../crates/jolt-sumcheck/src/prover.rs)
- [Current prefix/tail representation](../crates/jolt-kernels/src/optimized/prefix_product.rs)
- [Current semantic PCS sources](../crates/jolt-akita/src/trace_onehot/source.rs)
- [Current Akita profiling interval](../crates/jolt-prover/src/profile.rs)
- [Shared Metal arithmetic/runtime design](jolt-metal-field.md)
- [Shared Metal runtime](../crates/jolt-metal/README.md)
- [Donor Metal backend](https://github.com/a16z/jolt/tree/7aa549cc9ffce0abe1578609b233d8133d6c14f8/crates/jolt-kernels/src/metal)
- [Donor benchmark suite](https://github.com/a16z/jolt/blob/7aa549cc9ffce0abe1578609b233d8133d6c14f8/scripts/akita_metal_matrix.py)
- CPU padding improvements: `31e115820` (#2141), `e1d8e02b5` (#2182).


## InstructionInput native producer candidate

The candidate keeps CPU native round 0 and replaces first-bind operand
materialization plus subsequent packed/dense rounds. Its source is the canonical
`InstructionInputRow` extraction: four u64 operands and two raw immediate limbs
in a padding-free 48-byte record, plus one canonical u16 flag code per original
row pair. A bounded tile writes directly to shared `MaybeUninit` storage; there
is no host field-vector mirror or source-copy upload. First-bind output is
initialized once by disjoint completed tiles before it becomes a typed field
buffer. Later binds use distinct input/output allocations.

The CPU and GPU use the same flag-LUT owner. Packed words progress through
2/4/8/16 bits per flag; the fifth bind decodes the existing 16-bit LUT and writes
five dense flag columns. Dense GPU execution continues to a 4096-row CPU tail.
The fused reductions return q(0), the quadratic coefficient, and q(1) only when
the current equality endpoint requires it. Host code adds the constant suffix,
checks the cubic through the existing Gruen routine, and advances progress once.

At a full T27 prefix, compact source writes total about 6.125 GiB, initial
resident fields/words occupy 6 GiB, and the largest old/new bind overlap is
9 GiB. Source staging is normally bounded to 49 MiB. Admission checks single
buffer and working-set bounds; runtime allocation still accounts for other live
buffers. Setup owns only the device and pipelines. The member owns typed unused
bindings and witness-derived storage, which are released at CPU handoff.

The reusable typed initialization runtime change is independent of register
kernel routing. Signed-i128 conversion stays private to InstructionInput until
another active consumer requires that API. One existing reference-parity helper
covers multiple source tiles, an odd prefix, a nonzero tail, independent flags,
MIN/MAX immediates, every packed width, dense continuation, an exceptional GPU
equality endpoint, and the CPU handoff. Compilation, Metal validation and an
inclusive full-proof timing must pass before accepting this candidate; no
performance gain is inferred from its buffer sizes.
