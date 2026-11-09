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

Instruction RA and Booleanity retain the CPU's lazy branch tables, binding,
categorical startup, and output checks while offloading qualified round
reductions. They share one proof-owned compact device row buffer. The session
keeps a weak reference, so materialization by the last consumer releases it.

ReadRAF retains CPU address rounds and its grouped first cycle message. It
offloads lazy width-two/four messages, materializes directly at the existing
lazy-policy crossover, and keeps subsequent shrinking prefixes on the device
until the small CPU tail. Separate claim and selector extents retain their
implicit tails. Its packed claim byte shares the existing 32-byte instruction
row, which a proof-owned carry keeps alive for later RA and Booleanity use.
Focused round and Metal validation checks pass; the full-proof development
measurements above remain provisional.

The sparse register prototype at `5bd928644` passed its independent oracle,
Metal validation, and full proofs, but its data transfer cost erased the GPU
benefit. Preparation rose from 0.2682 to 1.4401 s, including 1.1721 s packing
and upload, while round time fell only from 1.9889 to 1.9240 s. The optimized
CPU implementation is restored. A further attempt must avoid the seed repacking
and transfer cost or use a later compact handoff; GPU dispatch time alone
(0.5208 s) is not evidence of an improvement. The frozen prototype and profile
remain in `registers-metal/` under the artifact directory.

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
dispatch. Coefficient packing still uses the current CPU kernel.

Stage 0 may retain its existing `DeviceK16Sources` windows for root folding.
For `P` physical prefix rows and `C` columns, their device payload is
`C * P + C * ceil(P / 32)` bytes: selectors plus ring classifications. Host masks
and packing buffers remain transient. The implicit suffix allocates nothing.
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
spent 0.9311 s packing/uploading and 0.3345 s on GPU execution. Residency has not
yet been measured. Its acceptance run must verify the same proof, report retained
bytes and reused window count, compare root-fold packing and total proof time,
and record peak RSS/footprint and memory after the proof's source handles drop.

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
