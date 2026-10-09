# Akita: one whole-bytecode commitment object

| Field | Value |
| --- | --- |
| Status | Implemented; functional validation passed |
| Date | 2026-10-09 |
| Repository | LayerZero-Research/jolt |
| Inspected base | `origin/main`, `eea40ca8960d9396da08a09d0835dedd069f2173` |
| Implementation branch | `omid/akita-single-bytecode` |
| Akita dependency | `d98400c555a7fc779bb4e29c35fbd4adad3232d1` |

## Decision and assessment

Remove application-level bytecode-table chunking from Akita. In committed-program
mode, Akita commits the entire padded expanded bytecode as one bounded-dense
polynomial in one independent group, followed by the existing program-image
group. Dory keeps its configurable `BytecodeChunk(i)` commitments.

The idea is sound as a protocol simplification. Akita already opens heterogeneous
groups at their own local points and arities; the bytecode object does not need
to fit the trace group's domain. At the inspected base, Akita committed-program callers
already exercised the mathematical one-object case by passing a count of one.
This change makes that case the only representable Akita design and removes its
chunk-specific interface, state, identities, and proof payloads.

There is a narrower claim than “chunking is never useful in lattices.” Splitting
a large dense object can still affect memory, producer schedule selection, and
opening cost. Akita's group abstraction removes the requirement to split
bytecode for trace-size compatibility; it does not prove that a single group is
always faster. Adopt the singleton design for its simpler contract, and validate
resource limits and schedule admission before implementation handoff.

This spec does not change Akita's underlying PCS construction or security policy.
Any need to expand the admitted size range is a separate, explicitly audited
schedule/deployment change, not an implicit consequence of this cleanup.

## Scope

“Bytecode chunking” here means dividing the padded static bytecode rows into
contiguous tables and making each table a separate committed polynomial.

Keep all of the following:

- `BytecodeRa(i)` address-bit decomposition and its one-hot checks.
- Instruction/RAM address chunks, balanced increment digits, and carry.
- `AkitaChunkProfile::{Single, Two, Four, Eight}`: these configure internal
  response/fold geometry, not the number of bytecode tables.
- Dense producers' existing internal response chunks and the fixed conservative
  eight-chunk catalog policy.
- Full-program mode, optional advice, and the separate program-image commitment.
- The committed-bytecode value claim reduction and its stage-6b/7 scheduling.
- The current prohibition of Akita committed-program mode with `field-inline`,
  and the mutual exclusion of `akita` and `zk`.

Do not add a new mode, hidden automatic partitioning, ignored count parameter,
or compatibility alias that calls the whole bytecode “chunk zero.”

## Current implementation and the size issue

The current Akita preprocessing API accepts `bytecode_chunk_count`.
`committed_program_packing_plan` makes that many singleton bytecode plans;
`commit_direct_program` materializes and commits each plan. Verifier preprocessing
stores a vector of direct commitments plus a separate count. Stage 8 reconstructs
the plans and opens them together with advice and `OneHotTrace`.

With padded bytecode length `B`, count `C`, and lane capacity `L`, each current
bytecode object has `L * B / C` logical coefficients. At the inspected revision,
`L = 512`; the implementation must continue deriving it from the canonical lane
layout instead of hard-coding 512.

The trace's one-hot columns have `K_chunk * T` coefficients. A particular
execution can be short while the static program is large. Akita does not impose
`L * B <= K_chunk * T`: its whole-bytecode group has its own physical arity and
local opening point. Grouped setup still needs enough capacity for all producer
profiles, including auxiliary groups larger than the final trace group.

Jolt's existing shared precommitted reduction schedule can also gain extra rounds
when bytecode or the image exceeds the trace domain. Independent PCS groups do
not eliminate that PIOP scheduling requirement. Do not remove the shared reference
schedule as part of removing bytecode splitting.

Relevant source:

| Owner | Current source |
| --- | --- |
| Akita preprocessing | `crates/jolt-prover/src/akita/preprocessing.rs` |
| Direct program witnesses/commitments | `crates/jolt-prover/src/akita/witness.rs` |
| Akita packing geometry and roles | `crates/jolt-claims/src/protocols/jolt/lattice/packing.rs` |
| Bytecode encoding | `crates/jolt-kernels/src/committed_program.rs` |
| Shared reduction geometry | `crates/jolt-claims/src/protocols/jolt/geometry/claim_reductions/bytecode.rs` |
| Canonical symbolic reduction | `crates/jolt-claims/src/protocols/jolt/relations/claim_reductions/bytecode/` |
| Preprocessing wire and digest | `crates/jolt-verifier/src/preprocessing.rs` |
| Validated reduction schedule | `crates/jolt-verifier/src/stages/mod.rs` |
| Prover/verifier final groups | `crates/jolt-prover/src/akita/stage8.rs`, `crates/jolt-verifier/src/stages/stage8/akita.rs` |
| Transcript absorption | `crates/jolt-verifier/src/verifier.rs` |
| Exact producer/grouped admission | `crates/jolt-akita/src/schedule_registry.rs`, `crates/jolt-akita/schedules/README.md` |

## Target contract

An Akita committed program has exactly two direct commitment objects:

1. `ProgramBytecode`: all padded expanded instruction rows and all canonical lanes.
2. `ProgramImageInit`: the existing padded initial program-image words.

Use named fields for this pair in verifier preprocessing, prover-retained objects,
and packing plans. Do not serialize a variable-length collection and a count that
must be kept in agreement. Generic Akita batch assembly may still use vectors:
those vectors describe heterogeneous groups, not bytecode partitions.

The committed-program batch order is:

```text
UntrustedAdvice? -> TrustedAdvice? -> ProgramBytecode -> ProgramImageInit -> OneHotTrace
```

Keep the distinction between batch order and stage-0 absorption order. Currently
stage 0 absorbs the trace commitment first, then advice, then direct program
commitments. Preserve that order, replacing the indexed chunk sequence with the
single bytecode commitment followed by the image. Both paths must use canonical
helpers; enum ordering must not determine group order.

### Singleton identity

Add `JoltCommittedPolynomial::ProgramBytecode` for the Akita whole-table object.
Append it after existing real variants, before any test-only tail, following the
repository's feature-gated-tail and append-only enum rules. Assign any explicit
codec tag without recycling or shifting an existing tag. Audit the actual serde
and positional codecs, derive macros, exhaustive matches, and golden fixtures.
Follow the existing unconditional lattice IDs if needed by `jolt-akita`, which
uses lattice packing geometry without selecting the `jolt-claims/akita` feature.
Do not introduce a feature gate that breaks that adapter build.

Keep `BytecodeChunk(usize)` for Dory with its existing positions and semantics.
Akita must neither produce nor accept it as an opening target. Remove Akita
chunk-index role assignment and use one fixed `program_bytecode` role, order 2;
the program image has order 3. Optional advice does not renumber those roles.

Give the whole-bytecode layout a new domain, for example
`program-bytecode-whole-v1`, and a fresh explicit packed-object ID tag. Bind the
identity, logical and physical dimensions, lane layout, trace order, and the
canonical packing metadata. Preserve the current canonical metadata binding;
do not invent a second encoding of the lane layout.

### Shape and supported envelope

For the whole table:

```text
logical_vars = committed_lane_vars() + log2(B)
physical_vars = max(MIN_DENSE_OBJECT_NUM_VARS, logical_vars)
```

`B` is the validated power-of-two bytecode length produced by preprocessing.
Keep the current slot-zero/zero-prefix embedding when the logical arity is below
the dense schedule floor. Preserve the common encoder's `TracePolynomialOrder`
encodings and exact point permutations. Akita's production preprocessing currently
admits only `CycleMajor`; keep its `AddressMajor` rejection rather than expanding
the backend's supported envelope in this change. Dory retains both orders.

The inspected direct-object bound is 34 physical variables and the floor is 14.
With nine lane variables, the single-table structural bound is consequently
`B <= 2^25` padded rows. This is an arity limit, not a practical memory guarantee
or proof that every such grouped request has an admitted schedule. Existing
splitting could represent bigger whole tables through smaller pieces; that
capability is intentionally removed. Reject oversized or unsupported shapes
with typed errors before allocating coefficient tables.

Resolve the exact bounded-dense producer row and heterogeneous grouped row from
trusted schedule artifacts. A proof must never choose a replacement row, source
contract, chunk profile, or shape. Do not promote bytecode to a full-width source
contract merely to make planning succeed.

## Implementation plan

### 1. Remove the Akita count at public and serialized boundaries

Remove the count argument from Akita `preprocess_committed`,
`preprocess_committed_with_advice`, and `commit_direct_program`, and update their
production/test callers. Preserve Dory APIs and SDK/recursion count arguments
where they actually select the Dory committed path.

Under `akita`, replace `direct_program_commitments: Vec<_>` and
`bytecode_chunk_count` with named `bytecode_commitment` and
`program_image_commitment` fields. Keep `trace_order`, metadata, memory layout,
and maximum trace length. Replace the prover's variable-length direct object
storage with named bytecode/image objects, retaining each object's plan, witness,
commitment, and opening hint.

Keep the Dory vector and inferred chunk count. The common
`bytecode_chunk_count()` accessor becomes Dory-only; Akita must not implement it
by returning one. Likewise, remove count from Akita's validated
`CommittedProgramSchedule` representation. Use compile-time backend-specific
fields/constructors at the existing feature seam, rather than a second runtime
backend selector.

### 2. Build one whole-table packing plan and witness

Replace Akita `PrecommittedPackingShape.bytecode_chunks`, per-chunk row size, and
`PrecommittedPackingPlan.bytecode_chunks` with whole-table dimensions and a named
bytecode plan. In committed-program mode the image is required. Remove optional
or generic plan surfaces that lose their last production caller.

Create one coefficient grid containing all `B` rows. Refactor the existing
materializer so singleton and Dory chunked producers share the canonical row/lane
encoder, immediate validation, and ordering logic. The Akita producer returns
one coefficient vector, not a one-element `Vec<Vec<_>>` obtained by requesting
chunk count one. Dory keeps a chunked materializer and all its legal counts.

Retain zero padding, no-op rows, register one-hot lanes, flags, original PC,
immediate encoding, and the existing signed-field interpretation. Size validation
must happen before allocation and retain checked arithmetic. Build the bytecode
grid once during preprocessing, retain it with its opening hint, and consume it
directly in Stage 8. Avoid an additional full-table clone or a second row encoder.

### 3. Specialize the existing reduction to a singleton target

Do not delete `BytecodeClaimReduction`. Read-RAF produces claims about selected
instruction attributes; a PCS opening of an unrelated polynomial would not
authenticate those claims. The reduction still connects them to `ProgramBytecode`.

The current split uses high address bits to weight separate chunks. For the whole
table there are no dropped bits: retain the full bytecode address point, and the
partition weight is the scalar one. The lane-weight, equality-polynomial,
stage-batching, and skipped-round scale computations remain canonical in
`jolt-claims`.

Refactor the existing reduction shape/layout to carry backend-selected targets:
Dory retains validated chunk count and indexed outputs; Akita carries a singleton
target without count, partition index, dropped-bit state, or weight vector.
Replace the raw `(dimensions, chunk_count)` shape where necessary with a typed
shape consumed by the existing relation implementations. Keep one owner for the
shared algebra; do not create a second Akita copy of the input/output formulas.

Akita address-phase output carries one `ProgramBytecode` cell. Cycle-phase output
must encode exactly one of intermediate handoff or final singleton output,
according to the validated schedule; replace the existing parallel optional
intermediate/vector representation with a typed state for the Akita path. Dory's
wire and indexed outputs remain unchanged. Update claims derives and stage
output adapters so they enumerate the singleton identity directly.

The final singleton expression is the same reduction coefficient times the
`ProgramBytecode` opening. Simplify only the partition factor to one; the lane
evaluation, bytecode-point equality, and skip-round normalization are not one.
Add an unindexed derived output-weight identity for Akita, appended to the
existing public-value ID enum; Dory keeps `ChunkOutputWeight(i)` and its codec.
The existing canonical final-expression builder must select the target and
weight identities from the validated shape, with no alternate claim formula.
Keep all six Akita staged value wires, including the Store wire used by fused
increment consumers. Keep both cycle-completed and address-completed schedules.

Adapt reference and optimized reduction kernels to read one whole-table witness
in Akita and the existing chunk list in Dory. Share reduction math and row
encoding. Eliminate Akita's multi-chunk accumulation and weights; do not retain
an indexed implementation behind a fixed literal count as the final design.

### 4. Update validation, transcript, and grouped setup together

Update the validated schedule construction in both verifier entry paths,
stage-6b/7 derived values and output points, precommitted leaf assembly, witness
dispatch, and canonical opening-target ownership. `ProgramBytecode` is a
preprocessing-held object, never a trace-backed witness polynomial.

Change Akita stage-0 program absorption to accept the named pair (or a typed
reference to it). Absorb one bytecode commitment with an unindexed label such as
`program_bytecode_commitment`, then the image. Remove chunk-index absorption and
`split_last()` interpretation of program commitments. Full mode contributes no
program pair; committed mode contributes both.

Provision setup from the two actual physical producer arities, preserving each
producer's fixed bounded source/profile. Keep the generic heterogeneous planner,
the selected trace response profile, and catalog audit. The bytecode producer's
arity grows relative to the previous `C > 1` configuration; regenerate adapted
verifier preprocessing accordingly. Do not change base `.aks` catalogs unless
exact-row admission demonstrates a need.

Stage 8 obtains the singleton logical leaf from the canonical reduction, applies
the existing physical embedding, and adds one bytecode group plus the image.
Validate roles, one polynomial per auxiliary group, dimensions, layout digests,
and point/evaluation lengths before PCS verification. Missing, extra, indexed,
duplicated, reordered, or stale program objects must be rejected; never silently
take the first element of an old chunk list.

### 5. Coordinate wire changes without changing Dory

This intentionally changes Akita preprocessing serialization, reduction claim
payloads, layout digests, and transcript binding. Existing Akita proofs and
preprocessing caches must be regenerated, including verifier/recursive-guest
fixtures that embed setup or preprocessing. Do not add migration aliases or
deserialize old multi-chunk payloads as the singleton format.

Bump the preprocessing digest domain for the changed Akita format. At the inspected base, the
domain was selected by `field-inline`, not PCS; introduce only the compile-time
distinction needed to give Akita a new domain while preserving current Dory
domains (`v2` ordinary, `v5` field-inline). Keep Dory enum discriminants, explicit
ID tags, serialization, absorption, reduction formulas, and commitments stable.
Update the concrete outer proof/fixture encoding guards wherever the audited
codec requires them; a changed digest alone must not be assumed to make an old
struct deserialize safely. Malformed/stale transport must fail with typed errors.

## Validation and acceptance

The implementation is complete only when Akita has no bytecode table count,
indexed bytecode role, chunk vector, or Dory `BytecodeChunk(i)` opening on a
production path. Search by semantic use: variables called bytecode chunks inside
`BytecodeRa` address decomposition are valid and must not be removed.

Required behavior coverage:

| Case | Observable contract |
| --- | --- |
| Full mode | No direct program groups; unchanged execution acceptance |
| Committed mode | Exactly one whole-bytecode group and one image group |
| Short execution, larger bytecode | Proof verifies with bytecode group larger than native trace group |
| Image dominates bytecode/trace | Existing image reduction and local group point remain valid |
| Small bytecode | Dense floor embedding verifies with independent polynomial evaluation |
| Trace order | Akita CycleMajor encoding/point/digest agree; AddressMajor remains rejected; Dory retains both orders |
| Advice combinations | Absent/present trusted/untrusted advice with the same singleton program contract |
| Internal trace profiles | Single/Two/Four/Eight do not change program group count |
| Setup transport | Serialized verifier setup/preprocessing verifies without source catalog discovery |
| Tampering | Changed bytecode leaf, commitment, role, order, shape, or digest is rejected |
| Unsupported shapes | Typed error before whole-table allocation; no hidden splitting |
| Dory | Counts 1, 2, and the 256 boundary retain their existing contracts |
| Field-inline | Existing committed-mode rejection and full-mode behavior retained |

Reuse existing committed-program and grouped-capacity tests; replace the Akita
one/two-bytecode-count loop with the singleton contract while retaining internal
trace-profile coverage. Preserve independent ground truth, frozen fixtures, and
soundness tests. Do not keep old-vs-new implementations or temporary parity probes
as permanent tests. Add a focused small-trace/whole-bytecode regression only where
the existing tests do not cover the combined Jolt reduction and PCS boundary.

During implementation, run affected tests sequentially with `cargo nextest`,
including Akita committed-program/advice tests, transported verifier fixtures,
reduction geometry, malformed-input rejection, and schedule admission. Run the
clear, ZK, and Akita acceptance matrices and verifier-fixture coverage required
for shared stage/public-value/transcript changes. Preserve the field-inline
matrix. Run both repository-required host Clippy configurations, the affected
Akita configuration, and formatting. Full suites remain CI work unless a failure
requires broader local checks.

Before performance sign-off, measure preprocessing time and peak memory, commit
time, reduction time, PCS opening time, verification time, and proof/setup bytes
for a large-bytecode short trace and an ordinary trace. Record bytecode/image
sizes, trace length, selected producer/grouped rows, and internal trace profile.
Compare the chosen whole-table policy with historical deployed chunk counts as a
manual transition benchmark, not a permanent equivalence oracle. If the whole
table exceeds the deployment's memory or schedule envelope, report that limit;
do not restore an application-level chunk knob inside this task.

## Documentation updates for the implementation

Update `book/src/how/akita.md`, `book/src/how/architecture/opening-proof.md`,
`specs/lattice-claims.md`, and `crates/jolt-akita/schedules/README.md` to distinguish
the whole-bytecode group from internal PCS chunks. Update Akita examples and
preprocessing/fixture instructions. Preserve Dory's committed-bytecode spec and
its CLI documentation. Remove misleading Akita descriptions of chunk counts,
chunk-index roles, and multiple direct bytecode groups.

## Evidence and limits of this specification

The assessment is based on the inspected fork revision and its pinned Akita
source, not on a blanket claim about lattice commitments. In particular:

- Current Akita Stage 8 already constructs local claims per direct program object.
- Existing `muldiv_e2e_akita_committed_program` exercises bytecode counts one and
  two under each internal trace profile; `advice_e2e_akita_committed_program`
  already uses one bytecode object with both advice kinds. These tests were read,
  not rerun as part of this specification.
- `grouped_opening_proves_advice_larger_than_the_trace_group` exercises an auxiliary
  object with 22 variables against a trace group with 16 variables, including
  transported verifier setup and transcript mismatch rejection. Its execution
  result is recorded below. This validates heterogeneous capacity at the adapter
  boundary; it is not a whole-bytecode Jolt end-to-end test.
- Arity bounds, catalog admission, and the materialized dense witness make
  performance and maximum usable program size deployment-dependent.

No implementation changes or new benchmark results are included in this spec.

Validation run on the inspected base, with no Rust changes:

```sh
cargo nextest run -p jolt-akita --cargo-quiet \
  -E 'test(grouped_opening_proves_advice_larger_than_the_trace_group)'
```

Result: one test passed (0.891 seconds test execution), 88 tests skipped. The
test covers the independent auxiliary-group capacity and transported-setup
behavior described above. It does not establish performance or catalog coverage
for every whole-bytecode size.

## Implementation record

The implementation uses appended `ProgramBytecode` and scalar `OutputWeight`
identities, with no Akita bytecode-table count parameter. Named bytecode and
image fields replace variable-length program-object lists in packing plans,
prover preprocessing, and verifier preprocessing. Cycle-phase output is an
exclusive intermediate/final enum; the final claim is one scalar. The bytecode
kernel binds the whole value table as its final committed polynomial, without
an auxiliary copy for per-chunk claims. Dory keeps its partition vectors and
indexed identities. Both encodings call the canonical row/lane materializer.

Producer and grouped schedule admission precede whole-table allocation. The
logical arity law and structural maximum are checked by the canonical geometry
and packing plan. This does not guarantee sufficient deployment memory for
all structurally admitted sizes.

The Akita digest domains are
`jolt/program-preprocessing/akita-whole-bytecode/v1` and, for the field-inline
profile, `jolt/program-preprocessing/akita-whole-bytecode/field-inline/v1`.
Regenerate all Akita preprocessing and proofs, including full-program proofs.
Dory domains and frozen digest vectors are retained. Existing base schedule
catalogs are reused; the grouped requests contain the two singleton program
objects.

The committed muldiv regression checks that whole-bytecode arity exceeds the
native trace arity, exactly two program objects exist, and the bytecode identity
is `ProgramBytecode`. It transports preprocessing and proofs through bincode,
verifies across Single/Two/Four/Eight response profiles, and rejects an altered
bytecode claim. The advice fixture and verifier soundness sweeps exercise the
named commitments and new scalar claims.

Performance has not been signed off: no speedup or deployment-memory reduction
is asserted by this change. The benchmark plan above remains the requirement
for a separate performance claim.

### Completed verification

- All six guest acceptance matrices passed: nine ordinary guests in each of
  clear, ZK, and Akita, plus field_ops and muldiv in each field-inline mode.
- Akita committed-program/advice proofs, serialization transport, all four
  response profiles, geometry, commitment/claim tampering, and verifier
  completeness passed. The new wrong intermediate/final phase-state mutation
  was also rejected.
- The selected clear suite passed 50 tests, including bytecode/precommitted
  geometry, Dory committed-program verifier fixtures and tampering. The selected
  ZK suite passed 14 tests, including committed-program proving and BlindFold
  tampering. Existing Dory digest vectors passed in both instruction profiles.
- The field-inline runs passed 4 clear, 4 ZK, and 8 Akita tests. Akita's
  committed-mode rejection remains covered.
- Both full-workspace host Clippy commands passed, as did affected-crate Akita
  Clippy with and without field-inline, with fixtures and all targets.
- The 21 focused Akita schedule/catalog/grouped-capacity tests passed. Three
  tests reported transient nextest output-handle leaks under parallel execution;
  isolated reruns passed without that status. No code change was needed.
- The Fiat-Shamir inventory diff was reviewed: the indexed Akita bytecode
  absorption and its index were replaced by one labeled whole-bytecode
  absorption, and named preprocessing replaced the old list/count. Dory
  absorption, challenge sites, and scope annotations are unchanged. The frozen
  inventory passed again without regeneration.
- Formatting, repository style invariants against origin/main, and git diff
  whitespace checks passed. Full crate suites remain CI work.

The Jolt CLI was reinstalled after updating main. Workspace linting found an
ignored recursion macro generated by an older API. Its original was saved at
`/tmp/jolt-bytecode-validation/provable_macro.rs.before`; the existing build
script regenerated the current default. No tracked recursion files changed.
Validation logs are in `/tmp/jolt-bytecode-validation/`.
