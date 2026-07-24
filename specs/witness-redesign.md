# Spec: Witness Redesign — Atomic Witnesses, Bundles, and the Trace→Witness Stream

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @moodlezoup, Claude            |
| Created     | 2026-07-15                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

`jolt-witness` is a trace→witness transformation layer that grew without a first-class notion of
the thing it transforms *into*. Witnesses are smeared across idioms: some grouped by protocol
stage (`stage5.rs`, `stage6.rs`), some by component (`ram.rs`, `registers.rs`), one giant match
(`trace_virtual_value`) for the cycle-domain virtuals, a parallel needs/row apparatus for the
committed batch stream (`JoltVmBatchNeeds`/`JoltVmBatchRow`), and a namespace abstraction
(`WitnessNamespace`, `OracleRef<N>`) whose generality serves two concrete protocols and whose
`ChallengeId`/`PublicId` associated types have zero uses. Meanwhile the crate's real consumers
foreshadow the missing design: `Stage5InstructionReadRafRow` and `JoltVmStage6Row` are witness
bundles — typed rows of exactly the values one consumer needs — hand-rolled because the layer
they belong to didn't exist.

This spec rebuilds the crate around that layer. A **witness** is an atomic value newtype
(`RamAddress(u64)`, `LookupIndex(u128)`, …) with a single-sourced derivation from a trace row.
A **bundle** is a `#[derive(WitnessBundle)]` struct whose fields are witnesses — a consumer's
data flow, stated as a type, with `#[opening(…)]` field annotations tying fields to jolt-claims
ids in the style of the `OutputClaims` macro. A **consumer** is the sketch-era
`StreamConsumer` (`type Witness; fn consume(&mut self, &[Self::Witness])`), and one
`stream_witnesses` pass drives a statically-known consumer set over the trace — the
trace→witnesses map is one-to-many, the witnesses→consumers map is many-to-many, and both are
explicit in types rather than in runtime id lists. The naive interpreter, which is
intrinsically id-driven, keeps a single `oracle_table` served by one hand-written exhaustive
match over the jolt-claims id enums — totality enforced by the compiler, no registry machinery.
The design is streaming-ready by construction: every contract is sequential-over-ranges (random
access is deliberately inexpressible), so the future streaming prover is a second caller of the
same pass, not a redesign.

`WitnessNamespace` is deleted rather than preserved: jolt-witness defines **no id vocabulary at
all** — `JoltPolynomialId` and the field-inline ids already live in jolt-claims, and
`OracleRef<JoltVmNamespace>` merely duplicated them. Every witness value is byte-pinned through
the proofs the byte-diff harness compares, so the entire migration is mechanically gated.

## Intent

### Goal

Rebuild `jolt-witness` as a typed trace→witness streaming layer: atomic witness newtypes with
single-sourced extractors, derive-composed consumer bundles annotated with their protocol ids,
a fused single-pass driver over statically-typed consumer sets, and one exhaustive id-indexed
table path for the naive interpreter — with no in-crate id types, no namespace generics, and no
API that presumes random access to the trace.

Key abstractions:

- **Atomic witness** (crate root, row-free): a value newtype per witness.

  ```rust
  pub struct RamAddress(pub u64);
  pub struct LookupIndex(pub u128);
  pub struct OpFlag(pub bool);        // indexed families: which flag is bound at the use site
  ```

- **`Extract`** (trace-backend module — the only row-coupled code in the crate): the
  single-sourced derivation. No `RowFacts`, no memoization; extractors recompute from row
  accessors. The two irreducible non-row inputs are the lookahead window (the `Next*` family
  is a function of rows t and t+1, with padding semantics at T−1) and the environment
  (`PC` requires `preprocessing.bytecode.get_pc`; RAM addresses go through
  `memory_layout.remap_word_address`).

  ```rust
  pub(crate) trait Extract: Sized {
      fn extract(row: &TraceRow, next: Option<&TraceRow>, env: &WitnessEnv<'_>)
          -> Result<Self, WitnessError>;
  }
  // WitnessEnv: preprocessing + memory layout (+ pc-lookup cache as an env detail)
  ```

- **`#[derive(WitnessBundle)]`** (new `jolt-witness-derive` crate): a consumer's input as a
  struct; fields are atomic witnesses; annotations bind fields to protocol ids.

  ```rust
  #[derive(WitnessBundle)]
  pub struct InstructionReadRafWitness {
      lookup_index: LookupIndex,                    // fact field: no protocol id
      table_index: TableIndex,
      #[opening(virtual = "OpFlags", flag = "InterleavedOperands")]
      interleaved: OpFlag,
  }
  ```

  The derive generates: the trace-backend constructor `from_row(row, next, env)` composing the
  field extractors; the bundle's annotated id set (for stage-0 validation and, later, linking a
  kernel's consumed witnesses to the opening claims it produces); and a consistency test per
  annotated field (bundle column ≡ `oracle_table(id)` on a sample trace) so the typed path and
  the id path cannot drift. If a profile ever shows per-field duplicate instruction decoding
  mattering, the hoist lives inside the generated `from_row` — generated code, not public API.

- **`StreamConsumer` + `ConsumerSet` + `stream_witnesses`**: the fused pass. Membership is
  static (a tuple type — the same fact the claims/verifier derives exploit: a stage's member
  list is known at compile time); presence is runtime (`enabled`/`Option` per slot, like batch
  members today).

  ```rust
  pub trait StreamConsumer: Send + Sync {
      type Witness: WitnessBundle;
      fn consume(&mut self, chunk: &[Self::Witness]);
  }

  pub fn stream_witnesses<C: ConsumerSet>(
      source: &impl RowSource,       // trace-backed today; segment-backed later
      range: CycleRange,             // [0, T) today; segments later
      chunk_size: usize,
      consumers: &mut C,
  ) -> Result<(), WitnessError>;
  ```

  One row walk per pass; each live consumer's bundle built via its generated `from_row`;
  chunked delivery. Ownership is plain: the caller owns the tuple and lends `&mut` — the
  sketch-era `Arc<Mutex<dyn StreamConsumer>>` registration graph is replaced by a type
  parameter. Segment parallelism (later) is one consumer-set instance per segment plus
  kernel-side digest aggregation, exactly as in the original streaming sketch.

- **`oracle_table` + the exhaustive match** (the naive interpreter's path — the one consumer
  that is irreducibly id-driven, since it discovers `Opening` leaves while walking an `Expr`):

  ```rust
  pub trait JoltWitnessOracle<F: Field> {     // object-safe: what &dyn consumers need
      fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError>;
      fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<F>, WitnessError>;
      fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError>;
      fn committed_batch_stream(&self, …) -> Result<…, WitnessError>;
  }
  ```

  The trace backend's `oracle_table` is **one hand-written exhaustive match** over
  `JoltPolynomialId` — each arm a one-liner dispatching to the same atomic extractor the
  bundles use (or to a private stateful materializer for the grid family). No wildcard arm:
  adding an enum variant in jolt-claims breaks the build until the variant is mapped or added
  to the explicit exclusion arm (protocol intermediates; ids served by other providers). The
  only macro in the crate is the bundle derive; where one function suffices, it's a function.

- **Grid/stateful witnesses** (`RamVal`, `RamRa`, `RegistersVal`, `Rs1Ra`/`Rs2Ra`/`RdWa`,
  `RamValFinal`, virtual `InstructionRa`): their `(init, advance, read)` sequential
  reconstruction becomes a private materializer behind `oracle_table`. Streaming consumers
  never see dense K×T grids — they consume the per-cycle delta witnesses (`RamAddress`,
  read/write values, register writes) and fold their own state, which is what the streaming
  sketch's consumers already did.

- **Committed batch stream**: kept as-is in role — stage-0 commitment is *one* consumer whose
  column list is proof-config-dependent with runtime arity (`InstructionRa(0..d)`, advice
  presence), so its dynamism stays contained inside it. Internally it re-expresses over the
  atomic extractors, deleting `JoltVmBatchNeeds`/`JoltVmBatchRow`.

- **No namespaces**: `WitnessNamespace`, `OracleRef<N>`, and the uninhabited namespace enums
  are deleted. Ids come from jolt-claims (`JoltPolynomialId` *is* the former
  `OracleRef<JoltVmNamespace>`). Field-inline becomes a sibling concrete backend speaking its
  own jolt-claims ids through the same pattern (atomic witnesses, exhaustive match). Error
  labels survive as plain `&'static str`.

### Invariants

1. **Sequential access only.** No public API of this crate exposes or requires random access
   to trace rows (`fn value(&self, index)` is deliberately inexpressible): every contract is a
   walk over a `CycleRange`. This is the streaming-readiness invariant — a checkpointed,
   re-emulating trace source can implement every signature honestly (random access would cost
   half a checkpoint interval per call and make default materialization quadratic; stateful
   witnesses would cost O(t) per query).
2. **Single-sourced derivation.** Each witness's derivation logic exists in exactly one
   `Extract` impl; the bundle path and the `oracle_table` path both dispatch to it, and the
   derive-emitted consistency tests pin the two paths together.
3. **Totality with explicit exclusions.** The `oracle_table` match has no wildcard arm. Every
   `JoltPolynomialId` variant is either mapped to an extractor/materializer or listed in the
   explicit exclusion arm with its reason (protocol intermediate owned by `ProofSession`;
   preprocessing-sourced, served by the committed-program path). A new enum variant is a
   compile error until classified.
4. **Byte-pinning.** Witness values are byte-pinned through proofs: the byte-diff harness
   (10 tests, both trace orders, advice and committed-program modes) passes unchanged after
   every slice. No slice may change any witness value.
5. **Round-free.** No rounds, challenges, digests, or binding in this crate. Sumcheck
   round structure (2^i folding, head/tail aggregation) lives in kernels and the future
   streaming engine, which consume this crate only through passes.
6. **Row coupling confined.** `TraceRow`/jolt-program types appear only in the trace-backend
   module. Atomic witness newtypes, bundle types, `StreamConsumer`, and `ConsumerSet` are
   row-free — a future backend (GPU-resident, fixed-column) constructs the same bundle types
   from its own representation.
7. **No in-crate ids.** jolt-witness defines no identifier types; all ids are jolt-claims'.
8. **Validated bundles.** A bundle's annotated id set is checkable at stage 0 against the
   backend's servable set (derived from the match, never curated) before proving starts.

### Non-Goals

- **The streaming engine.** The `prove_batch` pass hook, per-round consumer registration,
  digest aggregation, head/tail segment reconciliation, and streaming kernels are future
  jolt-sumcheck/jolt-kernels work. This spec delivers the substrate they call
  (`stream_witnesses` over ranges) and proves it live by using it for today's bundle
  materialization — nothing round-shaped lands here.
- **Trace segmentation.** `TraceSource::segments()`, checkpoint-seeded state
  (tracer's `LazyTraceIterator` snapshots), and segment-overlap windows are deferred with the
  engine. The `CycleRange` parameter and the `StatefulWitness` init hook are the reserved
  seams; `[0, T)` is the only range exercised now.
- **Performance work.** This is a re-plumbing with values pinned byte-identical; no witness
  generation speedups are claimed. (The `RowCtx`/laziness apparatus considered during design
  is explicitly *not* built.)
- **GPU or other exotic backends.** The second backend delivered here is a fixed-column test
  backend; anything further waits for a consumer.
- **Renaming jolt-program/tracer types** or otherwise reshaping the trace representation.
- **Changing which polynomials exist** or how any value is derived — jolt-claims ids and
  semantics are inputs to this spec, not outputs.

## Evaluation

### Acceptance Criteria

- [ ] The byte-diff harness (`-p jolt-prover --features prover-fixtures`, 10 tests) passes
      unchanged after every execution slice, and the full workspace suite + clippy under
      `host` and `host,zk` stay green throughout.
- [ ] Every cycle-domain arm of the old `trace_virtual_value` exists as an atomic witness with
      its own `Extract` impl; the function itself is deleted.
- [ ] `oracle_table`'s match is exhaustive with no wildcard: a test (or the build itself)
      demonstrates that an unclassified `JoltPolynomialId` variant fails compilation, and the
      exclusion arm's reasons are asserted in a test.
- [ ] `#[derive(WitnessBundle)]` generates `from_row`, the annotated id set, and per-annotated-
      field consistency tests; a UI/snapshot test covers the derive including the indexed-
      family attribute form.
- [ ] `Stage5InstructionReadRafRow` and `JoltVmStage6Row` (and their traits) are replaced by
      bundles declared next to their consumers; no stage-named module remains in jolt-witness;
      the prover's `W:` bound names the new backend trait instead of stage-row traits.
- [ ] Bundle materialization is implemented *via* `stream_witnesses` (a collecting consumer),
      so the pass driver is live, tested surface from day one — not speculative API.
- [ ] `JoltVmBatchNeeds` and `JoltVmBatchRow` are deleted; the committed batch stream produces
      byte-identical chunks over the atomic extractors (pinned by the byte-diff commitment
      path and existing stream tests).
- [ ] `WitnessNamespace`, `OracleRef`, and `namespace.rs` are deleted; `grep -r
      "WitnessNamespace\|OracleRef" crates/` returns nothing; consumers use
      `JoltPolynomialId` directly.
- [ ] A `FixedBackend` (stored columns) implements `JoltWitnessOracle` and serves at least one
      kernel unit test — the second implementor that validates the seam, and the replay
      substrate the eval-harness spec's slot fixtures want.
- [ ] `field_inline` is migrated to the same pattern (concrete backend, own ids, exhaustive
      match) and its test suite passes under `--features field-inline`.
- [ ] Stage-0 validation checks every bundle id set and the proof config's requested id set
      against the backend's servable set before witness generation.

### Testing Strategy

The byte-diff harness is the primary oracle: every witness value flows into commitments,
sumcheck messages, and openings that are compared byte-for-byte against `jolt-prover-legacy`
across base/advice/committed/address-major configurations — any derivation drift fails loudly.
Per-slice: `cargo nextest` on jolt-witness (both feature modes), jolt-kernels, jolt-prover;
workspace suite before each commit. New unit surface: derive UI tests, per-witness extractor
tests (migrating the existing `tests.rs` assertions onto named witnesses), the generated
bundle-vs-table consistency tests, exclusion-arm classification test, and `FixedBackend`-served
kernel tests. Existing jolt-witness tests are ported, not weakened — the current
materialization and stream assertions all have direct new homes.

### Performance

No improvement claimed; no regression tolerated beyond noise. The known cost delta is
per-field instruction re-decoding inside a bundle's `from_row` (accepted by design; the
contained escape hatch is a derive-level hoist, invisible to definitions). The committed batch
stream's one-pass structure is preserved. Byte-diff harness wall time is the informal
regression canary; a span objective over witness generation becomes available once
`specs/span-objectives.md` lands (the pass driver is a natural span). No existing jolt-eval
objectives are expected to move; no new ones are added by this spec.

## Design

### Architecture

Three maps, three homes:

```
trace rows ──(one-to-many: Extract impls, trace-backend module)──▶ atomic witnesses
atomic witnesses ──(many-to-many: #[derive(WitnessBundle)] structs)──▶ bundles
bundles ──(StreamConsumer / ConsumerSet / stream_witnesses)──▶ kernels & commitment
```

**Crate layout** (jolt_vm; field_inline mirrors it in miniature):

```
src/
  witnesses/           # atomic newtypes + Extract impls, one file per family-of-convenience
    pc.rs registers.rs ram.rs operands.rs flags.rs lookups.rs increments.rs one_hot.rs …
  bundle.rs            # WitnessBundle trait, re-export of the derive
  consumer.rs          # StreamConsumer, ConsumerSet (tuple impls), stream_witnesses
  backend/
    trace.rs           # TraceBackend (née TraceBackedJoltVmWitness): WitnessEnv, the
                       #   exhaustive oracle_table match, grid materializers, committed stream
    fixed.rs           # FixedBackend: stored columns, for tests and slot fixtures
  shape.rs error.rs chunk.rs
```

File grouping under `witnesses/` is packaging, not taxonomy — the unit is the newtype, and
nothing dispatches on modules. The stage-vs-component organization dispute dissolves because
stages now *request* (bundles) rather than *own* (modules).

**Object safety split.** The naive interpreter holds `&dyn JoltWitnessOracle` (id-indexed,
object-safe). Bundle materialization and `stream_witnesses` are statically dispatched —
generic over the consumer tuple — matching how stage recipes are already monomorphic over `W`.
The two paths meet at the `Extract` impls, and the generated consistency tests keep them met.

**Where dynamism legitimately survives** (and is contained): the naive interpreter (an
interpreter — runtime ids are its job) and the committed column list (config-dependent arity
inside the one commitment consumer). Everything else — every hand kernel, every future
streaming consumer — states its data flow as named struct fields, compile-checked.

**Streaming, later, in full:** the engine adds a per-round phase to the batched prover —
collect the round's consumer set, run `stream_witnesses` per segment (`rayon` across segments,
one consumer-set instance per segment), aggregate digests kernel-side, then absorb/challenge as
today — and jolt-program grows `TraceSource::segments()` returning checkpoint-seeded row
iterators with one row of overlap for the lookahead window. Nothing in this crate changes shape
for that: the range parameter narrows, `StatefulWitness::init` gains a checkpoint-seeded
constructor, and the same bundles flow. The streaming prover is a `JoltBackend` value in
jolt-kernels, per the backend seam.

### Alternatives Considered

- **A witness catalog + registry macro** (fn-pointer entries, declared fact-needs, lazy
  memoized `RowCtx`, id-superset stream ids, source-partitioned capability traits). Rejected as
  over-mechanized: the needs/laziness apparatus solved a fusion problem that unconditional
  per-row recomputation handles at acceptable cost; the registry duplicated what one exhaustive
  match provides; capability traits at "component" granularity drew arbitrary boundaries. This
  spec is that design after three rounds of pruning — the surviving ideas are the atomic unit,
  totality, and single-sourcing.
- **One producer trait per witness (+ marker aggregation).** Canonical taxonomy, but ~66 traits
  whose primary consumers cannot use them: the naive interpreter and the fused pass are
  runtime-id/-set driven; Rust has no multi-trait objects, so the `dyn`-able marker supertrait
  would demand every producer and destroy partial-provider precision exactly where it would be
  consumed; config-dependent index ranges (`InstructionRa(0..d)`) are unverifiable by trait
  bounds anyway. Bundles + newtypes deliver the wanted static naming without the wall; if a
  statically-bound per-witness consumer ever materializes, traits can be generated from the
  same definitions.
- **Random-access oracle traits** (`fn value(&self, index)`). The signature only the
  materialized backend can implement honestly: checkpointed traces pay ~half a checkpoint
  interval per call (the provided `materialize` becomes O(T·interval)), stateful witnesses are
  O(t) per query, and `&self` forces interior mutability into any cursor-based implementor.
  Sequential ranges are the intersection all intended backends implement cheaply — and what
  the original streaming sketch already assumed.
- **Runtime id-list streaming API** (`stream_columns(&[OracleRef], …)`) as the primary
  consumer path. Works, but hides data flow in runtime values; the house precedent
  (claims/verifier derives) is types-and-derives for statically-known structure. Retained only
  where dynamism is intrinsic (interpreter, commitment).
- **Keeping `WitnessNamespace`.** Two concrete protocols, two dead associated types, and an
  `OracleRef` that duplicates `JoltPolynomialId`. Since this redesign touches every signature
  anyway, deleting the generics now is nearly free; keeping them taxes every new piece with an
  `N` that serves nobody. Field-inline's separation is preserved by being a separate concrete
  backend with separate id types — enforced by jolt-claims' types, not by phantom parameters.
- **Push-based consumer registration** (`Arc<Mutex<dyn StreamConsumer>>` per segment, as in
  the original sketch). The lock-and-share graph exists only because registration decouples
  ownership; a statically-typed consumer set owned by the caller gives the same one-pass
  fan-out with plain `&mut`, and per-segment instances restore the parallelism.

## Documentation

- Crate docs: rewrite `jolt-witness/src/lib.rs`'s header around the three-map picture
  (trace → witnesses → bundles → consumers) and the sequential-access invariant.
- `specs/`: this spec cross-references `specs/span-objectives.md` (the pass as a profiling
  span; non-goal there, substrate here) and the eval-harness spec (FixedBackend ↔ slot
  fixtures).
- No `book/` changes: internal architecture with no user-facing surface.

## Execution

Every slice ends with: byte-diff 10/10, jolt-witness tests both feature modes, workspace
suite, clippy `host` + `host,zk`, `cargo fmt`. Values are byte-pinned throughout; any slice
that moves a byte is wrong by definition.

1. **Atomic witnesses.** Introduce `witnesses/` newtypes + `Extract` + `WitnessEnv`; split
   `trace_virtual_value`'s arms into per-witness impls one-for-one; the old function becomes a
   thin dispatcher and then disappears. Internal only — no public API change. Migrate the
   per-value assertions in `tests.rs` onto named witnesses.
2. **The exhaustive table + namespace deletion.** Replace `provider.rs`'s dispatch with the
   no-wildcard match over `JoltPolynomialId` (grid materializers become private); delete
   `WitnessNamespace`/`OracleRef`/`namespace.rs`; `Shape` replaces `OracleDescriptor`; update
   jolt-kernels/jolt-prover imports (`dense_view` takes `JoltPolynomialId`). Classification
   audit happens here, forced by the compiler: map or exclude every variant, including the
   currently-unserved oddballs (`Rd`, `InstructionRaf`, `RamValInit`) with documented reasons.
3. **Bundles + the pass.** New `jolt-witness-derive` crate (`WitnessBundle`, `#[opening]`);
   `StreamConsumer`/`ConsumerSet`/`stream_witnesses`; bundle materialization as a collecting
   consumer over the pass. Recast stage-5/6 rows as bundles declared beside their kernels;
   delete `stage5.rs`/`stage6.rs` and their traits; swap the prover's `W:` bound to the
   backend trait. Derive UI tests + generated consistency tests land here.
4. **Batch-stream unification + second backend.** Re-express the committed batch stream over
   the atomic extractors; delete `JoltVmBatchNeeds`/`JoltVmBatchRow` and the per-kind
   `value_from_row` duplication in `streams.rs`. Add `FixedBackend` + one kernel unit test on
   it. Stage-0 bundle/config validation.
5. **Field-inline.** Same pattern in miniature: atomic witnesses, exhaustive match over its
   jolt-claims ids, concrete backend; delete its namespace plumbing. Gate additionally on
   `--features field-inline` tests and clippy.

Deferred, with named seams: `TraceSource::segments()` + checkpoint seeding (tracer/
jolt-program), the engine's per-round pass hook (jolt-sumcheck), streaming kernels + digest
aggregation (jolt-kernels), segment-overlap windows — all consumers of `stream_witnesses(range)`
as specified here.

### Relocation table

| Today | Becomes | Slice |
|---|---|---|
| `trace_virtual_value` match arms | per-witness `Extract` impls in `witnesses/` | 1 |
| `supported_trace_virtual` | the exhaustive match itself (exclusion arm) | 2 |
| `provider.rs` `oracle_table` dispatch | the no-wildcard match in `backend/trace.rs` | 2 |
| `OracleRef<JoltVmNamespace>` | `JoltPolynomialId` (jolt-claims) | 2 |
| `WitnessNamespace`, `namespace.rs`, `OracleDescriptor` | deleted; `Shape` | 2 |
| `ram.rs` / `registers.rs` state loops | private grid materializers behind `oracle_table` | 2 |
| `stage5.rs` row + trait | `InstructionReadRafWitness` bundle beside its kernel | 3 |
| `stage6.rs` row + trait | its bundle beside its kernel | 3 |
| ad-hoc `Vec<Row>` materialization | collecting consumer over `stream_witnesses` | 3 |
| `JoltVmBatchNeeds` / `JoltVmBatchRow` | deleted; committed stream over `Extract` impls | 4 |
| `streams.rs` `*StreamKind::value_from_row` | the same witnesses' `Extract` impls | 4 |
| `PcLookupCache` | `WitnessEnv` internal | 1–2 |
| `field_inline/` provider + namespace | sibling concrete backend, same pattern | 5 |

## References

- `specs/clean-slate-prover.md` — the backend seam this crate serves; the #1637 postmortem
  that calibrates the trait-granularity decisions here
- `specs/span-objectives.md`, `specs/jolt-eval-harness.md` — profiling spans over the pass;
  FixedBackend as the slot-fixture replay substrate
- The streaming-sumcheck sketch (this design's origin): `StreamableTrace` segments,
  `StreamConsumer`/`StreamDigest`, checkpoint-seeded passes — its consumer/digest layer maps
  onto jolt-kernels; its pass structure onto `stream_witnesses`; its `Arc<Mutex>` registration
  onto the `ConsumerSet` type parameter
- `tracer::LazyTraceIterator` — existing checkpointed re-emulation, the future segment source
- jolt-claims-derive / jolt-verifier-derive — the house pattern for typed structs + derives
  over statically-known protocol structure (`OutputClaims`-style field annotations)
