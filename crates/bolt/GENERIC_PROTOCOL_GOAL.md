# Bolt Generic Protocol Goal

Bolt should be a compiler framework for SNARK/PIOP-style protocols, not a
Jolt-shaped compiler with configurable names. Jolt is the first complete
protocol package and the correctness/performance oracle, but generic Bolt
layers must remain reusable for other protocols.

## Objective

Refactor the compiler boundaries so generic IRs, passes, validation, and Rust
artifact assembly operate over protocol concepts:

```text
roles
stages
transcript events
oracles and commitments
claims and relations
sumcheck obligations
opening obligations
proof slots
role-specific execution plans
```

Jolt-specific facts should enter only through the Jolt protocol package:

```text
protocol params
stage ordering
oracle names
relation definitions
proof-slot names
transcript labels
artifact crate names
prover kernel ABI mappings
Jolt-specific evaluation-proof composition
```

The result should make adding another protocol a matter of adding a new
`src/protocols/<name>/` package plus artifact config and any required prover
kernels, not editing Bolt's generic compiler core.

## Non-Negotiables

- Generic Bolt modules must not branch on Jolt stage names, Jolt relation
  symbols, or Jolt artifact names.
- Jolt symbols may appear as ordinary MLIR symbol data carried by Jolt-built
  modules, but generic passes may only preserve, validate structurally, or emit
  that data.
- Generic lowering remains:

```text
protocol -> concrete -> party -> compute -> cpu -> Rust
```

- Rust emission is the final target. Protocol behavior should be represented
  in dialect ops or MLIR-derived typed plans before Rust is emitted.
- Verifier emission must remain kernel-free and protocol-auditable.
- Prover kernels are protocol-package implementation details below the
  dialect boundary.
- Checked-in generated role crates remain generated artifacts, not
  hand-maintained source.

## Target Source Layout

Generic compiler modules:

```text
crates/bolt/src/dialects.rs
crates/bolt/src/ir.rs
crates/bolt/src/mlir.rs
crates/bolt/src/schema/
crates/bolt/src/pass/
crates/bolt/src/emit/rust/
```

Jolt protocol package:

```text
crates/bolt/src/protocols/jolt/
  params.rs
  validate.rs
  oracles.rs
  phases/
  relations/
  emit/
    rust/
    artifacts.rs
```

The exact file split can evolve, but ownership should not: generic modules own
compiler mechanics; `protocols/jolt` owns Jolt instantiation facts.

## Generic IR Criteria

Generic IR should expose enough structure for passes and emitters to reason
without protocol-specific string matching:

- `protocol` declares roles, stage boundaries, protocol params, and proof
  boundaries.
- `transcript` declares absorb/squeeze events and explicit state threading.
- `piop` declares oracles, claims, sumchecks, relation obligations, opening
  claims, opening equalities, and proof slots.
- `pcs` declares commitment, opening, verification, and evaluation-aggregation
  obligations over abstract PCS schemes.
- `party` represents role projection without deleting semantic obligations
  needed by later validation.
- `compute` represents executable obligations and optional prover kernel hooks.
- `cpu` represents backend-ready execution plans while staying
  protocol-agnostic.

If a generic emitter needs to branch on a string like
`jolt.stage6.booleanity`, the IR has not been lowered into a sufficiently
typed plan.

## Generic Pass Criteria

Generic passes may branch on:

```text
dialect op name
role
phase
declared proof-slot kind
declared relation-plan kind
declared PCS/transcript operation kind
backend target
```

Generic passes must not branch on:

```text
Jolt stage names
Jolt relation symbols
Jolt oracle names
Jolt artifact crate names
Jolt kernel ABI strings
```

Jolt-specific lowering is allowed inside `protocols/jolt`, but it should
produce generic dialect ops and typed plans consumed by the shared compiler.

## Generic Artifact Criteria

The generic Rust artifact assembler should be driven by `ProtocolArtifactConfig`
and ordered `ProtocolRustArtifact` values:

- Protocol name, type prefix, transcript label, role crate names, dependencies,
  forbidden imports, and type paths are config data.
- Stage modules are emitted from `ProtocolStage` data, not hardcoded enums in
  generic artifact code.
- Top-level `prover.rs` and `verifier.rs` are generated from role/stage/proof
  plans.
- Protocol-specific proof extensions are represented by explicit extension
  config or generic PCS/evaluation IR, not by checks like
  `type_prefix == "Jolt"`.
- `jolt-prover` may import verifier-owned proof types; `jolt-verifier` must
  never import prover or kernel code.

## Jolt Quarantine Criteria

These are the only acceptable homes for Jolt-specific compiler knowledge:

```text
crates/bolt/src/protocols/jolt/**
crates/bolt/tests/** when the test explicitly targets Jolt
crates/jolt-prover/**
crates/jolt-verifier/**
crates/jolt-kernels/**
crates/jolt-equivalence/**
```

Generic Bolt modules should have a hygiene gate rejecting `jolt`, `Jolt`,
`stage6`, `stage7`, `stage8`, and Jolt relation/policy names, with a small
temporary allowlist during migration.

## Correctness Criteria

Every genericity cleanup slice must preserve the existing semantic oracles:

```text
generated role crates still compile
checked-in generated role crates match canonical generation
Bolt prover artifacts are accepted by the generated Bolt verifier
Bolt prover artifacts are accepted by the core oracle for implemented stages
Bolt/core transcript histories match for implemented stages
internal prover/verifier transcript histories match for implemented stages
tampering gates still reject malformed artifacts
generated verifier import boundaries remain intact
verifier CPU IR remains kernel-free
```

For pure file moves and namespace refactors, generated output should either be
byte-for-byte unchanged or intentionally regenerated with a clear explanation
of why the generated surface changed.

## Migration Algorithm

For each cleanup slice:

1. Identify the Jolt-specific fact currently living in a generic module.
2. Decide whether it is protocol data, relation semantics, artifact config, or
   a prover-kernel implementation detail.
3. Move that fact to `protocols/jolt` or encode it as generic IR/typed plan
   data.
4. Keep generic APIs protocol-named (`Protocol*`) and provide Jolt convenience
   wrappers only under `protocols::jolt`.
5. Add or tighten a hygiene gate so the leak does not reappear.
6. Regenerate artifacts only through the canonical generator.
7. Run the relevant schema, generation, import, equivalence, and tamper gates.

Do not hide protocol semantics in opaque Rust helpers to pass the hygiene gate.
If the generic emitter needs new information, add a typed plan field or dialect
operation and validate it.

## Active Goal-Mode Slice: Modular Compiler Core

The next goal-mode track is to make Bolt's generic compiler implementation
smaller, more navigable, and closer to the protocol/compiler notation we use in
the paper, while preserving the full Jolt/Core equivalence and performance
semantics that currently serve as the correctness oracle.

This should be an iterative cleanup, not a global rewrite. The first objective
is to expose stable module boundaries and duplicated mechanics before deciding
which abstractions deserve to exist.

Current pressure points:

```text
crates/bolt/src/pass.rs
crates/bolt/src/pass/*.rs
crates/bolt/src/emit/rust/artifacts.rs
crates/bolt/src/protocols/jolt/emit/rust/*.rs
crates/bolt/src/schema.rs
```

Desired direction:

```text
pass/
  mod.rs
  protocol_to_concrete.rs
  roles.rs
  party_to_compute.rs
  kernel_resolution.rs
  compute_to_cpu.rs
  verify.rs

emit/rust/
  mod.rs
  artifact.rs
  crate_graph.rs
  source.rs
  role_entrypoints.rs
  runtime_modules.rs

protocols/<name>/
  params.rs
  phases/
  relations/
  kernels.rs
  emit/
```

These names are starting hypotheses, not architecture commitments. Prefer file
names and APIs that match the paper vocabulary when the code already has the
same conceptual boundary.

### First Slice

Start with the pass module because it is now generic and has clearer seams:

1. Split `pass/lowering.rs` by actual compiler pass:
   `party_to_compute`, `kernel_resolution`, and `compute_to_cpu`.
2. Keep public root exports source-compatible:
   `lower_party_to_compute`, `resolve_compute_kernels_with`,
   `lower_compute_to_cpu`, `PartyToComputeLowering`, and
   `ComputeKernelSpec`.
3. Move shared pass helpers into a private support module only when two or more
   passes use them. Do not create a helper module just to move lines around.
4. Introduce traits only where they remove concrete duplication:
   `KernelRegistry` is a plausible candidate because kernel resolution already
   depends on a protocol-supplied relation-to-kernel mapping. A generic `Pass`
   trait is not required until two passes can use it without obscuring MLIR
   lifetimes or error reporting.
5. Add or tighten structure tests only after each boundary is real. The test
   should prevent regressions such as Jolt strings re-entering generic pass
   modules, not bless a speculative directory layout.

### Emit Cleanup Track

After the pass split is stable, inspect Rust emission rather than immediately
reshaping it. The likely seams are:

```text
artifact config and crate graph assembly
role entrypoint generation
runtime module/source rendering
typed stage/plan emission
protocol extension hooks
```

The emit track should first document duplicated patterns and protocol-specific
facts, then lift only the parts that can be represented by generic
`ProtocolArtifactConfig`, `ProtocolStage`, MLIR-derived plans, or explicit
extension traits. Jolt stage emitters may remain quarantined while we identify
which pieces are truly protocol-specific.

### Trait Guardrails

Use traits to make protocol inputs explicit, not to hide protocol semantics:

- Good candidates:
  `KernelRegistry`, protocol artifact extension hooks, relation plan emitters,
  typed runtime module providers.
- Risky candidates:
  a universal `Pass` abstraction over all MLIR transformations, a generic
  emitter trait that just moves string matching behind dynamic dispatch, or
  protocol hooks that accept untyped bags of strings.
- Prefer associated types and small traits over object-safe trait objects until
  there is a concrete need for dynamic dispatch.
- Keep error messages phase-aware and op-aware; abstraction should not erase the
  schema context that makes MLIR failures debuggable.

### Non-Negotiable Guardrails For This Track

- Preserve the Bolt pipeline:

  ```text
  protocol -> concrete -> party -> compute -> cpu -> Rust
  ```

- Preserve full-field non-zk Jolt semantics and transcript behavior.
- Preserve existing generated role crate semantics. File moves should not change
  generated output unless a slice explicitly says so.
- Preserve prover/verifier performance semantics. Any runtime or emitter
  consolidation that changes allocation patterns, relation evaluation shape, or
  kernel dispatch shape needs focused perf review.
- Keep Jolt facts under `protocols/jolt` or generated Jolt crates.
- Keep generic compiler hygiene gates green.
- Each slice must end with focused tests, plus equivalence/perf gates when the
  slice touches generated Rust behavior, relation evaluation, transcript order,
  kernel dispatch, or artifact shape.
- Do not combine modularization with semantic redesign. If a cleanup exposes a
  missing IR concept, write down the concept and land the mechanical split
  first unless the split is impossible without the new IR.

### Success Criteria For The First Few Iterations

- `pass.rs` becomes a thin module façade, not the home of pass logic.
- Each generic pass has one obvious file and one small public surface.
- Shared lowering mechanics are factored without reintroducing protocol names.
- `emit/rust` has a documented split plan based on current duplication, not a
  speculative final backend design.
- The next developer can add a small non-Jolt protocol fixture without editing
  Jolt files or discovering hidden Jolt assumptions in generic pass code.

### Iteration Log

- Split the generic pass façade into `party_to_compute`,
  `kernel_resolution`, `compute_to_cpu`, `roles`, `verify`, and support files.
  Jolt now supplies stage naming, parameterization, ordering, and kernel
  mappings through protocol-local wrappers.
- Split party-to-compute transcript, value/point, and proof/opening notation
  into private operation-family modules, keeping role-sensitive prover/verifier
  target selection as explicit typed helpers.
- Added `KernelRegistry` as the narrow trait boundary for protocol-supplied
  relation-to-kernel resolution. Avoided a universal pass trait until there are
  multiple concrete users.
- Split kernel-resolution sumcheck/opening proof notation into a private module,
  including the relation-to-kernel attr construction for claim and driver ops.
  The registry lookup and value mapping remain explicit in the pass loop.
- Split kernel-resolution opening-input, point, and field value notation into a
  private module, matching the operation-family boundary used by CPU lowering.
- Split kernel-resolution transcript notation into a private module so
  transcript attr/result conventions are named separately from kernel lookup.
- Split generic Rust artifact assembly into crate graph, role API, support, and
  type modules while preserving the `ProtocolArtifactConfig` /
  `ProtocolStage` public surface.
- Split generic role API source discovery from role API source rendering, so
  generated crate API assembly is not mixed with Rust item introspection.
- Split low-level role API source scanning helpers out of role API discovery,
  leaving discovery focused on stage/commitment metadata construction.
- Split role API discovery into stage metadata, commitment metadata, extension
  selection, and module-naming helper modules while keeping the discovery
  façade imports unchanged for prover/verifier role renderers.
- Split generic role API rendering further into prover and verifier renderer
  modules while leaving the generated role crate API and public artifact
  surface unchanged.
- Split prover proof-helper rendering and verifier proof/input struct rendering
  into focused role API submodules, keeping execution ordering in the main
  prover/verifier renderers.
- Factored shared role-program struct/default-program rendering out of the
  generic prover/verifier role API emitters. This keeps generated role API text
  unchanged while removing duplicated program field selection logic.
- Factored shared role artifact-field and error-variant rendering out of the
  generic prover/verifier role API emitters, preserving the prover kernel-module
  distinction through an explicit role-aware helper.
- Factored shared role program/artifact/error declaration wrapper rendering out
  of the prover/verifier role API emitters. Parent renderers still own role API
  ordering and extension hook placement.
- Factored generic role error-conversion rendering into the shared role item
  renderer while preserving the verifier macro form and prover kernel-module
  error source selection.
- Factored repeated generic prover role API `where` bound rendering into a
  local helper so the default and with-program entrypoints share the same
  executor/transcript bound construction.
- Split generic generated-crate manifest rendering out of crate graph assembly,
  leaving crate/file assembly separate from standalone/workspace `Cargo.toml`
  formatting.
- Split generic crate graph emission into artifact construction/validation,
  in-memory generated-crate assembly, and filesystem writing modules. The
  public artifact assembly exports remain unchanged.
- Split generic Rust artifact type definitions into config, extension, type
  reference, stage, and generated-crate family modules while preserving the
  public artifact type re-exports.
- Moved compute-to-cpu local MLIR attr parsing/source-formatting helpers into
  shared pass support to prepare that pass for operation-group splits without
  changing lowering behavior.
- Split the kernel-resolution protocol hook types into a dedicated
  `kernel_resolution::registry` module while keeping `ComputeKernelSpec` and
  `KernelRegistry` re-exported through the existing pass/root API.
- Split concrete transcript-state verification into a private `verify`
  submodule while keeping `VerifyError` and `verify_concrete_transcript`
  re-exported through the existing pass/root API.
- Factored concrete transcript-state verification into a local state-thread
  tracker so the verifier names initialization, consumption, operand presence,
  and advancement checks without changing diagnostics or transcript semantics.
- Split protocol-boundary declared-role checking out of the role projection
  pass. Role projection now keeps projection control flow separate from
  boundary role parsing diagnostics.
- Split compute-to-cpu PCS commitment/receive attribute construction into a
  private operation-group module, keeping the main CPU lowering pass focused on
  pass control flow and value mapping.
- Moved compute-to-cpu PCS commitment target-op naming and result-type notation
  into the same private operation-group module, leaving operand resolution in
  the main lowering loop.
- Split compute-to-cpu PCS commitment attr conversion and target-op naming
  further into a private `commitment::pcs` module. The lowering loop retains
  operand resolution and diagnostics.
- Split compute-to-cpu opening-input, point, and field value notation into a
  private operation-group module. This names the target CPU op/result-type
  conventions and copied attr sets without moving the value-map control flow.
- Split compute-to-cpu sumcheck/opening proof notation into a private
  operation-group module. This keeps the paper-level proof objects explicit
  while avoiding a larger rewrite of the pass loop.
- Split compute-to-cpu transcript and oracle notation into private
  operation-group modules, and moved transcript absorb attr construction out of
  the PCS commitment helper module. Operand resolution and transcript ordering
  remain in the main lowering loop.
- Added a local compute-to-cpu helper for the repeated single-result
  append-and-map pattern after checking the MLIR lifetime shape. Multi-result
  proof/transcript operations still map results explicitly at their call sites.
- Moved the single-result append-and-map helper into shared pass support and
  adopted it across party-to-compute, kernel-resolution, and compute-to-cpu.
  This factors the MLIR lifetime/value-map pattern without introducing a
  universal pass trait.
- Added a small shared value-family lowering trait across party-to-compute,
  kernel-resolution, and compute-to-CPU. The passes now declare source
  operation classification, target phase, result types, and operation naming
  policy while shared support owns the identical opening-input, point, and
  field lowering mechanics. The compute-source classifier is shared by
  kernel-resolution and compute-to-CPU, keeping the post-compute operation
  family table in one place.
- Added a shared append-and-map-results helper for explicit multi-result MLIR
  operations, keeping the expected result count visible at each lowering site.
- Moved sumcheck, opening, and PCS proof operation lowering out of the three
  main pass loops into their pass-local proof modules. The pass files now carry
  control flow and value mapping, while proof modules own proof-object notation.
- Split each pass-local proof module into `sumcheck`, `opening`, and `pcs`
  family files. Kernel resolution keeps relation-to-kernel lookup in the
  sumcheck family, matching where kernels are introduced.
- Split kernel-resolution sumcheck kernel-introduction helpers into a private
  `sumcheck::kernel` module. The parent sumcheck module still owns operation
  dispatch and keeps registry-backed kernelization local to kernel resolution.
- Moved non-proof operation-family lowering for party-to-compute,
  kernel-resolution, and compute-to-cpu into their existing transcript, value,
  oracle, and commitment modules. The main pass files are now short pass
  control-flow spines plus top-level declaration copying.
- Split shared pass support into attr parsing, diagnostics, source formatting,
  transcript type notation, and value-map/result plumbing modules. Existing
  pass call sites still import through the `support` façade.
- Added shared pass helpers for the repeated lowered-op append pattern and
  adopted them across sumcheck proof-family lowering. Kernel-resolution cases
  that introduce protocol-supplied kernels remain explicit.
- Adopted the same shared lowered-op append helpers across opening and PCS
  proof-family lowering, including no-result equality ops. The proof-family
  files now primarily spell target op names and paper-level attr/result shape.
- Added a shared opening-proof lowering trait across party-to-compute,
  kernel-resolution, and compute-to-CPU. The pass-local opening modules now
  declare source classification, target naming, and result types while shared
  support owns claim, equality, and batch lowering mechanics.
- Added the same trait-shaped support for PCS opening claims, opening batches,
  and batch open/verify operations. Party-to-compute still owns role-sensitive
  open-vs-verify target selection, while compute-source passes share their PCS
  operation classifier.
- Factored the common sumcheck eval and instance-result lowering tail into
  shared support. Sumcheck claim, batch, driver, verifier, and kernel
  introduction paths remain explicit in the pass-local proof modules.
- Split compute-to-CPU sumcheck kernel claim/driver lowering into a private
  `sumcheck::kernel` module, mirroring the kernel-resolution proof split while
  keeping verifier claim/batch/driver lowering in the parent CPU proof module.
- Factored fixed transcript init/absorb-bytes/squeeze lowering into shared
  transcript support for party-to-compute and kernel-resolution, and reused the
  same support for compute-to-CPU absorb-bytes/squeeze. CPU transcript init and
  artifact absorb remain explicit because they rewrite attrs and resolve
  artifact operands differently.
- Added a shared required-lowered-operand helper for CPU-stage custom lowering
  cases that need exact diagnostics. Commitment, oracle-family append, and
  transcript artifact absorb now share operand-key/value-map resolution while
  keeping their existing error text.
- Moved compute-to-CPU PCS commitment batch/optional lowering bodies into the
  private commitment PCS module. The parent commitment pass now only dispatches
  PCS operation families, while PCS target naming, attr conversion, operand
  resolution, and append/map mechanics stay grouped together.
- Split compute-to-CPU transcript attr conversion into a private transcript
  helper module so CPU lowering dispatch stays focused on operation order and
  value mapping while symbol/string/bool source formatting remains local to
  CPU transcript lowering.
- Added shared compute-to-CPU target-op naming support and adopted it across
  CPU value, proof, oracle, and transcript lowering policies. The individual
  modules still declare which operation families they accept.
- Adopted shared lowered-op append helpers across fixed-shape value,
  transcript, and oracle lowering cases. Custom field attrs and operand-specific
  CPU diagnostics remain explicit at their call sites.
- Split schema verification into a small façade plus operation and constraint
  modules, keeping schema context and error messages phase-aware.
- Split schema operation verification by dialect family (`foundation`, `piop`,
  lowered compute/CPU, and top-level PCS) and introduced a small lowered-dialect
  trait to share compute/CPU op shape checks without hiding protocol semantics.
- Split lowered compute/CPU schema verification further into dialect policy,
  lowered attr tables, and sumcheck validation helpers. The main lowered
  verifier remains the operation dispatch table.
- Split schema constraint helpers into attr parsing, naming, opening-claim
  equality, shape/count, and symbol lookup modules while preserving the
  `schema` re-export surface used by Jolt validation and generic schema ops.
- Added a protocol-local Jolt Rust emitter MLIR helper module for repeated
  attribute, operand, and operation-name parsing. This reduces duplicated stage
  emitter mechanics without moving Jolt stage semantics into generic emit code.
- Added a protocol-local Jolt Rust source helper module for repeated literal
  and const-array rendering. This keeps emitted Rust text mechanics shared
  without changing generated crate structure.
- Added a protocol-local Jolt Rust check helper module for repeated support
  target validation, count checks, missing binding diagnostics, and symbol-set
  collection while preserving emitter-specific error text where it existed.
- Split generated prover/verifier with-program execution-body rendering into
  role-specific execution modules. Entrypoint order, extension hook placement,
  and generated role API shape remain owned by the parent role renderers.
- Split generated prover/verifier entrypoint signature rendering into
  role-specific entrypoint modules. The parent role renderers now assemble
  imports, type/error declarations, entrypoints, and helper items in order.
- Split generated prover input struct rendering into a role-specific prover
  input module, matching the existing verifier input helper. The parent prover
  renderer still owns emitted section ordering and extension hook placement.
- Split generated prover/verifier import rendering into role-specific import
  modules. Default import formatting, extension import hooks, kernel imports,
  and stage-module aliases are now outside the parent renderers while emitted
  section ordering remains unchanged.
- Split generated prover/verifier type-alias rendering into role-specific alias
  modules. Parent renderers still call the alias block immediately after imports
  so emitted API ordering is unchanged.
- Split generated prover/verifier error enum and error-conversion rendering
  into role-specific error modules. Parent renderers still call the error block
  after artifacts and before entrypoints, preserving extension hook placement.
- Moved generated prover/verifier role API type-name construction into a shared
  role API names module and reused it for generated role-crate `lib.rs`
  exports. The role renderers still own section ordering and role-specific
  emitted items, but no longer duplicate the protocol-prefixed name inventory.
- Split compute-to-CPU oracle buffer and oracle-family lowering into private
  operation-family modules. The parent oracle pass now dispatches family cases
  while buffer/family modules own target naming, attrs, operand resolution, and
  append/map mechanics.
- Split shared pass value-map key derivation into a private support module.
  Operand/result key diagnostics are now separated from the append-and-map
  plumbing that lowering passes call, without changing the support façade.
- Split generated role-crate `src/lib.rs` rendering into a dedicated role API
  module. Crate graph assembly still calls the same `generated_lib` export, but
  role API façade types are no longer mixed with lib-module source formatting.
- Split the generated role API plan structs into a dedicated role API type
  module. Discovery, declarations, programs, and role renderers still use the
  same parent-local names, but the façade no longer owns the shared data shape.
- Split generated `src/stages/mod.rs` source rendering out of crate graph
  assembly. Crate assembly now owns file ordering, while the module-source
  helper owns runtime-module and stage-module declaration formatting.
- Split generic value-lowering operation-family classification and copied-attr
  notation into private support submodules. The shared value-lowering trait and
  lowering function remain the pass-facing API, while family tables and attr
  lists are now named independently.
- Split compute-to-CPU PCS commitment target-op naming and CPU attr/source
  formatting into a private notation module. The PCS lowering module still owns
  operand resolution, diagnostics, and append/map mechanics for commitment
  artifacts.
- Split shared transcript lowering support into operation-family
  classification, copied-attr notation, and squeeze result-type modules. The
  transcript dialect trait and lowering function remain the pass-facing API.
- Split shared sumcheck value lowering support into operation-family
  classification and copied-attr notation modules. The shared sumcheck value
  dialect trait and lowering function remain unchanged at call sites.
- Split shared PCS proof lowering support into operation-family classification
  and copied-attr notation modules. Role-sensitive target selection remains in
  the dialect implementation, while shared lowering keeps the same call shape.
- Split shared opening-proof lowering support into operation-family
  classification and copied-attr notation modules, matching the value,
  transcript, sumcheck, and PCS support layout.
- Factored shared sumcheck proof notation into pass support. Party projection,
  kernel resolution, and compute-to-CPU now reuse the same claim/batch/driver
  attr lists and phase-specific result-type constants, while each pass still
  owns role selection, kernel introduction, and operation dispatch.
- Factored opening proof result-type notation into pass support. Opening and
  PCS proof modules now depend on shared phase-specific opening claim, batch,
  and batch-opening result-type constants instead of importing notation through
  sibling proof modules.
- Consolidated generated role API role selection into a shared role helper.
  Program and item renderers now use the same prover/verifier role value and
  module-alias selection rule instead of carrying duplicate role enums.
- Split generated role API error item rendering out of the artifact-field item
  renderer. Error variants and prover/verifier conversion implementations now
  live in an `items::errors` module behind the same parent exports.
- Split PIOP schema operation attr notation into a private schema module. The
  PIOP schema verifier now keeps operation dispatch separate from relation,
  sumcheck, opening, and input attr-shape tables.
- Split foundation schema operation attr notation into a private schema module.
  Field, transcript, PCS, polynomial, and protocol-boundary attr tables are now
  separated from foundation operation dispatch.
- Split PCS schema operation attr notation into a private schema module and
  moved the remaining lowered compute/CPU schema inline attr lists into the
  lowered attr table. Schema operation dispatch now consistently references
  named notation instead of embedding protocol-object attr arrays in match arms.
- Split generated prover/verifier entrypoint bound formatting into role-local
  helper modules. Entrypoint modules still own function signature ordering and
  body assembly, while bounds helpers own executor/transcript `where` clauses.
- Moved field-expression attr selection into value-lowering notation. Primitive
  pass attr support now owns only generic attr extraction, while value lowering
  owns the field/polynomial expression attr-copy policy it consumes.
- Split generated prover/verifier execution call rendering into role-local
  `execution::calls` modules. Execution modules now own statement ordering,
  proof/artifact assembly, and extension placement, while call helpers own
  commitment/stage invocation formatting.
- Split compute-to-CPU custom transcript init/absorb lowering into a private
  transcript module. The parent transcript lowering file now owns shared
  transcript dialect dispatch while custom CPU cases own operand diagnostics,
  attr conversion, and append/map mechanics.
- Consolidated fixed-shape value lowering inside shared pass support. Point,
  field constant/unit, and opening-input lowering now share the same local
  target-name and append/map helper path, while expression lowering remains
  explicit because it resolves operands and dynamic attrs.
- Consolidated fixed-shape transcript, sumcheck value, opening, and PCS proof
  lowering inside shared pass support. The helpers now centralize target-name
  resolution and append/map plumbing while keeping family-specific result
  counts, attrs, role selection, and dynamic result typing explicit.
- Split shared pass value-map result append/mapping into a private support
  module. Operand lookup remains in the value support parent, key derivation
  stays in `values::keys`, and result insertion/append-and-map plumbing now
  lives in `values::results` behind the same support façade.
- Factored schema operand-owner result resolution into the symbol constraint
  helpers and reused it for opening-claim equality validation. Symbol lookup
  and opening equality now share the same operand/result diagnostics.
- Split low-level schema attr parsing into a private attrs parser module.
  Schema attr constraints now expose typed readers and required-attr checks,
  while integer and symbol-array parsing details live behind that surface.
- Consolidated generated role API public-function source scanning. Discovery
  still calls the same source-scan helpers, but ordinary and substring-matched
  public function discovery now share one parser for Rust function names.
- Factored schema shape operand/result count checks into local constraint
  helpers. Exact-shape and minimum-shape validation now share result-count
  diagnostics while preserving the existing schema error text.
- Factored pass typed-attr error construction into a local helper. String,
  symbol, and bool attr readers keep the same public surface and diagnostic
  text while sharing operation/attr error formatting.
- Moved sumcheck value result-type notation into shared support. Party
  projection, kernel resolution, and compute-to-CPU now reference common
  compute/CPU eval and instance-result type constants when implementing their
  sumcheck value dialects.
- Moved fixed value-lowering result-type notation into shared support. Party
  projection, kernel resolution, and compute-to-CPU now reuse common
  compute/CPU point, field, and opening-input result-type constants while
  preserving pass-local dispatch and target naming.
- Moved transcript state result-type notation into shared transcript support.
  Party projection, kernel resolution, and compute-to-CPU now reuse common
  compute/CPU transcript-state result constants while retaining pass-local
  target naming and CPU custom transcript lowering.
- Factored the repeated Jolt generated Rust module shell into a protocol-local
  source helper. Commitment and stages 1 through 7 now share the same
  dead-code/import/type/constants/entrypoint assembly path while stage files
  still own their role-specific imports, plans, constants, and entrypoints.
- Split shared value-lowering implementation bodies into fixed-result and
  field-expression helper modules. The parent value-lowering support file now
  reads as the operation-family dispatch table while the helper modules own
  append/map mechanics for fixed-shape values and dynamic field expressions.
- Split remaining shared lowering result append bodies into private result
  helper modules for sumcheck values, PCS, opening, and transcript lowering.
  Parent support files now keep trait definitions and operation-family dispatch
  separate from target-name/result append mechanics.
- Added a protocol-local plan-array source renderer and adopted it for simple
  generated stage constants in stages 4 through 7. Stage emitters still own
  their Jolt-specific plan rows and role logic, while the repeated Rust
  `pub const NAME: &[PlanType]` wrapper formatting now lives in one helper.
- Reused the protocol-local plan-array renderer for the matching simple
  constants in stages 2 and 3. The stage emitters still construct their own
  program-step, transcript, opening-input, field-constant, and kernel plan rows
  while sharing the constant-array wrapper formatting.
- Reused the protocol-local plan-array renderer for exact-match simple
  constants in stage 1. Stage 1 still owns its role-sensitive claim, batch, and
  driver rendering while shared source support owns the repeated constant-array
  wrapper for transcript, kernel, eval, instance-result, and opening-claim
  plans.
- Added a compact variant of the protocol-local plan-array renderer and used
  it for commitment plan arrays. Commitment emission keeps its existing tight
  generated source layout while oracle, batch, optional-commitment, and
  transcript plan rows share the same array wrapper mechanics.
- Reused the protocol-local plan-array renderer for tail value/opening
  constants in stages 2 through 5. Sumcheck instance/eval, point-slice, opening
  claim, and opening equality row construction stays stage-local, while the
  repeated constant-array wrapper is shared.
- Reused the protocol-local plan-array renderer for the plain tail value and
  opening constants in stages 6 and 7. The chunked sumcheck-eval macro and
  role-sensitive concat/batch emission remain stage-local because their emitted
  source shape is not a simple plan array.
- Reused the compact protocol-local plan-array renderer for role-sensitive
  point-concat and opening-batch constants in stages 2 through 7. Prover and
  verifier row construction still stays stage-local; only the final one-newline
  `pub const` wrapper is shared.
- Reused the same compact plan-array wrapper for sumcheck claim, batch, and
  driver constants in stages 2 through 7. Missing role-binding checks,
  round-schedule helper arrays, and prover/verifier row construction remain in
  the stage emitters.
- Reused the compact plan-array wrapper for plain field-expression constants in
  stages 2 through 5 and for the prover field-expression constants in stages 6
  and 7. The stage 6/7 verifier macro paths remain local because their
  rustfmt-skipped chunking is a distinct emitted-source shape.
- Reused the compact plan-array wrapper for stage 1 role-sensitive sumcheck
  claim, batch, driver, and opening-batch constants. Prover/verifier
  role-binding checks and helper-array construction remain in the stage 1
  emitter.
- Reused the protocol-local plan-array renderer for the plain stage 8 opening
  input and opening-claim arrays. The singleton evaluation-point, batch, PCS
  proof, and program constants remain stage 8-local because they encode the
  evaluation-proof composition shape.
- Added a protocol-local rustfmt-skipped macro-plan array renderer and reused
  it for stage 6/7 field-expression verifier arrays and sumcheck-eval arrays.
  Macro invocation rows and chunking stay stage-local.
- Added a protocol-local params-constant renderer and reused it for stages 1
  through 7. Each stage still owns its ordering and role-specific program
  assembly while the repeated `STAGE*_PARAMS` source wrapper is shared.
- Added a protocol-local struct-constant renderer and reused it for final
  `STAGE*_PROGRAM` constants in stages 1 through 7. Stage emitters still own
  the exact field lists, role-dependent type aliases, and stage ordering, while
  the repeated Rust struct literal wrapper is shared.
- Reused the protocol-local params and struct-constant renderers in commitment
  Rust emission. Commitment-specific oracle, batch, optional commitment, and
  transcript row construction stays local to the commitment emitter.
- Added protocol-local value and inline-struct constant renderers and reused
  them in stage 8 for singleton generated constants. Stage 8 still owns the
  evaluation-point selection, proof composition shape, and opening row
  construction.
- Added a compact protocol-local string-array renderer for commitment batch
  oracle helper arrays and removed the now-unused generic `push_format` helper.
  Commitment batch membership and plan rows stay local to commitment emission.
- Removed unused legacy `emit_types()` bodies from stages 4 through 7 after
  those stages had moved to the shared generated runtime aliases. The emitted
  prover/verifier source paths now rely only on the active alias/type emitters.
- Added a stage-common verifier type-alias renderer for stages 4 through 7,
  parameterized by stage number and point-zero support. Stage-specific verifier
  errors and relation helpers remain in their stage emitters.
- Reused the stage-common source helpers for stage 4 through 7 prover imports
  and default transcript aliases. Stage-specific verifier imports and runtime
  verifier logic remain local to each stage.
- Generalized the stage-common prover import renderer and reused it for stages
  2 through 7, with explicit stage-local shape flags for opening equalities,
  transcript byte absorbs, and point-zero plans.
- Replaced the stage-common prover import renderer's positional shape flags
  with named import-shape constants. Stage files now declare their shared
  source prelude shape explicitly while keeping the import text assembled in
  one protocol-local helper.
- Added a stage-common verifier type-alias renderer for stages 2 and 3, keeping
  their verifier-specific sumcheck claim/driver aliases distinct from the
  stage 4-7 runtime-plan aliases.
- Added a stage-common verifier error-enum renderer for stages 2 through 7,
  with a named RAM-shaped variant for stage 2. Verifier-specific data structs,
  relation symbols, and relation checks remain stage-local.
- Added a generic pass helper for copying named operations with selected
  attributes across phase lowerings. Party-to-compute relation copying,
  kernel-resolution structural op copying, and compute-to-cpu kernel copying
  now share that notation-level mechanic.
- Replaced the remaining stage-common verifier type-alias booleans with named
  shape descriptors for stages 2/3 and stages 4-7. Stage emitters now declare
  alias shape in terms of protocol stage families instead of positional
  `true`/`false` flags.
- Replaced stage 2/3 sumcheck claim and driver emitter boolean role flags with
  the existing `Role` enum. The generated prover/verifier constant paths now
  name their role at the helper boundary while preserving role-specific
  binding diagnostics.
- Replaced generic role API commitment-presence booleans with typed optional
  commitment API values. Prover/verifier import rendering, prover input
  rendering, and prover generic-parameter construction now receive the
  commitment artifact shape rather than an anonymous flag.
- Moved generated role API program-availability checks onto the discovered
  stage and commitment metadata types. Program struct fields, default-program
  fields, and prover/verifier execution calls now share one invariant for when
  a with-program entrypoint can receive `programs.<field>`.
- Moved generated role API entrypoint selection onto the discovered stage and
  commitment metadata types. Prover/verifier execution renderers now ask the
  metadata for the preferred with-program-or-default entrypoint while retaining
  their role-specific call formatting and fallback diagnostics.
- Added typed verifier-stage input descriptors to discovered stage metadata.
  Verifier input struct rendering and verifier execution argument rendering now
  share the same opening/RAM/data input inventory while keeping Rust source
  formatting in the verifier renderer modules.
- Added a typed generated role API program descriptor so program struct fields
  and default-program values consume already-checked program type/const pairs
  instead of pairing a boolean availability test with `expect("... checked")`.
- Factored generated role API metadata program and entrypoint selection into
  private metadata helpers shared by stage and commitment metadata. The
  role-specific renderers still consume typed descriptors and owned fallback
  diagnostics at their call sites.
- Replaced lowered schema dialect boolean feature flags with a named
  `LoweredDialectCapabilities` policy object. Compute/CPU schema validation now
  checks relation-op and kernel-sumcheck support through the dialect capability
  value instead of independent associated booleans.
- Replaced positional lowered-dialect capability construction with named
  compute/CPU capability constants, so schema dialect policy selection no
  longer depends on `true`/`false` ordering.
- Replaced counted ordered-claim schema shape call sites with named
  `CountedOperandShape` constants for zero-fixed and one-fixed operand forms.
  PIOP, PCS, and lowered schema validators now share the same counted-shape
  notation instead of repeating positional count arguments.
- Collapsed stage 8 generated type prelude assembly into a single stage-local
  `emit_types()` path. Evaluation-proof constants, point selection, and opening
  composition remain in the stage 8 emitter.
- Tightened generic compiler hygiene so Jolt protocol names and backend facts
  cannot re-enter `crates/bolt/src` outside `protocols/jolt`.
- Replaced the PCS lowering helper's optional role plumbing with a typed
  `PcsLoweringRole` boundary. Party-to-compute PCS batch open/verify selection
  now declares its role requirement explicitly and reports a schema error if
  invoked without one, while kernel-resolution and compute-to-CPU PCS lowerings
  declare that role input is unavailable.
- Added named `LoweredResultCount` descriptors for generic pass result mapping.
  Value, opening, transcript, sumcheck, and PCS lowering helpers now spell
  zero/one/two/three-result shapes explicitly instead of passing raw arities
  through repeated `if result_count == 1` branches.
- Replaced module-shell copying's anonymous optional role with a typed
  `PhaseCopyRole` policy. Protocol-to-concrete lowering now declares role-free
  copying, while role derivation and party projection declare the exact role
  they stamp into the copied module shell.
- Split generated role API metadata types into focused stage, commitment,
  program, and verifier-input descriptor modules. The role API façade still
  re-exports the same discovered metadata shapes, but each descriptor now owns
  its role-specific helpers instead of collecting all role API type behavior in
  one file.
- Replaced raw exact/minimum schema operation shape arguments with named
  `ExactOpShape` and `MinOpShape` descriptors. Foundation, PIOP, PCS, and
  lowered compute/CPU schema validators now describe common operand/result
  shapes by name while preserving the existing shape diagnostics.
- Replaced lowered schema dialect capability booleans with an enum-backed
  `LoweredDialectCapabilities` policy. Compute and CPU validators now select
  their capability set by variant rather than storing parallel relation/kernel
  support flags.
- Moved generated role-crate runtime-module selection onto
  `ProtocolArtifactConfig`. Crate file assembly and stage-module source
  generation now ask the artifact config for the modules active for a role
  instead of repeating verifier-only checks in crate-graph emitters.
- Split generated role-crate `src/lib.rs` role-module rendering into
  prover/verifier helpers and a separate generated-stage inventory helper.
  The parent `generated_lib` path now assembles common crate metadata while
  role-specific exports and extension overrides stay in role-local source
  helpers.
- Moved generated role API source-file path selection onto `RoleApiRole`.
  The role API façade now converts from the compiler `Role` once, then uses the
  role helper for `src/prover.rs` versus `src/verifier.rs` while keeping the
  prover/verifier API source renderers separate.
- Moved role-specific generated program-plan type suffixes onto `RoleApiRole`.
  Stage and commitment API discovery now convert artifact roles once and ask
  the role helper for prover/verifier program-plan naming instead of repeating
  role matches in each discovery module.
- Added local fixed-role constants to prover and verifier role API renderers,
  error renderers, and execution-call helpers. Role-specific files now pass
  their declared role through shared declaration/program/error helpers without
  repeating `RoleApiRole::Prover` or `RoleApiRole::Verifier` at each call site.
- Added a four-result `LoweredResultCount` case and used named result-count
  lowering for party-to-compute sumcheck claim, batch, and driver operations.
  Role-sensitive sumcheck claim/driver target names now come from one
  `RoleSumcheckTargets` descriptor instead of separate role match helpers.
- Reused named `LoweredResultCount` lowering for fixed-shape kernel-resolution
  and compute-to-CPU sumcheck verifier/kernel paths. The lower-level
  multi-result append helper is now internal to pass support instead of being
  exported through the support façade.
- Reused named `LoweredResultCount` lowering for compute-to-CPU oracle buffer
  and oracle-family init paths. The lower-level first-result append helper is
  now internal to pass support instead of being exported through the support
  façade.
- Moved `LoweredResultCount` into a dedicated pass support module and reused it
  for custom append-and-map result plumbing. Kernel-resolution sumcheck driver
  introduction now uses the same named four-result notation as lowered-op
  append paths.
- Factored generated crate artifact partitioning into an `ArtifactsByRole`
  helper. Crate graph assembly now separates artifact validation from
  prover/verifier bucket construction before generating role crates.
- Split schema operation shape notation into `schema/ops/support/shape.rs`.
  The schema support façade now owns validation helper functions while the
  named operand/result shapes stay in a dedicated notation module.
- Moved role API extension activation policy onto
  `ProtocolArtifactExtension::is_active_with`. Discovery now supplies
  availability predicates for commitment, proof stages, and artifact stages
  instead of inspecting extension requirement fields directly.
- Collapsed pass lowering result plumbing onto `append_lowered_result_count`.
  The raw multi-result append-and-map helper is now private to
  `support/values/results.rs`, keeping generic pass call sites on named
  `LoweredResultCount` notation instead of untyped result counts.
- Factored shared role API program metadata behavior into
  `RoleApiProgramSource`. Commitment and stage API metadata now share one
  trait-backed implementation for program presence, program descriptors, and
  prover/verifier entrypoint selection.
- Moved generated role API program-argument construction onto
  `RoleApiProgramSource`. Prover and verifier execution call emitters now ask
  metadata for the optional `programs.<field>` argument instead of duplicating
  field-based string construction.
- Split lowered value, point, field, and opening-input schema validation into
  `schema/ops/lowered/values.rs`. The parent lowered validator now dispatches
  dialect operations while value-family shape rules live in their own module.
- Split lowered transcript schema validation into
  `schema/ops/lowered/transcript.rs`, keeping transcript init/absorb/squeeze
  shape rules separate from the lowered dialect dispatcher.
- Split lowered oracle schema validation into `schema/ops/lowered/oracles.rs`,
  isolating dense trace, one-hot chunk, optional advice, oracle ref, and
  oracle-family shape rules from the lowered dialect dispatcher.
- Split lowered PCS schema validation into `schema/ops/lowered/pcs.rs` and
  lowered opening schema validation into `schema/ops/lowered/opening.rs`.
  Commitment, receive, opening-claim, opening-batch, and batch-open/verify
  shape rules now live in family-specific modules.
- Added shared sumcheck proof-op lowering helpers under
  `pass/support/sumcheck_proofs/lowering.rs`. Kernel-resolution and
  compute-to-CPU sumcheck passes now ask support to lower claim, batch, and
  driver proof ops with the named sumcheck attrs and result counts.
- Extended the shared sumcheck proof-op lowering helpers to kernel claim and
  kernel driver ops. Compute-to-CPU kernelized sumcheck lowering now reuses the
  same support notation, and kernel-resolution claim lowering now uses named
  `LoweredResultCount::One` mapping instead of the first-result wrapper.
- Removed the pass support first-result append-and-map wrapper. Custom
  single-result lowerings for field expressions, compute-to-CPU PCS,
  transcript, and oracle-family append now use `append_and_map_result_count`
  with explicit `LoweredResultCount::One`.
- Reused the shared sumcheck proof-op lowering helpers in party-to-compute
  sumcheck lowering. Role-specific prover/verifier claim and driver targets now
  share the same claim, batch, and driver notation helpers as later pass stages.
- Split the concrete transcript verifier state machine into
  `pass/verify/transcript/thread.rs`. The public transcript verification pass
  now owns traversal and op dispatch while `TranscriptThread` owns state
  initialization, input checks, and state advancement.
- Split concrete transcript verifier op classification into
  `pass/verify/transcript/ops.rs`. The verifier traversal now dispatches on a
  typed `TranscriptOp` family instead of matching raw operation-name groups
  inline.
- Moved generated crate artifact role bucketing into
  `emit/rust/artifacts/crate_graph/artifacts_by_role.rs`. Crate graph assembly
  now validates artifacts and maps role buckets into generated crates without
  owning the bucket data structure.
- Added a shared generated role API source context in
  `emit/rust/artifacts/role_api/context.rs`. Prover and verifier API renderers
  now share discovery of stages, modules, commitment metadata, active
  extensions, common names, protocol snake name, field type, and transcript
  trait while keeping role-specific rendering separate.
- Moved generated role API extension program-field and helper-item selection
  onto `RoleApiRole`. Prover and verifier renderers now ask their declared role
  for extension program fields, default program fields, and helper items instead
  of directly indexing prover/verifier extension branches.
- Moved generated role API extension error-field selection onto `RoleApiRole`.
  Prover and verifier error renderers now ask their declared role for extension
  error variants, error items, and error conversions instead of directly
  indexing prover/verifier extension branches.
- Moved generated role API extension import selection onto `RoleApiRole`.
  Prover and verifier import renderers now ask their declared role for extension
  imports instead of directly indexing prover/verifier extension branches.
- Moved common generated role API extension source slots onto `RoleApiRole`.
  Generated `lib.rs`, prover inputs/proof assembly, and verifier proof/input
  rendering now share role-selected `lib_module`, `input_fields`, and
  `proof_fields` access while leaving role-specific extension hooks local.
- Split `RoleApiRole` into a small `role` module plus `role::extension`.
  Role identity, source paths, and program type suffixes now stay separate from
  the extension source-slot selector table as the role API cleanup continues.
- Factored shared sumcheck proof lowering through a private
  `SumcheckProofShape` descriptor. Claim, batch, driver, and kernelized
  sumcheck proof helpers now name their operand/result/attr shape while one
  helper owns the repeated append-and-map plumbing.
- Moved PCS and opening proof-family lowering shape onto `PcsOpFamily` and
  `OpeningOpFamily`. The shared dispatchers now classify an operation family
  once, then lower using family-owned attrs, result types, and result counts.
- Moved fixed value-family lowering shape onto `ValueOpFamily`. Opening-input,
  point, and field value cases now share one fixed-result lowering path while
  field expressions remain the explicit custom lowering case.
- Moved transcript lowering shape onto `TranscriptOpFamily`. Init,
  absorb-bytes, and squeeze dispatch now share one transcript result lowering
  helper, with squeeze-specific result type computation kept explicit.
- Moved sumcheck value lowering shape onto `SumcheckValueFamily`. Eval and
  instance-result lowering now follow the same classify-then-family-owned-shape
  pattern as value, transcript, opening, and PCS support.
- Split sumcheck proof support notation into dedicated attr and result-type
  modules. The `sumcheck_proofs` façade now re-exports proof lowering helpers
  separately from the paper-level attr/result-type tables.
- Split value-lowering result-type notation out of value attrs and field
  expression attr construction. The value-lowering façade still re-exports the
  same compute/CPU result-type constants for pass-local dialect policies.
- Split sumcheck-value and transcript result-type notation out of attr
  notation. Sumcheck eval/instance and transcript state/squeeze result-type
  constants now live beside their result-type helper functions and stay
  re-exported through the existing façades.
- Split opening proof result-type notation behind an `opening_proofs` façade.
  Opening claim, batch, and batch-opening compute/CPU result types now live in
  a dedicated result-type module like the other proof/value support families.
- Split sumcheck proof shape descriptors out of sumcheck proof lowering. The
  lowering module now owns append/map mechanics while `shape.rs` owns claim,
  batch, driver, and kernelized result-count/attr descriptors.
- Split verifier role API proof and input struct rendering behind the existing
  `proof_inputs` façade. Proof type rendering and verifier input inventory
  rendering now live in separate verifier submodules.
- Split generated role-crate `src/lib.rs` rendering into inventory and role
  export submodules. The parent `generated_lib` path now assembles the crate
  shell while child modules own stage inventory rows and role-specific exports.
- Split generic role API source scanning into public-item scanners and
  kernel-import discovery. The `source_scan` façade keeps discovery call sites
  stable while separating Rust item introspection from kernel module lookup.
- Factored generated role API program discovery into a shared discovery helper.
  Stage and commitment metadata discovery now use one program type/const lookup
  shape before the typed program descriptors consume that metadata.
- Factored generated role API entrypoint discovery into a shared discovery
  helper. Stage and commitment metadata discovery now share verifier/prover
  default and with-program function scanning while declaring prover prefixes.
- Split generated stage API type-name discovery into a stage-local type
  inventory helper. Stage metadata assembly now consumes derived artifacts,
  output, eval, and verifier input type names instead of constructing them
  inline.
- Split generated stage API error-type discovery into a stage-local helper.
  Role-specific prover kernel error and verifier error fallback naming is now
  separate from the main stage metadata assembly path.
- Split generated prover entrypoint signature formatting out of entrypoint
  assembly. The parent prover entrypoint renderer now owns ordering, where
  bounds, and body insertion while a signature helper owns the repeated generic
  signature text.
- Split generated verifier entrypoint signature formatting out of entrypoint
  assembly. The parent verifier entrypoint renderer now owns ordering,
  extension hook placement, and body insertion while a signature helper owns
  transcript-bound signature text.
- Split generated role API error item rendering into variant, prover
  conversion, and verifier conversion modules. The `items::errors` façade keeps
  existing call sites stable while each child module owns one generated error
  construct.
- Split generated role API artifact-field rendering out of the `items` façade.
  Role API item rendering now separates artifact declarations from generated
  error notation while preserving the shared declaration call sites.
- Factored role API extension source selection through a borrowed role-slot
  view. The prover/verifier branch now happens once per selector call, and each
  generated role API accessor names the common extension slot it consumes.
- Split generated crate file-inventory construction out of crate graph
  assembly. Crate graph assembly now owns artifact validation, role bucketing,
  and role crate construction while `crate_graph::files` owns ordered generated
  file layout.
- Split generic role API public-item source scanning by discovery target. The
  `source_scan::items` façade now re-exports separate function, const, and type
  scanners instead of mixing Rust source introspection helpers in one file.
- Split party-to-compute sumcheck proof lowering into value-policy and
  role-target modules. The parent sumcheck proof file now dispatches proof ops
  while child modules name PIOP value lowering and prover/verifier target op
  choices.
- Split kernel-resolution and compute-to-CPU sumcheck value policies out of
  their proof dispatch files. The parent sumcheck modules now focus on
  kernel/verifier proof cases while child modules own post-compute value-family
  lowering policy.
- Split compute-to-CPU PCS commitment notation into target-op and attr
  conversion modules. The PCS commitment lowering body now imports through the
  existing notation façade while target naming and CPU attr rendering stay
  separate.
- Split fixed compute-to-CPU transcript lowering policy out of the transcript
  dispatcher. The parent CPU transcript module now chooses between shared
  absorb-bytes/squeeze lowering and CPU-specific init/artifact absorb cases.
- Split the role API extension borrowed-slot view out of the extension accessor
  module. `role::extension` now owns the public selector methods while
  `role::extension::slots` owns prover/verifier extension-slot projection.
- Split generated role API declaration wrappers by emitted declaration family.
  The `declarations` façade now re-exports program, artifact, and error
  declaration renderers from separate modules while role renderers keep stable
  call sites.
- Consolidated generated program field rendering through a typed
  `ProgramFieldStyle`. Program struct fields and default-program values now
  share one commitment/stage traversal while preserving the stage module alias
  distinction between struct types and default constants.
- Split pass-local PCS dialect policies out of party-to-compute,
  kernel-resolution, and compute-to-CPU proof dispatch files. The parent PCS
  modules now expose the lowering wrapper while child dialect modules own
  classification, result-type, and target-op policy.
- Factored kernel-resolution sumcheck kernelization through a named
  `KernelSumcheckShape`. Claim and driver kernel lowering now share registry
  lookup, operand lowering, kernel attr construction, and append/map mechanics
  while naming their target op, attrs, result types, and result count.
- Split compute-source value-family classification out of value-family shape
  notation. `value_lowering::family` now owns family/result-shape policy while
  `value_lowering::compute` owns post-compute operation-name classification.
- Split compute-source transcript classification out of transcript family
  notation. `transcript::family` now owns transcript attrs/result shape while
  `transcript::compute` owns post-compute operation-name classification.
- Split compute-source sumcheck-value classification out of sumcheck-value
  family notation. `sumcheck_lowering::family` now owns eval/instance result
  shape while `sumcheck_lowering::compute` owns post-compute op recognition.
- Split compute-source PCS and opening classification out of their proof-family
  notation modules. `pcs_lowering::family` and `opening_lowering::family` now
  own attrs/result-shape policy while their `compute` modules own post-compute
  op recognition.
- Split shared sumcheck proof lowering into standard wrappers, kernelized
  wrappers, and a private core append/map helper. The `sumcheck_proofs`
  lowering façade still re-exports the same public helpers while each child
  module owns one proof-lowering concern.
- Factored the common generated role API program/artifact declaration section
  shared by prover and verifier renderers. Role renderers now pass declaration
  type names to one helper while keeping errors, entrypoints, and role-specific
  ordering local.
- Split pass-local value dialect policies out of party-to-compute,
  kernel-resolution, and compute-to-CPU value lowering files. The parent value
  modules now expose only the lowering wrapper while child dialect modules own
  classification, result-type, and target-op policy.
- Split pass-local transcript dialect policies out of party-to-compute and
  kernel-resolution transcript lowering files. Together with the existing CPU
  split, transcript pass modules now consistently keep wrappers separate from
  dialect policy.
- Split pass-local opening dialect policies out of party-to-compute,
  kernel-resolution, and compute-to-CPU opening proof files. Opening pass
  modules now match the wrapper-plus-dialect-policy structure used by value,
  transcript, PCS, and sumcheck lowering.
- Split kernel-resolution sumcheck kernel shape descriptors out of the
  registry-driven lowering module. Kernel claim/driver target op, attr, result
  type, and result-count notation now lives beside the kernel lowering file
  instead of inside its control flow.
- Split generated role API program-field rendering out of the `programs`
  façade. The façade now exports field renderers while `programs::fields` owns
  typed struct/default field formatting and shared commitment/stage traversal.
- Consolidated generated role API program-aware entrypoint metadata into a
  shared `RoleApiProgramBinding`. Stage and commitment metadata now expose the
  same generic binding while discovery builds that binding through one helper
  instead of threading duplicate program/default function fields.
- Split generic protocol parameter attr notation into `pass::support::params`.
  Party-to-compute, kernel resolution, and compute-to-CPU drivers now share the
  protocol parameter attr list and compute-to-CPU symbol-ref conversion instead
  of carrying local parameter-copy mechanics.
- Added typed CPU attr-source descriptors through `pass::support::attr_sources`.
  PCS commitment, transcript, and parameter CPU attr notation now declare
  ordered attr specs with `LoweredAttr` while shared support owns
  symbol/string/int/bool source conversion mechanics.
- Split compute relation and CPU kernel attr lists into
  `pass::support::relations`. Party-to-compute, kernel resolution, and
  compute-to-CPU drivers now use shared pass notation for relation/kernel
  copied attrs instead of repeating literal attr arrays in pass control flow.
- Split artifact config behavior by concern. `ProtocolArtifactConfig` now keeps
  the data shape while child modules own dependency expansion, role-specific
  selections, and generated crate naming helpers used by manifest and crate
  assembly.
- Split generated crate manifest rendering by output mode. The manifest façade
  now dispatches on `ManifestMode` while standalone and workspace manifest
  formatting live in separate child modules with the same crate assembly call
  surface.
- Split kernel-resolution registry internals by boundary. The public
  `ComputeKernelSpec` now owns compute-kernel attr rendering in a spec module,
  while kernel materialization/caching lives in a private ensure module behind
  the existing registry façade and `KernelRegistry` API.
- Split kernelized sumcheck lowering internals under
  `kernel_resolution::proofs::sumcheck::kernel`. Claim/driver dispatch stays in
  the parent while kernel attr construction and shared append/map lowering live
  in dedicated child modules beside the existing shape descriptors.
- Split shared CPU attr-source support into descriptor and lowering modules.
  `LoweredAttr` now owns only typed attr notation, while the lowering child owns
  MLIR attr reads and source-value formatting behind the same support façade.
- Factored generated role API stage-module imports into a shared parent
  import helper. Prover and verifier import modules now retain role-specific
  imports while the common `crate::stages::{...}` alias block is emitted from
  one role API support path.
- Moved generated role API role-specific name selection onto `RoleApiNames`.
  Prover and verifier renderers now ask the shared name inventory for
  program/artifact/error type names by `RoleApiRole` instead of indexing
  role-specific fields at declaration and error call sites.
- Extended role-aware generated name selection to input and entrypoint type
  setup. Prover and verifier entrypoint renderers now receive input, program,
  artifact, and error names through `RoleApiNames` role selectors while keeping
  role-specific entrypoint structs unchanged.
- Reused `RoleApiNames` role selectors in generated role-crate `lib.rs`
  exports. Prover and verifier export renderers now share the same role-aware
  type-name access path as declarations and entrypoint setup.
- Split generated role API program-field style formatting out of field
  traversal. The `programs::fields` parent now walks commitment/stage program
  metadata while a child style module owns struct-field versus default-value
  source formatting.
- Split generated role API extension slot accessors by concern. Program-field,
  error, and source/lib/input/proof extension accessors now live in separate
  `role::extension` child modules while sharing the same role-selected slot
  projection.
- Split generated prover/verifier entrypoint type bundles out of entrypoint
  rendering modules. Entrypoint parents now own emitted function ordering and
  body assembly while child type modules own the role-specific input/program/
  artifact/error type-name bundle.
- Split generic function-source attr notation into `pass::support::functions`.
  Party-to-compute, kernel resolution, and compute-to-CPU now share function
  source attr formatting/copy policy instead of carrying inline `source`
  rewrites in the pass drivers.
- Split shared value-map result plumbing further under
  `pass::support::values::results`. Append-and-map flow and source/target
  result insertion now live in separate child modules behind the same support
  façade.
- Split primitive pass attr support into copied-attr collection and typed
  reader modules. The `attrs` façade still exposes `copy_attrs`,
  `string_attr`, `symbol_attr`, and `bool_attr` while child modules own
  collection versus reader diagnostics.
- Split module-copy support into role policy and copied-body rendering
  modules. `phase_copy_source` still owns module shell assembly while
  `PhaseCopyRole` owns role stamping and body rendering owns MLIR body text.
- Split the role pass façade into concrete role derivation and party
  projection child modules. Public role-pass entrypoints stay in `roles.rs`,
  while party projection owns `party.function` op notation and schema checks.
- Factored generated role API artifact value emission into the existing
  artifact item helper. Prover and verifier execution bodies now share the
  artifact construction-field path that sits beside artifact declaration
  fields.
- Split declaration handling out of party-to-compute, kernel-resolution, and
  compute-to-CPU pass spines. The parent passes now traverse and dispatch
  operation families, while declaration modules own params/function/relation
  copying and CPU params/function/kernel conversion.
- Moved generated role API declaration type-bundle construction onto
  `RoleApiSourceContext`. Prover and verifier renderers now consume the same
  role-aware declaration view instead of repeating program/artifact/field name
  selection.
- Added shared pass support for requiring role-bearing modules before
  role-dependent lowering. Party-to-compute, kernel resolution, and
  compute-to-CPU now share the role extraction path while preserving their
  phase-specific diagnostics.
- Replaced generated role API's independent role-name selectors with a typed
  `RoleApiRoleNames` view. Prover/verifier renderers, declaration context, and
  role-crate exports now consume one role-specific name bundle for input,
  program, artifact, and error type names.
- Split schema kernel-reference bookkeeping into a private tracker. The main
  schema verifier now owns phase/op validation flow and named lowered-phase
  policy checks, while the tracker owns compute/CPU kernel symbol collection
  and missing-kernel diagnostics.
- Split the public `SchemaError` type and its conversion impls out of the
  schema verifier façade. `schema.rs` now re-exports the error type while the
  error module owns display, `MlirError`, and transcript verifier conversion
  behavior.
- Split schema phase policy and module phase-attribute validation into a
  private phase module and removed the unused phase parameter from operation
  validation. The schema verifier now asks `ModulePhase` for lowered/verifier
  policy while op validation stays phase independent.
- Split the schema traversal loop into a private verifier module. The public
  `schema.rs` façade now owns phase-specific entrypoints and re-exports while
  `schema::verify` owns module traversal, op validation dispatch, verifier
  lowering policy, and kernel reference verification.
- Split counted ordered-operand schema validation out of primitive shape
  constraints. Exact/min operand-result count checks stay in `shape.rs`, while
  counted dynamic operands own count attr, ordered symbol attr, and operand
  owner diagnostics in a child module.
- Split schema operand-owner symbol helpers out of the symbol constraint
  module. Symbol lookup and attr equality stay in `symbols.rs`, while operand
  result/symbol ownership diagnostics live in a child module reused by opening
  and counted-shape validation.
- Split schema typed attr readers from required-attr presence checks.
  `constraints::attrs` now keeps the required-attribute façade and parsing
  module, while a reader child owns string/symbol/int/symbol-array extraction
  diagnostics.
- Split PIOP sumcheck operation validation out of the mixed PIOP dispatcher.
  Sumcheck claim, batch, driver, eval, and instance-result shape rules now live
  in a `piop::sumcheck` child module beside the shared PIOP attr notation.
- Split PIOP opening operation validation out of the mixed PIOP dispatcher.
  Opening input, claim, equality, and batch shape/equality rules now live in a
  `piop::opening` child module beside the shared PIOP attr notation.
- Split PIOP transcript operation validation out of the mixed PIOP dispatcher.
  Transcript state, absorb, absorb-bytes, and squeeze shape rules now live in a
  `piop::transcript` child module.
- Split PIOP setup/oracle/commitment validation out of the mixed PIOP
  dispatcher. Oracle declarations, oracle-family declarations, publish ops, and
  protocol-level PCS commit shape rules now live in a `piop::setup` child
  module.
- Moved lowered compute/CPU sumcheck suffix dispatch into the lowered
  sumcheck schema module. The lowered dispatcher now delegates sumcheck
  claim/batch/driver/eval/instance-result cases to the family module that owns
  their attrs, shapes, and dialect capability checks.
- Moved lowered compute/CPU declaration suffix dispatch into a declaration
  schema module. Params, function, relation, and kernel attr validation now
  lives beside declaration attr notation and dialect capability checks instead
  of in the lowered dispatcher.
- Split lowered declaration attr notation into an `attrs::declarations` child
  module. The lowered attrs façade still re-exports params/function/relation/
  kernel attr tables while declaration-specific notation no longer sits in the
  catch-all attr table file.
- Split lowered oracle attr notation into an `attrs::oracles` child module.
  Dense trace, one-hot chunk, optional advice, oracle ref, and oracle-family
  attr tables are now grouped beside the oracle schema family that consumes
  them.
- Split lowered transcript attr notation into an `attrs::transcript` child
  module. Transcript init, absorb, absorb-bytes, and squeeze attr tables are
  now grouped beside the transcript schema family that consumes them.
- Split lowered PCS attr notation into an `attrs::pcs` child module. PCS
  commitment, opening claim, opening batch, and batch opening attr tables are
  now grouped beside the PCS schema family that consumes them.
- Split lowered value attr notation into an `attrs::values` child module.
  Opening input, point, field, and polynomial-basis attr tables are now grouped
  beside the value schema family that consumes them.
- Split lowered opening attr notation into an `attrs::opening` child module.
  Opening claim and opening batch attr tables are now grouped beside the
  opening schema family, leaving the lowered attrs parent as a façade of
  family re-exports.
- Split lowered sumcheck attr notation into a child module under the sumcheck
  schema family. Claim, batch, driver, eval, and instance-result attr tables
  now sit beside the lowered sumcheck shape rules that consume them.
- Split schema op-shape descriptor structs into a child module under shape
  support. The shape support parent now reads as the named operand/result shape
  inventory while descriptor fields stay visible only to schema op support.
- Moved remaining inline foundation schema attr lists into foundation notation.
  Field binary/pow and party-function validation now use named attr tables
  instead of raw arrays in the operation dispatcher.
- Moved remaining PIOP transcript/setup inline attr arrays into PIOP notation.
  Oracle, oracle-family, publish, protocol PCS commit, and transcript shape
  validation now reference named attr tables from the PIOP notation module.
- Named the shared opening-claim equality attr table in schema op support.
  PIOP and lowered opening equality validators now share the same explicit
  support notation instead of embedding attr literals in the helper.
- Factored Jolt runtime-stage verifier program aliases into `stage_common`.
  Stages 4-7 now share the default-transcript and verifier-program alias
  renderer while keeping stage-specific verifier data and relation code local.
- Moved remaining inline pass attr notation for value expressions and
  party-to-compute function declarations into pass support helpers. Field pow,
  Lagrange basis eval, and compute-function source attrs are now named beside
  the lowering support that consumes them.
- Factored repeated Jolt runtime-stage role module source assembly into
  `stage_common`. Stages 4-7 now share the prover/verifier source wrapper while
  keeping stage-specific imports, types, constants, and entrypoints local.
- Added a fallible Jolt role-module source wrapper for stages 2 and 3. Their
  prover/verifier source assembly now shares the same source-wrapper boundary
  while preserving role-specific constant validation errors.
- Factored Jolt stage role-to-filename selection into `stage_common`. Stage
  files still own their concrete generated filenames, while the repeated
  prover/verifier role switch is shared across stages 1-8.
- Split Jolt stage role-source helpers into a `stage_common::source` child
  module. Shared filename and module-source wrappers now live separately from
  stage import/type alias notation.
- Split Jolt stage prover import notation into a `stage_common::imports` child
  module. Import-shape flags and generated prover import rendering now sit
  beside each other instead of in the mixed stage-common façade.
- Split Jolt stage verifier type/error notation into a
  `stage_common::verifier_types` child module. Stage verifier error shapes,
  type-alias shapes, default transcript aliases, and runtime verifier-program
  aliases now live apart from source-wrapper and import rendering helpers.
- Reused the fallible Jolt role-module source wrapper for stage 1. Stage 1 now
  shares the same prover/verifier source assembly path as stages 2 and 3 while
  preserving its borrowed import/type sections and constant validation errors.
- Moved protocol-local role-to-filename selection into the Jolt Rust source
  helper. Stage filename selection and commitment-phase filename selection now
  share the same role switch while keeping concrete filenames local.
- Split Jolt Rust literal rendering into a `source::literals` child module.
  String literal, optional string literal, and string-array literal helpers now
  sit behind the existing source façade instead of mixing with const/array
  item rendering.
- Split Jolt Rust array rendering into a `source::arrays` child module. Plan
  arrays, string arrays, usize arrays, rustfmt-skip arrays, and interned
  string-array rendering now live behind the existing source façade.
- Split Jolt Rust const/struct rendering into a `source::consts` child module.
  Params consts, struct consts, inline struct consts, and scalar value consts
  now sit behind the same source façade as literal and array rendering.
- Split Jolt Rust role-source rendering into a `source::roles` child module.
  Role filename selection and role module source assembly now live behind the
  same source façade as literal, array, and const rendering.
- Split Jolt MLIR operand-symbol helpers into an `mlir::operands` child
  module. Operand-owner diagnostics and symbol extraction now sit apart from
  typed attr readers behind the existing MLIR helper façade.
- Split Jolt MLIR typed attr readers into an `mlir::attrs` child module.
  String/symbol/int/bool/array readers and attr diagnostics now live apart from
  operand helpers and operation-name formatting behind the same MLIR façade.
- Split Jolt stage verifier notation one level deeper into `shapes`, `errors`,
  and `aliases`. The stage-common façade still exports the same helpers while
  shape descriptors, verifier error rendering, and type-alias rendering now
  have separate homes.
- Split commitment-phase MLIR extraction into a `commitment::parse` child
  module. Commitment source emission stays in the parent while CPU op scanning,
  typed attr reads, skip-policy parsing, and transcript artifact operand
  resolution now live behind the same commitment façade.
- Split commitment-phase generated constant rendering into a
  `commitment::constants` child module. The parent still assembles generated
  source sections while params, plan arrays, transcript plan constants, and
  skip-policy Rust notation sit in one source-rendering module.
- Split commitment-phase generated entrypoint/runtime source text into a
  `commitment::source` child module. The parent commitment façade keeps plan
  types, target validation, import/type assembly, oracle provider rendering,
  and public emit entrypoints apart from the generated prover/verifier runtime
  text.
- Split Stage 8 MLIR extraction into a `stage8::parse` child module. The
  parent stage emitter now keeps target validation and generated source
  rendering, while CPU op scanning, operand-symbol extraction, and claim
  source-stage backfilling live behind the same stage façade.
- Split Stage 8 generated source rendering into a `stage8::source` child
  module. The parent stage emitter now owns the public extraction/emission
  entrypoints plus target validation, while type text, plan constants, and
  opening-input literals live apart from MLIR parsing.
- Split Stage 1 MLIR extraction into a `stage1::parse` child module. Stage 1
  target validation and generated Rust rendering stay in the parent while CPU
  op scanning, typed attr reads, operand-symbol extraction, and plan
  construction are isolated behind the same stage façade.
- Split Stage 1 target validation into a `stage1::verify` child module.
  Parameter support checks, role-specific kernel/relation binding checks,
  batch-count validation, and opening-flow validation now sit apart from
  generated Rust source rendering.
- Split Stage 2 MLIR extraction into a `stage2::parse` child module. Stage 2
  field-expression, sumcheck, opening, point, transcript, and program-step plan
  construction now lives apart from target validation and generated Rust
  rendering behind the same stage façade.
- Split Stage 2 target validation into a `stage2::verify` child module.
  Field-flow checks, kernel ABI checks, role-specific driver validation, point
  source validation, and opening batch validation now live apart from generated
  Rust source rendering.
- Split Stage 3 MLIR extraction into a `stage3::parse` child module. Stage 3
  field-expression, sumcheck, opening, opening-equality, point, transcript, and
  program-step plan construction now lives apart from target validation and
  generated Rust rendering.
- Split Stage 3 target validation into a `stage3::verify` child module.
  Field-flow checks, kernel ABI checks, role-specific driver validation, point
  source validation, opening equality validation, and opening batch validation
  now live apart from generated Rust source rendering.
- Split Stage 4 MLIR extraction into a `stage4::parse` child module. Stage 4
  field-expression, sumcheck, transcript absorb-bytes, opening equality, point,
  and program-step plan construction now lives apart from target validation and
  generated Rust rendering.
- Split Stage 4 target validation into a `stage4::verify` child module.
  Transcript-step checks, field-flow checks, kernel ABI checks, role-specific
  driver validation, opening equality validation, and opening batch validation
  now live apart from generated Rust source rendering.
- Split Stage 5 MLIR extraction into a `stage5::parse` child module. Stage 5
  field-expression, sumcheck, transcript absorb-bytes, opening equality, point,
  and program-step plan construction now lives apart from target validation and
  generated Rust rendering.
- Split Stage 5 target validation into a `stage5::verify` child module.
  Transcript-step checks, field-flow checks, kernel ABI checks, role-specific
  driver validation, opening equality validation, and opening batch validation
  now live apart from generated Rust source rendering.
- Split Stage 6 MLIR extraction into a `stage6::parse` child module. Stage 6
  field-expression, sumcheck, transcript absorb-bytes, point-zero, opening
  equality, point, and program-step plan construction now lives apart from
  target validation and generated Rust rendering.
- Split Stage 6 target validation into a `stage6::verify` child module.
  Transcript-step checks, field-flow checks, kernel ABI checks, point-zero field
  validation, role-specific driver validation, opening equality validation, and
  opening batch validation now live apart from generated Rust source rendering.
- Split Stage 7 MLIR extraction into a `stage7::parse` child module. Stage 7
  field-expression, sumcheck, transcript absorb-bytes, point-zero, opening
  equality, point, and program-step plan construction now lives apart from
  target validation and generated Rust rendering.
- Split Stage 7 target validation into a `stage7::verify` child module.
  Transcript-step checks, field-flow checks, kernel ABI checks, point-zero field
  validation, role-specific driver validation, opening equality validation, and
  opening batch validation now live apart from generated Rust source rendering.
- Split Stage 8 target validation into a `stage8::verify` child module.
  Function-name validation, single opening-batch/proof validation, role-specific
  PCS mode validation, evaluation-point source checks, and opening-claim source
  checks now live apart from Stage 8 source rendering.
- Split Stage 4 generated constant rendering into a `stage4::constants` child
  module. Params, program-step, transcript, field, kernel, sumcheck, point, and
  opening plan constants now live apart from Stage 4 entrypoint/runtime source
  text.
- Split Stage 4 generated entrypoint/runtime source text into a
  `stage4::source` child module. The parent stage façade keeps source
  assembly, imports/types, constants-facing role helpers, target validation,
  and public emit entrypoints apart from the generated prover/verifier runtime
  text.
- Split Stage 5 generated constant rendering into a `stage5::constants` child
  module. Params, program-step, transcript, field, kernel, sumcheck, point, and
  opening plan constants now live apart from Stage 5 entrypoint/runtime source
  text.
- Split Stage 5 generated entrypoint/runtime source text into a
  `stage5::source` child module. The parent stage façade keeps source
  assembly, imports/types, constants-facing role helpers, target validation,
  and public emit entrypoints apart from the generated prover/verifier runtime
  text.
- Split Stage 6 generated constant rendering into a `stage6::constants` child
  module. Params, program-step, transcript, field, kernel, sumcheck, point-zero,
  point, and opening plan constants now live apart from Stage 6
  entrypoint/runtime source text.
- Split Stage 6 generated entrypoint/runtime source text into a
  `stage6::source` child module. The parent stage façade keeps source
  assembly, imports/types, constants-facing role helpers, target validation,
  and public emit entrypoints apart from the generated prover/verifier runtime
  text.
- Split Stage 6 generated verifier type/data rendering into a `stage6::types`
  child module. Stage 6-specific bytecode verifier data structs and
  relation/bytecode symbol tables now live apart from the parent source
  assembly façade and the generated runtime entrypoint text.
- Split Stage 7 generated constant rendering into a `stage7::constants` child
  module. Params, program-step, transcript, field, kernel, sumcheck, point-zero,
  point, and opening plan constants now live apart from Stage 7
  entrypoint/runtime source text.
- Split Stage 7 generated entrypoint/runtime source text into a
  `stage7::source` child module. The parent stage façade keeps source
  assembly, imports/types, constants-facing role helpers, target validation,
  and public emit entrypoints apart from the generated prover/verifier runtime
  text.
- Split Stage 8 generated constant rendering into a `stage8::constants` child
  module. Evaluation params, opening inputs, opening claims, opening batch, PCS
  proof, and program constants now live apart from Stage 8 generated type text.
- Split Stage 2 generated constant rendering into a `stage2::constants` child
  module. Fallible prover/verifier program constants, role-specific kernel and
  relation bindings, sumcheck plans, point plans, and opening plans now live
  apart from Stage 2 runtime verifier/prover source text.
- Split Stage 2 generated entrypoint/runtime source text into a
  `stage2::source` child module. The parent stage façade now keeps plan types,
  imports/types, source assembly, target validation, and public emit
  entrypoints apart from the large generated prover/verifier runtime text.
- Split Stage 3 generated constant rendering into a `stage3::constants` child
  module. Fallible prover/verifier program constants, role-specific kernel and
  relation bindings, transcript/opening inputs, sumcheck plans, point plans,
  opening equalities, and opening batches now live apart from Stage 3 runtime
  verifier/prover source text.
- Split Stage 3 generated entrypoint/runtime source text into a
  `stage3::source` child module. The parent stage façade now keeps plan types,
  imports/types, source assembly, target validation, and public emit
  entrypoints apart from the generated prover/verifier runtime text.
- Split Stage 1 generated constant rendering into a `stage1::constants` child
  module. Prover and verifier params, transcript, kernel/relation-bound
  sumcheck plans, sumcheck results/evals, and opening plans now live apart from
  Stage 1 import/type/entrypoint and runtime verifier source text.
- Split Stage 1 generated entrypoint/runtime source text into a
  `stage1::source` child module. The parent stage façade now keeps plan types,
  imports/types, source assembly, target validation, and public emit
  entrypoints apart from the generated prover/verifier runtime text.
- Moved the thin generic pass façade from `crates/bolt/src/pass.rs` to
  `crates/bolt/src/pass/mod.rs`. The public root exports stay source-compatible
  while the pass tree now matches the target module layout in this goal file.
- Moved the thin generic Rust artifact façade from
  `crates/bolt/src/emit/rust/artifacts.rs` to
  `crates/bolt/src/emit/rust/artifacts/mod.rs`. The crate-graph, role-API,
  support, and type submodules keep the same public exports while the artifact
  tree now uses the same directory-module layout as the pass tree.
- Moved the thin generic schema façade from `crates/bolt/src/schema.rs` to
  `crates/bolt/src/schema/mod.rs`. Protocol/concrete/party/compute/CPU schema
  entrypoints and `SchemaError` keep the same public exports while schema
  operation, constraint, phase, kernel-reference, and verifier modules live
  under the schema tree.
- Updated the generic artifact API quarantine test to read the new
  `emit/rust/artifacts/mod.rs` façade path. The recursive generic compiler
  Jolt-leakage test already covers the moved pass, schema, and artifact
  subtrees.

## Definition Of Done

- `crates/bolt/src/lib.rs` exports generic compiler APIs at the root and keeps
  Jolt APIs namespaced under `bolt::protocols::jolt`.
- `crates/bolt/src/emit/rust` contains generic Rust backend mechanics only.
- `JoltProtocolStage`, Jolt artifact config, Jolt stage emitters, Jolt relation
  mappings, and Jolt eval-proof composition live under `protocols/jolt`.
- Generic artifact assembly can produce role crates for a non-Jolt protocol
  fixture using only `ProtocolArtifactConfig` and `ProtocolStage` data.
- Generic passes and validators have automated hygiene tests preventing Jolt
  leakage.
- Existing Jolt correctness, transcript, tamper, import, generated-artifact,
  and performance gates remain available and green for implemented stages.
