# Bolt Code Quality Notes

This document distills compiler-engineering idioms from mature Rust compiler
projects into concrete expectations for Bolt. It is not a general Rust style
guide. The goal is to keep Bolt small, auditable, and pleasant to change while
it matures from the initial full-`Fr`, non-zk Jolt path into a generic protocol
compiler.

## Reference Projects

Use these projects as quality references, not as templates to copy wholesale:

- [Cranelift](https://cranelift.dev/): compiler backend, IR discipline,
  lowering phases, verifier culture, and embedder-facing APIs.
- [wasm-tools](https://github.com/bytecodealliance/wasm-tools): multi-crate
  parser/validator/printer/encoder structure, CLI/library separation, fuzzing,
  and fixture policy.
- [rust-analyzer architecture](https://rust-analyzer.github.io/book/contributing/architecture.html):
  explicit crate boundaries, architecture invariants, facade crates over
  implementation crates, and no hidden IO in core analysis.
- [rustc dev guide](https://rustc-dev-guide.rust-lang.org/overview.html):
  staged compiler docs, typed diagnostics, query-oriented thinking, and
  contributor-facing architecture explanations.
- [SWC](https://swc.rs/): production Rust compiler/tooling APIs, transform
  pipelines, performance culture, and generated/test artifact management.
- [Oxc](https://oxc.rs/): high-performance modular toolchain split into parser,
  transformer, resolver, linter, formatter, and minifier crates.
- [Gleam compiler](https://github.com/gleam-lang/gleam): readable compiler
  implementation, direct diagnostics, CLI/core separation, and approachable
  module boundaries.

## Meta Idioms

### Architecture Invariants Are First-Class

Mature compiler projects write down what must not happen. Bolt should do the
same in code comments, module docs, Semgrep rules, and CI.

Examples:

```text
generic Bolt code must not know Jolt relation names
generated verifier code must not import witness/prover/trace/core crates
MLIR plan data is source of truth; generated Rust should not infer protocol facts
fixtures under generated scratch paths are not checked-in review artifacts
```

When a boundary matters, encode it in more than prose:

- module layout
- crate dependency graph
- typed APIs
- Semgrep rules
- focused tests
- PR checklist items

### One Source Of Truth Per Layer

Compiler verbosity often comes from the same fact being represented several
ways. Pick one source of truth for each layer:

```text
protocol semantics: protocol package data under src/protocols/**
compiler IR: melior::ir::Module and typed schema wrappers
phase plans: typed plan structs derived from MLIR
Rust artifacts: generated output, not protocol source
runtime mechanics: shared handwritten support modules
```

If generated Rust has to rediscover protocol meaning from names, imports, or
string patterns, the compiler boundary has leaked.

### Public Facades, Plain Internals

Follow the rust-analyzer pattern: make the public boundary polished and stable,
but let internal compiler modules use direct data structures when that makes
the lowering clearer.

For Bolt:

- public API: small facade around parse/build/lower/emit/regenerate
- internal passes: explicit structs, enums, and plan data
- protocol packages: isolated Jolt-specific definitions
- generated crates: small public prove/verify/setup APIs, hidden stage details

Avoid turning every internal helper into a public abstraction. A compiler can
have direct internal code as long as the boundary is clean.

### Data Tables Beat Stage Copy-Paste

The Jolt path has many repeated stages. Repetition is acceptable in generated
artifacts, but the compiler should prefer data-driven descriptions over
hand-written stage forks.

Prefer:

```text
Stage::ALL
StagePlan
RelationPlan
RoleImportPlan
OpeningPlan
EvaluationClaimPlan
```

Avoid:

```text
stage1 special case
stage2 special case
stage3 special case
same import block copied into every emitter
same verifier check with only a relation name changed
```

The generated output may be repetitive if that improves auditability. The
generator should not be repetitive unless the repetition is genuinely clearer.

### Validators Are Compiler Phases

Cranelift and wasm-tools treat validation as real infrastructure, not as test
only code. Bolt should validate at phase boundaries:

```text
protocol package -> schema validation
protocol dialect -> concrete dialect validation
party projection -> role visibility validation
compute dialect -> executable dataflow validation
cpu dialect -> Rust emission validation
generated Rust -> import/API/artifact validation
```

Validation errors should say what invariant failed and where the bad fact came
from. Prefer typed error variants over stringly error paths.

## Bolt Style Rules

### Keep Names Short But Domain-Specific

Use names that carry compiler meaning without narrating implementation steps.

Good:

```text
StagePlan
RolePlan
ClaimSet
OpeningPlan
ImportPlan
LoweringError
EmitError
SchemaError
```

Usually too verbose:

```text
GeneratedVerifierStageSpecificNamedEvaluationContainer
JoltProtocolRustEmitterVerifierCommonHelperManager
```

Usually too vague:

```text
Data
Info
Thing
Manager
Helper
Context
```

`Context` is acceptable when it truly represents shared phase state. Otherwise
prefer the precise thing being passed.

### Prefer Typed IDs And Enums Over Strings

Strings are fine at parse boundaries and diagnostics. They are risky in core
compiler logic.

Prefer:

```text
StageId
Role
Dialect
RelationId
OracleId
ClaimId
```

Avoid matching on:

```text
"stage1"
"verifier"
"jolt_*"
"instruction_lookup_key"
```

If a string must cross a boundary, convert it into a typed value immediately and
emit diagnostics from that conversion point.

### Use Traits, Generics, And Enums For Compiler Concepts

The strongest Rust compiler projects use the type system to make illegal states
hard to express. They do not make everything generic. The pattern is:

```text
enum: closed sets the compiler owns
trait: narrow extension point with multiple implementations
generic: code that is genuinely parametric over a phase, role, dialect, or backend
newtype: domain string that should stop behaving like an arbitrary string
```

Bolt already has one good example: `BoltModule<'c, P: Phase>` prevents passing a
protocol module into a CPU-only API. That same style should spread to stages,
roles, dialect operations, attributes, symbols, and emit plans.

Reference-project idioms to borrow:

- Cranelift uses typed IR entities, target abstractions, and validators so that
  lowering code works over compiler concepts rather than raw strings.
- wasm-tools uses typed parser/encoder/validator APIs where binary format
  concepts become enums and structs early, then pass through library APIs in
  typed form.
- rust-analyzer distinguishes facade crates from implementation crates and
  heavily uses typed IDs, arenas, and traits at API boundaries.
- SWC and Oxc use visitor/fold/traverse traits for AST transformations instead
  of ad hoc recursive functions spread through callers.
- rustc uses typed compiler context, typed IDs, and phase-specific IR rather
  than passing source-level names through every layer.

For Bolt, use this decision rule:

```text
closed vocabulary?      enum
domain name?            newtype
phase-parametric pass?  generic over Phase markers
protocol extension?     trait with associated types
open backend/registry?  trait or closure-backed trait
syntax rendering?       Display/SourceFragment, not String helpers everywhere
```

Current Bolt examples that are Rust-native:

- `BoltModule<'c, P: Phase>` uses type-state for phase boundaries.
- `Role`, `ProtocolStageKind`, `ArtifactCrateRole`, and manifest mode enums
  encode closed sets.
- `KernelRegistry` gives kernel selection a small trait boundary and supports
  closure implementations.

Current Bolt examples that still read too much like C or Python:

- Large `match operation_name(op).as_str()` blocks.
- Attribute requirements represented as `&["sym_name", "stage", ...]`.
- Protocol concepts represented as `String` fields deep into codegen plans.
- Symbol construction via repeated `format!("stage6....{oracle}")`.
- Rust source generated through many free functions returning `String`.
- API extension points represented as raw source snippets in string fields.

Prefer newtypes for domain strings:

```rust
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolName(String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttrName(&'static str);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RustPath(String);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageId {
    Commitment,
    Stage1Outer,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6,
    Stage7,
    Evaluation,
}
```

These wrappers should implement `Display`, `TryFrom<&str>` where needed, and
small constructors that enforce naming rules. Once parsed, compiler logic
should pass `SymbolName`, `StageId`, `OracleName`, `RelationName`, `ClaimName`,
`ProofSlot`, `RustIdent`, `RustPath`, and `CrateName` instead of `String`.

Prefer enums for dialect operations:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeOpKind {
    Function,
    Params,
    Kernel,
    TranscriptInit,
    TranscriptAbsorb,
    TranscriptSqueeze,
    OpeningInput,
}

impl ComputeOpKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Function => "compute.function",
            Self::Params => "compute.params",
            Self::Kernel => "compute.kernel",
            Self::TranscriptInit => "compute.transcript_init",
            Self::TranscriptAbsorb => "compute.transcript_absorb",
            Self::TranscriptSqueeze => "compute.transcript_squeeze",
            Self::OpeningInput => "compute.opening_input",
        }
    }
}
```

Then validation, lowering, and emission can switch on `ComputeOpKind` instead
of string literals. Unknown operation names should be converted to typed errors
at the edge.

Prefer typed operation specs over repeated schema strings:

```rust
pub struct OpSpec {
    pub name: OpName,
    pub attrs: &'static [AttrName],
    pub operands: Shape,
    pub results: Shape,
}
```

This lets schema validation become table-driven and makes the required attrs
available to lowering and diagnostics without duplicating lists.

Prefer traits with associated types for extension points:

```rust
pub trait LoweringPass {
    type Source: Phase;
    type Target: Phase;

    fn run<'c>(
        &self,
        context: &'c MeliorContext,
        module: &BoltModule<'c, Self::Source>,
    ) -> Result<BoltModule<'c, Self::Target>, MlirError>;
}

pub trait ProtocolPackage {
    type Stage: Copy + Eq + Ord;
    type Params;

    fn artifact_config(&self) -> ProtocolArtifactConfig;
    fn stages(&self) -> &'static [Self::Stage];
}

pub trait RustEmitter {
    type Plan;

    fn emit(&self, plan: &Self::Plan, out: &mut SourceWriter) -> Result<(), EmitError>;
}
```

Associated types keep APIs readable. Prefer them over generic parameter lists
that make every function look like `fn f<P, R, E, S, T>(...)`.

Prefer enum dispatch when the set is closed:

```rust
pub enum StageEmitter {
    Commitment,
    Proof(StageId),
    Evaluation,
}
```

Use trait objects only when Bolt needs runtime plugin behavior. For current
Jolt-on-Bolt work, most sets are closed and should be enums or tables, not
boxed traits.

Prefer rendering traits over string helpers:

```rust
pub trait SourceFragment {
    fn write_to(&self, out: &mut SourceWriter);
}
```

This lets plans render themselves through a common writer and reduces helpers
like `rust_str_array`, `symbol_array_attr`, and many one-off `emit_* -> String`
functions. `String` remains the final output buffer, not the intermediate
representation.

Migration order:

1. Newtype the most dangerous strings: symbol names, Rust paths, crate names,
   stage names, relation names, oracle names, and claim names.
2. Add `TryFrom<&str>` parsers at MLIR/schema boundaries.
3. Introduce dialect op enums for `protocol`, `piop`, `compute`, and `cpu`.
4. Replace validation match blocks with `OpSpec` tables.
5. Make lowering passes implement a common `LoweringPass` trait after the
   typed source/target APIs are stable.
6. Introduce `SourceWriter`/`SourceFragment` for Rust emission.
7. Replace raw API extension strings with typed extension hooks or structured
   source fragments.

Avoid type-system overreach:

- Do not add a trait for a concept with one implementation and no clear
  extension boundary.
- Do not genericize code just to remove a match over a closed enum.
- Do not hide straightforward lowering behind deep trait stacks.
- Do not make generated Rust clever; use the type system in Bolt, then emit
  boring artifacts.

### Prefer Standard Traits For Rust-Native Notation

Rust-native compiler code should make common operations look common. Bolt should
lean on standard traits where the meaning is unsurprising, and reserve custom
traits for real compiler/protocol concepts.

Good standard-trait targets:

```text
Display            diagnostic names, generated labels, symbol wrappers
Debug              structural compiler state
TryFrom<&str>      parse strings at MLIR/schema boundaries
From<T>            lossless conversions and error aggregation
AsRef<str>         cheap access to symbol/name wrappers
Borrow<str>        lookup typed keys by string at boundary maps
IntoIterator       plan collections and typed table views
Index<Id>          dense generated stores when missing values are impossible
std::fmt::Write    source generation into a writer
Add/Sub/Mul/Neg    expression builders, not proof arithmetic side effects
```

Avoid using traits when the notation would surprise a reader. `Deref` is usually
wrong for domain wrappers because it hides boundary crossings. `Index` should
only be used for tables where the ID is compiler-generated and guaranteed valid;
proof-dependent lookup should return `Result`.

Domain wrappers should feel like normal Rust values:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Symbol(&'static str);

impl Symbol {
    pub const fn new(value: &'static str) -> Self {
        Self(value)
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl AsRef<str> for Symbol {
    fn as_ref(&self) -> &str {
        self.0
    }
}
```

Generated code can then read as typed data without losing auditability:

```rust
pub const BYTECODE_GAMMA: Symbol = Symbol::new("stage6.bytecode_read_raf.gamma");

pub const STAGE6_PROGRAM_STEPS: &[ProgramStepPlan] = &[
    ProgramStepPlan {
        kind: ProgramStepKind::TranscriptSqueeze,
        symbol: BYTECODE_GAMMA,
    },
];
```

For expression construction inside Bolt, use operator traits on an expression
builder type instead of constructing `"field.mul"` and `"field.add"` strings by
hand:

```rust
let gamma = FieldExpr::symbol(BYTECODE_GAMMA);
let input = FieldExpr::symbol(INPUT_IMM);
let term = gamma * input + FieldExpr::constant(1);
```

The result should still lower into explicit MLIR operations. The overloaded
operators are only notation for building typed compiler IR, not hidden proof
execution.

For source generation, prefer `std::fmt::Write` and small `Display` or
`SourceFragment` implementations over helpers that return string fragments:

```rust
impl SourceFragment for ProgramStepPlan {
    fn write_to(&self, out: &mut SourceWriter) -> std::fmt::Result {
        writeln!(
            out,
            "    ProgramStepPlan {{ kind: {:?}, symbol: {} }},",
            self.kind,
            self.symbol.rust_literal(),
        )
    }
}
```

This turns emission into "write this typed thing" rather than "join a pile of
strings." It also gives the generator one place to decide how symbols, labels,
arrays, and enum variants render.

For generated runtime stores, prefer typed IDs and standard lookup traits:

```rust
pub struct ScalarStore<F> {
    values: Vec<F>,
}

impl<F> std::ops::Index<ScalarId> for ScalarStore<F> {
    type Output = F;

    fn index(&self, id: ScalarId) -> &Self::Output {
        &self.values[id.0 as usize]
    }
}
```

Use this only after Bolt has generated and validated all IDs. For
proof-dependent values, keep fallible APIs:

```rust
pub fn scalar_or<E>(&self, id: ScalarId, missing: impl FnOnce(ScalarId) -> E) -> Result<F, E>;
```

Custom traits should capture protocol/compiler roles:

```rust
pub trait NamedPlan {
    fn symbol(&self) -> Symbol;
}

pub trait EvaluatesFieldExpr<F> {
    type Error;

    fn eval(&self, expr: &FieldExprPlan, inputs: &[F]) -> Result<F, Self::Error>;
}

pub trait ExecutesStage<F> {
    type Program;
    type Proof;
    type Artifacts;
    type Error;

    fn execute(
        program: &'static Self::Program,
        proof: &Self::Proof,
    ) -> Result<Self::Artifacts, Self::Error>;
}
```

Prefer custom traits when they let shared runtime code operate over
prover/verifier or stage-specific plans without string matching. Do not use a
trait simply to avoid writing a direct function.

Recommended notation upgrades:

```text
String symbol              -> Symbol, Oracle, Claim, Batch
string parse helper        -> TryFrom<&str>
manual label formatting    -> Display / SourceFragment
Vec lookup by name         -> Index<Id> for generated dense stores
kind == "scalar"           -> ChallengeKind::Scalar
formula == "field.mul"     -> FieldFormula::Mul
source.push_str(format!)   -> writeln!(SourceWriter, ...)
ad hoc stage API scraping  -> typed StageArtifactMetadata
```

### Use Result Flow Instead Of Panics

Production compiler paths should return typed errors.

Prefer:

```rust
let Some(plan) = plans.get(stage) else {
    return Err(EmitError::MissingStagePlan { stage });
};
```

Avoid in non-test code:

```rust
let plan = plans.get(stage).expect("stage plan exists");
```

Allowed panic-like behavior should be rare and reserved for impossible internal
invariants. Even then, prefer a narrow `InternalInvariant` error until the
boundary is fully hardened.

### Comments Explain Invariants, Not Syntax

Good comments explain why a boundary exists or why a representation is shaped a
certain way.

Useful:

```text
Verifier code must not depend on witness material; these claims are absorbed
from proof-owned data only.
```

Not useful:

```text
Loop over the stages.
```

Module docs should describe ownership, invariants, and allowed dependencies.
Function bodies should be obvious enough to need few comments.

### Keep Generated Code Boring

Generated Rust should be:

- deterministic
- formatted
- mechanically structured
- low on clever helper macros
- clear about generated ownership
- small enough for review when possible

Prefer moving reusable mechanics into handwritten runtime modules and emitting
declarative plan data. This reduces both generator complexity and generated
artifact size.

### Make Generated Plan Data Typed

The current generated Jolt artifacts are moving in the right direction: they
emit plan structs, stage proof types, stage error enums, input-provider traits,
and shared verifier runtime helpers. The main remaining weakness is that most
generated plan data is still linked by raw `&'static str` values.

Current generated-code patterns to improve:

- `StageParams`, `KernelPlan`, `OpeningInputPlan`, `FieldExprPlan`,
  `SumcheckClaimPlan`, `SumcheckBatchPlan`, and related structs use
  `&'static str` for almost every semantic field.
- `ProgramStepPlan.kind`, transcript squeeze `kind`, opening claim `claim_kind`,
  batch `policy`, point concat `layout`, equality `mode`, and field expression
  `formula` are closed vocabularies represented as strings.
- Runtime stores and evaluation code key values by `&'static str`, for example
  `ValueStore`, stage eval lookup, commitment lookup, and evaluation proof
  coefficient maps.
- Stage 8 proof code branches on string values like `source_stage == "stage7"`.
- The role API generator infers stage APIs by scanning generated source text for
  public item names. The emitter should already know that metadata.

Keep human-readable names for diagnostics, but do not make strings the runtime
linking mechanism.

Near-term generated runtime types:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Symbol(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Oracle(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Claim(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Batch(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramStepKind {
    TranscriptSqueeze,
    TranscriptAbsorbBytes,
    SumcheckDriver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChallengeKind {
    Scalar,
    Vector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimKind {
    Committed,
    Virtual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningEqualityMode {
    PointAndEval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldFormula {
    Add,
    Sub,
    Mul,
    Neg,
    Pow { exponent: usize },
}
```

With those wrappers, generated plans stay readable but become harder to misuse:

```rust
pub struct ProgramStepPlan {
    pub kind: ProgramStepKind,
    pub symbol: Symbol,
}

pub struct TranscriptSqueezePlan {
    pub symbol: Symbol,
    pub label: &'static [u8],
    pub kind: ChallengeKind,
    pub count: usize,
}

pub struct OpeningClaimPlan {
    pub symbol: Claim,
    pub oracle: Oracle,
    pub claim_kind: ClaimKind,
    pub point_source: Symbol,
    pub eval_source: Symbol,
}
```

This is a low-risk first step because the newtypes are `Copy`, preserve the
underlying labels, and can implement `Display`/`as_str()` for diagnostics and
transcript labels.

Medium-term generated reference types:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PointId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClaimId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchId(pub u16);
```

For generated stage-local plan data, prefer IDs over string joins:

```rust
pub struct SumcheckClaimPlan {
    pub id: ClaimId,
    pub symbol: Symbol,
    pub claim_value: ScalarId,
    pub input_openings: &'static [ClaimId],
}

pub struct SumcheckBatchPlan {
    pub id: BatchId,
    pub symbol: Batch,
    pub ordered_claims: &'static [ClaimId],
}

pub struct SumcheckDriverPlan {
    pub symbol: Symbol,
    pub batch: BatchId,
}
```

The generated module can still expose `const fn name(id) -> &'static str` tables
for diagnostics. Runtime execution should use typed IDs, not repeated string
lookups.

Use enums for closed generated decisions:

```rust
match plan.kind {
    ProgramStepKind::TranscriptSqueeze => execute_squeeze(...),
    ProgramStepKind::TranscriptAbsorbBytes => execute_absorb_bytes(...),
    ProgramStepKind::SumcheckDriver => execute_sumcheck(...),
}
```

Avoid:

```rust
match plan.kind {
    "transcript_squeeze" => ...
    "sumcheck_driver" => ...
    _ => unsupported(...)
}
```

Use traits where stages share behavior but keep stage-specific data explicit:

```rust
pub trait StageProgram {
    type Error;

    const STAGE: StageId;

    fn steps(&self) -> &'static [ProgramStepPlan];
    fn transcript_squeezes(&self) -> &'static [TranscriptSqueezePlan];
}

pub trait StageInputProvider<F> {
    fn opening(&self, symbol: Symbol) -> Option<StageOpeningInputValue<F>>;
}
```

Do not force every stage into one dynamic interface. Top-level prove/verify APIs
can remain explicit by stage because stage inputs differ. The trait target is
shared mechanics such as plan walking, value lookup, transcript work, and
diagnostics.

Generator changes needed to produce typed output:

1. Make Bolt's extracted CPU plan structs typed first. For example,
   `Stage6ProgramStepPlan.kind` should become `ProgramStepKind`, and
   `Stage6OpeningClaimPlan.claim_kind` should become `ClaimKind`.
2. Emit typed literals such as `ProgramStepKind::TranscriptSqueeze` instead of
   `"transcript_squeeze"`.
3. Generate `Symbol`, `Oracle`, `Claim`, and `Batch` newtype constructors in a
   shared runtime module or per-role common module.
4. Replace generated `BTreeMap<&'static str, F>` and `Vec<(&'static str, F)>`
   stores with typed-key equivalents.
5. For large stage-local graphs, generate compact IDs and name tables rather
   than comparing strings.
6. Stop scraping generated source in role API assembly. Stage emitters should
   return both `RustSourceFile` and typed `StageRustApi` metadata.
7. Replace raw source-string extension fields with typed extension hooks or
   `SourceFragment` values.

Suggested generated-code migration order:

```text
first:
  newtypes for Symbol/Oracle/Claim/Batch/RustPath/CrateName
  enums for ProgramStepKind, ChallengeKind, ClaimKind, BatchPolicy, FieldFormula

next:
  typed ValueStore keys
  typed stage/evaluation proof lookup
  typed commitment lookup by Oracle

later:
  generated stage-local IDs for dense plans
  StageProgram and StageInputProvider traits for shared mechanics
  source-free StageRustApi metadata from emitters
```

Be careful with giant generated enums. For thousands of local values, a compact
`ValueId(u16)` plus a generated name table is usually better than a huge enum.
For small closed vocabularies, enums are the right Rust shape.

## Verbosity Reduction Playbook

Use this checklist when a Bolt module starts feeling too large:

1. Does this code repeat across prover and verifier?
   Move shared mechanics into a role-independent helper and keep role data
   explicit.

2. Does this code repeat across stages?
   Introduce a typed stage plan or table. Keep per-stage differences as data.

3. Does this code branch on Jolt-specific names from a generic module?
   Move the fact into `src/protocols/jolt/**` and pass it as typed protocol
   data.

4. Does this emitter build Rust by ad hoc string fragments?
   Use a small source writer, shared import planner, and structured sections.
   Consider syntax-aware generation only where it pays for itself.

5. Does generated output link plan data by string names?
   Prefer newtyped symbols first, then generated IDs where lookup is dense or
   performance-sensitive.

6. Does the generated output match on closed string vocabularies?
   Emit enums for step kind, challenge kind, claim kind, field formula, batch
   policy, and equality mode.

7. Does the generator inspect generated source to discover its own API?
   Return typed artifact metadata from emitters instead.

8. Does an error message require reading the call stack to understand?
   Add a typed error variant with the phase, stage, role, and symbol involved.

9. Does a helper name include three or more nouns?
   Split it or name the compiler concept directly.

10. Does a function mix analysis, validation, and emission?
   Split it by phase. Analysis returns plan data, validation checks the plan,
   emission renders the plan.

11. Does a test compare a huge generated file by default?
   Prefer structural assertions. Keep full goldens only when they are
   deliberate review artifacts.

## Code Organization Targets

Target shape:

```text
src/schema/**
  typed schema, ids, attrs, operation wrappers

src/pass/**
  generic pass traits, pass runner, diagnostics, validation plumbing

src/emit/**
  role-independent source writing, import planning, artifact layout

src/protocols/jolt/**
  Jolt package definitions, Jolt-specific lowering facts, Jolt stage plans

src/protocols/jolt/emit/**
  Jolt-specific plan-to-Rust glue where generic emit is insufficient
```

Generic modules may manipulate protocol-provided typed data. They should not
learn protocol semantics by matching on Jolt strings or stage names.

## Testing Idioms To Borrow

From the reference projects, Bolt should adopt these testing habits:

- filetests or snapshot-style tests for IR lowering when the textual form is
  compact and intentional
- structural tests for generated Rust imports, public APIs, and artifact layout
- roundtrip or regeneration tests that fail on nondeterminism
- verifier negative tests for malformed, missing, reordered, or tampered proof
  data
- fuzz/property tests for parsers, schema conversion, and lowering invariants
- small performance smoke tests on every PR and larger perf gates on schedule

Fixture policy:

```text
tracked fixtures: small, intentional goldens
ignored fixtures: generated scratch output
generated artifacts: checked in only when they are crate deliverables
```

## Review Checklist

Before landing Bolt compiler changes, ask:

- Is the source of truth for this fact clear?
- Did this change add protocol knowledge to a generic module?
- Did this change keep domain names typed, or did it pass raw strings deeper
  into the compiler?
- Is this abstraction an enum, trait, generic, or newtype for a concrete
  reason?
- Did this change make generated verifier code depend on prover, witness,
  trace, kernels, or core?
- Is repeated stage logic represented as data where practical?
- Are errors typed and actionable?
- Are panics, unwraps, and expects absent from production paths?
- Are generated artifacts deterministic?
- Are fixtures intentional and small?
- Is the public API smaller than the internal implementation?
- Could a reviewer understand the phase boundary without reading every call
  site?

## Practical Near-Term Targets

For the current Jolt-on-Bolt cleanup, prioritize:

1. Move stale Jolt-specific branches out of generic Bolt modules.
2. Centralize import planning for generated role crates.
3. Keep verifier generation witness-free and kernel-free.
4. Newtype domain strings before they spread through more lowering/codegen
   paths.
5. Replace operation-name string matches with dialect op enums where the set is
   closed.
6. Make generated plan output use newtyped symbols and enums for closed
   vocabularies.
7. Stop deriving role APIs by scanning generated source strings; return typed
   emitter metadata.
8. Replace repeated stage emitter code with typed plan data where it reduces
   real duplication.
9. Tighten typed errors around schema, lowering, and emission failures.
10. Add Semgrep crate-boundary rules before expanding the generated surface.
11. Keep full generated fixtures out of ordinary PR diffs.
12. Document any remaining verbosity as explicit debt in the relevant cleanup
   plan.

The quality bar is not "minimal code at all costs." The bar is code where the
compiler facts are represented once, boundaries are enforceable, and generated
artifacts are boring enough to audit.
