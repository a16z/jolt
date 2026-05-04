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
