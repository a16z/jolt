# Spec: Recursion Protocol Composition, Dory Assist, Field Inline, And Wrapper

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-19 |
| Status | draft |
| PR | TBD |

## Architecture Goal

Modular Jolt recursion separates protocol contracts from prover computation.
This document defines the protocol side: which claims exist, which stages run,
how claims compose, how transcript challenges bind to proof data, which openings
the verifier checks, and how the selected verifier computation lowers to R1CS
for wrapping.

Related architecture specs:

```text
recursion protocol composition:
  protocol contracts, claim formulas, verifier stages, wrapper R1CS boundaries

jolt-prover model crate:
  orchestration, witness generation, kernels, proof assembly, compute artifacts

jolt-verifier model crate:
  stage-by-stage verifier dataflow and typed proof/check interfaces
```

See also:

- [`jolt-prover` model crate spec](jolt-prover-model-crate.md)
- [`jolt-verifier` model crate spec](jolt-verifier-model-crate.md)

This recursion architecture extends the verifier model with three composable
protocol features:

1. **Dory assist**: a Dory-recursion auxiliary proof that replaces expensive
   ordinary Dory stage-8 verifier work with additional Jolt verifier stages.
2. **Field inline**: native field instructions, an FR register memory instance,
   and field-op guest R1CS rows.
3. **Wrapper**: a Spartan + HyperKZG SNARK over an R1CS encoding of the selected
   verifier computation.

Core invariant: protocol features compose through the verifier stage schedule.
Dory assist, field inline, ZK mode, and wrapping are selected by protocol
config; the wrapper receives one selected verifier computation as input.

## Composition Model

The modular stack is organized around protocol ownership:

```text
jolt-claims
  formulas, opening specs, relation IDs, stage metadata, protocol shapes

jolt-verifier
  selected stage schedule, proof-shape validation, transcript ordering

component crates
  reusable native and R1CS encodings for sumcheck, openings, Hyrax, BlindFold

jolt-wrapper
  assembly of the selected verifier computation into R1CS

jolt-wrapper::snarks::spartan_hyperkzg
  a SNARK backend over arbitrary R1CS
```

### Protocol Selection

Protocol selection is an explicit verifier input. It controls proof-shape
validation, stage scheduling, transcript replay, and wrapper assembly:

```rust
pub struct ProtocolSelection {
    pub zk: ZkSelection,
    pub field_inline: FieldInlineConfig,
    pub dory_assist: DoryAssistConfig,
    pub wrapper: Option<WrapperSelection>,
}

pub enum ZkSelection {
    Transparent,
    BlindFold,
}

pub enum WrapperSelection {
    SpartanHyperKzg,
    Gnark,
}
```

Interpretation:

```text
selection.zk
  chooses transparent claims or BlindFold/ZK proof payloads

selection.field_inline
  enables FR memory-checking payloads and field-op rows

selection.dory_assist
  extends the ordinary Jolt schedule with Dory-assist stages

selection.wrapper
  asks jolt-wrapper to prove the selected verifier computation
```

The proof model can carry optional protocol payloads. `ProtocolSelection`
defines which payloads are required for the selected proof shape and which
stage schedule consumes them.

### Crate Roles

```text
jolt-claims
  base Jolt, field-inline, and Dory-assist protocol formulas and metadata

jolt-verifier
  explicit verifier stage flow and protocol composition

jolt-r1cs
  R1CS builder, matrices, guest constraints, claim-expression lowering

jolt-sumcheck
  sumcheck proof models, native verifier, generic sumcheck R1CS constraints

jolt-openings
  opening claims, consistency checks, reduction machinery

jolt-blindfold
  BlindFold verifier logic and verifier-equation R1CS

jolt-hyrax
  Hyrax commitment/opening verification and reusable Hyrax R1CS constraints

jolt-dory
  Dory PCS proof artifacts and ordinary native Dory verification

jolt-wrapper
  selected-verifier R1CS assembly and SNARK backend adapters

jolt-hyperkzg
  HyperKZG PCS used by the first wrapper backend
```

The important split is that protocol descriptions live in `jolt-claims`,
stage execution lives in `jolt-verifier`, reusable R1CS encodings live with
their owning component crates, and `jolt-wrapper` ties those pieces together for
the chosen wrapper backend.

## Dory Assist

Dory assist is the recursion-paper "extended Jolt" path. Conceptually, it adds
stages after ordinary Jolt's information-theoretic verifier work. Those stages
prove the expensive Dory verifier computation with an auxiliary proof. The
selected Jolt verifier then checks the Dory-assist stages as the stage-8 PCS
path.

Dory assist is independent of field inline. If field inline is enabled, its
effects are already reflected in the base Jolt stage outputs consumed by the
Dory-opening path.

### Protocol Intuition

The Dory assist proof treats the Dory verifier computation as a typed execution
trace:

```text
public boundary:
  Dory proof artifact
  Dory verifier setup inputs
  Jolt evaluation claims and commitments from stage 8
  transcript-derived scalars

private witness:
  typed Dory verifier trace:
    GT exponentiation and multiplication
    G1/G2 scalar multiplication and addition
    Miller-loop and final-exponentiation trace rows
    operation outputs and wiring values

commitment:
  Hyrax commitment to the packed dense trace
```

The proof then establishes:

```text
local correctness:
  each operation family satisfies its algebraic relation

wiring correctness:
  outputs consumed by later operations match the producer outputs

boundary correctness:
  public Dory proof inputs and Jolt evaluation claims are the values used in
  the operation trace

packing correctness:
  many native-size operation traces are packed into one dense polynomial

opening correctness:
  Hyrax opens that dense polynomial at the verifier's required point
```

The latest `quang/recursion-temp` branch provides the best concrete structure
for this module shape. It has moved from the earlier jagged-transform staging
toward a three-stage prefix-packing pipeline:

```text
stage 1:
  packed GT exponentiation sumcheck

stage 2:
  batched constraints:
    GT shift and claim reduction
    GT multiplication
    G1/G2 scalar multiplication
    G1/G2 addition
    multi-Miller-loop constraints
    AST-derived wiring and boundary constraints

stage 3:
  prefix packing reduction to one dense polynomial opening

PCS opening:
  Hyrax opening of the dense trace
```

The branch currently has an external pairing boundary for final verification.
The full Dory-assist target moves pairing and final exponentiation into the
Dory-assist trace and stage-2 constraint layer.

### `jolt-claims` Protocol Layout

Target layout:

```text
crates/jolt-claims/src/protocols/dory_assist/
  mod.rs
  config.rs
  ids.rs
  proof_shape.rs
  verifier_shape.rs
  public_inputs.rs
  witness_shape.rs
  metadata.rs
  transcript.rs
  opening_boundary.rs
  prefix_packing.rs
  wiring_plan.rs
  constraints/
    mod.rs
    config.rs
    system.rs
    sumcheck.rs
    poly_types.rs
  gt/
    mod.rs
    types.rs
    indexing.rs
    exponentiation.rs
    multiplication.rs
    shift.rs
    base_power.rs
    stage1_base_openings.rs
    stage2_base_openings.rs
    stage2_openings.rs
    wiring.rs
  g1/
    mod.rs
    types.rs
    indexing.rs
    addition.rs
    scalar_multiplication.rs
    wiring.rs
  g2/
    mod.rs
    types.rs
    indexing.rs
    addition.rs
    scalar_multiplication.rs
    wiring.rs
  pairing/
    mod.rs
    multi_miller_loop.rs
    final_exponentiation.rs
    shift.rs
```

This mirrors the recursion branch structure while moving protocol facts into
`jolt-claims`:

```text
recursion/constraints/*
  -> dory_assist::constraints

recursion/gt/*
  -> dory_assist::gt

recursion/g1/*
  -> dory_assist::g1

recursion/g2/*
  -> dory_assist::g2

recursion/pairing/*
  -> dory_assist::pairing

recursion/prefix_packing.rs
  -> dory_assist::prefix_packing

recursion/wiring_plan.rs
  -> dory_assist::wiring_plan
```

`jolt-claims::protocols::dory_assist` owns:

```text
operation family definitions:
  constraint type, public inputs, native arity, poly types, opening IDs

stage specs:
  stage 1 packed GT exp
  stage 2 batched constraints
  stage 3 prefix packing
  dense-trace opening boundary

wiring specs:
  typed copy constraints derived from the Dory verifier AST

packing specs:
  canonical dense layout, prefix codewords, opening-point normalization

public boundary:
  Dory proof artifact, verifier setup inputs, Jolt evaluation claims
```

Dory assist emits a generic dense-trace opening request. `jolt-hyrax` owns the
Hyrax commitment/opening protocol and its R1CS encoding.

### API Shape

Initial protocol surface:

```rust
pub struct DoryAssistConfig {
    pub enabled: bool,
    pub pack_bits: usize,
}

pub struct DoryAssistVerifierSpec<F> {
    pub transcript: DoryAssistTranscriptSpec,
    pub stages: DoryAssistStageSpecs<F>,
    pub opening_boundary: DoryAssistOpeningBoundary<F>,
    pub public_inputs: DoryAssistPublicInputSpec,
}

pub struct DoryAssistStageSpecs<F> {
    pub stage1: PackedGtExpStageSpec<F>,
    pub stage2: BatchedDoryConstraintStageSpec<F>,
    pub stage3: PrefixPackingStageSpec<F>,
}

pub struct DoryAssistPublicInputs<F> {
    pub jolt_evaluation_claims: JoltEvaluationClaims<F>,
    pub dory_proof: DoryProofPublicInput<F>,
    pub verifier_setup_digest: F,
}

pub struct DoryAssistOpeningBoundary<F> {
    pub dense_commitment: HyraxCommitment,
    pub dense_point: Vec<F>,
    pub dense_eval: F,
}
```

The proof shape follows the branch's `RecursionProof` structure:

```rust
pub struct DoryAssistProof<F, T, HyraxProof> {
    pub stage1_proof: SumcheckInstanceProof<F, T>,
    pub stage2_proof: SumcheckInstanceProof<F, T>,
    pub stage3_packed_eval: F,
    pub opening_proof: HyraxProof,
    pub opening_claims: DoryAssistOpeningClaims<F>,
    pub dense_commitment: HyraxCommitment,
}
```

The full pairing/final-exponentiation trace extends `ConstraintType`,
`PolyType`, and stage-2 operation-family specs.

### `jolt-verifier` Integration

`jolt-verifier` owns the selected stage schedule. Dory assist is a stage
extension in the same style as the base verifier model:

```text
crates/jolt-verifier/src/dory_assist/
  mod.rs
  proof.rs
  inputs.rs
  outputs.rs
  metadata.rs
  stage1/
    mod.rs
    verify.rs
  stage2/
    mod.rs
    verify.rs
  stage3/
    mod.rs
    verify.rs
  pcs.rs
```

Verifier flow:

```text
ordinary selected Jolt:
  stages 1-7
  -> ordinary stage-8 PCS/Dory opening verification

selected Jolt with Dory assist:
  stages 1-7
  -> build Dory assist public boundary from stage-8 opening data
  -> dory_assist::stage1::verify
  -> dory_assist::stage2::verify
  -> dory_assist::stage3::verify
  -> dory_assist::pcs::verify through jolt-hyrax
```

The branch's `api.rs` has the right high-level shape for the boundary:

```text
base verifier stages 1-7
  -> stage-8 opening snapshot
  -> Dory witness/proof artifact
  -> Dory assist proof
  -> verifier replays the same selected boundary
```

In the modular architecture, `jolt-verifier`/`jolt-claims` own that typed
boundary. Heavy witness extraction and dense trace construction remain
compute-side concerns in the `jolt-prover` spec.

### R1CS And Wrapper Hooks

Dory assist contributes verifier computation to the wrapper:

```text
stage 1/2 sumcheck checks
  -> jolt-sumcheck::r1cs

operation-family claim equations
  -> jolt-claims formulas + jolt-r1cs lowering

wiring/boundary equations
  -> dory_assist::wiring_plan + jolt-r1cs lowering

prefix packing
  -> dory_assist::prefix_packing R1CS helper

dense Hyrax opening
  -> jolt-hyrax::r1cs
```

The wrapper consumes these helpers through the selected verifier schedule.
Dory-specific formulas stay in `jolt-claims`.

## Field Inline

Field inline adds native field operations to the Jolt VM. From the proof
machinery perspective, it is a uniform extension: a small FR register file is
another memory-checking instance, and field operations add local guest R1CS rows
plus an explicit field-product relation.

### Protocol Intuition

Ordinary Jolt already splits instruction semantics across components:

```text
Twist/read-write checking:
  proves memory/register accesses are consistent over time

instruction lookups and flags:
  prove which instruction relation is active on each cycle

Spartan/R1CS rows:
  prove local per-cycle algebraic semantics

product virtualization:
  proves selected multiplication witness products used by local constraints
```

Field inline follows the same pattern:

```text
FR register Twist:
  tracks reads and writes of field-register slots

field instruction flags:
  select FADD, FSUB, FMUL, FINV, ASSERT_EQ, MOV, etc.

field-op R1CS rows:
  enforce local field instruction semantics

field-product relation:
  proves FieldRs1Value * FieldRs2Value for FMUL/FINV-style rows
```

The design is additive. Enabling field inline adds FR relations to the existing
stage schedule; downstream Dory assist and wrapper logic consume the resulting
selected verifier computation.

### `jolt-claims` Layout

Target layout:

```text
crates/jolt-claims/src/protocols/field_inline/
  mod.rs
  config.rs
  ids.rs
  openings.rs
  stages.rs
  stage3.rs
  stage4.rs
  stage5.rs
  r1cs.rs
```

Initial surface:

```rust
pub struct FieldInlineConfig {
    pub enabled: bool,
    pub field_register_log_k: usize, // 4 for 16 slots
}

pub enum FieldInlineRelation {
    FieldRegistersClaimReduction,
    FieldRegistersReadWrite,
    FieldRegistersValEvaluation,
    FieldProduct,
}

pub enum FieldInlineOpening {
    FieldRs1Value,
    FieldRs2Value,
    FieldRdValue,
    FieldRegistersVal,
    FieldRs1Ra,
    FieldRs2Ra,
    FieldRdWa,
    FieldRdInc,
    FieldProduct,
}
```

Stage placement:

```text
stage 3:
  field-register claim reductions

stage 4:
  field-register read/write checking over T * 16

stage 5:
  field-register val evaluation

Spartan/product layer:
  explicit field-register-native product relation for FMUL/FINV
```

These helpers are used by the appropriate `jolt-verifier` stages, the same way
register and RAM memory relations are batched with the rest of the protocol.

### Guest R1CS Rows

Target layout:

```text
crates/jolt-r1cs/src/constraints/
  mod.rs
  rv64.rs
  field_inline.rs
```

`field_inline.rs` owns field-instruction guest rows:

```text
FADD:
  IsFieldAdd * (FieldRs1Value + FieldRs2Value - FieldRdValue) = 0

FSUB:
  IsFieldSub * (FieldRs1Value - FieldRs2Value - FieldRdValue) = 0

FMUL:
  IsFieldMul * (FieldProduct - FieldRdValue) = 0

FINV:
  IsFieldInv * (FieldRs1Value * FieldRdValue - 1) = 0

ASSERT_EQ:
  IsFieldAssertEq * (FieldRs1Value - FieldRs2Value) = 0

MOV/SLL bridge:
  bridge rows from XReg Rs1Value into FieldRdValue
```

FMUL uses an explicit field-register-native product:

```text
FieldProduct = FieldRs1Value * FieldRs2Value
FieldProduct = FieldRdValue
```

That product can batch with existing product-check machinery, but the relation
keeps a field-specific name so the witness path stays FR-aware.

### Verifier And Proof Shape

Field-inline proof data is config-gated data in the normal Jolt proof model:

```text
selection.field_inline.enabled = true
  proof carries FR memory and field-product payloads
  stages 3/4/5 batch FR memory claims with existing stage work
  Spartan/product checks include field-product relations

selection.field_inline.enabled = false
  selected stage schedule is ordinary Jolt
```

The verifier uses the base Jolt stages as the field-inline integration point.
Those stages call field-inline claim helpers when the config enables them.

### Trace Semantics

For a normal arithmetic cycle:

```text
cycle 10:
  opcode = ADD
  x-register reads/writes are active
  FR memory flags are inactive
  RV64 rows enforce rd = rs1 + rs2
```

For a field cycle:

```text
cycle 11:
  opcode = FMUL
  FR register reads/writes are active
  field-product relation is active
  field R1CS rows enforce FieldProduct = FieldRdValue
```

Bridge instructions are the place where normal registers and FR registers meet:

```text
cycle 12:
  opcode = FIELD_MOV_FROM_X
  x-register read is active
  FR register write is active
  bridge row enforces FieldRdValue = XRegRs1Value
```

Trace encoding decision: pure field-op cycles either suppress incidental
x-register accesses entirely or prove those accesses inert. The clean protocol
target is:

```text
pure field op:
  FR metadata/events + FR Twist + field R1CS

bridge op:
  x-register Twist + FR Twist + bridge R1CS
```

### Interaction With Dory Assist And Wrapper

Field inline changes the base Jolt proof that Dory assist consumes. The selected
verifier output already includes the field-inline stage effects, so Dory assist
continues to consume the ordinary typed stage boundary.

The wrapper sees field inline through:

```text
selected verifier stages 3/4/5
  -> field-inline claim formulas
  -> sumcheck verifier constraints
  -> field-product and guest R1CS boundary constraints
```

## Wrapper

The wrapper proves an R1CS encoding of the selected verifier computation. The
first backend is Spartan + HyperKZG. The backend accepts arbitrary R1CS;
protocol-specific lowering belongs with the protocol crates and R1CS component
crates.

### Design Intuition

The wrapper is a compiler from verifier computation to R1CS plus a SNARK backend:

```text
selected verifier computation
  -> transcript constraints
  -> sumcheck verifier constraints
  -> claim equation constraints
  -> opening/PCS verifier constraints
  -> R1CS instance and witness
  -> Spartan + HyperKZG proof
```

This makes field inline and Dory assist regular inputs to wrapper assembly.
They change the selected verifier schedule while the wrapper SNARK backend
remains generic over R1CS.

### `jolt-wrapper` Layout

Target layout:

```text
crates/jolt-wrapper/
  Cargo.toml
  src/
    lib.rs
    config.rs
    proof.rs
    public_inputs.rs
    r1cs/
      mod.rs
      assembly.rs
      instance.rs
      public_inputs.rs
      witness.rs
    snarks/
      mod.rs
      spartan_hyperkzg/
        mod.rs
        setup.rs
        prover.rs
        verifier.rs
        proof.rs
      gnark/
        mod.rs
```

The wrapper uses the generic selected-verifier assembly namespace. Reusable
Hyrax R1CS helpers live in `jolt-hyrax::r1cs`. During bring-up, a wrapper-local
Hyrax adapter can bridge to that interface; stable ownership stays in
`jolt-hyrax`.

### Assembly API

```rust
pub struct WrapperAssemblyInputs<F, PCS, VC, ZkProof> {
    pub selection: ProtocolSelection,
    pub preprocessing: JoltVerifierPreprocessing<PCS, VC>,
    pub public_io: JoltDevice,
    pub proof: JoltProof<PCS, VC, ZkProof>,
    pub public_inputs: WrapperPublicInputs<F>,
    pub witness_inputs: WrapperWitnessInputs<F>,
}

pub struct WrapperR1csInstance<F> {
    pub matrices: ConstraintMatrices<F>,
    pub witness: Vec<F>,
    pub public_inputs: Vec<F>,
}

pub fn assemble_selected_verifier_r1cs<F, PCS, VC, ZkProof>(
    inputs: WrapperAssemblyInputs<F, PCS, VC, ZkProof>,
) -> Result<WrapperR1csInstance<F>, WrapperError>;
```

Backend API:

```rust
pub fn prove_spartan_hyperkzg<F, P>(
    setup: &SpartanHyperKzgProverSetup<F, P>,
    instance: WrapperR1csInstance<F>,
) -> Result<WrapperProof<P>, WrapperError>;

pub fn verify_spartan_hyperkzg<F, P>(
    setup: &SpartanHyperKzgVerifierSetup<F, P>,
    proof: &WrapperProof<P>,
    public_inputs: &WrapperPublicInputs<F>,
) -> Result<(), WrapperError>;
```

### R1CS Ownership

Reusable R1CS encodings live with the crates that own the corresponding
protocol machinery:

```text
R1CS builder/matrices:
  jolt-r1cs

transcript hashing:
  jolt-transcript R1CS helpers

sumcheck verifier equations:
  jolt-sumcheck::r1cs

claim formulas:
  jolt-claims formulas lowered through jolt-r1cs

opening consistency:
  jolt-openings + jolt-r1cs helpers

field-inline guest rows:
  jolt-r1cs::constraints::field_inline

BlindFold verifier checks:
  jolt-blindfold

Hyrax verifier checks:
  jolt-hyrax::r1cs

Dory-assist prefix packing:
  jolt-claims::protocols::dory_assist::prefix_packing R1CS helper
```

`jolt-wrapper::r1cs::assembly` sequences these helpers according to
`ProtocolSelection` and the same stage order as `jolt-verifier`.

### Challenge Binding

For a self-contained wrapper, transcript challenges must be derived inside the
R1CS from the same messages absorbed by the native verifier:

```text
proof data / public inputs
  -> transcript absorb constraints
  -> challenge variables
  -> sumcheck and claim constraints use those same variables
```

A split verifier can expose challenges as public inputs, but then the outer
verifier must recompute and bind them. The standalone wrapper target uses the
self-contained path.

### Wrapper Target Order

The first complete target:

```text
base Jolt stages 1-7
  + Dory assist stages
  -> selected verifier R1CS
  -> Spartan + HyperKZG
```

Field inline composes by changing the base Jolt stages. ZK composes through
`selection.zk`.

## ZK Composition

ZK is another protocol axis in `ProtocolSelection`:

```text
Transparent:
  clear sumcheck round polynomials
  clear opening/evaluation claims
  verifier checks claim equalities directly

BlindFold:
  committed sumcheck round polynomials
  hidden opening/evaluation claims where needed
  verifier checks BlindFold proof / verifier-equation R1CS
```

Composition points:

```text
ZK + field inline:
  field-inline claims and openings participate in BlindFold equations at the
  same stage placement, primarily stages 3/4/5.

ZK + Dory assist:
  Dory assist extends the selected schedule. Its public boundary must match
  the selected visibility mode for the base Jolt proof.

ZK + wrapper:
  wrapper proves the selected verifier computation. If BlindFold is selected,
  wrapper assembly includes the BlindFold verifier checks.
```

There is one cost-model question to keep open:

```text
Option A:
  BlindFold first:
    ZK Jolt / ZK Jolt with Dory assist
      -> wrapper proves the ZK verifier computation

Option B:
  transparent Jolt + Dory assist
    -> wrapper
    -> add ZK at the wrapper layer
```

Option A is the direct composition. Option B may have a better cost profile for
some proof shapes because it avoids carrying BlindFold through earlier wrapper
work.

## End-To-End Flows

### Ordinary Jolt

```text
guest executes RV64 / existing SDK inline code
  -> Jolt proof
  -> jolt-verifier runs selected transparent/BlindFold schedule
  -> ordinary stage-8 PCS/Dory opening verification
```

### Field-Inline Jolt

```text
guest uses native field instructions
  -> trace includes FR register events
  -> proof includes FR memory and field-product payloads
  -> stages 3/4/5 batch FR claims with existing stage work
  -> field-op R1CS rows enforce local field instruction semantics
```

### Jolt With Dory Assist

```text
Jolt stages 1-7 verify normally
  -> stage-8 boundary constructs Dory assist public inputs
  -> Dory assist proves the Dory verifier trace
  -> verifier checks Dory assist stages and Hyrax opening
```

### Wrapped Jolt With Dory Assist

```text
selected verifier computation:
  base Jolt stages 1-7
  + Dory assist stages
  + optional field-inline stage additions
  + optional BlindFold verifier checks

wrapper assembly:
  selected verifier computation -> R1CS
  R1CS -> Spartan + HyperKZG proof
```

### Simple Recursion

```text
compile selected verifier to RV64 guest
  -> run inside Jolt
  -> field inline/native inlines accelerate verifier hotspots
```

## Open Questions

### Pure FieldOp X-Register Accesses

Protocol target: pure FR access for pure field operations, with explicit
x-register/FR bridge semantics for bridge operations. The implementation needs
a concrete trace encoding for field-op cycles.

### Field Product Batching

`FieldProduct` is an explicit field-register-native product relation.
The implementation can batch it with existing product virtualization checks
once the FR-aware witness path is present.

### ZK And Wrapper Ordering

BlindFold-before-wrapper and wrapper-before-ZK remain two candidate orderings.
The choice depends on the wrapper/BlindFold cost model.

### Hyrax R1CS Placement

`jolt-hyrax::r1cs` is the stable home. If wrapper assembly is the first
consumer, a wrapper adapter can bridge to the final API during bring-up.

## Testing

### `jolt-claims`

- Field-inline claim formula tests.
- Field-product relation tests.
- Dory-assist operation-family formula tests.
- Dory-assist wiring, boundary, and prefix-packing tests.
- Public-input binding tests for Jolt evaluation claims and Dory proof
  artifacts.

### `jolt-r1cs`

- Base RV64 constraints unchanged when field inline is disabled.
- Field-inline row helpers append expected FR rows when enabled.
- `FieldProduct` tests cover FMUL/FINV and batching with existing product
  checks.
- Claim-expression lowering tests for Dory-assist prefix-packing and wiring
  formulas.

### `jolt-verifier`

- `ProtocolSelection` drives proof-shape validation and selected stage
  scheduling.
- Transparent and BlindFold selections reject mismatched proof payloads.
- Field-inline payloads are required only when enabled.
- Dory-assist payloads are required only when enabled.
- Dory-assist stage tests mirror the recursion branch stage fixtures.
- Field-inline and Dory-assist additions follow the existing checkpointed
  verifier test harness and tamper manifest style.

### `jolt-hyrax`

- Native Hyrax verifier fixtures for Dory-assist dense-trace openings.
- R1CS verifier constraints match native verification for the same opening
  transcript.

### `jolt-wrapper`

- Wrapper assembly produces deterministic R1CS for a selected computation.
- Spartan + HyperKZG proves and verifies synthetic wrapper fixtures.
- Real fixtures cover base Jolt + Dory assist once typed verifier outputs exist.
- Mutating transcript challenges, sumcheck claims, prefix-packing values, Hyrax
  boundary values, or public inputs causes wrapper verification failure.

## Milestones

1. Add `jolt-claims::protocols::field_inline`.
2. Add `jolt-r1cs::constraints::field_inline` and explicit `FieldProduct`.
3. Add `jolt-claims::protocols::dory_assist` using the `quang/recursion-temp`
   stage/module shape.
4. Factor or add `jolt-hyrax` native verification and R1CS helpers.
5. Extend `jolt-verifier` proof-shape validation around `ProtocolSelection`.
6. Wire field-inline payloads into stages 3/4/5.
7. Wire Dory-assist as selected stages after base Jolt stages 1-7.
8. Add wrapper R1CS assembly for the selected verifier computation.
9. Implement `jolt-wrapper::snarks::spartan_hyperkzg`.
10. Add `snarks/gnark` once the R1CS boundary is stable.
