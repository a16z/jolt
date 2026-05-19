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
  protocol contracts, claim formulas, verifier stages, wrapper R1CS interfaces

jolt-prover model crate:
  orchestration, witness generation, kernels, proof assembly, compute artifacts

jolt-verifier model crate:
  stage-by-stage verifier dataflow and typed proof/check interfaces
```

See also:

- [`jolt-prover` model crate spec](jolt-prover-model-crate.md)
- [`jolt-verifier` model crate spec](jolt-verifier-model-crate.md)
- Recursion paper repo: <https://github.com/markosg04/recursion-paper>

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

The implementation model mirrors base Jolt. New protocol work enters as new
stage IDs, relation IDs, dimensions, claim formulas, openings, publics,
challenges, and concrete verifier stages. Dory assist uses different algebraic
components and workloads, but it follows the same claim/stage organization as
Spartan, RAM, registers, bytecode, instruction lookups, and the other base Jolt
components.

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
  Hyrax commitment/opening proof types, native verifier, and reusable R1CS
  constraints

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

### R1CS Module Pattern

Modular protocol crates expose native verifier APIs and, when the protocol must
be wrapped, a sibling `r1cs` module that encodes the same verifier equations
against `jolt-r1cs` builders.

```text
jolt-sumcheck
  native sumcheck verifier
  r1cs/ for generic sumcheck verifier equations

jolt-hyrax
  native Hyrax verifier
  r1cs/ for Hyrax opening verification constraints

jolt-blindfold
  native BlindFold verifier
  r1cs/ for BlindFold verifier-equation constraints

jolt-claims
  semantic claim formulas and protocol dimensions
  helpers that lower claim expressions through jolt-r1cs
```

`jolt-wrapper` sequences these modules. It does not own the generic Hyrax,
sumcheck, BlindFold, or claim-lowering logic.

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
public inputs:
  Dory proof artifact
  Dory verifier setup inputs
  Jolt evaluation claims and commitments from stage 8
  transcript-derived scalars

private witness:
  typed Dory verifier trace:
    GT exponentiation and multiplication
    G1/G2 scalar multiplication and addition
    Miller-loop trace rows
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

public-input consistency:
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
    AST-derived wiring and public-input constraints

stage 3:
  prefix packing reduction to one dense polynomial opening

PCS opening:
  Hyrax opening of the dense trace
```

Quang's branch is a protocol reference for the Dory-assist SNARK: stage order,
operation families, wiring shape, prefix packing, and Hyrax opening flow. The
implementation here is new modular code. Protocol facts move into
`jolt-claims`, selected-stage execution moves into `jolt-verifier`, Hyrax moves
into `jolt-hyrax`, and wrapper R1CS assembly consumes those modular crates.

The Dory-assist target proves the multi-Miller-loop work in the assist proof.
The selected verifier receives the resulting public GT value, computes final
exponentiation directly, and checks the public pairing equality. This is the
same style as other cheap deterministic verifier work such as equality-polynomial
evaluation: keep the expensive trace inside the assist proof, compute the cheap
public check natively.

### `jolt-claims` Protocol Layout

Target layout:

```text
crates/jolt-claims/src/protocols/dory_assist/
  mod.rs
  config.rs
  ids.rs
  stage.rs
  proof_shape.rs
  public_inputs.rs
  dimensions.rs
  transcript.rs
  constraints/
    mod.rs
    shape.rs
    poly_types.rs
    sumcheck.rs
  gt/
    mod.rs
    claims.rs
    openings.rs
    wiring.rs
  g1/
    mod.rs
    claims.rs
    openings.rs
    wiring.rs
  g2/
    mod.rs
    claims.rs
    openings.rs
    wiring.rs
  pairing/
    mod.rs
    claims.rs
    openings.rs
    final_check.rs
  packing/
    mod.rs
    prefix.rs
    dense_opening.rs
```

This mirrors the base Jolt claim structure. Small top-level modules define the
stage IDs, proof shape, dimensions, public inputs, transcript challenge IDs,
opening IDs, and PIOP stage shapes. Component modules own relation-specific
claims, openings, wiring formulas, sumcheck domains, and dense-opening metadata.
The component split tracks the Dory verifier's algebraic domains:

```text
constraints:
  shared constraint families, poly types, arity, and sumcheck shapes

gt:
  GT exponentiation, GT multiplication, base powers, GT shift, GT wiring

g1:
  G1 addition, G1 scalar multiplication, scalar-mul shift, G1 wiring

g2:
  G2 addition, G2 scalar multiplication, scalar-mul shift, G2 wiring

pairing:
  multi-Miller-loop claims and the public final-exponentiation check

packing:
  prefix-packing layout and dense-opening claim
```

`jolt-claims::protocols::dory_assist` owns:

```text
operation family definitions:
  constraint type, public inputs, native arity, poly types, opening IDs

stage specs:
  stage 1 packed GT exp
  stage 2 batched constraints
  stage 3 prefix packing
  dense-trace opening claim

wiring specs:
  typed copy constraints derived from the Dory verifier AST

packing specs:
  canonical dense layout, prefix codewords, opening-point normalization

public inputs:
  Dory proof artifact, verifier setup inputs, Jolt evaluation claims
```

Dory assist emits a generic dense-trace opening request. `jolt-hyrax` owns the
Hyrax commitment/opening protocol and its R1CS encoding.

### API Shape

The Dory-assist API mirrors the existing `jolt-claims::protocols::jolt`
surface. It defines IDs, dimensions, stage claims, and protocol claims. The
verifier takes stage-specific inputs and runs the concrete selected stages in
the same way it runs base Jolt stages.

```rust
pub struct DoryAssistConfig {
    pub enabled: bool,
    pub pack_bits: usize,
}

pub enum DoryAssistStageId {
    PackedGtExp,
    BatchedConstraints,
    PrefixPacking,
}

pub enum DoryAssistChallengeId {
    Gt(GtChallenge),
    G1(G1Challenge),
    G2(G2Challenge),
    Pairing(PairingChallenge),
    Packing(PackingChallenge),
}

pub enum DoryAssistOpeningId {
    Gt(GtOpening),
    G1(G1Opening),
    G2(G2Opening),
    Pairing(PairingOpening),
    DenseTrace,
}

pub enum DoryAssistPublicId {
    DoryProofArtifact,
    VerifierSetupDigest,
    JoltEvaluationClaim,
    PairingFinalCheckInput,
}

pub struct DoryAssistDimensions {
    pub constraints: DoryAssistConstraintDimensions,
    pub gt: GtDimensions,
    pub g1: G1Dimensions,
    pub g2: G2Dimensions,
    pub pairing: PairingDimensions,
    pub packing: PrefixPackingDimensions,
}

pub struct DoryAssistStageClaims<F> {
    pub id: DoryAssistStageId,
    pub sumcheck: DoryAssistSumcheckSpec,
    pub input: DoryAssistInputClaimExpression<F>,
    pub output: DoryAssistOutputClaimExpression<F>,
    pub consistency: Vec<DoryAssistConsistencyClaim<F>>,
}

pub struct DoryAssistProtocolClaims<F> {
    pub stages: Vec<DoryAssistStageClaims<F>>,
}

pub struct DoryAssistPublicInputs<F> {
    pub jolt_evaluation_claims: JoltEvaluationClaims<F>,
    pub dory_proof: DoryProofPublicInput<F>,
    pub verifier_setup_digest: F,
    pub pairing_final_check_inputs: PairingFinalCheckInputs<F>,
}

pub struct DoryAssistDenseOpeningClaim<F> {
    pub dense_commitment: HyraxCommitment,
    pub dense_point: Vec<F>,
    pub dense_eval: F,
}
```

This is the same pattern as base Jolt:

```text
base Jolt:
  JoltStageId
  JoltOpeningId / JoltPublicId / JoltChallengeId
  JoltFormulaDimensions
  JoltStageClaims
  JoltProtocolClaims

Dory assist:
  DoryAssistStageId
  DoryAssistOpeningId / DoryAssistPublicId / DoryAssistChallengeId
  DoryAssistDimensions
  DoryAssistStageClaims
  DoryAssistProtocolClaims
```

The component set changes, but the organization stays uniform. Base Jolt has
Spartan, instruction, RAM, registers, bytecode, booleanity, and advice
components. Dory assist has GT, G1, G2, pairing, constraints, and packing
components.

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

Multi-Miller-loop proving extends `ConstraintType`, `PolyType`, and stage-2
operation-family specs. Final exponentiation remains a native verifier
computation over public values.

### `jolt-verifier` Integration

`jolt-verifier` owns the selected stage schedule. Dory assist adds concrete
stages to that schedule in the same style as the base verifier model:

```text
crates/jolt-verifier/src/
  proof.rs
  stages/
    mod.rs
    dory_assist/
      mod.rs
      public_inputs.rs
      stage1/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
      stage2/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
      stage3/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
      hyrax_opening/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
```

This mirrors the existing `crates/jolt-verifier/src/stages/stage{1,2,3,4}/`
shape. Dory-assist proof payloads live in the verifier proof model, alongside
the other optional proof payloads selected by `ProtocolSelection`; the stage
modules own typed inputs, typed outputs, and the concrete verification step.

Verifier flow:

```text
ordinary selected Jolt:
  stages 1-7
  -> ordinary stage-8 PCS/Dory opening verification

selected Jolt with Dory assist:
  stages 1-7
  -> build Dory assist public inputs from stage-8 opening data
  -> stages::dory_assist::stage1::verify
  -> stages::dory_assist::stage2::verify
  -> stages::dory_assist::stage3::verify
  -> stages::dory_assist::hyrax_opening::verify through jolt-hyrax
```

The branch's `api.rs` has the right high-level shape for the handoff:

```text
base verifier stages 1-7
  -> stage-8 opening snapshot
  -> Dory witness/proof artifact
  -> Dory assist proof
  -> verifier replays the same selected public inputs
```

In the modular architecture, `jolt-verifier`/`jolt-claims` own those typed
public inputs. Heavy witness extraction and dense trace construction remain
compute-side concerns in the `jolt-prover` spec.

### R1CS And Wrapper Hooks

Dory assist contributes verifier computation to the wrapper:

```text
stage 1/2 sumcheck checks
  -> jolt-sumcheck::r1cs

operation-family claim equations
  -> jolt-claims formulas + jolt-r1cs lowering

wiring and public-input equations
  -> dory_assist component claims + jolt-r1cs lowering

prefix packing
  -> dory_assist::packing R1CS helper

dense Hyrax opening
  -> jolt-hyrax::r1cs
```

The wrapper consumes these helpers through the selected verifier schedule.
Dory-specific formulas stay in `jolt-claims`.

## Hyrax

`jolt-hyrax` factors Hyrax out as a reusable modular crate. The Dory-assist
pipeline uses Hyrax to commit to and open the packed dense trace, while wrapper
assembly uses `jolt-hyrax::r1cs` to prove the same opening verification inside
R1CS.

Target layout:

```text
crates/jolt-hyrax/
  Cargo.toml
  src/
    lib.rs
    commitment.rs
    proof.rs
    setup.rs
    transcript.rs
    verifier.rs
    opening.rs
    r1cs/
      mod.rs
      inputs.rs
      witness.rs
      constraints.rs
```

Native API sketch:

```rust
use jolt_crypto::{
    HomomorphicCommitment, JoltGroup, Pedersen, PedersenSetup, VectorCommitment,
    VectorCommitmentOpening,
};
use jolt_openings::EvaluationClaim;
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, Transcript};

pub type PedersenHyrax<G> = Hyrax<Pedersen<G>>;
pub type PedersenHyraxSetup<G> = PedersenSetup<G>;

pub struct Hyrax<VC: VectorCommitment> {
    _marker: core::marker::PhantomData<VC>,
}

pub struct HyraxCommitment<C> {
    pub row_commitments: Vec<C>,
    pub row_len: usize,
}

pub struct HyraxOpeningPoint<F> {
    pub row: Point<F>,
    pub entry: Point<F>,
}

pub struct HyraxOpeningClaim<F, C> {
    pub commitment: HyraxCommitment<C>,
    pub point: HyraxOpeningPoint<F>,
    pub evaluation: EvaluationClaim<F>,
}

pub struct HyraxOpeningProof<F> {
    pub row_opening: VectorCommitmentOpening<F>,
}

pub fn verify_hyrax_opening<VC, T>(
    setup: &VC::Setup,
    claim: &HyraxOpeningClaim<VC::Field, VC::Output>,
    proof: &HyraxOpeningProof<VC::Field>,
    transcript: &mut T,
) -> Result<(), HyraxError>
where
    VC: VectorCommitment,
    VC::Output: HomomorphicCommitment<VC::Field> + AppendToTranscript,
    T: Transcript<Challenge = VC::Field>;
```

The default instantiation is Pedersen over a `JoltGroup`, e.g.
`PedersenHyrax<G>`, but the verifier is generic over `VectorCommitment`. This
keeps Hyrax aligned with the existing `jolt-crypto` abstractions and reuses
`VectorCommitment::verify_committed_rows` for row-combined openings. The
commitment output must be homomorphic because the verifier combines row
commitments with equality-polynomial weights before checking the committed row.

R1CS API sketch:

```rust
pub struct HyraxR1csInputs<F> {
    pub public_claim: HyraxOpeningClaim<F, HyraxCommitmentVar>,
    pub proof_witness: HyraxOpeningProofVars<F>,
}

pub fn append_hyrax_verifier_constraints<F>(
    builder: &mut R1csBuilder<F>,
    inputs: HyraxR1csInputs<F>,
) -> Result<(), HyraxR1csError>;
```

Ownership:

```text
jolt-hyrax
  commitment/proof/setup types over jolt-crypto VectorCommitment
  native verifier using Pedersen / homomorphic commitment traits
  transcript absorption order for Hyrax messages
  R1CS encoding of the Hyrax verifier

jolt-claims::protocols::dory_assist::packing
  dense-opening claim produced by prefix packing

jolt-verifier::stages::dory_assist::hyrax_opening
  typed stage wrapper around jolt-hyrax native verification

jolt-wrapper
  calls jolt-hyrax::r1cs while assembling the selected verifier R1CS
```

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

conversion R1CS rows:
  enforce movement between ordinary x-register values and field-register values
```

The design is additive. Enabling field inline adds FR relations to the existing
stage schedule; downstream Dory assist and wrapper logic consume the resulting
selected verifier computation.

Field inline must support both BN254-width field values and smaller field
targets. The protocol stays generic over field width where practical, with
explicit support for 254-bit BN254 field elements and 128-bit field elements.

### `jolt-claims` Layout

Target layout:

```text
crates/jolt-claims/src/protocols/field_inline/
  mod.rs
  config.rs
  ids.rs
  stage.rs
  dimensions.rs
  public_inputs.rs
  registers/
    mod.rs
    claims.rs
    openings.rs
  product/
    mod.rs
    claims.rs
    openings.rs
  conversion/
    mod.rs
    claims.rs
    openings.rs
```

This mirrors the base Jolt and Dory-assist structure. `jolt-claims` defines the
field-inline IDs, dimensions, public inputs, and stage-claim formulas. Concrete
stage execution stays in `jolt-verifier`; concrete guest R1CS rows stay in
`jolt-r1cs::constraints::field_inline`.

Component ownership:

```text
registers:
  FR memory-checking claims for claim reduction, read/write, and val evaluation

product:
  explicit field-register-native product relation for FMUL/FINV

conversion:
  claims/openings needed by x-register <-> field-register movement
```

Initial surface:

```rust
pub struct FieldInlineConfig {
    pub enabled: bool,
    pub field_register_log_k: usize, // 4 for 16 slots
    pub field_width: FieldInlineWidth,
}

pub enum FieldInlineWidth {
    Bits128,
    Bits254,
    Native,
}

pub enum FieldInlineStageId {
    FieldRegistersClaimReduction,
    FieldRegistersReadWrite,
    FieldRegistersValEvaluation,
    FieldProductVirtualization,
    FieldConversion,
}

pub enum FieldInlineOpening {
    Registers(FieldRegisterOpening),
    Product(FieldProductOpening),
    Conversion(FieldConversionOpening),
}

pub enum FieldInlinePublicId {
    FieldWidth,
    FieldRegisterCount,
}

pub enum FieldInlineChallengeId {
    Registers(FieldRegisterChallenge),
    Product(FieldProductChallenge),
    Conversion(FieldConversionChallenge),
}

pub struct FieldInlineDimensions {
    pub field_register_log_k: usize,
    pub field_width: FieldInlineWidth,
}

pub struct FieldInlineStageClaims<F> {
    pub id: FieldInlineStageId,
    pub sumcheck: FieldInlineSumcheckSpec,
    pub input: FieldInlineInputClaimExpression<F>,
    pub output: FieldInlineOutputClaimExpression<F>,
    pub consistency: Vec<FieldInlineConsistencyClaim<F>>,
}
```

The concrete opening enums live with their components:

```text
registers:
  FieldRs1Value, FieldRs2Value, FieldRdValue, FieldRegistersVal,
  FieldRs1Ra, FieldRs2Ra, FieldRdWa, FieldRdInc

product:
  FieldProduct, FieldProductLhs, FieldProductRhs

conversion:
  XRegisterValue, FieldRegisterValue, EncodedFieldValue
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

### Conversion Rows

Field inline also needs explicit R1CS rows for non-field-register to
field-register movement. These rows are separate from field arithmetic: they
prove that the value loaded into the FR register file is the same value encoded
in ordinary Jolt data.

```text
x-register -> field-register:
  IsFieldLoadFromX * (FieldRdValue - decode_x_register(Rs1Value, width)) = 0

field-register -> x-register:
  IsFieldStoreToX * (RdValue - encode_field_register(FieldRs1Value, width)) = 0

immediate/constant -> field-register:
  IsFieldLoadImm * (FieldRdValue - decode_immediate(imm, width)) = 0
```

The `width` parameter is part of the field-inline configuration. BN254 uses a
254-bit representation; 128-bit field targets use the smaller conversion
encoding. The conversion helpers are generic over width wherever the row
formula does not depend on a concrete field size.

### Verifier And Proof Shape

Field-inline proof data is config-gated data in the normal Jolt proof model:

```text
selection.field_inline.enabled = true
  proof carries FR memory and field-product payloads
  stages 3/4/5 batch FR memory claims with existing stage work
  Spartan/product checks include field-product relations
  field conversion rows are active for x-register/FR movement

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
continues to consume the ordinary typed stage output.

The wrapper sees field inline through:

```text
selected verifier stages 3/4/5
  -> field-inline claim formulas
  -> sumcheck verifier constraints
  -> field-product and guest R1CS constraints
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
  jolt-claims::protocols::dory_assist::packing R1CS helper
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
  Dory assist extends the selected schedule. Its public inputs must match the
  selected visibility mode for the base Jolt proof.

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
  -> stage-8 data constructs Dory assist public inputs
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

## Design Decisions

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

## Testing

### `jolt-claims`

- Field-inline claim formula tests.
- Field-product relation tests.
- Dory-assist operation-family formula tests.
- Dory-assist wiring, public-input, and prefix-packing tests.
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
  opening values, or public inputs causes wrapper verification failure.

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
10. Add `snarks/gnark` once the R1CS interface is stable.
