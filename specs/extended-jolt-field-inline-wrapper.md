# Spec: Field Inline, Dory Assist, And Wrapper Pipeline

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-19 |
| Status | draft |
| PR | TBD |

## Context

The recursion-paper architecture has three orthogonal goals that should stay
separate in the modular codebase:

1. **Field inline** is a Jolt VM/protocol extension. It adds native field
   instructions, an FR register-file memory-checking instance, and local R1CS
   constraints for field operations.
2. **Dory assist** is the auxiliary proof that certifies the expensive Dory
   verifier work. The verifier checks Jolt's information-theoretic stages
   directly and checks the assist proof instead of doing ordinary Dory
   stage-8 verification.
3. **Spartan + HyperKZG wrapper** is a generic SNARK wrapper for verifier
   computations. It should consume a typed verifier plan rather than depend on
   field inline or Dory assist as hard-coded concepts.

These goals compose, but they should not collapse into one protocol namespace.
The composition point is `jolt-verifier`, which decides which verifier
components are enabled:

```text
base Jolt information-theoretic verifier
  + optional field-inline Jolt protocol checks
  + optional Dory-assist verifier
  -> verifier plan
  -> optional wrapper R1CS
  -> Spartan + HyperKZG
```

Field inline is new proof machinery for a new kind of field-register memory
checking. Dory assist is another SNARK we compose with Jolt. The wrapper should
be generic over the verifier computation it is asked to prove.

## Goals

- Put field-inline claim/opening changes in `jolt-claims`.
- Put field-inline R1CS constraints in `jolt-r1cs`.
- Add a concrete `dory_assist` protocol module in `jolt-claims`.
- Compose enabled protocol components in `jolt-verifier`.
- Create a generic `jolt-wrapper` crate with `snarks/spartan_hyperkzg` first
  and room for `snarks/gnark` later.
- Keep the design additive and concrete. Do not build a broad plugin framework
  before the critical path exists.

## Non-Goals

- Do not move witness generation, sparse trace layout, kernel scheduling, or
  commitment streaming into `jolt-claims`.
- Do not design the final `jolt-prover` replacement in this spec.
- Do not over-design a universal expression IR before the concrete field-inline
  and Dory-assist paths require it.
- Do not replace existing SDK inline APIs.

## High-Level Crate Split

```text
jolt-claims
  protocol facts and concrete claim/opening formulas

jolt-r1cs
  R1CS builder, matrices, guest constraints, and common lowering helpers

jolt-verifier
  executable verifier and composition gate for enabled protocol components

jolt-wrapper
  generic R1CS wrapper for typed verifier plans

jolt-sumcheck / jolt-blindfold / jolt-openings / jolt-hyperkzg
  reusable generic machinery, including protocol-owned R1CS encodings
```

The important separation:

```text
jolt-claims says what must be checked.
jolt-verifier executes those checks.
jolt-wrapper proves the selected verifier plan was executed.
jolt-prover, later, computes proofs efficiently.
```

## `jolt-claims` Layout

Target:

```text
crates/jolt-claims/src/
  protocols/
    jolt/
      ...
    field_inline/
      mod.rs
      config.rs
      ids.rs
      stages.rs
      openings.rs
      stage3.rs
      stage4.rs
      stage5.rs
      r1cs.rs
    dory_assist/
      mod.rs
      config.rs
      proof_shape.rs
      verifier_plan.rs
      ids.rs
      stages.rs
      openings.rs
      transcript.rs
      packing.rs
      ops/
        mod.rs
        g1.rs
        g2.rs
        gt.rs
        pairing.rs
      pcs/
        mod.rs
        hyrax.rs
```

In this `jolt-claims` layout, `protocols/dory_assist/pcs/hyrax.rs` is not the
R1CS implementation. It owns the semantic Hyrax plan for Dory assist: which
commitments, points, claimed evaluations, transcript messages, and public
boundary values define the Hyrax openings that must be checked.

### `protocols/field_inline`

This module owns field-inline protocol facts. It should use concrete helpers
rather than a broad framework.

```rust
pub struct FieldInlineConfig {
    pub enabled: bool,
    pub field_register_log_k: usize, // 4 for 16 slots
}

pub enum FieldInlineRelation {
    FieldRegistersClaimReduction,
    FieldRegistersReadWrite,
    FieldRegistersValEvaluation,
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
}
```

Concrete claim helpers:

```rust
pub fn stage3_claim_reduction_input<F>(...) -> F;
pub fn stage3_claim_reduction_output<F>(...) -> F;

pub fn stage4_read_write_input<F>(...) -> F;
pub fn stage4_read_write_output<F>(...) -> F;

pub fn stage5_val_evaluation_input<F>(...) -> F;
pub fn stage5_val_evaluation_output<F>(...) -> F;

pub fn required_openings(config: FieldInlineConfig) -> Vec<FieldInlineOpening>;
```

These helpers are used by:

- `jolt-verifier` when verifying FR-active Jolt proofs;
- BlindFold/R1CS lowering;
- wrapper R1CS assembly through the field-inline and claim/R1CS helpers when
  the proof being verified has field inline enabled.

### `protocols/dory_assist`

This module owns the recursion-paper auxiliary proof protocol. It describes the
Dory-verifier SNARK, not the base Jolt VM.

Protocol pieces:

```text
operation families:
  GT multiplication / exponentiation
  G1 addition / scalar multiplication
  G2 addition / scalar multiplication
  Miller loop / final exponentiation
  full pairing checks

composition machinery:
  family packing
  prefix packing
  sumcheck-based wiring
  Hyrax commitment/opening over Grumpkin
  Dory proof artifact as public input

stage structure:
  auxiliary sumcheck stages
  claim edges in the Dory-verifier computation DAG
  Hyrax opening claims
```

Initial concrete surface:

```rust
pub struct DoryAssistConfig {
    pub enabled: bool,
    pub pack_bits: usize,
}

pub struct DoryAssistVerifierPlan<F> {
    pub transcript: DoryAssistTranscriptPlan,
    pub sumchecks: Vec<DoryAssistSumcheck<F>>,
    pub wiring_checks: Vec<DoryAssistWiringCheck<F>>,
    pub hyrax_opening: HyraxOpeningPlan<F>,
    pub public_inputs: DoryAssistPublicInputPlan,
}

pub struct DoryAssistPublicInputPlan {
    pub includes_jolt_evaluation_claims: bool,
    pub includes_dory_proof: bool,
}

pub struct DoryAssistPublicInputs<F> {
    pub jolt_evaluation_claims: JoltEvaluationClaims<F>,
    pub dory_proof: DoryProofPublicInput<F>,
}

pub fn verifier_plan<F>(
    config: DoryAssistConfig,
) -> DoryAssistVerifierPlan<F>;
```

This plan should be concrete and reviewable. The target is the full Dory
verifier: `pi_assist` proves that the Dory verifier accepts the public Dory
proof artifact and the Jolt evaluation claims. There should not be a config
mode for leaving pairings or final exponentiation outside the Dory-assist proof.
Generalization can wait until the first verifier/wrapper path is working.

### Protocol Composition

Protocol composition is handled in `jolt-verifier`. `jolt-claims` should expose
the facts/formulas for each protocol component; it should not decide which
combination of components is active.

Use small config/proof types near the verifier, or in `protocols/jolt` only if
they need to be shared:

```rust
pub struct JoltProtocolConfig {
    pub field_inline: FieldInlineConfig,
}

pub struct DoryAssistedConfig {
    pub jolt: JoltProtocolConfig,
    pub dory_assist: DoryAssistConfig,
}
```

The composed verifier plan is just:

```rust
pub struct DoryAssistedVerifierPlan<F> {
    pub jolt_info_theoretic: JoltInfoTheoreticVerifierPlan<F>,
    pub dory_assist: DoryAssistVerifierPlan<F>,
}
```

This should live in `jolt-verifier`. `jolt-wrapper` consumes the resulting plan
and should not independently decide which protocol components are enabled.

## `jolt-r1cs` Layout

Target:

```text
crates/jolt-r1cs/src/
  constraints/
    mod.rs
    rv64.rs
    field_inline.rs
```

`field_inline.rs` owns extra VM R1CS rows and variables for the native field
register path:

```text
FADD:       IsFieldAdd      * (FieldRs1Value + FieldRs2Value - FieldRdValue) = 0
FSUB:       IsFieldSub      * (FieldRs1Value - FieldRs2Value - FieldRdValue) = 0
FMUL:       field product relation
FINV:       inverse relation
ASSERT_EQ:  IsFieldAssertEq * (FieldRs1Value - FieldRs2Value) = 0
MOV/SLL:    bridge rows from XReg Rs1Value to FieldRdValue
```

Dory assist does not modify the Jolt guest trace R1CS at all, and field inline
can be added by appending the `field_inline.rs` rows when the selected Jolt
protocol config enables it.

```rust
pub fn append_field_inline_constraints<F: Field>(
    builder: &mut R1csBuilder<F>,
    config: FieldInlineR1csConfig,
);
```

This is for the **Jolt guest trace relation**. It is separate from the wrapper
R1CS, even though the Spartan + HyperKZG wrapper also ultimately proves an
R1CS. The wrapper R1CS is produced from verifier computation: transcript
events, claim equations, sumcheck verifier checks, opening consistency checks,
and PCS verifier checks. Those encodings should be owned by the modular crates
that own the corresponding protocol semantics.

The desired layering is:

```text
jolt-r1cs
  provides the R1CS builder, matrices, common lowering machinery, and guest
  trace relations, including optional field-inline rows

jolt-sumcheck::r1cs
  encodes generic sumcheck verifier checks

jolt-claims + jolt-r1cs
  encode claim equations and protocol formulas

jolt-transcript / transcript R1CS helpers
  encode transcript absorbs and challenge derivation

jolt-wrapper
  assembles the selected verifier plan into R1CS by calling those component
  encoders

jolt-wrapper::snarks::spartan_hyperkzg
  accepts an arbitrary R1CS instance and proves it
```

The choice to include field-inline guest rows is a protocol-composition
decision made by the verifier/prover config, not by a new RV64
constraint-system variant.

Open design question for FMUL/FINV:

```text
Option A:
  reuse Product = LeftInstructionInput * RightInstructionInput,
  but only if the witness path is actually FR-aware.

Option B:
  add a distinct FieldProduct relation for field-inline multiplication.
```

The spec preference is explicitness over accidental reuse. If full-width Fr
values are not truly routed through the normal product-virtual columns, add a
distinct field product relation.

## `jolt-verifier` Layout

Target additions:

```text
crates/jolt-verifier/src/
  config.rs
  field_inline/
    mod.rs
    stage3.rs
    stage4.rs
    stage5.rs
    openings.rs
  dory_assist/
    mod.rs
    proof.rs
    transcript.rs
    sumchecks.rs
    wiring.rs
    hyrax.rs
    pairing.rs
  dory_assisted.rs
```

### Ordinary Jolt Verifier

The existing `jolt-verifier` path verifies ordinary Jolt proofs. It should be
factored so the information-theoretic stages can be called without directly
performing ordinary Dory stage-8 verification.

Needed boundary:

```rust
pub fn verify_jolt_info_theoretic<F, PCS, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS>,
    proof: &JoltProof<PCS>,
    config: JoltProtocolConfig,
    transcript: &mut T,
) -> Result<JoltEvaluationClaims<F>, VerifierError>;
```

`JoltProtocolConfig` gates field-inline changes in the underlying Jolt proof.

### Dory-Assist Verifier

This verifies `pi_assist`, using the stage-7/evaluation claims from the Jolt
information-theoretic verifier and the Dory proof artifact as public input.

```rust
pub fn verify_dory_assist<F, T>(
    proof: &DoryAssistProof<F>,
    public_inputs: DoryAssistPublicInputs<F>,
    config: DoryAssistConfig,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = F>;
```

This includes:

- Dory-assist sumcheck verification;
- copy/wiring checks;
- packing checks;
- Hyrax opening verification;
- full Dory verifier checks, including pairing/final-exponentiation work,
  proved inside `pi_assist`.

### Dory-Assisted Jolt Verifier

This is a thin composition in `jolt-verifier`, not a separate `jolt-claims`
protocol:

```rust
pub struct DoryAssistedProof<PCS> {
    pub jolt: JoltProof<PCS>,
    pub assist: DoryAssistProof<PCS::Field>,
}

pub fn verify_dory_assisted_jolt<F, PCS, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS>,
    proof: &DoryAssistedProof<PCS>,
    config: DoryAssistedConfig,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    T: Transcript<Challenge = F>;
```

Flow:

```text
1. Bind DoryAssistedConfig into the transcript.
2. Verify Jolt information-theoretic stages.
3. Extract stage-7/evaluation claims and the public Dory proof artifact.
4. Verify that `pi_assist` proves the full Dory verifier accepts that public
   input.
5. Accept without running ordinary Dory stage-8 verification.
```

If `config.jolt.field_inline.enabled`, the Jolt information-theoretic verifier
includes the field-inline stage/opening additions.

## `jolt-wrapper` Layout

New crate:

```text
crates/jolt-wrapper/
  Cargo.toml
  src/
    lib.rs
    config.rs
    proof.rs
    public_inputs.rs
    verifier_plan.rs
    r1cs/
      mod.rs
      assembly.rs
      instance.rs
      public_inputs.rs
      witness.rs
      pcs/
        mod.rs
        hyrax.rs
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

The wrapper consumes a typed verifier plan. The first concrete plan should be
the Jolt information-theoretic verifier plus Dory assist, but the wrapper
boundary should stay generic:

```rust
pub struct WrapperConfig {
    pub backend: WrapperBackend,
}

pub struct WrappedVerifierPlan<F> {
    pub transcript: TranscriptPlan<F>,
    pub public_inputs: PublicInputPlan,
    pub witness_inputs: WitnessInputPlan,
    pub checks: Vec<VerifierCheck<F>>,
}

pub struct WrapperWitnessInputs<F> {
    pub plan: WrappedVerifierPlan<F>,
    pub private_inputs: WrapperWitness<F>,
    pub public_inputs: WrapperPublicInputs<F>,
}

pub struct WrapperR1csInstance<F> {
    pub matrices: ConstraintMatrices<F>,
    pub witness: Vec<F>,
    pub public_inputs: Vec<F>,
}

pub enum WrapperBackend {
    SpartanHyperKzg,
    Gnark,
}

pub fn build_dory_assisted_jolt_plan<F>(
    config: DoryAssistedConfig,
) -> Result<WrappedVerifierPlan<F>, WrapperError>;

pub fn assemble_wrapper_r1cs<F>(
    inputs: WrapperWitnessInputs<F>,
) -> Result<WrapperR1csInstance<F>, WrapperError>;
```

The proof-system-specific backend only sees an R1CS instance:

```rust
pub struct WrapperProverConfig {
    pub backend: WrapperBackend,
}
```

The first backend:

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

The first wrapper R1CS instantiation should include:

- transcript/challenge reconstruction or public challenge binding;
- whichever Jolt information-theoretic checks `jolt-verifier` enabled;
- whichever Dory-assist verifier checks `jolt-verifier` enabled;
- generic sumcheck constraints from `jolt-sumcheck::r1cs`;
- claim equations from `jolt-claims` lowered through `jolt-r1cs`;
- opening consistency constraints from the openings/verifier plan;
- Hyrax opening verification for `pi_assist` in the Dory-assist case;
- full Dory verifier checks represented by the Dory-assist plan, including
  pairing/final-exponentiation work inside `pi_assist`.

This matches the paper statement that wrapping encodes:

```text
information-theoretic checks
+ additional Dory-assist sumcheck stages
+ Hyrax opening verification
```

The outer Spartan + HyperKZG proof then has its own verifier and PCS checks.
Those are part of the wrapper SNARK, not part of the verifier being wrapped.

### R1CS Encoding Ownership

`jolt-wrapper` should not become the owner of all generic lowering logic. The
component crates should own their own verifier-to-R1CS encodings where that
encoding is generic or reusable:

```text
TranscriptPlan        -> transcript crate / transcript R1CS helpers
SumcheckPlan          -> jolt-sumcheck::r1cs
ClaimEquationPlan     -> jolt-claims formulas + jolt-r1cs lowering
OpeningConsistency    -> jolt-openings/verifier plan + jolt-r1cs constraints
Guest R1CS boundary   -> jolt-r1cs constraints
PcsVerifierPlan       -> PCS-specific verifier constraints
```

This keeps field inline black-boxed from the wrapper. If field inline is
enabled, `jolt-verifier` emits more claim equations, openings, sumchecks, and
guest-R1CS boundary checks. The wrapper assembles the resulting R1CS instance
by invoking the component encoders; it should not reimplement their protocol
semantics.

PCS is the least generic part initially. For the Dory-assist target, the first
concrete PCS verifier encoding should be Hyrax:

```text
dory_assist verifier plan
  -> sumcheck / packing / wiring checks
  -> Hyrax opening claims
  -> Hyrax verifier constraints
  -> wrapper R1CS
```

Do not design a universal PCS-in-circuit interface before the Hyrax path is
implemented. The first useful abstraction is a narrow one: enough for the
Dory-assist verifier plan to say which Hyrax openings must be checked, and
enough for a concrete Hyrax R1CS module to constrain those openings.

Ownership split:

```text
jolt-claims::protocols::dory_assist::pcs::hyrax
  semantic Hyrax opening plans:
    commitments, claimed evals, points, transcript messages, public boundary

jolt-verifier::dory_assist::hyrax
  executable native Hyrax verifier for pi_assist

jolt-wrapper::r1cs::pcs::hyrax
  bespoke Hyrax verifier constraints:
    allocate commitment/opening witness variables
    constrain transcript challenge derivation or public challenge binding
    constrain Hyrax inner-product/MSM/opening equations
    emit public input bindings expected by the wrapper instance
```

So the quoted `jolt-claims` `hyrax.rs` is the protocol/plan file. The bespoke
logic to put Hyrax verification into R1CS starts in
`jolt-wrapper::r1cs::pcs::hyrax`, using the opening plan emitted by
`jolt-claims` and the executable verifier shape in `jolt-verifier`. If another
crate needs the same Hyrax verifier constraints, move the R1CS module out then.

## End-To-End Flows

### Ordinary Jolt With Optional Field Inline

```text
guest uses SDK inline or normal RV64 code
  -> tracer emits cycles, plus FieldRegEvent sidecar if FR-active
  -> ordinary Jolt proof is generated
  -> ordinary Jolt verifier checks information-theoretic stages and Dory stage 8
```

If field inline is enabled, the Jolt protocol includes:

```text
extra field-register Twist relations
extra R1CS rows
extra openings
extra stage 3/4/5 claim formulas
```

### Dory-Assisted Jolt

```text
ordinary Jolt prover produces pi_jolt
  -> Dory-assist prover proves Dory verifier accepts pi_dory
  -> output pi_assisted = (pi_jolt, pi_assist)
```

Verifier:

```text
verify pi_jolt stages 1-7 / information-theoretic claims
  -> verify pi_assist proves the full Dory verifier accepts the public Dory proof
  -> accept without ordinary Dory stage-8 verification
```

If the Jolt proof is FR-active, `config.jolt.field_inline` changes the Jolt
information-theoretic verifier. Dory assist remains the auxiliary protocol for
replacing Dory stage-8 verification.

### Simple Recursion

```text
compile Dory-assisted verifier to RV64 guest
  -> run inside Jolt
  -> use inlines to accelerate verifier hotspots
```

This is where field inline or other native inlines can matter directly. The
paper specifically identifies native-field multiplication/inlines as useful for
making Jolt self-recursion faster when the verifier is executed as a guest.

### Wrapper

```text
typed verifier plan
  first target: Jolt information-theoretic verifier + Dory assist
  -> wrapper R1CS
  -> Spartan prover
  -> HyperKZG PCS
  -> ~KB wrapper proof
```

The wrapper proves the verifier, not the prover.

## Open Design Questions

### Field Inline Product Relation

For field-inline FMUL/FINV, decide whether to reuse existing product
virtualization or add a distinct field product relation.

Concern:

```text
The current branch's intended rows use:

  Product = LeftInstructionInput * RightInstructionInput
  LeftInstructionInput  = FieldRs1Value
  RightInstructionInput = FieldRs2Value
  Product               = FieldRdValue

but the product witness source I inspected appeared RV64-shaped.
```

If there is not a true FR-aware product witness path, add `FieldProduct`.

### Pure FieldOp Normal Register Accesses

Decide whether pure `FieldOp` cycles should suppress normal x-register
read/write accesses. Protocol preference:

```text
Pure FR arithmetic:
  FR metadata/events + FR Twist + FR R1CS

Bridge ops:
  normal register Twist + FR Twist + bridge R1CS
```

### Wrapper Challenge Binding

The wrapper must either:

- reconstruct Fiat-Shamir challenges inside R1CS; or
- expose challenges as public inputs and bind them to an external transcript
  computation.

For a self-contained wrapper, transcript construction should be constrained
inside the wrapper R1CS.

## Testing Plan

### `jolt-claims`

- Field-inline claim formula tests.
- Dory-assist stage/claim DAG shape tests.
- Dory-assist public-boundary tests bind the Jolt evaluation claims and public
  Dory proof artifact.

### `jolt-r1cs`

- Base RV64 constraints unchanged when field inline disabled.
- Field-inline row helpers append the expected FR rows when enabled.
- Dory assist does not change RV64/field-inline guest trace constraints.
- FMUL/FINV tests after product relation is decided.

### `jolt-verifier`

- Ordinary Jolt proofs verify on the ordinary verifier path.
- Dory-assisted proofs verify on `verify_dory_assisted_jolt`.
- Verification rejects when `pi_assist` is missing or mismatched.
- Config/transcript toggles are rejection-tested.
- Field-inline base proof fixtures verify only when
  `config.jolt.field_inline.enabled`.

### `jolt-wrapper`

- Assemble wrapper R1CS from a typed verifier plan by calling component-owned
  R1CS encoders.
- Synthetic Dory-assist verifier-plan proof passes Spartan + HyperKZG.
- Mutating sumcheck claims, Hyrax boundary values, transcript challenges, or
  config changes causes wrapper verification failure.
- Wrapper shape changes deterministically with the selected verifier plan.

## Milestones

1. Add `jolt-claims::protocols::field_inline` for FR Twist/opening/R1CS
   protocol facts.
2. Add `jolt-r1cs::constraints::field_inline` append helpers.
3. Add `jolt-claims::protocols::dory_assist` with the Dory-assist verifier
   plan shape.
4. Refactor `jolt-verifier` to expose a stage-1-through-7 information-theoretic
   verifier boundary.
5. Implement `jolt-verifier::dory_assist`.
6. Implement `jolt-verifier` protocol composition for ordinary, FR-active, and
   Jolt-plus-Dory-assist verifier plans.
7. Create `jolt-wrapper` with `snarks/spartan_hyperkzg`.
8. Assemble typed verifier plans into wrapper R1CS, with the Dory-assisted plan
   as the first target and component crates owning reusable encodings.
9. Add `snarks/gnark` only after the Spartan + HyperKZG path has stable
   boundaries.

## Review Checklist

- Is `dory_assist` concrete enough to implement without a broad IR?
- Is all protocol enablement/composition handled in `jolt-verifier`?
- Is field inline treated as an optional Jolt protocol extension?
- Is the wrapper clearly proving the verifier rather than the prover?
- Is the wrapper generic over typed verifier plans rather than hard-coded to
  field inline or Dory assist?
- Are generic R1CS encodings owned by the crates that own the protocol
  semantics, rather than centralized in `jolt-wrapper`?
- Does the wrapper include the Hyrax opening verification from the
  Dory-assist proof?
- Does Dory assist prove the full Dory verifier, with no external
  pairing/final-exponentiation boundary?
- Are Spartan + HyperKZG and later gnark separated cleanly under
  `jolt-wrapper/snarks/`?
