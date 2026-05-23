# Spec: Wrapper Protocol And SNARK Backend

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

The wrapper proves an R1CS encoding of the configured Jolt verifier computation.
The first backend is zero-knowledge Spartan + HyperKZG. The backend consumes
arbitrary R1CS; Jolt-specific protocol assembly happens before backend proving.

This spec owns the wrapper protocol axis:

```text
configured verifier computation
  -> R1CS assembly
  -> ZK Spartan + HyperKZG proof
```

Field inline and Dory assist add checks to the configured linear verifier flow.
Their protocol details are specified in
[field-inline-protocol.md](field-inline-protocol.md) and
[dory-assist-protocol.md](dory-assist-protocol.md). Proof-shape validation,
compile-time verifier config, and wrapper entry-point rules are specified in
[selected-verifier-integration.md](selected-verifier-integration.md).

## Scope

V1 scope:

```text
generic configured-verifier R1CS builder
transparent Jolt proof as private wrapper witness
in-circuit transcript replay
variable-challenge sumcheck R1CS
claim lowering over constants and variables
Hyrax R1CS integration when the configured verifier includes Dory assist
zero-knowledge Spartan + HyperKZG backend over arbitrary R1CS
```

Out of scope:

```text
making wrapper depend on BlindFold internals
wrapping the BlindFold verifier as the primary v1 ZK composition
folding / relaxed R1CS / row hiding
one universal circuit shape for every configured verifier mode
full gnark adapter
deleting the existing Groth16 transpiler path
```

## Design Intuition

The wrapper is a compiler from verifier computation to R1CS plus a SNARK
backend:

```text
configured verifier computation
  -> protocol-R1CS builder
  -> transcript constraints
  -> sumcheck verifier constraints
  -> claim equation constraints
  -> opening/PCS verifier constraints
  -> R1CS instance and witness
  -> ZK Spartan + HyperKZG proof
```

The backend must remain boring and zero-knowledge:

```text
input:
  ConstraintMatrices<F>
  witness Vec<F>
  public inputs Vec<F>

output:
  wrapper proof
```

It should not know about Jolt stages, field inline, Dory assist, BlindFold, or
transcript ordering. It must hide the R1CS witness, which includes the inner
transparent Jolt proof, clear opening claims, verifier intermediate values, and
any auxiliary variables allocated during wrapper assembly.

## Statement

For a fixed configured verifier, the wrapper proves verifier acceptance:

```text
Given public wrapper inputs, there exists a private transparent Jolt proof
witness and verifier auxiliary witness such that the configured Jolt verifier
accepts when its Fiat-Shamir transcript is replayed inside R1CS.
```

The public wrapper inputs include the public Jolt instance data and deployment
bindings:

```text
public IO
preprocessing / verifier-key digest
configured protocol identifier
selected verifier configuration digest
any deployment-specific public inputs required by the wrapper verifier
```

The private wrapper witness includes:

```text
transparent Jolt proof payload
clear opening claims and proof commitments
stage-local verifier intermediate values
opening snapshot data
optional selected PCS-assist proof payload
auxiliary variables needed by transcript, sumcheck, claim, opening, and assist
constraints
```

The R1CS acceptance relation is the configured linear verifier:

```text
1. Bind the public preamble and private proof payload into the Jolt transcript.
2. Derive verifier challenges inside R1CS.
3. Verify the transparent Jolt PIOP stages:
     sumcheck verifier equations
     stage input/output claim equations
     public-input and consistency equations
     final opening snapshot construction
4. Verify the configured opening phase:
     ordinary mode: ordinary PCS opening verifier path
     Dory-assist mode: selected DoryAssistProof verifier path
5. Enforce that the configured verifier accepts.
```

For the primary v1 wrapper target, the in-circuit transcript is the algebraic
Poseidon Jolt transcript. The wrapped inner proof must use the same transcript
backend, so the wrapper constrains Poseidon absorbs and challenges directly
over field elements rather than proving a Blake2b/Keccak byte hash circuit.
This remains true when the configured verifier includes Dory assist: the
wrapper transcript is Poseidon over BN254 `Fr`, while Dory-assist sumcheck
checks are `Fq` arithmetic represented non-natively inside the `Fr` R1CS.

In Dory-assist mode, step 4 does not encode the ordinary Dory stage-8 verifier.
It encodes the selected `PcsProofAssist` verifier for the same opening snapshot:

```text
opening snapshot
  -> Dory-assist stage 1/2/3 verifier checks
  -> prefix-packing checks
  -> Hyrax dense-trace opening check
  -> Dory-assist final public pairing check / bound acceptance condition
```

This statement is intentionally about configured verifier acceptance. The VM
execution semantics enter through the soundness of the configured Jolt verifier
and the selected PCS-assist verifier, not through a separate execution circuit
inside the wrapper.

## Relationship To BlindFold

BlindFold already establishes the useful R1CS assembly pattern:

```text
build a protocol statement from verifier-stage semantics
allocate verifier-equation witness variables
lower claim expressions
append generic sumcheck constraints
connect each stage's output claim to the next protocol component
```

The wrapper reuses this pattern, not the BlindFold protocol. The primary v1
wrapped-ZK composition is:

```text
transparent Jolt proof
  -> private wrapper witness
  -> wrapper R1CS proves the transparent configured verifier accepts
  -> ZK Spartan + HyperKZG hides the verifier witness
```

In this composition, the wrapper R1CS does not prove the BlindFold verifier.
Base-layer BlindFold remains the standalone Jolt ZK path. If a future product
needs to wrap an already-BlindFolded Jolt proof, then the configured verifier
R1CS can include BlindFold verification as an optional composition, but that is
not the primary v1 wrapper target.

```text
BlindFold builder:
  configured stage semantics
  + committed sumcheck consistency
  + challenges baked as constants
  -> verifier-equation R1CS for the BlindFold proof

Wrapper builder:
  configured verifier computation
  + transparent Jolt proof as private witness data
  + transcript absorbs inside R1CS
  + challenge variables derived inside R1CS
  -> full verifier R1CS for ZK Spartan + HyperKZG
```

The wrapper does not need:

```text
Nova folding
relaxed R1CS
random satisfying instance
BlindFold witness row hiding
BlindFold-specific committed output claim rows
in-circuit BlindFold verification for the primary wrapped-ZK path
```

The wrapper does need:

```text
in-circuit Fiat-Shamir
challenge variables
public-input ordering
proof-data allocation
component R1CS helpers
```

## `jolt-wrapper` Layout

Target layout:

```text
crates/jolt-wrapper/
  Cargo.toml
  src/
    lib.rs
    builder.rs
    error.rs
    proof.rs
    protocol.rs
    r1cs.rs
    statements.rs
    verify.rs
    witness.rs
    snark_backends/
      mod.rs
      spartan_hyperkzg/
        mod.rs
        proof.rs
        prover.rs
        setup.rs
        verifier.rs
      gnark/
        mod.rs
```

The crate mirrors the useful `jolt-blindfold` top-level organization:
`statements` describes the configured verifier object, `builder` assembles typed
inputs and transcript state, `protocol` owns the built R1CS instance and layout,
`r1cs` owns constraint appending, and `snark_backends` owns R1CS-to-proof
orchestration. Reusable component encodings live in the crates that own those
protocols.

## Protocol Builder

`jolt-wrapper::r1cs` owns a generic protocol builder for the configured verifier
computation:

```rust
pub struct WrapperProtocolBuilder<F, Tr> {
    pub builder: jolt_r1cs::R1csBuilder<F>,
    pub transcript: Tr,
    pub sources: WrapperClaimSources<F>,
}
```

The first implementation can use `R1csBuilder` directly. A separate assembly
object is not required to encode verifier equations; it is only useful once the
wrapper needs stable public-input ordering and a single export point for
matrices, witness, and public inputs. `WrapperProtocolBuilder` adds
wrapper-specific transcript state, verifier source maps, and proof artifact
allocation.

Helper shape:

```rust
impl<F, Tr> WrapperProtocolBuilder<F, Tr> {
    pub fn alloc_public(
        &mut self,
        id: WrapperPublicInputId,
        value: F,
    ) -> Variable;

    pub fn alloc_witness(
        &mut self,
        id: WrapperWitnessId,
        value: F,
    ) -> Variable;

    pub fn absorb_public(&mut self, variable: Variable);
    pub fn absorb_witness(&mut self, variable: Variable);

    pub fn challenge(&mut self, id: WrapperChallengeId) -> Variable;
}
```

Each configured verifier stage lowers through an R1CS hook:

```rust
pub trait WrapperR1csStage<F> {
    type Error;

    fn append_r1cs<Tr>(
        &self,
        builder: &mut WrapperProtocolBuilder<F, Tr>,
    ) -> Result<(), Self::Error>;
}
```

The stage hook mirrors native verifier execution:

```text
allocate exactly the proof data the native verifier reads
absorb the same proof/public data in the same order
derive challenge variables
call component R1CS helpers
register openings and claims needed by later stages
```

## `jolt-r1cs` Substrate

`jolt-r1cs` provides generic construction machinery. It should not know Jolt
stage semantics. Component encodings should accept `&mut R1csBuilder<F>` unless
they specifically need public-input registration or final instance export.

Possible additions:

```text
crates/jolt-r1cs/src/
  builder.rs
  constraint.rs
  lowering.rs
  matrices.rs
```

Responsibilities:

```text
lowering.rs:
  SourceValue, ClaimSources, lower_claim_expr, assert_claim_expr_eq

constraints/:
  execution-relation constraints such as field-inline field_constraints
```

When the wrapper reaches backend handoff, either `jolt-wrapper` or `jolt-r1cs`
can add a small assembly/export helper:

```rust
pub struct WrapperR1csAssembly<F> {
    pub builder: R1csBuilder<F>,
    pub public_inputs: Vec<PublicInputBinding>,
}

pub struct PublicInputBinding {
    pub id: PublicInputId,
    pub variable: Variable,
}
```

Stable public-input ordering is part of the wrapper statement. It must be
deterministic across prover and verifier.

## Claim Lowering

`jolt-r1cs::lowering` should remain the single claim-lowering API. It needs to
generalize source values instead of adding a parallel symbolic-lowering module.

Current lowering treats openings as variables and challenges/publics as
constants. Wrapper lowering needs challenges and some public values to be R1CS
variables or linear combinations.

Adjusted shape:

```rust
pub enum SourceValue<F> {
    Constant(F),
    LinearCombination(LinearCombination<F>),
}

pub trait ClaimSources<F> {
    type Opening;
    type Challenge;
    type Public;

    fn opening(&mut self, id: &Self::Opening)
        -> Result<SourceValue<F>, ClaimLoweringError>;
    fn challenge(&mut self, id: &Self::Challenge)
        -> Result<SourceValue<F>, ClaimLoweringError>;
    fn public(&mut self, id: &Self::Public)
        -> Result<SourceValue<F>, ClaimLoweringError>;
}
```

`lower_claim_expr` keeps a constant fast path:

```text
only constants:
  produce a constant linear combination

one non-constant factor:
  stay linear

multiple non-constant factors:
  allocate multiplication constraints through R1csBuilder::multiply
```

This gives both modes through one API:

```text
BlindFold:
  ChallengeId -> SourceValue::Constant(F)
  PublicId    -> SourceValue::Constant(F)

Wrapper:
  ChallengeId -> SourceValue::LinearCombination(challenge variable)
  PublicId    -> SourceValue::LinearCombination(public input variable)
```

The existing `ClaimSourceTable` remains useful as the constant-source
implementation. The wrapper provides `WrapperClaimSources`, backed by
transcript-derived challenges, public inputs, opening variables, and stage-local
aliases.

## Sumcheck R1CS

`jolt-sumcheck::r1cs` owns the generic sumcheck verifier equations. Its round
API uses a single challenge abstraction for both baked and in-circuit
challenges:

```text
round.challenge() -> LinearCombination<F>
```

BlindFold and other baked-verifier paths return `LinearCombination::constant(r)`.
The wrapper returns a variable-backed linear combination for the transcript
challenge. This keeps one sumcheck R1CS path rather than a constant path plus a
wrapper-only path.

For each round:

```text
round-sum check:
  sum over verifier domain equals claim_in
  this remains a linear equality

evaluation-at-challenge check:
  s_i(r_i) = claim_out
  this uses variable challenge r_i
```

Use Horner constraints for the evaluation:

```text
eval = c_d
eval = eval * r + c_{d-1}
...
assert eval == claim_out
```

The wrapper obtains `r` from `jolt-transcript::r1cs`, so the same challenge
variable is used by transcript binding, claim expressions, and sumcheck
verifier equations.

## Transcript R1CS

`jolt-transcript` needs an `r1cs` module for in-circuit Fiat-Shamir:

```text
crates/jolt-transcript/src/r1cs/
  mod.rs
  sponge.rs
  poseidon.rs
  absorb.rs
  challenge.rs
```

API shape:

```rust
pub struct R1csScalar<F> {
    pub value: F,
    pub lc: LinearCombination<F>,
}

pub trait R1csTranscript<F> {
    fn new(builder: &mut R1csBuilder<F>, label: &'static [u8]) -> Self;
    fn challenge_scalar(&mut self, builder: &mut R1csBuilder<F>) -> R1csScalar<F>;
}

pub trait R1csAlgebraicTranscript<F>: R1csTranscript<F> {
    fn absorb_scalar(&mut self, builder: &mut R1csBuilder<F>, value: R1csScalar<F>);
    fn absorb_constant_scalar(&mut self, builder: &mut R1csBuilder<F>, value: F);
    fn absorb_u64(&mut self, builder: &mut R1csBuilder<F>, value: u64);
    fn absorb_label(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8]);
    fn absorb_label_with_len(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8], len: u64);
}

pub trait R1csJoltTranscript<F>: R1csAlgebraicTranscript<F> {
    fn append_label(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8]);
    fn append_u64(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8], value: u64);
    fn append_scalar(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8], value: R1csScalar<F>);
    fn append_scalars(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8], values: &[R1csScalar<F>]);
}

pub trait R1csByteTranscript<F>: R1csTranscript<F> {
    type Byte;

    fn absorb_bytes(&mut self, builder: &mut R1csBuilder<F>, bytes: &[Self::Byte]);
    fn absorb_constant_bytes(&mut self, builder: &mut R1csBuilder<F>, bytes: &'static [u8]);
}

pub trait R1csJoltByteTranscript<F>: R1csJoltTranscript<F> + R1csByteTranscript<F> {
    fn append_bytes(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8], bytes: &[Self::Byte]);
    fn append_constant_bytes(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8], bytes: &'static [u8]);
}
```

The base trait owns transcript initialization and scalar challenge production.
Absorption is capability-based: algebraic backends implement
`R1csAlgebraicTranscript`, while byte-oriented backends can later implement
`R1csByteTranscript`. The initial wrapper target is the algebraic Poseidon path,
so wrapper assembly should absorb field elements and domain-separation words
directly instead of routing through byte decomposition. Native verifier replay
and wrapper assembly must share the same transcript order and scalar encoding.

The Poseidon R1CS backend must match the Jolt proof transcript used by
`jolt-core` under `transcript-poseidon`:

```text
state_0 = Poseidon(label_word, 0, 0)
raw_absorb(word): state <- Poseidon(state, n_rounds, word); n_rounds += 1
challenge_scalar(): state <- Poseidon(state, n_rounds, 0); n_rounds += 1; return state
```

Scalar payloads are absorbed as field elements. `append_label(label)` absorbs a
32-byte zero-padded label word. `append_u64(label, x)` absorbs the label and
then a little-endian `u64` word. `append_scalars(label, values)` first absorbs
the Jolt packed label/count word, where the label occupies bytes `[0..24)` and
the count is big-endian in bytes `[24..32)`, then absorbs each scalar. This is
not the byte-oriented `Blake2b`/`Keccak` serialization path.

For Poseidon byte absorbs used by the Jolt preamble, each 32-byte chunk is
converted to one little-endian scalar. The first chunk uses the current
`n_rounds` tag; continuation chunks use round tag zero; the transcript round
increments once for the whole byte payload. Assigned byte values must be range
constrained by the component that allocated them.

## R1CS Ownership

Reusable R1CS encodings live with the crates that own the corresponding
protocol machinery:

```text
R1CS builder/matrices:
  jolt-r1cs::builder, jolt-r1cs::assembly, jolt-r1cs matrices

transcript hashing:
  jolt-transcript::r1cs

sumcheck verifier equations:
  jolt-sumcheck::r1cs, including variable-challenge wrapper checks

claim formulas:
  jolt-claims formulas lowered through jolt-r1cs::lowering

opening consistency:
  jolt-openings + jolt-r1cs helpers

field-inline field_constraints:
  jolt-r1cs::constraints::field_constraints

Future BlindFold verifier checks:
  jolt-blindfold, using the same constant-source claim/sumcheck R1CS helpers

Hyrax verifier checks:
  jolt-hyrax::r1cs

Dory-assist prefix packing:
  jolt-claims::protocols::dory_assist::packing R1CS helper
```

`jolt-wrapper::r1cs::assembly` sequences these helpers according to the
configured verifier flow exported by `jolt-verifier`.

## Dory Assist In Wrapper

A Dory-assisted wrapper does not encode the ordinary Dory stage-8 verifier path.
It encodes the configured verifier flow after the selected `PcsProofAssist`
implementation has replaced that ordinary PCS opening check. Concretely:

```text
base Jolt stages
  -> opening snapshot
  -> selected DoryAssistProof::verify(...)
  -> Dory-assist stage 1/2/3 checks
  -> Hyrax opening check for the packed dense trace
  -> Dory-assist final public pairing check
  -> wrapper R1CS proves this configured verifier accepts
```

The wrapper-facing R1CS hooks follow the Dory-assist ownership split:

```text
Dory-assist stage sumchecks:
  jolt-sumcheck::r1cs

Dory-assist operation-family and wiring formulas:
  jolt-claims::protocols::dory_assist + jolt-r1cs lowering

Dory-assist prefix packing:
  jolt-claims::protocols::dory_assist::packing

Dory-assist dense-trace opening:
  jolt-hyrax::r1cs

Dory-assist final public pairing check:
  dory-assist verifier crate hook
```

`jolt-hyrax::r1cs` should only lower the Hyrax opening verifier used by the
assist proof:

```text
row_point, col_point = split(opening_point)
combined_commitment = Σ_i eq(row_point, i) * row_commitment_i
combined_commitment == Pedersen(combined_row, row_combination_randomness)
claimed_eval == Σ_j eq(col_point, j) * combined_row[j]
```

It should not know Dory-assist stage IDs, operation families, prefix codes, or
copy constraints. Those are Dory-assist protocol semantics.

For the BN254/Grumpkin recursion target, Dory assist uses Grumpkin-backed
Pedersen row commitments. This is not only a wrapper convenience: Grumpkin's
scalar field is BN254 `Fq`, matching the arithmetic needed by the assisted Dory
verifier trace. The wrapper benefit is narrower but important: Grumpkin's base
field is BN254 `Fr`, so when the configured verifier is compiled into a BN254
`Fr` R1CS, the Hyrax commitment group-law constraints are native-coordinate
constraints rather than BN254-G1-over-`Fq` non-native group arithmetic.

Any `Fq` scalar arithmetic, range constraints, operation-family semantics, and
copy/wiring checks remain Dory-assist constraints. The Hyrax R1CS module only
binds the already-packed dense trace to the assist verifier's claimed dense
opening.

Dory-assist `Fq` sumcheck challenges must be derived from the wrapper's
Poseidon-over-`Fr` transcript by an explicit, domain-separated
`Fr`-to-`Fq` challenge map. The simplest implementation is to represent the
resulting `Fq` challenge with the same non-native `Fq` variable type used by
Dory-assist verifier arithmetic. We should avoid a hidden Fq-Poseidon transcript:
if stronger challenge uniformity is needed than direct `Fr` injection, derive
two or more Poseidon-`Fr` words and reduce the combined integer modulo `Fq` with
an explicit conversion gadget.

The Dory-assist spec says final exponentiation and public pairing equality are
native verifier work in v1. For wrapper purposes, that means "outside the
Dory-assist auxiliary proof," not "outside the configured verifier." A
self-contained wrapper must still account for that selected assist-verifier
acceptance condition: either the Dory-assist verifier crate supplies the R1CS
hook for the final public check, or the wrapper verifier must recheck and bind
that public condition outside the wrapper proof. The primary self-contained
wrapper target should not rely on an unchecked native side condition.

## Assembly API

```rust
pub struct WrapperAssemblyInputs<F, PCS, VC, ZkProof, PcsAssistProof>
where
    PcsAssistProof: PcsProofAssist<PCS>,
{
    pub preprocessing: JoltVerifierPreprocessing<PCS, VC>,
    pub public_io: JoltDevice,
    pub proof: JoltProof<PCS, VC, ZkProof, PcsAssistProof>,
    pub public_inputs: WrapperPublicInputs<F>,
    pub witness_inputs: WrapperWitnessInputs<F>,
}

pub struct WrapperR1csInstance<F> {
    pub matrices: ConstraintMatrices<F>,
    pub witness: Vec<F>,
    pub public_inputs: Vec<F>,
}

pub fn assemble_configured_verifier_r1cs<
    F,
    PCS,
    VC,
    ZkProof,
    PcsAssistProof,
>(
    inputs: WrapperAssemblyInputs<F, PCS, VC, ZkProof, PcsAssistProof>,
) -> Result<WrapperR1csInstance<F>, WrapperError>
where
    PcsAssistProof: PcsProofAssist<PCS>,
{
    /* assemble configured verifier R1CS */
}
```

Wrapper assembly uses the same compile-time-derived `JOLT_VERIFIER_CONFIG` as
the configured inner verifier. `WrapperAssemblyInputs` does not contain a
runtime protocol selector; it validates `proof.protocol` against the configured
inner verifier before assembling R1CS.

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

## Challenge Binding

For a self-contained wrapper, transcript challenges must be derived inside the
R1CS from the same messages absorbed by native verification:

```text
proof data / public inputs
  -> transcript absorb constraints
  -> challenge variables
  -> sumcheck and claim constraints use those same variables
```

A split verifier can expose challenges as public inputs, but then the outer
verifier must recompute and bind them. The standalone wrapper target uses the
self-contained path.

## Wrapper Circuit Shape

V1 should use one wrapper circuit shape per configured verifier computation:

```text
ordinary Jolt wrapper:
  setup for ordinary configured verifier R1CS

field-inline Jolt wrapper:
  setup for FR-active configured verifier R1CS

Dory-assist Jolt wrapper:
  setup for Dory-assist configured verifier R1CS
```

Do not force one parameterized circuit with runtime inactive rows in v1. The
SNARK backend sees a fixed R1CS for each proof shape. If a future deployment
needs one universal circuit, selector-gated inactive protocol components can be
designed separately.

## Relationship To Existing Groth16 Transpiler

The existing `transpiler/` path symbolically runs the current verifier for
stages 1-7 and emits gnark code. It is useful reference infrastructure and a
near-term Groth16 path, but it is not the target modular architecture.

`jolt-wrapper` is the modular replacement path:

```text
current transpiler:
  jolt-core verifier execution -> MleAst -> gnark

modular wrapper:
  configured verifier computation -> component R1CS helpers -> generic SNARK backend
```

The future `snarks/gnark` adapter may reuse lessons from the transpiler, but
the wrapper should not use `MleAst` as the main verifier IR.

## ZK Composition

ZK configuration is owned by
[selected-verifier-integration.md](selected-verifier-integration.md).

The primary v1 wrapped-ZK composition is:

```text
inner proof:
  transparent Jolt proof, never published as a standalone artifact

wrapper R1CS:
  proves the transparent configured verifier accepts that proof

wrapper backend:
  zero-knowledge Spartan + HyperKZG hides the inner proof and all verifier
  witness data

outer public statement:
  public IO
  preprocessing / verifier-key digest
  configured protocol identifier
  wrapper public inputs required by the selected deployment
```

This avoids encoding the BlindFold verifier in the first wrapper circuit. It is
also the cleanest separation of concerns: transparent Jolt supplies a complete
information-theoretic verifier relation, while the wrapper SNARK supplies the
zero-knowledge hiding for the final proof.

The optional future composition remains:

```text
BlindFold before wrapper:
  Jolt prover produces a standalone BlindFold proof
  configured verifier checks that BlindFold proof
  wrapper proves that configured BlindFold-mode verifier computation
```

That path is useful only when the base Jolt proof must itself be a standalone
ZK artifact before wrapping. It should not drive the v1 wrapper architecture.

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Generalize `jolt-r1cs::lowering`.
   - Add `SourceValue::{Constant, LinearCombination}`.
   - Keep existing constant-source tests passing.
   - Review gate: mixed constant/variable claim tests cover products and
     aliases.

2. Add variable-challenge `jolt-sumcheck::r1cs`.
   - Use `round.challenge() -> LinearCombination<F>`.
   - Keep constant challenges as the baked fast path for BlindFold.
   - Add Horner evaluation constraints for non-constant challenges.
   - Review gate: variable-challenge tests reject bad challenge assignments.

3. Add `jolt-transcript::r1cs`.
   - Add capability traits for field-oriented and byte-oriented transcript
     gadgets.
   - Implement the field-oriented Poseidon transcript gadget first.
   - Match native Poseidon transcript order and scalar encoding.
   - Review gate: R1CS challenge outputs match native transcript outputs for
     fixed absorption sequences.

4. Add wrapper protocol builder skeleton.
   - Implement `WrapperProtocolBuilder`, `WrapperClaimSources`, and configured
     stage hook traits over `R1csBuilder`.
   - Review gate: fixture stage lowers to satisfied R1CS without becoming part
     of the public wrapper API.

5. Add wrapper instance export.
   - Track stable public-input ordering.
   - Export matrices, witness, and publics for the backend.
   - Review gate: deterministic public-input order tests.

6. Assemble base configured verifier R1CS.
   - Start with a narrow configured verifier computation before full Jolt.
   - Review gate: native verifier and wrapper R1CS agree on accept/reject for
     the same fixture.

7. Add Dory-assist and Hyrax hooks.
   - Compile the selected Dory-assist `PcsProofAssist` verifier path rather
     than the ordinary Dory stage-8 verifier path.
   - Call Dory-assist claim, packing, final-check, and `jolt-hyrax::r1cs`
     helpers when the configured verifier includes Dory assist.
   - Review gate: Dory-assist configured computation produces satisfied
     R1CS.

8. Add field-inline wrapper hooks.
   - Include FR stages when the configured verifier includes field inline.
   - Review gate: FR-off and FR-on configs produce distinct deterministic
     R1CS shapes.

9. Implement ZK `snark_backends/spartan_hyperkzg`.
   - Consume arbitrary `WrapperR1csInstance`.
   - Keep the backend independent from Jolt protocol types.
   - Hide the R1CS witness, including transparent inner proof data.
   - Review gate: arbitrary R1CS proof verifies and witness
     randomization changes proof bytes for the same public statement.

10. Add configured-verifier wrapper fixture.
   - Prove a small transparent configured verifier computation end to end with
     the transparent proof held as private wrapper witness.
   - Review gate: mutating transcript challenges, public inputs, sumcheck
     claims, or opening values causes wrapper verification failure.
