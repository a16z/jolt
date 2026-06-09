# Spec: Wrapper Verifier Protocol

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

The wrapper proves an R1CS encoding of the configured Jolt verifier computation.
The near-term implementation target is `jolt-wrapper-verifier`: a concrete
wrapper verifier crate that owns relation construction and the Spartan +
HyperKZG verifier-side protocol. The production prover crate is intentionally
deferred until shared prover machinery lands.

This spec owns the wrapper protocol axis:

```text
configured verifier computation
  -> R1CS assembly
  -> Spartan + HyperKZG wrapper verification
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
jolt-wrapper-verifier crate structure
generic configured-verifier R1CS builder
transparent Jolt proof as private wrapper witness
in-circuit transcript replay
variable-challenge sumcheck R1CS
claim lowering over constants and variables
Hyrax R1CS integration when the configured verifier includes Dory assist
concrete Spartan + HyperKZG verifier-side protocol
integration-test fixtures for composed R1CS protocols
```

Out of scope:

```text
jolt-wrapper-prover production crate
generic snark-backends abstraction
making wrapper depend on BlindFold internals
wrapping the BlindFold verifier as the primary v1 ZK composition
folding / relaxed R1CS / row hiding
one universal circuit shape for every configured verifier mode
full gnark adapter
deleting the existing Groth16 transpiler path
```

## Design Intuition

`jolt-wrapper-verifier` is a concrete protocol verifier, not a generic backend
registry. It mirrors `jolt-verifier`: typed statement inputs, explicit
transcript flow, explicit protocol checks, and a small public `verify` API.

```text
configured verifier computation
  -> wrapper relation builder
  -> transcript constraints
  -> sumcheck verifier constraints
  -> claim equation constraints
  -> opening/PCS verifier constraints
  -> canonical R1CS relation and public input layout
  -> Spartan verifier checks
  -> HyperKZG opening checks
  -> wrapper verifier accepts
```

The verifier crate may expose test-gated slow fixture machinery that builds a
witness and proof for integration tests, but that machinery is not the
production prover API. The eventual `jolt-wrapper-prover` crate should consume
the same relation semantics and shared prover infrastructure rather than
duplicating protocol definitions.

There is no `snark_backends/` abstraction in v1. If a future Groth16/gnark
pipeline consumes the same R1CS relation, that adapter should be added after
the verifier relation is stable.

The relation/proof boundary remains simple:

```text
input:
  ConstraintMatrices<F>
  public inputs Vec<F>

output:
  wrapper verifier decision
```

Spartan + HyperKZG proof objects should not know about Jolt stages, field
inline, Dory assist, BlindFold, or transcript ordering. Those semantics are
compiled into the canonical R1CS relation by `jolt-wrapper-verifier`.

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
reduced opening statement data
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
     final reduced opening statement construction
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
It encodes the selected `PcsProofAssist` verifier for the same reduced opening
statement:

```text
reduced opening statement
  -> Dory-assist stage 1/2/3 verifier checks
  -> prefix-packing checks
  -> Hyrax dense-witness opening check
  -> Dory-assist final public pairing check / bound acceptance condition
```

This statement is intentionally about configured verifier acceptance. The VM
execution semantics enter through the soundness of the configured Jolt verifier
and the selected PCS-assist verifier, not through a separate execution circuit
inside the wrapper.

## Field Semantics Contract

The primary wrapper target is the BN254 recursion path. Field boundaries must be
explicit protocol objects, not implicit casts between witness representations:

```text
Fr-native:
  wrapper R1CS field
  Spartan + HyperKZG verifier field
  algebraic Poseidon wrapper transcript
  ordinary transparent Jolt verifier checks

Fq-nonnative inside Fr R1CS:
  Dory-assist verifier trace arithmetic
  Dory-assist sumcheck claims and challenges
  Dory-assist operation-family values, copy edges, and wiring values

Grumpkin native-coordinate constraints:
  Hyrax/Pedersen row commitments
  Grumpkin point addition and scalar multiplication coordinates over Fr
  scalar semantics remain Fq and must use the same non-native Fq representation
```

The wrapper verifier binds only the canonical relation identity and public
inputs. Test fixtures and the eventual production prover may consume
`ConstraintMatrices<Fr>`, `witness Vec<Fr>`, and `public_inputs Vec<Fr>`, but
that proving machinery must not know which entries encode native `Fr` values,
non-native `Fq` limbs, Grumpkin coordinates, transcript state, or Jolt proof
data.

The configured verifier R1CS assembly must make every field crossing explicit:

```text
Poseidon-Fr challenge
  -> domain-separated transcript label absorbed by the caller
  -> constrained Fr-to-Fq integer injection
  -> canonical Fq variable
  -> Dory-assist sumcheck / operation constraints
```

There should be no hidden `Fq` transcript, no unconstrained `Fr` witness standing
in for an `Fq` value, and no component-local conversion rule that is not visible
at the wrapper assembly boundary. If stronger challenge uniformity is needed
than direct `Fr` injection, the conversion gadget should derive multiple
Poseidon-`Fr` words and reduce the combined integer modulo `Fq` with explicit
canonicality constraints.

The v1 challenge map is the canonical integer injection `Fr -> Fq`. The
transcript remains a Poseidon transcript over `Fr`; the caller domain-separates
before squeezing the `Fr` challenge, then calls the non-native helper that
constrains the `Fr` canonical limbs and reuses those limbs as the `Fq`
challenge. This is valid because BN254 has `q_Fr < q_Fq`. The API and callsites
should use injection terminology so the code does not look like an unchecked
cast or a second `Fq` transcript.

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

## Crate Split

The wrapper stack is split by role:

```text
jolt-wrapper-verifier:
  owns relation construction, public wrapper verification, concrete
  Spartan + HyperKZG verifier checks, and test-gated slow fixtures

jolt-wrapper-prover:
  deferred; later owns production witness/proof generation using shared prover
  machinery

jolt-claims:
  owns wrapper-spartan-hyperkzg protocol semantics, statement shapes,
  dimension facts, Spartan round/claim structure, and formula facts shared by
  verifier/prover code
```

`jolt-wrapper-verifier` should be usable before the Dory-assist verifier is
complete. It can assemble and test composed R1CS protocols over the reusable
component gadgets already provided by `jolt-r1cs`, `jolt-poly`, `jolt-sumcheck`,
`jolt-openings`, `jolt-crypto`, and `jolt-hyrax`.

## `jolt-wrapper-verifier` Layout

Target layout:

```text
crates/jolt-wrapper-verifier/
  Cargo.toml
  src/
    lib.rs
    config.rs
    error.rs
    proof.rs
    r1cs_builder.rs
    verifier.rs
    stages/
      mod.rs
      r1cs_relation/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
      spartan/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
      hyperkzg/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
```

The crate should mirror the useful `jolt-verifier` organization, but with only
the protocol phases the wrapper actually needs. `config`, `proof`, and
`verifier` are top-level verifier concepts. `r1cs_builder` is the only
test-facing relation-construction helper for now; it wraps `R1csBuilder`,
tracks public-input layout, and exposes witness checking for integration tests.
`verifier` owns the small public `verify(...)` entry point and threads the
transcript in a linear order. `stages/*` own typed inputs, typed outputs, and
phase-local checks.

Do not add top-level `protocol`, `statement`, `witness`, `preprocessing`, or
`test_support` modules until they carry real verifier-side responsibilities.
Witness construction belongs to the future prover side; verifier inputs can
stay as typed structs in `verifier.rs`.

The wrapper stages are intentionally coarse:

```text
r1cs_relation:
  canonicalize/bind public wrapper inputs, R1CS dimensions, public input
  layout, and relation identity using jolt-claims wrapper facts

spartan:
  verify the Spartan transcript, claims, and verifier equations for the
  configured relation

hyperkzg:
  verify the HyperKZG opening checks required by the Spartan verifier
```

Do not add a wrapper-local `components/` directory for generic math. Reusable
encodings for transcript, scalar arithmetic, sumcheck, opening reductions,
Hyrax, curve groups, and commitments live in the crates that own those
abstractions. `jolt-wrapper-verifier` sequences those APIs.

The crate must not grow a generic backend registry. Spartan + HyperKZG is the
concrete verifier path for v1.

## Verifier Flow

The top-level verifier should read like a smaller `jolt-verifier`:

```rust
pub fn verify<F, PCS, T>(
    config: &WrapperVerifierConfig,
    inputs: WrapperVerifierInputs<'_, F>,
    proof: &WrapperProof<PCS>,
) -> Result<(), WrapperError>
where
    T: Transcript<Challenge = F>,
{
    validate_proof_config(config, proof)?;

    let checked = validate_inputs(config, inputs)?;

    let mut transcript = T::new(config.transcript_label);
    absorb_public_inputs(&checked, inputs.public_inputs, &mut transcript);
    absorb_wrapper_commitments(proof, &mut transcript);

    let relation = stages::r1cs_relation::verify(
        &checked,
        &config.key.relation,
        inputs.public_inputs,
        proof,
        &mut transcript,
    )?;
    let spartan = stages::spartan::verify(
        &checked,
        proof,
        &mut transcript,
        stages::spartan::deps(&relation),
    )?;
    stages::hyperkzg::verify(
        &checked,
        &config.key.hyperkzg,
        proof,
        &mut transcript,
        stages::hyperkzg::deps(&relation, &spartan),
    )?;

    Ok(())
}
```

The exact helper names can change during implementation, but the shape should
not: validate first, bind public wrapper inputs, then advance protocol
phases linearly through explicit typed outputs. There should be no hidden
state object that accumulates implicit claims across phases.

## R1CS Builder

`jolt-wrapper-verifier::r1cs_builder` owns the thin wrapper around
`jolt_r1cs::R1csBuilder` used while constructing the configured verifier
r1cs_relation:

```rust
pub struct WrapperR1csBuilder<F, Tr> {
    pub builder: jolt_r1cs::R1csBuilder<F>,
    pub transcript: Tr,
    pub layout: WrapperR1csLayout,
    pub public_inputs: Vec<F>,
}
```

The first implementation can use `R1csBuilder` directly. A separate assembly
object is not required to encode verifier equations; it is only useful once the
wrapper needs stable public-input ordering and a single export point for
matrices, witness, and public inputs. `WrapperR1csBuilder` only adds
wrapper-specific transcript state and public-input layout tracking. Public
scalar allocation records the stable public-input slot and emits an equality
row tying the committed witness slot to the public value; the wrapper verifier
must never rely on an out-of-band layout table for that consistency.

Helper shape:

```rust
impl<F, Tr> WrapperR1csBuilder<F, Tr> {
    pub fn alloc_public_scalar(&mut self, value: F) -> AssignedScalar<F>;
    pub fn alloc_witness_scalar(&mut self, value: F) -> AssignedScalar<F>;
    pub fn finish(self) -> Result<WrapperR1csProtocol<F>, WrapperError>;
}
```

Each configured verifier stage lowers through an R1CS hook:

```rust
pub trait WrapperR1csStage<F> {
    type Error;

    fn append_r1cs<Tr>(
        &self,
        builder: &mut WrapperR1csBuilder<F, Tr>,
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

When the wrapper relation needs a single export point for test fixtures or a
future prover, either `jolt-wrapper-verifier` or `jolt-r1cs` can add a small
assembly/export helper:

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
implementation. The wrapper relation stage should provide explicit source
tables backed by transcript-derived challenges, public inputs, opening
variables, and stage-local aliases. Do not hide those behind a long-lived
wrapper state object.

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

The Poseidon R1CS transcript gadget must match the Jolt proof transcript used by
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

non-native field and range helpers:
  jolt-r1cs::nonnative, including canonical Fq-in-Fr variables

scalar gadget abstraction:
  jolt-r1cs helpers/traits over native Fr and non-native Fq variables

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

curve and group constraints:
  jolt-crypto::r1cs, including Grumpkin point and Pedersen constraints

Hyrax verifier checks:
  jolt-hyrax::r1cs

Dory-assist verifier staging:
  jolt-dory-assist-verifier crate, mirroring jolt-verifier for Dory-assist semantics

Dory-assist protocol facts and packing:
  jolt-claims::protocols::dory_assist

Wrapper Spartan + HyperKZG protocol semantics:
  jolt-claims::protocols::wrapper_spartan_hyperkzg owns the wrapper statement
  facts, canonical protocol/transcript labels, raw and padded R1CS dimension
  facts, Spartan protocol shape, public-input layout, round/claim structure, and
  formula semantics consumed by verifier/prover code
```

Wrapper implementation should be foundation-first. Before adding bespoke
Dory-assist verifier R1CS wiring, each reusable component should expose a
generic, crate-owned R1CS interface with local tests and at least one
cross-crate composition test. The final configured-verifier assembly should be
mostly sequencing typed component hooks, not reimplementing component math in
`jolt-wrapper-verifier`.

The safe implementation rule is: when a wrapper check feels Dory-assist-specific
but is really a generic algebraic, transcript, sumcheck, opening, or group
constraint, stop and upstream that piece to the crate that owns the abstraction.
Only after the generic component has local soundness tests and a composed R1CS
test should the configured verifier call it.

`jolt-wrapper-verifier::stages::r1cs_relation` sequences these helpers according to
the configured verifier flow exported by `jolt-verifier`. It should not
implement component math directly. It allocates the proof and public-input
witness, threads transcript state, records aliases and claim sources, and calls
the R1CS hooks owned by the component crates.

## Generic R1CS Building Blocks

The wrapper should be assembled from reusable R1CS components before any
full Dory-assist verifier circuit is attempted. The main foundation blocks are:

```text
field arithmetic:
  native Fr constraints from jolt-r1cs::builder
  non-native Fq constraints from jolt-r1cs::nonnative

transcript:
  Poseidon over Fr from jolt-transcript::r1cs
  no hidden Fq transcript

challenge crossing:
  caller absorbs protocol label in Poseidon-Fr transcript
  transcript squeezes AssignedScalar<Fr>
  jolt-r1cs::nonnative injects the Fr challenge into Fq
  jolt-r1cs::nonnative exposes constrained little-endian Fq bits for scalar gadgets

sumcheck:
  jolt-sumcheck::r1cs owns verifier equations
  native-Fr path uses LinearCombination<Fr> / AssignedScalar<Fr>
  non-native-Fq path uses FqVar inside R1csBuilder<Fr>
  same semantic path for baked and in-circuit challenges

claims/formulas:
  jolt-claims describes expressions
  jolt-r1cs::lowering turns those expressions into constraints

polynomial evaluation:
  jolt-poly::r1cs owns equality-polynomial and multilinear-evaluation gadgets
  these gadgets are generic over jolt-poly::r1cs::PolynomialScalarGadget
  jolt-r1cs implements that trait for AssignedScalar and FqVar

openings:
  jolt-openings owns opening semantics
  R1CS helpers only constrain the algebra needed by the selected verifier

groups/commitments:
  jolt-crypto::r1cs owns group gadgets and vector-commitment verifier gadgets
  concrete Grumpkin/Pedersen is one implementation
  the generic R1CS surface is defined before concrete Pedersen verification

Hyrax:
  jolt-hyrax::r1cs owns Hyrax opening checks
  it is generic over jolt-crypto::r1cs vector-commitment gadgets
```

Each block needs two classes of tests:

```text
local tests:
  valid witness satisfies constraints
  malformed witness violates constraints
  representative single-variable tampering rejects

composition tests:
  at least two crates interact in one R1CS
  the composed witness verifies
  tampering across every boundary rejects
```

The current composition-test pattern lives under `crates/jolt-r1cs/tests/` and
should remain the place for broad R1CS interoperability checks. A useful
composition test should cross module boundaries, for example:

```text
Poseidon transcript -> Fr challenge -> Fq injection -> Fq arithmetic
Poseidon transcript -> variable challenge -> sumcheck R1CS
Fq arithmetic -> claim lowering -> sumcheck/output relation
Grumpkin commitment -> Hyrax opening equation -> Fq scalar claim
```

`jolt-hyrax::r1cs` should mirror the native Hyrax crate's generic shape. It
should be generic over a vector-commitment R1CS interface rather than hard-code
Grumpkin or Pedersen into Hyrax. The concrete wrapper target instantiates that
interface with Grumpkin-backed Pedersen commitments, but the Hyrax verifier
lowering should only see:

```text
scalar variables:
  S: jolt-poly::r1cs::PolynomialScalarGadget

row commitments:
  VC::OutputVar

vector commitment verifier:
  VC::verify_opening_r1cs(...)
  VC::linear_combine_commitments_r1cs(...)
```

The intended Hyrax R1CS flow is:

```text
row_point, col_point = split(opening_point)
row_weights = jolt-poly::r1cs::eq_evals(row_point)
combined_commitment = VC::linear_combine_commitments_r1cs(row_commitments, row_weights)
VC::verify_opening_r1cs(combined_commitment, combined_row, combined_blinding)
col_weights = jolt-poly::r1cs::eq_evals(col_point)
opened_eval = jolt-poly::r1cs::inner_product(combined_row, col_weights)
opened_eval == claimed_eval
```

This keeps field arithmetic and polynomial evaluation out of `jolt-hyrax`,
keeps group laws and Pedersen/MSM logic out of `jolt-hyrax`, and leaves
Dory-assist stage/copy/packing semantics out of `jolt-hyrax`.

### Scalar Gadget Interface

Several verifier equations are field-generic mathematically but not
representation-generic in R1CS. For the BN254 wrapper target we need both:

```text
native Fr variables:
  AssignedScalar<Fr> / LinearCombination<Fr>
  cheap native multiplication constraints

non-native Fq variables:
  FqVar inside R1csBuilder<Fr>
  canonical limbs and explicit reduction constraints
```

The reusable equation crates should therefore target a small scalar-gadget API
rather than hard-coding either `LinearCombination<Fr>` or Dory-assist-specific
types. The API should expose only the operations verifier equations need:

```text
constant
allocated witness/public value
assert_equal
add/sub/neg
mul
select by native boolean when needed
affine combination / dot product when useful
```

`jolt-r1cs` should own this trait or helper layer because it owns the builder
and both native and non-native scalar representations. `jolt-sumcheck::r1cs`
can then express round-sum and polynomial-evaluation constraints once, with a
native implementation for `Fr` and a non-native implementation for `FqVar`.
Dory-assist verifier code should consume that generic sumcheck helper instead
of containing its own Horner/evaluation/reduction logic.

## Dory Assist In Wrapper

A Dory-assisted wrapper does not encode the ordinary Dory stage-8 verifier path.
It encodes the configured verifier flow after the selected `PcsProofAssist`
implementation has replaced that ordinary PCS opening check. Concretely:

```text
base Jolt stages
  -> reduced opening statement
  -> selected DoryAssist::verify_clear / verify_zk(...)
  -> Dory-assist stage 1/2/3 checks
  -> Hyrax opening check for the packed dense witness
  -> Dory-assist final public pairing check
  -> wrapper R1CS proves this configured verifier accepts
```

The wrapper-facing R1CS hooks follow the Dory-assist ownership split:

```text
Dory-assist staged verifier flow:
  jolt-dory-assist-verifier crate

Dory-assist stage sumchecks:
  jolt-sumcheck::r1cs

Dory-assist operation-family and wiring formulas:
  jolt-claims::protocols::dory_assist + jolt-r1cs lowering

Dory-assist prefix packing:
  jolt-claims::protocols::dory_assist::packing

Dory-assist dense-witness opening:
  jolt-hyrax::r1cs

Dory-assist final public pairing check:
  jolt-dory-assist-verifier crate hook
```

The `jolt-dory-assist-verifier` crate mirrors `jolt-verifier` in scope, but for
the Dory-assist protocol only. It owns the stage ordering, proof payload
structure, transcript order, native verification, wrapper-facing R1CS hook, and
final acceptance condition for the selected assist proof.
`jolt-wrapper-verifier` should not match on Dory-assist operation families or
copy-edge internals. It receives a typed reduced opening statement from the
base Jolt stage-8 assembly, passes it to the selected Dory-assist verifier
hook, and constrains the returned acceptance condition as part of the
configured verifier relation.

`jolt-hyrax::r1cs` should only lower the Hyrax opening verifier used by the
assist proof and should mirror the native Hyrax genericity over vector
commitments:

```text
row_point, col_point = split(opening_point)
row_weights = jolt-poly::r1cs::eq_evals(row_point)
combined_commitment = VC::linear_combine_commitments_r1cs(row_commitments, row_weights)
VC::verify_opening_r1cs(combined_commitment, combined_row, row_combination_randomness)
col_weights = jolt-poly::r1cs::eq_evals(col_point)
claimed_eval == jolt-poly::r1cs::inner_product(combined_row, col_weights)
```

It should not know Dory-assist stage IDs, operation families, prefix codes, or
copy constraints. It also should not implement Grumpkin group laws, Pedersen
MSMs, equality-polynomial evaluation, or scalar field arithmetic directly.
Those are owned by `jolt-crypto::r1cs`, `jolt-poly::r1cs`, and `jolt-r1cs`
respectively. The concrete BN254/Grumpkin wrapper path instantiates the generic
Hyrax vector-commitment interface with Grumpkin-backed Pedersen, but Hyrax
itself should not be specialized to that commitment instantiation.

For the BN254/Grumpkin recursion target, Dory assist uses Grumpkin-backed
Pedersen row commitments. This is not only a wrapper convenience: Grumpkin's
scalar field is BN254 `Fq`, matching the arithmetic needed by the assisted Dory
verifier trace. The wrapper benefit is narrower but important: Grumpkin's base
field is BN254 `Fr`, so when the configured verifier is compiled into a BN254
`Fr` R1CS, the Hyrax commitment group-law constraints are native-coordinate
constraints rather than BN254-G1-over-`Fq` non-native group arithmetic.

Any `Fq` scalar arithmetic, range constraints, operation-family semantics, and
copy/wiring checks remain Dory-assist constraints. The Hyrax R1CS module only
binds the already-packed dense witness to the assist verifier's claimed dense
opening.

Dory-assist `Fq` sumcheck challenges must be derived from the wrapper's
Poseidon-over-`Fr` transcript by an explicit, domain-separated
`Fr`-to-`Fq` injection. The transcript module only produces `Fr` challenges. The
non-native field module constrains the injection into `Fq`. Dory-assist verifier
code owns the protocol decision about which labels to absorb before each
challenge. We should avoid a hidden Fq-Poseidon transcript: if stronger
challenge uniformity is needed than direct `Fr` injection, derive two or more
Poseidon-`Fr` words and reduce the combined integer modulo `Fq` with an explicit
conversion gadget.

The Dory-assist spec says final exponentiation and public pairing equality are
native verifier work in v1. For wrapper purposes, that means "outside the
Dory-assist auxiliary proof," not "outside the configured verifier." A
self-contained wrapper must still account for that selected assist-verifier
acceptance condition: either the `jolt-dory-assist-verifier` crate supplies the
R1CS hook for the final public check, or the wrapper verifier must recheck and
bind that public condition outside the wrapper proof. The primary
self-contained wrapper target should not rely on an unchecked native side
condition.

Implementation preference for v1: keep final exponentiation out of the
Dory-assist PIOP and implement the self-contained wrapper path as an R1CS
final-check hook owned by `jolt-dory-assist-verifier`. This avoids adding
final-exp witness columns, copy edges, prefix-packing entries, and Hyrax-bound
claims before we know the wrapper cost is a bottleneck. If it later becomes too
expensive as non-native R1CS arithmetic, the same acceptance condition can be
moved behind a dedicated Dory-assist final-exp PIOP component.

## Assembly API

```rust
pub struct WrapperVerifierConfig {
    pub transcript_label: &'static [u8],
    pub key: WrapperVerifierKey,
}

pub struct WrapperVerifierKey<F, P> {
    pub relation: ConstraintMatrices<F>,
    pub relation_statement: R1csRelationStatement,
    pub hyperkzg: HyperKZGVerifierSetup<P>,
}

pub struct WrapperVerifierInputs<'a, F> {
    pub public_inputs: &'a [F],
}
```

`jolt-wrapper-verifier` uses the same compile-time-derived
`JOLT_VERIFIER_CONFIG` as the configured inner verifier once full Jolt wrapping
is wired. In the current arbitrary-R1CS scaffold, the verifier key owns the
public R1CS relation, its canonical relation statement, and the HyperKZG
verifier setup. Per-proof verifier inputs are intentionally only public wrapper
inputs; richer proof/preprocessing objects should be introduced only when a
verifier stage consumes them.

Verifier API:

```rust
pub fn verify<F, Tr>(
    config: &WrapperVerifierConfig,
    inputs: WrapperVerifierInputs<'_, F>,
    proof: &WrapperProof<F>,
) -> Result<(), WrapperError>;
```

The public API verifies the concrete Spartan + HyperKZG wrapper proof. Internal
modules may expose narrower helpers such as `verify_spartan` and
`verify_hyperkzg`, but those helpers remain verifier-side and protocol-specific.
They are not a generic backend interface.

`r1cs_builder.rs` may expose test-facing relation construction and witness
checking helpers. It should not become a prover abstraction; production witness
construction belongs to the future `jolt-wrapper-prover`.

## R1CS Protocol Test Harness

Until the concrete Spartan + HyperKZG proof path is implemented, wrapper
integration tests should use complete small R1CS protocol cases with the raw
R1CS witness checker as the acceptance oracle. This keeps the target relation
rigorous without leaking test-only prover machinery into the public verifier
API.

The harness should be manifest-style:

```text
named protocol case
  -> explicit component coverage list
  -> constructed WrapperR1csProtocol
  -> named tamper targets
```

Each protocol case must first accept its valid witness, then reject every named
tamper target after cloning and mutating the constructed protocol. Public input
mutations are added uniformly: scalar mutation, truncation, extension, and
public-input-layout corruption. Component-specific targets should cover every
allocated value whose consistency is material to the protocol check.

Current foundation cases:

```text
poseidon_public_arithmetic:
  Poseidon-over-Fr transcript, byte absorption, Fr arithmetic, public outputs

poseidon_fq_sumcheck_opening_reduction:
  Poseidon-over-Fr transcript, Fr -> Fq challenge injection, non-native Fq
  arithmetic, native and non-native sumcheck constraints, same-point opening
  reduction

poseidon_hyrax_pedersen_opening:
  Poseidon-over-Fr transcript, Fr -> Fq challenge injection, Grumpkin/Pedersen
  Hyrax opening verification
```

When the Spartan verifier lands, the same cases should be lifted from raw
`verify_r1cs_witness` checks to test-gated Spartan + HyperKZG proof generation
and verification. The tamper manifest should remain the source of truth for
which protocol values must reject.

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
Spartan + HyperKZG wrapper verifier sees a fixed R1CS identity for each proof
shape. If a future deployment needs one universal circuit, selector-gated
inactive protocol components can be designed separately.

## Relationship To Existing Groth16 Transpiler

The existing `transpiler/` path symbolically runs the current verifier for
stages 1-7 and emits gnark code. It is useful reference infrastructure and a
near-term Groth16 path, but it is not the target modular architecture.

`jolt-wrapper-verifier` is the modular replacement path:

```text
current transpiler:
  jolt-core verifier execution -> MleAst -> gnark

modular wrapper:
  configured verifier computation -> component R1CS helpers
  -> concrete Spartan + HyperKZG wrapper verifier
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

wrapper verifier proof:
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

The implemented wrapper-ZK verifier path follows the same separation. Spartan
sumcheck claims are carried as committed output-claim rows. The HyperKZG stage
uses the PCS ZK opening verifier and returns a hidden evaluation commitment for
`Z(ry)`. A small BlindFold statement then proves the committed Spartan claim
relations and binds the hidden `Z(ry)` scalar to that hidden HyperKZG evaluation
commitment. The verifier continues the same Fiat-Shamir transcript through
Spartan, HyperKZG, and BlindFold; it does not start a detached BlindFold
transcript.

For this to compose, the wrapper VC setup used by BlindFold must be compatible
with the ZK HyperKZG evaluation commitment basis: the single-value VC message
generator and blinding generator must match the PCS basis used for the hidden
evaluation commitment. Tests instantiate this with Grumpkin-backed Pedersen.

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

2. Add variable-challenge native `jolt-sumcheck::r1cs`.
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

4. Add explicit field-semantics helpers for wrapper assembly.
   - Add the canonical non-native `Fq`-inside-`Fr` representation in
     `jolt-r1cs::nonnative`.
   - Add boolean, range, equality, add/sub/mul/inv/select helpers needed by
     Dory-assist verifier constraints.
   - Add the constrained `Poseidon-Fr -> Fq` integer injection used by
     Dory-assist sumchecks. Domain separation remains caller-owned transcript
     logic; the injection helper only constrains canonicality and equality.
   - Review gate: bad canonicality, bad reduction, and bad challenge-conversion
     witnesses are rejected.

5. Add cross-crate R1CS composition tests.
   - Create integration tests under `crates/jolt-r1cs/tests/` that pull in
     R1CS-facing modules across crates.
   - Compose Poseidon transcript, native challenges, non-native Fq injection,
     non-native arithmetic, wrapper public inputs, and sumcheck constraints in
     one circuit.
   - Review gate: composed witness verifies and representative tampering across
     every module boundary rejects.

6. Add wrapper R1CS builder skeleton.
   - Implement `WrapperR1csBuilder`, public-input layout tracking, and
     configured stage hook traits over `R1csBuilder`.
   - Review gate: fixture stage lowers to satisfied R1CS without becoming part
     of the public wrapper API.

7. Add wrapper instance export.
   - Track stable public-input ordering.
   - Export matrices, witness, and publics for test fixtures and the future
     production prover.
   - Review gate: deterministic public-input order tests.

8. Add scalar-gadget foundation helpers.
   - Define the minimal native/non-native scalar helper surface needed by
     verifier equations: constants, witnesses, equality, arithmetic, select,
     affine combinations, and dot products.
   - Implement native `Fr` helpers over `AssignedScalar<Fr>` /
     `LinearCombination<Fr>`.
   - Implement non-native `Fq` helpers over `FqVar`.
   - Review gate: the same toy algebraic relation runs over native `Fr` and
     non-native `FqVar`, and tampering either representation rejects.
   - Status: implemented in `jolt-r1cs::scalar`; covered by shared native and
     non-native relation tests plus boundary tampering checks.

9. Generalize sumcheck R1CS over scalar gadgets.
   - Keep the existing native path for BlindFold and ordinary wrapper checks.
   - Add a non-native `FqVar` path for Dory-assist sumcheck semantics.
   - Reuse the same round-sum and polynomial-evaluation logic for both paths
     instead of duplicating Horner/evaluation logic in Dory-assist code.
   - Review gate: a toy sumcheck relation verifies over native `Fr` and
     non-native `Fq`, with bad challenge, bad coefficient, bad round sum, and
     bad output claim witnesses rejected.
   - Status: implemented as an additive `jolt-sumcheck::r1cs` gadget-round API.
     Existing native layout APIs remain intact for BlindFold. Tests cover native
     and non-native valid witnesses plus bad round sum, bad challenge, bad output
     claim, and representative tampering.

10. Add claim/formula lowering for scalar gadgets.
   - Keep `jolt-claims` as the formula owner.
   - Lower formulas into native scalar helpers or non-native `FqVar` helpers
     depending on the configured verifier field semantics.
   - Do not make `jolt-claims` aware of R1CS builders or witness allocation.
   - Review gate: a formula involving constants, transcript challenges,
     openings, and public values lowers to equivalent native and non-native
     test relations.
   - Status: implemented in `jolt-r1cs::lowering` as an additive
     scalar-gadget source table and lowering path. Existing native
     `LinearCombination` lowering remains intact for BlindFold. Tests cover
     native and non-native formula lowering, missing sources, bad expected
     output, and representative tampering.

11. Add generic opening/evaluation helper tests.
   - Identify opening consistency algebra that is independent of Dory-assist.
   - Put generic helper APIs in `jolt-openings`/`jolt-r1cs` as appropriate.
   - Review gate: opening/evaluation toy relations compose with transcript and
     scalar-gadget helpers; tampered point or claim rejects. Commitment
     tampering belongs to the selected PCS/commitment R1CS component.
   - Status: implemented the first `jolt-openings/r1cs` slice behind the
     `r1cs` feature. It constrains same-point opening claims and scalar RLC of
     opening claims over both native `AssignedScalar<Fr>` and non-native
     `FqVar`. Tests now check equivalence with native
     `rlc_combine_scalars`, Poseidon-R1CS transcript challenge composition,
     independent same-point groups, typed error cases, and tampering of point,
     claim, challenge, and expected reduced claim witnesses. It intentionally
     does not constrain commitment equations; those remain owned by
     `jolt-crypto::r1cs`, `jolt-hyrax::r1cs`, or the selected PCS-assist
     verifier.

12. Add Grumpkin/Pedersen R1CS primitives.
   - Use `jolt-crypto::r1cs` for native-coordinate Grumpkin point constraints
     over `Fr`.
   - Define generic group/vector-commitment R1CS traits that Hyrax can consume
     without naming Grumpkin or Pedersen.
   - Add point equality, on-curve checks, addition/doubling, scalar
     multiplication, MSM, and Pedersen commitment helpers with explicit scalar
     semantics.
   - Review gate: invalid point, invalid scalar link, and tampered commitment
     witnesses reject.
   - Status: implemented the first `jolt-crypto/r1cs` slice behind the `r1cs`
     feature. It adds affine `GrumpkinPointVar`, coordinate allocation from
     native `GrumpkinPoint`, equality, on-curve constraints for
     `y^2 = x^3 - 17`, and non-exceptional affine addition with an explicitly
     constrained denominator inverse. The helper rejects identity points and
     `x_1 == x_2` exceptional additions instead of silently applying an
     incomplete formula. Tests cover valid points, invalid points, coordinate
     tampering, valid addition, output tampering, and exceptional-addition
     rejection. Doubling, scalar multiplication, MSM, and Pedersen commitment
     helpers remain future sub-slices.
   - Status update: added the generic R1CS trait surface:
     `GroupElementVar`, `NonExceptionalAddGroupVar`,
     `VectorCommitmentOpeningVar`, and `VectorCommitmentR1cs`.
     `GrumpkinPointVar` implements the group-element and non-exceptional-add
     traits. This intentionally does not claim full Pedersen verification yet:
     the current Grumpkin point gadget is affine and non-identity, so a full
     vector-commitment verifier still needs a complete group representation,
     variable scalar multiplication, MSM, and Pedersen opening constraints.
   - Status update: added `FqVar::bits_le()` in `jolt-r1cs` for constrained
     canonical little-endian scalar bits, plus `DoubleGroupVar` and affine
     Grumpkin doubling constraints in `jolt-crypto::r1cs`. Tests cover valid
     bit decompositions, bit tampering, valid doubling, output tampering, and
     zero-y exceptional doubling rejection. This is the immediate substrate for
     variable scalar multiplication; it is not yet scalar multiplication or
     Pedersen verification.
   - Status update: added an identity-aware Grumpkin point gadget and
     fixed-base Grumpkin scalar multiplication over constrained `FqVar` bits.
     The identity-aware gadget carries an explicit boolean identity flag,
     enforces on-curve constraints for non-identity points, enforces zero
     coordinates for identity witnesses, and handles identity cases in addition
     without witness-branching. The fixed-base scalar-mul helper walks the
     canonical little-endian `Fq` bit decomposition, conditionally selects each
     fixed base power, and constrains the accumulator with the identity-aware
     addition gadget. Tests cover zero scalar, nonzero scalar, the generic
     fixed-base trait surface, malformed identity witnesses, identity addition
     cases, ordinary addition, output tampering, and exceptional non-identity
     additions. MSM and Pedersen commitment verification remain future
     sub-slices.
   - Status update: added a fixed-base Grumpkin MSM helper and a
     Grumpkin-backed Pedersen opening helper over `FqVar` values/blinding. This
     constrains the Pedersen equation
     `C = sum_i values[i] * G_i + blinding * H` using the fixed-base scalar
     multiplication substrate. Tests cover valid openings, typed capacity
     errors, fixed-base MSM length errors, and tampering of committed values,
     blinding, and commitment coordinates.
   - Status update: added complete identity-aware Grumpkin addition, variable-
     base Grumpkin scalar multiplication, Grumpkin commitment linear
     combination, and the concrete `VectorCommitmentR1cs` implementation for
     `Pedersen<GrumpkinPoint>`. Complete addition covers identity, ordinary
     addition, doubling, and inverse-pair addition to identity with constrained
     zero-test flags and gated equations. Variable-base scalar multiplication
     uses constrained canonical `FqVar` bits and complete addition throughout.
     Tests cover complete-add identity/ordinary/doubling/inverse cases,
     generic trait surfaces, variable-base scalar multiplication, commitment
     linear combination, Pedersen opening verification through the generic VC
     trait, typed length errors, and tampering of base, coefficient, output,
     values, blinding, and commitment coordinates. This gives Hyrax the generic
     row-commitment linear-combination primitive it needs without owning
     Grumpkin or Pedersen logic.

13. Add Hyrax R1CS component hooks.
   - Use `jolt-hyrax::r1cs` for dense-witness opening verification.
   - Consume generic `jolt-crypto::r1cs` vector-commitment helpers rather than
     owning group logic or specializing to Grumpkin/Pedersen.
   - Use `jolt-poly::r1cs` for equality-polynomial evaluation and inner
     products.
   - Keep Hyrax unaware of Dory-assist stages and copy-edge semantics.
   - Review gate: tampering the dense-witness opening, Grumpkin commitment, or
     packed evaluation rejects.
   - Status: implemented the `jolt-poly/r1cs` foundation behind the `r1cs`
     feature. `jolt-poly` defines `PolynomialScalarGadget` so it can own
     polynomial equations without depending on `jolt-r1cs` and creating a
     dependency cycle. `jolt-r1cs` implements the trait for `AssignedScalar`
     and `FqVar`. The first helpers cover `eq_eval`, `eq_evals`,
     `scaled_eq_evals`, `inner_product`, and `multilinear_eval`. Tests cover
     plain polynomial equivalence in `jolt-poly` and real native/non-native
     constraint composition plus tampering in `crates/jolt-r1cs/tests`.
   - Status update: implemented the first `jolt-hyrax/r1cs` slice behind the
     `r1cs` feature. The module exposes `HyraxOpeningR1csInput` and a generic
     `verify_opening<VC>` hook over `jolt_crypto::r1cs::VectorCommitmentR1cs`.
     It computes row eq weights with `jolt-poly::r1cs`, linearly combines row
     commitments through the VC interface, verifies the combined-row vector
     commitment opening, computes the entry-point inner product, and constrains
     it to the claimed evaluation. Production code stays generic over VC and
     does not mention Dory assist, Grumpkin formulas, or Pedersen internals.
     Tests instantiate the hook with Grumpkin-backed Pedersen and non-native
     `FqVar`, using nonzero row blindings. They cover valid openings,
     row-commitment count mismatch, combined-row length mismatch, and tampering
     of row commitments, row point, entry point, combined row, combined
     blinding, and claimed evaluation.

14. Stand up `jolt-wrapper-verifier` and assemble the base relation.
   - Migrate or replace the current wrapper scaffold so the verifier crate owns
     `config`, `proof`, `r1cs_builder`, `verifier`, and coarse
     `stages/{r1cs_relation,spartan,hyperkzg}` modules.
   - Keep the top-level `verify(...)` flow aligned with `jolt-verifier`:
     validate config and inputs, bind public wrapper inputs, then run
     explicit typed stages in protocol order.
   - Remove any `snark_backends/` abstraction from the near-term design.
   - Start with a narrow configured verifier computation before full Jolt.
   - Review gate: native verifier and wrapper R1CS agree on accept/reject for
     the same fixture.
   - Status: migrated the scaffold from `jolt-wrapper` to
     `jolt-wrapper-verifier`, removed the `snark_backends` directory, added the
     verifier-style top-level modules and coarse `r1cs_relation`, `spartan`, and
     `hyperkzg` stages, then slimmed the top-level surface down to
     `config`, `proof`, `r1cs_builder`, `verifier`, and `stages`. The
     `r1cs_relation` stage is wired to the new
     `jolt-claims::protocols::wrapper_spartan_hyperkzg` dimension/fact module.
     Slice 1 of the wrapper facts is complete: `jolt-claims` now exposes the
     canonical wrapper protocol labels, transcript labels, raw relation
     dimensions, padded/log Spartan dimensions, and contiguous public-input
     layout with checked constructors; `jolt-wrapper-verifier` consumes those
     facts instead of hard-coding labels locally. At this checkpoint the
     Spartan and HyperKZG stages are explicit verifier placeholders; the
     existing R1CS fixture path remains test-facing and passes under the new
     crate name.

15. Implement concrete Spartan + HyperKZG verifier checks in
    `jolt-wrapper-verifier`.
   - Bind the canonical wrapper statement, relation identity, and public
     inputs into the wrapper transcript.
   - Verify the Spartan proof for the configured wrapper relation.
   - Verify the HyperKZG opening proofs required by Spartan.
   - Keep helper modules protocol-specific rather than generic backend
     abstractions.
   - Review gate: test-gated fixture proofs verify, and mutating proof bytes,
     public inputs, relation identity, Spartan messages, or HyperKZG openings
     rejects.
   - Status update: the first verifier slice is wired. `WrapperProof` now
     has an explicit relation statement plus `spartan` and `hyperkzg` proof
     payloads. The `r1cs_relation` stage derives canonical wrapper statement
     facts from the verifier's public statement, checks that the proof's
     relation statement matches those public facts, binds the protocol id, raw
     relation dimensions, padded/log Spartan dimensions, public-input layout,
     and public inputs into the transcript, then hands typed statement facts to
     Spartan.
   - Status update: the non-preprocessed Spartan verifier path is wired through
     the HyperKZG opening boundary. `WrapperVerifierKey` now carries the public
     `ConstraintMatrices<F>` relation rather than loose dimension integers,
     and `r1cs_relation` derives the canonical relation dimensions from that
     matrix object. `SpartanProof` carries the clear compressed outer sumcheck,
     explicit outer evaluation claims `A(rx), B(rx), C(rx)`, the clear
     compressed inner sumcheck, and the witness opening claim `Z(ry)`.
     The Spartan stage derives `tau`, verifies the degree-3 outer sumcheck,
     checks `final_claim == eq(tau, rx) * (A(rx) * B(rx) - C(rx))`, absorbs the
     outer evaluation claims, samples inner batching coefficients, verifies the
     degree-2 inner sumcheck, evaluates `A~(rx, ry), B~(rx, ry), C~(rx, ry)`
     directly from the public R1CS matrices, and checks the inner final claim
     against the claimed `Z(ry)`.
     `A(rx), B(rx), C(rx)` and `Z(ry)` are prover claims, not trusted
     verifier-computed values: the verifier discharges the former through the
     inner sumcheck and public matrix MLE evaluations, and must discharge the
     latter through the HyperKZG opening check before accepting.
     Integration tests assert that a matching zero-matrix relation plus valid
     Spartan transcript reaches the HyperKZG check, while relation mismatches,
     unpaddable dimensions, malformed outer/inner round counts, Spartan
     degree-bound violations, outer-claim mismatches, outer claims that satisfy
     only the outer product relation, and `tau` transcript ordering regressions
     are caught before HyperKZG.
   - Status update: the wrapper verifier now calls the real HyperKZG opening
     verifier. The witness commitment is transcript-bound immediately after
     the public wrapper statement and before Spartan samples `tau`; the
     HyperKZG stage binds `ry` and the claimed `Z(ry)` into the same transcript
     and verifies the committed witness polynomial opening against the
     configured verifier setup. `WrapperVerifierConfig` is now parameterized by
     the pairing group and carries a typed `WrapperVerifierKey`, while
     `WrapperVerifierInputs` is reduced to per-proof public inputs.
   - Status update: the wrapper statement binding now includes the full public
     R1CS matrices, not only their dimensions. Matrix rows are absorbed in the
     canonical A/B/C order with explicit row lengths and column indices before
     public inputs are bound. This makes the verifier transcript commit to the
     exact relation being proven.
   - Status update: the ZK wrapper verifier path is wired for the arbitrary
     R1CS case. `WrapperZkProof` carries committed Spartan outer/inner
     sumchecks, a ZK HyperKZG opening proof, and a native BlindFold proof.
     Spartan verifies committed sumcheck consistency and committed output-row
     shapes without revealing `A(rx), B(rx), C(rx)`, or `Z(ry)`. HyperKZG
     verifies the hidden witness opening and binds the hidden evaluation
     commitment into the transcript. The wrapper BlindFold statement then
     enforces the two hidden Spartan claim equations and the final hidden
     `Z(ry)` equality against the HyperKZG evaluation commitment. The verifier
     uses one continuous transcript across all three stages.

16. Add configured-verifier wrapper fixtures.
   - Keep fixture relation construction in `r1cs_builder.rs` and integration
     tests until the production `jolt-wrapper-prover` exists.
   - Prove small composed R1CS protocols end to end in tests.
   - Review gate: mutating transcript challenges, public inputs, sumcheck
     claims, opening values, Spartan messages, or HyperKZG openings causes
     wrapper verification failure.
   - Status: added the first manifest-style R1CS protocol harness in
     `jolt-wrapper-verifier` integration tests. It covers Poseidon-over-`Fr`
     transcript replay, constrained byte absorption, public input layout,
     `Fr -> Fq` challenge injection, non-native `Fq` arithmetic, native and
     non-native sumcheck constraints, same-point opening reduction, and
     Grumpkin/Pedersen Hyrax opening verification. The current oracle is
     `verify_r1cs_witness`; once the Spartan + HyperKZG verifier path exists,
     these same protocol cases should be proven and verified with the wrapper
     proof path.
   - Status update: added the gold-star wrapper mini protocol e2e fixture. The
     fixture constructs an R1CS relation containing a Poseidon-over-`Fr`
     transcript with scalar and byte absorption, native sumcheck constraints,
     and a Grumpkin/Pedersen Hyrax opening over injected `Fq` values. A
     test-gated prover commits the full witness polynomial with HyperKZG,
     proves the Spartan outer and inner sumchecks, opens `Z(ry)`, and feeds the
     native wrapper verifier. The prover uses an optimized sum-of-products
     sumcheck helper: the outer Spartan polynomial is represented as
     `eq_tau * A * B - eq_tau * C`, and the inner polynomial as
     `combined_matrix * witness`, avoiding repeated sparse matrix evaluation
     during proof generation.
   - Status update: the mini protocol fixture is cached as a deterministic
     artifact under the system temp directory, with regeneration controlled by
     `JOLT_WRAPPER_REGENERATE_FIXTURES`. This mirrors the verifier fixture
     pattern: the expensive honest proof/setup generation happens once, while
     completeness and tampering reuse the cached proof. Tampering coverage
     mutates public inputs, representative R1CS matrix coefficients, each
     relation-statement dimension, the witness commitment, every Spartan outer
     and inner round polynomial, each outer evaluation claim scalar, the
     witness opening claim, every HyperKZG fold commitment, edge indices in
     each clear HyperKZG evaluation vector, and every HyperKZG witness
     commitment element. The fixture cache key is versioned with the transcript
     binding format, so relation-binding changes force regeneration instead of
     silently reusing stale proofs.
   - Status update: added a focused public-input binding regression. It mutates
     the committed witness slot for a public input while leaving the public
     input vector unchanged, then checks the raw R1CS relation rejects. This
     locks in the invariant that public-input consistency is enforced by
     constraints, not only by wrapper-side layout bookkeeping.
   - Status update: added the ZK mini protocol path with a test-gated prover
     that produces committed Spartan sumchecks, a ZK HyperKZG opening proof, and
     a BlindFold proof over the hidden wrapper equations. Completeness verifies
     the honest ZK wrapper proof through `verify_zk`. Soundness tampering starts
     from the real generated ZK proof and mutates public inputs, relation
     coefficients and dimensions, the witness commitment, committed Spartan
     round commitments, committed output-claim row commitments, ZK HyperKZG fold
     and hidden-evaluation commitments, and BlindFold folding, opening, and
     embedded sumcheck payloads. Each registered mutation is required to reject.
     ZK fixtures are cached under the same system-temp fixture directory with a
     separate versioned magic and per-seed file names, so the completeness,
     soundness, and statistical tracks reuse deterministic proof artifacts while
     still regenerating when `JOLT_WRAPPER_REGENERATE_FIXTURES` is set.
   - Status update: added explicit ZK setup-compatibility hardening. Tests now
     reject existing proofs under mutated VC message generators, VC blinding
     generator, and VC capacity. A separate fixture generates committed
     Spartan rows with the expected VC setup while using a different HyperKZG
     hidden-evaluation basis, then checks that the final BlindFold binding
     rejects the proof instead of accepting a hidden `Z(ry)` commitment that is
     not openable under the wrapper VC setup.
   - Status update: added an empirical independence check for the ZK wrapper
     proof. It regenerates multiple proofs for the same public relation shape,
     verifies each proof, and projects the witness commitment, hidden HyperKZG
     evaluation commitment, BlindFold folding scalar, auxiliary commitment, and
     witness opening into distribution and pairwise-correlation checks. The
     default sample count is 32 and can be increased with
     `JOLT_WRAPPER_ZK_STATISTICAL_SAMPLES`; the debug-mode run is intentionally
     slow, so use release mode for this check during review.

17. Add field-inline wrapper hooks.
   - Include FR stages when the configured verifier includes field inline.
   - Review gate: FR-off and FR-on configs produce distinct deterministic
     R1CS shapes.

18. Add Dory-assist verifier hook integration.
   - Compile the selected Dory-assist `PcsProofAssist` verifier path rather
     than the ordinary Dory stage-8 verifier path.
   - Treat the `jolt-dory-assist-verifier` crate as the owner of Dory-assist
     stage ordering, native verification, R1CS hook, and final acceptance
     condition.
   - Pass the typed reduced opening statement from wrapper stage-8 assembly
     into the selected Dory-assist verifier hook.
   - Use only the generic R1CS building blocks above for transcript, Fq
     arithmetic, sumcheck, claim lowering, openings, Hyrax, and Grumpkin
     commitments.
   - Review gate: Dory-assist configured computation produces satisfied R1CS;
     tampering each Dory-assist stage payload rejects without adding protocol
     math to `jolt-wrapper-verifier`.
