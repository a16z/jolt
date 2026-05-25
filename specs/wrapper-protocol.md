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
  Spartan + HyperKZG backend field
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

The wrapper backend receives only `ConstraintMatrices<Fr>`, `witness Vec<Fr>`,
and `public_inputs Vec<Fr>`. It must not know which entries encode native `Fr`
values, non-native `Fq` limbs, Grumpkin coordinates, transcript state, or Jolt
proof data.

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
```

Wrapper implementation should be foundation-first. Before adding bespoke
Dory-assist verifier R1CS wiring, each reusable component should expose a
generic, crate-owned R1CS interface with local tests and at least one
cross-crate composition test. The final configured-verifier assembly should be
mostly sequencing typed component hooks, not reimplementing component math in
`jolt-wrapper`.

The safe implementation rule is: when a wrapper check feels Dory-assist-specific
but is really a generic algebraic, transcript, sumcheck, opening, or group
constraint, stop and upstream that piece to the crate that owns the abstraction.
Only after the generic component has local soundness tests and a composed R1CS
test should the configured verifier call it.

`jolt-wrapper::r1cs::assembly` sequences these helpers according to the
configured verifier flow exported by `jolt-verifier`. It should not implement
component math directly. It allocates the proof and public-input witness,
threads transcript state, records aliases and claim sources, and calls the R1CS
hooks owned by the component crates.

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
final acceptance condition for the selected assist proof. `jolt-wrapper` should
not match on Dory-assist operation families or copy-edge internals. It receives
a typed reduced opening statement from the base Jolt stage-8 assembly, passes it
to the selected Dory-assist verifier hook, and constrains the returned
acceptance condition as part of the configured verifier relation.

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
itself should not be specialized to that backend.

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
pub struct WrapperAssemblyInputs<F, PCS, VC, ZkProof, PcsAssist>
where
    PcsAssist: PcsProofAssist<PCS>,
{
    pub preprocessing: JoltVerifierPreprocessing<PCS, VC>,
    pub public_io: JoltDevice,
    pub proof: JoltProof<PCS, VC, ZkProof, PcsAssist>,
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
    PcsAssist,
>(
    inputs: WrapperAssemblyInputs<F, PCS, VC, ZkProof, PcsAssist>,
) -> Result<WrapperR1csInstance<F>, WrapperError>
where
    PcsAssist: PcsProofAssist<PCS>,
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

6. Add wrapper protocol builder skeleton.
   - Implement `WrapperProtocolBuilder`, `WrapperClaimSources`, and configured
     stage hook traits over `R1csBuilder`.
   - Review gate: fixture stage lowers to satisfied R1CS without becoming part
     of the public wrapper API.

7. Add wrapper instance export.
   - Track stable public-input ordering.
   - Export matrices, witness, and publics for the backend.
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

14. Assemble base configured verifier R1CS.
   - Start with a narrow configured verifier computation before full Jolt.
   - Review gate: native verifier and wrapper R1CS agree on accept/reject for
     the same fixture.

15. Add Dory-assist verifier hook integration.
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
   - Review gate: Dory-assist configured computation produces satisfied
     R1CS; tampering each Dory-assist stage payload rejects without adding
     protocol math to `jolt-wrapper`.

16. Add field-inline wrapper hooks.
   - Include FR stages when the configured verifier includes field inline.
   - Review gate: FR-off and FR-on configs produce distinct deterministic
     R1CS shapes.

17. Implement ZK `snark_backends/spartan_hyperkzg`.
   - Consume arbitrary `WrapperR1csInstance`.
   - Keep the backend independent from Jolt protocol types.
   - Hide the R1CS witness, including transparent inner proof data.
   - Review gate: arbitrary R1CS proof verifies and witness
     randomization changes proof bytes for the same public statement.

18. Add configured-verifier wrapper fixture.
   - Prove a small transparent configured verifier computation end to end with
     the transparent proof held as private wrapper witness.
   - Review gate: mutating transcript challenges, public inputs, sumcheck
     claims, or opening values causes wrapper verification failure.
