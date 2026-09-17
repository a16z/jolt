# U1: An algebra boundary for reusable sumcheck

Status: implemented on `omid/sumcheck-boundry`.

Main baseline: `6b234c98ae` (`feat(sumcheck): support standalone use
(#1865)`), the squash merge of `omid/refactor-sumcheck-crate` plus its review
fixes. Working branch: `omid/sumcheck-boundry`, rebased directly onto that
main commit with only this specification carried over.

## 1. Decision and scope

Use the existing `jolt_field::Field` trait for sumcheck arithmetic and its
polynomial containers. Require transcript absorption only at clear transcript
adapters, serialization only at serialization implementations, and optimized
accumulation only at code that actually consumes an accumulator.

Keep Jolt's `JoltField` bundle and its optimized prover kernels. A Jolt caller
already satisfying that bundle will satisfy the relaxed arithmetic bounds.
Preserve the current proving loop, scheduler, recorder, proof representations,
transcript labels, and BN254 bytes.

The deliverable is a reusable **clear sumcheck reduction**, including the
existing batched prover and verifier. It is not a generic replacement for all
of Jolt, Dory, Pedersen, or BlindFold. Extension fields do not automatically
become supported commitment scalars. No new backend framework or sumcheck
algorithm is proposed.

## 2. Architectural context

Sumcheck reduces a claim that a polynomial sums to a known value on a Boolean
hypercube to a claim about one evaluation at a random point. In each round the
prover supplies a univariate polynomial `s`. The verifier checks
`s(0) + s(1) == running_claim`, draws a challenge `r`, and replaces the claim
with `s(r)`. After all rounds, the application must check the final evaluation
against its polynomial or commitment scheme. Sumcheck alone does not discharge
that final obligation.

The relevant crates have separate jobs:

| Component | Responsibility |
| --- | --- |
| `jolt-field` | Arithmetic traits, concrete fields, optional encoding and accumulation capabilities |
| `jolt-poly` | Polynomial representations, evaluation, compression, interpolation |
| `jolt-sumcheck` | Claims, round scheduling, batched proving, reduction verification, proof containers |
| `jolt-transcript` | Message absorption and deterministic Fiat–Shamir challenges |
| Jolt prover kernels | Compute individual round polynomials efficiently from witness data |
| Commitment / BlindFold stack | Authenticate final evaluations or prove hidden sumcheck relations |

`ProveRounds` is the interface between a caller's polynomial computation and
the shared proving engine. `RoundScheduler` controls how members are visited.
`SumcheckRecorder` controls what reaches the transcript and proof: the clear
recorder sends coefficients; the committed recorder sends commitments.
These existing interfaces are sufficient for this change.

Compression is arithmetic, not byte encoding. For
`s(X) = c0 + c1 X + ... + cd X^d`, the claim gives
`c1 = running_claim - 2*c0 - c2 - ... - cd`. A compressed polynomial stores
`[c0, c2, ..., cd]`. Computing this representation does not require Serde,
canonical bytes, or a deferred-reduction accumulator.

## 3. Review findings in merged main

The standalone-use merge established the dependency boundary, empty default
features, public clear prover example, and isolated external-consumer fixture.
It did not narrow the scalar capability boundary described by U1. The review's
central diagnosis therefore remains correct, with two additional details that
matter to implementation.

1. [The field traits](../crates/jolt-field/src/algebra.rs) already separate
   `Field`, `CanonicalBytes`, `CanonicalEncoding`, and `WithAccumulator`.
   `JoltField` bundles `Field + CanonicalEncoding + WithAccumulator + Serialize
   + DeserializeOwned + MaybeAllocative`. We should use this existing split.
2. [SumcheckScalar](../crates/jolt-sumcheck/src/scalar.rs) includes canonical
   encoding and repeats bounds already implied by `Field`. Its blanket impl
   also repeats several predicates. Removing encoding here is necessary but
   insufficient.
3. [UnivariatePoly](../crates/jolt-poly/src/univariate.rs),
   [CompressedPoly](../crates/jolt-poly/src/compressed_univariate.rs), and
   [interpolation helpers](../crates/jolt-poly/src/lagrange.rs) require
   `JoltField` for arithmetic. The inspected implementations do not use
   `WithAccumulator`. Interpolation is a transitive dependency that must also
   be relaxed; changing only the polynomial struct declaration will not work.
4. [The prover](../crates/jolt-sumcheck/src/prover.rs),
   [proof containers](../crates/jolt-sumcheck/src/proof.rs),
   [batch validation](../crates/jolt-sumcheck/src/batch.rs), and
   [clear recorder](../crates/jolt-sumcheck/src/recorder.rs) propagate the
   broad bound. Compressed verification also requires it, while generic full
   verification uses `SumcheckScalar`.
5. [Transcript](../crates/jolt-transcript/src/legacy.rs) independently declares
   `type Challenge: CanonicalEncoding`. This prevents even a caller-supplied
   transcript from returning an algebra-only extension field. Its
   `challenge_scalar_powers` helper also requires `JoltField`, although it
   only needs ring multiplication and one.
6. The transcript already has a narrower absorption abstraction:
   `AppendToTranscript`, blanket-implemented for `CanonicalBytes`. Absorbing a
   value does not require decoding it, converting it to an integer, or
   defining the field's challenge-byte reduction.
7. [The Solinas extensions](../crates/jolt-field/src/solinas/ext.rs) implement
   arithmetic and conditional Serde but do not implement `CanonicalBytes`,
   `CanonicalEncoding`, or `WithAccumulator`. The separate unreduced
   interfaces do not imply `WithAccumulator`.
8. [The Mersenne61 test](../crates/jolt-sumcheck/tests/mersenne61_compat.rs)
   exercises a custom verifier round, and its field implements canonical
   encoding and an accumulator. It does not prove that the stock prover and
   polynomial containers accept a field lacking those capabilities.
9. [The isolated consumer](../crates/jolt-sumcheck/tests/external-consumer/Cargo.toml)
   compiles the public example as an integration test outside workspace
   feature unification. The example uses Jolt's own `Prime64Offset59` and a
   stateful but fixed-challenge transcript. It checks package isolation and
   prover/verifier scheduling, but does not yet test a field defined outside
   `jolt-field` or extension-field challenge generation.

The pasted review did not include resolved targets for `[field]` or
`[extensions]`, or the contents of `akita-jolt-sumcheck-migration-review.md`.
This spec verifies merged main's local code, not Akita's exact modulus aliases,
extension configuration, or transcript. The test choices below use concrete
local field implementations matching the requested degrees and base widths.
They must not be described as proof of complete Akita compatibility.

## 4. Proposed capability boundaries

| Surface | Proposed bound |
| --- | --- |
| Polynomial storage and arithmetic, including compression | `F: Field` |
| Interpolation and division | `F: Field`, with existing domain preconditions |
| `SumcheckScalar` | Blanket marker over `Field` only |
| `ProveRounds`, scheduler handles, batch validation, prover engine | `F: Field` / equivalent `SumcheckScalar` |
| Proof containers and recorded results | `F: Field`, no encoding or Serde prerequisite |
| `SumcheckRecorder` interface | `F: Field` |
| Stock clear recorder and built-in clear round transcript implementations | `F: Field + AppendToTranscript` |
| Full verifier with caller-provided round messages | `F: SumcheckScalar`; round type supplies its transcript behavior |
| Stock full-proof / compressed-proof verification conveniences | `F: Field + AppendToTranscript` |
| `Transcript::Challenge` | No associated-type capability bound |
| `Transcript::challenge_scalar_powers` | `Self::Challenge: Ring` |
| Existing digest/sponge transcript implementations | Keep `CanonicalEncoding` where challenge derivation uses it |
| Serde implementations | `F: Serialize` / `F: Deserialize<'de>` and corresponding commitment bounds |
| Actual accumulator kernels | Keep `WithAccumulator`, or existing `JoltField` where other bundled capabilities are used |
| Commitment builders / R1CS adapters | Retain backend-required bounds; do not propagate them into clear containers |

Here, “minimal” means removing unrelated capabilities from the existing
practical algebra interface. `Field` itself still includes sampling and the
`Ring` requirements for formatting, hashing, copying, and thread safety. This
proposal does not redesign the field trait hierarchy or create an additional
ladder of arithmetic traits. Some individual operations could work over a
ring, but uniform `Field` bounds keep this change small and easy to use.

### Transcript separation

Relaxing `Transcript::Challenge` allows an external implementation to return
its own field type. It does not change how any existing Jolt transcript
samples challenges. Generic consumers that actually call canonical-encoding
methods on a challenge must state that bound locally after this relaxation.

Use `AppendToTranscript` on clear encoding implementations, rather than
introducing a new codec generic parameter on every sumcheck API. A field
owned by an external crate can implement this trait itself. An external user
working with another crate's field may need a local newtype because Rust does
not allow implementing a foreign trait for a foreign type. Existing custom
round/recorder interfaces remain available for application-specific formats.

Keep `RoundMessage` and `ClearRound` signatures and their inheritance intact.
They require message behavior, not a particular field codec. Their built-in
polynomial implementations gain `AppendToTranscript` bounds; polynomial
evaluation, compression, and round generation remain available without them.
The generic verifier needs an absorbable round message because it performs
Fiat–Shamir, whereas a polynomial by itself need not be absorbable.

### A small byte adapter for the concrete extensions

To make the *actual*, unwrapped local quadratic and quartic fields usable with
the stock clear recorder, add `CanonicalBytes` implementations to `FpExt2`
and `FpExt4`, conditional on canonical bytes for their base fields. Do not add
`CanonicalEncoding` or `WithAccumulator` merely to satisfy sumcheck.

Define the little-endian payload as the concatenation of fixed-width base
coefficient encodings in the existing basis order: `[c0, c1]` for quadratic,
and `[c0, c1, c2, c3]` for quartic. `NUM_BYTES` is the extension degree times
the base width. The coefficient tuple uniquely identifies the element in the
configured basis. The protocol must separately fix the field and basis; the
payload does not identify them.

The existing `AppendToTranscript` blanket implementation reverses the entire
payload. Consequently the absorbed extension byte order is the reverse of
this concatenation, not merely each coefficient independently reversed.
Document and pin both forms with byte tests. Do not change this existing
blanket implementation or claim these new bytes match Akita's format.

This is an absorption adapter, not a full field codec. Built-in
`Blake2bTranscript<Extension>` / `KeccakTranscript<Extension>` still need their
existing `CanonicalEncoding` challenge contract. For this scope, extension
integration tests use an explicitly supplied transcript returning extension
challenges. A production Akita integration supplies its own agreed transcript
and challenge sampler. No production extension challenge mapping is invented
as part of U1.

### Serde and allocation profiling

Keep derived Serde implementations conditional. Remove `DeserializeOwned`
or higher-ranked deserialization requirements where `Deserialize<'de>` is
sufficient for the owned coefficient vectors. Preserve field order, enum
order, and wire representation. Do not add a new serialization feature or
use Serde as the definition of transcript bytes.

If an affected type derives `Allocative`, its profiling implementation must
require the necessary element bound locally. Do not reintroduce
`MaybeAllocative` into `Field` or sumcheck's algebra marker to make a derive
compile. Jolt's existing `JoltField` callers retain their profiling support.

### Optimized arithmetic and optional features

The batch engine combines the small round coefficient vectors using normal
field arithmetic; it does not require deferred reduction. Heavy witness
kernels are behind `ProveRounds` implementations, which can keep their
specialized requirements. Do not replace their optimized accumulators with
naive arithmetic, and do not provide a blanket `WithAccumulator` impl that
would conflict with specialized implementations.

Pure committed-proof consistency data and witness containers should not
require a commitment-capable field merely to exist. Actual vector commitment
operations and R1CS adapters can retain the bounds required by their current
dependencies. U1 does not promise Pedersen/BlindFold proving for the Solinas
extensions.

## 5. Resulting code shape

These excerpts show the implemented capability boundary. Method bodies and
public argument order remain unchanged except where a bound moved to the
method that uses it.

```rust
// jolt-sumcheck/src/scalar.rs
use jolt_field::Field;

pub trait SumcheckScalar: Field {}
impl<F: Field> SumcheckScalar for F {}
```

```rust
// jolt-poly/src/univariate.rs; CompressedPoly follows the same pattern.
use jolt_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct UnivariatePoly<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> UnivariatePoly<F> {
    // Existing new/evaluate/compress/interpolate/divide methods.
}
```

```rust
// Transcript retains its existing methods and supertraits.
pub trait Transcript: Default + Sync + Send + 'static {
    type Challenge;
    // ... existing methods ...
    fn challenge_scalar_powers(&mut self, len: usize) -> Vec<Self::Challenge>
    where
        Self::Challenge: Ring;
}

// Concrete digest/sponge implementations retain F: CanonicalEncoding.
// The stock clear recorder requires only the absorption capability it uses.
impl<F: Field + AppendToTranscript, C> SumcheckRecorder<F>
    for ClearSumcheckRecorder<F, C>
{
    // Existing input claims, compressed rounds, and output claims.
}
```

```rust
pub trait ProveRounds<F: Field> {
    fn num_rounds(&self) -> usize;
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>>;
}

pub fn prove_batch<F, R, T>(
    prelude: &BatchPrelude<F>,
    members: &mut [&mut dyn ProveRounds<F>],
    scheduler: &mut dyn RoundScheduler<F>,
    recorder: &mut R,
    transcript: &mut T,
) -> Result<ProvedBatch<F>, SumcheckError<F>>
where
    F: Field,
    R: SumcheckRecorder<F>,
    T: Transcript<Challenge = F>;
```

`prove_batch` itself has no absorption bound on `F`: the recorder owns that
responsibility. `prove_uniskip_clear` does absorb scalar coefficients and an
output claim directly, so it gains `F: Field + AppendToTranscript`.

For external callers the normal sequence remains: implement `ProveRounds`,
construct `BatchPrelude`, use `ClearSumcheckRecorder`, call `prove_batch`,
finish recording, run the corresponding verifier transcript, and check the
returned evaluation against the application polynomial. No new builder or
mode selection is required. Jolt's existing callers keep this same sequence.

## 6. The review's last bullet: what to test and why

Keep the intent of the bullet, but make it precise. These are two distinct
regression risks, not a requirement to bring the Akita application into this
repository.

### A. Actual extension arithmetic through prover and verifier

Use the real local types `Ext2<Prime64Offset59>` and
`FpExt4<Prime32Offset99>`, plus `Prime128Offset275` as the direct prime-field
case. Confirm the applicable existing non-residue/basis configuration in
`jolt-field`; use its arithmetic rather than defining another extension.
If Akita's concrete choices differ, these are representative local tests,
not a claim that the exact Akita types were tested.

Run one shared small, multi-round polynomial fixture through:

1. `ProveRounds`, `SequentialRounds`, and the real `prove_batch` engine.
2. The stock clear recorder, actual full/compressed polynomial containers,
   and `SumcheckVerifier` (compressed Boolean verification and a full clear
   round path, including a centered-domain uni-skip case).
3. The final evaluation check against a separately specified polynomial.

For example, specify a product of two affine multivariate polynomials, compute
its initial claim by explicitly summing the few Boolean points, and compute
its final value by direct substitution. Neither oracle copies the round
generator. Use nonzero higher extension coefficients in both polynomial data
and challenges; values embedded only from the base field would miss the
extension-specific behavior. Include values that exercise every quartic
basis coordinate across the fixture.

The test transcript must absorb messages and deterministically derive
challenges from them, with separately initialized prover/verifier states.
A stateful test-only adapter may mix the absorbed bytes into a deterministic
state and construct each extension challenge with nonzero values in every
basis coordinate. Pin its coordinate order and compare final transcript states
after the same output-claim absorption. The existing example's transcript
tracks absorbed bytes but always returns the same challenge; do not use it to
claim challenge-generation correctness or cryptographic soundness. This test
adapter is not a published secure extension-transcript implementation.

Tamper with a higher-coordinate coefficient and require failure of the
complete verification procedure. For compressed proofs, round consistency is
reconstructed from the claim, so rejection may occur at the application's
final evaluation check rather than inside `verify_compressed`. Also exercise
the existing wrong-round-count and excess-degree error paths with an extension
field where they add coverage, without duplicating the entire soundness suite.

Retain polynomial compression/evaluation and interpolation checks over these
fields, and conditional Serde round trips using the actual extension Serde
implementations. Pin the newly defined coefficient byte ordering separately.

### B. A field defined outside the library

Extend the existing isolated consumer fixture with a locally defined small
prime field (a local wrapper around existing prime arithmetic is sufficient).
Implement the public algebra traits and `AppendToTranscript`, and provide a
local transcript. Deliberately omit `CanonicalBytes`, `CanonicalEncoding`,
`WithAccumulator`, Serde, and allocation profiling from this type.

Run the stock prover, clear recorder, proof containers, and compressed verifier
with this field, then independently check the final evaluation. Also instantiate
polynomial arithmetic and round generation through a helper whose only field
bound is `Field`; this catches an encoding bound leaking into pure arithmetic.
Use proper field sampling if the wrapper implements `Field::random`; do not
copy the test-only Mersenne61 reduction sampler as an exact-uniform sampler.

This is a separate Cargo package with its own `[workspace]`, so Jolt's dev
dependencies and workspace feature unification cannot silently supply missing
capabilities. The local wrapper counts as an external field type for Rust's
trait/coherence boundary. It does not prove compatibility with every third-party
field library; those libraries can require a newtype adapter.

An integration test under `tests/` already compiles as a separate Rust crate,
but the isolated package adds the dependency/feature isolation that matters
here. Reuse [check-external-sumcheck.sh](../scripts/check-external-sumcheck.sh)
and keep its minimal dependency checks. Do not add an unrelated third-party
field dependency just to make the word “external” literal.

The test field is a small API-compatibility fixture, not a production security
recommendation. Preserve the useful existing Mersenne61 hash-transcript test;
avoid duplicating its handwritten arithmetic or adding old-versus-new oracles.

## 7. Implementation sequence

1. Record baseline focused tests and available polynomial benchmarks. Audit
   uses of `Transcript::Challenge` that rely on implied canonical methods.
2. Relax the univariate, compressed, and required Lagrange helper bounds in
   `jolt-poly`. Keep formulas and evaluation order unchanged. Make Serde bounds
   conditional at their impls.
3. Relax `Transcript::Challenge` and the powers helper. Keep encoding bounds
   on concrete transcript implementations; add explicit bounds only at generic
   consumers that truly use them.
4. Simplify `SumcheckScalar`, then migrate the clear sumcheck path end to end:
   polynomial round implementations, batch validation, scheduler interfaces,
   recorder interface, containers, prover, and verifier. Split mixed impl
   blocks so convenience verification bounds do not constrain construction.
5. Inspect committed witness/consistency containers for inherited broad bounds.
   Leave backend-bound builders and R1CS adapters intact where required.
   Compile Jolt consumers without changing protocol expressions or generated
   stage behavior.
6. Add the two narrow extension byte implementations and their byte vectors.
   Add the extension integration fixture and extend the isolated external
   consumer. Update README/example guidance with the capability boundary and
   final-evaluation obligation.
7. Run the validation matrix below. Compare existing Jolt fixtures/transcript
   results, inspect the final diff for unintended protocol or kernel changes,
   and remove temporary diagnostics before handoff.

Expected production edits are limited to `jolt-poly`'s univariate/compressed/
Lagrange modules, `jolt-sumcheck`, the transcript facade and necessary bound
propagation, and the two extension absorption implementations in `jolt-field`.
Any wider algorithm change requires a separate proposal.

## 8. Validation and acceptance criteria

Focused validation (use `cargo nextest`, never `cargo test`):

```bash
cargo nextest run -p jolt-poly -p jolt-transcript -p jolt-sumcheck --cargo-quiet
cargo nextest run -p jolt-sumcheck --features committed,r1cs,parallel --cargo-quiet
cargo nextest run --manifest-path crates/jolt-sumcheck/tests/external-consumer/Cargo.toml --locked
bash scripts/check-external-sumcheck.sh
```

Run the new extension byte tests in `jolt-field` with `solinas`, and the relevant
allocation profiling checks with `allocative`. Confirm package feature names
and test targets when implementing; explicitly enable transcript backends for
backend-specific tests. If the external fixture gains dependencies, update its
lockfile before using the script's `--locked --offline` check. Wire its check
and extension tests into the appropriate CI job if not already exercised.

Required Jolt regression validation:

```bash
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo fmt -q
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-prover --features prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,zk --cargo-quiet
```

Use the existing living-reference byte-parity suites and transcript fixtures;
do not retain a copy of pre-change production code as a test oracle. Compare
available univariate/Lagrange polynomial benchmark cases before and after
because shared polynomial code is changing. No performance optimization is
intended; unexpected regressions require investigation before acceptance.

Acceptance requires all of the following:

- Pure polynomial and round-generation APIs compile with an algebra-only
  bound; clear proving does not require the `JoltField` bundle.
- The isolated field runs the public prover and verifier without canonical
  encoding, optimized accumulation, Serde, or profiling implementations.
- Actual quadratic/quartic types exercise higher coordinates through both
  sides, with matching transcript replay and independent final evaluation.
- Conditional extension serialization and the new absorption byte vectors
  pass; existing Jolt serialized proofs and transcript bytes stay unchanged.
- Jolt's standard and ZK acceptance suites pass, and optimized kernels retain
  their current arithmetic and capability bounds.

## 9. Limits and risks to preserve in documentation

Field-generic is not the same as working for every field characteristic or
domain. The current batched prover requires invertible two and already reports
`TwoNotInvertible`. Integer-grid interpolation requires distinct embedded
nodes and invertible denominators; its existing helpers can panic when these
preconditions fail. Extension fields retain the characteristic of their base
field, so a larger extension does not repair colliding integer-domain nodes.
Do not advertise arbitrary-characteristic or fully fallible interpolation
support as an outcome of this change.

Relaxing a trait's associated-type bound can expose downstream source code
that relied on it implicitly. Such callers need a local encoding bound if
they use encoding methods; this is a source-compatibility consideration even
though all existing concrete field implementations remain usable.

The largest security-sensitive risk is accidental transcript drift. Preserve
labels, counts, coefficient order, compression convention, challenge calls,
and claim absorption order. The new extension payload needs an explicit basis
contract, and a production external transcript needs a reviewed challenge
distribution over the intended field, not just its prime subfield.

No sumcheck input/output formula changes are planned. Jolt's BlindFold
claim/constraint synchronization remains unchanged. If implementation reveals
a required formula change, stop and revise the scope rather than folding it
into this bound refactor.

## 10. Implementation status

The algebra, transcript, proof-container, and clear prover/verifier bounds are
implemented. Quadratic and quartic coefficient encodings are pinned by golden
tests. The integration suite covers the direct 128-bit field, the actual
quadratic and quartic extensions, transcript replay, conditional proof Serde,
centered-domain uni-skip, and a higher-coordinate tamper. The isolated consumer
defines its own field wrapper and runs the stock clear prover and verifier
without implementing canonical encoding, optimized accumulation, Serde, or
allocation profiling. Existing verifier error tests continue to cover malformed
round counts and degree bounds without duplicating them for each scalar type.
