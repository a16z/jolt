# Spec: `jolt-openings` Crate API Cutover

| Field       | Value        |
|-------------|--------------|
| Author(s)   | @quangvdao   |
| Created     | 2026-05-12   |
| Status      | active       |
| PR          | [#1521](https://github.com/a16z/jolt/pull/1521) |

## Summary

This PR makes the extracted PCS crates the canonical place for polynomial
commitment and opening APIs, without migrating legacy `jolt-core` in this PR.

The earlier integration experiment cut `jolt-core` over to the new trait family.
That was useful for finding the right abstraction boundary, especially around
streaming commitment and Stage 8 opening fusion. The merge target is narrower:
ship the crate-level API and concrete backend implementations now, while leaving
`jolt-core` unchanged as the end-to-end reference implementation.

The intended consumer for the new API is the upcoming Bolt-generated
prover/verifier split, such as PR [#1514](https://github.com/a16z/jolt/pull/1514).
Generated prover and verifier crates should use these abstractions directly when
they land, rather than porting through the old in-core PCS trait family.

## Intent

### Goals

1. Make `crates/jolt-openings` the backend-neutral PCS/opening API.
2. Make `crates/jolt-dory` implement that API while preserving the existing
   Dory streaming and ZK behavior.
3. Keep `crates/jolt-hyperkzg` aligned with the same base trait family.
4. Provide source-backed commitment and opening abstractions that can express
   current Jolt/Dory performance behavior without baking Dory matrix vocabulary
   into generic traits.
5. Preserve legacy `jolt-core` as the old implementation for reference tests
   and protocol-parity checks.
6. Document the intended generated-prover/generated-verifier integration path.

### Non-Goals

1. No `jolt-core` cutover in this PR.
2. No SDK, example, transpiler, CLI, or extractor churn caused by a legacy
   `jolt-core` migration.
3. No proof-format or transcript change in legacy `jolt-core`.
4. No Akita or Hachi integration.
5. No generic source partition type whose real values are Dory-specific.
6. No permanent compatibility shim around the old in-core PCS traits.

If a future PR makes a protocol-breaking change, it must either be implemented
on both the legacy `jolt-core` reference path and the generated-role path, or be
explicitly staged behind a separate compatibility plan.

## Source Of Truth

PR [#1467](https://github.com/a16z/jolt/pull/1467), branch
`quang/pcs-prover-verifier-split`, is the starting point for the abstract PCS
split. Current `main` remains the source of truth for legacy `jolt-core`
protocol behavior.

Port or adapt from #1467:

1. `crates/jolt-openings/src/schemes.rs`: verifier/prover split and extension
   traits.
2. `crates/jolt-openings/src/sources.rs`: backend-neutral commitment and opening
   source traits.
3. `crates/jolt-openings/src/homomorphic.rs`: ordinary homomorphic batch helper.
4. `crates/jolt-openings/src/claims.rs`: prover/verifier claim and batch-output
   vocabulary.
5. `crates/jolt-openings/src/mock.rs`: test PCS under the split trait family.
6. `crates/jolt-dory/src/scheme.rs`: Dory implementation of the extracted
   traits.
7. Focused tests and benches for the new crate-level API.

Preserve from current `main`:

1. Legacy `jolt-core` PCS implementation and proof flow.
2. Existing end-to-end reference behavior.
3. Current Dory proof hardening and ZK opening semantics.

## Design

### Architecture

After this PR, the extracted crate dependency direction is:

```text
jolt-field
jolt-poly
jolt-transcript
jolt-crypto
    |
    v
jolt-openings
    |
    +--> jolt-dory
    |
    +--> jolt-hyperkzg
    |
    +--> future PCS adapters
```

`jolt-openings` owns the abstract PCS API, source traits, batch-opening
claim/result vocabulary, and generic ordinary homomorphic batch helper.
Concrete PCS crates own backend-specific setup, commitment, proof, transcript,
hint, batching, and ZK details.

Legacy `jolt-core` stays outside this cutover. It remains the old end-to-end
reference implementation until generated prover/verifier crates are ready to
consume the extracted API directly.

### Trait Layers

`jolt-openings` defines a verifier-first trait hierarchy:

```rust
pub trait CommitmentSchemeVerifier: Commitment + Clone + Send + Sync + 'static {
    type Field: Field;
    type Proof;
    type BatchProof;
    type VerifierSetup;

    fn verify(...);
    fn verify_batch(...);
    fn bind_opening_inputs(...);
}

pub trait CommitmentScheme: CommitmentSchemeVerifier {
    type ProverSetup;
    type OpeningHint;
    type SetupParams;

    fn setup(...);
    fn project_verifier_setup(...);
    fn commit<S: CommitmentSource<Self::Field> + ?Sized>(...);
    fn commit_batch<B: BatchCommitmentSource<Self::Field>>(...);
    fn open<S: CommitmentSource<Self::Field> + ?Sized>(...);
    fn prove_batch<S: CommitmentSource<Self::Field>>(...);
}
```

Single-claim `open` and `verify` belong on the base PCS traits because they are
semantic PCS operations, not homomorphic-only operations.

Verifier setup construction is split out:

```rust
pub trait PublicVerifierSetup: CommitmentSchemeVerifier {
    type PublicParams;

    fn verifier_setup(params: Self::PublicParams) -> Self::VerifierSetup;
}
```

This keeps KZG-style schemes honest: verifier setup may contain
trapdoor-derived elements and cannot always be reconstructed from public
generators alone.

### Source Traversal

`CommitmentSource` describes how a polynomial-like object can be evaluated and
traversed. It does not describe a backend partition.

```rust
pub trait CommitmentSource<F: Field>: Send + Sync {
    fn num_vars(&self) -> usize;
    fn evaluate(&self, point: &[F]) -> F;
    fn natural_chunk_len(&self) -> Option<usize> { None }
    fn for_each_row<V>(&self, chunk_len: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>);
    fn map_rows<R, V>(&self, chunk_len: usize, visit: V) -> Vec<R>
    where
        R: Send,
        V: for<'row> Fn(usize, SourceRow<'row, F>) -> R + Send + Sync;
    fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F>;
}
```

The key point is `natural_chunk_len`. A source may say, “this is the row length I
can stream efficiently.” Dory privately interprets that as its row width and
derives its own matrix split. Another backend may ignore it, use it as a tile
size, or choose a different schedule.

`SourceRow` supports field rows, compact integer rows, strided rows, and one-hot
rows so Dory can preserve the existing fast paths without forcing every caller
to materialize field elements.

### Batch Sources

Commitment batching is a source registry plus a shared traversal hint:

```rust
pub trait BatchCommitmentSource<F: Field> {
    type Id: SourceId;
    type Source<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    fn source(&self, id: Self::Id) -> Self::Source<'_>;
    fn natural_chunk_len(&self, ids: &[Self::Id]) -> Option<usize> { ... }
}
```

Opening batching has two layers:

```rust
pub trait BatchOpeningSource<F: Field, OpeningHint>: Send + Sync {
    type Id: SourceId;
    type Source<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    fn source(&self, id: Self::Id) -> Self::Source<'_>;
    fn opening_hint(&self, id: Self::Id) -> &OpeningHint;
}

pub trait LinearCombinationOpeningSource<F: Field, OpeningHint>:
    BatchOpeningSource<F, OpeningHint>
{
    type LinearCombination<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    fn linear_combination<'a>(
        &'a mut self,
        terms: &[LinearSourceTerm<F, Self::Id>],
    ) -> Self::LinearCombination<'a>;
}
```

The base opening source does not assume linear combination. The linear extension
is the extra capability needed by homomorphic/RLC-style PCS implementations such
as Dory. This leaves room for future schemes whose native batching is
concatenation, quotienting, folding, GPU tiling, or something else.

### Batch Opening Output

Source-backed batch opening returns the proof plus a public relation between the
PCS output and raw protocol claims.

```rust
pub trait LinearOpeningScheme: CommitmentScheme + LinearOpeningSchemeVerifier {
    fn prove_batch_opening<B, ClaimId>(
        terms: Vec<ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>>,
        source_batch: &mut B,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> BatchOpeningProverResult<Self, ClaimId>
    where
        B: LinearCombinationOpeningSource<Self::Field, Self::OpeningHint>;
}
```

In transparent mode, the output value is public. In ZK mode, the output may be a
hiding commitment plus prover-only value/blinding witnesses.

This is the abstraction generated Stage 8 should use: generated code supplies
raw opening terms and a source batch; the PCS owns the fusion challenge schedule,
source fusion, proof construction, and output-relation metadata.

### Ordinary Homomorphic Batching

The ordinary homomorphic batch helper implements the standard group-by-point RLC
protocol for schemes whose commitments and hints can be linearly combined.

Singleton batches are deliberately special: a one-claim batch calls `open` /
`verify` directly and wraps the result in `PCS::BatchProof`. It does not absorb
`rlc_claims` or draw a batch challenge.

For multi-claim batches, the prover and verifier:

1. absorb the claim count under `rlc_claims`;
2. absorb all claimed evaluations in the same order;
3. group claims by opening point;
4. draw one RLC challenge per point group;
5. combine polynomials, commitments, evaluations, and hints with the same
   challenge powers; and
6. produce or verify one `PCS::Proof` per opening-point group.

For Dory this means `PCS::BatchProof = Vec<DoryProof>`, with one proof per
opening-point group.

### Dory

`DoryScheme` implements the extracted traits while keeping Dory-specific details
inside `crates/jolt-dory`:

1. `CommitmentSchemeVerifier`
2. `PublicVerifierSetup`
3. `CommitmentScheme`
4. `AdditivelyHomomorphicVerifier`
5. `AdditivelyHomomorphic`
6. `ZkOpeningSchemeVerifier`
7. `ZkOpeningScheme`
8. `LinearOpeningSchemeVerifier`
9. `LinearOpeningScheme`
10. `ZkLinearOpeningSchemeVerifier`
11. `ZkLinearOpeningScheme`

Dory-specific facts remain private to Dory:

1. `sigma` / `nu`
2. row commitments
3. row-major proof coordinate order
4. transcript adapter details
5. bounded proof deserialization
6. ZK evaluation commitment plumbing

`DoryScheme::BatchProof = Vec<DoryProof>`. The ordinary homomorphic batch helper
returns one `DoryProof` per opening-point group. A singleton batch is therefore a
single-element vector, not a different proof type.

### Bolt / Generated Roles

Generated prover/verifier crates should integrate at the crate boundary, not by
first porting through legacy `jolt-core`.

Generated commitment code should build a `BatchCommitmentSource` from its
witness provider:

```rust
let commitments = PCS::commit_batch(&witness_sources, &source_ids, &prover_setup);
```

Generated Stage 8 code should collect raw opening terms and invoke the linear
opening extension only when the selected PCS advertises that capability:

```rust
let result = PCS::prove_batch_opening(
    opening_terms,
    &mut opening_sources,
    &prover_setup,
    transcript,
);
```

Verifier code should call the matching verifier extension and use the returned
relation to bind the generated protocol's output constraints:

```rust
let public = PCS::verify_batch_opening(
    verifier_terms,
    &proof.joint_opening_proof,
    &verifier_setup,
    transcript,
)?;
```

The generated pipeline owns protocol-specific ids, claim ordering, transcript
placement, and constraint binding. `jolt-openings` owns only the PCS-level source
and proof API.

The abstraction should stay strategy-oblivious above the PCS boundary. Bolt's
oracle buffers map to `CommitmentSource`; oracle families and
`compute.pcs_commit_batch` map to `BatchCommitmentSource` plus
`PCS::commit_batch`; opening obligations map to raw opening terms plus
`BatchOpeningSource`; and RLC-style opening obligations additionally require
`LinearCombinationOpeningSource`.

A `SourceId` names a committed source, not necessarily one logical Jolt
polynomial. Dory can use one source id per logical polynomial. A future packed
scheme can use one source id for a packed witness group and let its adapter route
logical polynomial openings into packed-source points.

### Generated ZK Boundary

ZK openings need more than a verifier call. They also need the hiding commitment
to the opened value and prover-only witness data such as the hidden scalar and
blinding.

The source-backed ZK API returns this as:

1. `BatchOpeningPublic` containing `Hidden(y_com)`-style public outputs;
2. `ZkBatchOpeningWitness` containing prover-only output values and blinds; and
3. a relation describing how the PCS output is derived from raw opening terms.

Generated ZK code should use that relation to bind its proof constraints. The
PCS should not mention BlindFold or any specific proof system by name; it should
only expose the commitment/opening facts the surrounding protocol needs.

Legacy `jolt-core` proof serialization is intentionally unchanged in this PR.
When generated-role proofs consume this API, their opening proof storage should
use the scheme-defined `PCS::BatchProof` rather than assuming that every batch is
one `PCS::Proof`.

## Invariants

1. `jolt-openings` does not depend on `jolt-core`, `jolt-dory`, `dory`,
   `common`, `tracer`, `jolt-sdk`, Akita, or Hachi.
2. Generic traits do not expose Dory's matrix shape, `sigma`, `nu`, or a generic
   associated partition type.
3. Source traversal APIs are allowed to expose backend-neutral facts such as
   `natural_chunk_len`.
4. `StreamingCommitment` is not part of the canonical public API.
5. `BatchOpeningSource` is a source/hint registry. Linear fusion is only on
   `LinearCombinationOpeningSource`.
6. Dory hints carry enough backend-owned information to replay the commitment
   traversal at opening time.
7. Dory source-backed commitment/opening must preserve current streaming
   behavior and avoid materializing full trace-sized field tables on hot paths.
8. Prover and verifier batch helpers bind the same transcript data in the same
   order.
9. Legacy `jolt-core` remains unchanged by this PR.
10. Protocol-specific ids, claim ordering, transcript placement, and constraint
    binding stay outside `jolt-openings`.
11. The PR introduces no Akita dependency.

## Acceptance Criteria

- [x] `crates/jolt-openings/src/schemes.rs` defines the verifier/prover split.
- [x] `PublicVerifierSetup` is separate from the base verifier trait.
- [x] Base traits expose single `open` / `verify`.
- [x] Base traits expose ordinary `prove_batch` / `verify_batch`.
- [x] `crates/jolt-openings/src/sources.rs` defines source and batch-source
      traits with `natural_chunk_len`.
- [x] `crates/jolt-openings/src/sources.rs` separates `BatchOpeningSource` from
      `LinearCombinationOpeningSource`.
- [x] `crates/jolt-openings/src/claims.rs` defines raw batch-opening terms and
      output-relation metadata.
- [x] `crates/jolt-openings/src/homomorphic.rs` provides ordinary homomorphic
      batch helpers.
- [x] `crates/jolt-openings/src/mock.rs` implements the split traits for tests.
- [x] `crates/jolt-dory` implements the split trait family.
- [x] `DoryScheme::BatchProof = Vec<DoryProof>`.
- [x] Dory ordinary batch opening returns one proof per opening-point group.
- [x] Dory source-backed commitment uses source traversal hints rather than a
      public shaped trait.
- [x] Dory source-backed linear opening owns fusion and returns output-relation
      metadata.
- [x] `crates/jolt-hyperkzg` compiles against the same base trait family.
- [x] The final diff contains no `jolt-core` source changes.
- [x] Focused crate tests and clippy pass.

## Testing Strategy

Focused `jolt-openings` tests should validate:

1. single-claim commit/open/verify;
2. multi-claim batches with shared and distinct points;
3. tampered-evaluation rejection;
4. RLC polynomial/scalar consistency;
5. prover/verifier transcript sync for the batch helper;
6. source-backed opening output relations; and
7. ZK source-backed output metadata with prover-only witnesses kept private.

Focused `jolt-dory` tests should validate:

1. transparent and ZK round trips;
2. homomorphic commitment and hint combination;
3. single-claim and multi-claim `prove_batch` / `verify_batch`;
4. source-batch commitment equivalence to direct commitment;
5. proof deserialization hardening;
6. non-default `natural_chunk_len` replay through `DoryHint`; and
7. source-backed transparent and ZK batch-opening output relations.

Focused `jolt-hyperkzg` tests should validate that the base trait family remains
usable by a non-Dory backend, including verifier setup that is not publicly
derivable.

Dependency checks should verify that `jolt-openings` has no dependency on
`jolt-core`, `jolt-dory`, `dory`, `tracer`, or Akita, and that backend crates
depend on `jolt-openings` rather than the reverse.

## Performance

This PR is scoped to crate-level API and backend behavior, but the performance
contract is still important because these traits are intended for generated
provers.

The API must preserve two Dory hot paths:

1. source-batch commitment should allow one scan over source rows while producing
   many Dory commitments; and
2. source-backed linear opening should allow RLC-style fusion without forcing
   callers to materialize all committed polynomials as dense field tables.

The closure-based row visitors are generic over the visitor type. Rust
monomorphizes the concrete closure at the call site, like `Iterator::map` or
Rayon closures; the higher-ranked row lifetime only says the visitor accepts a
short-lived borrowed row. It does not require dynamic dispatch or heap-allocated
trait objects.

Performance-sensitive invariants:

1. keep row order and row encodings stable;
2. keep compact integer and one-hot row paths available;
3. keep Dory row-commitment hints reusable at opening time;
4. keep backend-owned parallelism in commitment, folding, and hint combination;
5. avoid materializing trace-sized field tables on Dory streaming paths; and
6. benchmark generated-role integration against legacy `jolt-core` before
   replacing any end-to-end path.

## Validation

Required before merge:

```bash
cargo fmt -q --check
cargo check -p jolt-openings -q --features test-utils
cargo check -p jolt-dory -q
cargo check -p jolt-hyperkzg -q
cargo nextest run -p jolt-openings --cargo-quiet --features test-utils
cargo nextest run -p jolt-dory --cargo-quiet
cargo nextest run -p jolt-hyperkzg --cargo-quiet
cargo clippy -p jolt-openings -q --features test-utils --all-targets -- -D warnings
cargo clippy -p jolt-dory -q --all-targets -- -D warnings
cargo clippy -p jolt-hyperkzg -q --all-targets -- -D warnings
```

Useful reference checks, but not required by this PR's scope because legacy
`jolt-core` is intentionally untouched:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

## Alternatives Considered

1. **Full legacy `jolt-core` cutover in this PR.**
   Rejected for merge scope. The experiment found useful trait boundaries, but
   the long-term integration target is the generated prover/verifier split, and
   legacy `jolt-core` should stay as a reference implementation for now.

2. **Keep only the old `reduce_prover` / `reduce_verifier` API.**
   Rejected because it leaves batching as external orchestration and forces
   future non-Dory schemes into homomorphic assumptions.

3. **Put single-claim `open` and `verify` only on homomorphic extensions.**
   Rejected because singleton openings are semantic PCS operations.

4. **Make source partitioning an associated type on `CommitmentSource`.**
   Rejected because it would either encode Dory's matrix split generically or
   force a universal partition enum that tries to predict future backends.

5. **Let Stage 8 pre-fuse raw claims and call singleton opening.**
   Rejected for future generated-role integration because it keeps the protocol
   layer responsible for PCS fusion challenges and output-relation bookkeeping.

## Future Work

1. Wire Bolt-generated prover/verifier crates directly to the source-backed
   opening API.
2. Add reference tests comparing legacy `jolt-core` outputs against the
   generated-role implementation.
3. If a later protocol-breaking change lands, mirror it on both sides or stage a
   dedicated compatibility plan.
4. Revisit whether non-linear native batch APIs deserve their own extension
   traits once a concrete non-Dory backend needs them.
