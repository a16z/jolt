# Spec: Unify Field Trait Hierarchy


| Field     | Value      |
| --------- | ---------- |
| Author(s) | @quangvdao |
| Created   | 2026-04-30 |
| Status    | proposed   |
| PR        | #1484      |


## Summary

`jolt_field::Field` is a single monolithic trait that bakes in BN254-shaped assumptions: infix division via Rust's `Div` operator (panics on zero with no caller signal), fixed `[u8; 32]` byte serialization, a hardcoded `type Accumulator` associated type, canonical-representation introspection (`to_u64`, `num_bits`), and 254-bit-specific helpers (`mul_pow_2`, `mul_u64`/`i64`/`u128`/`i128`).
The lattice PCS in `[LayerZero-Labs/hachi](https://github.com/LayerZero-Labs/hachi)` (small-prime fields `Fp32`/`Fp64`/`Fp128`) cannot satisfy this trait and so currently re-implements the same machinery on a parallel and finer hierarchy: `AdditiveGroup` → `FieldCore` → `{Invertible, FromSmallInt, FieldSampling}` → canonical representation traits, with capability extensions for accumulator and fixed-byte encodings.
After auditing Hachi together with sibling field ecosystems in `binius64`, `plonky2`, and `plonky3`, the decomposition must also avoid prime-field assumptions in byte decoding, Fiat-Shamir challenge sampling, integer embedding, and halving.
Jolt should make the algebraic ladder one step more explicit by inserting `RingCore` between additive groups and fields; this gives cyclotomic rings a first-class home without pretending they are fields.
Adopt that slim hierarchy as a new architectural layer underneath `jolt_field::Field`; keep the existing `Field` trait name for the Jolt compatibility umbrella while moving non-algebraic operations to named orthogonal capabilities; refactor `Fr`'s impls to satisfy each layer; relax bounds on shared verifier code so non-BN254 fields can substitute; gate `light-poseidon`/`ark-bn254`/`ark-ff`/`ark-serialize` in `jolt-transcript` behind a `poseidon` feature.
This is the first PR in a series that gradually makes Jolt field-generic enough to support many field families (small prime fields, binary fields, extension fields, and BN254); this PR's immediate compatibility target is Hachi's small-prime fields.

The extension-field API (`LiftBase`, `ExtField`, `Fp2`, `Fp4`) is explicitly **deferred to a follow-up PR** — see "Follow-Up Work" — to keep this PR's blast radius reviewable.

## Intent

### Goal

Refactor `jolt-field` so its public trait surface is a layered algebraic hierarchy (`AdditiveGroup`, `RingCore`, `FieldCore`, `Invertible`, `MulPow2`, `MulPrimitiveInt`) plus orthogonal capabilities (`FromPrimitiveInt`, `RandomSampling`, `CanonicalBytes`, `ReducingBytes`, `TranscriptChallenge`, `WithAccumulator`, `FixedBytes<const N: usize>`, `FixedByteSize`, `CanonicalBitLength`, `CanonicalU64`).
The existing `Field` trait name remains as the Jolt compatibility umbrella: it includes the algebraic field operators plus the BN254/Jolt capabilities current callsites expect, but those capability traits themselves are not modeled as children of `FieldCore`.
Decompose `Fr`'s impls per layer; relax the bounds on the shared verifier-side code (transcript absorption, transcript challenge generation, sumcheck verifier, multilinear poly traits, RLC reduction) from `F: Field` to the needed slim layers (`CanonicalBytes`, `TranscriptChallenge`, `FromPrimitiveInt`, or finer) so a non-BN254 field can substitute; gate `light-poseidon`/`ark-bn254`/`ark-ff`/`ark-serialize` in `jolt-transcript` behind a `poseidon` feature.

> **Naming decision.** Keep the existing `jolt_field::Field` name in this PR. It is technically imprecise once the slim hierarchy exists, but preserving the identifier keeps this PR focused on the trait decomposition and avoids mixing a workspace-wide rename into the same review. Track any rename as follow-up work.

### Series Scope

This PR is one step in a staged migration toward a generic field hierarchy for Jolt.
The long-term direction is that shared verifier/protocol crates can be parameterized over the smallest trait layer they actually need, making room for binary fields, small-prime fields, extension fields, and BN254-shaped fields without forcing every field family to mimic BN254's serialization and helper API.

This PR intentionally takes the narrow first step:

1. Introduce the slim hierarchy in `jolt-field`.
2. Keep the existing `Field` trait name, but narrow it to algebraic field operators and add explicit capability bounds where callsites need serialization, sampling, accumulators, integer scaling, or canonical bytes.
3. Relax only the shared verifier/protocol bounds needed to demonstrate a non-BN254 field.
4. Use Hachi's `Fp32`/`Fp64`/`Fp128` needs as the concrete design target, validated by a `Mersenne61` compat test, while keeping the trait names and semantics compatible with the audited Binius and Plonky field APIs.

Later PRs should broaden the hierarchy and adoption surface: extension fields, Hachi consuming `jolt-field`, granular bound tightening across `jolt-prover-legacy`, binary-field-specific requirements, and any eventual rename of the top `Field` bundle.

### Invariants

This is a refactor.
The relevant `jolt-eval` invariants (`SplitEqBindLowHigh`, `SplitEqBindHighLow`, `FieldMulScalar`, `Soundness` per `jolt-eval/src/invariant/mod.rs:134-160`) must continue to pass byte-identically.
No new invariants required.

The implementation must additionally preserve:

1. End-to-end `Fr` behavior is identical: same Fiat-Shamir state stream, same proof bytes for `muldiv` and other end-to-end tests, same verifier outcomes. Wire format unchanged.
2. Every operation currently callable on `F: jolt_field::Field` remains available post-refactor, either under the same name through a named capability or under a clearer capability-specific name, except panicking infix `Div`, which is intentionally removed from the trait bound and migrated to explicit inversion at generic callsites.
   `square` moves to `RingCore`; `inverse` and `inv_or_zero` move to `Invertible`; `FieldCore` becomes the algebraic field marker `RingCore + Invertible`; `mul_pow_2` moves to `MulPow2`; `mul_u64`/`mul_i64`/`mul_u128`/`mul_i128` move to `MulPrimitiveInt`; `to_bytes` moves to `FixedBytes<32>::to_bytes_array`; `from_bytes` moves to `ReducingBytes::from_le_bytes_mod_order`; transcript challenge construction moves to `TranscriptChallenge`; `random` moves to `RandomSampling::random`; `from_bool` and the integer constructors move to `FromPrimitiveInt`; `NUM_BYTES`, `Accumulator`, `num_bits`, and `to_u64` move to `FixedByteSize`, `WithAccumulator`, `CanonicalBitLength`, and `CanonicalU64` respectively.
   `WithAccumulator` is scoped at the additive layer because redundant accumulator representations only require an additive result type; multiply-add helpers are an additional ring-level accumulator capability. Some callsites use the names from the capability trait rather than the old `Field` body, but no behavior changes.
3. The slim hierarchy (`AdditiveGroup` … `RingCore` … byte-oriented `CanonicalBytes`/`ReducingBytes` plus `TranscriptChallenge`/`CanonicalU64`/`WithAccumulator`/`FixedBytes`/`FixedByteSize`/`CanonicalBitLength`) is implementable by a non-BN254 field with no arkworks dependency. Demonstrated by a `Mersenne61` (the prime $2^{61} - 1$) compat test that compiles and passes against `jolt-field --no-default-features` and `jolt-transcript --no-default-features`.
4. `Invertible` is the inversion capability over rings: it owns `inverse() -> Option<Self>` and the convenience `inv_or_zero(self) -> Self`, whose default is `self.inverse().unwrap_or_else(Self::zero)`. `FieldCore` is just `RingCore + Invertible`; it does not claim that 2 is invertible, so binary fields can implement `FieldCore`. The `Div` operator is not required by `Field`, `RingCore`, or `FieldCore`; shared code must spell zero handling through `Invertible`.
5. `jolt-transcript --no-default-features` builds with no arkworks dependency and exposes `DigestTranscript`, `Blake2bTranscript`, `KeccakTranscript`, `AppendToTranscript`, and the `domain::`* helpers. Poseidon is feature-gated.
6. The shared verifier/protocol code touched by the `Mersenne61` compatibility test bounds on the slim layer: `F: RingCore`, `F: TranscriptChallenge`, `F: CanonicalBytes + FixedByteSize`, `F: RingCore + MulPow2`, etc., not on `F: Field`.
   Demonstrated by the `Mersenne61` test driving `SumcheckVerifier::verify` to completion.
7. No new dynamic dispatch or extra clones on hot paths. Every method that is `#[inline]` or `#[inline(always)]` today retains the annotation on both the trait method and each impl, since the impl now lives in a sibling module rather than in the same file as the trait.

### Non-Goals

1. Migrating Hachi to consume `jolt-field`. Lands as a separate Hachi PR after this one merges.
2. Changing wire format or transcript byte stream.
3. Extracting any of this into a new crate. The refactor stays within `jolt-field`, `jolt-transcript`, and downstream callsite-bound-relaxation sweeps.
4. Adopting Hachi's lattice-specific traits (`PseudoMersenneField`, `SmoothFftField`, `Module`) or Hachi's exact `u128` canonical-representation trait into `jolt-field`. They stay in Hachi as further refinements above `RingCore` / `CanonicalBytes`.
5. **Adopting Hachi's extension-field API (`LiftBase`, `ExtField`, `Fp2`, `Fp4`, `Fp2Config`, `Fp4Config`, `NegOneNr`, `TwoNr`).** Cleanly portable from Hachi (verbatim) but adds blast radius to the PR. Tracked in "Follow-Up Work" below; lands as a separate PR after this one.
6. Removing BN254-specific capabilities from the existing top-of-hierarchy trait entirely. This PR moves method families to named capability traits: ring-core methods (`square`) to `RingCore`, inversion (`inverse`, `inv_or_zero`) to `Invertible`, primitive integer embedding to `FromPrimitiveInt`, byte encoding to `CanonicalBytes`/`ReducingBytes`/`FixedBytes`, transcript challenge decoding to `TranscriptChallenge`, random sampling to `RandomSampling`, fixed byte-size metadata to `FixedByteSize`, canonical bit-length introspection to `CanonicalBitLength`, checked `u64` extraction to `CanonicalU64`, accumulator support to `WithAccumulator`, power-of-two multiplication to `MulPow2`, and primitive-scalar multiplication to `MulPrimitiveInt`. The Jolt `Field` umbrella still includes the compatibility capabilities; smaller shared code can ask for only the capabilities it uses.
7. Touching `jolt-poly`, `jolt-sumcheck`, `jolt-openings`, `jolt-r1cs`, `jolt-crypto`, `jolt-prover-legacy` semantics. Their bounds get relaxed where possible to enable Hachi reuse, but no behavioral changes.

## Evaluation

### Acceptance Criteria

- `crates/jolt-field/src/lib.rs` exports the slim hierarchy: `AdditiveGroup`, `RingCore`, `FieldCore`, `Invertible`, `FromPrimitiveInt`, `RandomSampling`, `CanonicalBytes`, `ReducingBytes`, `TranscriptChallenge`, `CanonicalU64`, `WithAccumulator`, `FixedBytes`, `FixedByteSize`, `CanonicalBitLength`, `MulPow2`, `MulPrimitiveInt`. Plus the existing `Field` name as the Jolt compatibility umbrella, and the existing helper traits `OptimizedMul`, `MaybeAllocative`.
- The `Field` trait remains the Jolt compatibility umbrella. It supertraits `FieldCore + FromPrimitiveInt + CanonicalBytes + ReducingBytes + TranscriptChallenge + FixedBytes<32> + FixedByteSize + CanonicalBitLength + CanonicalU64 + RandomSampling + WithAccumulator + MulPow2 + MulPrimitiveInt + Serialize + DeserializeOwned + MaybeAllocative` plus structural Rust bounds (`'static + Sized + Copy + Send + Sync + Default + Eq + Hash + Debug + Display`) and has an empty body. The key point is that `FromPrimitiveInt`, `RandomSampling`, serde, `WithAccumulator`, byte-encoding traits, fixed byte-size metadata, canonical bit-length introspection, transcript challenge decoding, and checked canonical-integer extraction are orthogonal capabilities; they are included by the Jolt `Field` umbrella but are not algebraic descendants of `FieldCore`.
- `Fr` implements every leaf trait in the slim hierarchy in sibling modules. The `impl Field for Fr` and `impl FieldCore for Fr` become empty: the supertraits now carry the whole surface.
- `RingCore` exists between `AdditiveGroup` and `FieldCore`. It owns `One`, multiplication, `Sum`/`Product`, and `square`, but not inversion. `AdditiveGroup` owns `Zero`, so additive-only accumulator types and ring types share the same zero API without a duplicate `ZERO` associated const. `CyclotomicRing<F, D>` can implement this in a follow-up without pretending to be a field.
- `Invertible: RingCore` owns both `inverse() -> Option<Self>` and `inv_or_zero(self) -> Self`. `inv_or_zero` has the default `self.inverse().unwrap_or_else(Self::zero)`.
- `FieldCore` is the algebraic field marker `RingCore + Invertible`; it does not require an inverse of 2. Binary fields can therefore implement `FieldCore`.
- `FromPrimitiveInt` is an orthogonal primitive-integer embedding capability. It means "reduce or embed this primitive integer into the implementor's scalar object," not "interpret raw field bytes" or "pack tower-basis coordinates." Signed constructors are retained for Jolt compatibility but should not be used as a shared verifier bound unless the callsite really needs signed embedding. `MulPow2` and `MulPrimitiveInt` are ring-level capabilities that depend on `RingCore + FromPrimitiveInt`, so primitive-scalar multiplication is available to rings as well as fields without making integer embedding an algebraic child of `RingCore`.
- `Field`, `RingCore`, and `FieldCore` do NOT supertrait `Div`. Concrete `Fr` may keep its existing operator impls, but generic code must use `inverse()` or `inv_or_zero()` explicitly.
- `AdditiveGroup` keeps Jolt's existing `num_traits::Zero` expectations, while `RingCore` adds `One`, so current helpers like zero-vector allocation and `OptimizedMul::mul_1_optimized` keep compiling under relaxed slim bounds. The blanket `OptimizedMul` impl relaxes from `F: Field` to `F: RingCore`.
- `CanonicalBytes` is byte-oriented and does not require exact `u128` canonical conversion; `Fr` can implement it because BN254 Fr has a canonical 32-byte encoding. Reducing byte parsing is deliberately split into `ReducingBytes`, matching the current `Field::from_bytes` behavior without making every canonical encoder claim a strict or reducing decoder. Exact `u128` canonical conversion remains Hachi-only for now.
- `TranscriptChallenge` is distinct from `RandomSampling`. `RandomSampling::random` is RNG-backed random field element generation for tests and witnesses, while `TranscriptChallenge::from_challenge_bytes` is Fiat-Shamir challenge decoding from squeezed bytes. Future non-uniform sampling APIs, such as sparse challenges over Hachi rings, should use separate capability traits rather than widening `RandomSampling`. BN254 and `Mersenne61` can implement `TranscriptChallenge` by reducing bytes; binary fields and Plonky-style challengers may use field-specific deserialization, rejection sampling, or native sponge elements instead.
- `FixedByteSize` is standalone and owns fixed-size encoding metadata: `const NUM_BYTES: usize`.
- `CanonicalBitLength` is standalone and owns element-dependent canonical bit-length introspection: `fn num_bits(&self) -> u32`. This remains a method, not a const, because it reports significant bits of the specific canonical representative, not the field modulus width.
- `FixedBytes<const N: usize>` extends `CanonicalBytes + ReducingBytes + FixedByteSize`; it owns only the fixed-array convenience API `fn to_bytes_array(&self) -> [u8; N]` and `fn from_bytes_array(bytes: &[u8; N]) -> Self`, with default implementations that delegate to `to_bytes_le` and `from_le_bytes_mod_order`. Non-field fixed byte blobs should use a separate future trait such as `FixedEncoding<N>` rather than this canonical field/value encoding trait.
- `CanonicalU64` is standalone and owns `fn to_canonical_u64_checked(&self) -> Option<u64>`. Implementors may use a canonical-byte scanning helper when they also implement `CanonicalBytes + FixedBytes<N>`, and `Fr` may use the current arkworks bigint path if that is faster.
- `AppendToTranscript` for field-like values is not bounded by `CanonicalBytes` alone. It requires `CanonicalBytes + FixedByteSize` so the blanket impl can allocate exactly `F::NUM_BYTES` bytes, fill them with little-endian canonical bytes, reverse them, and preserve today's EVM-compatible big-endian transcript absorption.
- `WithAccumulator: AdditiveGroup` points at an additive redundant representation: `type Accumulator: AdditiveAccumulator<Element = Self>`. `AdditiveAccumulator` supports `add`, `merge`, and `reduce`. The current FMA accumulator behavior is represented by a ring-level extension trait, `RingAccumulator: AdditiveAccumulator` with `Element: RingCore + FromPrimitiveInt` and `fmadd`/small-scalar helpers. `NaiveAccumulator<R>` works for any ring `R` satisfying those slim bounds, and also serves as the additive fallback. This lets `Mersenne61` and cyclotomic rings use `NaiveAccumulator<_>` without making accumulator support a child of `FieldCore`.
- `crates/jolt-transcript/Cargo.toml`: `light-poseidon`, `ark-bn254`, `ark-ff`, `ark-serialize` are `optional = true`. A new `poseidon = ["dep:light-poseidon", "dep:ark-bn254", "dep:ark-ff", "dep:ark-serialize", "jolt-field/bn254"]` feature exists. `default = ["poseidon"]`. The `jolt-field` dependency is `default-features = false`; `crates/jolt-transcript/src/poseidon.rs` and the corresponding `pub use` in `lib.rs` are gated `#[cfg(feature = "poseidon")]`.
- `cargo build -p jolt-field --no-default-features` succeeds.
- `cargo build -p jolt-transcript --no-default-features` succeeds.
- `cargo nextest run --workspace --features host` passes.
- `cargo nextest run --workspace --features host,zk` passes.
- `cargo nextest run -p jolt-prover-legacy muldiv --features host` and `--features host,zk` pass.
- `cargo clippy --all --features host --all-targets -- -D warnings` passes. Same for `host,zk`.
- `crates/jolt-sumcheck/tests/mersenne61_compat.rs`: a `Mersenne61` struct (modulus $2^{61} - 1$, no arkworks dep) implements `AdditiveGroup + RingCore + FieldCore + FromPrimitiveInt + RandomSampling + CanonicalBytes + ReducingBytes + TranscriptChallenge + CanonicalU64 + WithAccumulator + FixedByteSize + CanonicalBitLength + FixedBytes<8>` (using `NaiveAccumulator<Mersenne61>`). The test lives downstream of `jolt-field` so `jolt-field` remains a leaf crate. The test:
  - Substitutes `Mersenne61` into `Blake2bTranscript<Mersenne61>` and `KeccakTranscript<Mersenne61>` from `jolt-transcript --no-default-features`.
  - Drives a hand-rolled 4-round `RoundProof<Mersenne61>` impl through `SumcheckVerifier::verify` to completion.
  - Compiles and passes without the `bn254` feature on `jolt-field` and without `poseidon` on `jolt-transcript`.
- `crates/jolt-field/tests/binary_field_core_compat.rs`: a tiny characteristic-2 field or GF(2) wrapper implements `AdditiveGroup + RingCore + FieldCore + Invertible + CanonicalBytes + ReducingBytes + FixedByteSize` but not `MulPow2` or the Jolt `Field` umbrella. The test is compile-oriented: it proves the algebraic layer admits binary fields, and it prevents future verifier-bound relaxations from accidentally smuggling odd-prime integer-scaling assumptions back into `FieldCore`.
- Wire-format roundtrip test (in `crates/jolt-field/tests/` or `crates/jolt-sumcheck/tests/`): bincode-encode a deterministic `SumcheckProof<Fr>` (10 rounds, degree 3) and a `Vec<Fr>` of length 32 with a fixed RNG seed; assert the byte blob matches a hardcoded expected value captured before the refactor.
- Transcript-determinism test: replay a fixed sequence of `append`/`challenge` calls on `Blake2bTranscript<Fr>`, `KeccakTranscript<Fr>`, and `PoseidonTranscript`; assert each `state()` snapshot matches a hardcoded expected value captured before the refactor.

### Testing Strategy

**Existing tests that must keep passing:**

- All tests in `crates/jolt-field/`, `crates/jolt-transcript/`, `crates/jolt-poly/`, `crates/jolt-sumcheck/`, `crates/jolt-openings/`, `crates/jolt-crypto/`, `crates/jolt-r1cs/`, `jolt-prover-legacy/`.
- `jolt-eval` invariant suites (`field_mul_scalar`, `split_eq_bind`, `soundness`).
- `jolt-prover-legacy muldiv` end-to-end in both `host` and `host,zk` modes.

**New tests:**

- `crates/jolt-sumcheck/tests/mersenne61_compat.rs` (above). The single most important new test: it gates the slim hierarchy (so it stays implementable by non-BN254 fields) and exercises shared verifier code with a non-arkworks field. Without it, regressions toward BN254-shaped trait surface slip through.
- Wire-format and transcript-determinism roundtrip tests above.

**Feature coverage:**

- `--features host` and `--features host,zk` for the standard workspace gate.
- `--no-default-features` for `jolt-field` and `jolt-transcript` to verify BN254/Poseidon gating works.

### Performance

No regression.
This is a pure refactor.

**Concrete checks:**

- `cargo bench -p jolt-field --bench field_arith`: throughput within ±2% of pre-refactor baseline.
- `cargo bench -p jolt-transcript --bench transcript_ops`: same.
- `cargo bench -p jolt-poly --bench poly_ops`: same.
- `cargo bench -p jolt-openings --bench rlc`: same.
- `cargo bench -p jolt-crypto --bench crypto`: same.

The `jolt-eval` performance objectives (`jolt-eval/src/objective/performance/`, including `field_mul.rs`) must not regress.
No objectives are expected to move; this is a trait decomposition with the same impl bodies.

The risk surface is `#[inline]` annotations being lost when methods cross from the old monolithic impl to per-trait sibling-module impls.
The implementer MUST audit every method moved off the monolithic impl and confirm `#[inline]` (or `#[inline(always)]` where present today) is preserved on both the trait method and the impl.
If LLVM cannot inline across the new module boundaries, add `#[inline]` to the impls explicitly.

## Design

### Architecture

The algebraic hierarchy after refactor:

```
AdditiveGroup                       // Zero + Add/Sub/Neg + AddAssign/SubAssign
                                    // (also implementable by wide accumulators)
   └── RingCore                     // One, Mul/MulAssign + Sum/Product,
       │                            //   square. NO inverse, NO Div.
       ├── Invertible               // inverse() -> Option<Self>,
       │                            // inv_or_zero(self) -> Self (default via inverse)
       ├── MulPow2                  // mul_pow_2
       ├── MulPrimitiveInt          // mul_u64/i64/u128/i128
       └── FieldCore                // RingCore + Invertible. NO Div.
```

Orthogonal capability traits (not algebraic children of `FieldCore`):

```
RandomSampling                      // random<R: RngCore>(rng) -> Self
WithAccumulator: AdditiveGroup      // type Accumulator: AdditiveAccumulator<Element = Self>
FromPrimitiveInt                    // reducing primitive integer embedding;
                                    // signed constructors are compatibility surface
CanonicalBytes                      // to_bytes_le(&mut [u8]),
ReducingBytes                       // from_le_bytes_mod_order(&[u8])
TranscriptChallenge                 // from_challenge_bytes(&[u8])
FixedBytes<const N: usize>          // CanonicalBytes + ReducingBytes + FixedByteSize,
                                    // to_bytes_array() -> [u8; N],
                                    // from_bytes_array(&[u8; N]) -> Self
FixedByteSize                       // const NUM_BYTES
CanonicalBitLength                  // num_bits() -> u32
CanonicalU64                        // to_canonical_u64_checked() -> Option<u64>
(PseudoMersenneField → SmoothFftField stays in hachi)

External capability bounds used by the Jolt compatibility umbrella:
Serialize + DeserializeOwned + MaybeAllocative
```

Trait signatures (stable shape; adapted from `hachi/src/primitives/arithmetic.rs`, with `num_traits::Zero + One` retained for Jolt callsites):

```rust
// crates/jolt-field/src/additive_group.rs
pub trait AdditiveGroup:
    Sized + Clone + Copy + Send + Sync
    + Zero
    + Add<Output = Self> + AddAssign + for<'a> Add<&'a Self, Output = Self>
    + Sub<Output = Self> + SubAssign + for<'a> Sub<&'a Self, Output = Self>
    + Neg<Output = Self>
{}

// crates/jolt-field/src/ring_core.rs
pub trait RingCore:
    AdditiveGroup
    + One
    + PartialEq + Eq + Default + Debug + Display + Hash
    + Mul<Output = Self> + MulAssign + for<'a> Mul<&'a Self, Output = Self>
    + Sum<Self> + for<'a> Sum<&'a Self>
    + Product<Self> + for<'a> Product<&'a Self>
{
    fn square(&self) -> Self { *self * *self }
}

// crates/jolt-field/src/invertible.rs
pub trait Invertible: RingCore {
    /// Branching (public-value) inverse. Returns `None` on zero.
    fn inverse(&self) -> Option<Self>;

    /// Inverse with zero-mapping behavior.
    ///
    /// Default may branch through `inverse`.
    fn inv_or_zero(self) -> Self {
        self.inverse().unwrap_or_else(Self::zero)
    }
}

// crates/jolt-field/src/field_core.rs
/// Algebraic field marker: ring arithmetic plus explicit inversion.
///
/// This trait intentionally does not require Rust's infix `Div` operator.
/// `a / b` cannot expose the `b = 0` case in its type, and existing backend
/// implementations panic on zero denominators by unwrapping `inverse()`.
/// Generic code should use `inverse()` or `inv_or_zero()` so zero handling is
/// visible at the callsite.
pub trait FieldCore: RingCore + Invertible {}

// crates/jolt-field/src/from_primitive_int.rs
/// Embed primitive integer values into a type.
///
/// This is intentionally orthogonal to `RingCore`: byte/challenge parsing and
/// lookup helpers may need integer embedding without also needing generic ring
/// arithmetic. Ring-specific helpers add `RingCore` where multiplication is
/// actually used.
///
/// Implementations reduce or embed integers into the scalar object. This is not
/// a byte decoder and not a tower-basis packing API. Signed constructors are
/// retained for current Jolt compatibility; shared verifier code should prefer
/// unsigned constructors unless signed semantics are required.
pub trait FromPrimitiveInt: Sized {
    fn from_bool(v: bool) -> Self { if v { Self::from_u64(1) } else { Self::from_u64(0) } }
    fn from_u8(v: u8)   -> Self { Self::from_u64(v as u64) }
    fn from_i8(v: i8)   -> Self { Self::from_i64(v as i64) }
    fn from_u16(v: u16) -> Self { Self::from_u64(v as u64) }
    fn from_i16(v: i16) -> Self { Self::from_i64(v as i64) }
    fn from_u32(v: u32) -> Self { Self::from_u64(v as u64) }
    fn from_i32(v: i32) -> Self { Self::from_i64(v as i64) }
    fn from_u64(v: u64) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_u128(v: u128) -> Self;
    fn from_i128(v: i128) -> Self;
}

// crates/jolt-field/src/random_sampling.rs
pub trait RandomSampling {
    fn random<R: RngCore>(rng: &mut R) -> Self;
}

// crates/jolt-field/src/mul_pow_2.rs
pub trait MulPow2: RingCore + FromPrimitiveInt {
    /// Multiplication of a ring element by the integer `2^pow`.
    ///
    /// In characteristic two, this is zero for `pow > 0`. Protocol code that
    /// expects nonzero powers of two must keep an explicit odd-characteristic
    /// assumption rather than relying on `RingCore`.
    /// Default: 63-bit-chunked path. `Fr` overrides with the existing implementation.
    fn mul_pow_2(&self, pow: usize) -> Self {
        assert!(pow <= 255);
        let mut res = *self;
        let mut p = pow;
        while p >= 64 {
            res = res * Self::from_u64(1u64 << 63);
            p -= 63;
        }
        res * Self::from_u64(1u64 << p)
    }
}

// crates/jolt-field/src/mul_primitive_int.rs
pub trait MulPrimitiveInt: RingCore + FromPrimitiveInt {
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }

    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }

    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }

    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }
}

// crates/jolt-field/src/canonical_bytes.rs
pub trait CanonicalBytes: Sized {
    /// Caller-provided canonical byte buffer; panics if the length is invalid for this type.
    fn to_bytes_le(&self, out: &mut [u8]);

    fn to_bytes_le_vec(&self) -> Vec<u8>
    where
        Self: FixedByteSize,
    {
        let mut out = vec![0u8; Self::NUM_BYTES];
        self.to_bytes_le(&mut out);
        out
    }
}

// crates/jolt-field/src/reducing_bytes.rs
pub trait ReducingBytes: Sized {
    /// Deserialize little-endian bytes by reducing into this type.
    ///
    /// This preserves the current `Field::from_bytes` semantics for BN254.
    /// It is intentionally separate from `CanonicalBytes`, because Binius and
    /// Plonky-style APIs distinguish canonical or unique encodings from
    /// reducing constructors and transcript sampling.
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self;
}

// crates/jolt-field/src/transcript_challenge.rs
pub trait TranscriptChallenge: Sized {
    /// Construct a Fiat-Shamir challenge from squeezed transcript bytes.
    ///
    /// BN254 and Mersenne-style prime fields may implement this by reducing
    /// bytes. Binary fields may deserialize a field-specific byte layout, and
    /// Plonky-style challengers may prefer rejection sampling or native sponge
    /// elements to avoid modulo bias.
    fn from_challenge_bytes(bytes: &[u8]) -> Self;
}

// crates/jolt-field/src/fixed_bytes.rs
pub trait FixedBytes<const N: usize>: CanonicalBytes + ReducingBytes + FixedByteSize {
    /// Returns a fixed-size byte encoding.
    fn to_bytes_array(&self) -> [u8; N] {
        debug_assert_eq!(Self::NUM_BYTES, N);
        let mut out = [0u8; N];
        self.to_bytes_le(&mut out);
        out
    }

    fn from_bytes_array(bytes: &[u8; N]) -> Self {
        Self::from_le_bytes_mod_order(bytes)
    }
}

// crates/jolt-field/src/fixed_byte_size.rs
pub trait FixedByteSize {
    /// Byte length of the fixed-size encoding.
    const NUM_BYTES: usize;
}

// crates/jolt-field/src/canonical_bit_length.rs
pub trait CanonicalBitLength {
    /// Number of significant bits in the canonical little-endian encoding.
    fn num_bits(&self) -> u32;
}

// crates/jolt-field/src/canonical_u64.rs
pub trait CanonicalU64 {
    /// Return the canonical representative as `u64` if it fits.
    fn to_canonical_u64_checked(&self) -> Option<u64>;
}

// crates/jolt-field/src/with_accumulator.rs
pub trait WithAccumulator: AdditiveGroup {
    type Accumulator: AdditiveAccumulator<Element = Self>;
}
```

The existing trait keeps the `Field` name as Jolt's compatibility umbrella:

```rust
// crates/jolt-field/src/field.rs
pub trait Field:
    'static + Sized + Copy + Send + Sync + Default
    + Eq + Hash + Debug + Display
    + FieldCore
    + FromPrimitiveInt
    + CanonicalBytes + ReducingBytes + TranscriptChallenge
    + FixedBytes<32> + FixedByteSize + CanonicalBitLength
    + CanonicalU64 + RandomSampling + WithAccumulator
    + MulPow2 + MulPrimitiveInt
    + Serialize + DeserializeOwned + MaybeAllocative
{
    // Empty compatibility marker bundle. Methods and associated items live on
    // named capability traits.
}
```

The universal `square` method lives on `RingCore`; inversion lives on `Invertible`; `FieldCore` is the marker combining `RingCore + Invertible`.
None are duplicated on `Field`.
The multiplication helper families live on `MulPow2` and `MulPrimitiveInt`; they are not duplicated on `Field`.
The serialization, random sampling, accumulator, fixed byte-size metadata, canonical bit-length introspection, transcript challenge decoding, and checked canonical-integer entry points live only on `CanonicalBytes`, `ReducingBytes`, `TranscriptChallenge`, `FixedBytes<32>`, `FixedByteSize`, `CanonicalBitLength`, `CanonicalU64`, `RandomSampling`, and `WithAccumulator`; `Field` inherits those capability traits for compatibility but does not duplicate `to_bytes`/`from_bytes`/`random`/`to_u64`/`num_bits` wrapper methods in its body.
`Fr` keeps its `mul_pow_2` override verbatim.
Code that relaxes away from `F: Field` and uses non-algebraic operations gains explicit capability bounds, e.g. `F: CanonicalBytes + FixedByteSize`, `F: ReducingBytes`, `F: TranscriptChallenge`, `F: CanonicalBitLength + CanonicalU64`, or `F: RandomSampling`.
Callsites that specifically invoke the old wrapper names migrate mechanically: `x.to_bytes()` becomes `<F as FixedBytes<32>>::to_bytes_array(&x)` (or `x.to_bytes_array()` when the bound is visible), `F::from_bytes(bytes)` becomes `<F as ReducingBytes>::from_le_bytes_mod_order(bytes)`, `F::random(rng)` becomes `<F as RandomSampling>::random(rng)`, and `x.to_u64()` becomes `x.to_canonical_u64_checked()`.
Existing `x.num_bits()` callsites keep the same method name but get it from `CanonicalBitLength` rather than `Field`.

Usage audit before the refactor: `to_u64` is only exercised by field tests and concrete field impls, not by Jolt protocol logic.
`num_bits` is used by BN254-specific GLV code and transcript tests; both uses are canonical-representation introspection and do not justify keeping methods directly on the top `Field` bundle.

`CanonicalBytes` is the universal canonical encoding layer and is byte-oriented, not `u128`-oriented.
Fixed-size canonical encodings pair `CanonicalBytes` with `FixedByteSize`; `CanonicalBytes` itself does not own `NUM_BYTES`.
This matters because BN254 Fr has a canonical 32-byte representation but cannot expose an exact `to_canonical_u128`.
`ReducingBytes::from_le_bytes_mod_order` is intentionally a reducing constructor rather than a strict canonical decoder, matching today's `Field::from_bytes`.
If a future caller needs rejection of non-canonical encodings, add a separate checked decoder instead of changing this method's behavior.
Do not use `CanonicalBytes` alone as a transcript-sampling contract: Binius samples challenges by deserializing field-specific bytes, while Plonky-style challengers distinguish unique/hash encodings from rejection-sampled field elements.
`TranscriptChallenge` is the place to encode those Fiat-Shamir semantics.
`FromPrimitiveInt::from_u128` remains the universal "embed/reduce this primitive integer into the algebraic object" API for scalar construction and lookup constants.
It is no longer the default transcript challenge hook, because reducing fixed-width integers introduces the wrong abstraction for binary fields and can hide bias concerns for Plonky-style prime fields.
Exact `u128` canonical conversion stays in Hachi for now, where `Fp32`/`Fp64`/`Fp128` and pseudo-Mersenne/FFT refinements actually need it.
Jolt can add that capability later if a shared protocol callsite needs exact small-field representatives, but this PR only needs byte encoding plus primitive integer embedding.

`FixedBytes<N>` deliberately extends `CanonicalBytes + ReducingBytes + FixedByteSize`.
It is the fixed-array convenience layer for canonical field/value encodings, not a generic fixed-byte trait for commitments, hashes, or arbitrary blobs.
If those non-field values need a shared fixed-byte trait later, add a separate `FixedEncoding<N>` trait rather than weakening the semantics of `FixedBytes<N>`.
`FixedByteSize` is standalone because byte length metadata is a representation fact, not a field law.
`CanonicalBitLength` is separate because `num_bits(&self)` is element-dependent and cannot be a const unless the API changes to mean modulus or encoding capacity.
`CanonicalU64` is a separate narrow capability because checked extraction into a machine integer is neither fixed-byte encoding nor fixed byte-size metadata.

`RingCore` is intentionally explicit even though the immediate `Mersenne61` compatibility test is a field.
Cyclotomic rings like Hachi's `CyclotomicRing<F, D> = F[X]/(X^D + 1)` have addition, multiplication, zero, one, sums/products, and squaring, but they do not have inverses for every nonzero element.
They should therefore implement `RingCore`, not `FieldCore`.

`Invertible` is a ring-level capability because some rings or ring-like representations may have a usable inversion operation without being the default field scalar type.
It owns both `inverse` and `inv_or_zero`; the default `inv_or_zero` is intentionally simple and delegates to `inverse`.
The first implementation PR should not introduce a separate inverse-of-two capability.
Jolt currently computes halving at the few relevant callsites with `F::from_u64(2).inverse().unwrap()`, and this PR leaves those `jolt-prover-legacy` callsites under the existing `F: Field` umbrella.

Current `Fr` division implementations may stay as concrete operator impls: they are already centralized by the existing `delegate_binop!(Div, div)` macro in `crates/jolt-field/src/arkworks/bn254.rs`.
This PR does not add a new division trait, and `Field` no longer requires `Div`.
`Div` intentionally stays off `Field`, `RingCore`, and `FieldCore` because the Rust operator cannot express zero-handling in its type: arkworks `Fp` division computes `other.inverse().unwrap()`, so BN254 division by zero panics, while verifier/protocol code often wants explicit branching via `inverse() -> Option<Self>`.
The one current generic field infix-division callsite in `jolt-prover-legacy/src/subprotocols/mles_product_sum.rs` should migrate mechanically from `(claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0` to multiplication by `eq_eval_at_0.inverse().expect(...)`, preserving the same partial-operation behavior while making the zero case explicit.
Dropping `Div` from the top `Field` bundle prevents newly relaxed shared code (`F: RingCore`, `F: FieldCore`, `F: CanonicalBytes`, etc.) from accidentally depending on panicking infix division.
Code that needs explicit zero handling should use `Invertible::inv_or_zero` or `Invertible::inverse`, not infix `Div`.

In mathematical terms, `FieldCore` is the field-like algebraic layer (`RingCore` plus `Invertible`), while the existing `Field` trait is Jolt's compatibility umbrella.
Serialization, sampling, canonical bytes, and accumulator support are intentionally orthogonal capabilities even when the `Field` umbrella includes them.

`OptimizedMul` keeps the same API but its blanket impl moves down from `F: Field` to `F: RingCore`.
The methods only need multiplication plus `Zero`/`One` checks, and keeping the impl tied to the compatibility umbrella would make relaxed `RingCore` callsites lose the fast-path helpers.

`AdditiveGroup` separated from `RingCore` lets wide accumulator types (`WideAccumulator`, hachi's `Fp128x8i32`, etc.) implement `+`/`-` without claiming to support multiplication or a multiplicative identity.
`RingCore` separated from `FieldCore` lets cyclotomic ring representations support generic ring arithmetic without claiming that every nonzero element is invertible.

`FieldAccumulator` must also move down to the slim hierarchy and split into additive and ring-level pieces.
The base capability is additive because the point of the associated accumulator is a redundant representation of an element and a way to merge/reduce those redundant sums.
The current hot-loop `acc += a * b` behavior remains available as a ring-level extension.
Today `FieldAccumulator` requires `type Field: Field`, which would force `Mersenne61` or a cyclotomic ring to implement unrelated field capabilities merely to use `NaiveAccumulator`.
After this refactor the additive layer needs only additive group structure, and the FMA layer needs only the ring arithmetic used by its default methods:

```rust
pub trait AdditiveAccumulator: Default + Copy + Send + Sync {
    type Element: AdditiveGroup;
    fn add(&mut self, value: Self::Element);
    fn merge(&mut self, other: Self);
    fn reduce(self) -> Self::Element;
}

pub trait RingAccumulator: AdditiveAccumulator
where
    Self::Element: RingCore + FromPrimitiveInt,
{
    fn fmadd(&mut self, a: Self::Element, b: Self::Element);
    fn fmadd_u8(&mut self, a: Self::Element, b: u8) { ... }
    fn fmadd_u64(&mut self, a: Self::Element, b: u64) { ... }
    fn fmadd_i64(&mut self, a: Self::Element, b: i64) { ... }
    fn fmadd_bool(&mut self, a: Self::Element, b: bool) { ... }
}

pub struct NaiveAccumulator<F: RingCore + FromPrimitiveInt>(F);
```

Code that needs current hot-loop fused multiply-add behavior must ask for both layers:
`F: WithAccumulator` and `<F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>`.
The bare `WithAccumulator` bound only promises additive `add`/`merge`/`reduce`.

**`jolt-transcript` Poseidon gating:**

`crates/jolt-transcript/Cargo.toml` becomes:

```toml
[features]
default = ["poseidon"]
poseidon = ["dep:light-poseidon", "dep:ark-bn254", "dep:ark-ff", "dep:ark-serialize", "jolt-field/bn254"]

[dependencies]
blake2.workspace = true
digest.workspace = true
sha3.workspace = true
jolt-field = { workspace = true, default-features = false }
ark-bn254 = { workspace = true, optional = true }
ark-ff = { workspace = true, optional = true }
ark-serialize = { workspace = true, optional = true }
light-poseidon = { workspace = true, optional = true }
```

`crates/jolt-transcript/src/poseidon.rs` and the corresponding `pub use poseidon::*;` in `lib.rs` are gated `#[cfg(feature = "poseidon")]`.
The hash-based transcript aliases cannot default to `jolt_field::Fr` when `poseidon`/`bn254` is disabled. Gate the defaults accordingly:

```rust
#[cfg(feature = "poseidon")]
pub type Blake2bTranscript<F = jolt_field::Fr> = DigestTranscript<Blake2b<U32>, F>;
#[cfg(not(feature = "poseidon"))]
pub type Blake2bTranscript<F> = DigestTranscript<Blake2b<U32>, F>;
```

Apply the same pattern to `KeccakTranscript`.
Remove `DigestTranscript`'s unconditional default type parameter entirely; downstream users should get the old default through `Blake2bTranscript` / `KeccakTranscript` when the `poseidon` feature is enabled.
Default features stay on for in-workspace consumers (jolt-prover-legacy, etc.), so no jolt-internal callsites change.
Hachi opts into `default-features = false`.

**Bound relaxation (minimal sweep).**
Where shared verifier/protocol code currently bounds `F: Field` and the body only uses slim algebraic or representation capabilities, the bound is relaxed to whichever subset is actually used.
This is the only knob needed to make the `Mersenne61` test pass; it is also the migration path Hachi will follow without further jolt-side changes.
Initial relaxation list:

- `jolt_sumcheck::SumcheckClaim` / `EvaluationClaim` / `SumcheckError`: relax from `F: Field` to structural bounds required by derives and formatting, e.g. `F: Clone + Debug + PartialEq + Eq + Display` where applicable.
- Add `jolt_sumcheck::SumcheckScalar` as a local marker trait bundling the scalar capabilities used across the sumcheck verifier, batched verifier, round proof, and proof types:
  `RingCore + MulPow2 + CanonicalBytes + FixedByteSize + TranscriptChallenge + FromPrimitiveInt + Clone + Debug + Display + Eq`.
  This is intentionally a little stricter than each individual function needs, because the named marker keeps downstream signatures readable while still avoiding the old BN254-shaped `Field` umbrella.
- `jolt_sumcheck::SumcheckVerifier::verify`: relax from `F: Field` to the bounds implied by `SumcheckClaim<F>`, `EvaluationClaim<F>`, `SumcheckError<F>`, `T: Transcript<Challenge = F>`, and `P: RoundProof<F>`. Where these bounds become noisy, use `F: SumcheckScalar`.
- `jolt_sumcheck::BatchedSumcheckVerifier::verify`: relax from `F: Field` to `F: SumcheckScalar` plus the structural bounds required by claims/errors and `P: RoundProof<F>`. `SumcheckScalar` covers absorbing claimed sums before drawing `alpha`, `zero`, `one`, `+`, `*`, `*=`, and claim scaling. Because `MulPow2` is integer multiplication by `2^k`, any future binary-field version of this verifier must either avoid that scaling path or keep an explicit odd-characteristic assumption.
- `jolt_sumcheck::RoundProof` trait and impls: relax where the body permits. The clear `UnivariatePoly<F>` impl needs enough ring arithmetic for `evaluate`, `zero`, `one`, and equality; labeled impls additionally need `AppendToTranscript`, which is supplied by `CanonicalBytes + FixedByteSize`.
- `jolt_transcript::DigestTranscript` / hash transcript challenge generation: from `F: Field` to `F: TranscriptChallenge`, because challenges are derived from squeezed bytes but the reducing strategy is field-family-specific.
- `jolt_transcript::AppendToTranscript` blanket impl: from `F: Field` to `F: CanonicalBytes + FixedByteSize`, because transcript absorption needs canonical bytes and the fixed byte length. Preserve today's little-endian-to-big-endian reversal before absorption.
- `jolt_poly` multilinear traits (`MultilinearPoly`, `MultilinearBinding`, `RlcSource`): `F: Field` → `F: FieldCore + WithAccumulator` (or finer, per body).
- `jolt_openings::reduce_`* and `rlc_combine*`: `F: Field` → `F: FieldCore` (or `+ RandomSampling` if the body samples random elements).

Every other `F: Field` bound in the workspace stays as `F: Field`.

### Cross-Field Compatibility Audit

The trait split above is based on Hachi's small-prime hierarchy, then checked against the sibling `../binius64`, `../plonky2`, and `../plonky3` field APIs.
Those crates agree with the high-level goal, but they expose a few portability traps that this spec now accounts for.

- **Binius binary fields.** Binius has characteristic-2 fields and tower fields with field-specific `SerializeBytes` / `DeserializeBytes` and transcript `CanSample<F>` paths. Integer embedding is not a substitute for tower-basis packing, and `2 = 0`, so `FieldCore` must not imply halving or odd-characteristic behavior.
- **Plonky2 fields.** Plonky2 distinguishes canonical constructors (`from_canonical_u64`) from reducing constructors (`from_noncanonical_u128`) and serializes extension elements as base-field coefficient arrays. Jolt should not conflate canonical encoding, reducing byte decoding, and transcript challenge sampling in one trait.
- **Plonky3 fields.** Plonky3 separates canonical representatives, unique hashing encodings, quotient maps, raw byte streams, and challenger-specific rejection sampling. `TranscriptChallenge` is therefore a separate capability rather than a blanket `FromPrimitiveInt` or `CanonicalBytes` consequence.
- **Extension fields.** Plonky2 and Plonky3 both represent extension elements by ordered base-field coordinates. The extension-field follow-up must specify coefficient order and transcript byte order explicitly rather than borrowing an unstable implementation basis from another crate.

### Alternatives Considered

1. **Delete BN254-shaped capabilities entirely instead of moving them to named traits.** Rejected — loses callable functionality from the existing API and forces churn at every jolt-prover-legacy callsite that uses fixed-width bytes, sampling, accumulators, `mul_pow_2`, etc. Better to keep the capabilities available through explicit bounds.
2. **Keep `Field` monolithic, add a `BackendField: Field` extension.** Rejected — keeps every BN254-shaped method directly on the trait body, which is exactly what Hachi cannot satisfy. The right direction is to keep `Field` as a compatibility umbrella with an empty body while moving the surface into named capabilities that can also be requested independently.
3. **Extract a new `iop-core` crate containing the slim trait surface.** Rejected — extra crate boundary with no compile-time benefit. Hachi-Jolt sharing works as long as both implement the same traits, regardless of which crate owns them.
4. **Drop `jolt_field::Field` entirely; force every callsite to bound exactly what it needs.** Rejected for now — too much downstream churn in one PR. Granular tightening can happen incrementally in follow-up PRs once the hierarchy is in place. The minimal bound-relaxation sweep (above) is the only change to existing bounds in this PR.
5. **Put `Div` on `FieldCore`, `Field`, or anywhere reachable by shared verifier code.** Rejected — hides the zero-denominator case behind one infix operator. Arkworks BN254 division by zero panics through `inverse().unwrap()`, and the only current generic field infix-division callsite in `jolt-prover-legacy` is migrated to explicit inversion instead.
6. **Don't separate `AdditiveGroup` from `FieldCore`.** Rejected — `AdditiveGroup` is the right home for wide accumulators which need `+`/`-` but not multiplication. Hachi already exploits this for `Fp128x8i32` and the wide-cyclotomic-ring shift-accumulate kernels; Jolt's `WideAccumulator` benefits similarly.
7. **Adopt all of Hachi's lattice-specific traits (`PseudoMersenneField`, `SmoothFftField`, `Module`) into `jolt-field`.** Rejected — these only make sense for lattice/post-quantum constructions; BN254 doesn't satisfy them. Stay in Hachi.
8. **Adopt the extension-field API (`LiftBase`, `ExtField`, `Fp2`, `Fp4`, non-residue configs) in this PR.** Rejected for blast-radius reasons — adopting it is cheap (verbatim port from Hachi, ~5 new files, no new logic), but it extends the surface that needs review and the `Mersenne61` test gates that need writing. Tracked in "Follow-Up Work" so the design reasoning isn't lost.
9. **Variable-length serialization via associated type `type Bytes: AsRef<[u8]>`.** Rejected — heavier API for callers (need to name `<F as CanonicalBytes>::Bytes` everywhere) with no compile-time win over `to_bytes_le(&mut [u8])` plus the standalone `FixedByteSize` capability. The `FixedBytes<N>` trait covers the array-form case for canonical field/value encodings.
10. **Name the integer-embedding trait `FromSmallInt` (matching Hachi today) or `FromPrimInt`.** Rejected — "small" misleads once `u128`/`i128` are included, while `PrimInt` reads like a dependency or exact match to `num_traits::PrimInt`. `FromPrimitiveInt` is explicit about embedding primitive integer widths into a ring or field without implying that the trait covers every `PrimInt` detail.
11. **Name the compat field `StubField` or `TestField`.** Rejected — $2^{61} - 1$ is the Mersenne prime $M_{61}$, a real and historically named field. `Mersenne61` is honest about what it is and matches the precedent set by Plonky3's `Mersenne31`.
12. **Put `digit_lut(log_basis)` on `FromPrimitiveInt`.** Rejected for this PR — Jolt currently needs primitive integer embedding (`from_u64`, `from_i128`, etc.) but not Hachi's balanced-digit lookup-table helper. Keep the slim trait minimal; if a shared balanced-digit decomposition API is needed later, add it as a separate Hachi/Jolt capability trait.
13. **Put exact `u128` canonical conversion on `CanonicalBytes` or in this Jolt PR.** Rejected — BN254 Fr can implement canonical byte encoding but its canonical element representation is 254 bits and cannot fit in `u128`. Exact small-field canonical representation is useful for Hachi, but not needed by the Jolt shared verifier/protocol sweep in this PR.
14. **Leave `mul_pow_2` and primitive-scalar multiplication directly on `Field`.** Rejected — these are separate capabilities from the BN254-shaped top bundle. `MulPow2` and `MulPrimitiveInt` keep the operations available to `F: Field` while allowing future code to request only the multiplication helper it actually needs.
15. **Name the ring layer `Ring` or `AdditiveRing`.** Rejected — `Ring` is too broad and likely to collide conceptually with concrete polynomial/cyclotomic ring types, while `AdditiveRing` is mathematically odd because rings are already additive groups with multiplication. `RingCore` matches `FieldCore`: it is the minimal algebraic API for ring arithmetic, not a concrete ring object.
16. **Add a `TWO_INV` / halving capability in this PR.** Rejected — Jolt does not currently expose a `TWO_INV` trait or constant. The few existing halving callsites are in `jolt-prover-legacy`, stay under the existing `F: Field` umbrella in this PR, and can be revisited when Jolt is ready to make those paths binary-field-aware.

## Decisions and Open Questions

The naming question is resolved for this PR. The inlining question remains open until benchmarks run after implementation.

### D-1: Keep `Field` for this PR

The current `jolt_field::Field` trait is BN254-shaped: it requires fixed `[u8; 32]` serialization, `mul_pow_2`, primitive-scalar multiplication helpers, and other large-prime-field capabilities.
After this refactor, "Field" in the abstract sense is more accurately `FieldCore`; the existing `Field` trait is specifically the bundle for "large prime field with the BN254-shaped surface".
Keeping the name `Field` is technically imprecise, but every alternative (`LargeField`, `LargePrimeField`, `BigPrimeField`, `Bn254ShapedField`) forces a workspace-wide identifier sweep on top of the trait restructuring.
This PR keeps `Field`; if maintainers want the precision win, do it as a separate rename-only PR after the hierarchy lands.

### OQ-1: Fast-path defaults on multiplication helper traits.

The spec proposes default bodies on `MulPow2::mul_pow_2` and `MulPrimitiveInt::{mul_u64,mul_i64,mul_u128,mul_i128}` so that future implementors can get workable behavior automatically.
`Fr` overrides them with BN254-specific fast paths (the current bodies in `crates/jolt-field/src/arkworks/bn254.rs`).

Question: do the defaults need any inlining hints to avoid a perf cliff if some path winds up calling the default by accident?
The next agent should confirm by running the `field_arith` bench against `Fr` post-refactor and comparing to baseline.

## Documentation

- `crates/jolt-field/src/lib.rs` doc-comment rewritten to walk through the layered hierarchy with a small diagram and one-line summaries. Cross-link to Hachi as the second consumer.
- `crates/jolt-transcript/src/lib.rs` doc-comment updated to note the `poseidon` feature flag and what builds without it.
- No `book/` changes required. The trait decomposition is a developer-facing API change with no zkVM-user-facing semantics.

## Execution

### Files to create

- `crates/jolt-field/src/additive_group.rs` (port `hachi::AdditiveGroup`)
- `crates/jolt-field/src/ring_core.rs` (new; algebraic ring layer between additive groups and fields)
- `crates/jolt-field/src/field_core.rs` (port `hachi::FieldCore`, minus ring-level operations moved to `RingCore` and without Hachi's `TWO_INV` constant)
- `crates/jolt-field/src/invertible.rs` (port `hachi::Invertible`)
- `crates/jolt-field/src/from_primitive_int.rs` (rename Hachi's `FromSmallInt` to `FromPrimitiveInt`, omitting Hachi's `digit_lut` helper)
- `crates/jolt-field/src/random_sampling.rs` (new orthogonal random sampling capability; method name remains `random`)
- `crates/jolt-field/src/transcript_challenge.rs` (new Fiat-Shamir challenge decoding capability; keeps transcript sampling distinct from RNG sampling and primitive integer embedding)
- `crates/jolt-field/src/mul_pow_2.rs` (new; `MulPow2` capability with current default body)
- `crates/jolt-field/src/mul_primitive_int.rs` (new; `MulPrimitiveInt` capability with current default bodies)
- `crates/jolt-field/src/canonical_bytes.rs` (byte-oriented canonical encoding with `to_bytes_le`)
- `crates/jolt-field/src/reducing_bytes.rs` (byte-oriented reducing constructor with `from_le_bytes_mod_order`)
- `crates/jolt-field/src/canonical_u64.rs` (new; checked extraction of canonical representatives that fit in `u64`)
- `crates/jolt-field/src/fixed_bytes.rs` (new; `FixedBytes<const N: usize>: CanonicalBytes + ReducingBytes + FixedByteSize` with fixed-array encoding helpers)
- `crates/jolt-field/src/fixed_byte_size.rs` (new; standalone `FixedByteSize` with `const NUM_BYTES`)
- `crates/jolt-field/src/canonical_bit_length.rs` (new; standalone `CanonicalBitLength` with `num_bits`)
- `crates/jolt-field/src/with_accumulator.rs` (new; `WithAccumulator: AdditiveGroup`)
- `crates/jolt-sumcheck/tests/mersenne61_compat.rs` (the BN254-free odd-prime compat test; downstream of `jolt-field`)
- `crates/jolt-field/tests/binary_field_core_compat.rs` (the characteristic-2 algebraic-layer compat test; verifies no odd-prime scaling leaks into `FieldCore`)

### Files to modify

- `crates/jolt-field/src/field.rs` — keep `trait Field { ... }` as the Jolt compatibility umbrella; restructure the supertrait list to include the named algebraic and orthogonal capabilities; make the trait body empty. Do not rename it in this PR.
- `crates/jolt-field/src/accumulator.rs` — rename `FieldAccumulator` to `RingAccumulator` and relax its element bound and `NaiveAccumulator<R>` from `F: Field` to the ring-level bounds they actually use.
- `crates/jolt-field/src/lib.rs` — module reorg, re-exports of every public trait.
- `crates/jolt-field/src/arkworks/bn254.rs` — decompose `impl Field for Fr` into per-trait `impl AdditiveGroup`, `impl RingCore`, `impl Invertible`, `impl FieldCore`, `impl FromPrimitiveInt`, `impl RandomSampling`, `impl CanonicalBytes`, `impl ReducingBytes`, `impl TranscriptChallenge`, `impl FixedBytes<32>`, `impl FixedByteSize`, `impl CanonicalBitLength`, `impl CanonicalU64`, `impl WithAccumulator`, `impl MulPow2`, `impl MulPrimitiveInt` for `Fr`. The existing concrete `Div` impls may remain centralized by `delegate_binop!(Div, div)`, but they are not required by `Field`. The `impl Field for Fr` and `impl FieldCore for Fr` become empty. `square` moves into `impl RingCore for Fr`; `inverse` and `inv_or_zero` move into `impl Invertible for Fr`; `random` moves into `impl RandomSampling for Fr`; byte serialization moves into `impl CanonicalBytes`/`impl ReducingBytes`/`impl FixedBytes<32>`; challenge decoding moves into `impl TranscriptChallenge`; `NUM_BYTES` moves into `impl FixedByteSize`; `num_bits` moves into `impl CanonicalBitLength`; checked canonical `u64` extraction moves into `impl CanonicalU64`; `mul_pow_2` moves into `impl MulPow2 for Fr`; `mul_u64`/`mul_i64`/`mul_u128`/`mul_i128` move into `impl MulPrimitiveInt for Fr`. Audit `#[inline]` on every moved method.
- `crates/jolt-field/src/arkworks/wide_accumulator.rs` — update `WideAccumulator` to implement `RingAccumulator<Element = Fr>` and confirm `WithAccumulator::Accumulator = WideAccumulator` for `Fr`.
- `crates/jolt-sumcheck/src/scalar.rs` (or `traits.rs`) — add the `SumcheckScalar` marker trait and blanket impl for types satisfying the bundled sumcheck capabilities.
- `crates/jolt-transcript/Cargo.toml` — make `light-poseidon`, `ark-bn254`, `ark-ff`, `ark-serialize` optional; add `poseidon` feature; `default = ["poseidon"]`; depend on `jolt-field` with `default-features = false`.
- `crates/jolt-transcript/src/lib.rs` — gate `pub use poseidon::*;` on `#[cfg(feature = "poseidon")]`.
- `crates/jolt-transcript/src/poseidon.rs` — file-level `#![cfg(feature = "poseidon")]`.
- `crates/jolt-transcript/src/{blake2b.rs,keccak.rs,digest.rs}` — remove unconditional default generic parameters that mention `jolt_field::Fr` when `poseidon`/`bn254` is disabled; keep the old defaults only under `#[cfg(feature = "poseidon")]`.

### Bound-relaxation sweep (minimal)

The crates touched are limited to those needed to make `crates/jolt-sumcheck/tests/mersenne61_compat.rs` compile and pass:

- `crates/jolt-sumcheck/src/{verifier.rs, batched_verifier.rs, round_proof.rs, claim.rs, proof.rs}`: `F: Field` → `F: SumcheckScalar` where a named sumcheck scalar bundle is clearer, or to the narrower structural bounds listed in "Bound relaxation (minimal sweep)" where the type truly does not need arithmetic.
- `crates/jolt-transcript/src/{transcript.rs, blanket.rs, digest.rs}`: challenge generation bounds move from `F: Field` to `F: TranscriptChallenge`; the blanket `AppendToTranscript` impl moves from `F: Field` to `F: CanonicalBytes + FixedByteSize`.
- `crates/jolt-poly/src/{multilinear.rs, binding.rs, dense.rs, eq.rs, eq_plus_one.rs, lt.rs, lagrange.rs, univariate.rs, compressed_univariate.rs}`: `F: Field` → `F: FieldCore + WithAccumulator` (or finer).
- `crates/jolt-openings/src/{claims.rs, reduction.rs}`: `F: Field` → `F: FieldCore + RandomSampling` where applicable.

`jolt-prover-legacy`, `jolt-r1cs`, `jolt-crypto`, `jolt-eval` keep their `F: Field` bounds unchanged.
Use clippy errors during step 6 (below) to verify the relaxation is consistent.

### Order of operations

1. Add the new trait modules in `crates/jolt-field/src/`. The crate compiles standalone with empty-of-impls traits.
2. Rename `FieldAccumulator` to `RingAccumulator` and relax `NaiveAccumulator` to slim ring bounds so non-`Field` rings can use the fallback accumulator.
3. Decompose `Fr`'s monolithic `Field` impl into per-trait impls in `arkworks/bn254.rs`. The top-bundle trait preserves Jolt compatibility through inherited named capabilities; ring-core methods (`square`) live only in `RingCore`, inversion methods (`inverse`, `inv_or_zero`) live only in `Invertible`, `FieldCore` is an empty marker impl, serialization/random-sampling/introspection live only in `CanonicalBytes`/`ReducingBytes`/`TranscriptChallenge`/`FixedBytes<32>`/`FixedByteSize`/`CanonicalBitLength`/`CanonicalU64`/`RandomSampling`, accumulator support lives only in `WithAccumulator`, and multiplication helper methods live only in `MulPow2` / `MulPrimitiveInt`. `cargo build -p jolt-field --features bn254` green.
4. Snapshot the wire-format and transcript-determinism golden blobs against `main` (capture them in a one-off test, paste expected values into `tests/`).
5. Add the `poseidon` feature in `jolt-transcript`. Verify `cargo build -p jolt-transcript --no-default-features` green.
6. Write `crates/jolt-sumcheck/tests/mersenne61_compat.rs`, including `Mersenne61` impls for every slim-hierarchy trait and the sumcheck verifier walkthrough. Run it and watch which `F: Field` bounds in shared code make it fail to compile.
7. Write `crates/jolt-field/tests/binary_field_core_compat.rs`, including a minimal characteristic-2 field that implements the algebraic layer and explicitly does not implement `MulPow2` or the Jolt `Field` umbrella. This is a guardrail from the Binius audit, not a claim that Jolt protocols are binary-field-ready in this PR.
8. Run the bound-relaxation sweep guided by step-6 errors. Use `cargo clippy --workspace --features host` and `--features host,zk` as the worklist. Iterate until the `Mersenne61` test compiles and clippy is clean in both modes.
9. Run the full gate: `cargo nextest run --workspace --features host`, `--features host,zk`, the explicit `muldiv` checks in both, and all listed benches.
10. Open the Hachi PR consuming `jolt-field` once this lands.

## Follow-Up Work

These were considered for inclusion in this PR and explicitly deferred for review-burden reasons.
File one separate spec/PR per item once this lands.

1. **Extension-field API port.** Add `LiftBase<F>` and `ExtField<F: FieldCore>: FieldCore + LiftBase<F> + FromPrimitiveInt`, plus concrete `Fp2<F, C>`/`Fp4<F, C2, C4>` types, `Fp2Config<F>`/`Fp4Config<F, C2>`, and pre-canned non-residue configs `NegOneNr` (for $p \equiv 3 \pmod 4$) and `TwoNr` (for $p \equiv 5 \pmod 8$). Verbatim port from `hachi/src/algebra/fields/{lift,ext}.rs`. Add `Fp2<Mersenne61, NegOneNr>` round-trip tests (add/mul/square/inverse, conjugate, norm). Needed for any future Jolt FRI / recursive verification / non-BN254 work. The Binius and Plonky audits show this follow-up must explicitly specify coefficient order, base-field byte order, and transcript absorption for extension elements.
2. **Hachi adoption of `jolt-field`.** Land in `LayerZero-Labs/hachi`. Implement the slim hierarchy for `Fp32`/`Fp64`/`Fp128`; implement `RingCore` for `CyclotomicRing<F, D>` where the coefficient type supports the needed field arithmetic. Delete `hachi/src/primitives/{arithmetic,transcript}.rs`, `hachi/src/algebra/uni_poly.rs`, and the verifier half of `hachi/src/protocol/sumcheck/`. Replace with re-exports from the slim hierarchy. Keep `PseudoMersenneField`/`SmoothFftField`/`Module` as hachi-only refinements.
3. **Granular bound tightening across `jolt-prover-legacy`.** Audit every `F: Field` bound in `jolt-prover-legacy` and relax to the minimum required (e.g. `F: RingCore + CanonicalBytes + FixedByteSize`). Mechanical, but blast radius is workspace-wide. Optional follow-up; no behavioral change.
4. **Transcript challenge sampling.** Once a second non-BN254 protocol uses the transcript crate, consider replacing the simple `TranscriptChallenge::from_challenge_bytes` hook with a challenger-side trait closer to Binius `CanSample<F>` or Plonky3 `CanSampleUniformBits<F>`, especially if uniform rejection sampling or native-sponge field elements matter.
5. **`Field` naming finalization.** If maintainers want a more precise name after this lands, file a rename-only PR.

## References

- `hachi/src/primitives/arithmetic.rs`: source of the slim hierarchy. `AdditiveGroup` (lines 11-25), `FieldCore` (lines 28-65, split here into `RingCore` plus field-only inversion, without carrying over Hachi's `TWO_INV` constant), `Invertible` (lines 71-74), `FromSmallInt` (lines 84-159, renamed to `FromPrimitiveInt` here and slimmed by omitting `digit_lut`), and `FieldSampling` (lines 180-183, renamed here to `RandomSampling`). Hachi's current `CanonicalField` (lines 166-175), `PseudoMersenneField` (lines 186-192), `SmoothFftField` (lines 212-220), and `Module` stay in Hachi.
- `hachi/src/algebra/fields/lift.rs` and `hachi/src/algebra/fields/ext.rs`: source of the deferred extension-field API (Follow-Up Work item 1).
- `crates/jolt-field/src/field.rs`: current monolithic trait, source of every method to relocate or keep.
- `crates/jolt-field/src/accumulator.rs`: current `FieldAccumulator` / `NaiveAccumulator` bound on the top `Field` trait; rename to `RingAccumulator` and move down to slim ring bounds.
- `crates/jolt-field/src/arkworks/bn254.rs`: current `impl Field for Fr` to decompose.
- `crates/jolt-field/src/arkworks/wide_accumulator.rs`: existing accumulator implementation; updates to `RingAccumulator<Element = Fr>` and remains `WithAccumulator::Accumulator` for `Fr`.
- `crates/jolt-transcript/Cargo.toml` lines 14-22: source of the unconditional `light-poseidon` / `ark-`* deps that the `poseidon` feature gates, and the `jolt-field` dependency edge that must use `default-features = false`.
- `hachi/src/protocol/sumcheck/mod.rs` lines 11-17: Hachi's standing duplication notice naming the shared-trait adoption as the planned fix.
- `binius64/crates/field/src/field.rs` and `binius64/crates/field/src/binary_field.rs`: binary-field audit source. Binius keeps characteristic-2 fields, inversion, serialization, and transcript sampling separate enough that Jolt must not put halving or integer-pow2 assumptions on `FieldCore`.
- `binius64/crates/transcript/src/transcript.rs` and `binius64/crates/utils/src/serialization.rs`: source for the distinction between field-specific byte deserialization and RNG sampling.
- `plonky2/field/src/types.rs` and `plonky2/plonky2/src/util/serialization/mod.rs`: source for canonical vs noncanonical constructors and base-field-coordinate extension serialization.
- `plonky3/field/src/field.rs`, `plonky3/field/src/integers.rs`, and `plonky3/challenger/src/serializing_challenger.rs`: source for `QuotientMap`, canonical vs unique encodings, raw byte streams, and challenger-specific sampling.
- `num_traits::PrimInt`: terminology reference considered when choosing the more explicit `FromPrimitiveInt` name.
- Mersenne prime $M_{61} = 2^{61} - 1$: the test field for `tests/mersenne61_compat.rs`. Same prime family used by Plonky3's `Mersenne31` for $M_{31} = 2^{31} - 1$.

