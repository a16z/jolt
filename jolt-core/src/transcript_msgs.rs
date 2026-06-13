//! jolt-core's field-typed Fiat–Shamir vocabulary over the spongefish NARG.
//!
//! jolt-core proof code is generic over `F: JoltField` (concretely `ark_bn254::Fr`,
//! or `TrackedFr` in profiling builds). spongefish's built-in `Encoding`/`Decoding`
//! are implemented only for the concrete arkworks types, so they can't be used
//! *directly* on a generic `F`. As a **host bridge** we therefore move values through
//! the NARG as a length-prefixed `CanonicalSerialize` blob (reusing
//! [`jolt_transcript::BytesMsg`]). Byte sponges (Blake2b/Keccak) ship the anonymous
//! `BytesMsg` blob; the field-aligned Poseidon sponge (`U = Fr`) routes frames through
//! the typed `FieldFrameMsg`/`CommitmentsMsg`/`RawBytesMsg` wrappers, whose NARG bytes
//! are `BytesMsg`-identical but whose sponge absorption is typed by value kind (spec §4.2).
//!
//! Challenge width is selected **per sponge type** (matching legacy `transcripts/`),
//! via per-sponge [`FsChallenge`] impls — NOT via a Cargo feature, so enabling
//! `transcript-poseidon` cannot change what a Blake2b/Keccak-backed state derives:
//! - **Byte sponges (Blake2b/Keccak):** 128-bit
//!   [`OptimizedChallenge::challenge_u128`]; `F::Challenge` is `MontU128Challenge`. They stay
//!   128-bit even under a hand-set `challenge-254-bit` — as in legacy `blake2b.rs`/`keccak.rs`.
//! - **Poseidon (`transcript-poseidon` → `challenge-254-bit`, maintainer decision on #1586):**
//!   GENUINE full-field `Fr` squeezes (`verifier_message::<ark_bn254::Fr>`), `F::Challenge` is
//!   `Mont254BitChallenge`. Poseidon's natural unit (128-bit truncation is costly for recursion);
//!   restores legacy `transcripts/poseidon.rs` so Poseidon works end-to-end and never hits its
//!   `unimplemented!()` `challenge_u128`.
//!
//! **What actually crosses the NARG:** prover-only payload — the sumcheck/uniskip round
//! polynomials via `write_scalars`/`read_scalars`, the witness-commitments and
//! untrusted-advice presence frames via `write_commitments`/`read_commitments`, and the
//! type-opaque ZK/BlindFold payloads (round-poly commitments, per-field BlindFold
//! values) via `write_slice`/`read_slice`. *Shared* values (flushed opening claims,
//! trusted/preprocessing commitments) are `absorb`'d ([`public_message`]); the dory
//! `joint_opening_proof` and — in non-ZK mode — the `opening_claims` stay **structured
//! proof fields** (see `JoltProof::narg` in `proof_serialization.rs` for the full
//! inventory).
//!
//! Three concerns, three traits:
//! - [`FsChallenge`] — squeezed verifier randomness; implemented per sponge type for
//!   `ProverState`/`VerifierState`, so prover and verifier share it.
//! - [`ProverFs`] — `absorb` shared values ([`public_message`], not shipped) and
//!   `write_slice` prover-only payload ([`prover_message`], into the NARG).
//! - [`VerifierFs`] — `absorb` the same shared values, and `read_slice` the prover
//!   payload back from the NARG in order.
//!
//! [`public_message`]: spongefish::ProverState::public_message
//! [`prover_message`]: spongefish::ProverState::prover_message

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_transcript::{
    serialize_slice, BytesMsg, OptimizedChallenge, ProverState, VerificationError,
    VerificationResult, VerifierState,
};

/// Absorbing shared values is the field-agnostic [`jolt_transcript::FsAbsorb`] surface
/// (`absorb` = spongefish `public_message`); jolt-core re-exports it so the whole
/// transcript vocabulary lives behind `crate::transcript_msgs`, and there is a *single*
/// absorb implementation shared with the modular crates (no hand-kept second copy).
pub use jolt_transcript::FsAbsorb;
use rand::{CryptoRng, RngCore};

use crate::field::JoltField;

// WIRE-FORMAT — RESOLVED (DEV-25 trigger #2 fired: the on-chain transpiler reader landed).
// Typed per-kind messages (`FieldFrameMsg`/`CommitmentsMsg`/`RawBytesMsg`) were introduced
// for the Poseidon (`U = Fr`) path, because the field-aligned sponge absorbs by value KIND;
// byte sponges keep the single anonymous `BytesMsg` path (see `impl_byte_sponge_fs!`).
// NARG wire bytes are identical on every path (8-byte LE length ‖ concatenated compressed
// serializations) — only Poseidon ABSORPTION is typed, so converting a call site never
// changes byte-sponge proof bytes. History: typed wrappers (DEV-12) → collapsed to
// `BytesMsg` (DEV-16) → partially reintroduced here.

/// Squeezed field challenges, shared by both transcript roles.
///
/// Implemented per sponge type for `ProverState`/`VerifierState` (Blake2b/Keccak:
/// 128-bit; Poseidon: full-field), so the prover and verifier derive challenges
/// identically and the width can never depend on feature unification.
///
/// ## Plain vs optimized — the #1 silent Fiat–Shamir hazard
///
/// (Describes the byte-sponge path — Blake2b/Keccak, the default build. Poseidon
/// leaves `challenge_u128` `unimplemented!()` and uses full-field `challenge-254-bit`;
/// see the module docs.)
///
/// Both methods draw the *same* 128-bit value ([`OptimizedChallenge::challenge_u128`]),
/// but they produce **different field elements** — they are NOT interchangeable:
/// - [`challenge_field`](Self::challenge_field) → the plain element `v` (`from_u128`),
///   all 128 bits.
/// - [`challenge_optimized`](Self::challenge_optimized) → the fast-multiply
///   [`MontU128Challenge`](crate::field::challenge::MontU128Challenge) form, which
///   **masks the top 3 bits** (125-bit) and represents the element `v_masked · 2¹²⁸`.
///   So it differs from `challenge_field` in *value*, not merely in the returned type.
///
/// Soundness therefore requires the prover and verifier to call the *same* method at a
/// given transcript position; mixing them silently diverges the challenge. This mirrors
/// the modular [`jolt_transcript::FsChallenge`], but **the names are inverted** there:
/// jolt-core's `challenge_field`/`challenge_optimized` map to that trait's
/// `challenge_scalar` (plain) / `challenge` (optimized) — and jolt-core's optimized form
/// additionally returns the distinct `F::Challenge` type, not `F`. Map by *semantics*,
/// never by like name.
pub trait FsChallenge<F: JoltField> {
    /// Squeeze the **plain** field challenge `v` (`from_u128`), keeping all 128 bits.
    /// Not interchangeable with [`challenge_optimized`](Self::challenge_optimized) — see the
    /// trait docs. (Byte-sponge semantics; full-field `Fr` on the Poseidon path.)
    fn challenge_field(&mut self) -> F;

    /// Squeeze the fast-multiply [`JoltField::Challenge`] form (125-bit, `v_masked · 2¹²⁸`)
    /// — a *different* value than [`challenge_field`](Self::challenge_field). **Byte-sponge
    /// only:** the Poseidon path has no masking, so it returns the same full `Fr` as
    /// `challenge_field` (wrapped in `Mont254BitChallenge`).
    fn challenge_optimized(&mut self) -> F::Challenge;

    /// `n` independent field challenges.
    fn challenge_vec(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.challenge_field()).collect()
    }

    /// `n` independent optimized challenges.
    fn challenge_optimized_vec(&mut self, n: usize) -> Vec<F::Challenge> {
        (0..n).map(|_| self.challenge_optimized()).collect()
    }

    /// Powers `(1, q, q², …, q^(n-1))` from a single squeezed `q`.
    fn challenge_powers(&mut self, n: usize) -> Vec<F> {
        let q = self.challenge_field();
        let mut powers = Vec::with_capacity(n);
        let mut cur = F::from_u64(1);
        for _ in 0..n {
            powers.push(cur);
            cur *= q;
        }
        powers
    }
}

// `transcript-poseidon` forces `challenge-254-bit`, so `F::Challenge` is the
// `#[repr(transparent)]` `Mont254BitChallenge` the full-field transmute below relies on.
// Enforce that coupling (else the transmute could hit the 16-byte `MontU128Challenge`).
#[cfg(all(feature = "transcript-poseidon", not(feature = "challenge-254-bit")))]
compile_error!(
    "transcript-poseidon requires challenge-254-bit (F::Challenge = Mont254BitChallenge); \
     the Poseidon full-field challenge transmute in transcript_msgs depends on it"
);

/// Byte-sponge path (Blake2b/Keccak): 128-bit [`OptimizedChallenge::challenge_u128`].
/// They stay 128-bit even under a hand-set `challenge-254-bit` (a 128-bit value in the
/// wider type) — dispatching on the *sponge type*, not a Cargo feature, preserves
/// legacy's per-sponge behaviour and keeps challenge derivation independent of feature
/// unification across the workspace.
impl<F: JoltField, R: RngCore + CryptoRng> FsChallenge<F>
    for ProverState<jolt_transcript::Blake2b512, R>
{
    fn challenge_field(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }
}

impl<F: JoltField> FsChallenge<F> for VerifierState<'_, jolt_transcript::Blake2b512> {
    fn challenge_field(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }
}

impl<F: JoltField, R: RngCore + CryptoRng> FsChallenge<F>
    for ProverState<jolt_transcript::Keccak, R>
{
    fn challenge_field(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }
}

impl<F: JoltField> FsChallenge<F> for VerifierState<'_, jolt_transcript::Keccak> {
    fn challenge_field(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }
}

/// Reconstruct `F` from a native BN254 `Fr` (runs per challenge / per read scalar).
///
/// When `F` *is* `ark_bn254::Fr` (the production instantiation) this is the identity:
/// a single `transmute_copy` skips the `into_bigint → LE bytes → from_le_bytes_mod_order`
/// round-trip. For any other `F` (e.g. the `repr(Rust)` `TrackedFr` profiling newtype,
/// whose in-memory layout is NOT guaranteed to match `Fr`) it falls back to the canonical
/// 32-byte LE round-trip. `from_bytes` is `from_le_bytes_mod_order` and the bytes are
/// `< modulus`, so the round-trip is exact. The `TypeId` branch is on a monomorphized
/// `F`, so the compiler folds it to a single path per instantiation (no runtime cost).
fn native_to_field<F: JoltField>(native: ark_bn254::Fr) -> F {
    use ark_ff::PrimeField;
    if core::any::TypeId::of::<F>() == core::any::TypeId::of::<ark_bn254::Fr>() {
        // SAFETY: the `TypeId` guard proves `F` is exactly `ark_bn254::Fr` (both are
        // `'static`), so this reinterprets `Fr → Fr` — an identity copy of identical size
        // and layout. Equals the round-trip below: `into_bigint`/`from_le_bytes_mod_order`
        // compose to the identity on a canonical `Fr` value.
        return unsafe { core::mem::transmute_copy::<ark_bn254::Fr, F>(&native) };
    }
    let limbs = native.into_bigint().0;
    let mut le = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        le[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    F::from_bytes(&le)
}

/// The inverse of [`native_to_field`]: a native BN254 `Fr` unit carrying the value of `F`
/// (sumcheck/uni-skip round-poly coeffs — runs per element in the `write_scalars` hot path).
///
/// When `F` *is* `ark_bn254::Fr` (the production instantiation) this is the identity:
/// a single `transmute_copy` replaces the `serialize_compressed → deserialize_compressed`
/// no-op round-trip. For any other `F` (e.g. the `repr(Rust)` `TrackedFr` profiling newtype)
/// it falls back to the canonical serialization round-trip, exact for every `F` this crate
/// instantiates (BN254-backed, `NUM_BYTES == 32`). The `TypeId` branch is on a monomorphized
/// `F`, so the compiler folds it to a single path per instantiation (no runtime cost).
#[expect(
    clippy::expect_used,
    reason = "32-byte canonical field serialization round-trips infallibly"
)]
fn field_to_native<F: JoltField>(value: &F) -> ark_bn254::Fr {
    use ark_serialize::CanonicalDeserialize;
    if core::any::TypeId::of::<F>() == core::any::TypeId::of::<ark_bn254::Fr>() {
        // SAFETY: the `TypeId` guard proves `F` is exactly `ark_bn254::Fr` (both are
        // `'static`), so this reinterprets `Fr → Fr` — an identity copy of identical size
        // and layout. Equals the round-trip below: `serialize_compressed` writes `Fr`'s
        // canonical (non-Montgomery) LE bytes and `deserialize_compressed::<Fr>` reads them
        // back into the same in-memory `Fr`, i.e. the identity on the value.
        return unsafe { core::mem::transmute_copy::<F, ark_bn254::Fr>(value) };
    }
    let mut buf = [0u8; 32];
    value
        .serialize_compressed(&mut buf[..])
        .expect("32-byte field element serializes into a 32-byte buffer");
    ark_bn254::Fr::deserialize_compressed(&buf[..]).expect("canonical field bytes parse as Fr")
}

/// Reinterpret a full field element as its `F::Challenge` (Poseidon-only: there it's the
/// `#[repr(transparent)]` `Mont254BitChallenge<F>`). One shared spot so the `unsafe`
/// prover/verifier reinterpretation can't drift.
fn wrap_full_field<F: JoltField>(f: F) -> F::Challenge {
    // Turn a layout mismatch into a compile error — the check `transmute` makes but
    // `transmute_copy` skips for generics (e.g. rejects the 16-byte `MontU128Challenge`).
    // Instantiated only by the Poseidon impls below, so a build whose `F::Challenge` is
    // `MontU128Challenge` only fails if it actually uses a Poseidon-backed transcript.
    const { assert!(core::mem::size_of::<F>() == core::mem::size_of::<F::Challenge>()) }
    // SAFETY: `F::Challenge` is the `#[repr(transparent)]` `Mont254BitChallenge<F>` newtype
    // of `F` — identical size (asserted) and layout. Mirrors legacy `challenge_scalar_optimized`.
    unsafe { core::mem::transmute_copy::<F, F::Challenge>(&f) }
}

/// Poseidon path (`transcript-poseidon` → `challenge-254-bit`): the optimized challenge is a
/// GENUINE full-field `Fr` (Poseidon's natural unit), not a 128-bit truncation, so Poseidon
/// never hits its `unimplemented!()` `challenge_u128` (kept unimplemented per the maintainer's
/// decision on #1586). With the field-aligned `U = Fr` sponge the squeeze is **one native unit
/// = exactly one permute** ([`jolt_transcript::NativeChallenge`], identity decode) — exactly
/// uniform under the ideal-permutation model, with zero in-circuit decode cost. NB: here
/// `challenge_field` and `challenge_optimized` return the SAME value (no `v·2¹²⁸` masking).
impl<F, R> FsChallenge<F> for ProverState<jolt_transcript::PoseidonSponge, R>
where
    F: JoltField,
    R: RngCore + CryptoRng,
{
    fn challenge_field(&mut self) -> F {
        native_to_field(ProverState::verifier_message::<jolt_transcript::NativeChallenge>(self).0)
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        let f: F = self.challenge_field();
        wrap_full_field(f)
    }
}

impl<F: JoltField> FsChallenge<F> for VerifierState<'_, jolt_transcript::PoseidonSponge> {
    fn challenge_field(&mut self) -> F {
        native_to_field(VerifierState::verifier_message::<jolt_transcript::NativeChallenge>(self).0)
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        let f: F = self.challenge_field();
        wrap_full_field(f)
    }
}

/// Decode every `T` in a single self-delimiting frame.
///
/// The frame body length (carried by [`BytesMsg`]'s prefix, which the sponge already
/// bounds to the remaining NARG) determines the count — so a sequence is read back
/// without shipping or trusting a separate length, and the per-round element count
/// may vary. This is also why we never deserialize a `Vec<T>` directly:
/// `CanonicalDeserialize` for `Vec` reads its OWN length prefix from the NARG and
/// pre-allocates from it, so an adversarial proof could capacity-overflow panic / OOM.
/// Here every allocation is bounded by the actual frame bytes, which are bounded by the
/// actual proof.
fn read_all<T: CanonicalDeserialize>(body: &[u8]) -> VerificationResult<Vec<T>> {
    let mut cursor = body;
    let mut out = Vec::new();
    while !cursor.is_empty() {
        out.push(T::deserialize_compressed(&mut cursor).map_err(|_| VerificationError)?);
    }
    Ok(out)
}

/// Prover-side message vocabulary over the spongefish NARG.
///
/// The typed `write_scalars`/`write_commitments` variants exist because the
/// `Fr`-unit Poseidon sponge absorbs by value *kind* (spec §4.2) and
/// `write_slice` is type-opaque. Their NARG bytes are identical to
/// [`write_slice`](Self::write_slice) on every sponge — only what the sponge
/// absorbs differs on the Poseidon path — so converting a call site never
/// changes the proof bytes of the byte-sponge builds.
pub trait ProverFs<F: JoltField>: FsChallenge<F> + FsAbsorb {
    /// Write a sequence of prover-only values as one self-delimiting frame
    /// (read back with [`VerifierFs::read_slice`]). No length prefix is shipped; the
    /// frame is bounded by the NARG, so the per-round element count may vary.
    /// On the Poseidon path the sponge absorbs the frame under the byte rule —
    /// the right classification for type-opaque payloads (e.g. ZK G1 points,
    /// whose Fq coordinates must never embed as native `Fr` units).
    fn write_slice<T: CanonicalSerialize>(&mut self, values: &[T]);

    /// [`write_slice`](Self::write_slice) for a frame of **field elements**
    /// (sumcheck/uni-skip round polynomials). Poseidon absorbs the count-led
    /// field frame `[Fr(2k+1), e₁, …, e_k]`; read back with
    /// [`VerifierFs::read_scalars`].
    fn write_scalars(&mut self, values: &[F]) {
        self.write_slice(values);
    }

    /// [`write_slice`](Self::write_slice) for a frame of **commitments**
    /// (witness commitments, the advice presence frame). Poseidon absorbs a
    /// frame count unit `Fr(2k+1)` then per-commitment byte-rule groups (one
    /// GT ↦ `[Fr(768), 13 chunks]`); an empty frame is the count-led
    /// `[Fr(1)]`. Read back with [`VerifierFs::read_commitments`].
    fn write_commitments<T: CanonicalSerialize + Clone>(&mut self, values: &[T]) {
        self.write_slice(values);
    }
}

/// Byte-sponge `ProverFs`/`VerifierFs` impls. Per-sponge (like [`FsChallenge`])
/// rather than blanket over `H: DuplexSpongeInterface<U = u8>`: coherence
/// cannot prove a foreign-sponge blanket disjoint from the concrete
/// `PoseidonSponge` (`U = Fr`) impls below, because the `U` projection lives in
/// a foreign crate.
macro_rules! impl_byte_sponge_fs {
    ($sponge:ty) => {
        impl<F, R> ProverFs<F> for ProverState<$sponge, R>
        where
            F: JoltField,
            R: RngCore + CryptoRng,
        {
            fn write_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
                self.prover_message(&BytesMsg(serialize_slice(values)));
            }
        }

        impl<F: JoltField> VerifierFs<F> for VerifierState<'_, $sponge> {
            fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>> {
                let bytes = self.prover_message::<BytesMsg>()?;
                read_all(&bytes.0)
            }
        }
    };
}

/// Poseidon (`U = Fr`) NARG path: every frame ships the **same bytes** as the
/// byte-sponge path (`8-byte LE length ‖ concatenated compressed
/// serializations`), while the sponge absorbs the typed unit encoding of the
/// decoded values (spec §4.2). The seam is spongefish's
/// `prover_message`/`prover_message::<T>()`, which couples "write/read these
/// NARG bytes" with "absorb `T`'s `Encoding<[H::U]>`" — so the typed message
/// types ([`jolt_transcript::FieldFrameMsg`] / [`jolt_transcript::CommitmentsMsg`]
/// / [`jolt_transcript::RawBytesMsg`]) carry a `BytesMsg`-identical NARG codec
/// next to their `Fr`-unit encoding, and prover/verifier absorption cannot
/// drift from the shipped bytes.
impl<F, R> ProverFs<F> for ProverState<jolt_transcript::PoseidonSponge, R>
where
    F: JoltField,
    R: RngCore + CryptoRng,
{
    fn write_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
        self.prover_message(&jolt_transcript::RawBytesMsg(serialize_slice(values)));
    }

    fn write_scalars(&mut self, values: &[F]) {
        let units: Vec<ark_bn254::Fr> = values.iter().map(field_to_native).collect();
        self.prover_message(&jolt_transcript::FieldFrameMsg(units));
    }

    fn write_commitments<T: CanonicalSerialize + Clone>(&mut self, values: &[T]) {
        self.prover_message(&jolt_transcript::CommitmentsMsg(values.to_vec()));
    }
}

/// Verifier-side message vocabulary over the spongefish NARG.
///
/// The typed `read_scalars`/`read_commitments` variants mirror
/// [`ProverFs::write_scalars`]/[`ProverFs::write_commitments`]; a frame must be
/// read with the same kind it was written with or the Poseidon sponges diverge.
pub trait VerifierFs<F: JoltField>: FsChallenge<F> + FsAbsorb {
    /// Read every value in the next frame written by [`ProverFs::write_slice`]; the
    /// count is the frame's (self-delimiting, so a varying per-round length is fine).
    /// Bounded allocation — see [`read_all`].
    fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>>;

    /// Read a frame that must contain exactly one value — the single-element analogue of
    /// [`read_slice`]. Errors if the frame is empty or carries more than one value, closing
    /// the silent-truncation gap of `read_slice()?.into_iter().next()`.
    fn read_single<T: CanonicalDeserialize>(&mut self) -> VerificationResult<T> {
        match <[T; 1]>::try_from(self.read_slice::<T>()?) {
            Ok([value]) => Ok(value),
            Err(_) => Err(VerificationError),
        }
    }

    /// Read back a [`ProverFs::write_scalars`] frame of field elements.
    fn read_scalars(&mut self) -> VerificationResult<Vec<F>> {
        self.read_slice()
    }

    /// Read back a [`ProverFs::write_commitments`] frame of commitments.
    fn read_commitments<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> VerificationResult<Vec<T>> {
        self.read_slice()
    }
}

impl_byte_sponge_fs!(jolt_transcript::Blake2b512);
impl_byte_sponge_fs!(jolt_transcript::Keccak);

impl<F: JoltField> VerifierFs<F> for VerifierState<'_, jolt_transcript::PoseidonSponge> {
    fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>> {
        let bytes = self.prover_message::<jolt_transcript::RawBytesMsg>()?;
        read_all(&bytes.0)
    }

    fn read_scalars(&mut self) -> VerificationResult<Vec<F>> {
        let frame = self.prover_message::<jolt_transcript::FieldFrameMsg>()?;
        Ok(frame.0.into_iter().map(native_to_field).collect())
    }

    fn read_commitments<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> VerificationResult<Vec<T>> {
        Ok(self
            .prover_message::<jolt_transcript::CommitmentsMsg<T>>()?
            .0)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use jolt_transcript::{prover_transcript, verifier_transcript, Blake2b512, VerifierTranscript};

    const SESSION: &[u8] = b"jolt-transcript-msgs-test/v1";
    type Bl = Blake2b512;

    /// A frame of values round-trips through the NARG and `check_eof` succeeds.
    #[test]
    fn write_then_read_round_trips() {
        let mut r = test_rng();
        let scalars: Vec<Fr> = (0..5).map(|_| Fr::random(&mut r)).collect();

        let instance = [7u8; 32];
        let mut p = prover_transcript(SESSION, instance, Bl::default());
        ProverFs::<Fr>::write_slice(&mut p, &scalars);
        let narg = p.narg_string().to_vec();

        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        let read: Vec<Fr> = VerifierFs::<Fr>::read_slice(&mut v).unwrap();
        assert_eq!(read, scalars);
        VerifierTranscript::<Bl>::check_eof(v).unwrap();
    }

    /// Mirrors a batched non-ZK sumcheck (`BatchedSumcheck::prove`/`verify`):
    ///   shared input claims → `absorb` (recomputed by verifier; not in NARG)
    ///   batching coeffs      → `challenge_vec`
    ///   per round: round poly → `write_slice` (prover-only, in NARG); challenge → `challenge_optimized`
    ///   flushed claims       → `absorb` (shared; not in NARG)
    /// Verifies the verifier reconstructs every prover-only value AND both sides
    /// derive identical challenges, then `check_eof`.
    #[test]
    fn sumcheck_shaped_narg_round_trips_and_challenges_agree() {
        let mut r = test_rng();
        let n_instances = 2usize;
        let n_rounds = 4usize;
        let input_claims: Vec<Fr> = (0..n_instances).map(|_| Fr::random(&mut r)).collect();
        let round_polys: Vec<Vec<Fr>> = (0..n_rounds)
            .map(|i| (0..(2 + i)).map(|_| Fr::random(&mut r)).collect())
            .collect();
        let flushed_claims: Vec<Fr> = (0..5).map(|_| Fr::random(&mut r)).collect();
        let instance = [0x5C; 32];

        let mut p = prover_transcript(SESSION, instance, Bl::default());
        for c in &input_claims {
            FsAbsorb::absorb(&mut p, c);
        }
        let p_batching = FsChallenge::<Fr>::challenge_vec(&mut p, n_instances);
        let mut p_round_challenges = Vec::with_capacity(n_rounds);
        for poly in &round_polys {
            ProverFs::<Fr>::write_slice(&mut p, poly);
            p_round_challenges.push(FsChallenge::<Fr>::challenge_optimized(&mut p));
        }
        // Flushed opening claims are SHARED (both sides hold them) → `absorb`, matching
        // the real `flush_to_transcript` (opening_proof.rs); they are NOT in the NARG.
        for c in &flushed_claims {
            FsAbsorb::absorb(&mut p, c);
        }
        let narg = p.narg_string().to_vec();

        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        for c in &input_claims {
            FsAbsorb::absorb(&mut v, c);
        }
        let v_batching = FsChallenge::<Fr>::challenge_vec(&mut v, n_instances);
        let mut v_round_challenges = Vec::with_capacity(n_rounds);
        for expected in &round_polys {
            // round-poly counts vary, so read the self-delimiting frame (like real sumcheck)
            let read: Vec<Fr> = VerifierFs::<Fr>::read_slice(&mut v).unwrap();
            assert_eq!(
                &read, expected,
                "round poly reconstructed incorrectly from NARG"
            );
            v_round_challenges.push(FsChallenge::<Fr>::challenge_optimized(&mut v));
        }
        // Verifier absorbs the same shared flushed claims (not read from the NARG).
        for c in &flushed_claims {
            FsAbsorb::absorb(&mut v, c);
        }
        VerifierTranscript::<Bl>::check_eof(v).unwrap();

        assert_eq!(p_batching, v_batching, "batching challenges diverged");
        assert_eq!(
            p_round_challenges, v_round_challenges,
            "round challenges diverged"
        );
    }

    /// Trailing garbage must be rejected by `check_eof` (malleability guard).
    #[test]
    fn trailing_garbage_is_rejected_by_check_eof() {
        let instance = [1u8; 32];
        let mut p = prover_transcript(SESSION, instance, Bl::default());
        ProverFs::<Fr>::write_slice(&mut p, &[Fr::from(42u64)]);
        let mut narg = p.narg_string().to_vec();
        narg.push(0xFF);

        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        let _: Vec<Fr> = VerifierFs::<Fr>::read_slice(&mut v).unwrap();
        assert!(VerifierTranscript::<Bl>::check_eof(v).is_err());
    }

    /// Reading fewer frames than the prover wrote cannot silently pass: the unconsumed
    /// frame is caught by `check_eof`, so a desynced/short read order is rejected.
    #[test]
    fn under_reading_narg_is_rejected_by_check_eof() {
        let mut r = test_rng();
        let frame_a: Vec<Fr> = vec![Fr::random(&mut r)];
        let frame_b: Vec<Fr> = (0..4).map(|_| Fr::random(&mut r)).collect();
        let instance = [0x0D; 32];

        let mut p = prover_transcript(SESSION, instance, Bl::default());
        ProverFs::<Fr>::write_slice(&mut p, &frame_a);
        ProverFs::<Fr>::write_slice(&mut p, &frame_b);
        let narg = p.narg_string().to_vec();

        // Verifier reads only the first frame, leaving frame_b unconsumed.
        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        let read_a: Vec<Fr> = VerifierFs::<Fr>::read_slice(&mut v).unwrap();
        assert_eq!(read_a, frame_a);
        assert!(
            VerifierTranscript::<Bl>::check_eof(v).is_err(),
            "an unconsumed NARG frame must be rejected"
        );
    }

    /// Under `transcript-poseidon` the optimized challenge is a GENUINE full-field
    /// `Fr`, not a 128-bit truncation, and prover/verifier must agree on it. Exercises
    /// the Poseidon `FsChallenge` impls (the legacy-restored full-field path) on the
    /// actual `PoseidonSponge`.
    #[cfg(feature = "transcript-poseidon")]
    #[test]
    fn full_field_optimized_challenge_agrees_and_is_not_truncated() {
        use ark_ff::PrimeField;
        use jolt_transcript::PoseidonSponge;
        let instance = [0x2E; 32];
        let n = 8usize;

        let mut p = prover_transcript(SESSION, instance, PoseidonSponge::default());
        let p_ch: Vec<<Fr as JoltField>::Challenge> = (0..n)
            .map(|_| FsChallenge::<Fr>::challenge_optimized(&mut p))
            .collect();
        let narg = p.narg_string().to_vec();

        let mut v = verifier_transcript(SESSION, instance, PoseidonSponge::default(), &narg);
        for expected in &p_ch {
            let got = FsChallenge::<Fr>::challenge_optimized(&mut v);
            assert_eq!(*expected, got, "full-field optimized challenge diverged");
        }
        VerifierTranscript::<PoseidonSponge>::check_eof(v).unwrap();

        // A 128-bit-truncated path would leave the top two limbs (bits 128+) zero
        // for every challenge; a genuine full-field squeeze sets them w.h.p.
        let uses_high_bits = p_ch.iter().any(|c| {
            let f: Fr = (*c).into();
            let limbs = f.into_bigint().0;
            limbs[2] != 0 || limbs[3] != 0
        });
        assert!(
            uses_high_bits,
            "optimized challenge is 128-bit truncated, not full-field 254-bit"
        );
    }
}
