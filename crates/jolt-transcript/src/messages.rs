//! Field-typed Fiat–Shamir vocabulary for the *verifier-only* modular crates
//! (jolt-crypto / jolt-openings / jolt-sumcheck / jolt-hyperkzg / jolt-dory /
//! jolt-blindfold / jolt-verifier).
//!
//! These crates are **symmetric** (the spec's "Option-A" shape): a value is
//! `absorb`'d on *both* the (test-)prover and the verifier side and never
//! written to a NARG. There is intentionally **no** `prover_message` /
//! `check_eof` here — the modular verify paths consume a *structured* proof
//! and re-absorb its values to derive challenges. The malleability guard lives
//! at the structured-proof deserialize boundary, outside this surface.
//!
//! This is the field-agnostic sibling of jolt-core's internal
//! [`transcript_msgs`](../../jolt-core/src/transcript_msgs.rs) vocabulary: it
//! is generic over `jolt_field::Field` (the modular crates' field newtype)
//! rather than jolt-core's `JoltField`, and it does NOT drive a NARG. It emits
//! the identical sponge messages jolt-core does
//! (`public_message(&BytesMsg(serialize_compressed(value)))`; a 16-byte
//! challenge squeeze), so a future jolt-core cross-verifier agrees by
//! construction.
//!
//! Two concerns, two traits:
//! - [`FsAbsorb`] — absorb shared values (spongefish `public_message`).
//! - [`FsChallenge`] — squeeze field challenges.
//!
//! [`FsTranscript`] combines them; it is the bound that replaces the legacy
//! `Transcript<Challenge = F>` facade at every modular callsite.
//!
//! ## Challenge embedding — the #1 silent Fiat–Shamir hazard
//!
//! [`challenge`](FsChallenge::challenge) / [`challenge_vector`](FsChallenge::challenge_vector)
//! use the **optimized** `MontU128` embedding ([`from_challenge_bytes`], the
//! field element `v · 2¹²⁸`), matching jolt-core's `challenge_optimized`.
//! [`challenge_scalar`](FsChallenge::challenge_scalar) /
//! [`challenge_scalar_powers`](FsChallenge::challenge_scalar_powers) use the
//! **plain** embedding ([`from_u128`], the field element `v`), matching
//! jolt-core's `challenge_field`. The legacy facade drew the *same* line
//! (`challenge` → `from_challenge_bytes`, `challenge_scalar` →
//! `from_scalar_challenge_bytes`), so a migrated callsite keeps whichever
//! method it already called and stays value-consistent. Collapsing the two to
//! one embedding stays internally consistent but silently diverges from
//! jolt-core at the optimized sites (sumcheck round challenges, uni-skip `r0`,
//! Spartan `tau`, opening-RLC `rho`, …).
//!
//! **Byte sponges only (Poseidon caveat).** This 128-bit `MontU128` vocabulary matches
//! jolt-core's `challenge_optimized` for the byte sponges (Blake2b/Keccak) — the only ones
//! modular consumers instantiate. [`FsChallenge`] is deliberately implemented per byte-sponge
//! type and NOT for `PoseidonSponge` (which uses full-field `challenge-254-bit`; maintainer
//! decision on #1586), so instantiating a modular verifier over a Poseidon-backed state is a
//! **compile error** rather than a latent runtime panic. A full-field model verifier for
//! Poseidon is the deferred gnark/on-chain follow-up.
//!
//! [`from_challenge_bytes`]: jolt_field::TranscriptChallenge::from_challenge_bytes
//! [`from_u128`]: jolt_field::FromPrimitiveInt::from_u128

use ark_serialize::CanonicalSerialize;
use jolt_field::Field;
use rand::{CryptoRng, RngCore};
use spongefish::{DuplexSpongeInterface, ProverState, VerifierState};

use crate::codec::BytesMsg;
#[cfg(any(feature = "transcript-blake2b", feature = "transcript-keccak"))]
use crate::prover::OptimizedChallenge;

#[expect(clippy::expect_used)]
fn serialize_one<T: CanonicalSerialize>(value: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(value.compressed_size());
    value
        .serialize_compressed(&mut buf)
        .expect("CanonicalSerialize into a Vec is infallible");
    buf
}

/// Compressed serialization of every element, concatenated with no length prefix.
///
/// The single shared encoder for "a sequence of `CanonicalSerialize` values as one
/// frame": used here by [`FsAbsorb::absorb_slice`] and by jolt-core's NARG
/// `write_slice`, so both produce byte-identical frames and cannot drift.
#[expect(clippy::expect_used)]
pub fn serialize_slice<T: CanonicalSerialize>(values: &[T]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.iter().map(CanonicalSerialize::compressed_size).sum());
    for v in values {
        v.serialize_compressed(&mut buf)
            .expect("CanonicalSerialize into a Vec is infallible");
    }
    buf
}

/// Absorb shared values into the sponge (spongefish `public_message`); emits
/// no NARG bytes.
///
/// Shared by both transcript roles so symmetric modular code absorbs
/// identically on the prover and verifier sides. Every absorb is one
/// length-prefixed [`BytesMsg`], so `absorb(a) ; absorb(b)` stays distinct
/// from `absorb(a ‖ b)`.
pub trait FsAbsorb {
    /// Absorb one `CanonicalSerialize` value (e.g. a commitment) as a single
    /// message, via its compressed serialization.
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T);

    /// Absorb a slice of `CanonicalSerialize` values as a single message
    /// (their compressed serializations concatenated).
    fn absorb_slice<T: CanonicalSerialize>(&mut self, values: &[T]);

    /// Absorb raw bytes as a single message.
    fn absorb_bytes(&mut self, bytes: &[u8]);

    /// Absorb one field element as a single message. Uses `to_bytes_le`, which
    /// equals `serialize_compressed` for BN254 `Fr`, so this matches
    /// [`absorb`](Self::absorb) for the same value.
    fn absorb_field<F: Field>(&mut self, value: &F) {
        self.absorb_bytes(&value.to_bytes_le_vec());
    }

    /// Absorb a slice of field elements as a *single* message (their
    /// little-endian bytes concatenated).
    fn absorb_field_slice<F: Field>(&mut self, values: &[F]) {
        let mut buf = Vec::with_capacity(values.len() * F::NUM_BYTES);
        for v in values {
            buf.extend_from_slice(&v.to_bytes_le_vec());
        }
        self.absorb_bytes(&buf);
    }
}

impl<H, R> FsAbsorb for ProverState<H, R>
where
    H: DuplexSpongeInterface<U = u8>,
    R: RngCore + CryptoRng,
{
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
        self.public_message(&BytesMsg(serialize_one(value)));
    }

    fn absorb_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
        self.public_message(&BytesMsg(serialize_slice(values)));
    }

    fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.public_message(&BytesMsg(bytes.to_vec()));
    }
}

impl<H> FsAbsorb for VerifierState<'_, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
        self.public_message(&BytesMsg(serialize_one(value)));
    }

    fn absorb_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
        self.public_message(&BytesMsg(serialize_slice(values)));
    }

    fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.public_message(&BytesMsg(bytes.to_vec()));
    }
}

/// Squeeze field challenges. Implemented per byte-sponge type for
/// `ProverState` / `VerifierState` (Blake2b/Keccak), so prover and verifier
/// derive challenges identically. Deliberately NOT implemented for
/// `PoseidonSponge` — see the module docs.
///
/// See the module docs for the optimized-vs-plain embedding distinction.
pub trait FsChallenge<F: Field> {
    /// Squeeze an **optimized** (`MontU128`-embedded, `v · 2¹²⁸`) challenge.
    fn challenge(&mut self) -> F;

    /// Squeeze a **plain** (`from_u128`, `v`) scalar challenge.
    fn challenge_scalar(&mut self) -> F;

    /// `n` independent optimized challenges.
    fn challenge_vector(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.challenge()).collect()
    }

    /// Powers `(1, γ, γ², …, γⁿ⁻¹)` from a single **plain** squeezed `γ`.
    fn challenge_scalar_powers(&mut self, n: usize) -> Vec<F> {
        let gamma = self.challenge_scalar();
        let mut powers = vec![F::from_u64(1); n];
        for index in 1..n {
            powers[index] = powers[index - 1] * gamma;
        }
        powers
    }
}

#[cfg(any(feature = "transcript-blake2b", feature = "transcript-keccak"))]
fn optimized_embed<F: Field>(v: u128) -> F {
    F::from_challenge_bytes(&v.to_le_bytes())
}

#[cfg(feature = "transcript-blake2b")]
impl<F: Field, R: RngCore + CryptoRng> FsChallenge<F>
    for ProverState<spongefish::instantiations::Blake2b512, R>
{
    fn challenge(&mut self) -> F {
        optimized_embed(self.challenge_u128())
    }

    fn challenge_scalar(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }
}

#[cfg(feature = "transcript-blake2b")]
impl<F: Field> FsChallenge<F> for VerifierState<'_, spongefish::instantiations::Blake2b512> {
    fn challenge(&mut self) -> F {
        optimized_embed(self.challenge_u128())
    }

    fn challenge_scalar(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }
}

#[cfg(feature = "transcript-keccak")]
impl<F: Field, R: RngCore + CryptoRng> FsChallenge<F>
    for ProverState<spongefish::instantiations::Keccak, R>
{
    fn challenge(&mut self) -> F {
        optimized_embed(self.challenge_u128())
    }

    fn challenge_scalar(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }
}

#[cfg(feature = "transcript-keccak")]
impl<F: Field> FsChallenge<F> for VerifierState<'_, spongefish::instantiations::Keccak> {
    fn challenge(&mut self) -> F {
        optimized_embed(self.challenge_u128())
    }

    fn challenge_scalar(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }
}

/// The combined absorb + challenge surface that replaces the legacy
/// `Transcript<Challenge = F>` facade bound at modular callsites.
///
/// Blanket-implemented for every state that is both [`FsAbsorb`] and
/// [`FsChallenge<F>`] — i.e. every `ProverState` / `VerifierState` over a
/// supported sponge.
pub trait FsTranscript<F: Field>: FsAbsorb + FsChallenge<F> {}

impl<F: Field, T: FsAbsorb + FsChallenge<F>> FsTranscript<F> for T {}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{prover_transcript, verifier_transcript, VerifierTranscript};
    use jolt_field::{Fr, FromPrimitiveInt};
    use spongefish::instantiations::Blake2b512;

    const SESSION: &[u8] = b"jolt-transcript-messages-test/v1";
    type Bl = Blake2b512;

    /// A symmetric absorb sequence + challenge derivation agrees between a
    /// (test-)prover and an independently-built verifier — the modular crates'
    /// model. No NARG: every value is `absorb`'d on both sides.
    #[test]
    fn symmetric_absorb_and_challenges_agree() {
        let scalars: Vec<Fr> = (0..6).map(|i| Fr::from_u64(i * 7 + 3)).collect();
        let instance = [0x42; 32];

        let mut p = prover_transcript(SESSION, instance, Bl::default());
        FsAbsorb::absorb_field_slice(&mut p, &scalars);
        let p_opt = FsChallenge::<Fr>::challenge(&mut p);
        let p_plain = FsChallenge::<Fr>::challenge_scalar(&mut p);
        let p_vec = FsChallenge::<Fr>::challenge_vector(&mut p, 3);
        let p_pow = FsChallenge::<Fr>::challenge_scalar_powers(&mut p, 4);

        // Symmetric: the verifier re-absorbs the SAME shared values (no NARG read).
        let empty: &[u8] = &[];
        let mut v = verifier_transcript(SESSION, instance, Bl::default(), empty);
        FsAbsorb::absorb_field_slice(&mut v, &scalars);
        let v_opt = FsChallenge::<Fr>::challenge(&mut v);
        let v_plain = FsChallenge::<Fr>::challenge_scalar(&mut v);
        let v_vec = FsChallenge::<Fr>::challenge_vector(&mut v, 3);
        let v_pow = FsChallenge::<Fr>::challenge_scalar_powers(&mut v, 4);
        VerifierTranscript::<Bl>::check_eof(v).unwrap();

        assert_eq!(p_opt, v_opt, "optimized challenge diverged");
        assert_eq!(p_plain, v_plain, "plain scalar challenge diverged");
        assert_eq!(p_vec, v_vec, "challenge vector diverged");
        assert_eq!(p_pow, v_pow, "challenge powers diverged");
    }

    /// The optimized and plain embeddings differ (the DEV-41 distinction):
    /// the same 128-bit squeeze must NOT produce the same field element.
    #[test]
    fn optimized_and_plain_embeddings_differ() {
        let instance = [7u8; 32];
        let mut a = prover_transcript(SESSION, instance, Bl::default());
        let optimized = FsChallenge::<Fr>::challenge(&mut a);

        let mut b = prover_transcript(SESSION, instance, Bl::default());
        let plain = FsChallenge::<Fr>::challenge_scalar(&mut b);

        // Same transcript position, same 128-bit squeeze, different embedding.
        assert_ne!(
            optimized, plain,
            "optimized (v·2^128) and plain (v) embeddings must differ"
        );
    }
}
