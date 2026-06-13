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
/// identically on the prover and verifier sides. On byte sponges every absorb
/// is one length-prefixed [`BytesMsg`], so `absorb(a) ; absorb(b)` stays
/// distinct from `absorb(a ‖ b)`; on the `Fr`-unit Poseidon sponge every
/// absorb is one leading-tagged unit message (see [`crate::codec`]) with the
/// same length-binding property.
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

    // ── Typed vocabulary (spec §4.4) ────────────────────────────────────────
    //
    // `absorb` is type-opaque (a `CanonicalSerialize` blob), so a field-unit
    // sponge cannot recover the value's kind from it — and length-sniffing is
    // unsound (12 field elements serialize to exactly one GT's 384 bytes).
    // These typed methods carry the kind. Their DEFAULTS route through the
    // required byte methods above, so byte sponges (and symbolic implementors
    // like the transpiler's `SymbolicVerifierFs`) keep today's behavior
    // bit-for-bit with zero impl changes; the `Fr`-unit Poseidon state
    // overrides them with the spec §4.2 unit encodings.

    /// Absorb one **field element** as a single message.
    ///
    /// Default: identical to [`absorb`](Self::absorb). Poseidon override: the
    /// count-led field frame `[Fr(3), value]`.
    fn absorb_scalar<T: CanonicalSerialize>(&mut self, value: &T) {
        self.absorb(value);
    }

    /// Absorb a slice of **field elements** as a single message.
    ///
    /// Default: identical to `absorb(&values.to_vec())` (ark's `Vec`
    /// serialization: 8-byte LE count ‖ elements) — the byte form every
    /// converted call site used. Poseidon override: the count-led field frame
    /// `[Fr(2k+1), e₁, …, e_k]`.
    fn absorb_scalars<T: CanonicalSerialize + Clone>(&mut self, values: &[T]) {
        self.absorb(&values.to_vec());
    }

    /// Absorb one **commitment / group element** (Dory GT, G1/G2 points) as a
    /// single message.
    ///
    /// Default: identical to [`absorb`](Self::absorb). Poseidon override: the
    /// byte rule over the compressed serialization (one GT = 384 bytes ↦
    /// `[Fr(768), 13 chunks]`); group coordinates (q > r) must NEVER be
    /// absorbed as native field units.
    fn absorb_commitment<T: CanonicalSerialize>(&mut self, value: &T) {
        self.absorb(value);
    }

    /// [`absorb_commitment`](Self::absorb_commitment) for a group element
    /// already given as its canonical compressed bytes (the Dory bridge's
    /// `append_group`/`append_serde`, whose `DorySerialize` values are not
    /// ark-`CanonicalSerialize`).
    ///
    /// Default: identical to `absorb(&bytes.to_vec())` (8-byte LE length ‖
    /// bytes inside the message) — today's byte form at those sites. Poseidon
    /// override: the byte rule directly over `bytes`.
    fn absorb_commitment_bytes(&mut self, bytes: &[u8]) {
        self.absorb(&bytes.to_vec());
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

/// Poseidon (`U = Fr`) absorb path. The prover- and verifier-state impls are
/// emitted from one `macro_rules!` body (`impl_poseidon_fs_absorb`), so the
/// two roles cannot drift. Type-opaque absorbs and raw bytes go under the
/// byte rule ([`RawBytesMsg`](crate::RawBytesMsg)); the typed methods carry
/// the spec §4.2 unit encodings.
#[cfg(feature = "transcript-poseidon")]
mod poseidon_absorb {
    use super::*;
    use crate::codec::{FieldFrameMsg, RawBytesMsg};
    use crate::poseidon::PoseidonSponge;
    use ark_bn254::Fr;
    use ark_serialize::CanonicalDeserialize;

    /// Parse the concatenated 32-byte-LE canonical serializations of field
    /// elements (e.g. jolt-core's `F: JoltField` scalars) into native `Fr`
    /// units. Canonical inputs round-trip exactly; a non-32-multiple length
    /// means a non-scalar reached a scalar-typed absorb — a call-site bug.
    #[expect(clippy::expect_used, reason = "caller-contract violation, not data")]
    fn parse_scalar_units(bytes: &[u8]) -> Vec<Fr> {
        assert!(
            bytes.len().is_multiple_of(32),
            "absorb_scalar(s): value is not a sequence of 32-byte field elements ({} bytes)",
            bytes.len()
        );
        bytes
            .chunks_exact(32)
            .map(|c| Fr::deserialize_compressed(c).expect("non-canonical scalar absorbed"))
            .collect()
    }

    /// Emits the full Poseidon `FsAbsorb` method set under the given impl
    /// header. Invoked once per transcript role so both impls share one
    /// token-for-token method body and cannot drift.
    macro_rules! impl_poseidon_fs_absorb {
        ($($header:tt)+) => {
            $($header)+ {
                fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
                    self.public_message(&RawBytesMsg(serialize_one(value)));
                }

                fn absorb_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
                    self.public_message(&RawBytesMsg(serialize_slice(values)));
                }

                fn absorb_bytes(&mut self, bytes: &[u8]) {
                    self.public_message(&RawBytesMsg(bytes.to_vec()));
                }

                // The trait default routes `absorb_field`/`absorb_field_slice` through
                // `absorb_bytes` (byte rule), which on Poseidon diverges from
                // `absorb_scalar`/`absorb_scalars` (count-led field frame). Override here so a
                // field element absorbed via `absorb_field` produces the SAME `FieldFrameMsg`
                // it would via `absorb_scalar`: `Field::to_bytes_le_vec()` for BN254 `Fr`
                // equals `serialize_compressed` (canonical 32-byte LE), so feeding it through
                // `parse_scalar_units` yields the identical native `Fr` units.
                fn absorb_field<F: Field>(&mut self, value: &F) {
                    self.public_message(&FieldFrameMsg(parse_scalar_units(
                        &value.to_bytes_le_vec(),
                    )));
                }

                fn absorb_field_slice<F: Field>(&mut self, values: &[F]) {
                    let mut buf = Vec::with_capacity(values.len() * F::NUM_BYTES);
                    for v in values {
                        buf.extend_from_slice(&v.to_bytes_le_vec());
                    }
                    self.public_message(&FieldFrameMsg(parse_scalar_units(&buf)));
                }

                fn absorb_scalar<T: CanonicalSerialize>(&mut self, value: &T) {
                    self.public_message(&FieldFrameMsg(parse_scalar_units(&serialize_one(value))));
                }

                fn absorb_scalars<T: CanonicalSerialize + Clone>(&mut self, values: &[T]) {
                    self.public_message(&FieldFrameMsg(parse_scalar_units(&serialize_slice(
                        values,
                    ))));
                }

                fn absorb_commitment<T: CanonicalSerialize>(&mut self, value: &T) {
                    // The byte rule over one compressed serialization —
                    // unit-identical to one per-GT group inside a
                    // `CommitmentsMsg` frame (the frame's leading `Fr(2k+1)`
                    // count unit binds the frame partition and is not emitted
                    // for a lone, schedule-fixed commitment absorb).
                    self.absorb_commitment_bytes(&serialize_one(value));
                }

                fn absorb_commitment_bytes(&mut self, bytes: &[u8]) {
                    self.public_message(&RawBytesMsg(bytes.to_vec()));
                }
            }
        };
    }

    impl_poseidon_fs_absorb!(
        impl<R: RngCore + CryptoRng> FsAbsorb for ProverState<PoseidonSponge, R>
    );
    impl_poseidon_fs_absorb!(impl FsAbsorb for VerifierState<'_, PoseidonSponge>);
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

// Gated on `transcript-blake2b`: the suite fixes its sponge to `Blake2b512`
// (and exercises that sponge's `FsChallenge` impl), so it only compiles when
// that feature — the one pulling `spongefish/blake2` — is on. Without this
// gate `--no-default-features --features transcript-poseidon --all-targets`
// fails to build (no `Blake2b512` in `spongefish::instantiations`).
#[cfg(all(test, feature = "transcript-blake2b"))]
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
