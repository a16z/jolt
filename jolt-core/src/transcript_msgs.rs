//! jolt-core's field-typed Fiat–Shamir vocabulary over the spongefish NARG.
//!
//! jolt-core proof code is generic over `F: JoltField` (concretely `ark_bn254::Fr`,
//! or `TrackedFr` in profiling builds). spongefish's built-in `Encoding`/`Decoding`
//! are implemented only for the concrete arkworks types, so they can't be used
//! *directly* on a generic `F`. As a **host bridge** we therefore move values through
//! the NARG as a length-prefixed `CanonicalSerialize` blob (reusing
//! [`jolt_transcript::BytesMsg`]). This is intentionally the simplest transport, not
//! the end-state: a typed-codec surface (`Encoding` impls on local field/blob newtypes,
//! or the native codec for concrete `Fr`) is the cleaner design and a deliberate follow-up.
//!
//! Challenges come from the 128-bit [`OptimizedChallenge::challenge_u128`] primitive
//! (the legacy `challenge_scalar` likewise squeezed 16 bytes). NB this 128-bit
//! description is the Blake2b/Keccak default: `transcript-poseidon` forces
//! `challenge-254-bit`, so under Poseidon `F::Challenge` is full-field (the open D5b
//! coupling), and Poseidon's `challenge_u128` impl is correspondingly flagged optional.
//!
//! **What actually crosses the NARG (host "hybrid" milestone):** *only* prover-only
//! payload — the sumcheck/uniskip round polynomials — is `write`/`read` through the NARG.
//! *Shared* values (the public statement, flushed opening claims) are `absorb`'d
//! ([`public_message`]); commitments and the dory opening proof stay **structured proof
//! fields** and are also `absorb`'d, NOT written to the NARG. (Pushing commitments into
//! the NARG — full Option-B — is a follow-up.)
//!
//! Three concerns, three traits:
//! - [`FsChallenge`] — squeezed verifier randomness; blanket-implemented for any
//!   transcript with [`OptimizedChallenge`], so prover and verifier share it.
//! - [`ProverFs`] — `absorb` shared values ([`public_message`], not shipped) and
//!   `write` prover-only payload ([`prover_message`], into the NARG).
//! - [`VerifierFs`] — `absorb` the same shared values, and `read` the prover payload
//!   back from the NARG in order.
//!
//! [`public_message`]: spongefish::ProverState::public_message
//! [`prover_message`]: spongefish::ProverState::prover_message

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_transcript::{
    BytesMsg, DuplexSpongeInterface, OptimizedChallenge, ProverState, VerificationError,
    VerificationResult, VerifierState,
};
use rand::{CryptoRng, RngCore};

use crate::field::JoltField;

// ─── WIRE-FORMAT — FOLLOW-UP DECISION (read this before changing the codec) ─────────────
//
// Every NARG message below is transported as ONE anonymous length-prefixed byte blob
// (`BytesMsg(CanonicalSerialize)`), regardless of the value's kind. This is the host *bridge*
// choice: DRY (a single codec path) and adequate while only ONE kind of value crosses the NARG
// — the sumcheck/uniskip round polynomials. History: typed per-kind wrappers
// (`FieldMsg` / `FieldVecMsg` / `Blob`) once existed (DEV-12) and were deliberately collapsed
// into this single `BytesMsg` path for less redundancy (DEV-16). Trade-off: the wire format is
// now anonymous — type identity is recovered only by read-order + the deserialize target, not by
// the bytes themselves.
//
// SWITCH to a typed per-kind codec surface — local newtypes that `impl Encoding`/`NargDeserialize`
// (e.g. `FsField<F>` / `FsFieldVec<F>` / `FsBlob<T>`, exposing `write_field`/`read_fields`/…) — when
// EITHER trigger lands:
//   1. Full Option-B moves commitments + the dory opening proof + claims INTO the NARG (today they
//      stay structured proof fields and are only `absorb`'d). With many value kinds in the NARG, an
//      explicit, self-describing wire format prevents read-order / wrong-type mixups.
//   2. The on-chain verifier (gnark / Solidity / Lean) must re-read the NARG inside a circuit /
//      contract — a typed wire format is far easier to mirror there than N anonymous blobs.
// At either point you are ALREADY restructuring what the NARG contains, so introduce the typed
// types here and route `write`/`read` through them instead of `BytesMsg` — one change, made when
// the explicit format is load-bearing rather than cosmetic.
//
// DO NOT do it before then. With a single value kind, typed vs. anonymous is a wash; doing it now
// only reverses DEV-16 and forces a fresh `muldiv` re-verify, then gets redone at full-B = churn.
// The host path is `muldiv`-verified as-is. Decision rule: typed codec IFF (full-Option-B || on-chain
// reader); otherwise keep `BytesMsg`. See DEV-16 (the collapse) + DEV-25 (this decision).
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// Single-value [`slice_to_bytes`]: compressed serialization, no length prefix.
fn to_bytes<T: CanonicalSerialize>(value: &T) -> Vec<u8> {
    slice_to_bytes(std::slice::from_ref(value))
}

/// Squeezed field challenges, shared by both transcript roles.
///
/// Blanket-implemented for any state exposing [`OptimizedChallenge`] (i.e. every
/// `ProverState`/`VerifierState` over a supported sponge), so the prover and
/// verifier derive challenges identically.
pub trait FsChallenge<F: JoltField> {
    /// Squeeze a full-width field challenge. Per jolt's design this is the
    /// 128-bit challenge embedded in the field (the optimized and full forms
    /// squeeze the same 128 bits; they differ only in the returned type).
    fn challenge_field(&mut self) -> F;

    /// Squeeze the fast-multiply [`JoltField::Challenge`] form.
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

impl<F: JoltField, S: OptimizedChallenge> FsChallenge<F> for S {
    fn challenge_field(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }
}

#[expect(clippy::expect_used)]
fn slice_to_bytes<T: CanonicalSerialize>(values: &[T]) -> Vec<u8> {
    // Reserve the exact serialized size up front: `write_slice` runs once per
    // sumcheck round on the prover hot path, so the default-grow reallocations
    // are pure churn. The byte output is unchanged.
    let mut buf = Vec::with_capacity(values.iter().map(|v| v.compressed_size()).sum());
    for v in values {
        v.serialize_compressed(&mut buf)
            .expect("CanonicalSerialize into a Vec is infallible");
    }
    buf
}

/// Decode every `T` in a single self-delimiting frame.
///
/// The frame body length (carried by [`BytesMsg`]'s prefix, which the sponge already
/// bounds to the remaining NARG) determines the count — so a sequence is read back
/// without shipping or trusting a separate length, and the per-round element count
/// may vary. This is also why we never `read::<Vec<T>>()`: `CanonicalDeserialize` for
/// `Vec` reads its OWN length prefix from the NARG and pre-allocates from it, so an
/// adversarial proof could capacity-overflow panic / OOM. Here every allocation is
/// bounded by the actual frame bytes, which are bounded by the actual proof.
fn read_all<T: CanonicalDeserialize>(body: &[u8]) -> VerificationResult<Vec<T>> {
    let mut cursor = body;
    let mut out = Vec::new();
    while !cursor.is_empty() {
        out.push(T::deserialize_compressed(&mut cursor).map_err(|_| VerificationError)?);
    }
    Ok(out)
}

/// Absorbing shared values, common to both transcript roles.
///
/// A value both sides recompute is `absorb`'d (spongefish `public_message`): it
/// is mixed into the sponge but NOT shipped in the NARG. Shared by `ProverFs` /
/// `VerifierFs` so role-agnostic code (e.g. `fiat_shamir_preamble`) can absorb
/// the public statement on either side.
pub trait AbsorbFs<F: JoltField> {
    /// Absorb a shared value both sides recompute (emits no NARG bytes).
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T);
}

impl<F, H, R> AbsorbFs<F> for ProverState<H, R>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
    R: RngCore + CryptoRng,
{
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
        self.public_message(&BytesMsg(to_bytes(value)));
    }
}

impl<F, H> AbsorbFs<F> for VerifierState<'_, H>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
{
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
        self.public_message(&BytesMsg(to_bytes(value)));
    }
}

/// Prover-side message vocabulary over the spongefish NARG.
pub trait ProverFs<F: JoltField>: FsChallenge<F> + AbsorbFs<F> {
    /// Write a fixed-size prover-only value into the NARG.
    fn write<T: CanonicalSerialize>(&mut self, value: &T);

    /// Write a sequence whose length the verifier already knows from the protocol
    /// (read back with [`VerifierFs::read_vec`]). No length prefix is shipped.
    fn write_slice<T: CanonicalSerialize>(&mut self, values: &[T]);
}

impl<F, H, R> ProverFs<F> for ProverState<H, R>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
    R: RngCore + CryptoRng,
    Self: OptimizedChallenge,
{
    fn write<T: CanonicalSerialize>(&mut self, value: &T) {
        self.prover_message(&BytesMsg(to_bytes(value)));
    }

    fn write_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
        self.prover_message(&BytesMsg(slice_to_bytes(values)));
    }
}

/// Verifier-side message vocabulary over the spongefish NARG.
pub trait VerifierFs<F: JoltField>: FsChallenge<F> + AbsorbFs<F> {
    /// Read the next fixed-size prover-only value back from the NARG, in order.
    fn read<T: CanonicalDeserialize>(&mut self) -> VerificationResult<T>;

    /// Read every value in the next frame written by [`ProverFs::write_slice`]; the
    /// count is the frame's (self-delimiting, so a varying per-round length is fine).
    /// Bounded allocation — see [`read_all`].
    fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>>;

    /// Like [`read_slice`](Self::read_slice) but assert the frame holds exactly `n`
    /// values (use when the count is protocol-fixed, for defense in depth).
    fn read_vec<T: CanonicalDeserialize>(&mut self, n: usize) -> VerificationResult<Vec<T>> {
        let values = self.read_slice()?;
        if values.len() != n {
            return Err(VerificationError);
        }
        Ok(values)
    }
}

impl<F, H> VerifierFs<F> for VerifierState<'_, H>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
    Self: OptimizedChallenge,
{
    fn read<T: CanonicalDeserialize>(&mut self) -> VerificationResult<T> {
        let bytes = self.prover_message::<BytesMsg>()?;
        let mut cursor = &bytes.0[..];
        let value = T::deserialize_compressed(&mut cursor).map_err(|_| VerificationError)?;
        if !cursor.is_empty() {
            return Err(VerificationError);
        }
        Ok(value)
    }

    fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>> {
        let bytes = self.prover_message::<BytesMsg>()?;
        read_all(&bytes.0)
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

    /// A single value round-trips through the NARG and `check_eof` succeeds.
    #[test]
    fn write_then_read_round_trips() {
        let mut r = test_rng();
        let scalars: Vec<Fr> = (0..5).map(|_| Fr::random(&mut r)).collect();

        let instance = [7u8; 32];
        let mut p = prover_transcript(SESSION, instance, Bl::default());
        for s in &scalars {
            ProverFs::<Fr>::write(&mut p, s);
        }
        let narg = p.narg_string().to_vec();

        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        for s in &scalars {
            let read: Fr = VerifierFs::<Fr>::read(&mut v).unwrap();
            assert_eq!(read, *s);
        }
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
            AbsorbFs::<Fr>::absorb(&mut p, c);
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
            AbsorbFs::<Fr>::absorb(&mut p, c);
        }
        let narg = p.narg_string().to_vec();

        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        for c in &input_claims {
            AbsorbFs::<Fr>::absorb(&mut v, c);
        }
        let v_batching = FsChallenge::<Fr>::challenge_vec(&mut v, n_instances);
        let mut v_round_challenges = Vec::with_capacity(n_rounds);
        for expected in &round_polys {
            // round-poly counts vary, so read the self-delimiting frame (like real sumcheck)
            let read: Vec<Fr> = VerifierFs::<Fr>::read_slice(&mut v).unwrap();
            assert_eq!(&read, expected, "round poly mis-reconstructed from NARG");
            v_round_challenges.push(FsChallenge::<Fr>::challenge_optimized(&mut v));
        }
        // Verifier absorbs the same shared flushed claims (not read from the NARG).
        for c in &flushed_claims {
            AbsorbFs::<Fr>::absorb(&mut v, c);
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
        ProverFs::<Fr>::write(&mut p, &Fr::from(42u64));
        let mut narg = p.narg_string().to_vec();
        narg.push(0xFF);

        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        let _: Fr = VerifierFs::<Fr>::read(&mut v).unwrap();
        assert!(VerifierTranscript::<Bl>::check_eof(v).is_err());
    }

    /// Reading in the wrong order cannot silently pass: a scalar's bytes read as a
    /// `Vec<F>` either errors or fails `check_eof` — never a clean accept.
    #[test]
    fn wrong_read_order_does_not_silently_pass() {
        let mut r = test_rng();
        let scalar = Fr::random(&mut r);
        let vec_msg: Vec<Fr> = (0..4).map(|_| Fr::random(&mut r)).collect();
        let instance = [0x0D; 32];

        let mut p = prover_transcript(SESSION, instance, Bl::default());
        ProverFs::<Fr>::write(&mut p, &scalar);
        ProverFs::<Fr>::write_slice(&mut p, &vec_msg);
        let narg = p.narg_string().to_vec();

        // Verifier reads the slice first (wrong order): it consumes the scalar's
        // frame and tries to decode `vec_msg.len()` field elements from it — which
        // cannot succeed (and must NOT panic, even on adversarial bytes).
        let mut v = verifier_transcript(SESSION, instance, Bl::default(), &narg);
        let first: VerificationResult<Vec<Fr>> = VerifierFs::<Fr>::read_vec(&mut v, vec_msg.len());
        let silently_ok = match first {
            Err(_) => false,
            Ok(read) => read == vec_msg && VerifierTranscript::<Bl>::check_eof(v).is_ok(),
        };
        assert!(!silently_ok, "wrong read order silently accepted");
    }
}
