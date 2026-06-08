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
//! Challenge width is selected **per sponge** (matching legacy `transcripts/`), via the
//! `transcript-poseidon` feature:
//! - **Byte sponges (Blake2b/Keccak), default build:** 128-bit
//!   [`OptimizedChallenge::challenge_u128`]; `F::Challenge` is `MontU128Challenge`. They stay
//!   128-bit even under a hand-set `challenge-254-bit` — as in legacy `blake2b.rs`/`keccak.rs`.
//! - **Poseidon (`transcript-poseidon` → `challenge-254-bit`, #1586 reviewer / D5b):** GENUINE
//!   full-field `Fr` squeezes (`verifier_message::<ark_bn254::Fr>`), `F::Challenge` is
//!   `Mont254BitChallenge`. Poseidon's natural unit (128-bit truncation is costly for recursion);
//!   restores legacy `transcripts/poseidon.rs` so Poseidon works end-to-end and never hits its
//!   `unimplemented!()` `challenge_u128`.
//!
//! **What actually crosses the NARG (host "hybrid" milestone):** *only* prover-only
//! payload — the sumcheck/uniskip round polynomials — is `write_slice`/`read_slice`
//! through the NARG. *Shared* values (the public statement, flushed opening claims) are
//! `absorb`'d ([`public_message`]); commitments and the dory opening proof stay
//! **structured proof fields** and are also `absorb`'d, NOT written to the NARG.
//! (Pushing commitments into the NARG — full Option-B — is a follow-up.)
//!
//! Three concerns, three traits:
//! - [`FsChallenge`] — squeezed verifier randomness; blanket-implemented for any
//!   transcript with [`OptimizedChallenge`], so prover and verifier share it.
//! - [`ProverFs`] — `absorb` shared values ([`public_message`], not shipped) and
//!   `write_slice` prover-only payload ([`prover_message`], into the NARG).
//! - [`VerifierFs`] — `absorb` the same shared values, and `read_slice` the prover
//!   payload back from the NARG in order.
//!
//! [`public_message`]: spongefish::ProverState::public_message
//! [`prover_message`]: spongefish::ProverState::prover_message

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_transcript::{
    serialize_slice, BytesMsg, DuplexSpongeInterface, OptimizedChallenge, ProverState,
    VerificationError, VerificationResult, VerifierState,
};

/// Absorbing shared values is the field-agnostic [`jolt_transcript::FsAbsorb`] surface
/// (`absorb` = spongefish `public_message`); jolt-core re-exports it so the whole
/// transcript vocabulary lives behind `crate::transcript_msgs`, and there is a *single*
/// absorb implementation shared with the modular crates (no hand-kept second copy).
pub use jolt_transcript::FsAbsorb;
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

/// Squeezed field challenges, shared by both transcript roles.
///
/// Blanket-implemented for any state exposing [`OptimizedChallenge`] (i.e. every
/// `ProverState`/`VerifierState` over a supported sponge), so the prover and
/// verifier derive challenges identically.
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

/// Non-Poseidon path (default): 128-bit [`OptimizedChallenge::challenge_u128`], blanket
/// over the byte sponges (Blake2b/Keccak). They stay 128-bit even under a hand-set
/// `challenge-254-bit` (a 128-bit value in the wider type) — gating on the *sponge*, not
/// the width, preserves legacy's per-sponge behaviour.
#[cfg(not(feature = "transcript-poseidon"))]
impl<F: JoltField, S: OptimizedChallenge> FsChallenge<F> for S {
    fn challenge_field(&mut self) -> F {
        F::from_u128(self.challenge_u128())
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }
}

/// Reconstruct `F` from a full BN254 `Fr` squeeze via its canonical 32-byte LE bytes
/// (stack buffer, no `Vec` — runs per challenge). `from_bytes` is `from_le_bytes_mod_order`
/// and the bytes are `< modulus`, so the round-trip is exact.
#[cfg(feature = "transcript-poseidon")]
fn full_field_squeeze<F: JoltField>(squeezed: ark_bn254::Fr) -> F {
    use ark_ff::PrimeField;
    let limbs = squeezed.into_bigint().0;
    let mut le = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        le[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    F::from_bytes(&le)
}

/// Reinterpret a full field element as its `F::Challenge` (Poseidon-only: there it's the
/// `#[repr(transparent)]` `Mont254BitChallenge<F>`). One shared spot so the `unsafe`
/// prover/verifier reinterpretation can't drift.
#[cfg(feature = "transcript-poseidon")]
fn wrap_full_field<F: JoltField>(f: F) -> F::Challenge {
    // Turn a layout mismatch into a compile error — the check `transmute` makes but
    // `transmute_copy` skips for generics (e.g. rejects the 16-byte `MontU128Challenge`).
    const { assert!(core::mem::size_of::<F>() == core::mem::size_of::<F::Challenge>()) }
    // SAFETY: `F::Challenge` is the `#[repr(transparent)]` `Mont254BitChallenge<F>` newtype
    // of `F` — identical size (asserted) and layout. Mirrors legacy `challenge_scalar_optimized`.
    unsafe { core::mem::transmute_copy::<F, F::Challenge>(&f) }
}

/// Poseidon path (`transcript-poseidon` → `challenge-254-bit`): the optimized challenge is a
/// GENUINE full-field `Fr` (Poseidon's natural unit), not a 128-bit truncation — restoring
/// legacy `transcripts/poseidon.rs`, so Poseidon never hits its `unimplemented!()`
/// `challenge_u128`. Per transcript-state (not blanket over `OptimizedChallenge`) to reach
/// the native `verifier_message::<ark_bn254::Fr>` squeeze. NB: here `challenge_field` and
/// `challenge_optimized` return the SAME value (no `v·2¹²⁸` masking).
#[cfg(feature = "transcript-poseidon")]
impl<F, H, R> FsChallenge<F> for ProverState<H, R>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
    R: RngCore + CryptoRng,
{
    fn challenge_field(&mut self) -> F {
        full_field_squeeze(ProverState::verifier_message::<ark_bn254::Fr>(self))
    }

    fn challenge_optimized(&mut self) -> F::Challenge {
        let f: F = self.challenge_field();
        wrap_full_field(f)
    }
}

#[cfg(feature = "transcript-poseidon")]
impl<F, H> FsChallenge<F> for VerifierState<'_, H>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
{
    fn challenge_field(&mut self) -> F {
        full_field_squeeze(VerifierState::verifier_message::<ark_bn254::Fr>(self))
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
pub trait ProverFs<F: JoltField>: FsChallenge<F> + FsAbsorb {
    /// Write a sequence of prover-only values as one self-delimiting frame
    /// (read back with [`VerifierFs::read_slice`]). No length prefix is shipped; the
    /// frame is bounded by the NARG, so the per-round element count may vary.
    fn write_slice<T: CanonicalSerialize>(&mut self, values: &[T]);
}

impl<F, H, R> ProverFs<F> for ProverState<H, R>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
    R: RngCore + CryptoRng,
    Self: OptimizedChallenge,
{
    fn write_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
        self.prover_message(&BytesMsg(serialize_slice(values)));
    }
}

/// Verifier-side message vocabulary over the spongefish NARG.
pub trait VerifierFs<F: JoltField>: FsChallenge<F> + FsAbsorb {
    /// Read every value in the next frame written by [`ProverFs::write_slice`]; the
    /// count is the frame's (self-delimiting, so a varying per-round length is fine).
    /// Bounded allocation — see [`read_all`].
    fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>>;
}

impl<F, H> VerifierFs<F> for VerifierState<'_, H>
where
    F: JoltField,
    H: DuplexSpongeInterface<U = u8>,
    Self: OptimizedChallenge,
{
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
            assert_eq!(&read, expected, "round poly mis-reconstructed from NARG");
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
