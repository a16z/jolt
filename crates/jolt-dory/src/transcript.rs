//! Bridges the `jolt-transcript` framework into dory-pcs's `DoryTranscript` trait.
//!
//! Prover/verifier parity within `jolt-dory` is by construction: both sides
//! traverse this adapter. The surrounding Jolt transcript is responsible for
//! matching the core Fiat-Shamir byte layout before this adapter is entered.
//!
//! Dory's transcript trait draws challenges infallibly. A draw the Jolt
//! transcript refuses is recorded and reported by
//! [`JoltToDoryTranscript::finish`], which every caller checks once Dory
//! returns.

use dory::backends::arkworks::BN254;
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use jolt_field::{Fr, Ring};
use jolt_openings::OpeningsError;
use jolt_transcript::{FieldTranscript, Label, LabelWithCount};

use crate::scheme::{ark_to_jolt_fr, jolt_fr_to_ark, ArkFr};

pub struct JoltToDoryTranscript<'a, T: FieldTranscript<Fr>> {
    transcript: &'a mut T,
    rejected: bool,
}

impl<'a, T: FieldTranscript<Fr>> JoltToDoryTranscript<'a, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        Self {
            transcript,
            rejected: false,
        }
    }

    /// Ends the adapter's use.
    ///
    /// # Errors
    /// [`OpeningsError::TranscriptRejected`] if the transcript refused one of
    /// Dory's challenge draws; the challenges Dory used after it are void.
    pub fn finish(self) -> Result<(), OpeningsError> {
        if self.rejected {
            Err(OpeningsError::TranscriptRejected)
        } else {
            Ok(())
        }
    }
}

/// The loss factor of each Dory challenge drawn as a short scalar, by the
/// label dory-pcs 0.4.2 draws it under; `None` draws a full-width scalar.
///
/// The verifier multiplies group elements by `alpha`, `beta` and `d`
/// directly, so a short challenge makes those multiplications cheaper. Its
/// width comes from the loss ([`FieldTranscript::challenge_short_scalar`]): a
/// bad set of `L` values is hit with probability at most `2^-128`. The losses
/// come from the verifier's checks, as Laurent polynomials in the challenge:
///
/// - `beta`: the round's checks span `beta^-1` to `beta`, so at most 2 values
///   of `beta` make a false claim pass. `beta` is drawn alone. `L = 2`.
/// - `alpha`: the fold spans `alpha^-1` to `alpha`, 2 values. The last
///   round's `alpha` is drawn together with `gamma`, whose check spans
///   `gamma^-1` to `gamma` (2 more values), so the width covers both.
///   `L = 2 + 2 = 4`.
/// - `d`: the final pairing check spans `d^-1` to `d^2`. `L = 3`.
///
/// `gamma` stays full width: the verifier only uses `-gamma` and
/// `gamma^-1`, which are full width anyway. So do the zero-knowledge
/// Sigma-protocol challenges, which the verifier squares or multiplies by
/// other challenges.
///
/// These widths hold each burst of challenges to the `2^-128` error the rest
/// of Jolt's transcript is priced to. For Dory that bound is round by round:
/// a check that passes at a challenge outside its bad set yields a break of
/// Dory's binding (an algebraic-group-model argument under SXDH), which the
/// prover cannot do. A bound summed over all of Dory's challenges instead
/// would need about 5 more bits.
///
/// The table depends on how dory-pcs uses each label and on its draw order,
/// so the workspace pins dory-pcs to exactly 0.4.2, and
/// `tests/challenge_bursts.rs` checks that each short draw follows an
/// absorb and that only `gamma` follows the last `alpha` with no absorb in
/// between. A label this table does not know draws full width.
fn short_challenge_loss(label: &[u8]) -> Option<u64> {
    match label {
        b"beta" => Some(2),
        b"alpha" => Some(4),
        b"d" => Some(3),
        _ => None,
    }
}

impl<T: FieldTranscript<Fr>> DoryTranscript for JoltToDoryTranscript<'_, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript
            .append(&LabelWithCount(b"dory_bytes", bytes.len() as u64));
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        let jolt_scalar = ark_to_jolt_fr(x);
        self.transcript.append(&Label(b"dory_field"));
        self.transcript.append(&jolt_scalar);
    }

    #[expect(
        clippy::expect_used,
        reason = "transcript serialization failures are fatal"
    )]
    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        let mut buffer = Vec::new();
        g.serialize_compressed(&mut buffer)
            .expect("group serialization should not fail");
        self.transcript
            .append(&LabelWithCount(b"dory_group", buffer.len() as u64));
        self.transcript.append_bytes(&buffer);
    }

    #[expect(
        clippy::expect_used,
        reason = "transcript serialization failures are fatal"
    )]
    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        self.transcript
            .append(&LabelWithCount(b"dory_serde", buffer.len() as u64));
        self.transcript.append_bytes(&buffer);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> ArkFr {
        let drawn = match short_challenge_loss(label) {
            Some(loss) => self.transcript.challenge_short_scalar(loss),
            None => self.transcript.challenge(),
        };
        // Dory cannot take an error here: a refused draw yields zero and
        // fails `finish`.
        let challenge: Fr = drawn.unwrap_or_else(|_| {
            self.rejected = true;
            Fr::from_u64(0)
        });
        jolt_fr_to_ark(&challenge)
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("reset is not invoked by dory-pcs and is intentionally unsupported")
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::indexing_slicing,
        reason = "tests unwrap infallible serialization of well-formed group elements"
    )]

    use super::*;
    use crate::scheme::jolt_fr_to_ark;
    use ark_bn254::{Fr as ArkBn254Fr, G1Projective};
    use ark_ec::PrimeGroup;
    use ark_serialize::CanonicalSerialize;
    use dory::backends::arkworks::{ArkFr as DoryArkFr, ArkG1};
    use jolt_transcript::{prover_transcript, Blake2b512, ProverState, Transcript};

    // The framing words below are reconstructed from the documented layout
    // (`Label` / `LabelWithCount` in jolt-transcript's codec), NOT by calling
    // those helpers: a `Label` is one 32-byte zero-padded word; a
    // `LabelWithCount` packs the label into bytes 0..24 and the count
    // big-endian into bytes 24..32. A word is absorbed as its 32 raw bytes, a
    // byte payload through `append_bytes`, which prefixes its length.

    fn label_word(label: &[u8]) -> [u8; 32] {
        assert!(label.len() <= 32);
        let mut word = [0u8; 32];
        word[..label.len()].copy_from_slice(label);
        word
    }

    fn label_with_count_word(label: &[u8], count: u64) -> [u8; 32] {
        assert!(label.len() <= 24);
        let mut word = [0u8; 32];
        word[..label.len()].copy_from_slice(label);
        word[24..].copy_from_slice(&count.to_be_bytes());
        word
    }

    fn transcript() -> ProverState<Blake2b512> {
        prover_transcript::<Blake2b512>(b"dory-adapter-framing", b"")
    }

    /// `ArkFr` in scope is a type alias, which cannot be used in constructor
    /// position; this builds the dory wrapper explicitly.
    fn dory_ark_fr(inner: ArkBn254Fr) -> ArkFr {
        DoryArkFr(inner)
    }

    #[test]
    fn append_bytes_frames_as_dory_bytes_count_word_then_payload() {
        let payload = [0xaa, 0xbb, 0xcc, 0xdd, 0xee];

        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_bytes(b"caller-label", &payload);

        let mut expected = transcript();
        expected.append(&label_with_count_word(b"dory_bytes", 5));
        expected.append_bytes(&payload);
        let actual = actual.verifier_message::<[u8; 32]>();
        assert_eq!(actual, expected.verifier_message::<[u8; 32]>());

        // The count is load-bearing: the same payload under a wrong count
        // must diverge, otherwise the golden comparison proves nothing.
        let mut wrong_count = transcript();
        wrong_count.append(&label_with_count_word(b"dory_bytes", 4));
        wrong_count.append_bytes(&payload);
        assert_ne!(actual, wrong_count.verifier_message::<[u8; 32]>());

        // The payload is a length-prefixed message of its own, not part of a
        // concatenated buffer with the word.
        let mut merged = transcript();
        let mut buffer = label_with_count_word(b"dory_bytes", 5).to_vec();
        buffer.extend_from_slice(&payload);
        merged.append_bytes(&buffer);
        assert_ne!(actual, merged.verifier_message::<[u8; 32]>());
    }

    #[test]
    fn append_field_frames_as_dory_field_label_then_big_endian_scalar() {
        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_field(
            b"caller-label",
            &dory_ark_fr(ArkBn254Fr::from(0xdead_beefu64)),
        );

        // Fr absorbs as its 32-byte little-endian canonical form: the value
        // then 24 zero bytes, reconstructed here without CanonicalBytes.
        let mut scalar_le = [0u8; 32];
        scalar_le[..8].copy_from_slice(&0xdead_beefu64.to_le_bytes());

        let mut expected = transcript();
        expected.append(&label_word(b"dory_field"));
        expected.append(&scalar_le);
        let actual = actual.verifier_message::<[u8; 32]>();
        assert_eq!(actual, expected.verifier_message::<[u8; 32]>());

        // Big-endian absorption would be an invisible-to-roundtrip bug.
        let mut big_endian = transcript();
        let mut scalar_be = [0u8; 32];
        scalar_be[24..].copy_from_slice(&0xdead_beefu64.to_be_bytes());
        big_endian.append(&label_word(b"dory_field"));
        big_endian.append(&scalar_be);
        assert_ne!(actual, big_endian.verifier_message::<[u8; 32]>());
    }

    #[test]
    fn append_group_frames_as_dory_group_count_word_then_compressed_point() {
        let generator = ArkG1(G1Projective::generator());

        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_group(b"caller-label", &generator);

        // Payload reconstructed via arkworks compressed serialization of the
        // inner point; the adapter's framing is the labeled count word.
        let mut payload = Vec::new();
        generator
            .0
            .serialize_compressed(&mut payload)
            .expect("compressed G1 serialization should not fail");
        assert_eq!(payload.len(), 32);

        let mut expected = transcript();
        expected.append(&label_with_count_word(b"dory_group", payload.len() as u64));
        expected.append_bytes(&payload);
        assert_eq!(
            actual.verifier_message::<[u8; 32]>(),
            expected.verifier_message::<[u8; 32]>()
        );
    }

    #[test]
    fn append_serde_frames_as_dory_serde_count_word_then_compressed_value() {
        let value = dory_ark_fr(ArkBn254Fr::from(42u64));

        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_serde(b"caller-label", &value);

        let mut payload = Vec::new();
        value
            .0
            .serialize_compressed(&mut payload)
            .expect("compressed Fr serialization should not fail");
        assert_eq!(payload.len(), 32);

        let mut expected = transcript();
        expected.append(&label_with_count_word(b"dory_serde", payload.len() as u64));
        expected.append_bytes(&payload);
        assert_eq!(
            actual.verifier_message::<[u8; 32]>(),
            expected.verifier_message::<[u8; 32]>()
        );
    }

    /// Domain separation comes from the fixed `dory_*` labels; the caller's
    /// dory-side label is never absorbed (it only picks a challenge's width).
    #[test]
    fn caller_labels_do_not_reach_the_transcript() {
        let payload = [1u8, 2, 3];

        let mut first = transcript();
        JoltToDoryTranscript::new(&mut first).append_bytes(b"label-one", &payload);

        let mut second = transcript();
        JoltToDoryTranscript::new(&mut second).append_bytes(b"label-two", &payload);

        let first = first.verifier_message::<[u8; 32]>();
        assert_eq!(first, second.verifier_message::<[u8; 32]>());
        assert_ne!(first, transcript().verifier_message::<[u8; 32]>());
    }

    /// `alpha`, `beta` and `d` are the Jolt transcript's short scalars for
    /// their losses, at 130, 129 and 130 bits; other labels stay full width.
    #[test]
    fn group_scalar_challenges_are_short() {
        for (label, bits) in [(&b"alpha"[..], 130), (b"beta", 129), (b"d", 130)] {
            let loss = short_challenge_loss(label).expect("short label");
            assert_eq!(
                jolt_transcript::challenge_bits_for_loss(loss, jolt_transcript::SECURITY_BITS),
                Ok(bits)
            );

            let mut adapted = transcript();
            let mut adapter = JoltToDoryTranscript::new(&mut adapted);
            let adapter_challenge = adapter.challenge_scalar(label);
            adapter.finish().expect("short Dory challenges fit BN254");
            let mut direct = transcript();
            let direct_challenge: Fr = direct.challenge_short_scalar(loss).expect("fits BN254");
            assert_eq!(adapter_challenge, jolt_fr_to_ark(&direct_challenge));
            assert_eq!(
                adapted.verifier_message::<[u8; 32]>(),
                direct.verifier_message::<[u8; 32]>()
            );
        }
        for label in [&b"gamma"[..], b"sigma_c", b"sigma1_c", b"sigma2_c"] {
            assert_eq!(short_challenge_loss(label), None);
        }
    }

    /// The adapter's challenge is the Jolt transcript's scalar challenge,
    /// converted — so after identical absorptions both sides must squeeze the
    /// same scalar, and the adapter's state advances exactly like the direct
    /// transcript's.
    #[test]
    fn challenge_scalar_matches_underlying_jolt_transcript() {
        let mut adapted = transcript();
        let mut adapter = JoltToDoryTranscript::new(&mut adapted);
        adapter.append_bytes(b"caller-label", b"shared-absorption");
        let adapter_challenge = adapter.challenge_scalar(b"caller-label");
        adapter
            .finish()
            .expect("the adapter draws a supported challenge");

        let mut direct = transcript();
        direct.append(&label_with_count_word(b"dory_bytes", 17));
        direct.append_bytes(b"shared-absorption");
        let direct_challenge: Fr = direct.challenge().expect("Fr supports one challenge");

        assert_eq!(adapter_challenge, jolt_fr_to_ark(&direct_challenge));
        assert_ne!(adapter_challenge, dory_ark_fr(ArkBn254Fr::from(0u64)));
        assert_eq!(
            adapted.verifier_message::<[u8; 32]>(),
            direct.verifier_message::<[u8; 32]>()
        );

        // Fr conversion sanity: the transmute-based bridge is the identity on
        // canonical values.
        assert_eq!(
            jolt_fr_to_ark(&Fr::from_u64(7)),
            dory_ark_fr(ArkBn254Fr::from(7u64)),
        );
    }
}
