//! Coverage for the native split-trait surface — `ProverTranscript`,
//! `VerifierTranscript`, `OptimizedChallenge`. The existing per-backend
//! `transcript_tests!` macro exercises only the `legacy::Transcript`
//! facade.

#![cfg(any(feature = "transcript-blake2b", feature = "transcript-keccak"))]
#![expect(clippy::expect_used, reason = "tests")]

use jolt_field::Fr;
use jolt_transcript::{prover_transcript, verifier_transcript, BytesMsg, OptimizedChallenge};

const SESSION: &[u8] = b"native-traits";
const INSTANCE: [u8; 32] = [0x77; 32];

#[cfg(feature = "transcript-blake2b")]
mod blake2b {
    use super::*;
    use spongefish::instantiations::Blake2b512;

    #[test]
    fn prover_verifier_round_trip() {
        let mut prover = prover_transcript(SESSION, INSTANCE, Blake2b512::default());
        prover.public_message(&BytesMsg(b"pub".to_vec()));
        prover.prover_message(&BytesMsg(b"private".to_vec()));
        let _c1: Fr = prover.challenge_128();
        let narg = prover.narg_string().to_vec();

        let mut verifier = verifier_transcript(SESSION, INSTANCE, Blake2b512::default(), &narg);
        verifier.public_message(&BytesMsg(b"pub".to_vec()));
        let got: BytesMsg = verifier
            .prover_message()
            .expect("prover_message must deserialize");
        assert_eq!(got.as_slice(), b"private");
        let _c2: Fr = verifier.challenge_128();
        verifier.check_eof().expect("eof");
    }

    #[test]
    fn optimized_challenge_is_128_bit_truncated() {
        use ark_ff::PrimeField;
        let mut prover = prover_transcript(SESSION, INSTANCE, Blake2b512::default());
        let c: Fr = prover.challenge_128();
        let ark_c: ark_bn254::Fr = c.into();
        let bigint = ark_c.into_bigint().0;
        assert_eq!(
            bigint[2], 0,
            "challenge_128 must fit in the low 128 bits — limb 2 leaked",
        );
        assert_eq!(
            bigint[3], 0,
            "challenge_128 must fit in the low 128 bits — limb 3 leaked",
        );
    }

    #[test]
    fn distinct_sessions_diverge() {
        let mut a = prover_transcript(b"a", INSTANCE, Blake2b512::default());
        let mut b = prover_transcript(b"b", INSTANCE, Blake2b512::default());
        let ca: Fr = a.challenge_128();
        let cb: Fr = b.challenge_128();
        assert_ne!(
            ca, cb,
            "distinct session bytes must yield distinct first challenge"
        );
    }

    #[test]
    fn distinct_instances_diverge() {
        let mut a = prover_transcript(SESSION, [0x11; 32], Blake2b512::default());
        let mut b = prover_transcript(SESSION, [0x22; 32], Blake2b512::default());
        let ca: Fr = a.challenge_128();
        let cb: Fr = b.challenge_128();
        assert_ne!(
            ca, cb,
            "distinct instance digests must yield distinct first challenge",
        );
    }
}

#[cfg(feature = "transcript-keccak")]
mod keccak {
    use super::*;
    use spongefish::instantiations::Keccak;

    #[test]
    fn prover_verifier_round_trip() {
        let mut prover = prover_transcript(SESSION, INSTANCE, Keccak::default());
        prover.prover_message(&BytesMsg(b"a".to_vec()));
        let _c: Fr = prover.challenge_128();
        let narg = prover.narg_string().to_vec();

        let mut verifier = verifier_transcript(SESSION, INSTANCE, Keccak::default(), &narg);
        let got: BytesMsg = verifier.prover_message().expect("ok");
        assert_eq!(got.as_slice(), b"a");
        let _c2: Fr = verifier.challenge_128();
        verifier.check_eof().expect("eof");
    }
}

/// The split traits exist so protocol code can be written once, generic
/// over the sponge. These drivers touch every trait method through the
/// trait bounds only, then assert the prover and verifier transcripts agree
/// on the squeezed challenge.
mod trait_generic_surface {
    use super::*;
    use jolt_transcript::{ProverTranscript, VerifierTranscript};
    use spongefish::DuplexSpongeInterface;

    fn drive_prover<H, P>(prover: &mut P) -> (u128, Vec<u8>)
    where
        H: DuplexSpongeInterface<U = u8>,
        P: ProverTranscript<H>,
    {
        prover.public_message(&BytesMsg(b"statement".to_vec()));
        prover.prover_message(&BytesMsg(b"witness commitment".to_vec()));
        let challenge: u128 = prover.verifier_message();
        (challenge, prover.narg_string().to_vec())
    }

    fn replay_verifier<H, V>(mut verifier: V) -> u128
    where
        H: DuplexSpongeInterface<U = u8>,
        V: VerifierTranscript<H>,
    {
        verifier.public_message(&BytesMsg(b"statement".to_vec()));
        let witness: BytesMsg = verifier
            .prover_message()
            .expect("prover message must replay from the NARG");
        assert_eq!(witness.as_slice(), b"witness commitment");
        let challenge: u128 = verifier.verifier_message();
        verifier.check_eof().expect("exact NARG must pass EOF");
        challenge
    }

    #[cfg(feature = "transcript-blake2b")]
    #[test]
    fn blake2b_prover_and_verifier_agree_through_the_traits() {
        use spongefish::instantiations::Blake2b512;
        let mut prover = prover_transcript(SESSION, INSTANCE, Blake2b512::default());
        let (prover_challenge, narg) = drive_prover(&mut prover);
        let verifier = verifier_transcript(SESSION, INSTANCE, Blake2b512::default(), &narg);
        assert_eq!(prover_challenge, replay_verifier(verifier));
    }

    #[cfg(feature = "transcript-keccak")]
    #[test]
    fn keccak_prover_and_verifier_agree_through_the_traits() {
        use spongefish::instantiations::Keccak;
        let mut prover = prover_transcript(SESSION, INSTANCE, Keccak::default());
        let (prover_challenge, narg) = drive_prover(&mut prover);
        let verifier = verifier_transcript(SESSION, INSTANCE, Keccak::default(), &narg);
        assert_eq!(prover_challenge, replay_verifier(verifier));
    }
}

/// The `transcript_builder` escape hatch must land on the same transcript
/// state as the `(session, instance)` factory when fed the same bytes, or
/// power users would silently fork the Fiat-Shamir stream.
#[cfg(feature = "transcript-blake2b")]
mod builder_escape_hatch {
    use super::*;
    use jolt_transcript::transcript_builder;
    use spongefish::instantiations::Blake2b512;

    struct RawDigest([u8; 32]);

    impl spongefish::Encoding<[u8]> for RawDigest {
        fn encode(&self) -> impl AsRef<[u8]> {
            self.0
        }
    }

    #[test]
    fn builder_with_factory_wiring_matches_the_factory() {
        let mut factory = prover_transcript(SESSION, INSTANCE, Blake2b512::default());
        let mut manual = transcript_builder()
            .session(BytesMsg(SESSION.to_vec()))
            .instance(RawDigest(INSTANCE))
            .to_prover(Blake2b512::default());
        let factory_challenge: Fr = factory.challenge_128();
        let manual_challenge: Fr = manual.challenge_128();
        assert_eq!(
            factory_challenge, manual_challenge,
            "escape-hatch wiring must match prover_transcript"
        );
    }
}
