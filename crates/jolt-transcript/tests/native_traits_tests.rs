//! Coverage for the native split-trait surface — `ProverTranscript`,
//! `VerifierTranscript`, `OptimizedChallenge`. The existing per-backend
//! `transcript_tests!` macro exercises only the `legacy::Transcript`
//! facade.

#![cfg(any(
    feature = "transcript-blake2b",
    feature = "transcript-keccak",
    feature = "transcript-poseidon"
))]
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

#[cfg(feature = "transcript-poseidon")]
mod poseidon {
    use super::*;
    use jolt_transcript::PoseidonSponge;

    /// Prover and verifier squeeze the SAME `challenge_128` at the same
    /// transcript position — the field-native Poseidon sponge agreement.
    #[test]
    fn prover_verifier_round_trip_and_challenge_agree() {
        let mut prover = prover_transcript(SESSION, INSTANCE, PoseidonSponge::default());
        prover.public_message(&BytesMsg(b"pub".to_vec()));
        prover.prover_message(&BytesMsg(b"private".to_vec()));
        let c_prover: Fr = prover.challenge_128();
        let narg = prover.narg_string().to_vec();

        let mut verifier = verifier_transcript(SESSION, INSTANCE, PoseidonSponge::default(), &narg);
        verifier.public_message(&BytesMsg(b"pub".to_vec()));
        let got: BytesMsg = verifier
            .prover_message()
            .expect("prover_message must deserialize");
        assert_eq!(got.as_slice(), b"private");
        let c_verifier: Fr = verifier.challenge_128();
        verifier.check_eof().expect("eof");

        assert_eq!(
            c_prover, c_verifier,
            "Poseidon challenge_128 must agree between prover and verifier",
        );
    }

    /// Poseidon's `challenge_128` keeps only the low 128 bits of the field
    /// squeeze (the D5b decision), so the high two limbs are zero.
    #[test]
    fn optimized_challenge_is_128_bit_truncated() {
        use ark_ff::PrimeField;
        let mut prover = prover_transcript(SESSION, INSTANCE, PoseidonSponge::default());
        let c: Fr = prover.challenge_128();
        let ark_c: ark_bn254::Fr = c.into();
        let bigint = ark_c.into_bigint().0;
        assert_eq!(
            bigint[2], 0,
            "limb 2 leaked — challenge_128 exceeds 128 bits"
        );
        assert_eq!(
            bigint[3], 0,
            "limb 3 leaked — challenge_128 exceeds 128 bits"
        );
    }
}
