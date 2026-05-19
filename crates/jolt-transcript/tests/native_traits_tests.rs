//! Coverage for the native split-trait surface — `ProverTranscript`,
//! `VerifierTranscript`, `OptimizedChallenge`. The existing per-backend
//! `transcript_tests!` macro exercises only the `legacy::Transcript`
//! facade.

#![cfg(any(feature = "transcript-blake2b", feature = "transcript-keccak"))]
#![expect(clippy::expect_used, reason = "tests")]

use jolt_field::Fr;
use jolt_transcript::{
    prover_transcript, verifier_transcript, BytesMsg, OptimizedChallenge, ProverTranscript,
    VerifierTranscript,
};

const SESSION: &[u8] = b"native-traits";
const INSTANCE: [u8; 32] = [0x77; 32];

#[cfg(feature = "transcript-blake2b")]
mod blake2b {
    use super::*;
    use spongefish::instantiations::Blake2b512;

    #[test]
    fn prover_verifier_round_trip() {
        let mut prover = prover_transcript(SESSION, INSTANCE, Blake2b512::default());
        ProverTranscript::<Blake2b512>::public_message(&mut prover, &BytesMsg(b"pub".to_vec()));
        ProverTranscript::<Blake2b512>::prover_message(&mut prover, &BytesMsg(b"private".to_vec()));
        let _c1: Fr =
            <spongefish::ProverState<Blake2b512> as OptimizedChallenge>::challenge_128(&mut prover);
        let narg = ProverTranscript::<Blake2b512>::narg_string(&prover).to_vec();

        let mut verifier = verifier_transcript(SESSION, INSTANCE, Blake2b512::default(), &narg);
        VerifierTranscript::<Blake2b512>::public_message(&mut verifier, &BytesMsg(b"pub".to_vec()));
        let got: BytesMsg = VerifierTranscript::<Blake2b512>::prover_message(&mut verifier)
            .expect("prover_message must deserialize");
        assert_eq!(got.as_slice(), b"private");
        let _c2: Fr =
            <spongefish::VerifierState<'_, Blake2b512> as OptimizedChallenge>::challenge_128(
                &mut verifier,
            );
        VerifierTranscript::<Blake2b512>::check_eof(verifier).expect("eof");
    }

    #[test]
    fn optimized_challenge_is_128_bit_truncated() {
        use ark_ff::PrimeField;
        let mut prover = prover_transcript(SESSION, INSTANCE, Blake2b512::default());
        let c: Fr =
            <spongefish::ProverState<Blake2b512> as OptimizedChallenge>::challenge_128(&mut prover);
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
        let ca: Fr =
            <spongefish::ProverState<Blake2b512> as OptimizedChallenge>::challenge_128(&mut a);
        let cb: Fr =
            <spongefish::ProverState<Blake2b512> as OptimizedChallenge>::challenge_128(&mut b);
        assert_ne!(
            ca, cb,
            "distinct session bytes must yield distinct first challenge"
        );
    }

    #[test]
    fn distinct_instances_diverge() {
        let mut a = prover_transcript(SESSION, [0x11; 32], Blake2b512::default());
        let mut b = prover_transcript(SESSION, [0x22; 32], Blake2b512::default());
        let ca: Fr =
            <spongefish::ProverState<Blake2b512> as OptimizedChallenge>::challenge_128(&mut a);
        let cb: Fr =
            <spongefish::ProverState<Blake2b512> as OptimizedChallenge>::challenge_128(&mut b);
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
        ProverTranscript::<Keccak>::prover_message(&mut prover, &BytesMsg(b"a".to_vec()));
        let _c: Fr =
            <spongefish::ProverState<Keccak> as OptimizedChallenge>::challenge_128(&mut prover);
        let narg = ProverTranscript::<Keccak>::narg_string(&prover).to_vec();

        let mut verifier = verifier_transcript(SESSION, INSTANCE, Keccak::default(), &narg);
        let got: BytesMsg =
            VerifierTranscript::<Keccak>::prover_message(&mut verifier).expect("ok");
        assert_eq!(got.as_slice(), b"a");
        let _c2: Fr = <spongefish::VerifierState<'_, Keccak> as OptimizedChallenge>::challenge_128(
            &mut verifier,
        );
        VerifierTranscript::<Keccak>::check_eof(verifier).expect("eof");
    }
}
