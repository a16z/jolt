#![no_main]

//! Differential check of `verify_compressed` against an in-harness reference
//! that decompresses each wire round and evaluates it in full.
//!
//! The compressed wire format omits every round's linear coefficient; the
//! verifier recovers it from the running sum. The harness writes fuzzer-chosen
//! compressed rounds `[c0, c2, ..., cd]` into a prover transcript, computing
//! the reference along the way (decompress with the running-sum hint, draw the
//! prover-side challenge, evaluate the full polynomial). It then runs
//! `SumcheckVerifier::verify_compressed` over the resulting argument string
//! and requires it to accept, consume every byte, and return the reference's
//! challenges and final claim. A divergence means either the c₁-recovery
//! arithmetic disagrees with the polynomial it defines or the verifier's
//! challenges disagree with the prover's for the same bytes.

use jolt_field::{CanonicalEncoding, Fr};
use jolt_poly::CompressedPoly;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};
use libfuzzer_sys::fuzz_target;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck-fuzz/diff");
const SESSION: &[u8] = b"jolt-sumcheck-diff-fuzz";
const SCALAR_BYTES: usize = 32;
const MAX_NUM_VARS: usize = 8;
const MAX_DEGREE: usize = 4;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 + SCALAR_BYTES {
        return;
    }
    let num_vars = ((data[0] as usize) % MAX_NUM_VARS) + 1;
    let degree = ((data[1] as usize) % MAX_DEGREE) + 1;
    let claimed_sum = read_scalar(&data[2..2 + SCALAR_BYTES]);
    let claim = SumcheckClaim::new(num_vars, degree, claimed_sum);

    let round_bytes = degree * SCALAR_BYTES;
    let rounds = data[2 + SCALAR_BYTES..].chunks_exact(round_bytes);
    if rounds.len() < num_vars {
        return;
    }

    let mut prover = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let mut running_sum = claimed_sum;
    let mut challenges: Vec<Fr> = Vec::with_capacity(num_vars);
    for round in rounds.take(num_vars) {
        let stored: Vec<Fr> = round.chunks_exact(SCALAR_BYTES).map(read_scalar).collect();
        prover.send_all(&stored);
        let r: Fr = prover.challenge_small();
        running_sum = CompressedPoly::new(stored)
            .decompress(running_sum)
            .evaluate(r);
        challenges.push(r);
    }
    let reference = EvaluationClaim::new(challenges, running_sum);
    let narg = prover.finish();

    let mut verifier = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, &narg);
    let production = SumcheckVerifier::verify_compressed(&claim, &mut verifier)
        .expect("verify_compressed rejected well-formed wire rounds");
    verifier
        .finish()
        .expect("verify_compressed left wire rounds unread");
    assert_eq!(
        production, reference,
        "verify_compressed disagrees with decompress-then-evaluate"
    );
});

#[inline]
fn read_scalar(bytes: &[u8]) -> Fr {
    debug_assert_eq!(bytes.len(), SCALAR_BYTES);
    <Fr as CanonicalEncoding>::from_bytes_le_reduced(bytes)
}
