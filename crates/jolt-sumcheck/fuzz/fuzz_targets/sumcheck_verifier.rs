//! Fuzz `SumcheckVerifier::verify_compressed` — the production wire path —
//! with an attacker-controlled claim and an attacker-controlled argument
//! string. The verifier MUST never panic on any input: it must either return
//! `Ok(EvaluationClaim)` or a typed [`SumcheckError`], and the transcript's
//! `finish` must then report trailing bytes or poison without panicking.
//!
//! Round width is fixed by the claim's degree bound, so the argument string
//! is read as `num_vars` rounds of `degree` coefficients. Short inputs hit
//! `TranscriptError::Truncated`, out-of-range 32-byte words hit
//! `TranscriptError::NonCanonical`, and canonical words exercise the
//! c₁-recovery arithmetic (`evaluate_with_hint`) that the full-round path
//! never reaches. Full-depth accept-path coverage lives in
//! `valid_prefix_proof`.

#![no_main]

use jolt_field::{CanonicalEncoding, Fr};
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{Blake2b512, ProtocolId, VerifierTranscript};
use libfuzzer_sys::fuzz_target;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck-fuzz/verifier");
const SESSION: &[u8] = b"jolt-sumcheck-fuzz";

/// Bytes per BN254 scalar.
const SCALAR_BYTES: usize = 32;

/// Cap on `num_vars` to keep the fuzz iteration cheap. Real sumchecks bind up
/// to ~30 variables, but fuzzing the verifier panic surface only needs a
/// handful of rounds.
const MAX_NUM_VARS: usize = 8;

/// Cap on `degree` to keep round polys small. Real sumchecks use degree
/// 2..=4; we go up to 6 to exercise the high-degree path.
const MAX_DEGREE: usize = 6;

fuzz_target!(|data: &[u8]| {
    // Header: 1 byte num_vars + 1 byte degree + 32 bytes claimed_sum; the
    // rest is the argument string.
    if data.len() < 2 + SCALAR_BYTES {
        return;
    }

    let num_vars = (data[0] as usize) % (MAX_NUM_VARS + 1);
    let degree = ((data[1] as usize) % MAX_DEGREE) + 1; // SumcheckClaim::new requires >= 1
    let claimed_sum = <Fr as CanonicalEncoding>::from_bytes_le_reduced(&data[2..2 + SCALAR_BYTES]);
    let claim = SumcheckClaim::new(num_vars, degree, claimed_sum);
    let narg = &data[2 + SCALAR_BYTES..];

    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    let _ = SumcheckVerifier::verify_compressed(&claim, &mut transcript);
    let _ = transcript.finish();
});
