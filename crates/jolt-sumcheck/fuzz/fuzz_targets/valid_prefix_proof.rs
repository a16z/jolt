//! Fuzz `SumcheckVerifier::verify` with argument strings whose first `K`
//! rounds are written by an honest-shaped prover to satisfy the sum-check
//! invariant (`s_i(0) + s_i(1) = running_sum_i`), followed by raw fuzzer
//! bytes.
//!
//! Why this exists alongside `sumcheck_verifier`:
//! raw bytes almost always fail the first round (non-canonical word or round
//! check) and return early, leaving every Fiat-Shamir step after round 0
//! unexercised. Valid leading rounds drive the verifier round by round
//! through `receive_full_round`, `challenge_small`, and `evaluate` over the
//! full depth of the protocol, with the raw tail landing at any round.
//!
//! Per-round bytes pick `c_0` and `c_2 .. c_d` for valid rounds; the
//! linear coefficient `c_1` is derived from the sum-check invariant. When
//! `K == num_vars` no tail is appended, the argument string is valid by
//! construction, and the verifier MUST accept, consume every byte, and
//! return the running sum computed prover-side.
//!
//! Addresses review on PR #1493.

#![no_main]

use jolt_field::{CanonicalEncoding, Fr};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{send_full_round, BooleanHypercube, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};
use libfuzzer_sys::fuzz_target;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck-fuzz/valid-prefix");
const SESSION: &[u8] = b"jolt-sumcheck-valid-fuzz";
const SCALAR_BYTES: usize = 32;
const MAX_NUM_VARS: usize = 8;
const MAX_DEGREE: usize = 6;

fuzz_target!(|data: &[u8]| {
    // Header: 1 byte num_vars + 1 byte degree + 1 byte valid_rounds + 32 bytes claimed_sum.
    if data.len() < 3 + SCALAR_BYTES {
        return;
    }

    let num_vars = (data[0] as usize) % (MAX_NUM_VARS + 1);
    let degree = ((data[1] as usize) % MAX_DEGREE) + 1;
    let valid_rounds = (data[2] as usize) % (num_vars + 1);
    let claimed_sum = read_scalar(&data[3..3 + SCALAR_BYTES]);
    let claim = SumcheckClaim::new(num_vars, degree, claimed_sum);

    let mut cursor = 3 + SCALAR_BYTES;
    let mut prover = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let mut running_sum = claimed_sum;

    for _ in 0..valid_rounds {
        // `degree` scalars: c_0 + (degree - 1) high-order coefficients.
        let needed = SCALAR_BYTES * degree;
        if cursor + needed > data.len() {
            return;
        }
        let mut scalars = data[cursor..cursor + needed]
            .chunks_exact(SCALAR_BYTES)
            .map(read_scalar);
        cursor += needed;
        let c0 = scalars.next().expect("degree >= 1");
        let c_high: Vec<Fr> = scalars.collect();

        // s(0) + s(1) = 2·c_0 + c_1 + c_2 + … + c_d = running_sum
        let c1 = c_high.iter().fold(running_sum - c0 - c0, |acc, &c| acc - c);
        let mut coeffs = vec![c0, c1];
        coeffs.extend_from_slice(&c_high);
        let poly = UnivariatePoly::new(coeffs);

        send_full_round(&poly, degree, &mut prover).expect("round fits its degree bound");
        let r: Fr = prover.challenge_small();
        running_sum = poly.evaluate(r);
    }

    let mut narg = prover.finish();
    let fully_valid = valid_rounds == num_vars;
    if !fully_valid {
        narg.extend_from_slice(&data[cursor..]);
    }

    let mut verifier = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, &narg);
    let result = SumcheckVerifier::verify(&claim, BooleanHypercube, &mut verifier);
    let finished = verifier.finish();

    if fully_valid {
        let eval_claim = result.expect("fully valid proof must verify");
        finished.expect("fully valid proof must be consumed exactly");
        assert_eq!(
            eval_claim.value, running_sum,
            "verifier final eval disagrees with prover-side running sum",
        );
        assert_eq!(eval_claim.point.len(), num_vars);
    }
    // Otherwise the raw tail may or may not happen to verify; the contract
    // is just no panic.
});

#[inline]
fn read_scalar(bytes: &[u8]) -> Fr {
    debug_assert_eq!(bytes.len(), SCALAR_BYTES);
    <Fr as CanonicalEncoding>::from_bytes_le_reduced(bytes)
}
