#![no_main]

use jolt_field::{Fr, CanonicalEncoding};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

const SCALAR_BYTES: usize = 32;
const MAX_NUM_VARS: usize = 8;
const MAX_DEGREE: usize = 6;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 + SCALAR_BYTES {
        return;
    }

    let num_vars = (data[0] as usize) % (MAX_NUM_VARS + 1);
    let degree = ((data[1] as usize) % MAX_DEGREE) + 1;
    let valid_rounds = (data[2] as usize) % (num_vars + 1);
    let claimed_sum = read_scalar(&data[3..3 + SCALAR_BYTES]);
    let claim = SumcheckClaim::new(num_vars, degree, claimed_sum);

    let mut cursor = 3 + SCALAR_BYTES;

    let mut prover_transcript = Blake2bTranscript::new(b"jolt-sumcheck-valid-fuzz");
    let mut running_sum = claimed_sum;
    let mut round_proofs: Vec<UnivariatePoly<Fr>> = Vec::with_capacity(num_vars);

    for round in 0..num_vars {
        if round < valid_rounds {
            let needed = SCALAR_BYTES * degree;
            if cursor + needed > data.len() {
                return;
            }
            let c0 = read_scalar(&data[cursor..cursor + SCALAR_BYTES]);
            cursor += SCALAR_BYTES;

            let mut c_high: Vec<Fr> = Vec::with_capacity(degree - 1);
            for _ in 0..(degree - 1) {
                c_high.push(read_scalar(&data[cursor..cursor + SCALAR_BYTES]));
                cursor += SCALAR_BYTES;
            }

            let mut c1 = running_sum - c0 - c0;
            for c in &c_high {
                c1 -= *c;
            }

            let mut coeffs = vec![c0, c1];
            coeffs.extend_from_slice(&c_high);
            let poly = UnivariatePoly::new(coeffs);

            for c in poly.coefficients() {
                c.append_to_transcript(&mut prover_transcript);
            }
            let r: Fr = prover_transcript.challenge();
            running_sum = poly.evaluate(r);
            round_proofs.push(poly);
        } else {
            if cursor >= data.len() {
                return;
            }
            let coeff_count = (data[cursor] as usize) % (degree + 2);
            cursor += 1;
            let needed = coeff_count * SCALAR_BYTES;
            if cursor + needed > data.len() {
                return;
            }
            let coeffs: Vec<Fr> = (0..coeff_count)
                .map(|i| {
                    read_scalar(&data[cursor + i * SCALAR_BYTES..cursor + (i + 1) * SCALAR_BYTES])
                })
                .collect();
            cursor += needed;
            round_proofs.push(UnivariatePoly::new(coeffs));
        }
    }

    let mut verifier_transcript = Blake2bTranscript::new(b"jolt-sumcheck-valid-fuzz");
    let result = SumcheckVerifier::verify::<Fr, _, UnivariatePoly<Fr>, _>(
        &claim,
        &round_proofs,
        BooleanHypercube,
        &mut verifier_transcript,
    );

    if valid_rounds == num_vars {
        let eval_claim = result.expect("fully valid proof must verify");
        assert_eq!(
            eval_claim.value, running_sum,
            "verifier final eval disagrees with prover-side running sum",
        );
        assert_eq!(eval_claim.point.len(), num_vars);
    }
});

#[inline]
fn read_scalar(bytes: &[u8]) -> Fr {
    debug_assert_eq!(bytes.len(), SCALAR_BYTES);
    <Fr as CanonicalEncoding>::from_bytes_le_reduced(bytes)
}
