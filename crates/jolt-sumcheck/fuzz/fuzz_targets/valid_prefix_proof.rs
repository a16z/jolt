//! Fuzz `SumcheckVerifier::verify` with proofs whose first `K` round
//! polynomials are constructed to satisfy the sum-check invariant
//! (`s_i(0) + s_i(1) = running_sum_i`). The remaining rounds use raw
//! random bytes from the fuzzer.
//!
//! Why this exists alongside `sumcheck_verifier`:
//! the panic-guard target tends to fail the verifier's first sum-check
//! comparison (round 0) and return `Err` early, leaving every Fiat-Shamir
//! transcript step after round 0 unexercised. By feeding valid leading
//! rounds, the verifier proceeds round-by-round, exercising
//! `append_to_transcript`, `challenge`, and `evaluate` over the full
//! depth of the protocol.
//!
//! Per-round bytes pick `c_0` and `c_2 .. c_d` for valid rounds; the
//! linear coefficient `c_1` is derived from the sum-check invariant
//! exactly as the standard compressed-unipoly format does. When
//! `K == num_vars` the entire proof is valid by construction and the
//! verifier MUST accept and return the running sum we computed
//! prover-side.
//!
//! Addresses review on PR #1493.

#![no_main]

use jolt_field::{Fr, ReducingBytes};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{prover_transcript, Blake2b512, FsAbsorb, FsChallenge};
use libfuzzer_sys::fuzz_target;

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

    // Mirror the verifier's transcript steps prover-side over the first
    // `valid_rounds` rounds, so the proof is valid by construction up to
    // round `valid_rounds - 1` and the verifier's running sum after
    // ingesting those rounds matches what we compute below.
    let mut pt = prover_transcript(b"jolt-sumcheck-valid-fuzz", [0u8; 32], Blake2b512::default());
    let mut running_sum = claimed_sum;
    let mut round_proofs: Vec<UnivariatePoly<Fr>> = Vec::with_capacity(num_vars);

    for round in 0..num_vars {
        if round < valid_rounds {
            // Need `degree` scalars: c_0 + (degree - 1) high-order coefficients.
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

            // Derive c_1 from the sum-check invariant:
            //   s(0) + s(1) = c_0 + (c_0 + c_1 + c_2 + … + c_d) = running_sum
            //   ⇒ c_1 = running_sum − 2·c_0 − c_2 − … − c_d
            let mut c1 = running_sum - c0 - c0;
            for c in &c_high {
                c1 -= *c;
            }

            let mut coeffs = vec![c0, c1];
            coeffs.extend_from_slice(&c_high);
            let poly = UnivariatePoly::new(coeffs);

            // Mirror the verifier's transcript / challenge / evaluate sequence
            // EXACTLY. The verifier absorbs the round polynomial as a single
            // `absorb_field_slice(coefficients())` message (see
            // `RoundMessage for UnivariatePoly`), so the prover side must too —
            // absorbing coefficients one-by-one is a different sponge input and
            // yields a different challenge. `pt` is a concrete `ProverState`,
            // which carries spongefish's deprecated inherent `challenge`; call
            // the `FsChallenge` trait method explicitly so we get the migrated
            // challenge, not the inherent one.
            pt.absorb_field_slice(poly.coefficients());
            let r: Fr = FsChallenge::<Fr>::challenge(&mut pt);
            running_sum = poly.evaluate(r);
            round_proofs.push(poly);
        } else {
            // Tail rounds: variable-length raw polynomial from fuzzer bytes.
            // Most of these will fail the verifier's sum check; the contract
            // is that `verify` returns an `Err` (or `Ok` if the random bytes
            // happen to land on a valid value) without panicking.
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

    let mut vt = prover_transcript(b"jolt-sumcheck-valid-fuzz", [0u8; 32], Blake2b512::default());
    let result = SumcheckVerifier::verify::<Fr, _, UnivariatePoly<Fr>, _>(
        &claim,
        &round_proofs,
        BooleanHypercube,
        &mut vt,
    );

    if valid_rounds == num_vars {
        // Proof is valid by construction across every round. The verifier
        // MUST accept and return the same running sum we computed.
        let eval_claim = result.expect("fully valid proof must verify");
        assert_eq!(
            eval_claim.value, running_sum,
            "verifier final eval disagrees with prover-side running sum",
        );
        assert_eq!(eval_claim.point.len(), num_vars);
    }
    // Otherwise (`valid_rounds < num_vars`) `verify` may return Ok or Err
    // depending on the random tail; the contract is just no panic.
});

#[inline]
fn read_scalar(bytes: &[u8]) -> Fr {
    debug_assert_eq!(bytes.len(), SCALAR_BYTES);
    <Fr as ReducingBytes>::from_le_bytes_mod_order(bytes)
}
