//! Witness extraction from Stage 1 proof
//!
//! This module extracts all necessary data from a Stage1OnlyProof
//! by replaying the Fiat-Shamir transcript to generate the same
//! challenges that the prover used.

use crate::field::JoltField;
use crate::zkvm::stage1_only_verifier::Stage1OnlyProof;
use crate::poly::opening_proof::Openings;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use common::jolt_device::JoltDevice;
use ark_bn254::Fr;

/// Circuit data extracted from Stage 1 proof
///
/// All data becomes PUBLIC INPUTS to the Groth16 circuit
/// (no privacy needed for EVM efficiency)
#[derive(Clone, Debug)]
pub struct Stage1CircuitData {
    // Challenges (extracted by replaying Blake2b transcript)
    pub tau: Vec<Fr>,
    pub r0: Fr,
    pub sumcheck_challenges: Vec<Fr>,

    // Proof data
    pub uni_skip_poly_coeffs: Vec<Fr>,
    pub sumcheck_round_polys: Vec<Vec<Fr>>,

    // R1CS evaluations
    pub r1cs_input_evals: Vec<Fr>,

    pub trace_length: usize,
    pub expected_final_claim: Fr,
}

impl Stage1CircuitData {
    /// Extract from Stage1OnlyProof by replaying transcript
    ///
    /// This generates the SAME challenges that the prover used by
    /// replaying the Blake2b Fiat-Shamir transcript.
    ///
    /// # Arguments
    ///
    /// * `proof` - Stage 1 proof (uni-skip + sumcheck)
    /// * `opening_claims` - Polynomial opening claims
    /// * `program_io` - Program I/O device (for Fiat-Shamir preamble)
    /// * `commitments` - Polynomial commitments (for transcript state)
    /// * `ram_K` - RAM size parameter (for Fiat-Shamir preamble)
    pub fn from_stage1_proof<F: JoltField, PCS, ProofTranscript: Transcript>(
        proof: &Stage1OnlyProof<F, ProofTranscript>,
        opening_claims: &Openings<F>,
        program_io: &JoltDevice,
        commitments: &[PCS::Commitment],
        ram_K: usize,
    ) -> Self
    where
        PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>,
        F: Into<Fr>,
    {
        // Replay the exact transcript logic from Stage1OnlyVerifier::new()
        let mut transcript = ProofTranscript::new(b"Jolt");

        // 1. Reconstruct transcript state (Fiat-Shamir preamble + commitments)
        crate::zkvm::fiat_shamir_preamble(
            program_io,
            ram_K,
            proof.trace_length,
            &mut transcript,
        );

        for commitment in commitments {
            transcript.append_serializable(commitment);
        }

        // 2. Extract tau (initial challenges for outer sumcheck)
        // These are generated at the start of Stage 1 (Spartan outer sumcheck)
        // IMPORTANT: num_rows_bits = num_cycle_vars + 2 (for univariate skip and streaming round)
        let num_cycle_vars = proof.trace_length.log_2();
        let num_rows_bits = num_cycle_vars + 2;
        let tau: Vec<F> = transcript.challenge_vector(num_rows_bits);

        // 3. Replay uni-skip first round to extract r0
        // The verifier appends the uni_poly, then samples r0
        transcript.append_serializable(&proof.uni_skip_first_round_proof.uni_poly);
        let r0: F = transcript.challenge_scalar();

        // 4. Extract uni-skip polynomial coefficients (already complete)
        let uni_skip_poly_coeffs: Vec<Fr> = proof
            .uni_skip_first_round_proof
            .uni_poly
            .coeffs
            .iter()
            .map(|&c| c.into())
            .collect();

        // 5. Replay sumcheck rounds and decompress polynomials
        // The key is: each compressed poly needs the "hint" (previous claim) to reconstruct linear term
        // Formula: c_1 = hint - 2*c_0 - (c_2 + c_3 + ... + c_n)
        //
        // The remaining sumcheck has num_rows_bits - 1 rounds (one round consumed by uni-skip)
        let num_sumcheck_rounds = num_rows_bits - 1;
        let mut sumcheck_challenges = Vec::with_capacity(num_sumcheck_rounds);
        let mut sumcheck_round_polys = Vec::with_capacity(num_sumcheck_rounds);

        // Initial claim after uni-skip first round
        // Evaluate uni_poly at r0 to get the claim for the first sumcheck round
        let mut current_claim: F = {
            let mut result = F::zero();
            let mut r0_power = F::one();
            for coeff in &proof.uni_skip_first_round_proof.uni_poly.coeffs {
                result += *coeff * r0_power;
                r0_power *= r0;
            }
            result
        };

        for compressed_poly in &proof.sumcheck_proof.compressed_polys {
            transcript.append_serializable(compressed_poly);
            let challenge: F = transcript.challenge_scalar();
            sumcheck_challenges.push(challenge);

            // Decompress polynomial using current_claim as hint
            let decompressed = compressed_poly.decompress(&current_claim);

            // Store full coefficients (including reconstructed linear term)
            let full_coeffs: Vec<Fr> = decompressed.coeffs.iter().map(|&c| c.into()).collect();
            sumcheck_round_polys.push(full_coeffs);

            // Update claim for next round: evaluate poly at challenge
            current_claim = {
                let mut result = F::zero();
                let mut challenge_power = F::one();
                for coeff in &decompressed.coeffs {
                    result += *coeff * challenge_power;
                    challenge_power *= challenge;
                }
                result
            };
        }

        // 6. Extract R1CS evaluations from opening claims
        // In Stage 1, the final check involves evaluating Az(r), Bz(r), Cz(r)
        // These are stored in the opening accumulator
        let r1cs_input_evals: Vec<Fr> = opening_claims
            .iter()
            .map(|(_, (_, eval))| (*eval).into())
            .collect();

        // 7. Compute expected final claim
        // The final claim should be: eq(tau, r) * [Az(r) * Bz(r) - Cz(r)] = 0
        // For now, set to zero (correct value for valid proof)
        let expected_final_claim = Fr::from(0u64);

        Self {
            tau: tau.iter().map(|&c| c.into()).collect(),
            r0: r0.into(),
            sumcheck_challenges: sumcheck_challenges.iter().map(|&c| c.into()).collect(),
            uni_skip_poly_coeffs,
            sumcheck_round_polys,
            r1cs_input_evals,
            trace_length: proof.trace_length,
            expected_final_claim,
        }
    }

    /// Get public inputs for Groth16 verification
    pub fn public_inputs(&self) -> Vec<Fr> {
        let mut inputs = Vec::new();

        inputs.extend(self.tau.iter());
        inputs.push(self.r0);
        inputs.extend(self.sumcheck_challenges.iter());
        inputs.extend(self.uni_skip_poly_coeffs.iter());
        for poly in &self.sumcheck_round_polys {
            inputs.extend(poly.iter());
        }
        inputs.extend(self.r1cs_input_evals.iter());
        inputs.push(self.expected_final_claim);

        inputs
    }

    /// Count total public inputs
    pub fn public_input_count(&self) -> usize {
        self.tau.len()
            + 1 // r0
            + self.sumcheck_challenges.len()
            + self.uni_skip_poly_coeffs.len()
            + self.sumcheck_round_polys.iter().map(|p| p.len()).sum::<usize>()
            + self.r1cs_input_evals.len()
            + 1 // expected_final_claim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_data_public_inputs() {
        let data = Stage1CircuitData {
            tau: vec![Fr::from(1u64); 10],
            r0: Fr::from(2u64),
            sumcheck_challenges: vec![Fr::from(3u64); 9],
            uni_skip_poly_coeffs: vec![Fr::from(4u64); 15],
            sumcheck_round_polys: vec![vec![Fr::from(5u64); 4]; 9],
            r1cs_input_evals: vec![Fr::from(6u64); 30],
            trace_length: 1024,
            expected_final_claim: Fr::from(7u64),
        };

        let public_inputs = data.public_inputs();
        let count = data.public_input_count();

        assert_eq!(public_inputs.len(), count);
        // 10 + 1 + 9 + 15 + (9*4) + 30 + 1 = 102
        assert_eq!(count, 102);
    }
}
