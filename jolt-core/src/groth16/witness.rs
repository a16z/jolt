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
        _opening_claims: &Openings<F>,
        _program_io: &JoltDevice,
        _commitments: &[PCS::Commitment],
        _ram_K: usize,
    ) -> Self
    where
        PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>,
    {
        // TODO: Implement transcript replay to extract challenges
        // This needs to match the logic in Stage1OnlyVerifier::new()

        // For now, return dummy data
        let trace_length = proof.trace_length;
        let num_rounds = trace_length.log_2();

        Self {
            tau: vec![Fr::from(1u64); num_rounds],
            r0: Fr::from(2u64),
            sumcheck_challenges: vec![Fr::from(3u64); num_rounds - 1],
            uni_skip_poly_coeffs: vec![Fr::from(4u64); 15], // Typical size
            sumcheck_round_polys: vec![vec![Fr::from(5u64); 4]; num_rounds - 1],
            r1cs_input_evals: vec![Fr::from(6u64); 30], // ~30 R1CS inputs
            trace_length,
            expected_final_claim: Fr::from(7u64),
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
