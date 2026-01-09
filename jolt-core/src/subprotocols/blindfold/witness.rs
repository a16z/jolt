//! BlindFold Witness Assignment
//!
//! Provides structures for assigning witness values to the verifier R1CS
//! from sumcheck proof transcripts.

use super::r1cs::VerifierR1CS;
use super::StageConfig;
use crate::field::JoltField;

/// Witness data for a single sumcheck round
#[derive(Clone, Debug)]
pub struct RoundWitness<F> {
    /// Polynomial coefficients [c0, c1, c2, c3]
    pub coeffs: Vec<F>,
    /// Challenge for this round
    pub challenge: F,
}

impl<F: JoltField> RoundWitness<F> {
    pub fn new(coeffs: Vec<F>, challenge: F) -> Self {
        Self { coeffs, challenge }
    }

    /// Evaluate the polynomial at a point using Horner's method
    pub fn evaluate(&self, x: F) -> F {
        let mut result = F::zero();
        for coeff in self.coeffs.iter().rev() {
            result = result * x + *coeff;
        }
        result
    }

    /// Compute Horner intermediates for degree-3 polynomial
    ///
    /// For g(X) = c0 + c1*X + c2*X^2 + c3*X^3 evaluated at r:
    /// - t1 = c2 + r * c3
    /// - t2 = c1 + r * t1
    /// - g(r) = c0 + r * t2
    pub fn compute_horner_intermediates(&self, r: F) -> (Vec<F>, F) {
        assert_eq!(self.coeffs.len(), 4, "Expected degree-3 polynomial");

        let c0 = self.coeffs[0];
        let c1 = self.coeffs[1];
        let c2 = self.coeffs[2];
        let c3 = self.coeffs[3];

        let t1 = c2 + r * c3;
        let t2 = c1 + r * t1;
        let next_claim = c0 + r * t2;

        (vec![t1, t2], next_claim)
    }
}

/// Witness data for a single sumcheck stage
#[derive(Clone, Debug)]
pub struct StageWitness<F> {
    /// Witness for each round in the stage
    pub rounds: Vec<RoundWitness<F>>,
}

impl<F: JoltField> StageWitness<F> {
    pub fn new(rounds: Vec<RoundWitness<F>>) -> Self {
        Self { rounds }
    }
}

/// Complete witness for the BlindFold verifier circuit
#[derive(Clone, Debug)]
pub struct BlindFoldWitness<F> {
    /// Initial claimed sum (public input)
    pub initial_claim: F,
    /// Witness data for each stage
    pub stages: Vec<StageWitness<F>>,
}

impl<F: JoltField> BlindFoldWitness<F> {
    pub fn new(initial_claim: F, stages: Vec<StageWitness<F>>) -> Self {
        Self {
            initial_claim,
            stages,
        }
    }

    /// Assign the witness to the Z vector for R1CS satisfaction checking
    ///
    /// The Z vector layout is:
    /// ```text
    /// Z = [1, challenges..., initial_claim, witness_vars...]
    /// ```
    pub fn assign(&self, r1cs: &VerifierR1CS<F>) -> Vec<F> {
        let mut z = vec![F::zero(); r1cs.num_vars];
        z[0] = F::one(); // Constant 1

        // Compute total rounds and build variable indices
        let total_rounds: usize = r1cs.stage_configs.iter().map(|s| s.num_rounds).sum();

        // Public input layout:
        // - Indices 1..=total_rounds are challenges
        // - Index total_rounds+1 is initial_claim
        let challenge_start = 1;
        let initial_claim_idx = total_rounds + 1;
        let witness_start = initial_claim_idx + 1;

        // Assign initial claim
        z[initial_claim_idx] = self.initial_claim;

        // Track current claim for chaining
        let mut current_claim = self.initial_claim;
        let mut challenge_idx = challenge_start;
        let mut witness_idx = witness_start;

        for (stage_idx, stage_witness) in self.stages.iter().enumerate() {
            let config = &r1cs.stage_configs[stage_idx];
            assert_eq!(
                stage_witness.rounds.len(),
                config.num_rounds,
                "Stage {stage_idx} has wrong number of rounds"
            );

            for round_witness in &stage_witness.rounds {
                let num_coeffs = config.poly_degree + 1;
                let num_intermediates = config.poly_degree - 1;

                assert_eq!(
                    round_witness.coeffs.len(),
                    num_coeffs,
                    "Wrong number of coefficients"
                );

                // Assign challenge (public input)
                let challenge = round_witness.challenge;
                z[challenge_idx] = challenge;
                challenge_idx += 1;

                // Compute Horner intermediates and next_claim
                let (intermediates, next_claim) =
                    round_witness.compute_horner_intermediates(challenge);

                // Verify the sum check: 2*c0 + c1 + c2 + c3 = current_claim
                let sum_check: F = F::from_u64(2) * round_witness.coeffs[0]
                    + round_witness.coeffs[1]
                    + round_witness.coeffs[2]
                    + round_witness.coeffs[3];
                assert_eq!(
                    sum_check, current_claim,
                    "Sum check failed: {sum_check} != {current_claim}"
                );

                // Assign coefficients
                for (i, coeff) in round_witness.coeffs.iter().enumerate() {
                    z[witness_idx + i] = *coeff;
                }
                witness_idx += num_coeffs;

                // Assign intermediates
                for (i, intermediate) in intermediates.iter().enumerate() {
                    z[witness_idx + i] = *intermediate;
                }
                witness_idx += num_intermediates;

                // Assign next_claim
                z[witness_idx] = next_claim;
                witness_idx += 1;

                // Chain to next round
                current_claim = next_claim;
            }
        }

        z
    }

    /// Create witness from compressed polynomial coefficients extracted from proof
    ///
    /// The compressed polynomial format stores coefficients [c1, c2, ...] (excluding c0)
    /// because c0 can be derived from the sum check: c0 = (claimed_sum - c1 - c2 - c3) / 2
    pub fn from_compressed_polys(
        initial_claim: F,
        stage_configs: &[StageConfig],
        compressed_coeffs: &[Vec<Vec<F>>], // [stage][round][coeffs without c0]
        challenges: &[Vec<F>],             // [stage][round]
    ) -> Self {
        assert_eq!(compressed_coeffs.len(), stage_configs.len());
        assert_eq!(challenges.len(), stage_configs.len());

        let mut stages = Vec::with_capacity(stage_configs.len());
        let mut current_claim = initial_claim;

        for (stage_idx, config) in stage_configs.iter().enumerate() {
            let stage_compressed = &compressed_coeffs[stage_idx];
            let stage_challenges = &challenges[stage_idx];

            assert_eq!(stage_compressed.len(), config.num_rounds);
            assert_eq!(stage_challenges.len(), config.num_rounds);

            let mut rounds = Vec::with_capacity(config.num_rounds);

            for round in 0..config.num_rounds {
                let compressed = &stage_compressed[round];
                let challenge = stage_challenges[round];

                // Decompress: c0 = (claimed_sum - c1 - c2 - c3) / 2
                // compressed = [c1, c2, c3]
                let sum_without_2c0: F = compressed.iter().copied().sum();
                let two = F::from_u64(2);
                let c0 = (current_claim - sum_without_2c0) * two.inverse().unwrap();

                let mut coeffs = vec![c0];
                coeffs.extend_from_slice(compressed);

                let round_witness = RoundWitness::new(coeffs, challenge);

                // Compute next claim for chaining
                current_claim = round_witness.evaluate(challenge);

                rounds.push(round_witness);
            }

            stages.push(StageWitness::new(rounds));
        }

        Self::new(initial_claim, stages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use ark_bn254::Fr;

    #[test]
    fn test_witness_assignment() {
        type F = Fr;

        let configs = [StageConfig::new(2, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Create valid round witnesses
        // Round 1: coeffs such that 2*c0 + c1 + c2 + c3 = 100
        let c0_1 = F::from_u64(40);
        let c1_1 = F::from_u64(5);
        let c2_1 = F::from_u64(10);
        let c3_1 = F::from_u64(5);
        // 2*40 + 5 + 10 + 5 = 100
        let r1 = F::from_u64(3);
        let round1 = RoundWitness::new(vec![c0_1, c1_1, c2_1, c3_1], r1);
        let next1 = round1.evaluate(r1);

        // Round 2: coeffs such that 2*c0 + c1 + c2 + c3 = next1
        // next1 = 40 + 3*5 + 9*10 + 27*5 = 40 + 15 + 90 + 135 = 280
        // We need 2*c0 + c1 + c2 + c3 = 280
        // Let c0 = 135, c1 = 5, c2 = 3, c3 = 2 => 270 + 5 + 3 + 2 = 280
        let c0_2 = F::from_u64(135);
        let c1_2 = F::from_u64(5);
        let c2_2 = F::from_u64(3);
        let c3_2 = F::from_u64(2);
        let r2 = F::from_u64(5);
        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], r2);

        let witness = BlindFoldWitness::new(
            F::from_u64(100),
            vec![StageWitness::new(vec![round1, round2])],
        );

        let z = witness.assign(&r1cs);
        assert!(r1cs.is_satisfied(&z), "R1CS should be satisfied");
    }

    #[test]
    fn test_compressed_poly_decompression() {
        type F = Fr;

        let initial_claim = F::from_u64(100);

        // Compressed format: [c1, c2, c3] without c0
        // If 2*c0 + c1 + c2 + c3 = 100 and [c1,c2,c3] = [5,10,5]
        // Then c0 = (100 - 5 - 10 - 5) / 2 = 40
        let compressed_coeffs = vec![vec![vec![F::from_u64(5), F::from_u64(10), F::from_u64(5)]]];
        let challenges = vec![vec![F::from_u64(3)]];
        let configs = [StageConfig::new(1, 3)];

        let witness = BlindFoldWitness::from_compressed_polys(
            initial_claim,
            &configs,
            &compressed_coeffs,
            &challenges,
        );

        assert_eq!(witness.stages[0].rounds[0].coeffs[0], F::from_u64(40));
        assert_eq!(witness.stages[0].rounds[0].coeffs[1], F::from_u64(5));
        assert_eq!(witness.stages[0].rounds[0].coeffs[2], F::from_u64(10));
        assert_eq!(witness.stages[0].rounds[0].coeffs[3], F::from_u64(5));

        // Build R1CS and verify
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();
        let z = witness.assign(&r1cs);
        assert!(r1cs.is_satisfied(&z));
    }
}
