//! BlindFold Witness Assignment
//!
//! Provides structures for assigning witness values to the verifier R1CS
//! from sumcheck proof transcripts.

use super::r1cs::VerifierR1CS;
use super::StageConfig;
use crate::field::JoltField;
use crate::poly::unipoly::CompressedUniPoly;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::transcripts::{AppendToTranscript, Transcript};

/// Witness data for a single sumcheck round
#[derive(Clone, Debug)]
pub struct RoundWitness<F> {
    /// Polynomial coefficients [c0, c1, c2, c3, ...]
    pub coeffs: Vec<F>,
    /// Challenge for this round
    pub challenge: F,
    /// Claimed sum for this round (g(0) + g(1) = 2*c0 + c1 + c2 + ...)
    pub claimed_sum: F,
}

impl<F: JoltField> RoundWitness<F> {
    pub fn new(coeffs: Vec<F>, challenge: F) -> Self {
        // Compute claimed_sum from coefficients: 2*c0 + c1 + c2 + ...
        let claimed_sum = F::from_u64(2) * coeffs[0] + coeffs[1..].iter().copied().sum::<F>();
        Self {
            coeffs,
            challenge,
            claimed_sum,
        }
    }

    pub fn with_claimed_sum(coeffs: Vec<F>, challenge: F, claimed_sum: F) -> Self {
        Self {
            coeffs,
            challenge,
            claimed_sum,
        }
    }

    /// Evaluate the polynomial at a point using Horner's method
    pub fn evaluate(&self, x: F) -> F {
        let mut result = F::zero();
        for coeff in self.coeffs.iter().rev() {
            result = result * x + *coeff;
        }
        result
    }

    /// Compute Horner intermediates for variable degree polynomial
    ///
    /// For g(X) = c0 + c1*X + c2*X^2 + ... + cd*X^d evaluated at r:
    /// Using Horner's method: g(r) = c0 + r*(c1 + r*(c2 + ... + r*cd))
    ///
    /// For degree d >= 2, we compute d-1 intermediates:
    /// - t[d-2] = c_{d-1} + r * c_d
    /// - t[i-1] = c_i + r * t[i]  for i from d-2 down to 1
    /// - g(r) = c0 + r * t[0]
    pub fn compute_horner_intermediates(&self, r: F) -> (Vec<F>, F) {
        let degree = self.coeffs.len() - 1;

        if degree == 0 {
            // Constant polynomial: g(r) = c0
            return (vec![], self.coeffs[0]);
        }

        if degree == 1 {
            // Linear: g(r) = c0 + c1*r (no intermediates)
            let next_claim = self.coeffs[0] + r * self.coeffs[1];
            return (vec![], next_claim);
        }

        // Degree >= 2: use Horner's method
        // Build intermediates from highest degree down
        let mut intermediates = vec![F::zero(); degree - 1];

        // First intermediate: t[d-2] = c_{d-1} + r * c_d
        intermediates[degree - 2] = self.coeffs[degree - 1] + r * self.coeffs[degree];

        // Middle intermediates: t[i-1] = c_i + r * t[i] for i from d-2 down to 1
        for i in (1..degree - 1).rev() {
            intermediates[i - 1] = self.coeffs[i] + r * intermediates[i];
        }

        // Final evaluation: g(r) = c0 + r * t[0]
        let next_claim = self.coeffs[0] + r * intermediates[0];

        (intermediates, next_claim)
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
    /// Initial claimed sums for each independent chain (public inputs).
    /// Stages with `starts_new_chain: true` use their corresponding initial claim.
    pub initial_claims: Vec<F>,
    /// Witness data for each stage
    pub stages: Vec<StageWitness<F>>,
}

impl<F: JoltField> BlindFoldWitness<F> {
    /// Create a new BlindFold witness with a single initial claim (legacy API).
    pub fn new(initial_claim: F, stages: Vec<StageWitness<F>>) -> Self {
        Self {
            initial_claims: vec![initial_claim],
            stages,
        }
    }

    /// Create a new BlindFold witness with multiple initial claims.
    /// Each initial claim corresponds to an independent chain in the R1CS.
    pub fn with_multiple_claims(initial_claims: Vec<F>, stages: Vec<StageWitness<F>>) -> Self {
        Self {
            initial_claims,
            stages,
        }
    }

    /// Assign the witness to the Z vector for R1CS satisfaction checking
    ///
    /// The Z vector layout is:
    /// ```text
    /// Z = [u, challenges..., initial_claim, witness_vars...]
    /// ```
    /// For non-relaxed instances, u = 1.
    pub fn assign(&self, r1cs: &VerifierR1CS<F>) -> Vec<F> {
        self.assign_with_u(r1cs, F::one())
    }

    /// Assign with a specific u value (for relaxed R1CS)
    pub fn assign_with_u(&self, r1cs: &VerifierR1CS<F>, u: F) -> Vec<F> {
        let mut z = vec![F::zero(); r1cs.num_vars];
        z[0] = u; // u scalar (1 for non-relaxed)

        // Compute total rounds and number of chains
        let total_rounds: usize = r1cs.stage_configs.iter().map(|s| s.num_rounds).sum();
        let num_chains = 1 + r1cs
            .stage_configs
            .iter()
            .skip(1)
            .filter(|s| s.starts_new_chain)
            .count();

        // Public input layout:
        // - Indices 1..=total_rounds are challenges
        // - Indices total_rounds+1..=total_rounds+num_chains are initial_claims
        let challenge_start = 1;
        let initial_claims_start = total_rounds + 1;
        let witness_start = initial_claims_start + num_chains;

        // Assign initial claims
        for (i, claim) in self.initial_claims.iter().enumerate() {
            z[initial_claims_start + i] = *claim;
        }

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
                let num_intermediates = config.poly_degree.saturating_sub(1);

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
            }
        }

        z
    }

    /// Create witness from compressed polynomial coefficients extracted from proof
    ///
    /// The compressed polynomial format from `CompressedUniPoly` stores `[c0, c2, c3, ...]`
    /// (excluding c1, the linear term). c1 can be derived from the sum check:
    ///   claimed_sum = g(0) + g(1) = c0 + (c0 + c1 + c2 + c3) = 2*c0 + c1 + c2 + c3
    ///   c1 = claimed_sum - 2*c0 - c2 - c3 - ...
    pub fn from_compressed_polys(
        initial_claim: F,
        stage_configs: &[StageConfig],
        compressed_coeffs: &[Vec<Vec<F>>], // [stage][round][c0, c2, c3, ...]
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

                // Decompress: c1 = claimed_sum - 2*c0 - c2 - c3 - ...
                // compressed = [c0, c2, c3, ...]
                let c0 = compressed[0];
                let sum_higher_coeffs: F = compressed[1..].iter().copied().sum();
                let c1 = current_claim - c0 - c0 - sum_higher_coeffs;

                // Build full coefficients: [c0, c1, c2, c3, ...]
                let mut coeffs = vec![c0, c1];
                coeffs.extend_from_slice(&compressed[1..]);

                let round_witness = RoundWitness::new(coeffs, challenge);

                // Compute next claim for chaining
                current_claim = round_witness.evaluate(challenge);

                rounds.push(round_witness);
            }

            stages.push(StageWitness::new(rounds));
        }

        Self::new(initial_claim, stages)
    }

    /// Create witness from a sumcheck proof by replaying verification to extract challenges.
    ///
    /// This method extracts the round polynomials from the proof and replays the
    /// transcript operations to derive the Fiat-Shamir challenges.
    ///
    /// Note: The transcript must be in the same state it was when the sumcheck
    /// verification would begin (after appending all prior protocol messages).
    pub fn from_sumcheck_proof<ProofTranscript: Transcript>(
        initial_claim: F,
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F::Challenge>) {
        let num_rounds = proof.compressed_polys.len();
        let degree = if num_rounds > 0 {
            proof.compressed_polys[0].degree()
        } else {
            3 // Default to degree 3
        };

        let _configs = [StageConfig::new(num_rounds, degree)];
        let mut current_claim = initial_claim;
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut rounds = Vec::with_capacity(num_rounds);

        for poly in &proof.compressed_polys {
            // Replay transcript operations exactly as in verification
            if let Some(ref commitments) = proof.round_commitments {
                // ZK mode: append commitment
                transcript.append_message(b"UniPolyCommitment");
                transcript.append_bytes(&commitments[challenges.len()]);
            } else {
                // Non-ZK mode: append raw coefficients
                poly.append_to_transcript(transcript);
            }

            // Derive challenge via Fiat-Shamir
            let challenge: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            challenges.push(challenge);

            // Extract compressed coefficients and decompress
            let compressed = &poly.coeffs_except_linear_term;
            let c0 = compressed[0];
            let sum_higher_coeffs: F = compressed[1..].iter().copied().sum();
            let c1 = current_claim - c0 - c0 - sum_higher_coeffs;

            let mut coeffs = vec![c0, c1];
            coeffs.extend_from_slice(&compressed[1..]);

            let round_witness = RoundWitness::new(coeffs, challenge.into());

            // Compute next claim for chaining
            current_claim = round_witness.evaluate(challenge.into());

            rounds.push(round_witness);
        }

        let witness = Self::new(initial_claim, vec![StageWitness::new(rounds)]);

        (witness, challenges)
    }

    /// Create witness from multiple sumcheck proofs (one per stage).
    ///
    /// This is useful for Jolt which has 6 sumcheck stages.
    pub fn from_multiple_sumcheck_proofs<ProofTranscript: Transcript>(
        initial_claims: &[F],
        proofs: &[&SumcheckInstanceProof<F, ProofTranscript>],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<Vec<F::Challenge>>) {
        assert!(!proofs.is_empty());
        assert_eq!(initial_claims.len(), proofs.len());

        let mut all_stages = Vec::with_capacity(proofs.len());
        let mut all_challenges = Vec::with_capacity(proofs.len());
        let mut current_claim = initial_claims[0];

        for (stage_idx, proof) in proofs.iter().enumerate() {
            let num_rounds = proof.compressed_polys.len();
            let mut stage_challenges = Vec::with_capacity(num_rounds);
            let mut rounds = Vec::with_capacity(num_rounds);

            // For batched sumchecks, the initial claim of subsequent stages
            // is computed from the batching. For simplicity, we use the
            // provided initial claims.
            if stage_idx > 0 {
                current_claim = initial_claims[stage_idx];
            }

            for poly in &proof.compressed_polys {
                // Replay transcript operations
                if let Some(ref commitments) = proof.round_commitments {
                    transcript.append_message(b"UniPolyCommitment");
                    transcript.append_bytes(&commitments[stage_challenges.len()]);
                } else {
                    poly.append_to_transcript(transcript);
                }

                let challenge: F::Challenge = transcript.challenge_scalar_optimized::<F>();
                stage_challenges.push(challenge);

                // Decompress
                let compressed = &poly.coeffs_except_linear_term;
                let c0 = compressed[0];
                let sum_higher_coeffs: F = compressed[1..].iter().copied().sum();
                let c1 = current_claim - c0 - c0 - sum_higher_coeffs;

                let mut coeffs = vec![c0, c1];
                coeffs.extend_from_slice(&compressed[1..]);

                let round_witness = RoundWitness::new(coeffs, challenge.into());
                current_claim = round_witness.evaluate(challenge.into());

                rounds.push(round_witness);
            }

            all_stages.push(StageWitness::new(rounds));
            all_challenges.push(stage_challenges);
        }

        let witness = Self::new(initial_claims[0], all_stages);
        (witness, all_challenges)
    }
}

impl<F: JoltField> CompressedUniPoly<F> {
    /// Get the compressed coefficients (c0, c2, c3, ...) excluding c1
    pub fn get_compressed_coeffs(&self) -> &[F] {
        &self.coeffs_except_linear_term
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
        let _next1 = round1.evaluate(r1);

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

        // Compressed format: [c0, c2, c3] without c1 (linear term omitted)
        // For g(x) = c0 + c1*x + c2*x^2 + c3*x^3
        // claimed_sum = g(0) + g(1) = 2*c0 + c1 + c2 + c3 = 100
        // If [c0, c2, c3] = [40, 10, 5], then c1 = 100 - 2*40 - 10 - 5 = 5
        let compressed_coeffs = vec![vec![vec![F::from_u64(40), F::from_u64(10), F::from_u64(5)]]];
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
