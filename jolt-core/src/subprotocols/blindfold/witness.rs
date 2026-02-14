//! BlindFold Witness Assignment
//!
//! Provides structures for assigning witness values to the verifier R1CS
//! from sumcheck proof transcripts.

use std::collections::HashSet;

use super::r1cs::VerifierR1CS;
use super::{OutputClaimConstraint, StageConfig, ValueSource};
use crate::field::JoltField;
use crate::poly::opening_proof::OpeningId;
use crate::poly::unipoly::CompressedUniPoly;

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
    /// Final output witness (if this stage has final_output constraint)
    pub final_output: Option<FinalOutputWitness<F>>,
    /// Initial input witness (if this stage has initial_input constraint)
    pub initial_input: Option<FinalOutputWitness<F>>,
}

impl<F: JoltField> StageWitness<F> {
    pub fn new(rounds: Vec<RoundWitness<F>>) -> Self {
        Self {
            rounds,
            final_output: None,
            initial_input: None,
        }
    }

    pub fn with_final_output(
        rounds: Vec<RoundWitness<F>>,
        final_output: FinalOutputWitness<F>,
    ) -> Self {
        Self {
            rounds,
            final_output: Some(final_output),
            initial_input: None,
        }
    }

    pub fn with_initial_input(
        rounds: Vec<RoundWitness<F>>,
        initial_input: FinalOutputWitness<F>,
    ) -> Self {
        Self {
            rounds,
            final_output: None,
            initial_input: Some(initial_input),
        }
    }

    pub fn with_both(
        rounds: Vec<RoundWitness<F>>,
        initial_input: FinalOutputWitness<F>,
        final_output: FinalOutputWitness<F>,
    ) -> Self {
        Self {
            rounds,
            final_output: Some(final_output),
            initial_input: Some(initial_input),
        }
    }
}

/// Witness data for final output binding constraint.
///
/// Supports two modes:
/// 1. Simple linear: uses batching_coefficients and evaluations
/// 2. General constraint: uses challenge_values and opening_values
#[derive(Clone, Debug, Default)]
pub struct FinalOutputWitness<F> {
    /// Batching coefficients αⱼ (public inputs, derived from transcript)
    /// Used for simple linear constraints.
    pub batching_coefficients: Vec<F>,
    /// Expected polynomial evaluations yⱼ (witness variables).
    pub evaluations: Vec<F>,
    /// Challenge values for general constraints.
    /// Layout: [batching_coeffs..., instance0_challenges..., instance1_challenges..., ...]
    pub challenge_values: Vec<F>,
    /// Opening values for general constraints.
    /// Maps OpeningId to its evaluation value.
    pub opening_values: Vec<F>,
}

/// Witness data for extra constraints appended after all stages.
#[derive(Clone, Debug, Default)]
pub struct ExtraConstraintWitness<F> {
    /// Output value for the constraint (witness variable).
    pub output_value: F,
    /// Blinding used for the evaluation commitment (witness variable).
    pub blinding: F,
    /// Challenge values for the constraint (public inputs).
    pub challenge_values: Vec<F>,
    /// Opening values required by the constraint (witness variables).
    pub opening_values: Vec<F>,
}

impl<F: JoltField> FinalOutputWitness<F> {
    pub fn new(batching_coefficients: Vec<F>, evaluations: Vec<F>) -> Self {
        debug_assert_eq!(
            batching_coefficients.len(),
            evaluations.len(),
            "Batching coefficients and evaluations must have same length"
        );
        Self {
            batching_coefficients,
            evaluations,
            challenge_values: Vec::new(),
            opening_values: Vec::new(),
        }
    }

    /// Create a general constraint witness with challenge and opening values.
    pub fn new_general(challenge_values: Vec<F>, opening_values: Vec<F>) -> Self {
        Self {
            batching_coefficients: Vec::new(),
            evaluations: Vec::new(),
            challenge_values,
            opening_values,
        }
    }

    /// Compute the expected final claim: Σⱼ αⱼ · yⱼ
    pub fn compute_expected_final_claim(&self) -> F {
        self.batching_coefficients
            .iter()
            .zip(&self.evaluations)
            .map(|(alpha, y)| *alpha * *y)
            .sum()
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
    /// Extra constraints appended after all stages (e.g., PCS binding)
    pub extra_constraints: Vec<ExtraConstraintWitness<F>>,
}

impl<F: JoltField> BlindFoldWitness<F> {
    /// Create a new BlindFold witness with a single initial claim (legacy API).
    pub fn new(initial_claim: F, stages: Vec<StageWitness<F>>) -> Self {
        Self {
            initial_claims: vec![initial_claim],
            stages,
            extra_constraints: Vec::new(),
        }
    }

    /// Create a new BlindFold witness with multiple initial claims.
    /// Each initial claim corresponds to an independent chain in the R1CS.
    pub fn with_multiple_claims(initial_claims: Vec<F>, stages: Vec<StageWitness<F>>) -> Self {
        Self {
            initial_claims,
            stages,
            extra_constraints: Vec::new(),
        }
    }

    /// Create a new BlindFold witness with extra constraints appended after all stages.
    pub fn with_extra_constraints(
        initial_claims: Vec<F>,
        stages: Vec<StageWitness<F>>,
        extra_constraints: Vec<ExtraConstraintWitness<F>>,
    ) -> Self {
        Self {
            initial_claims,
            stages,
            extra_constraints,
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

    /// Assign with a specific u value (for relaxed R1CS).
    ///
    /// Z layout (public inputs baked into matrix coefficients):
    /// ```text
    /// Z = [u, W_grid...]
    /// ```
    /// W_grid is R' × C: coefficient rows, then non-coefficient values.
    pub fn assign_with_u(&self, r1cs: &VerifierR1CS<F>, u: F) -> Vec<F> {
        let mut z = vec![F::zero(); r1cs.num_vars];
        z[0] = u;

        let hyrax_C = r1cs.hyrax.C;
        let hyrax_R_coeff = r1cs.hyrax.R_coeff;

        // Witness starts at index 1 (right after u, no public inputs)
        let witness_start = 1;
        // Non-coeff witness variables start after the coefficient grid
        let mut witness_idx = witness_start + hyrax_R_coeff * hyrax_C;
        let mut round_idx = 0usize;

        let mut assigned_openings: HashSet<OpeningId> = HashSet::new();

        for (stage_idx, stage_witness) in self.stages.iter().enumerate() {
            let config = &r1cs.stage_configs[stage_idx];
            assert_eq!(
                stage_witness.rounds.len(),
                config.num_rounds,
                "Stage {stage_idx} has wrong number of rounds"
            );

            // Initial input witness (opening vars + aux vars only, no challenge public inputs)
            if let Some(ref ii_config) = config.initial_input {
                if let Some(constraint) = &ii_config.constraint {
                    let num_aux_vars = constraint.estimate_aux_var_count();

                    if let Some(iw) = stage_witness.initial_input.as_ref() {
                        for (opening_id, val) in
                            constraint.required_openings.iter().zip(&iw.opening_values)
                        {
                            if !assigned_openings.contains(opening_id) {
                                z[witness_idx] = *val;
                                witness_idx += 1;
                                assigned_openings.insert(*opening_id);
                            }
                        }

                        let aux_values = Self::compute_aux_vars(
                            constraint,
                            &iw.opening_values,
                            &iw.challenge_values,
                        );
                        for val in aux_values {
                            z[witness_idx] = val;
                            witness_idx += 1;
                        }
                    } else {
                        let num_new_openings = constraint
                            .required_openings
                            .iter()
                            .filter(|id| {
                                if assigned_openings.contains(id) {
                                    false
                                } else {
                                    assigned_openings.insert(**id);
                                    true
                                }
                            })
                            .count();
                        witness_idx += num_new_openings + num_aux_vars;
                    }
                }
            }

            // Round witnesses: coefficients to grid, next_claim to non-coeff section
            for round_witness in &stage_witness.rounds {
                assert_eq!(
                    round_witness.coeffs.len(),
                    config.poly_degree + 1,
                    "Wrong number of coefficients"
                );

                for (k, coeff) in round_witness.coeffs.iter().enumerate() {
                    z[witness_start + round_idx * hyrax_C + k] = *coeff;
                }

                let next_claim = round_witness.evaluate(round_witness.challenge);
                z[witness_idx] = next_claim;
                witness_idx += 1;

                round_idx += 1;
            }

            // Final output witness
            if let Some(ref fo_config) = config.final_output {
                if let Some(exact_vars) = fo_config.exact_num_witness_vars {
                    witness_idx += exact_vars;
                } else if let Some(constraint) = &fo_config.constraint {
                    let num_aux_vars = constraint.estimate_aux_var_count();
                    let fo_witness = stage_witness.final_output.as_ref();

                    if let Some(fw) = fo_witness {
                        for (opening_id, val) in
                            constraint.required_openings.iter().zip(&fw.opening_values)
                        {
                            if !assigned_openings.contains(opening_id) {
                                z[witness_idx] = *val;
                                witness_idx += 1;
                                assigned_openings.insert(*opening_id);
                            }
                        }

                        let aux_values = Self::compute_aux_vars(
                            constraint,
                            &fw.opening_values,
                            &fw.challenge_values,
                        );
                        for val in aux_values {
                            z[witness_idx] = val;
                            witness_idx += 1;
                        }
                    } else {
                        let num_new_openings = constraint
                            .required_openings
                            .iter()
                            .filter(|id| {
                                if assigned_openings.contains(id) {
                                    false
                                } else {
                                    assigned_openings.insert(**id);
                                    true
                                }
                            })
                            .count();
                        witness_idx += num_new_openings + num_aux_vars;
                    }
                } else {
                    // Simple linear: evaluation values only (no batching coeffs, no accumulators)
                    let fw = stage_witness
                        .final_output
                        .as_ref()
                        .expect("Stage has final_output config but witness has no final_output");
                    assert_eq!(fw.evaluations.len(), fo_config.num_evaluations);

                    for eval in &fw.evaluations {
                        z[witness_idx] = *eval;
                        witness_idx += 1;
                    }
                }
            }
        }

        // Extra constraints: opening vars + output + aux vars + blinding (no challenge public inputs)
        for (extra_idx, constraint) in r1cs.extra_constraints.iter().enumerate() {
            let extra_witness = self.extra_constraints.get(extra_idx);
            let num_aux_vars = constraint.estimate_aux_var_count();

            if let Some(witness) = extra_witness {
                for (opening_id, val) in constraint
                    .required_openings
                    .iter()
                    .zip(&witness.opening_values)
                {
                    if !assigned_openings.contains(opening_id) {
                        z[witness_idx] = *val;
                        witness_idx += 1;
                        assigned_openings.insert(*opening_id);
                    }
                }

                z[witness_idx] = witness.output_value;
                witness_idx += 1;

                let aux_values = Self::compute_aux_vars(
                    constraint,
                    &witness.opening_values,
                    &witness.challenge_values,
                );
                for val in aux_values {
                    z[witness_idx] = val;
                    witness_idx += 1;
                }

                z[witness_idx] = witness.blinding;
                witness_idx += 1;
            } else {
                let num_new_openings = constraint
                    .required_openings
                    .iter()
                    .filter(|id| {
                        if assigned_openings.contains(id) {
                            false
                        } else {
                            assigned_openings.insert(**id);
                            true
                        }
                    })
                    .count();
                witness_idx += num_new_openings + 1 + num_aux_vars + 1;
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

    /// Compute auxiliary variables for a general constraint.
    ///
    /// This must match the order of aux var allocation in `add_sum_of_products_constraint`.
    /// The R1CS builder allocates aux vars in this order per term:
    /// - No factors: 1 aux var (coeff * 1)
    /// - Single factor: 1 aux var (coeff * factor)
    /// - Multiple factors: (n-1) aux vars for chain multiplication + 1 for coeff*product
    fn compute_aux_vars(
        constraint: &OutputClaimConstraint,
        opening_values: &[F],
        challenge_values: &[F],
    ) -> Vec<F> {
        // Build a map from OpeningId to value index
        let opening_map: std::collections::HashMap<OpeningId, usize> = constraint
            .required_openings
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        // Helper to resolve a ValueSource to a field element
        let resolve = |vs: &ValueSource| -> F {
            match vs {
                ValueSource::Opening(id) => {
                    let idx = *opening_map
                        .get(id)
                        .unwrap_or_else(|| panic!("Opening {id:?} not found in required_openings"));
                    opening_values[idx]
                }
                ValueSource::Challenge(idx) => challenge_values[*idx],
                ValueSource::Constant(val) => F::from_i128(*val),
            }
        };

        let mut aux_vars = Vec::new();

        for term in &constraint.terms {
            let coeff = resolve(&term.coeff);

            if term.factors.is_empty() {
                // No factors: aux = coeff * 1 = coeff
                aux_vars.push(coeff);
            } else if term.factors.len() == 1 {
                // Single factor: aux = coeff * factor
                let factor = resolve(&term.factors[0]);
                aux_vars.push(coeff * factor);
            } else {
                // Multiple factors: chain multiplication
                // aux0 = f0 * f1
                let f0 = resolve(&term.factors[0]);
                let f1 = resolve(&term.factors[1]);
                let mut current_product = f0 * f1;
                aux_vars.push(current_product);

                // aux_i = aux_{i-1} * f_{i+1}
                for factor in &term.factors[2..] {
                    let f = resolve(factor);
                    current_product *= f;
                    aux_vars.push(current_product);
                }

                // final_aux = coeff * product
                aux_vars.push(coeff * current_product);
            }
        }

        aux_vars
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
    use crate::subprotocols::blindfold::BakedPublicInputs;
    use ark_bn254::Fr;

    #[test]
    fn test_witness_assignment() {
        type F = Fr;

        let configs = [StageConfig::new(2, 3)];

        let round1 = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let round2 = RoundWitness::new(
            vec![
                F::from_u64(135),
                F::from_u64(5),
                F::from_u64(3),
                F::from_u64(2),
            ],
            F::from_u64(5),
        );

        let witness = BlindFoldWitness::new(
            F::from_u64(100),
            vec![StageWitness::new(vec![round1, round2])],
        );

        let baked = BakedPublicInputs::from_witness(&witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let z = witness.assign(&r1cs);
        assert!(r1cs.is_satisfied(&z), "R1CS should be satisfied");
    }

    #[test]
    fn test_compressed_poly_decompression() {
        type F = Fr;

        let initial_claim = F::from_u64(100);
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

        let baked = BakedPublicInputs::from_witness(&witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();
        let z = witness.assign(&r1cs);
        assert!(r1cs.is_satisfied(&z));
    }
}
