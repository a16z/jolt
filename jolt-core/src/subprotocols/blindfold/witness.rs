//! BlindFold Witness Assignment
//!
//! Provides structures for assigning witness values to the verifier R1CS
//! from sumcheck proof transcripts.

use std::collections::HashSet;

use super::r1cs::VerifierR1CS;
use super::{OutputClaimConstraint, StageConfig};
use crate::field::JoltField;
use crate::poly::opening_proof::OpeningId;
use crate::subprotocols::constraint_types::{SumOfProductsVisitor, ValueSource};

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

/// Witness data for final output or initial input binding constraint.
#[derive(Clone, Debug)]
pub enum FinalOutputWitness<F> {
    /// Simple linear: final_claim = Σⱼ αⱼ · yⱼ
    Linear {
        batching_coefficients: Vec<F>,
        evaluations: Vec<F>,
    },
    /// General sum-of-products constraint
    General {
        challenge_values: Vec<F>,
        opening_values: Vec<F>,
    },
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
    pub fn linear(batching_coefficients: Vec<F>, evaluations: Vec<F>) -> Self {
        debug_assert_eq!(batching_coefficients.len(), evaluations.len());
        Self::Linear {
            batching_coefficients,
            evaluations,
        }
    }

    pub fn general(challenge_values: Vec<F>, opening_values: Vec<F>) -> Self {
        Self::General {
            challenge_values,
            opening_values,
        }
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
        use super::layout::{compute_witness_layout, ConstraintKind, LayoutStep};

        let mut z = vec![F::zero(); r1cs.num_vars];
        z[0] = u;

        let hyrax_C = r1cs.hyrax.C;
        let hyrax_R_coeff = r1cs.hyrax.R_coeff;
        let witness_start = 1;
        let mut noncoeff_idx = witness_start + hyrax_R_coeff * hyrax_C;

        let layout = compute_witness_layout(&r1cs.stage_configs, &r1cs.extra_constraints);
        let mut assigned_openings: HashSet<OpeningId> = HashSet::new();

        for step in &layout {
            match step {
                LayoutStep::ConstantInitialClaim { .. } => {}
                LayoutStep::InitialClaimVar { chain_idx } => {
                    z[noncoeff_idx] = self.initial_claims[*chain_idx];
                    noncoeff_idx += 1;
                }
                LayoutStep::ConstraintVars {
                    constraint,
                    new_opening_count,
                    aux_var_count,
                    kind,
                    stage_idx,
                } => {
                    let witness_data = match kind {
                        ConstraintKind::InitialInput => self.stages[*stage_idx]
                            .initial_input
                            .as_ref()
                            .and_then(|w| match w {
                                FinalOutputWitness::General {
                                    opening_values,
                                    challenge_values,
                                } => Some((opening_values.as_slice(), challenge_values.as_slice())),
                                _ => None,
                            }),
                        ConstraintKind::FinalOutput => self.stages[*stage_idx]
                            .final_output
                            .as_ref()
                            .and_then(|w| match w {
                                FinalOutputWitness::General {
                                    opening_values,
                                    challenge_values,
                                } => Some((opening_values.as_slice(), challenge_values.as_slice())),
                                _ => None,
                            }),
                    };

                    if let Some((opening_values, challenge_values)) = witness_data {
                        for (opening_id, val) in
                            constraint.required_openings.iter().zip(opening_values)
                        {
                            if assigned_openings.insert(*opening_id) {
                                z[noncoeff_idx] = *val;
                                noncoeff_idx += 1;
                            }
                        }

                        let aux_values =
                            Self::compute_aux_vars(constraint, opening_values, challenge_values);
                        debug_assert_eq!(aux_values.len(), *aux_var_count);
                        for val in aux_values {
                            z[noncoeff_idx] = val;
                            noncoeff_idx += 1;
                        }
                    } else {
                        noncoeff_idx += new_opening_count + aux_var_count;
                    }
                }
                LayoutStep::CoeffRow {
                    round_idx,
                    num_coeffs,
                    stage_idx,
                    round_in_stage,
                } => {
                    let round_witness = &self.stages[*stage_idx].rounds[*round_in_stage];
                    assert_eq!(round_witness.coeffs.len(), *num_coeffs);

                    for (k, coeff) in round_witness.coeffs.iter().enumerate() {
                        z[witness_start + round_idx * hyrax_C + k] = *coeff;
                    }
                }
                LayoutStep::NextClaim {
                    stage_idx,
                    round_in_stage,
                } => {
                    let round_witness = &self.stages[*stage_idx].rounds[*round_in_stage];
                    let next_claim = round_witness.evaluate(round_witness.challenge);
                    z[noncoeff_idx] = next_claim;
                    noncoeff_idx += 1;
                }
                LayoutStep::LinearFinalOutput {
                    num_evaluations,
                    stage_idx,
                } => {
                    let fw = self.stages[*stage_idx]
                        .final_output
                        .as_ref()
                        .expect("Missing final_output witness for LinearFinalOutput step");
                    let evaluations = match fw {
                        FinalOutputWitness::Linear { evaluations, .. } => evaluations,
                        _ => panic!("Expected Linear variant for simple final output"),
                    };
                    assert_eq!(evaluations.len(), *num_evaluations);

                    for eval in evaluations {
                        z[noncoeff_idx] = *eval;
                        noncoeff_idx += 1;
                    }
                }
                LayoutStep::PlaceholderVars { num_vars } => {
                    noncoeff_idx += num_vars;
                }
                LayoutStep::ExtraConstraintVars {
                    constraint,
                    new_opening_count,
                    aux_var_count,
                    extra_idx,
                } => {
                    if let Some(witness) = self.extra_constraints.get(*extra_idx) {
                        for (opening_id, val) in constraint
                            .required_openings
                            .iter()
                            .zip(&witness.opening_values)
                        {
                            if assigned_openings.insert(*opening_id) {
                                z[noncoeff_idx] = *val;
                                noncoeff_idx += 1;
                            }
                        }

                        z[noncoeff_idx] = witness.output_value;
                        noncoeff_idx += 1;

                        let aux_values = Self::compute_aux_vars(
                            constraint,
                            &witness.opening_values,
                            &witness.challenge_values,
                        );
                        debug_assert_eq!(aux_values.len(), *aux_var_count);
                        for val in aux_values {
                            z[noncoeff_idx] = val;
                            noncoeff_idx += 1;
                        }

                        z[noncoeff_idx] = witness.blinding;
                        noncoeff_idx += 1;
                    } else {
                        noncoeff_idx += new_opening_count + 1 + aux_var_count + 1;
                    }
                }
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
        let mut visitor = WitnessAuxVisitor::new(constraint, opening_values, challenge_values);
        let mut aux_vars = Vec::new();
        constraint.visit(&mut visitor, &mut aux_vars);
        aux_vars
    }
}

struct WitnessAuxVisitor<'a, F> {
    opening_map: std::collections::HashMap<OpeningId, usize>,
    opening_values: &'a [F],
    challenge_values: &'a [F],
    current_product: F,
}

impl<'a, F: JoltField> WitnessAuxVisitor<'a, F> {
    fn new(
        constraint: &OutputClaimConstraint,
        opening_values: &'a [F],
        challenge_values: &'a [F],
    ) -> Self {
        let opening_map = constraint
            .required_openings
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        Self {
            opening_map,
            opening_values,
            challenge_values,
            current_product: F::zero(),
        }
    }
}

impl<F: JoltField> SumOfProductsVisitor for WitnessAuxVisitor<'_, F> {
    type Resolved = F;
    type Acc = Vec<F>;

    fn resolve(&self, vs: &ValueSource) -> F {
        match vs {
            ValueSource::Opening(id) => {
                let idx = *self.opening_map.get(id).expect("Opening not found");
                self.opening_values[idx]
            }
            ValueSource::Challenge(idx) => self.challenge_values[*idx],
            ValueSource::Constant(val) => F::from_i128(*val),
        }
    }

    fn on_no_factors(&mut self, acc: &mut Vec<F>, coeff: F) {
        acc.push(coeff);
    }

    fn on_single_factor(&mut self, acc: &mut Vec<F>, coeff: F, factor: F) {
        acc.push(coeff * factor);
    }

    fn on_chain_start(&mut self, acc: &mut Vec<F>, f0: F, f1: F) {
        self.current_product = f0 * f1;
        acc.push(self.current_product);
    }

    fn on_chain_step(&mut self, acc: &mut Vec<F>, factor: F) {
        self.current_product *= factor;
        acc.push(self.current_product);
    }

    fn on_chain_finalize(&mut self, acc: &mut Vec<F>, coeff: F) {
        acc.push(coeff * self.current_product);
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
