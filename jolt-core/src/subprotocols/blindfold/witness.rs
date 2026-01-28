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
    /// Expected polynomial evaluations yⱼ (witness, proven via ZK-Dory)
    /// Used for simple linear constraints.
    pub evaluations: Vec<F>,
    /// Challenge values for general constraints.
    /// Layout: [batching_coeffs..., instance0_challenges..., instance1_challenges..., ...]
    pub challenge_values: Vec<F>,
    /// Opening values for general constraints.
    /// Maps OpeningId to its evaluation value.
    pub opening_values: Vec<F>,
    /// Whether this uses the general constraint format.
    pub is_general_constraint: bool,
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
            is_general_constraint: false,
        }
    }

    /// Create a general constraint witness with challenge and opening values.
    pub fn new_general(challenge_values: Vec<F>, opening_values: Vec<F>) -> Self {
        Self {
            batching_coefficients: Vec::new(),
            evaluations: Vec::new(),
            challenge_values,
            opening_values,
            is_general_constraint: true,
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

        // Count total batching coefficients (only for simple constraints, not general ones)
        let total_batching_coeffs: usize = r1cs
            .stage_configs
            .iter()
            .filter_map(|s| s.final_output.as_ref())
            .filter(|fo| fo.constraint.is_none())
            .map(|fo| fo.num_evaluations)
            .sum();

        // Count total output constraint challenge values (for general constraints)
        let total_constraint_challenges: usize = r1cs
            .stage_configs
            .iter()
            .filter_map(|s| s.final_output.as_ref())
            .filter_map(|fo| fo.constraint.as_ref())
            .map(|c| c.num_challenges)
            .sum();

        // Count total input constraint challenge values
        let total_input_constraint_challenges: usize = r1cs
            .stage_configs
            .iter()
            .filter_map(|s| s.initial_input.as_ref())
            .filter_map(|ii| ii.constraint.as_ref())
            .map(|c| c.num_challenges)
            .sum();

        // Count total extra constraint challenge values
        let total_extra_constraint_challenges: usize = r1cs
            .extra_constraints
            .iter()
            .map(|c| c.num_challenges)
            .sum();

        // Public input layout:
        // - Indices 1..=total_rounds are sumcheck challenges
        // - Indices total_rounds+1..=total_rounds+num_chains are initial_claims
        // - Indices after that are batching_coefficients (simple constraints)
        // - Indices after that are output constraint_challenge_values
        // - Indices after that are input constraint_challenge_values
        // - Indices after that are extra constraint_challenge_values
        let challenge_start = 1;
        let initial_claims_start = total_rounds + 1;
        let batching_coeffs_start = initial_claims_start + num_chains;
        let constraint_challenges_start = batching_coeffs_start + total_batching_coeffs;
        let input_constraint_challenges_start =
            constraint_challenges_start + total_constraint_challenges;
        let extra_constraint_challenges_start =
            input_constraint_challenges_start + total_input_constraint_challenges;
        let witness_start = extra_constraint_challenges_start + total_extra_constraint_challenges;

        // Assign initial claims
        for (i, claim) in self.initial_claims.iter().enumerate() {
            z[initial_claims_start + i] = *claim;
        }

        let mut challenge_idx = challenge_start;
        let mut batching_coeff_idx = batching_coeffs_start;
        let mut constraint_challenge_idx = constraint_challenges_start;
        let mut input_constraint_challenge_idx = input_constraint_challenges_start;
        let mut extra_constraint_challenge_idx = extra_constraint_challenges_start;
        let mut witness_idx = witness_start;

        // Track which openings have been assigned (mirrors R1CS's global_opening_vars)
        let mut assigned_openings: HashSet<OpeningId> = HashSet::new();

        for (stage_idx, stage_witness) in self.stages.iter().enumerate() {
            let config = &r1cs.stage_configs[stage_idx];
            assert_eq!(
                stage_witness.rounds.len(),
                config.num_rounds,
                "Stage {stage_idx} has wrong number of rounds"
            );

            // Assign initial input witness if present (before processing rounds)
            if let Some(ref ii_config) = config.initial_input {
                if let Some(constraint) = &ii_config.constraint {
                    let ii_witness = stage_witness.initial_input.as_ref();

                    let num_challenges = constraint.num_challenges;
                    let num_aux_vars = Self::estimate_aux_var_count(constraint);

                    if let Some(iw) = ii_witness {
                        // Assign opening values only for NEW openings (matching R1CS allocation)
                        debug_assert_eq!(
                            iw.opening_values.len(),
                            constraint.required_openings.len(),
                            "Input opening values count mismatch"
                        );
                        for (opening_id, val) in
                            constraint.required_openings.iter().zip(&iw.opening_values)
                        {
                            if !assigned_openings.contains(opening_id) {
                                z[witness_idx] = *val;
                                witness_idx += 1;
                                assigned_openings.insert(*opening_id);
                            }
                        }

                        // Assign challenge values (public inputs)
                        debug_assert_eq!(
                            iw.challenge_values.len(),
                            num_challenges,
                            "Input challenge values count mismatch"
                        );
                        for val in &iw.challenge_values {
                            z[input_constraint_challenge_idx] = *val;
                            input_constraint_challenge_idx += 1;
                        }

                        // Compute and assign aux vars for intermediate products
                        let aux_values = Self::compute_aux_vars(
                            constraint,
                            &iw.opening_values,
                            &iw.challenge_values,
                        );
                        debug_assert_eq!(
                            aux_values.len(),
                            num_aux_vars,
                            "Input aux var count mismatch"
                        );
                        for val in aux_values {
                            z[witness_idx] = val;
                            witness_idx += 1;
                        }
                    } else {
                        // No witness provided - skip past allocated variables
                        // Count only NEW openings (matching R1CS allocation)
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
                        input_constraint_challenge_idx += num_challenges;
                    }
                }
            }

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

            // Assign final output witness if present
            if let Some(ref fo_config) = config.final_output {
                let fo_witness = stage_witness.final_output.as_ref();

                if let Some(constraint) = &fo_config.constraint {
                    // General constraint path
                    // R1CS allocates: opening_vars (witness) + aux_vars (witness)
                    // Challenge vars are public inputs (assigned separately)
                    let num_challenges = constraint.num_challenges;
                    let num_aux_vars = Self::estimate_aux_var_count(constraint);

                    if let Some(fw) = fo_witness {
                        // Assign opening values only for NEW openings (matching R1CS allocation)
                        debug_assert_eq!(
                            fw.opening_values.len(),
                            constraint.required_openings.len(),
                            "Opening values count mismatch"
                        );
                        for (opening_id, val) in
                            constraint.required_openings.iter().zip(&fw.opening_values)
                        {
                            if !assigned_openings.contains(opening_id) {
                                z[witness_idx] = *val;
                                witness_idx += 1;
                                assigned_openings.insert(*opening_id);
                            }
                        }

                        // Assign challenge values (public inputs)
                        debug_assert_eq!(
                            fw.challenge_values.len(),
                            num_challenges,
                            "Challenge values count mismatch"
                        );
                        for val in &fw.challenge_values {
                            z[constraint_challenge_idx] = *val;
                            constraint_challenge_idx += 1;
                        }

                        // Compute and assign aux vars for intermediate products
                        let aux_values = Self::compute_aux_vars(
                            constraint,
                            &fw.opening_values,
                            &fw.challenge_values,
                        );
                        debug_assert_eq!(
                            aux_values.len(),
                            num_aux_vars,
                            "Aux var count mismatch: computed {} but expected {}",
                            aux_values.len(),
                            num_aux_vars
                        );
                        for val in aux_values {
                            z[witness_idx] = val;
                            witness_idx += 1;
                        }
                    } else {
                        // No witness provided - skip past allocated variables
                        // Count only NEW openings (matching R1CS allocation)
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
                        constraint_challenge_idx += num_challenges;
                    }
                } else {
                    // Simple linear constraint path
                    let fw = fo_witness
                        .expect("Stage has final_output config but witness has no final_output");

                    assert_eq!(
                        fw.batching_coefficients.len(),
                        fo_config.num_evaluations,
                        "Wrong number of batching coefficients"
                    );
                    assert_eq!(
                        fw.evaluations.len(),
                        fo_config.num_evaluations,
                        "Wrong number of evaluations"
                    );

                    // Assign batching coefficients (public inputs)
                    for coeff in &fw.batching_coefficients {
                        z[batching_coeff_idx] = *coeff;
                        batching_coeff_idx += 1;
                    }

                    // Assign evaluations (witness variables)
                    for eval in &fw.evaluations {
                        z[witness_idx] = *eval;
                        witness_idx += 1;
                    }

                    // Assign accumulator variables for multi-evaluation constraints
                    let n = fo_config.num_evaluations;
                    if n > 1 {
                        // Compute accumulator values: acc_j = Σ_{i=0}^j αᵢ · yᵢ
                        let mut acc = F::zero();
                        for j in 0..n - 1 {
                            acc += fw.batching_coefficients[j] * fw.evaluations[j];
                            z[witness_idx] = acc;
                            witness_idx += 1;
                        }
                    }
                }
            }
        }

        // Assign extra constraints appended after all stages
        for (extra_idx, constraint) in r1cs.extra_constraints.iter().enumerate() {
            let extra_witness = self.extra_constraints.get(extra_idx);
            let num_challenges = constraint.num_challenges;
            let num_aux_vars = Self::estimate_aux_var_count(constraint);

            if let Some(witness) = extra_witness {
                debug_assert_eq!(
                    witness.opening_values.len(),
                    constraint.required_openings.len(),
                    "Extra opening values count mismatch"
                );

                // Assign opening values only for NEW openings (matching R1CS allocation)
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

                // Assign output value
                z[witness_idx] = witness.output_value;
                witness_idx += 1;

                // Assign challenge values (public inputs)
                debug_assert_eq!(
                    witness.challenge_values.len(),
                    num_challenges,
                    "Extra challenge values count mismatch"
                );
                for val in &witness.challenge_values {
                    z[extra_constraint_challenge_idx] = *val;
                    extra_constraint_challenge_idx += 1;
                }

                // Compute and assign aux vars for intermediate products
                let aux_values = Self::compute_aux_vars(
                    constraint,
                    &witness.opening_values,
                    &witness.challenge_values,
                );
                debug_assert_eq!(
                    aux_values.len(),
                    num_aux_vars,
                    "Extra aux var count mismatch"
                );
                for val in aux_values {
                    z[witness_idx] = val;
                    witness_idx += 1;
                }

                // Assign blinding
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
                extra_constraint_challenge_idx += num_challenges;
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

    /// Estimate the number of auxiliary variables needed for a general constraint.
    ///
    /// The R1CS builder allocates aux vars for intermediate products in sum-of-products.
    /// This needs to match the allocation in `add_sum_of_products_constraint`.
    fn estimate_aux_var_count(constraint: &OutputClaimConstraint) -> usize {
        let mut count = 0;
        for term in &constraint.terms {
            if term.factors.is_empty() {
                // Single coefficient: 1 aux var
                count += 1;
            } else if term.factors.len() == 1 {
                // Single factor: 1 aux var
                count += 1;
            } else {
                // Multiple factors: (n-1) aux vars for chain multiplication + 1 for final
                count += term.factors.len();
            }
        }
        // Plus 1 for the final sum constraint (if more than 1 term)
        // Actually the final sum uses a linear combination, not aux vars
        count
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
