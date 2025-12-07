//! Two-phase sumcheck for proving that Dory hints are well-formed
//! Proves: 0 = Σ_{i,x} eq(r_i, i) * eq(r_x, x) * C_i(x)
//! Where C_i(x) = ρ_curr(x) - ρ_prev(x)² × a(x)^{b_i} - Q_i(x) × g(x)

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        recursion_constraints::{MatrixConstraint, RowOffset},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::CommittedPolynomial,
};
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;
use std::array;

/// Parameters shared between prover and verifier
///
/// M has structure: M(offset_bits, constraint_index_bits, x_bits)
/// - Phase 1 binds x_bits (4 rounds)
/// - Phase 2 binds constraint_index_bits (num_constraint_index_vars rounds)
/// - offset_bits (2 bits) remain unbound → 4 final openings
#[derive(Clone)]
pub struct RecursionSumcheckParams {
    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraint index variables - ceil(log2(num_constraints))
    pub num_constraint_index_vars: usize,

    /// Number of constraints (actual, no padding)
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,

    /// Reference to recursion polynomial
    pub polynomial: CommittedPolynomial,
}

impl RecursionSumcheckParams {
    pub fn new(
        num_constraint_index_vars: usize,
        num_constraints: usize,
        polynomial: CommittedPolynomial,
    ) -> Self {
        Self {
            num_constraint_vars: 4, // Fixed for Fq12
            num_constraint_index_vars,
            num_constraints,
            sumcheck_id: SumcheckId::RecursionZeroCheck,
            polynomial,
        }
    }

    /// Total sumcheck rounds: x_vars + constraint_index_vars
    /// Note: offset bits (2) are NOT bound during sumcheck
    pub fn num_rounds(&self) -> usize {
        self.num_constraint_vars + self.num_constraint_index_vars
    }

    /// Construct opening point for M evaluation at (offset, r_i, r_x)
    fn construct_opening_point(
        &self,
        offset: RowOffset,
        i_challenges: &[<Fq as JoltField>::Challenge],
        x_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Vec<<Fq as JoltField>::Challenge> {
        let bits = offset.to_bits();
        let mut offset_bits: Vec<<Fq as JoltField>::Challenge> = bits
            .into_iter()
            .map(|bit| <Fq as JoltField>::Challenge::from(bit))
            .collect();
        offset_bits.reverse();

        let mut reversed_x = x_challenges.to_vec();
        reversed_x.reverse();
        let mut reversed_i = i_challenges.to_vec();
        reversed_i.reverse();

        [&offset_bits[..], &reversed_i[..], &reversed_x[..]].concat()
    }

    pub fn get_opening_point<const E: crate::poly::opening_proof::Endianness>(
        &self,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<E, Fq> {
        OpeningPoint::new(sumcheck_challenges.to_vec())
    }
}

/// Prover for recursion zero-check sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct RecursionSumcheckProver {
    /// Materialized M(s, x) as a multilinear polynomial
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub m_poly: MultilinearPolynomial<Fq>,

    /// g(x) polynomial for constraint evaluation
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub g_poly: MultilinearPolynomial<Fq>,

    /// Constraint bits for direct evaluation
    pub constraint_bits: Vec<bool>,

    /// Equality polynomial for constraint variables x (phase 1)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_x: MultilinearPolynomial<Fq>,

    /// Equality polynomial for constraint indices i (phase 2)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_i: MultilinearPolynomial<Fq>,

    /// Random challenge for eq(r, x)
    pub r_x: Vec<<Fq as JoltField>::Challenge>,

    /// Random challenge for eq(r', i)
    pub r_i: Vec<<Fq as JoltField>::Challenge>,

    /// Scalar from phase 1 completion: eq(r', x_bound)
    pub eq_r_x: Fq,

    /// Current round number
    pub round: usize,

    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: RecursionSumcheckParams,

    /// Optional: public exponent MLE for cross-checks
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub exponent_mle: DensePolynomial<Fq>,
}

impl RecursionSumcheckProver {
    pub fn gen<T: Transcript>(
        params: RecursionSumcheckParams,
        constraint_system: &super::recursion_constraints::ConstraintSystem,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let r_i: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_index_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let m_poly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_x));
        let eq_i = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_i));

        let mut constraint_bits = vec![false; params.num_constraints];
        for constraint in &constraint_system.constraints {
            constraint_bits[constraint.constraint_index] = constraint.bit;
        }

        Self {
            m_poly,
            g_poly: MultilinearPolynomial::LargeScalars(constraint_system.g_poly.clone()),
            constraint_bits,
            eq_x,
            eq_i,
            r_x,
            r_i,
            eq_r_x: Fq::one(),
            round: 0,
            params,
            exponent_mle: constraint_system.exponent_mle.clone(),
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for RecursionSumcheckProver {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars + self.params.num_constraint_index_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    #[tracing::instrument(skip_all, name = "RecursionSumcheck::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        if round < self.params.num_constraint_vars {
            // Phase 1: Sum over constraint variables (x)
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Sum over constraint index variables (i)
            self.compute_phase2_message(round - self.params.num_constraint_vars, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RecursionSumcheck::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        if round < self.params.num_constraint_vars {
            // Phase 1: Bind constraint variable x (low-order bits in M)
            self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

            if round == self.params.num_constraint_vars - 1 {
                self.eq_r_x = self.eq_x.get_bound_coeff(0);
                debug_assert_eq!(
                    self.eq_x.len(),
                    1,
                    "eq_x should be fully bound after Phase 1"
                );
            }
        } else {
            // Phase 2: Bind constraint index variable i
            self.eq_i.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let base_val = self.m_poly.get_bound_coeff(RowOffset::Base as usize);
        let rho_prev_val = self.m_poly.get_bound_coeff(RowOffset::RhoPrev as usize);
        let rho_curr_val = self.m_poly.get_bound_coeff(RowOffset::RhoCurr as usize);
        let quotient_val = self.m_poly.get_bound_coeff(RowOffset::Quotient as usize);

        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        for (offset, value) in [
            (RowOffset::Base, base_val),
            (RowOffset::RhoPrev, rho_prev_val),
            (RowOffset::RhoCurr, rho_curr_val),
            (RowOffset::Quotient, quotient_val),
        ] {
            let bits = offset.to_bits();
            let mut offset_bits: Vec<<Fq as JoltField>::Challenge> = bits
                .into_iter()
                .map(|bit| <Fq as JoltField>::Challenge::from(bit))
                .collect();
            offset_bits.reverse();

            let mut reversed_x = x_challenges.to_vec();
            reversed_x.reverse();
            let mut reversed_i = i_challenges.to_vec();
            reversed_i.reverse();
            let opening_point = [&offset_bits[..], &reversed_i[..], &reversed_x[..]].concat();

            accumulator.append_dense(
                transcript,
                self.params.polynomial,
                self.params.sumcheck_id,
                opening_point,
                value,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl RecursionSumcheckProver {
    fn compute_phase1_message(&self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 4;
        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_x_evals = self
                    .eq_x
                    .sumcheck_evals_array::<{ DEGREE + 1 }>(x_idx, BindingOrder::LowToHigh);
                let g_evals = self
                    .g_poly
                    .sumcheck_evals_array::<{ DEGREE + 1 }>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [Fq::zero(); DEGREE + 1];

                // Sum over all constraint indices
                for constraint_idx in 0..self.params.num_constraints {
                    let eq_i_val = self.eq_i.get_bound_coeff(constraint_idx);

                    // For each offset row type, compute the constraint evaluation
                    // M layout: M(x_bits, constraint_index_bits, offset_bits)
                    let num_i_bits = self.params.num_constraint_index_vars;

                    // Helper to get M evaluations for a specific offset and constraint
                    // M polynomial layout: M(x_bits(4), constraint_index_bits, offset_bits(2))
                    let get_m_evals = |offset: RowOffset| -> [Fq; DEGREE + 1] {
                        let m_idx = ((offset as usize) << (num_i_bits + num_x_remaining - 1))  // offset bits (MSB)
                                  | (constraint_idx << (num_x_remaining - 1))                   // constraint bits
                                  | x_idx; // remaining x bits (LSB)
                        self.m_poly
                            .sumcheck_evals_array::<{ DEGREE + 1 }>(m_idx, BindingOrder::LowToHigh)
                    };

                    let base_evals = get_m_evals(RowOffset::Base);
                    let rho_prev_evals = get_m_evals(RowOffset::RhoPrev);
                    let rho_curr_evals = get_m_evals(RowOffset::RhoCurr);
                    let quotient_evals = get_m_evals(RowOffset::Quotient);

                    // Get the bit for this constraint
                    let bit = if constraint_idx < self.params.num_constraints {
                        self.constraint_bits[constraint_idx]
                    } else {
                        false // padding
                    };
                    let bit_f = if bit { Fq::one() } else { Fq::zero() };

                    // Compute constraint at each evaluation point
                    for t in 0..=DEGREE {
                        // base^{b_i} = 1 + (base - 1) * b_i
                        let base_power = Fq::one() + (base_evals[t] - Fq::one()) * bit_f;
                        let constraint_val = rho_curr_evals[t]
                            - rho_prev_evals[t] * rho_prev_evals[t] * base_power
                            - quotient_evals[t] * g_evals[t];

                        x_evals[t] += eq_x_evals[t] * eq_i_val * constraint_val;
                    }
                }
                x_evals
            })
            .reduce(
                || [Fq::zero(); DEGREE + 1],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            );

        // Use all evaluations since we computed them anyway
        UniPoly::from_evals(&total_evals)
    }

    fn compute_phase2_message(&self, _phase2_round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 4;

        let g_val = self.g_poly.get_bound_coeff(0);

        let num_i_remaining = self.eq_i.get_num_vars();
        let i_half = 1 << (num_i_remaining - 1);

        let total_evals = (0..i_half)
            .into_par_iter()
            .map(|i| {
                // Get eq_i evaluations at this index
                let eq_i_evals = self
                    .eq_i
                    .sumcheck_evals_array::<{ DEGREE + 1 }>(i, BindingOrder::LowToHigh);

                // For each offset, we need to get M evaluations
                // The index into m_poly is: (i << 2) | offset
                let get_m_evals = |offset: RowOffset| -> [Fq; DEGREE + 1] {
                    let m_idx = (i << 2) | (offset as usize);
                    self.m_poly
                        .sumcheck_evals_array::<{ DEGREE + 1 }>(m_idx, BindingOrder::LowToHigh)
                };

                let base_evals = get_m_evals(RowOffset::Base);
                let rho_prev_evals = get_m_evals(RowOffset::RhoPrev);
                let rho_curr_evals = get_m_evals(RowOffset::RhoCurr);
                let quotient_evals = get_m_evals(RowOffset::Quotient);

                let mut i_evals = [Fq::zero(); DEGREE + 1];

                for t in 0..=DEGREE {
                    // For phase 2, we need to evaluate the exponent MLE at the current point
                    // The current index position is i (offset by t for sumcheck evaluation)
                    // We'll compute it by checking constraint_bits directly
                    let actual_constraint_idx = (i << 1) | (t & 1);
                    let bit = if actual_constraint_idx < self.params.num_constraints {
                        self.constraint_bits[actual_constraint_idx]
                    } else {
                        false
                    };
                    let bit_f = if bit { Fq::one() } else { Fq::zero() };

                    // base^{b_i} = 1 + (base - 1) * b_i
                    let base_power = Fq::one() + (base_evals[t] - Fq::one()) * bit_f;
                    let constraint_val = rho_curr_evals[t]
                        - rho_prev_evals[t] * rho_prev_evals[t] * base_power
                        - quotient_evals[t] * g_val;

                    // In Phase 2, we accumulate: eq_r_x * eq(r_i, i) * C_i(r_x)
                    i_evals[t] = self.eq_r_x * eq_i_evals[t] * constraint_val;
                }
                i_evals
            })
            .reduce(
                || [Fq::zero(); DEGREE + 1],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            );
        UniPoly::from_evals(&total_evals)
    }
}

/// Verifier for recursion zero-check sumcheck
///
/// After sumcheck completes, M has 2 unbound offset bits, giving 4 openings:
/// - M(00, r_i, r_x) = base
/// - M(01, r_i, r_x) = rho_prev
/// - M(10, r_i, r_x) = rho_curr
/// - M(11, r_i, r_x) = quotient
///
/// The verifier computes the constraint:
/// C(r_i, r_x) = rho_curr - rho_prev² × base^{b_i} - quotient × g(r_x)
/// where b_i comes from the exponent MLE
pub struct RecursionSumcheckVerifier {
    pub params: RecursionSumcheckParams,
    /// Random challenge for eq(r, x) - constraint variables
    pub r_x: Vec<<Fq as JoltField>::Challenge>,
    /// Random challenge for eq(r', i) - constraint indices
    pub r_i: Vec<<Fq as JoltField>::Challenge>,
    /// Precomputed g(x) polynomial for constraint evaluation
    pub g_poly: DensePolynomial<Fq>,
    /// Public exponent MLE over index bits: Σ_i b_i eq(z, i)
    pub exponent_mle: DensePolynomial<Fq>,
}

impl RecursionSumcheckVerifier {
    pub fn new<T: Transcript>(
        params: RecursionSumcheckParams,
        _constraints: Vec<MatrixConstraint>,
        g_poly: DensePolynomial<Fq>,
        exponent_mle: DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let r_i: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_index_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        Self {
            params,
            r_x,
            r_i,
            g_poly,
            exponent_mle,
        }
    }

    /// Evaluate the exponent MLE at the given point
    pub fn exponent_eval_at_point(&self, eval_point: &[Fq]) -> Fq {
        self.exponent_mle.evaluate(eval_point)
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for RecursionSumcheckVerifier {
    fn degree(&self) -> usize {
        // C_i(x) = ρ_curr - ρ_prev² × a^b - Q × g has degree 3 due to ρ_prev² × a
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        let mut values = vec![];
        for _ in 0..4 {
            let (_point, value) = accumulator
                .get_committed_polynomial_opening(self.params.polynomial, self.params.sumcheck_id);
            values.push(value);
        }

        let [base_val, rho_prev_val, rho_curr_val, quotient_val] =
            values.try_into().expect("Should have exactly 4 values");

        let mut x_challenges_reversed = x_challenges.to_vec();
        x_challenges_reversed.reverse();
        let g_val = self.g_poly.evaluate(&x_challenges_reversed);

        // Get the exponent bit at r_i
        let mut i_challenges_reversed = i_challenges.to_vec();
        i_challenges_reversed.reverse();
        let r_i_fq: Vec<Fq> = i_challenges_reversed.iter().map(|c| (*c).into()).collect();
        let bit_eval = self.exponent_eval_at_point(&r_i_fq);

        // Compute constraint: C(r_i, r_x) = ρ_curr - ρ_prev² × base^{b_i} - quotient × g(r_x)
        // where base^{b_i} = 1 + (base - 1) * b_i
        let base_power = Fq::one() + (base_val - Fq::one()) * bit_eval;
        let constraint_eval =
            rho_curr_val - rho_prev_val.square() * base_power - quotient_val * g_val;

        let x_bits_msb = x_challenges.to_vec();
        let i_bits_msb = i_challenges.to_vec();

        let eq_r_x = EqPolynomial::<Fq>::mle(&self.r_x, &x_bits_msb);
        let eq_r_i = EqPolynomial::<Fq>::mle(&self.r_i, &i_bits_msb);
        let expected_claim = eq_r_x * eq_r_i * constraint_eval;

        expected_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // The prover now appends 4 dense openings for the 4 M evaluations
        // at points M(offset, r_i, r_x) for offset ∈ {0,1,2,3}

        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        // Each uses opening point: [offset_bits || i_challenges || x_challenges]
        for offset in [
            RowOffset::Base,
            RowOffset::RhoPrev,
            RowOffset::RhoCurr,
            RowOffset::Quotient,
        ] {
            let opening_point =
                self.params
                    .construct_opening_point(offset, i_challenges, x_challenges);
            accumulator.append_dense(
                transcript,
                self.params.polynomial,
                self.params.sumcheck_id, // Use same sumcheck ID for all openings
                opening_point,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{
            commitment::{
                commitment_scheme::CommitmentScheme,
                dory::{DoryCommitmentScheme, DoryGlobals},
            },
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        },
        subprotocols::{recursion_constraints::ConstraintSystem, sumcheck::BatchedSumcheck},
        transcripts::Blake2bTranscript,
    };
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_grumpkin::Projective as GrumpkinProjective;
    use rand::thread_rng;
    use serial_test::serial;

    #[test]
    #[serial]
    // #[ignore]
    fn test_dory_witness_recursion_sumcheck_hyrax_reduce_and_prove() {
        use crate::poly::commitment::commitment_scheme::RecursionExt;
        use crate::poly::commitment::hyrax::Hyrax;
        use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
        use std::collections::HashMap;

        const RATIO: usize = 1;

        // Setup using Dory commitment
        DoryGlobals::reset();
        DoryGlobals::initialize(1 << 2, 1 << 2);
        let num_vars = 4;
        let mut rng = thread_rng();

        // 1. Create Dory proof and extract witnesses
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::rand(&mut rng))
            .collect();

        let mut prover_transcript = Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint.clone()),
            &mut prover_transcript,
        );

        // Extract witnesses using witness_gen
        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
        let mut witness_transcript = Blake2bTranscript::new(b"test");
        let (_witnesses, _hints) = DoryCommitmentScheme::witness_gen(
            &proof,
            &verifier_setup,
            &mut witness_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Witness generation should succeed");

        // 2. Build constraint system from witnesses
        let mut constraint_transcript = Blake2bTranscript::new(b"test");
        let (constraint_system, _constraint_hints) = ConstraintSystem::new(
            &proof,
            &verifier_setup,
            &mut constraint_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Constraint system creation should succeed");

        // 3. Setup Hyrax and commit to constraint matrix M
        let m_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            constraint_system.matrix.evaluations.clone(),
        ));

        let hyrax_prover_setup =
            Hyrax::<RATIO, GrumpkinProjective>::setup_prover(m_poly.get_num_vars());
        let hyrax_verifier_setup =
            Hyrax::<RATIO, GrumpkinProjective>::setup_verifier(&hyrax_prover_setup);

        let (m_commitment, _) =
            Hyrax::<RATIO, GrumpkinProjective>::commit(&m_poly, &hyrax_prover_setup);

        // 3.5. Sanity check: Verify all constraints are zero over the hypercube
        constraint_system.verify_constraints_are_zero();

        // 4. Run recursion sumcheck (two-phase)
        let params = RecursionSumcheckParams::new(
            constraint_system.matrix.num_constraint_index_vars,
            constraint_system.matrix.num_constraints,
            CommittedPolynomial::DoryConstraintMatrix,
        );

        let mut sumcheck_transcript = Blake2bTranscript::new(b"recursion_sumcheck");

        let mut verifier_transcript = sumcheck_transcript.clone();

        let log_T = m_poly.get_num_vars();
        let mut prover_accumulator = ProverOpeningAccumulator::<Fq>::new(log_T);

        let mut prover = RecursionSumcheckProver::gen(
            params.clone(),
            &constraint_system,
            &mut sumcheck_transcript,
        );

        let verifier = RecursionSumcheckVerifier::new(
            params.clone(),
            constraint_system.constraints.clone(),
            constraint_system.g_poly.clone(),
            constraint_system.exponent_mle.clone(),
            &mut verifier_transcript,
        );

        let (sumcheck_proof, sumcheck_challenges) = BatchedSumcheck::prove(
            vec![&mut prover],
            &mut prover_accumulator,
            &mut sumcheck_transcript,
        );

        // 5. Reduce 4 offset claims to single claim using Hyrax
        let mut committed_polynomials = HashMap::new();
        let mut committed_hints = HashMap::new();

        committed_polynomials.insert(CommittedPolynomial::DoryConstraintMatrix, m_poly.clone());
        committed_hints.insert(CommittedPolynomial::DoryConstraintMatrix, ());

        // Use reduce_and_prove to reduce the 4 offset claims to a single claim
        let reduced_proof = prover_accumulator
            .reduce_and_prove::<Blake2bTranscript, Hyrax<RATIO, GrumpkinProjective>>(
                committed_polynomials,
                committed_hints,
                &hyrax_prover_setup,
                &mut sumcheck_transcript,
                None, // No streaming context for this test
            );

        // 6. Verify the sumcheck and reduced openings
        // Create verifier accumulator
        let mut verifier_accumulator = VerifierOpeningAccumulator::<Fq>::new(log_T);

        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &prover_accumulator.openings {
            verifier_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        let verified_challenges = BatchedSumcheck::verify(
            &sumcheck_proof,
            vec![&verifier],
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .expect("Sumcheck verification should succeed");

        assert_eq!(
            verified_challenges.len(),
            sumcheck_challenges.len(),
            "Challenge count mismatch"
        );
        for (i, (prover_challenge, verifier_challenge)) in sumcheck_challenges
            .iter()
            .zip(verified_challenges.iter())
            .enumerate()
        {
            assert_eq!(
                prover_challenge, verifier_challenge,
                "Challenge mismatch at round {}",
                i
            );
        }
        let mut commitment_map = HashMap::new();
        commitment_map.insert(
            CommittedPolynomial::DoryConstraintMatrix,
            m_commitment.clone(),
        );

        // Reduce and verify with Hyrax
        let result = verifier_accumulator
            .reduce_and_verify::<Blake2bTranscript, Hyrax<RATIO, GrumpkinProjective>>(
                &hyrax_verifier_setup,
                &mut commitment_map,
                &reduced_proof,
                &mut verifier_transcript,
            );

        result.expect("Sumcheck verification should succeed");
    }
}
