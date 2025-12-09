//! Square-and-multiply sumcheck for proving GT exponentiation constraints
//! Proves: 0 = Σ_x eq(r_x, x) * Σ_i γ^i * C_i(x)
//! Where C_i(x) = ρ_{i+1}(x) - ρ_i(x)² × a(x)^{b_i} - Q_i(x) × g(x)
//!
//! This is Phase 1 of the new two-phase recursion protocol.
//! Output: Virtual polynomial claims for each polynomial in each constraint

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        recursion_constraints::ConstraintType, sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Helper to append all virtual claims for a constraint
fn append_constraint_virtual_claims<T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<Fq>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
    base_claim: Fq,
    rho_prev_claim: Fq,
    rho_curr_claim: Fq,
    quotient_claim: Fq,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionBase(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        base_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoPrev(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        rho_prev_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoCurr(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        rho_curr_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionQuotient(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        quotient_claim,
    );
}

/// Helper to retrieve all virtual claims for a constraint
fn get_constraint_virtual_claims(
    accumulator: &VerifierOpeningAccumulator<Fq>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (Fq, Fq, Fq, Fq) {
    let (_, base_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionBase(constraint_idx),
        sumcheck_id,
    );
    let (_, rho_prev_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionRhoPrev(constraint_idx),
        sumcheck_id,
    );
    let (_, rho_curr_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionRhoCurr(constraint_idx),
        sumcheck_id,
    );
    let (_, quotient_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionQuotient(constraint_idx),
        sumcheck_id,
    );

    (base_claim, rho_prev_claim, rho_curr_claim, quotient_claim)
}

/// Helper to append virtual opening points for a constraint (verifier side)
fn append_constraint_virtual_openings<T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<Fq>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionBase(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoPrev(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionRhoCurr(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionQuotient(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
}

/// Individual polynomial data for a single constraint
#[derive(Clone)]
pub struct ConstraintPolynomials {
    pub base: Vec<Fq>,
    pub rho_prev: Vec<Fq>,
    pub rho_curr: Vec<Fq>,
    pub quotient: Vec<Fq>,
    pub bit: bool,
    pub constraint_index: usize,
}

/// Parameters for square-and-multiply sumcheck
#[derive(Clone)]
pub struct SquareAndMultiplyParams {
    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraints
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl SquareAndMultiplyParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 4, // Fixed for Fq12
            num_constraints,
            sumcheck_id: SumcheckId::SquareAndMultiply,
        }
    }
}

/// Prover for square-and-multiply sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct SquareAndMultiplyProver {
    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: SquareAndMultiplyParams,

    /// Constraint bits for base^{b_i} evaluation
    pub constraint_bits: Vec<bool>,

    /// Global constraint indices for each constraint
    pub constraint_indices: Vec<usize>,

    /// g(x) polynomial for constraint evaluation
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub g_poly: MultilinearPolynomial<Fq>,

    /// Equality polynomial for constraint variables x
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_x: MultilinearPolynomial<Fq>,

    /// Random challenge for eq(r_x, x)
    pub r_x: Vec<<Fq as JoltField>::Challenge>,

    /// Gamma coefficient for batching constraints
    pub gamma: Fq,

    /// Base polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub base_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Rho_prev polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rho_prev_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Rho_curr polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rho_curr_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Quotient polynomials as multilinear
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub quotient_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Individual claims for each constraint (not batched)
    pub base_claims: Vec<Fq>,
    pub rho_prev_claims: Vec<Fq>,
    pub rho_curr_claims: Vec<Fq>,
    pub quotient_claims: Vec<Fq>,

    /// Current round
    pub round: usize,
}

impl SquareAndMultiplyProver {
    pub fn new<T: Transcript>(
        params: SquareAndMultiplyParams,
        constraint_polys: Vec<ConstraintPolynomials>,
        g_poly: DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<Fq>();

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_x));
        let mut constraint_bits = Vec::new();
        let mut constraint_indices = Vec::new();
        let mut base_mlpoly = Vec::new();
        let mut rho_prev_mlpoly = Vec::new();
        let mut rho_curr_mlpoly = Vec::new();
        let mut quotient_mlpoly = Vec::new();

        for poly in constraint_polys {
            constraint_bits.push(poly.bit);
            constraint_indices.push(poly.constraint_index);
            base_mlpoly.push(MultilinearPolynomial::from(poly.base));
            rho_prev_mlpoly.push(MultilinearPolynomial::from(poly.rho_prev));
            rho_curr_mlpoly.push(MultilinearPolynomial::from(poly.rho_curr));
            quotient_mlpoly.push(MultilinearPolynomial::from(poly.quotient));
        }

        Self {
            params,
            constraint_bits,
            constraint_indices,
            g_poly: MultilinearPolynomial::LargeScalars(g_poly),
            eq_x,
            r_x,
            gamma: gamma.into(),
            base_mlpoly,
            rho_prev_mlpoly,
            rho_curr_mlpoly,
            quotient_mlpoly,
            base_claims: vec![],
            rho_prev_claims: vec![],
            rho_curr_claims: vec![],
            quotient_claims: vec![],
            round: 0,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for SquareAndMultiplyProver {
    fn degree(&self) -> usize {
        4 // Degree from constraint: rho_prev^2 * base
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    #[tracing::instrument(skip_all, name = "SquareAndMultiply::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 4;
        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_x_evals = self
                    .eq_x
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                let g_evals = self
                    .g_poly
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [Fq::zero(); DEGREE];
                let mut gamma_power = self.gamma;

                for i in 0..self.constraint_bits.len() {
                    let base_evals_hint = self.base_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let rho_prev_evals_hint = self.rho_prev_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let rho_curr_evals_hint = self.rho_curr_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let quotient_evals_hint = self.quotient_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        // base^{b_i}: if bit is true, use base; else use 1
                        let base_power = if self.constraint_bits[i] {
                            base_evals_hint[t]
                        } else {
                            Fq::one()
                        };
                        let constraint_val = rho_curr_evals_hint[t]
                            - rho_prev_evals_hint[t] * rho_prev_evals_hint[t] * base_power
                            - quotient_evals_hint[t] * g_evals[t];

                        x_evals[t] += eq_x_evals[t] * gamma_power * constraint_val;
                    }

                    gamma_power *= self.gamma;
                }
                x_evals
            })
            .reduce(
                || [Fq::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &total_evals)
    }

    #[tracing::instrument(skip_all, name = "SquareAndMultiply::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

        for poly in &mut self.base_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.rho_prev_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.rho_curr_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.quotient_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;

        if self.round == self.params.num_constraint_vars {
            self.base_claims.clear();
            self.rho_prev_claims.clear();
            self.rho_curr_claims.clear();
            self.quotient_claims.clear();

            for i in 0..self.constraint_bits.len() {
                self.base_claims
                    .push(self.base_mlpoly[i].get_bound_coeff(0));
                self.rho_prev_claims
                    .push(self.rho_prev_mlpoly[i].get_bound_coeff(0));
                self.rho_curr_claims
                    .push(self.rho_curr_mlpoly[i].get_bound_coeff(0));
                self.quotient_claims
                    .push(self.quotient_mlpoly[i].get_bound_coeff(0));
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        for i in 0..self.constraint_bits.len() {
            append_constraint_virtual_claims(
                accumulator,
                transcript,
                self.constraint_indices[i], // Use global constraint index
                self.params.sumcheck_id,
                &opening_point,
                self.base_claims[i],
                self.rho_prev_claims[i],
                self.rho_curr_claims[i],
                self.quotient_claims[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for square-and-multiply sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct SquareAndMultiplyVerifier {
    pub params: SquareAndMultiplyParams,
    pub r_x: Vec<<Fq as JoltField>::Challenge>,
    pub gamma: Fq,
    pub num_constraints: usize,
    pub constraint_bits: Vec<bool>,
    pub constraint_indices: Vec<usize>,
}

impl SquareAndMultiplyVerifier {
    pub fn new<T: Transcript>(
        params: SquareAndMultiplyParams,
        constraint_bits: Vec<bool>,
        constraint_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<Fq>();
        let num_constraints = params.num_constraints;

        Self {
            params,
            r_x,
            gamma: gamma.into(),
            num_constraints,
            constraint_bits,
            constraint_indices,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for SquareAndMultiplyVerifier {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        use crate::poly::eq_poly::EqPolynomial;

        let r_x_fq: Vec<Fq> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_star_fq: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_eval = EqPolynomial::mle(&r_x_fq, &r_star_fq);
        let g_eval = {
            use crate::poly::dense_mlpoly::DensePolynomial;
            use crate::poly::multilinear_polynomial::MultilinearPolynomial;
            use jolt_optimizations::get_g_mle;

            let g_poly =
                MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(get_g_mle()));
            g_poly.evaluate_dot_product(&r_star_fq)
        };

        let mut total = Fq::zero();
        let mut gamma_power = self.gamma;

        for i in 0..self.num_constraints {
            let (base_claim, rho_prev_claim, rho_curr_claim, quotient_claim) =
                get_constraint_virtual_claims(
                    accumulator,
                    self.constraint_indices[i],
                    self.params.sumcheck_id,
                );

            // Compute the constraint: ρ_{i+1} - ρ_i^2 * base^{b_i} - q_i * g(x)
            let base_power = if self.constraint_bits[i] {
                base_claim
            } else {
                Fq::one()
            };

            let constraint_value = rho_curr_claim
                - rho_prev_claim * rho_prev_claim * base_power
                - quotient_claim * g_eval;

            total += gamma_power * constraint_value;
            gamma_power *= self.gamma;
        }

        eq_eval * total
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        for i in 0..self.num_constraints {
            append_constraint_virtual_openings(
                accumulator,
                transcript,
                self.constraint_indices[i],
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{
            commitment::commitment_scheme::CommitmentScheme,
            opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        },
        subprotocols::{recursion_constraints::ConstraintSystem, sumcheck::BatchedSumcheck},
        transcripts::Blake2bTranscript,
        zkvm::witness::CommittedPolynomial,
    };
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_two_phase_recursion_protocol_e2e() {
        use crate::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
        use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
        use ark_bn254::Fr;
        use ark_ff::UniformRand;
        use rand::thread_rng;

        // Initialize Dory globals
        DoryGlobals::reset();
        DoryGlobals::initialize(1 << 2, 1 << 2);

        let num_vars = 4; // For Fq12
        let mut rng = thread_rng();

        // Setup Dory prover and verifier
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Create a random polynomial to commit to
        let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Generate random evaluation point
        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Generate Dory proof
        let mut dory_transcript = Blake2bTranscript::new(b"recursion_test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut dory_transcript,
        );

        // Compute evaluation
        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);

        // Extract constraint system from Dory proof
        let mut extract_transcript = Blake2bTranscript::new(b"recursion_test");
        let (constraint_system, _hints) = ConstraintSystem::new(
            &proof,
            &verifier_setup,
            &mut extract_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Constraint system creation should succeed");

        let num_constraints = constraint_system.num_constraints();

        // Debug: Print constraint system info
        let num_gt_exp = constraint_system
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::GtExp { .. }))
            .count();
        let num_gt_mul = constraint_system
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::GtMul))
            .count();


        // Create transcripts for the two-phase protocol
        let mut prover_transcript = Blake2bTranscript::new(b"two_phase_recursion");
        let mut verifier_transcript = Blake2bTranscript::new(b"two_phase_recursion");

        // Extract g polynomial
        let g_poly = constraint_system.g_poly.clone();

        // ============ PHASE 1: Constraint Sumchecks ============

        // Extract GT exp constraints for square-and-multiply
        let gt_exp_constraint_polys = constraint_system.extract_constraint_polynomials();

        // Create prover accumulator for Phase 1
        // log_T should match the constraint system size
        let log_T = constraint_system.num_vars();
        let mut prover_accumulator = ProverOpeningAccumulator::<Fq>::new(log_T);

        // Create provers based on what constraints we have
        let mut gamma = Fq::zero();

        // Prepare square-and-multiply prover if we have GT exp constraints
        let mut sq_mul_prover = if !gt_exp_constraint_polys.is_empty() {
            let params_sq_mul = SquareAndMultiplyParams::new(gt_exp_constraint_polys.len());
            let prover = SquareAndMultiplyProver::new(
                params_sq_mul.clone(),
                gt_exp_constraint_polys,
                g_poly.clone(),
                &mut prover_transcript,
            );
            gamma = prover.gamma; // Save gamma for later
            Some(prover)
        } else {
            None
        };

        // Prepare GT mul prover if we have GT mul constraints
        let gt_mul_constraints = constraint_system.extract_gt_mul_constraints();
        let mut gt_mul_prover = if !gt_mul_constraints.is_empty() {
            use crate::subprotocols::gt_mul::{
                GtMulConstraintPolynomials, GtMulParams, GtMulProver,
            };

            let mut gt_mul_polys = Vec::new();
            for (idx, lhs, rhs, result, quotient) in gt_mul_constraints {
                gt_mul_polys.push(GtMulConstraintPolynomials {
                    lhs,
                    rhs,
                    result,
                    quotient,
                    constraint_index: idx,
                });
            }

            let params_gt_mul = GtMulParams::new(gt_mul_polys.len());
            let prover = GtMulProver::<Blake2bTranscript>::new(
                params_gt_mul,
                gt_mul_polys,
                g_poly.clone(),
                &mut prover_transcript,
            );
            if gamma == Fq::zero() {
                gamma = prover.gamma; // Use GT mul gamma if no square-and-multiply
            }
            Some(prover)
        } else {
            None
        };

        // Run Phase 1 sumcheck with all provers
        let mut phase1_instances: Vec<&mut dyn SumcheckInstanceProver<Fq, Blake2bTranscript>> =
            Vec::new();

        if let Some(ref mut prover) = sq_mul_prover {
            phase1_instances.push(prover);
        }

        if let Some(ref mut prover) = gt_mul_prover {
            phase1_instances.push(prover);
        }

        let (phase1_proof, r_phase1) = BatchedSumcheck::prove(
            phase1_instances,
            &mut prover_accumulator,
            &mut prover_transcript,
        );

        // ============ PHASE 2: Virtualization Sumcheck ============

        // ============ VERIFICATION ============

        // Create verifier accumulator with same log_T
        let mut verifier_accumulator = VerifierOpeningAccumulator::<Fq>::new(log_T);

        // Populate virtual polynomial claims in verifier accumulator from prover
        // In a real proof, these would come from proof.opening_claims
        for (key, (_, claim)) in &prover_accumulator.openings {
            verifier_accumulator
                .openings
                .insert(key.clone(), (OpeningPoint::default(), *claim));
        }

        // Count GT exp and GT mul constraints
        let num_gt_exp = constraint_system
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::GtExp { .. }))
            .count();
        let num_gt_mul = constraint_system
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::GtMul))
            .count();

        // Create verifiers based on what constraints we have
        let mut phase1_ver_instances: Vec<
            Box<dyn SumcheckInstanceVerifier<Fq, Blake2bTranscript>>,
        > = Vec::new();

        // Add square-and-multiply verifier if we have GT exp constraints
        if num_gt_exp > 0 {
            // Get constraint bits and indices for verifier (only GT exp constraints have bits)
            let (constraint_bits, constraint_indices): (Vec<bool>, Vec<usize>) = constraint_system
                .constraints
                .iter()
                .filter_map(|c| match &c.constraint_type {
                    ConstraintType::GtExp { bit } => Some((*bit, c.constraint_index)),
                    ConstraintType::GtMul => None,
                })
                .unzip();

            let params_sq_mul = SquareAndMultiplyParams::new(num_gt_exp);
            let verifier = SquareAndMultiplyVerifier::new(
                params_sq_mul,
                constraint_bits,
                constraint_indices,
                &mut verifier_transcript,
            );
            phase1_ver_instances.push(Box::new(verifier));
        }

        // Add GT mul verifier if we have GT mul constraints
        if num_gt_mul > 0 {
            use crate::subprotocols::gt_mul::{GtMulParams, GtMulVerifier};

            // Get constraint indices for GT mul constraints
            let constraint_indices: Vec<usize> = constraint_system
                .constraints
                .iter()
                .filter_map(|c| match &c.constraint_type {
                    ConstraintType::GtMul => Some(c.constraint_index),
                    ConstraintType::GtExp { .. } => None,
                })
                .collect();

            let params_gt_mul = GtMulParams::new(num_gt_mul);
            let verifier =
                GtMulVerifier::new(params_gt_mul, constraint_indices, &mut verifier_transcript);
            phase1_ver_instances.push(Box::new(verifier));
        }

        // Verify Phase 1 with all verifiers
        let phase1_ver_instances_refs: Vec<&dyn SumcheckInstanceVerifier<Fq, Blake2bTranscript>> =
            phase1_ver_instances.iter().map(|v| &**v).collect();

        let r_phase1_ver = BatchedSumcheck::verify(
            &phase1_proof,
            phase1_ver_instances_refs,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .expect("Phase 1 verification should succeed");


        // ============ PHASE 2: Virtualization Sumcheck ============

        use crate::subprotocols::recursion_virtualization::{
            RecursionVirtualizationParams, RecursionVirtualizationProver,
            RecursionVirtualizationVerifier,
        };

        // Create Phase 2 parameters
        let num_s_vars = constraint_system.matrix.num_s_vars;
        let num_constraints_padded = constraint_system.matrix.num_constraints_padded;
        let phase2_params = RecursionVirtualizationParams::new(
            num_s_vars,
            num_constraints,
            num_constraints_padded,
            CommittedPolynomial::DoryConstraintMatrix,
        );

        // Create Phase 2 prover
        let phase2_prover = RecursionVirtualizationProver::new(
            phase2_params.clone(),
            &constraint_system,
            &mut prover_transcript,
            r_phase1.clone(), // x_star from Phase 1
            &prover_accumulator,
            gamma,
        );

        // Run Phase 2 sumcheck
        let mut phase2_prover = phase2_prover;
        let phase2_instances: Vec<&mut dyn SumcheckInstanceProver<Fq, Blake2bTranscript>> =
            vec![&mut phase2_prover];

        let (phase2_proof, r_phase2) = BatchedSumcheck::prove(
            phase2_instances,
            &mut prover_accumulator,
            &mut prover_transcript,
        );


        // Add Phase 2 dense polynomial claims to verifier accumulator
        // In a real proof, these would come from proof.opening_claims
        for (key, (_, claim)) in &prover_accumulator.openings {
            if !verifier_accumulator.openings.contains_key(key) {
                verifier_accumulator
                    .openings
                    .insert(key.clone(), (OpeningPoint::default(), *claim));
            }
        }

        // Get all constraint types for Phase 2 verifier
        let constraint_types: Vec<ConstraintType> = constraint_system
            .constraints
            .iter()
            .map(|c| c.constraint_type.clone())
            .collect();

        // Create Phase 2 verifier
        let phase2_verifier = RecursionVirtualizationVerifier::new(
            phase2_params,
            constraint_types,
            &mut verifier_transcript,
            r_phase1_ver.clone(), // x_star from Phase 1 verification
            gamma,
        );

        // Verify Phase 2
        let phase2_ver_instances: Vec<&dyn SumcheckInstanceVerifier<Fq, Blake2bTranscript>> =
            vec![&phase2_verifier];

        let r_phase2_ver = BatchedSumcheck::verify(
            &phase2_proof,
            phase2_ver_instances,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .expect("Phase 2 verification should succeed");

        assert_eq!(
            r_phase2.len(),
            r_phase2_ver.len(),
            "Phase 2 challenge lengths should match"
        );


        // Import necessary types for Hyrax PCS
        use crate::poly::commitment::hyrax::{Hyrax, HyraxCommitment};
        use ark_grumpkin::Projective as GrumpkinProjective;
        use std::collections::HashMap;

        // Create polynomial map for opening proof
        let mut polynomials_map: HashMap<CommittedPolynomial, MultilinearPolynomial<Fq>> =
            HashMap::new();

        // Add the constraint matrix polynomial
        let matrix_poly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
        polynomials_map.insert(
            CommittedPolynomial::DoryConstraintMatrix,
            matrix_poly.clone(),
        );

        // Create empty hints map for Hyrax (doesn't need hints)
        let _opening_hints: HashMap<CommittedPolynomial, ()> = HashMap::new();

        // Setup Hyrax with RATIO=2 (standard for Hyrax)
        const RATIO: usize = 1;
        type HyraxPCS = Hyrax<RATIO, GrumpkinProjective>;

        // Setup prover generators
        let prover_setup = <HyraxPCS as crate::poly::commitment::commitment_scheme::CommitmentScheme>::setup_prover(
            constraint_system.matrix.num_vars
        );

        // Create polynomial map for prove_single
        let matrix_poly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
        let mut polynomials_map: HashMap<CommittedPolynomial, MultilinearPolynomial<Fq>> =
            HashMap::new();
        polynomials_map.insert(
            CommittedPolynomial::DoryConstraintMatrix,
            matrix_poly.clone(),
        );

        // Run prove_single for the single opening from Phase 2
        let opening_proof = prover_accumulator
            .prove_single::<Blake2bTranscript, HyraxPCS>(
                polynomials_map,
                &prover_setup,
                &mut prover_transcript,
            )
            .expect("prove_single should succeed");


        // Create commitments map for verifier
        let mut commitments_map: HashMap<
            CommittedPolynomial,
            HyraxCommitment<RATIO, GrumpkinProjective>,
        > = HashMap::new();

        // Commit to the matrix polynomial using Hyrax (reuse matrix_poly from above)
        let (matrix_commitment, _) =
            <HyraxPCS as crate::poly::commitment::commitment_scheme::CommitmentScheme>::commit(
                &matrix_poly,
                &prover_setup,
            );
        commitments_map.insert(CommittedPolynomial::DoryConstraintMatrix, matrix_commitment);

        // Setup verifier
        let verifier_setup = <HyraxPCS as crate::poly::commitment::commitment_scheme::CommitmentScheme>::setup_verifier(
            &prover_setup
        );

        // Get the matrix commitment for verify_single
        let matrix_commitment = commitments_map
            .get(&CommittedPolynomial::DoryConstraintMatrix)
            .expect("Matrix commitment should exist")
            .clone();

        // Verify the single opening proof
        let verification_result = verifier_accumulator
            .verify_single::<Blake2bTranscript, HyraxPCS>(
                &opening_proof,
                matrix_commitment,
                &verifier_setup,
                &mut verifier_transcript,
            );

        assert!(
            verification_result.is_ok(),
            "Opening proof verification should succeed: {:?}",
            verification_result.err()
        );

    }
}
