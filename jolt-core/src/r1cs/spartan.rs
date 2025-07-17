use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use tracer::instruction::RV32IMCycle;
use tracing::{span, Level};

use crate::dag::stage::{StagedSumcheck, SumcheckStages};
use crate::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::field::JoltField;
use crate::jolt::instruction::CircuitFlags;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::vm::JoltProverPreprocessing;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::OpeningsKeys::{OuterSumcheckAz, OuterSumcheckBz, OuterSumcheckCz};
use crate::poly::opening_proof::{
    OpeningPoint, OpeningsExt, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
    BIG_ENDIAN,
};
use crate::r1cs::builder::Constraint;
use crate::r1cs::constraints::{JoltRV32IMConstraints, R1CSConstraints};
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::r1cs::inputs::ALL_R1CS_INPUTS;
use crate::r1cs::inputs::COMMITTED_R1CS_INPUTS;
use crate::r1cs::key::UniformSpartanKey;
use crate::subprotocols::sumcheck::CacheSumcheckOpenings;
use crate::utils::math::Math;

use crate::utils::transcript::Transcript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use thiserror::Error;

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPlusOnePolynomial},
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::small_value::NUM_SVO_ROUNDS,
};

use super::builder::CombinedUniformBuilder;

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct OuterClaims<F: JoltField> {
    az: F,
    bz: F,
    cz: F,
}

#[derive(Clone, Debug)]
pub struct InnerSumcheckParams<F: JoltField> {
    r_cycle: Vec<F>,
    rx_var: Vec<F>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned when the outer sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidInnerSumcheckProof")]
    InvalidInnerSumcheckProof,

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidShiftSumcheckProof")]
    InvalidShiftSumcheckProof,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct UniformSpartanProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) shift_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) claimed_witness_evals: Vec<F>,
    pub(crate) shift_sumcheck_witness_eval: Vec<F>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F, ProofTranscript> UniformSpartanProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Spartan::setup")]
    pub fn setup(
        constraint_builder: &CombinedUniformBuilder<F>,
        padded_num_steps: usize,
    ) -> UniformSpartanKey<F> {
        assert_eq!(
            padded_num_steps,
            constraint_builder.uniform_repeat().next_power_of_two()
        );
        UniformSpartanKey::from_builder(constraint_builder)
    }

    #[tracing::instrument(skip_all, name = "Spartan::prove")]
    pub fn prove<PCS>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        constraint_builder: &CombinedUniformBuilder<F>,
        key: UniformSpartanKey<F>,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<Field = F>,
    {
        let input_polys: Vec<MultilinearPolynomial<F>> = ALL_R1CS_INPUTS
            .par_iter()
            .map(|var| var.generate_witness(trace, preprocessing))
            .collect();

        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck
           Proves: \sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0

           The matrices A, B, C have a block-diagonal structure with repeated blocks
           A_small, B_small, C_small corresponding to the uniform constraints.
        */

        let tau: Vec<F> = transcript.challenge_vector(num_rounds_x);
        let uniform_constraints_only_padded = constraint_builder
            .uniform_builder
            .constraints
            .len()
            .next_power_of_two();
        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            Self::prove_outer_sumcheck(
                num_rounds_x,
                uniform_constraints_only_padded,
                &constraint_builder.uniform_builder.constraints,
                &input_polys,
                &tau,
                transcript,
            );
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        /* Sumcheck 2: Inner sumcheck
           Proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                   \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)

           Evaluates the uniform constraint matrices A_small, B_small, C_small at the point
           determined by the outer sumcheck.
        */

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();

        let (r_cycle, rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let claims = OuterClaims {
            az: claim_Az,
            bz: claim_Bz,
            cz: claim_Cz,
        };
        let params = InnerSumcheckParams {
            r_cycle: r_cycle.to_vec(),
            rx_var: rx_var.to_vec(),
        };

        let (inner_sumcheck_proof, _inner_sumcheck_r) = Self::prove_inner_sumcheck(
            key.into(),
            &input_polys,
            &claims,
            &params,
            inner_sumcheck_RLC,
            transcript,
        );

        // Evaluate all witness polynomials P_i at r_cycle for the verifier
        // Verifier computes: z(r_inner, r_cycle) = Σ_i eq(r_inner, i) * P_i(r_cycle)
        let flattened_polys_ref: Vec<_> = input_polys.iter().collect();
        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys_ref, r_cycle);

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::evals(r_cycle, None);

        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */
        let (shift_sumcheck_proof, shift_sumcheck_witness_eval) = Self::prove_pc_sumcheck(
            &input_polys,
            &claimed_witness_evals,
            eq_plus_one_r_cycle,
            r_cycle.to_vec(),
            transcript,
        );

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof. Virtual polynomial openings
        // are proven in some subsequent sumcheck.
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| &input_polys[input.to_index()])
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();

        opening_accumulator.append_dense(
            &committed_polys,
            chis,
            r_cycle.to_vec(),
            &committed_poly_claims,
            transcript,
            None,
        );

        let outer_sumcheck_claims = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );
        Ok(UniformSpartanProof {
            outer_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_proof,
            shift_sumcheck_proof,
            claimed_witness_evals,
            shift_sumcheck_witness_eval,
            _marker: PhantomData,
        })
    }

    #[tracing::instrument(skip_all)]
    fn prove_outer_sumcheck(
        num_rounds_x: usize,
        uniform_constraints_only_padded: usize,
        uniform_constraints: &[Constraint],
        input_polys: &[MultilinearPolynomial<F>],
        tau: &[F],
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, [F; 3]) {
        SumcheckInstanceProof::prove_spartan_small_value::<NUM_SVO_ROUNDS>(
            num_rounds_x,
            uniform_constraints_only_padded,
            uniform_constraints,
            input_polys,
            tau,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn prove_inner_sumcheck(
        key: Arc<UniformSpartanKey<F>>,
        input_polys: &[MultilinearPolynomial<F>],
        claims: &OuterClaims<F>,
        params: &InnerSumcheckParams<F>,
        inner_sumcheck_RLC: F,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let mut inner_sumcheck =
            InnerSumcheck::new_prover(key, input_polys, claims, params, inner_sumcheck_RLC);

        let (inner_sumcheck_proof, r) = inner_sumcheck.prove_single(transcript);

        (inner_sumcheck_proof, r)
    }

    fn prove_pc_sumcheck(
        _input_polys: &[MultilinearPolynomial<F>],
        _claimed_witness_evals: &[F],
        _eq_plus_one_r_cycle: Vec<F>,
        _r_cycle: Vec<F>,
        _transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        todo!()
    }

    #[tracing::instrument(skip_all, name = "Spartan::verify")]
    pub fn verify<PCS>(
        &self,
        _key: UniformSpartanKey<F>,
        _commitments: &JoltCommitments<F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
        _transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        todo!()
    }
}

struct InnerSumcheckProverState<F: JoltField> {
    poly_abc_small: MultilinearPolynomial<F>,
    poly_z: MultilinearPolynomial<F>,
}

struct InnerSumcheckVerifierState<F: JoltField> {
    key: Arc<UniformSpartanKey<F>>,
    rx_var: Vec<F>,
    claimed_witness_evals: Vec<F>,
    inner_sumcheck_RLC: F,
}

pub struct InnerSumcheck<F: JoltField> {
    input_claim: F,
    prover_state: Option<InnerSumcheckProverState<F>>,
    verifier_state: Option<InnerSumcheckVerifierState<F>>,
    cached_claims: Option<(F, F)>, // (final_poly_abc_eval, final_poly_z_eval)
}

impl<F: JoltField> InnerSumcheck<F> {
    pub fn new_prover(
        key: Arc<UniformSpartanKey<F>>,
        input_polys: &[MultilinearPolynomial<F>],
        claims: &OuterClaims<F>,
        params: &InnerSumcheckParams<F>,
        inner_sumcheck_RLC: F,
    ) -> Self {
        let num_vars_uniform = key.num_vars_uniform_padded();
        let claim_inner_joint =
            claims.az + inner_sumcheck_RLC * claims.bz + inner_sumcheck_RLC.square() * claims.cz;

        let (eq_r_cycle, _) = EqPlusOnePolynomial::evals(&params.r_cycle, None);

        // Evaluate A_small, B_small, C_small combined with RLC at point rx_var
        let poly_abc_small =
            DensePolynomial::new(key.evaluate_small_matrix_rlc(&params.rx_var, inner_sumcheck_RLC));

        let span = span!(Level::INFO, "binding_z_second_sumcheck");
        let _guard = span.enter();

        // Bind witness polynomials z at point r_cycle
        let mut bind_z = vec![F::zero(); num_vars_uniform];

        input_polys
            .par_iter()
            .take(num_vars_uniform)
            .zip(bind_z.par_iter_mut())
            .for_each(|(poly, eval)| {
                *eval = poly.dot_product(&eq_r_cycle);
            });

        // Set the constant value at the appropriate position
        if key.uniform_r1cs.num_vars < num_vars_uniform {
            bind_z[key.uniform_r1cs.num_vars] = F::one();
        }

        drop(_guard);
        drop(span);

        let poly_z = DensePolynomial::new(bind_z);
        assert_eq!(poly_z.len(), poly_abc_small.len());

        Self {
            input_claim: claim_inner_joint,
            prover_state: Some(InnerSumcheckProverState {
                poly_abc_small: MultilinearPolynomial::LargeScalars(poly_abc_small),
                poly_z: MultilinearPolynomial::LargeScalars(poly_z),
            }),
            verifier_state: None,
            cached_claims: None,
        }
    }

    pub fn new_verifier(
        input_claim: F,
        key: Arc<UniformSpartanKey<F>>,
        rx_var: Vec<F>,
        claimed_witness_evals: Vec<F>,
        inner_sumcheck_RLC: F,
    ) -> Self {
        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(InnerSumcheckVerifierState {
                key,
                rx_var,
                claimed_witness_evals,
                inner_sumcheck_RLC,
            }),
            cached_claims: None,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for InnerSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.poly_abc_small.original_len().log_2()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.key.num_vars_uniform_padded().log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.poly_abc_small.len() / 2)
            .into_par_iter()
            .map(|i| {
                let abc_evals = prover_state
                    .poly_abc_small
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let z_evals = prover_state
                    .poly_z
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    abc_evals[0] * z_evals[0], // eval at 0
                    abc_evals[1] * z_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        // Bind both polynomials in parallel
        rayon::join(
            || {
                prover_state
                    .poly_abc_small
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
            || {
                prover_state
                    .poly_z
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
        );
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // The verifier needs to compute:
        // (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)

        // Evaluate uniform matrices A_small, B_small, C_small at point (rx_var, ry_var)
        let eval_a = verifier_state
            .key
            .evaluate_uniform_a_at_point(&verifier_state.rx_var, r);
        let eval_b = verifier_state
            .key
            .evaluate_uniform_b_at_point(&verifier_state.rx_var, r);
        let eval_c = verifier_state
            .key
            .evaluate_uniform_c_at_point(&verifier_state.rx_var, r);

        let left_expected = eval_a
            + verifier_state.inner_sumcheck_RLC * eval_b
            + verifier_state.inner_sumcheck_RLC * verifier_state.inner_sumcheck_RLC * eval_c;

        // Evaluate z(ry)
        let eval_z = verifier_state.key.evaluate_z_mle_with_segment_evals(
            &verifier_state.claimed_witness_evals,
            r,
            true,
        );

        left_expected * eval_z
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for InnerSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        _accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.cached_claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        // Note that these claims are never used by the verifier hence we do not add to the state manager
        let final_poly_abc = prover_state.poly_abc_small.final_sumcheck_claim();
        let final_poly_z = prover_state.poly_z.final_sumcheck_claim();

        self.cached_claims = Some((final_poly_abc, final_poly_z));
    }
}

struct PCSumcheckProverState<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_poly: MultilinearPolynomial<F>,
}

struct PCSumcheckVerifierState<F: JoltField> {
    r_cycle: Vec<F>,
    unexpanded_pc_eval_at_shift_r: F,
    pc_eval_at_shift_r: F,
    is_noop_eval_at_shift_r: F,
}

pub struct PCSumcheck<F: JoltField> {
    input_claim: F,
    gamma: F,
    gamma_squared: F,
    log_T: usize,
    prover_state: Option<PCSumcheckProverState<F>>,
    verifier_state: Option<PCSumcheckVerifierState<F>>,
}

impl<F: JoltField> PCSumcheck<F> {
    pub fn new_prover(
        input_polys: &[MultilinearPolynomial<F>],
        claimed_witness_evals: &[F],
        eq_plus_one_r_cycle: Vec<F>,
        gamma: F,
        r_cycle: Vec<F>,
    ) -> Self {
        let gamma_squared = gamma.square();
        let unexpanded_pc_index = JoltR1CSInputs::UnexpandedPC.to_index();
        let pc_index = JoltR1CSInputs::PC.to_index();
        let next_unexpanded_pc_index = JoltR1CSInputs::NextUnexpandedPC.to_index();
        let next_pc_index = JoltR1CSInputs::NextPC.to_index();
        let noop_index = JoltR1CSInputs::OpFlags(CircuitFlags::IsNoop).to_index();
        let next_is_noop_index = JoltR1CSInputs::NextIsNoop.to_index();

        let input_claim = claimed_witness_evals[next_unexpanded_pc_index]
            + gamma * claimed_witness_evals[next_pc_index]
            + gamma_squared * claimed_witness_evals[next_is_noop_index];

        Self {
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(PCSumcheckProverState {
                unexpanded_pc_poly: input_polys[unexpanded_pc_index].clone(),
                pc_poly: input_polys[pc_index].clone(),
                is_noop_poly: input_polys[noop_index].clone(),
                eq_plus_one_poly: MultilinearPolynomial::from(eq_plus_one_r_cycle),
            }),
            gamma,
            gamma_squared,
            verifier_state: None,
        }
    }

    pub fn new_verifier(
        input_claim: F,
        r_cycle: Vec<F>,
        gamma: F,
        unexpanded_pc_eval_at_shift_r: F,
        pc_eval_at_shift_r: F,
        is_noop_eval_at_shift_r: F,
    ) -> Self {
        let gamma_squared = gamma.square();
        Self {
            input_claim,
            prover_state: None,
            log_T: r_cycle.len(),
            verifier_state: Some(PCSumcheckVerifierState {
                r_cycle,
                unexpanded_pc_eval_at_shift_r,
                pc_eval_at_shift_r,
                is_noop_eval_at_shift_r,
            }),
            gamma,
            gamma_squared,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for PCSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.unexpanded_pc_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let unexpanded_pc_evals = prover_state
                    .unexpanded_pc_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let pc_evals = prover_state
                    .pc_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let eq_evals = prover_state
                    .eq_plus_one_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let is_noop_evals = prover_state
                    .is_noop_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    (unexpanded_pc_evals[0]
                        + self.gamma * pc_evals[0]
                        + self.gamma_squared * is_noop_evals[0])
                        * eq_evals[0], // eval at 0
                    (unexpanded_pc_evals[1]
                        + self.gamma * pc_evals[1]
                        + self.gamma_squared * is_noop_evals[1])
                        * eq_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::scope(|s| {
            s.spawn(|_| {
                prover_state
                    .unexpanded_pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .is_noop_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .eq_plus_one_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
        });
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        let batched_eval_at_shift_r = verifier_state.unexpanded_pc_eval_at_shift_r
            + self.gamma * verifier_state.pc_eval_at_shift_r
            + self.gamma_squared * verifier_state.is_noop_eval_at_shift_r;

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::new(verifier_state.r_cycle.clone()).evaluate(r);

        batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for PCSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let unexpanded_pc_eval = prover_state.unexpanded_pc_poly.final_sumcheck_claim();
        let pc_eval = prover_state.pc_poly.final_sumcheck_claim();
        let is_noop_eval = prover_state.is_noop_poly.final_sumcheck_claim();

        let accumulator = accumulator.expect("accumulator is needed");

        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::PCSumcheckUnexpandedPC,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::PCSumcheckPC,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::PCSumcheckIsNoop,
            opening_point,
            is_noop_eval,
        );
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let accumulator = accumulator.expect("accumulator is needed");

        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::PCSumcheckUnexpandedPC, opening_point.clone());
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::PCSumcheckPC, opening_point.clone());
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::PCSumcheckIsNoop, opening_point);
    }
}

pub struct SpartanDag<F: JoltField> {
    /// Cached key to avoid recomputation across stages
    key: Arc<UniformSpartanKey<F>>,
}

impl<F: JoltField> SpartanDag<F> {
    pub fn new<ProofTranscript: Transcript>(padded_trace_length: usize) -> Self {
        let constraint_builder = JoltRV32IMConstraints::construct_constraints(padded_trace_length);
        let key = Arc::new(UniformSpartanProof::<F, ProofTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        ));
        Self { key }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS> for InnerSumcheck<F> {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS> for PCSumcheck<F> {}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for SpartanDag<F>
{
    fn stage1_prove(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        /* Sumcheck 1: Outer sumcheck
           Proves: \sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0

           The matrices A, B, C have a block-diagonal structure with repeated blocks
           A_small, B_small, C_small corresponding to the uniform constraints.
        */
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        let padded_trace_length = trace.len().next_power_of_two();
        let key = self.key.clone();

        // Create input polynomials from trace
        let input_polys: Vec<MultilinearPolynomial<F>> = ALL_R1CS_INPUTS
            .par_iter()
            .map(|var| var.generate_witness(trace, preprocessing))
            .collect();

        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(num_rounds_x);

        // Recreate constraint_builder from padded_trace_length
        let constraint_builder: CombinedUniformBuilder<F> =
            JoltRV32IMConstraints::construct_constraints(padded_trace_length);

        let uniform_constraints_only_padded = constraint_builder
            .uniform_builder
            .constraints
            .len()
            .next_power_of_two();

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) = {
            let mut transcript = state_manager.transcript.borrow_mut();
            UniformSpartanProof::<F, ProofTranscript>::prove_outer_sumcheck(
                num_rounds_x,
                uniform_constraints_only_padded,
                &constraint_builder.uniform_builder.constraints,
                &input_polys,
                &tau,
                &mut transcript,
            )
        };

        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(
            &mut *state_manager.transcript.borrow_mut(),
            &outer_sumcheck_claims,
        );

        // Store Az, Bz, Cz claims with the outer sumcheck point
        let accumulator = state_manager.get_prover_accumulator();
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::OuterSumcheckAz,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[0],
        );
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::OuterSumcheckBz,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[1],
        );
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::OuterSumcheckCz,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[2],
        );

        // Append the outer sumcheck proof to the state manager
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::SpartanOuterSumcheck,
            ProofData::SpartanOuterData(outer_sumcheck_proof),
        );

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Evaluate all witness polynomials P_i at r_cycle for the verifier
        // Verifier computes: z(r_inner, r_cycle) = Σ_i eq(r_inner, i) * P_i(r_cycle)
        let flattened_polys_ref: Vec<_> = input_polys.iter().collect();
        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys_ref, r_cycle);

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof, which we add for future opening proof here
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| &input_polys[input.to_index()])
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();

        let accumulator = state_manager.get_prover_accumulator();

        let openings_keys: Vec<OpeningsKeys> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| OpeningsKeys::SpartanZ(*input))
            .collect();

        accumulator.borrow_mut().append_dense(
            &committed_polys,
            chis,
            r_cycle.to_vec(),
            &committed_poly_claims,
            &mut *state_manager.transcript.borrow_mut(),
            Some(openings_keys),
        );

        // Add virtual polynomial evaluations to the accumulator
        // These are needed by the verifier for future sumchecks and are not part of the PCS opening proof
        for (input, eval) in ALL_R1CS_INPUTS.iter().zip(claimed_witness_evals.iter()) {
            // Skip if it's a committed input (already added above)
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                accumulator.borrow_mut().append_virtual(
                    OpeningsKeys::SpartanZ(*input),
                    OpeningPoint::new(r_cycle.to_vec()),
                    *eval,
                );
            }
        }

        Ok(())
    }

    fn stage1_verify(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let key = self.key.clone();

        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(num_rounds_x);

        // Get the outer sumcheck proof
        let proofs = state_manager.proofs.borrow();
        let proof_data = {
            proofs
                .get(&ProofKeys::SpartanOuterSumcheck)
                .expect("Outer sumcheck proof not found")
        };

        let outer_sumcheck_proof = match proof_data {
            ProofData::SpartanOuterData(proof) => proof,
            _ => panic!("Invalid proof data type"),
        };

        // Get the claims:
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_ref = accumulator.borrow();
        let claim_Az = accumulator_ref.get_opening(OpeningsKeys::OuterSumcheckAz);
        let claim_Bz = accumulator_ref.get_opening(OpeningsKeys::OuterSumcheckBz);
        let claim_Cz = accumulator_ref.get_opening(OpeningsKeys::OuterSumcheckCz);
        drop(accumulator_ref);
        let outer_sumcheck_claims = [claim_Az, claim_Bz, claim_Cz];

        // Run the main sumcheck verifier:
        let (claim_outer_final, outer_sumcheck_r_original) = {
            let transcript = &mut state_manager.transcript.borrow_mut();
            match outer_sumcheck_proof.verify(F::zero(), num_rounds_x, 3, transcript) {
                Ok(result) => result,
                Err(_) => return Err(anyhow::anyhow!("Outer sumcheck verification failed")),
            }
        };

        // Outer sumcheck is bound from the top, reverse the challenge
        // TODO(markosg04): Make use of Endianness here?
        let outer_sumcheck_r_reversed: Vec<F> =
            outer_sumcheck_r_original.iter().rev().cloned().collect();
        let opening_point = OpeningPoint::new(outer_sumcheck_r_reversed.clone());

        // Populate the opening points for Az, Bz, Cz claims now that we have outer_sumcheck_r
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::OuterSumcheckAz, opening_point.clone());
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::OuterSumcheckBz, opening_point.clone());
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::OuterSumcheckCz, opening_point.clone());

        let tau_bound_rx = EqPolynomial::mle(&tau, &outer_sumcheck_r_reversed);
        let claim_outer_final_expected = tau_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(anyhow::anyhow!("Invalid outer sumcheck claim"));
        }

        ProofTranscript::append_scalars(
            &mut state_manager.transcript.borrow_mut(),
            &outer_sumcheck_claims[..],
        );

        // Add the commitments to verifier accumulator
        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r_reversed.split_at(num_cycles_bits);

        let accumulator = state_manager.get_verifier_accumulator();

        // Get commitments - TODO(moodlezoup): This relies on ordering of commitments
        let commitments = state_manager.get_commitments();
        let r1cs_input_commitments: Vec<_> = commitments
            .commitments
            .iter()
            .take(COMMITTED_R1CS_INPUTS.len())
            .collect();

        // Get claims for committed inputs
        let claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| {
                accumulator
                    .borrow()
                    .evaluation_openings()
                    .get_spartan_z(*input)
            })
            .collect();

        // let r_cycle: Vec<F> = r_cycle.into_iter().cloned().rev().collect();

        accumulator.borrow_mut().append(
            &r1cs_input_commitments,
            r_cycle.to_vec(),
            &claims,
            &mut *state_manager.transcript.borrow_mut(),
        );

        ALL_R1CS_INPUTS.iter().for_each(|input| {
            accumulator.borrow_mut().populate_claim_opening(
                OpeningsKeys::SpartanZ(*input),
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });

        Ok(())
    }

    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        /* Sumcheck 2: Inner sumcheck
            Proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)

            Evaluates the uniform constraint matrices A_small, B_small, C_small at the point
            determined by the outer sumcheck.
        */

        // Get the program data
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        let key = self.key.clone();

        let input_polys: Vec<MultilinearPolynomial<F>> = ALL_R1CS_INPUTS
            .par_iter()
            .map(|var| var.generate_witness(trace, preprocessing))
            .collect();

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let inner_sumcheck_RLC: F = state_manager.transcript.borrow_mut().challenge_scalar();

        // Get opening_point from accumulator (Az, Bz, Cz all have the same point)
        let accumulator = state_manager.get_prover_accumulator();
        let outer_sumcheck_r = accumulator
            .borrow()
            .get_opening_point(OuterSumcheckAz)
            .unwrap();

        let (r_cycle, rx_var) = outer_sumcheck_r.r.split_at(num_cycles_bits);

        let accumulator_ref = accumulator.borrow();
        let claim_Az = accumulator_ref.get_opening(OuterSumcheckAz);
        let claim_Bz = accumulator_ref.get_opening(OuterSumcheckBz);
        let claim_Cz = accumulator_ref.get_opening(OuterSumcheckCz);
        drop(accumulator_ref);

        let claims = OuterClaims {
            az: claim_Az,
            bz: claim_Bz,
            cz: claim_Cz,
        };
        let params = InnerSumcheckParams {
            r_cycle: r_cycle.to_vec(),
            rx_var: rx_var.to_vec(),
        };

        let inner_sumcheck =
            InnerSumcheck::new_prover(key, &input_polys, &claims, &params, inner_sumcheck_RLC);

        vec![Box::new(inner_sumcheck)]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        /* Sumcheck 2: Inner sumcheck
           Verifies: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)
        */

        let key = self.key.clone();

        let inner_sumcheck_RLC: F = state_manager.transcript.borrow_mut().challenge_scalar();

        // Get outer sumcheck claims from accumulator
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_ref = accumulator.borrow();
        let claim_Az = accumulator_ref.get_opening(OuterSumcheckAz);
        let claim_Bz = accumulator_ref.get_opening(OuterSumcheckBz);
        let claim_Cz = accumulator_ref.get_opening(OuterSumcheckCz);
        drop(accumulator_ref);

        // Compute joint claim
        let claim_inner_joint =
            claim_Az + inner_sumcheck_RLC * claim_Bz + inner_sumcheck_RLC.square() * claim_Cz;

        let outer_sumcheck_r = accumulator
            .borrow()
            .get_opening_point(OuterSumcheckAz)
            .unwrap();
        let num_cycles_bits = key.num_steps.log_2();

        let (_r_cycle, rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let claimed_witness_evals: Vec<F> = ALL_R1CS_INPUTS
            .iter()
            .map(|input| {
                accumulator
                    .borrow()
                    .evaluation_openings()
                    .get_spartan_z(*input)
            })
            .collect();

        let inner_sumcheck = InnerSumcheck::<F>::new_verifier(
            claim_inner_joint,
            key,
            rx_var.to_vec(),
            claimed_witness_evals,
            inner_sumcheck_RLC,
        );

        vec![Box::new(inner_sumcheck)]
    }

    fn stage3_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */

        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        let key = self.key.clone();

        // We need only pc and unexpanded pc for the next sumcheck
        let pc_poly = JoltR1CSInputs::PC.generate_witness(trace, preprocessing);
        let unexpanded_pc_poly =
            JoltR1CSInputs::UnexpandedPC.generate_witness(trace, preprocessing);
        let is_noop_poly =
            JoltR1CSInputs::OpFlags(CircuitFlags::IsNoop).generate_witness(trace, preprocessing);

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        // Get opening_point from accumulator (Az, Bz, Cz all have the same point)
        let accumulator = state_manager.get_prover_accumulator();
        let outer_sumcheck_r = accumulator
            .borrow()
            .get_opening_point(OuterSumcheckAz)
            .unwrap();
        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Get the NextPC and NextUnexpandedPC claims from the accumulator from stage 1
        let next_pc_eval = accumulator
            .borrow()
            .get_opening(OpeningsKeys::SpartanZ(JoltR1CSInputs::NextPC));
        let next_unexpanded_pc_eval = accumulator
            .borrow()
            .get_opening(OpeningsKeys::SpartanZ(JoltR1CSInputs::NextUnexpandedPC));
        let next_is_noop_eval = accumulator
            .borrow()
            .get_opening(OpeningsKeys::SpartanZ(JoltR1CSInputs::NextIsNoop));

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::evals(r_cycle, None);

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let gamma_squared = gamma.square();

        let pc_sumcheck = PCSumcheck {
            input_claim: next_unexpanded_pc_eval
                + gamma * next_pc_eval
                + gamma_squared * next_is_noop_eval,
            log_T: r_cycle.len(),
            prover_state: Some(PCSumcheckProverState {
                unexpanded_pc_poly,
                pc_poly,
                is_noop_poly,
                eq_plus_one_poly: MultilinearPolynomial::from(eq_plus_one_r_cycle),
            }),
            gamma,
            gamma_squared,
            verifier_state: None,
        };

        vec![Box::new(pc_sumcheck)]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        /* Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
           Verifies the batched constraint for both NextUnexpandedPC and NextPC
        */
        let key = self.key.clone();

        // Get batching challenge for combining NextUnexpandedPC and NextPC
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();

        // Get r_cycle from outer sumcheck opening point
        let accumulator = state_manager.get_verifier_accumulator();
        let outer_sumcheck_r = accumulator
            .borrow()
            .get_opening_point(OuterSumcheckAz)
            .unwrap();
        let num_cycles_bits = key.num_steps.log_2();

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let next_unexpanded_pc_eval = accumulator
            .borrow()
            .evaluation_openings()
            .get_spartan_z(JoltR1CSInputs::NextUnexpandedPC);
        let next_pc_eval = accumulator
            .borrow()
            .evaluation_openings()
            .get_spartan_z(JoltR1CSInputs::NextPC);
        let next_is_noop_eval = accumulator
            .borrow()
            .evaluation_openings()
            .get_spartan_z(JoltR1CSInputs::NextIsNoop);
        let shift_sumcheck_claim =
            next_unexpanded_pc_eval + gamma * next_pc_eval + gamma.square() * next_is_noop_eval;

        // Get shift sumcheck witness evaluations from openings (UnexpandedPC and PC at shift_r)
        let unexpanded_pc_eval_at_shift_r = accumulator
            .borrow()
            .get_opening(OpeningsKeys::PCSumcheckUnexpandedPC);
        let pc_eval_at_shift_r = accumulator.borrow().get_opening(OpeningsKeys::PCSumcheckPC);
        let is_noop_eval_at_shift_r = accumulator
            .borrow()
            .get_opening(OpeningsKeys::PCSumcheckIsNoop);

        let pc_sumcheck = PCSumcheck::<F>::new_verifier(
            shift_sumcheck_claim,
            r_cycle.to_vec(),
            gamma,
            unexpanded_pc_eval_at_shift_r,
            pc_eval_at_shift_r,
            is_noop_eval_at_shift_r,
        );

        vec![Box::new(pc_sumcheck)]
    }
}
