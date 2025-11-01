use itertools::chain;
use num_traits::Zero;
use std::{array, marker::PhantomData, sync::Arc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        lt_poly::LtPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::{
        dag::state_manager::StateManager,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::REGISTER_COUNT;
use rayon::prelude::*;

// Register value evaluation sumcheck
//
// Proves the relation:
//   Val(r) = Σ_{j=0}^{T-1} inc(j) ⋅ wa(r_address, j) ⋅ LT(r_cycle, j)
// where:
// - r = (r_address, r_cycle) is the evaluation point from the read-write checking sumcheck.
// - Val(r) is the claimed value of register r_address at time r_cycle.
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise.
// - wa is the MLE of the write-indicator (1 on matching {0,1}-points).
// - LT is the MLE of strict less-than on bitstrings; evaluated at (r_cycle, j) as field points.
//
// This sumcheck ensures that the claimed final value of a register is consistent
// with all the writes that occurred to it over time (assuming initial value of 0).

const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// Degree bound of the sumcheck round polynomials in [`ValEvaluationSumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative)]
pub(crate) struct ValEvaluationSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: RaPolynomial<u8, F>,
    lt: LtPolynomial<F>,
    #[allocative(skip)]
    params: ValEvaluationSumcheckParams<F>,
}

impl<F: JoltField> ValEvaluationSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheckProver::gen")]
    pub fn gen<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let params = ValEvaluationSumcheckParams::new(state_manager);

        // The opening point is r_address || r_cycle
        let registers_val_input_sample = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, r_cycle) = registers_val_input_sample.0.split_at(LOG_K);

        let (preprocessing, _, trace, _, _) = state_manager.get_prover_data();
        let inc =
            CommittedPolynomial::RdInc.generate_witness(preprocessing, trace, state_manager.ram_d);

        let eq_r_address = EqPolynomial::evals(&r_address.r);
        let wa: Vec<Option<u8>> = trace
            .par_iter()
            .map(|cycle| {
                let instr = cycle.instruction().normalize();
                Some(instr.operands.rd)
            })
            .collect();
        let wa = RaPolynomial::new(Arc::new(wa), eq_r_address);
        let lt = LtPolynomial::new(&r_cycle);

        Self {
            inc,
            wa,
            lt,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ValEvaluationSumcheckProver<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersValEvaluationSumcheckProver::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let half_n = self.inc.len() / 2;

        let [eval_at_1, eval_at_2, eval_at_inf] = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_at_1_j = self.inc.get_bound_coeff(j + half_n);
                let inc_at_inf_j = inc_at_1_j - self.inc.get_bound_coeff(j);
                let inc_at_2_j = inc_at_1_j + inc_at_inf_j;

                let wa_at_1_j = self.wa.get_bound_coeff(j + half_n);
                let wa_at_inf_j = wa_at_1_j - self.wa.get_bound_coeff(j);
                let wa_at_2_j = wa_at_1_j + wa_at_inf_j;

                let lt_at_1_j = self.lt.get_bound_coeff(j + half_n);
                let lt_at_inf_j = lt_at_1_j - self.lt.get_bound_coeff(j);
                let lt_at_2_j = lt_at_1_j + lt_at_inf_j;

                // Eval inc * wa * lt.
                [
                    (inc_at_1_j * wa_at_1_j).mul_unreduced::<9>(lt_at_1_j),
                    (inc_at_2_j * wa_at_2_j).mul_unreduced::<9>(lt_at_2_j),
                    (inc_at_inf_j * wa_at_inf_j).mul_unreduced::<9>(lt_at_inf_j),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_montgomery_reduce);

        let eval_at_0 = previous_claim - eval_at_1;
        let poly = UniPoly::from_evals_toom(&[eval_at_0, eval_at_1, eval_at_2, eval_at_inf]);
        let domain = chain!([0], 2..).take(DEGREE_BOUND).map(F::from_u64);
        domain.map(|x| poly.evaluate::<F>(&x)).collect()
    }

    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::HighToLow);
        self.wa.bind_parallel(r_j, BindingOrder::HighToLow);
        self.lt.bind(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = get_opening_point::<F>(sumcheck_challenges);
        let registers_val_input_sample = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, _) = registers_val_input_sample.0.split_at(LOG_K);

        let inc_claim = self.inc.final_sumcheck_claim();
        let wa_claim = self.wa.final_sumcheck_claim();

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
            inc_claim,
        );

        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
            wa_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ValEvaluationSumcheckVerifier<F: JoltField> {
    params: ValEvaluationSumcheckParams<F>,
}

impl<F: JoltField> ValEvaluationSumcheckVerifier<F> {
    pub fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let params = ValEvaluationSumcheckParams::new(state_manager);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ValEvaluationSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let registers_val_input_sample = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle) = registers_val_input_sample.0.split_at(LOG_K);

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();

        let r = get_opening_point::<F>(sumcheck_challenges);
        for (x, y) in r.r.iter().zip(r_cycle.r.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        // Return inc_claim * wa_claim * lt_eval
        inc_claim * wa_claim * lt_eval
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = get_opening_point::<F>(sumcheck_challenges);
        let registers_val_input_sample = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, _) = registers_val_input_sample.0.split_at(LOG_K);

        // Append claims to accumulator
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
        );

        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
        );
    }
}

struct ValEvaluationSumcheckParams<F: JoltField> {
    n_cycle_vars: usize,
    _phantom: PhantomData<F>,
}

impl<F: JoltField> ValEvaluationSumcheckParams<F> {
    pub fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        Self {
            n_cycle_vars: state_manager.get_trace_len().log_2(),
            _phantom: PhantomData,
        }
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    pub fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, registers_val_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        registers_val_input_claim
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::new(sumcheck_challenges.to_vec())
}
