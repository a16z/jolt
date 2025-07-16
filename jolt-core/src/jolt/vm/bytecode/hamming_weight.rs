use std::{cell::RefCell, rc::Rc};

use crate::{
    dag::{stage::StagedSumcheck, state_manager::StateManager},
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningPoint, OpeningsKeys, ProverOpeningAccumulator, BIG_ENDIAN},
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::sumcheck::{BatchableSumcheckInstance, CacheSumcheckOpenings},
    utils::{math::Math, transcript::Transcript},
};
use rayon::prelude::*;

pub struct HammingWeightProverState<F: JoltField> {
    ra: MultilinearPolynomial<F>,
    unbound_ra_poly: Option<MultilinearPolynomial<F>>,
}

pub struct HammingWeightSumcheck<F: JoltField> {
    log_K: usize,
    prover_state: Option<HammingWeightProverState<F>>,
    ra_claim: Option<F>,
    r_cycle: Vec<F>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        F: Vec<F>,
        unbound_ra_poly: MultilinearPolynomial<F>,
    ) -> Self {
        let log_K = sm.get_bytecode().len().log_2();
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        Self {
            log_K,
            prover_state: Some(HammingWeightProverState {
                ra: MultilinearPolynomial::from(F),
                unbound_ra_poly: Some(unbound_ra_poly),
            }),
            ra_claim: None,
            r_cycle,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let log_K = sm.get_bytecode().len().log_2();
        let ra_claim = sm.get_opening(OpeningsKeys::BytecodeHammingWeightRa);
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        Self {
            log_K,
            prover_state: None,
            ra_claim: Some(ra_claim),
            r_cycle,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }

    fn input_claim(&self) -> F {
        F::one()
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let univariate_poly_eval: F = (0..prover_state.ra.len() / 2)
            .into_par_iter()
            .map(|i| prover_state.ra.get_bound_coeff(2 * i))
            .sum();

        vec![univariate_poly_eval]
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state.ra.bind_parallel(r_j, BindingOrder::LowToHigh)
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        self.ra_claim.expect("ra_claim not set")
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for HammingWeightSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        let accumulator = accumulator.expect("accumulator is needed");

        let ra_claim = ps.ra.final_sumcheck_claim();
        accumulator.borrow_mut().append_sparse(
            vec![ps.unbound_ra_poly.take().unwrap()],
            opening_point.r,
            self.r_cycle.clone(),
            vec![ra_claim],
            Some(vec![OpeningsKeys::BytecodeHammingWeightRa]),
        );
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(self.r_cycle.iter().cloned())
            .collect::<Vec<_>>();
        let accumulator = accumulator.expect("accumulator is needed");
        accumulator.borrow_mut().populate_claim_opening(
            OpeningsKeys::BytecodeHammingWeightRa,
            OpeningPoint::new(r.clone()),
        );
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for HammingWeightSumcheck<F>
{
}
