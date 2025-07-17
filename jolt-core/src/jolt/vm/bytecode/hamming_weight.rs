use std::{cell::RefCell, rc::Rc};

use crate::{
    dag::{stage::StagedSumcheck, state_manager::StateManager},
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::sumcheck::{BatchableSumcheckInstance, CacheSumcheckOpenings},
    utils::{math::Math, transcript::Transcript},
};
use rayon::prelude::*;

pub struct HammingWeightProverState<F: JoltField> {
    ra: Vec<MultilinearPolynomial<F>>,
    unbound_ra_polys: Vec<MultilinearPolynomial<F>>,
}

pub struct HammingWeightSumcheck<F: JoltField> {
    gamma: Vec<F>,
    log_K_chunk: usize,
    d: usize,
    prover_state: Option<HammingWeightProverState<F>>,
    ra_claims: Option<Vec<F>>,
    r_cycle: Vec<F>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeHammingWeightSumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        F: Vec<Vec<F>>,
        unbound_ra_polys: Vec<MultilinearPolynomial<F>>,
    ) -> Self {
        let d = sm.get_prover_data().0.shared.bytecode.d;
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let log_K = sm.get_bytecode().len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        let ra = F
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>();
        Self {
            gamma: gamma_powers,
            log_K_chunk,
            d,
            prover_state: Some(HammingWeightProverState {
                ra,
                unbound_ra_polys,
            }),
            ra_claims: None,
            r_cycle,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let log_K = sm.get_bytecode().len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let ra_claims = (0..d)
            .map(|i| sm.get_opening(OpeningsKeys::BytecodeHammingWeightRa(i)))
            .collect::<Vec<F>>();
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        Self {
            gamma: gamma_powers,
            log_K_chunk,
            d,
            prover_state: None,
            ra_claims: Some(ra_claims),
            r_cycle,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        self.log_K_chunk
    }

    fn input_claim(&self) -> F {
        self.gamma.iter().sum()
    }

    #[tracing::instrument(skip_all, name = "BytecodeHammingWeight::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        vec![prover_state
            .ra
            .iter()
            .zip(self.gamma.iter())
            .map(|(ra, gamma)| {
                (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .sum::<F>()
                    * gamma
            })
            .sum()]
    }

    #[tracing::instrument(skip_all, name = "BytecodeHammingWeight::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        self.prover_state
            .as_mut()
            .unwrap()
            .ra
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh))
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        self.gamma
            .iter()
            .zip(self.ra_claims.as_ref().unwrap())
            .map(|(gamma, ra)| *ra * gamma)
            .sum()
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for HammingWeightSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_mut().unwrap();
        let ra_claims = ps
            .ra
            .iter()
            .map(|ra| ra.final_sumcheck_claim())
            .collect::<Vec<F>>();
        let ra_keys = (0..self.d)
            .map(OpeningsKeys::BytecodeHammingWeightRa)
            .collect::<Vec<_>>();
        let accumulator = accumulator.expect("accumulator is needed");
        ps.unbound_ra_polys
            .iter_mut()
            .zip(ra_claims)
            .zip(ra_keys)
            .for_each(|((ra, claim), key)| {
                accumulator.borrow_mut().append_sparse(
                    vec![std::mem::take(ra)],
                    opening_point.r.clone(),
                    self.r_cycle.clone(),
                    vec![claim],
                    Some(vec![key]),
                );
            });
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let accumulator = accumulator.expect("accumulator is needed");
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(self.r_cycle.iter().cloned())
            .collect::<Vec<_>>();
        (0..self.d).for_each(|i| {
            accumulator
                .borrow_mut()
                .populate_claim_opening(OpeningsKeys::BytecodeHammingWeightRa(i), r.clone().into())
        });
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for HammingWeightSumcheck<F>
{
}
