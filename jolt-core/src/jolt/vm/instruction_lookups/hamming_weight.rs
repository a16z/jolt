use std::{cell::RefCell, rc::Rc};

use rayon::prelude::*;

use super::{D, LOG_K_CHUNK};

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
    utils::transcript::Transcript,
};

const DEGREE: usize = 1;

struct HammingProverState<F: JoltField> {
    /// ra_i polynomials
    ra: [MultilinearPolynomial<F>; D],
    /// For the opening proofs
    unbound_ra_polys: Vec<MultilinearPolynomial<F>>,
}

pub struct HammingWeightSumcheck<F: JoltField> {
    gamma: [F; D],
    prover_state: Option<HammingProverState<F>>,
    ra_claims: Option<[F; D]>,
    r_cycle: Vec<F>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionHammingWeightSumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        F: [Vec<F>; D],
        unbound_ra_polys: Vec<MultilinearPolynomial<F>>,
    ) -> Self {
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let ra = F
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        Self {
            gamma: gamma_powers,
            prover_state: Some(HammingProverState {
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
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let ra_claims = (0..D)
            .map(|i| sm.get_opening(OpeningsKeys::InstructionHammingRa(i)))
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        Self {
            gamma: gamma_powers,
            prover_state: None,
            ra_claims: Some(ra_claims),
            r_cycle,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    fn input_claim(&self) -> F {
        self.gamma.iter().sum()
    }

    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
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

    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        self.prover_state
            .as_mut()
            .unwrap()
            .ra
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh))
    }

    fn expected_output_claim(&self, _r: &[F]) -> F {
        self.gamma
            .iter()
            .zip(self.ra_claims.unwrap())
            .map(|(gamma, ra)| ra * gamma)
            .sum()
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> CacheSumcheckOpenings<F, PCS>
    for HammingWeightSumcheck<F>
{
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
        let ra_keys = (0..D)
            .map(OpeningsKeys::InstructionHammingRa)
            .collect::<Vec<_>>();
        let accumulator = accumulator.expect("accumulator is needed");
        accumulator.borrow_mut().append_sparse(
            std::mem::take(&mut ps.unbound_ra_polys),
            opening_point.r.clone(),
            self.r_cycle.clone(),
            ra_claims,
            Some(ra_keys),
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
        (0..D).for_each(|i| {
            accumulator.borrow_mut().populate_claim_opening(
                OpeningsKeys::InstructionHammingRa(i),
                OpeningPoint::new(r.clone()),
            )
        });
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for HammingWeightSumcheck<F>
{
}
