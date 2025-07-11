use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

use super::{D, LOG_K_CHUNK};

use crate::{
    dag::state_manager::{Openings, OpeningsKeys, StateManager},
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::ProverOpeningAccumulator,
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::transcript::Transcript,
};

const DEGREE: usize = 1;

struct HammingProverState<F: JoltField> {
    r: Vec<F>,
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
            .openings_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .r;
        Self {
            gamma: gamma_powers,
            prover_state: Some(HammingProverState {
                r: Vec::with_capacity(LOG_K_CHUNK),
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
            .map(|i| sm.openings(OpeningsKeys::InstructionHammingRa(i)))
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let r_cycle = sm
            .openings_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .r;
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

    fn bind(&mut self, r_j: F, _round: usize) {
        self.prover_state.as_mut().unwrap().r.push(r_j);
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
    fn cache_openings(
        &mut self,
        openings: Option<Rc<RefCell<Openings<F>>>>,
        _accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
    ) {
        let ra_claims = self
            .prover_state
            .as_ref()
            .unwrap()
            .ra
            .iter()
            .map(|ra| ra.final_sumcheck_claim())
            .collect::<Vec<F>>();
        let r_address_prime = self.prover_state.as_ref().unwrap().r.clone();
        let r = r_address_prime
            .iter()
            .chain(self.r_cycle.iter())
            .cloned()
            .collect::<Vec<F>>();
        ra_claims.iter().enumerate().for_each(|(i, claim)| {
            openings.as_ref().unwrap().borrow_mut().insert(
                OpeningsKeys::InstructionHammingRa(i),
                (r.clone().into(), *claim),
            );
        });
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; D],
}

impl<F: JoltField, T: Transcript> HammingWeightProof<F, T> {
    pub fn new(sumcheck_proof: SumcheckInstanceProof<F, T>, openings: &Openings<F>) -> Self {
        Self {
            sumcheck_proof,
            ra_claims: (0..D)
                .map(|i| openings[&OpeningsKeys::InstructionHammingRa(i)].1)
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        }
    }

    pub fn populate_openings(&self, openings: &mut Openings<F>) {
        for (i, claim) in self.ra_claims.iter().enumerate() {
            openings.insert(
                OpeningsKeys::InstructionHammingRa(i),
                (vec![].into(), *claim),
            );
        }
    }
}
