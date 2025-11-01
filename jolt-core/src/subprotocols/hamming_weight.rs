use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::iter::zip;

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

/// Degree bound of the sumcheck round polynomials in [`HammingWeightSumcheckVerifier`].
const DEGREE_BOUND: usize = 1;

#[derive(Allocative)]
pub struct HammingWeightSumcheckProver<F: JoltField> {
    ra: Vec<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: HammingWeightSumcheckParams<F>,
}

impl<F: JoltField> HammingWeightSumcheckProver<F> {
    pub fn gen(params: HammingWeightSumcheckParams<F>, G: Vec<Vec<F>>) -> Self {
        let ra = G.into_iter().map(MultilinearPolynomial::from).collect();
        Self { ra, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for HammingWeightSumcheckProver<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(skip_all, name = "HammingWeightProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_msg = self
            .ra
            .par_iter()
            .zip(self.params.gamma_powers.par_iter())
            .map(|(ra, gamma)| {
                let ra_sum = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .fold_with(F::Unreduced::<5>::zero(), |running, new| {
                        running + new.as_unreduced_ref()
                    })
                    .reduce(F::Unreduced::zero, |running, new| running + new);
                ra_sum.mul_trunc::<4, 9>(gamma.as_unreduced_ref())
            })
            .reduce(F::Unreduced::zero, |running, new| running + new);

        vec![F::from_montgomery_reduce(prover_msg)]
    }

    #[tracing::instrument(skip_all, name = "HammingWeightProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = get_opening_point::<F>(sumcheck_challenges);
        let claims: Vec<F> = self.ra.iter().map(|ra| ra.final_sumcheck_claim()).collect();

        let r_cycle = self.params.get_r_cycle(accumulator);

        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            self.params.sumcheck_id,
            opening_point.r,
            r_cycle,
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct HammingWeightSumcheckVerifier<F: JoltField> {
    params: HammingWeightSumcheckParams<F>,
}

impl<F: JoltField> HammingWeightSumcheckVerifier<F> {
    pub fn new(params: HammingWeightSumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingWeightSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claims = (0..self.params.d).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    self.params.polynomial_types[i],
                    self.params.sumcheck_id,
                )
                .1
        });

        // Compute batched claim: sum_{i=0}^{d-1} gamma^i * ra_i
        zip(ra_claims, &self.params.gamma_powers)
            .map(|(ra_claim, gamma)| ra_claim * gamma)
            .sum()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.get_r_cycle(accumulator);
        let r = get_opening_point::<F>(sumcheck_challenges)
            .r
            .iter()
            .cloned()
            .chain(r_cycle.iter().cloned())
            .collect::<Vec<_>>();

        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            self.params.sumcheck_id,
            r,
        );
    }
}

pub struct HammingWeightSumcheckParams<F: JoltField> {
    pub d: usize,
    pub num_rounds: usize,
    pub gamma_powers: Vec<F>,
    pub polynomial_types: Vec<CommittedPolynomial>,
    pub sumcheck_id: SumcheckId,
    pub virtual_poly: Option<VirtualPolynomial>,
    pub r_cycle_sumcheck_id: SumcheckId,
}

impl<F: JoltField> HammingWeightSumcheckParams<F> {
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    pub fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        if self.sumcheck_id == SumcheckId::RamHammingWeight {
            let (_, hamming_booleanity_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            );
            hamming_booleanity_claim * self.gamma_powers.iter().sum::<F>()
        } else {
            self.gamma_powers.iter().sum()
        }
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        let virtual_poly = self
            .virtual_poly
            .expect("virtual_poly must be set for hamming weight");
        accumulator
            .get_virtual_polynomial_opening(virtual_poly, self.r_cycle_sumcheck_id)
            .0
            .r
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
