//! Fused increment virtualization sumcheck (lattice/packed mode).
//!
//! Replaces [`IncClaimReduction`](super::increments) when the packed witness
//! carries one fused inc stream: it consumes the same four reduced `Inc`
//! claims but reduces them to the `FusedInc` virtual opening plus the
//! `OpFlags(Store)` destination selector instead of the per-polynomial
//! `RamInc`/`RdInc` openings.
//!
//! Sumcheck (log T rounds, degree 3):
//!
//! ```text
//! Σ_j FusedInc(j) · [eq_ram(j)·Store(j) + γ²·eq_rd(j)·(1 − Store(j))]
//!   = v_1 + γ·v_2 + γ²·w_1 + γ³·w_2
//! eq_ram = eq(r_cycle_stage2, ·) + γ·eq(r_cycle_stage4, ·)
//! eq_rd  = eq(s_cycle_stage4, ·) + γ·eq(s_cycle_stage5, ·)
//! ```
//!
//! The hypercube identities `FusedInc·Store = RamInc` and
//! `FusedInc·(1 − Store) = RdInc` (per-cycle store/rd disjointness) make the
//! claimed sum equal the base reduction's. The summand is a genuine product
//! of three multilinears, so the base instance's bilinear prefix/suffix
//! split does not apply: all four polynomials bind densely from round 0.

#[cfg(feature = "prover")]
use std::sync::Arc;

use allocative::Allocative;
#[cfg(feature = "prover")]
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
#[cfg(feature = "prover")]
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
#[cfg(feature = "prover")]
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
#[cfg(feature = "prover")]
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
#[cfg(feature = "prover")]
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
#[cfg(feature = "prover")]
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::instruction::CircuitFlags;
#[cfg(feature = "prover")]
use crate::zkvm::packed_witness::FusedIncCycle;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct IncVirtualizationSumcheckParams<F: JoltField> {
    /// γ, γ², γ³ for batching
    pub gamma_powers: [F; 3],
    pub n_cycle_vars: usize,
    pub r_cycle_stage2: OpeningPoint<BIG_ENDIAN, F>, // RamInc from RamReadWriteChecking
    pub r_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>, // RamInc from RamValCheck
    pub s_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>, // RdInc from RegistersReadWriteChecking
    pub s_cycle_stage5: OpeningPoint<BIG_ENDIAN, F>, // RdInc from RegistersValEvaluation
}

impl<F: JoltField> IncVirtualizationSumcheckParams<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;

        let (r_cycle_stage2, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_cycle_stage4, _) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
        let (s_cycle_stage4, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (s_cycle_stage5, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );

        Self {
            gamma_powers: [gamma, gamma_sqr, gamma_cub],
            n_cycle_vars: trace_len.log_2(),
            r_cycle_stage2,
            r_cycle_stage4,
            s_cycle_stage4,
            s_cycle_stage5,
        }
    }

    /// `(eq_ram_combined, eq_rd_combined)` at the bound opening point — the
    /// verifier-side closed form, kept for the round-loop test (the live
    /// verification is jolt-verifier's IncVirtualization relation).
    #[cfg(test)]
    fn combined_eq_evals(&self, opening_point: &OpeningPoint<BIG_ENDIAN, F>) -> (F, F) {
        let gamma = self.gamma_powers[0];
        let eq_r2: F = EqPolynomial::mle(&opening_point.r, &self.r_cycle_stage2.r);
        let eq_r4: F = EqPolynomial::mle(&opening_point.r, &self.r_cycle_stage4.r);
        let eq_s4: F = EqPolynomial::mle(&opening_point.r, &self.s_cycle_stage4.r);
        let eq_s5: F = EqPolynomial::mle(&opening_point.r, &self.s_cycle_stage5.r);
        (eq_r2 + gamma * eq_r4, eq_s4 + gamma * eq_s5)
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for IncVirtualizationSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let [gamma, gamma_sqr, gamma_cub] = self.gamma_powers;

        let (_, v_1) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, v_2) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
        let (_, w_1) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, w_2) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );

        v_1 + gamma * v_2 + gamma_sqr * w_1 + gamma_cub * w_2
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        unimplemented!(
            "zk x lattice is rejected fail-closed; IncVirtualization carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; IncVirtualization carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; IncVirtualization carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; IncVirtualization carries no BlindFold plumbing"
        )
    }
}

#[cfg(feature = "prover")]
#[derive(Allocative)]
pub struct IncVirtualizationSumcheckProver<F: JoltField> {
    fused_inc: MultilinearPolynomial<F>,
    store: MultilinearPolynomial<F>,
    /// eq(r_cycle_stage2, ·) + γ·eq(r_cycle_stage4, ·)
    eq_ram: MultilinearPolynomial<F>,
    /// eq(s_cycle_stage4, ·) + γ·eq(s_cycle_stage5, ·)
    eq_rd: MultilinearPolynomial<F>,
    pub params: IncVirtualizationSumcheckParams<F>,
}

#[cfg(feature = "prover")]
impl<F: JoltField> IncVirtualizationSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "IncVirtualizationSumcheckProver::initialize")]
    pub fn initialize(params: IncVirtualizationSumcheckParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        use rayon::prelude::*;

        let gamma = params.gamma_powers[0];
        let (fused, store): (Vec<i128>, Vec<u8>) = trace
            .par_iter()
            .map(|cycle| {
                // One predicate read for both columns — the selector is the
                // same Store flag `from_cycle_with_store` keys the delta off.
                let (fused, store) = FusedIncCycle::from_cycle_with_store(cycle);
                (fused.delta, u8::from(store))
            })
            .unzip();

        let (eq_ram, eq_rd) = rayon::join(
            || {
                let (eq_r2, eq_r4) = rayon::join(
                    || EqPolynomial::<F>::evals(&params.r_cycle_stage2.r),
                    || EqPolynomial::<F>::evals(&params.r_cycle_stage4.r),
                );
                eq_r2
                    .par_iter()
                    .zip(eq_r4.par_iter())
                    .map(|(e2, e4)| *e2 + gamma * e4)
                    .collect::<Vec<F>>()
            },
            || {
                let (eq_s4, eq_s5) = rayon::join(
                    || EqPolynomial::<F>::evals(&params.s_cycle_stage4.r),
                    || EqPolynomial::<F>::evals(&params.s_cycle_stage5.r),
                );
                eq_s4
                    .par_iter()
                    .zip(eq_s5.par_iter())
                    .map(|(e4, e5)| *e4 + gamma * e5)
                    .collect::<Vec<F>>()
            },
        );

        Self {
            fused_inc: fused.into(),
            store: store.into(),
            eq_ram: eq_ram.into(),
            eq_rd: eq_rd.into(),
            params,
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for IncVirtualizationSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "IncVirtualizationSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        use rayon::prelude::*;

        let gamma_sqr = self.params.gamma_powers[1];
        let half_n = self.fused_inc.len() / 2;

        let evals = (0..half_n)
            .into_par_iter()
            .fold(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, j| {
                    let fused = self
                        .fused_inc
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let store = self
                        .store
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let eq_ram = self
                        .eq_ram
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let eq_rd = self
                        .eq_rd
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                    for k in 0..DEGREE_BOUND {
                        acc[k] += fused[k]
                            * (eq_ram[k] * store[k] + gamma_sqr * eq_rd[k] * (F::one() - store[k]));
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut a, b| {
                    for k in 0..DEGREE_BOUND {
                        a[k] += b[k];
                    }
                    a
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "IncVirtualizationSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.fused_inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.store.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_ram.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_rd.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        accumulator.append_virtual(
            VirtualPolynomial::FusedInc,
            SumcheckId::IncVirtualization,
            opening_point.clone(),
            self.fused_inc.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::OpFlags(CircuitFlags::Store),
            SumcheckId::IncVirtualization,
            opening_point,
            self.store.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;
    use ark_std::One;

    type Challenge = <Fr as JoltField>::Challenge;

    const LOG_T: usize = 4;
    const T: usize = 1 << LOG_T;

    fn point(seed: u64) -> OpeningPoint<BIG_ENDIAN, Fr> {
        OpeningPoint::new(
            (0..LOG_T)
                .map(|i| Challenge::from((seed + 3 * i as u64 + 1) as u128))
                .collect(),
        )
    }

    /// Synthetic fused/store columns obeying the per-cycle disjointness, the
    /// implied RamInc/RdInc columns, and the four upstream openings; then the
    /// full round loop: every round polynomial satisfies
    /// `h(0) + h(1) = claim`, and the final claim equals the verifier's
    /// closed-form expected output over the cached openings.
    #[test]
    fn round_loop_reduces_to_the_fused_openings() {
        let fused: Vec<i128> = (0..T as i128)
            .map(|j| (j - 7) * (j + 3) % 97 - 40)
            .collect();
        let store: Vec<u8> = (0..T).map(|j| u8::from(j % 3 == 0)).collect();
        let ram_inc: Vec<i128> = fused
            .iter()
            .zip(&store)
            .map(|(delta, flag)| if *flag == 1 { *delta } else { 0 })
            .collect();
        let rd_inc: Vec<i128> = fused
            .iter()
            .zip(&store)
            .map(|(delta, flag)| if *flag == 0 { *delta } else { 0 })
            .collect();

        let eval = |values: &[i128], r: &OpeningPoint<BIG_ENDIAN, Fr>| -> Fr {
            EqPolynomial::<Fr>::evals(&r.r)
                .iter()
                .zip(values)
                .map(|(eq, value)| *eq * Fr::from_i128(*value))
                .sum()
        };

        let (r2, r4, s4, s5) = (point(2), point(11), point(23), point(31));
        let mut accumulator = ProverOpeningAccumulator::<Fr>::new(LOG_T);
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r2.r.clone(),
            eval(&ram_inc, &r2),
        );
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
            r4.r.clone(),
            eval(&ram_inc, &r4),
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            s4.r.clone(),
            eval(&rd_inc, &s4),
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            s5.r.clone(),
            eval(&rd_inc, &s5),
        );

        let mut transcript = Blake2bTranscript::new(b"inc-virtualization-test");
        let params = IncVirtualizationSumcheckParams::<Fr>::new(T, &accumulator, &mut transcript);

        let gamma = params.gamma_powers[0];
        let (eq_ram, eq_rd) = rayon::join(
            || {
                let eq_r2 = EqPolynomial::<Fr>::evals(&params.r_cycle_stage2.r);
                let eq_r4 = EqPolynomial::<Fr>::evals(&params.r_cycle_stage4.r);
                eq_r2
                    .iter()
                    .zip(&eq_r4)
                    .map(|(e2, e4)| *e2 + gamma * e4)
                    .collect::<Vec<Fr>>()
            },
            || {
                let eq_s4 = EqPolynomial::<Fr>::evals(&params.s_cycle_stage4.r);
                let eq_s5 = EqPolynomial::<Fr>::evals(&params.s_cycle_stage5.r);
                eq_s4
                    .iter()
                    .zip(&eq_s5)
                    .map(|(e4, e5)| *e4 + gamma * e5)
                    .collect::<Vec<Fr>>()
            },
        );
        let mut prover = IncVirtualizationSumcheckProver {
            fused_inc: fused.clone().into(),
            store: store.clone().into(),
            eq_ram: eq_ram.into(),
            eq_rd: eq_rd.into(),
            params: params.clone(),
        };

        let mut claim = params.input_claim(&accumulator);
        let mut challenges = Vec::new();
        for round in 0..LOG_T {
            let message = SumcheckInstanceProver::<Fr, Blake2bTranscript>::compute_message(
                &mut prover,
                round,
                claim,
            );
            assert_eq!(
                message.eval_at_zero() + message.eval_at_one(),
                claim,
                "round {round} message must fold the previous claim"
            );
            let r_j = Challenge::from((100 + 7 * round) as u128);
            claim = message.evaluate(&r_j);
            challenges.push(r_j);
            SumcheckInstanceProver::<Fr, Blake2bTranscript>::ingest_challenge(
                &mut prover,
                r_j,
                round,
            );
        }

        SumcheckInstanceProver::<Fr, Blake2bTranscript>::cache_openings(
            &prover,
            &mut accumulator,
            &challenges,
        );
        let opening_point = params.normalize_opening_point(&challenges);
        let (fused_point, fused_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedInc,
            SumcheckId::IncVirtualization,
        );
        let (_, store_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Store),
            SumcheckId::IncVirtualization,
        );
        assert_eq!(fused_point.r, opening_point.r);

        let (eq_ram_combined, eq_rd_combined) = params.combined_eq_evals(&opening_point);
        assert_eq!(
            claim,
            fused_claim
                * (eq_ram_combined * store_claim
                    + params.gamma_powers[1] * eq_rd_combined * (Fr::one() - store_claim)),
            "final claim must equal the verifier's closed form"
        );
    }
}
