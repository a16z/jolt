use jolt_claims::protocols::jolt::geometry::ram::ram_ra_claim_reduction;
use jolt_claims::protocols::jolt::relations::ram::{
    RamRaClaimReductionInputClaims, RamRaClaimReductionOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::context::CudaKernelContext;
use super::device::fr_into;
use super::ram_ra_reduction::{CyclePoints, DeviceRamRaReduction};
use super::{require_context, CudaBackend};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct RamRaReductionKernel<F: Field> {
    state: DeviceRamRaReduction,
    relation: RamRaClaimReduction<F>,
    context: &'static CudaKernelContext,
}

impl<F: Field> PrepareKernel<F, RamRaClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRaClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaClaimReduction<F>>>, KernelError<F>> {
        let context = require_context()?;
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let ram_log_k = relation.ram_log_k();
        let points = inputs.points;
        let expected = ram_log_k + log_t;
        for point in [points.raf(), points.read_write(), points.val_check()] {
            if point.len() != expected {
                return Err(KernelError::InvariantViolation {
                    reason: "RAM RA claim-reduction input point has the wrong variable count",
                });
            }
        }
        let r_address = &points.read_write()[..ram_log_k];
        let cycle_points = CyclePoints {
            raf: &points.raf()[ram_log_k..],
            read_write: &points.read_write()[ram_log_k..],
            val_check: &points.val_check()[ram_log_k..],
        };
        let ra_id = ram_ra_claim_reduction().polynomial_id();
        let state = DeviceRamRaReduction::new(
            context,
            &witness.hot_indices(ra_id)?,
            &eq_table(r_address),
            &cycle_points,
            inputs.challenges.gamma,
            log_t,
        )?;
        Ok(Box::new(RamRaReductionKernel {
            state,
            relation: relation.clone(),
            context,
        }))
    }
}

impl<F: Field> ProveRounds<F> for RamRaReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.state
                .bind(self.context, challenge)
                .map_err(|_| SumcheckError::MissingEvaluationSource { kind: "cuda bind" })?;
        }
        let evals: Vec<F> = self.state.round_evals(self.context).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda round_evals",
            }
        })?;
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.state
            .bind(self.context, bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource { kind: "cuda bind" })
    }
}

impl<F: Field> SumcheckKernel<F> for RamRaReductionKernel<F> {
    type Relation = RamRaClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &RamRaClaimReductionInputClaims<F>,
    ) -> Result<RamRaClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.state.rounds_bound();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let claim = self.state.final_claim(self.context).map_err(|_| {
            SumcheckKernelError::InvariantViolation {
                reason: "CUDA RAM RA reduction claim readback failed",
            }
        })?;
        let ram_ra = fr_into(claim).ok_or(SumcheckKernelError::InvariantViolation {
            reason: "CUDA kernels support only the BN254 scalar field",
        })?;
        Ok(RamRaClaimReductionOutputClaims { ram_ra })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::relations::ram::{
        RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial, TraceDimensions};
    use jolt_claims::{OutputClaims, SumcheckChallenges};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::{arb_point, drive, fr, reference_input_claim, FixedPlane};
    use super::CudaBackend;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 6;
    const RAM_LOG_K: usize = 3;

    fn witness(seed: u64) -> FixedPlane {
        let cycles = 1usize << LOG_T;
        let addresses = 1usize << RAM_LOG_K;
        let mut backend = FixedBackend::new();
        let mut ra = vec![Fr::from_u64(0); addresses * cycles];
        for cycle in 0..cycles {
            let address = ((cycle as u64 * 5 + seed) % addresses as u64) as usize;
            ra[address * cycles + cycle] = Fr::from_u64(1);
        }
        backend
            .insert(
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamRa),
                Shape::new(RAM_LOG_K + LOG_T, PolynomialEncoding::Dense),
                ra,
            )
            .expect("insert ram_ra");
        FixedPlane::with_log_t(backend, "cuda ram_ra_claim_reduction fixture", Some(LOG_T))
    }

    proptest! {
        #[test]
        fn ram_ra_claim_reduction_matches_reference(
            seed in any::<u64>(),
            address in arb_point(RAM_LOG_K),
            raf_cycle in arb_point(LOG_T),
            rw_cycle in arb_point(LOG_T),
            val_cycle in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let witness = witness(seed);
            let relation =
                RamRaClaimReduction::<Fr>::new(TraceDimensions::new(LOG_T), RAM_LOG_K);
            let challenge_set =
                RamRaClaimReductionChallenges::from_transcript_values([gamma].into_iter())
                    .expect("challenges");

            let join = |cycle: &[Fr]| {
                let mut point = address.clone();
                point.extend_from_slice(cycle);
                point
            };
            let claims = RamRaClaimReductionInputClaims {
                raf: Fr::from_u64(0),
                read_write: Fr::from_u64(0),
                val_check: Fr::from_u64(0),
            };
            let points = RamRaClaimReductionInputClaims {
                raf: join(&raf_cycle),
                read_write: join(&rw_cycle),
                val_check: join(&val_cycle),
            };
            let make_inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };

            let input_claim = reference_input_claim(&witness, make_inputs);
            let mut expected_kernel = ReferenceBackend
                .prepare(&mut ProofSession::default(), &witness, make_inputs())
                .expect("reference prepare");
            let mut got_kernel = CudaBackend
                .prepare(&mut ProofSession::default(), &witness, make_inputs())
                .expect("cuda prepare");

            let expected = drive(&mut *expected_kernel, input_claim, &challenges);
            let got = drive(&mut *got_kernel, input_claim, &challenges);
            prop_assert_eq!(got, expected);

            let expected_claims = expected_kernel.output_claims(&claims).expect("reference claims");
            let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
            prop_assert_eq!(got_claims.opening_values(), expected_claims.opening_values());
        }
    }
}
