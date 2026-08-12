use jolt_claims::protocols::jolt::geometry::ram::{ram_inc_val_check, ram_ra_val_check};
use jolt_claims::protocols::jolt::relations::ram::{
    RamValCheckChallenges, RamValCheckInputClaims, RamValCheckOutputClaims,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::BindingOrder;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
use jolt_witness::JoltWitnessPlane;

use super::{require_context, CudaBackend};
use crate::cuda::common::dense_product::{DenseProductKernel, DeviceDenseProduct};
use crate::cuda::common::lt_poly::DeviceLtPolynomial;
use crate::cuda::common::ra_poly::DeviceRaPolynomial;
use crate::reference::views::{dense_view, eq_table};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

impl<F: Field> PrepareKernel<F, RamValCheck<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
        let context = require_context()?;
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let ram_log_k = relation.ram_log_k();
        let point: &[F] = &inputs.points.ram_val;
        if point.len() != ram_log_k + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM value-check input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = point.split_at(ram_log_k);

        let weights: [(F, Vec<F>); 0] = [];
        let factors = [dense_view(witness, ram_inc_val_check())?];
        let ra_id = ram_ra_val_check().polynomial_id();
        let ra = DeviceRaPolynomial::new(
            context,
            &witness.hot_indices(ra_id)?,
            &eq_table(r_address),
            BindingOrder::LowToHigh,
        )?;
        let lt = DeviceLtPolynomial::shifted(
            context,
            r_cycle,
            BindingOrder::LowToHigh,
            Some(inputs.challenges.gamma),
        )?;
        let state = DeviceDenseProduct::new(
            context,
            &weights,
            &factors,
            Some(ra),
            Some(lt),
            log_t,
            relation.degree(),
        )?;
        Ok(Box::new(DenseProductKernel {
            state,
            relation: relation.clone(),
            context,
            field: core::marker::PhantomData,
        }))
    }
}

impl<F: Field> SumcheckKernel<F> for DenseProductKernel<F, RamValCheck<F>> {
    type Relation = RamValCheck<F>;

    fn output_claims(
        &mut self,
        inputs: &RamValCheckInputClaims<F>,
    ) -> Result<RamValCheckOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.state.rounds_bound();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let finals: Vec<F> =
            self.finals()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA dense-product factor readback failed",
                })?;
        let [ram_inc, ram_ra] = finals.as_slice() else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "RAM value-check expects exactly two bound factors",
            });
        };
        Ok(RamValCheckOutputClaims {
            untrusted_advice: inputs.untrusted_advice,
            trusted_advice: inputs.trusted_advice,
            program_image: inputs.program_image,
            ram_ra: *ram_ra,
            ram_inc: *ram_inc,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &RamValCheck<F>,
        input_points: &RamValCheckInputClaims<Vec<F>>,
        output_points: &RamValCheckOutputClaims<Vec<F>>,
        challenges: &RamValCheckChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let id = JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self
            .state
            .lt_final(self.context)
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "CUDA RAM value-check split-LT readback failed",
            })?
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA RAM value-check has no split-LT factor",
            })?;
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
    use jolt_claims::protocols::jolt::relations::ram::{
        RamValCheckChallenges, RamValCheckInputClaims,
    };
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial, TraceDimensions,
    };
    use jolt_claims::{OutputClaims, SumcheckChallenges};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, drive, fr, reference_input_claim, FixedPlane};
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
        let inc: Vec<Fr> = (0..cycles as u64).map(|c| fr(c + seed)).collect();
        backend
            .insert(
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RamInc),
                Shape::new(LOG_T, PolynomialEncoding::Dense),
                inc,
            )
            .expect("insert ram_inc");
        FixedPlane::with_log_t(backend, "cuda ram_val_check fixture", Some(LOG_T))
    }

    proptest! {
        #[test]
        fn ram_val_check_matches_reference(
            seed in any::<u64>(),
            point in arb_point(RAM_LOG_K + LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let witness = witness(seed);
            let relation = RamValCheck::<Fr>::new(
                TraceDimensions::new(LOG_T),
                RAM_LOG_K,
                RamValCheckInit::full(Fr::from_u64(0)),
            );
            let challenge_set =
                RamValCheckChallenges::from_transcript_values([gamma].into_iter())
                    .expect("challenges");

            let claims = RamValCheckInputClaims {
                ram_val: Fr::from_u64(0),
                ram_val_final: Fr::from_u64(0),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let points = RamValCheckInputClaims {
                ram_val: point.clone(),
                ram_val_final: point.clone(),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
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
