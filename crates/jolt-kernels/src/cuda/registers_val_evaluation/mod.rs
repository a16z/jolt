use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::relations::registers::{
    RegistersValEvaluationInputClaims, RegistersValEvaluationOutputClaims,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersValEvaluationPublic};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::common::trace_columns::cached_bundles;

use super::{require_context, CudaBackend};
use crate::cuda::common::dense_product::{DenseProductKernel, DeviceDenseProduct};
use crate::cuda::common::device::require_fr_slice;
use crate::cuda::common::lt_poly::DeviceLtPolynomial;
use crate::cuda::common::ra_poly::DeviceRaPolynomial;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_poly::BindingOrder;
use jolt_verifier::stages::relations::ConcreteSumcheck;

pub(crate) mod witness;

impl<F: Field> PrepareKernel<F, RegistersValEvaluation<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersValEvaluation<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let point: &[F] = &inputs.points.registers_val;
        if point.len() != REGISTER_ADDRESS_BITS + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = point.split_at(REGISTER_ADDRESS_BITS);

        let cycles = 1usize << log_t;
        let registers = 1usize << REGISTER_ADDRESS_BITS;
        let rows =
            cached_bundles::<witness::RegistersValEvaluationWitness, _>(session, witness, cycles)?;
        let (inc, hot) = witness::device_columns(context, &rows, registers)?;
        drop(rows);

        let eq_address = context.eq_evals(require_fr_slice(r_address)?)?;
        let wa = DeviceRaPolynomial::from_device_tables(
            context,
            &hot,
            eq_address,
            BindingOrder::LowToHigh,
        )?;
        let lt = DeviceLtPolynomial::new(context, r_cycle, BindingOrder::LowToHigh)?;
        let state = DeviceDenseProduct::from_device_factors(
            None,
            vec![inc],
            Some(wa),
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

impl<F: Field> SumcheckKernel<F> for DenseProductKernel<F, RegistersValEvaluation<F>> {
    type Relation = RegistersValEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &RegistersValEvaluationInputClaims<F>,
    ) -> Result<RegistersValEvaluationOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.state.rounds_bound();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let finals: Vec<F> =
            self.finals()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA dense-product factor readback failed",
                })?;
        let [rd_inc, rd_wa] = finals.as_slice() else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers value-evaluation expects exactly two bound factors",
            });
        };
        Ok(RegistersValEvaluationOutputClaims {
            rd_inc: *rd_inc,
            rd_wa: *rd_wa,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &RegistersValEvaluation<F>,
        input_points: &RegistersValEvaluationInputClaims<Vec<F>>,
        output_points: &RegistersValEvaluationOutputClaims<Vec<F>>,
        challenges: &NoChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let id = JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self
            .state
            .lt_final(self.context)
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers value-evaluation split-LT readback failed",
            })?
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers value-evaluation has no split-LT factor",
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
    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    use jolt_claims::protocols::jolt::geometry::registers::rd_wa_val_evaluation;
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
    use jolt_witness::{collect_bundles, JoltWitnessPlane, OneHotSource};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};
    use proptest::prelude::*;

    use super::{witness::RegistersValEvaluationWitness, CudaBackend};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, reference_input_claim, register_rows, RowPlane, REGISTER_ACTIVITY,
    };
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};
    use jolt_claims::protocols::jolt::TraceDimensions;

    const LOG_T: usize = 5;

    fn witness(seed: u64) -> RowPlane {
        let fixture = register_rows(LOG_T, REGISTER_ADDRESS_BITS, seed);
        let mut backend = FixedBackend::new();
        backend
            .insert(
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RdWa),
                Shape::new(REGISTER_ADDRESS_BITS + LOG_T, PolynomialEncoding::Dense),
                fixture.rd_wa,
            )
            .expect("insert rd_wa");
        backend
            .insert(
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
                Shape::new(LOG_T, PolynomialEncoding::Dense),
                fixture.inc,
            )
            .expect("insert rd_inc");
        RowPlane::new(
            backend,
            "cuda registers_val_evaluation fixture",
            LOG_T,
            fixture.rows,
        )
    }

    #[test]
    fn fixture_bundles_match_the_one_hot_oracle_and_leave_rd_idle() {
        let cycles = 1usize << LOG_T;
        for seed in 0..REGISTER_ACTIVITY.len() as u64 {
            let plane = witness(seed);
            let rows = collect_bundles::<RegistersValEvaluationWitness>(
                &plane as &dyn JoltWitnessPlane<Fr>,
                cycles,
            )
            .expect("bundle walk");
            let expected = plane
                .hot_indices(rd_wa_val_evaluation().polynomial_id())
                .expect("oracle hot indices");
            assert_eq!(rows.len(), cycles, "bundle row count");
            for (cycle, row) in rows.iter().enumerate() {
                assert_eq!(
                    row.address.0.map(usize::from),
                    expected[cycle],
                    "seed {seed} cycle {cycle}: the bundle address disagrees with hot_indices",
                );
            }
            let idle = rows.iter().filter(|row| row.address.0.is_none()).count();
            assert!(
                idle > 0 && idle < cycles,
                "seed {seed}: {idle} of {cycles} cycles leave rd unwritten, so one of the two \
                 paths is unexercised",
            );
            let distinct: std::collections::BTreeSet<u8> =
                rows.iter().filter_map(|row| row.address.0).collect();
            assert!(
                distinct.len() > 1,
                "seed {seed}: every rd write targets the same register, so the address column \
                 cannot detect a wrong index",
            );
        }
    }

    proptest! {
        #[test]
        fn registers_val_evaluation_matches_reference(
            seed in any::<u64>(),
            point in arb_point(REGISTER_ADDRESS_BITS + LOG_T),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let witness = witness(seed);
            let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(LOG_T));
            let challenge_set = NoChallenges::default();

            let claims = RegistersValEvaluationInputClaims {
                registers_val: Fr::from_u64(0),
            };
            let points = RegistersValEvaluationInputClaims {
                registers_val: point.clone(),
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
