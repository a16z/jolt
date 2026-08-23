use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionInputInputClaims, InstructionInputOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::witness::session_window_residency;

use super::{require_context, CudaBackend};
use crate::cuda::common::context::{context_for, CudaKernelContext};
use crate::cuda::common::devices::witness_windows;
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::SumcheckKernelError;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::BindingOrder;

pub(crate) mod columns;
pub(crate) mod rounds;
pub(crate) mod witness;

use columns::{ColumnShard, DeviceInstructionColumns, ShardedInstructionColumns};
use rounds::{Basis, RoundBasis};

pub struct InstructionInputKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: InstructionInput<F>,
    columns: ShardedInstructionColumns<F>,
    basis: Basis<F>,
    rounds_bound: usize,
    finals: Option<Vec<F>>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for InstructionInputKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("finals"),
            self.finals.as_ref().map_or(0, |v| v.len() * size_of::<F>()),
        );
        visitor.exit();
    }
}

impl<F: Field> InstructionInputKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda instruction input-virtualization bind",
        };
        let bound = self.rounds_bound;
        self.columns.bind(challenge, bound).map_err(|_| failed())?;
        self.basis
            .bind(self.context, challenge)
            .map_err(|_| failed())?;
        self.rounds_bound += 1;
        if self.rounds_bound == self.relation.symbolic().rounds() {
            self.finals = Some(self.columns.finals().map_err(|_| failed())?);
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for InstructionInputKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        self.basis
            .round_poly(
                self.context,
                &self.columns,
                previous_claim,
                self.relation.symbolic().degree(),
            )
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda instruction input-virtualization round",
            })
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for InstructionInputKernel<F> {
    type Relation = InstructionInput<F>;

    fn output_claims(
        &mut self,
        _inputs: &InstructionInputInputClaims<F>,
    ) -> Result<InstructionInputOutputClaims<F>, SumcheckKernelError<F>> {
        let rounds = self.relation.symbolic().rounds();
        if self.rounds_bound != rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: rounds.saturating_sub(self.rounds_bound),
            });
        }
        let finals = self
            .finals
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA instruction input-virtualization never read back its bound columns",
            })?;
        let &[left_operand_is_rs1, rs1_value, left_operand_is_pc, unexpanded_pc, right_operand_is_rs2, rs2_value, right_operand_is_imm, imm] =
            finals.as_slice()
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA instruction input-virtualization bound the wrong column count",
            });
        };
        Ok(InstructionInputOutputClaims {
            left_operand_is_rs1,
            rs1_value,
            left_operand_is_pc,
            unexpanded_pc,
            right_operand_is_rs2,
            rs2_value,
            right_operand_is_imm,
            imm,
        })
    }
}

pub(crate) fn prepare_with_basis<F: Field>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    inputs: &ProverInputs<'_, F, InstructionInput<F>>,
    basis: RoundBasis,
) -> Result<InstructionInputKernel<F>, KernelError<F>> {
    let context = require_context()?;
    let relation = inputs.relation;
    let log_t = relation.symbolic().rounds();
    if log_t == 0 || relation.product_remainder_opening_point().len() != log_t {
        return Err(KernelError::InvariantViolation {
            reason: "the instruction-input product-remainder point spans the cycle variables",
        });
    }

    let cycles = 1usize << log_t;
    let point = relation.product_remainder_opening_point();
    let windows = if basis == RoundBasis::Gruen {
        witness_windows(cycles)
    } else {
        vec![crate::cuda::common::devices::CycleWindow {
            start: 0,
            len: cycles,
        }]
    };
    let shards = windows.len();
    let mut column_shards = Vec::with_capacity(shards);
    for (ordinal, window) in windows.iter().enumerate() {
        let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
            reason: "an instruction-input window names an absent device",
        })?;
        let (trace, atoms) = session_window_residency(device, session, witness, cycles, window)?;
        column_shards.push(ColumnShard {
            ordinal,
            columns: DeviceInstructionColumns::from_device(device, &trace, &atoms, window.len)?,
            eq: DeviceSplitEq::new_window(device, point, BindingOrder::LowToHigh, ordinal, shards)?,
            form: rounds::gruen_form(device, inputs.challenges.gamma)?,
        });
    }
    let basis = Basis::new(context, basis, point, inputs.challenges.gamma)?;

    Ok(InstructionInputKernel {
        context,
        relation: relation.clone(),
        columns: ShardedInstructionColumns::new(column_shards, log_t)?,
        basis,
        rounds_bound: 0,
        finals: None,
    })
}

impl<F: Field> PrepareKernel<F, InstructionInput<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        Ok(Box::new(prepare_with_basis(
            session,
            witness,
            &inputs,
            RoundBasis::Gruen,
        )?))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::{
        imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
        rs1_value, rs2_value, unexpanded_pc,
    };
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionInputChallenges, InstructionInputInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltOpeningId, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage3::outputs::InstructionInput;
    use jolt_witness::JoltWitnessPlane;
    use proptest::prelude::*;

    use super::rounds::RoundBasis;
    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_r1cs_witness,
    };
    use crate::reference::views::dense_view;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    fn flag_ids() -> [JoltOpeningId; 4] {
        [
            left_operand_is_rs1(),
            left_operand_is_pc(),
            right_operand_is_rs2(),
            right_operand_is_imm(),
        ]
    }

    fn value_ids() -> [JoltOpeningId; 4] {
        [rs1_value(), unexpanded_pc(), rs2_value(), imm()]
    }

    #[test]
    fn fixture_instruction_input_columns_vary() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let plane: &dyn JoltWitnessPlane<Fr> = witness;
            let zero = Fr::from_u64(0);
            let one = Fr::from_u64(1);
            for id in flag_ids() {
                let column = dense_view::<Fr>(plane, id).expect("the fixture serves the flag");
                assert!(
                    column.contains(&zero) && column.contains(&one),
                    "{id:?} takes one value across the fixture, so dropping its term would not \
                     change the round polynomials",
                );
            }
            for id in value_ids() {
                let column = dense_view::<Fr>(plane, id).expect("the fixture serves the operand");
                assert!(
                    column.iter().any(|value| *value != zero),
                    "{id:?} is zero at every cycle, so its coefficient could be anything",
                );
                assert!(
                    column.iter().any(|value| *value != column[0]),
                    "{id:?} is constant across the fixture, so a mis-indexed read would pass",
                );
            }
        });
    }

    #[test]
    fn device_instruction_columns_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let cycles = 1usize << LOG_T;
            let rows: Vec<super::witness::InstructionInputWitness> =
                jolt_witness::collect_bundles(witness, cycles).expect("reference rows");
            let expected: Vec<Vec<Fr>> = super::columns::DeviceInstructionColumns::new(
                context,
                &super::witness::pack(&rows),
            )
            .expect("host-encoded columns")
            .handles()
            .iter()
            .map(|column| column.to_host().expect("download"))
            .collect();

            let mut session = ProofSession::default();
            let trace = crate::cuda::witness::session_device_trace::<Fr>(
                context,
                &mut session,
                witness,
                cycles,
            )
            .expect("device residency");
            let atoms = crate::cuda::witness::session_atom_columns::<Fr>(
                context,
                &mut session,
                witness,
                cycles,
            )
            .expect("atom columns");
            let got: Vec<Vec<Fr>> = super::columns::DeviceInstructionColumns::from_device(
                context, &trace, &atoms, cycles,
            )
            .expect("device-gathered columns")
            .handles()
            .iter()
            .map(|column| column.to_host().expect("download"))
            .collect();

            assert!(
                expected[1].iter().any(|value| *value != expected[1][0]),
                "every rs1 value is identical, so a kernel ignoring the row would pass",
            );
            assert_eq!(
                got, expected,
                "the device instruction columns diverge from the host encoder",
            );
        });
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn instruction_input_matches_reference_round_for_round(
            seed in any::<u64>(),
            product_remainder_point in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = InstructionInput::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    product_remainder_point.clone(),
                );
                let claims = InstructionInputInputClaims::default();
                let points = InstructionInputInputClaims::default();
                let challenge_set = InstructionInputChallenges { gamma };
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };

                let input_claim = reference_input_claim(witness, make_inputs);
                let mut expected_kernel = ReferenceBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("reference prepare");
                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let expected_claims =
                    expected_kernel.output_claims(&claims).expect("reference claims");

                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("cuda prepare");
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "round polynomials diverged");

                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged"
                );
                Ok(())
            })?;
        }

        #[test]
        fn gruen_basis_matches_eval_point_basis_round_for_round(
            seed in any::<u64>(),
            product_remainder_point in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = InstructionInput::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    product_remainder_point.clone(),
                );
                let claims = InstructionInputInputClaims::default();
                let points = InstructionInputInputClaims::default();
                let challenge_set = InstructionInputChallenges { gamma };
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };

                let input_claim = reference_input_claim(witness, make_inputs);
                let mut session = ProofSession::default();
                let mut expected_kernel = super::prepare_with_basis(
                    &mut session, witness, &make_inputs(), RoundBasis::EvalPoints,
                ).expect("eval-point prepare");
                let mut got_kernel = super::prepare_with_basis(
                    &mut session, witness, &make_inputs(), RoundBasis::Gruen,
                ).expect("gruen prepare");

                let expected = drive(&mut expected_kernel, input_claim, &challenges);
                let got = drive(&mut got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "the two round bases diverged");
                Ok(())
            })?;
        }
    }
}
