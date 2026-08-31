use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims,
    BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_witness::backend::cuda::NarrowColumn;
use jolt_witness::JoltWitnessPlane;

use std::sync::Arc;

use crate::cuda::common::device_columns::DeviceTraceColumns;
use crate::cuda::witness::session_window_residency;

use super::pushforward::{DeviceBytecodePushforward, PushforwardInputs};
use crate::cuda::common::context::{context_for, CudaKernelContext};
use crate::cuda::common::devices::{witness_windows, CycleWindow};
use crate::cuda::common::error::backend;
use crate::cuda::common::one_hot_fold::{DeviceOneHotColumns, OneHotShards};
use crate::cuda::{require_context, CudaBackend};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct BytecodeReadRafAddressKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: BytecodeReadRafAddressPhase<F>,
    state: DeviceBytecodePushforward,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for BytecodeReadRafAddressKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: Field> BytecodeReadRafAddressKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.state
            .bind(self.context, challenge)
            .map_err(backend("cuda bytecode read-RAF address-phase bind"))?;
        self.rounds_bound += 1;
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for BytecodeReadRafAddressKernel<F> {
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
        let (at_zero, at_two) = self.state.round_lanes(self.context).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda bytecode read-RAF address-phase round",
            }
        })?;
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &[at_zero, at_two],
        ))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for BytecodeReadRafAddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &BytecodeReadRafAddressPhaseInputClaims<F>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        if self.state.len() != 1 {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA bytecode read-RAF address phase counted every round but its tables \
                         are not fully bound",
            });
        }
        let readback = || SumcheckKernelError::InvariantViolation {
            reason: "CUDA bytecode read-RAF address phase could not read its fully bound tables",
        };
        let intermediate = self.state.intermediate_claim().map_err(|_| readback())?;
        let val_stages = if self.relation.committed_program() {
            self.state.val_claims().map_err(|_| readback())?
        } else {
            Vec::new()
        };
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate,
            val_stages,
        })
    }
}

fn device_bytecode_pc_words<F: Field>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
    addresses: usize,
) -> Result<NarrowColumn, KernelError<F>> {
    let (trace, columns) = session_window_residency(context, session, witness, cycles, window)?;
    Ok(trace.narrow_u64_column(&columns.bytecode_pc, addresses as u64)?)
}

#[tracing::instrument(skip_all, name = "brap_pc_shards")]
fn bytecode_pc_shards<F: Field>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    addresses: usize,
    log_k: usize,
) -> Result<(OneHotShards, u64), KernelError<F>> {
    let windows = witness_windows(cycles);
    let mut columns = Vec::with_capacity(windows.len());
    let mut entry = None;
    for (ordinal, window) in windows.iter().enumerate() {
        let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
            reason: "a bytecode read-RAF window names an absent device",
        })?;
        let pcs =
            device_bytecode_pc_words::<F>(device, session, witness, cycles, window, addresses)?;
        if ordinal == 0 {
            entry = Some(pcs.first);
        }
        columns.push(DeviceOneHotColumns::from_device(
            DeviceTraceColumns {
                lookup: Arc::new(device.alloc_u64(0)?),
                pc: Arc::new(device.alloc_u32(0)?),
                ram: Arc::new(pcs.column),
            },
            [0, 0, 1],
            log_k,
            window.len,
        )?);
    }
    let entry = entry.ok_or(KernelError::InvariantViolation {
        reason: "the bytecode read-RAF partition produced no window",
    })?;
    Ok((OneHotShards::from_windows(windows, columns)?, entry))
}

impl<F: Field> PrepareKernel<F, BytecodeReadRafAddressPhase<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let addresses = 1usize << dimensions.log_k();
        let cycles = 1usize << dimensions.log_t();
        if relation.register_read_write_point().len() < REGISTER_ADDRESS_BITS
            || relation.register_val_evaluation_point().len() < REGISTER_ADDRESS_BITS
        {
            return Err(KernelError::InvariantViolation {
                reason: "a bytecode read-RAF register point is shorter than its address prefix",
            });
        }
        if relation
            .stage_cycle_points()
            .iter()
            .any(|point| point.len() != dimensions.log_t())
        {
            return Err(KernelError::InvariantViolation {
                reason: "a bytecode read-RAF stage cycle point has the wrong variable count",
            });
        }

        let program = witness.program_preprocessing();
        let stage_gammas = inputs.challenges.stage_gamma_powers();
        let stage_values = tracing::info_span!("brap_stage_values").in_scope(|| {
            read_raf_stage_values(BytecodeReadRafStageValueInputs {
                bytecode: &program.bytecode.bytecode,
                register_read_write_point: &relation.register_read_write_point()
                    [..REGISTER_ADDRESS_BITS],
                register_val_evaluation_point: &relation.register_val_evaluation_point()
                    [..REGISTER_ADDRESS_BITS],
                stage1_gammas: &stage_gammas[0],
                stage2_gammas: &stage_gammas[1],
                stage3_gammas: &stage_gammas[2],
                stage4_gammas: &stage_gammas[3],
                stage5_gammas: &stage_gammas[4],
            })
        });
        if stage_values.len() != addresses {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode stage values".to_owned(),
                expected: addresses,
                got: stage_values.len(),
            });
        }

        let (shards, entry) =
            bytecode_pc_shards::<F>(session, witness, cycles, addresses, dimensions.log_k())?;
        let entry_trace_index =
            usize::try_from(entry).map_err(|_| KernelError::InvariantViolation {
                reason: "the entry bytecode PC exceeds the host word",
            })?;

        let state = DeviceBytecodePushforward::new(
            context,
            &shards,
            PushforwardInputs {
                stage_cycle_points: relation.stage_cycle_points(),
                stage_values: &stage_values,
                entry_trace_index,
                entry_expected_index: relation.entry_bytecode_index(),
                gamma: inputs.challenges.gamma,
            },
        )?;
        drop(shards);

        Ok(Box::new(BytecodeReadRafAddressKernel {
            context,
            relation: relation.clone(),
            state,
            rounds_bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use std::collections::BTreeSet;

    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltRelationId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::{
        BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims, BytecodeStagePoints,
    };
    use jolt_witness::{collect_bundles, JoltWitnessPlane};

    use crate::cuda::booleanity::address::fixture_support::{slot_for_cycle, with_witness, SLOTS};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{fr, probe_input_claim};
    use crate::cuda::CudaBackend;
    use crate::optimized::OptimizedBytecodeReadRafAddress;
    use crate::reference::bytecode_read_raf::BytecodeReadRafWitness;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 9;

    const SEED: u64 = 20_260_816;

    const SAMPLE_POINTS: usize = 4;

    const STAGE_COUNT: usize = 5;

    fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        }
    }

    #[test]
    fn fixture_bytecode_pushforward_is_not_degenerate() {
        with_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let dimensions = formula_dimensions_from_parts(
                one_hot(),
                LOG_T,
                fixture.bytecode.code_size,
                fixture.ram_k,
                JoltRelationId::BytecodeReadRaf,
            )
            .expect("formula dimensions")
            .bytecode_read_raf;
            assert_eq!(
                dimensions.log_t(),
                LOG_T,
                "the read-RAF domain does not span the fixture's cycles",
            );

            let rows: Vec<BytecodeReadRafWitness> =
                collect_bundles(witness as &dyn JoltWitnessPlane<Fr>, 1usize << LOG_T)
                    .expect("bytecode read-RAF bundles");
            let addresses = 1usize << dimensions.log_k();
            let mut touched = BTreeSet::new();
            for (cycle, row) in rows.iter().enumerate() {
                assert!(
                    row.bytecode_pc.0 < addresses,
                    "cycle {cycle}: PC {} escapes the padded bytecode domain",
                    row.bytecode_pc.0,
                );
                assert_eq!(
                    row.bytecode_pc.0,
                    slot_for_cycle(cycle) + 1,
                    "cycle {cycle}: the read-RAF pushforward source disagrees with the fixture's \
                     own slot schedule",
                );
                let _ = touched.insert(row.bytecode_pc.0);
            }
            assert_eq!(
                touched.len(),
                SLOTS,
                "the pushforward reaches {} of {SLOTS} mapped bytecode rows, so some Val rows \
                 carry no cycle weight at all",
                touched.len(),
            );
            assert!(
                touched.len() < addresses,
                "every padded bytecode row is touched, so the fixture cannot detect a kernel \
                 that ignores the pushforward's zeros",
            );

            let entry = fixture
                .bytecode
                .entry_bytecode_index()
                .expect("the fixture bytecode has an entry mapping");
            assert_eq!(
                entry, rows[0].bytecode_pc.0,
                "the entry term is degenerate unless cycle 0 lands on the entry index",
            );

            let mut distinct_values = BTreeSet::new();
            for row in &fixture.bytecode.bytecode[..=SLOTS] {
                let _ = distinct_values.insert((
                    format!("{:?}", row.instruction_kind),
                    row.operands.imm,
                    row.operands.rd,
                    row.operands.rs1,
                    row.operands.rs2,
                ));
            }
            assert!(
                distinct_values.len() > 1,
                "every mapped bytecode row decodes identically, so the per-stage Val tables are \
                 constant and cannot detect a wrong Val bind",
            );
        });
    }

    #[test]
    fn bytecode_read_raf_address_matches_optimized_round_for_round() {
        let Some(_) = shared_context() else {
            return;
        };
        with_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let dimensions = formula_dimensions_from_parts(
                one_hot(),
                LOG_T,
                fixture.bytecode.code_size,
                fixture.ram_k,
                JoltRelationId::BytecodeReadRaf,
            )
            .expect("formula dimensions")
            .bytecode_read_raf;
            let rounds = dimensions.log_k();
            let cycles = 1usize << LOG_T;
            let rows = collect_bundles::<BytecodeReadRafWitness>(
                witness as &dyn JoltWitnessPlane<Fr>,
                cycles,
            )
            .expect("bundle walk");
            let entry_bytecode_index = rows[0].bytecode_pc.0;

            let point = |offset: u64, len: usize| -> Vec<Fr> {
                (0..len)
                    .map(|index| fr(offset * 31 + 7 * index as u64 + 3))
                    .collect()
            };
            let stage_cycle_points: [Vec<Fr>; STAGE_COUNT] =
                core::array::from_fn(|stage| point(stage as u64 + 1, LOG_T));
            let register_point = point(11, REGISTER_ADDRESS_BITS + LOG_T);

            for committed_program in [false, true] {
                let relation = BytecodeReadRafAddressPhase::<Fr>::new(
                    dimensions,
                    committed_program,
                    BytecodeStagePoints {
                        stage_cycle_points: stage_cycle_points.clone(),
                        register_read_write_point: register_point.clone(),
                        register_val_evaluation_point: register_point.clone(),
                    },
                    entry_bytecode_index,
                );
                let claims = BytecodeReadRafAddressPhaseInputClaims::default();
                let points = BytecodeReadRafAddressPhaseInputClaims::default();
                let challenge_set = jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges {
                    gamma: fr(101),
                    stage1_gamma: fr(103),
                    stage2_gamma: fr(107),
                    stage3_gamma: fr(109),
                    stage4_gamma: fr(113),
                    stage5_gamma: fr(127),
                };
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };

                let input_claim = probe_input_claim(
                    &mut *ReferenceBackend
                        .prepare(&mut ProofSession::default(), witness, make_inputs())
                        .expect("reference prepare"),
                );
                let mut expected_kernel = OptimizedBytecodeReadRafAddress
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("optimized prepare");
                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("cuda prepare");

                let challenges: Vec<Fr> =
                    (0..rounds).map(|index| fr(17 * index as u64 + 5)).collect();
                let mut expected_claim = input_claim;
                let mut got_claim = expected_claim;
                let mut bind = None;
                for (round, &challenge) in challenges.iter().enumerate() {
                    let expected = expected_kernel
                        .prove_round(bind, round, expected_claim)
                        .expect("optimized prove_round");
                    let got = got_kernel
                        .prove_round(bind, round, got_claim)
                        .expect("cuda prove_round");
                    for at in 0..SAMPLE_POINTS {
                        let x = Fr::from_u64(at as u64);
                        assert_eq!(
                            got.evaluate(x),
                            expected.evaluate(x),
                            "committed_program {committed_program} round {round} message diverged \
                             at X = {at}",
                        );
                    }
                    expected_claim = expected.evaluate(challenge);
                    got_claim = got.evaluate(challenge);
                    bind = Some(challenge);
                }
                let last = challenges[challenges.len() - 1];
                expected_kernel
                    .finish_rounds(last)
                    .expect("optimized finish_rounds");
                got_kernel.finish_rounds(last).expect("cuda finish_rounds");

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("optimized output claims");
                let got_claims = got_kernel
                    .output_claims(&claims)
                    .expect("cuda output claims");
                assert_eq!(
                    got_claims.intermediate, expected_claims.intermediate,
                    "committed_program {committed_program}: staged claim diverged",
                );
                assert_eq!(
                    got_claims.val_stages, expected_claims.val_stages,
                    "committed_program {committed_program}: per-stage Val claims diverged",
                );
                assert_eq!(
                    got_claims.val_stages.is_empty(),
                    !committed_program,
                    "committed_program {committed_program}: the Val-claim branch did not run",
                );
            }
        });
    }
}
