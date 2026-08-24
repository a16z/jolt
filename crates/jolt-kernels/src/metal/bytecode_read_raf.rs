use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::JoltWitnessPlane;

#[cfg(all(test, feature = "parallel"))]
use rayon::prelude::*;

use super::backend::MetalBackend;
use super::solinas::bytecode_read_raf_address::{
    carrier::{ADDRESS_LOG2, INNER_LOG2},
    worklist::BYTECODE_ADDRESS_PUSHFORWARD_STAGES,
    BytecodeAddressSparseStage1Carrier,
};
use super::solinas::{
    BooleanityRows, BytecodeCycleRowInputs, BytecodeCycleRowSequence, BytecodeCycleSequenceConfig,
    BytecodeCycleTablesMut, MetalError,
};
#[cfg(test)]
use crate::optimized::bytecode_read_raf::MetalBytecodeCycleInputs;
use crate::optimized::bytecode_read_raf::{
    prepare_bytecode_read_raf_address, prepare_bytecode_read_raf_address_from_pushforwards,
    prepare_metal_bytecode_cycle_shell, BytecodeCycleAlgebra, BytecodeCycleDenseState, CycleKernel,
    OptimizedBytecodeReadRafCycle,
};
#[cfg(test)]
use crate::optimized::instruction_read_raf::InstructionCycleRow;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BytecodeReadRafAddressImplementation {
    Cpu,
    AddressMajor,
}

impl BytecodeReadRafAddressImplementation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::AddressMajor => "address_major",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafAddressMetalConfig {
    pub implementation: BytecodeReadRafAddressImplementation,
    pub trace_cutoff_elements: usize,
}

impl Default for BytecodeReadRafAddressMetalConfig {
    fn default() -> Self {
        Self {
            implementation: BytecodeReadRafAddressImplementation::Cpu,
            trace_cutoff_elements: 1 << 20,
        }
    }
}

pub(super) fn bytecode_address_major_supported(witness: &dyn JoltWitnessPlane<AkitaField>) -> bool {
    witness.program_preprocessing().bytecode.bytecode.len() == 1usize << ADDRESS_LOG2
}

struct BytecodeAddressEqTables {
    e_lo: Vec<Vec<AkitaField>>,
    e_hi: Vec<Vec<AkitaField>>,
}

fn split_bytecode_address_eq_tables(
    stage_points: &[Vec<AkitaField>],
    log_rows: usize,
    address_elements: usize,
) -> Result<BytecodeAddressEqTables, KernelError<AkitaField>> {
    if stage_points.len() != BYTECODE_ADDRESS_PUSHFORWARD_STAGES {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode address stage count is invalid",
        });
    }
    if log_rows < INNER_LOG2 as usize || address_elements != 1usize << ADDRESS_LOG2 {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode address geometry is invalid",
        });
    }
    let hi_bits = log_rows - INNER_LOG2 as usize;
    let mut e_lo = Vec::with_capacity(stage_points.len());
    let mut e_hi = Vec::with_capacity(stage_points.len());
    for point in stage_points {
        if point.len() != log_rows {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address stage point has the wrong variable count",
            });
        }
        e_hi.push(EqPolynomial::evals(&point[..hi_bits], None));
        e_lo.push(EqPolynomial::evals(&point[hi_bits..], None));
    }
    Ok(BytecodeAddressEqTables { e_lo, e_hi })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: BytecodeCycleSequenceConfig,
    pub cpu_tail_algebra: BytecodeCycleAlgebra,
}

impl Default for BytecodeReadRafMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            cutoff_elements: 1 << 16,
            dispatch: BytecodeCycleSequenceConfig::default(),
            cpu_tail_algebra: BytecodeCycleAlgebra::Q10Accum,
        }
    }
}

#[cfg(test)]
fn exact_bytecode_cycle_input_claim(
    rows: &[InstructionCycleRow],
    inputs: &MetalBytecodeCycleInputs,
    address_elements: usize,
) -> Result<AkitaField, KernelError<AkitaField>> {
    if rows.len() < 4
        || !rows.len().is_power_of_two()
        || inputs.stage_points.len() != 9
        || inputs.stage_weights.len() != 9
        || inputs.ra0.len() != 256
        || inputs.ra1.len() != 256
    {
        return Err(KernelError::InvariantViolation {
            reason: "Bytecode evaluator input-claim geometry is invalid",
        });
    }
    let log_t = rows.len().ilog2() as usize;
    if inputs.stage_points.iter().any(|point| point.len() != log_t) {
        return Err(KernelError::InvariantViolation {
            reason: "Bytecode evaluator stage point has the wrong variable count",
        });
    }
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let lo_length = 1usize << lo_bits;
    let roots = inputs
        .stage_points
        .iter()
        .map(|point| {
            (
                EqPolynomial::<AkitaField>::evals(&point[..hi_bits], None),
                EqPolynomial::<AkitaField>::evals(&point[hi_bits..], None),
            )
        })
        .collect::<Vec<_>>();
    let invalid_pc = |row: &InstructionCycleRow| {
        row.mapped_pc()
            .is_some_and(|mapped_pc| mapped_pc >= address_elements)
    };
    #[cfg(feature = "parallel")]
    let has_invalid_pc = rows.par_iter().any(invalid_pc);
    #[cfg(not(feature = "parallel"))]
    let has_invalid_pc = rows.iter().any(invalid_pc);
    if has_invalid_pc {
        return Err(KernelError::InvariantViolation {
            reason: "Bytecode evaluator mapped PC exceeds the address domain",
        });
    }

    let term = |index: usize| {
        let row = &rows[index];
        let Some(mapped_pc) = row.mapped_pc() else {
            return AkitaField::zero();
        };
        let ra = inputs.ra0[mapped_pc >> 8] * inputs.ra1[mapped_pc & 0xff];
        let hi = index >> lo_bits;
        let lo = index & (lo_length - 1);
        let stage_eq = |stage: usize| roots[stage].0[hi] * roots[stage].1[lo];
        let combined = (0..5).fold(AkitaField::zero(), |sum, stage| {
            sum + inputs.stage_weights[stage] * stage_eq(stage)
        });
        let fused_combined = (5..9).fold(AkitaField::zero(), |sum, stage| {
            sum + inputs.stage_weights[stage] * stage_eq(stage)
        });
        let entry = if index == 0 {
            inputs.entry_weight
        } else {
            AkitaField::zero()
        };
        ra * (combined + row.fused_inc::<AkitaField>() * fused_combined + entry)
    };
    #[cfg(feature = "parallel")]
    let claim = (0..rows.len())
        .into_par_iter()
        .fold(AkitaField::zero, |sum, index| sum + term(index))
        .reduce(AkitaField::zero, |left, right| left + right);
    #[cfg(not(feature = "parallel"))]
    let claim = (0..rows.len()).fold(AkitaField::zero(), |sum, index| sum + term(index));
    Ok(claim)
}

impl PrepareKernel<AkitaField, BytecodeReadRafAddressPhase<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BytecodeReadRafAddressPhase<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafAddressPhase<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let cpu_inputs = || ProverInputs {
            relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let cpu = |session: &mut ProofSession| {
            prepare_bytecode_read_raf_address(session, witness, cpu_inputs())
        };
        let config = self.config.bytecode_read_raf_address;
        let dimensions = relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let route_span = tracing::info_span!(
            "MetalBytecodeReadRafAddress::route",
            cycles = trace_elements,
            requested = config.implementation.as_str(),
            realized_route = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _route_guard = route_span.enter();
        if config.implementation == BytecodeReadRafAddressImplementation::Cpu {
            let _ = route_span.record("realized_route", "cpu");
            let _ = route_span.record("fallback_reason", "configured_cpu");
            return Ok(Box::new(cpu(session)?));
        }

        let address_elements = 1usize << dimensions.log_k();
        if trace_elements < config.trace_cutoff_elements {
            let _ = route_span.record("realized_route", "cpu");
            let _ = route_span.record("fallback_reason", "trace_cutoff");
            return Ok(Box::new(cpu(session)?));
        }
        if address_elements != 1usize << ADDRESS_LOG2 {
            let _ = route_span.record("realized_route", "cpu");
            let _ = route_span.record("fallback_reason", "address_domain");
            return Ok(Box::new(cpu(session)?));
        }
        let _ = route_span.record("realized_route", "address_major_fused_stage1_grouped_v1");
        let _ = route_span.record("fallback_reason", "none");
        let stage_points = relation
            .stage_cycle_points()
            .iter()
            .chain(relation.fused_inc_cycle_points())
            .cloned()
            .collect::<Vec<_>>();
        let prepare_span =
            tracing::info_span!("MetalBytecodeReadRafAddress::address_major_prepare").entered();
        let tables =
            split_bytecode_address_eq_tables(&stage_points, dimensions.log_t(), address_elements)?;
        let carrier = session.take::<BytecodeAddressSparseStage1Carrier>().ok_or(
            KernelError::InvariantViolation {
                reason: "bytecode address-major carrier is missing",
            },
        )?;
        let receipt = carrier.receipt();
        let invocation = self
            .context
            .prepare_bytecode_address_sparse_resident(carrier, &tables.e_lo, &tables.e_hi)
            .map_err(|error| {
                KernelError::Sumcheck(metal_error(format!(
                    "bytecode address-major carrier preparation failed: {error}"
                )))
            })?;
        drop(prepare_span);

        let _join_span =
            tracing::info_span!("MetalBytecodeReadRafAddress::address_major_join").entered();
        let observation = invocation.execute_timed().map_err(|error| {
            KernelError::Sumcheck(metal_error(format!(
                "bytecode address-major completion failed: {error}"
            )))
        })?;
        if observation.receipt != receipt {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address-major source identity changed",
            });
        }
        drop(invocation);
        let expected_fields = stage_points.len().checked_mul(address_elements).ok_or(
            KernelError::InvariantViolation {
                reason: "bytecode address-major output size overflow",
            },
        )?;
        if observation.output.len() != expected_fields {
            return Err(KernelError::TableSizeMismatch {
                table: "Metal bytecode address pushforwards".to_owned(),
                expected: expected_fields,
                got: observation.output.len(),
            });
        }
        let pushforwards = observation
            .output
            .chunks_exact(address_elements)
            .map(<[AkitaField]>::to_vec)
            .collect::<Vec<_>>();
        let prepared = prepare_bytecode_read_raf_address_from_pushforwards(
            witness,
            cpu_inputs(),
            pushforwards,
            receipt.first_push_pc(),
        )?;
        Ok(Box::new(prepared))
    }
}

impl PrepareKernel<AkitaField, BytecodeReadRafCycle<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BytecodeReadRafCycle<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafCycle<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let config = self.config.bytecode_read_raf_cycle;
        let cpu_inputs = || ProverInputs {
            relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let fallback = |session: &mut ProofSession| {
            OptimizedBytecodeReadRafCycle::new(config.cpu_tail_algebra).prepare(
                session,
                witness,
                cpu_inputs(),
            )
        };

        if trace_elements < config.trace_cutoff_elements
            || config.cutoff_elements > trace_elements / 2
            || relation.degree() != 4
            || dimensions.num_committed_ra_polys() != 2
            || relation.committed_chunk_bits() != 8
        {
            return fallback(session);
        }
        let Some(rows) = session.state::<BooleanityRows>().cloned() else {
            return fallback(session);
        };
        if rows.len() != trace_elements || self.context.validate_booleanity_rows(&rows).is_err() {
            return fallback(session);
        }

        let _span = tracing::info_span!("MetalBytecodeReadRafCycle::prepare").entered();
        let (cpu, metadata) =
            prepare_metal_bytecode_cycle_shell(cpu_inputs(), config.cpu_tail_algebra)?;
        let sequence = match self.context.prepare_bytecode_cycle_row_sequence(
            rows,
            BytecodeCycleRowInputs {
                stage_points: &metadata.stage_points,
                stage_weights: &metadata.stage_weights,
                entry_weight: metadata.entry_weight,
                ra0: &metadata.ra0,
                ra1: &metadata.ra1,
            },
            config.dispatch,
        ) {
            Ok(sequence) => sequence,
            Err(error) if bytecode_prepare_can_fallback(&error) => {
                tracing::warn!(
                    error = %error,
                    "bytecode Metal preparation unavailable; using the optimized CPU kernel"
                );
                return fallback(session);
            }
            Err(error) => return Err(metal_error(error.to_string()).into()),
        };
        Ok(Box::new(MetalBytecodeReadRafKernel::new(
            cpu,
            sequence,
            config.cutoff_elements,
        )))
    }
}

pub(crate) struct MetalBytecodeReadRafKernel {
    cpu: CycleKernel<AkitaField>,
    sequence: Option<BytecodeCycleRowSequence>,
    host_tail: Option<[Vec<AkitaField>; 5]>,
    cutoff_elements: usize,
}

impl MetalBytecodeReadRafKernel {
    fn new(
        cpu: CycleKernel<AkitaField>,
        sequence: BytecodeCycleRowSequence,
        cutoff_elements: usize,
    ) -> Self {
        Self {
            cpu,
            sequence: Some(sequence),
            host_tail: Some(std::array::from_fn(|_| {
                vec![AkitaField::zero(); cutoff_elements]
            })),
            cutoff_elements,
        }
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let sequence = self
            .sequence
            .take()
            .ok_or_else(|| metal_error("bytecode cycle sequence disappeared before readback"))?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "bytecode cycle CPU handoff requires materialized factor tables",
            ));
        }
        let elements = sequence.current_elements();
        let readback_bytes = elements
            .checked_mul(5 * std::mem::size_of::<AkitaField>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| metal_error("bytecode cycle readback byte count overflowed"))?;
        let _span = tracing::info_span!(
            "MetalBytecodeReadRafCycle::readback",
            bytes = readback_bytes
        )
        .entered();
        let mut tables = self
            .host_tail
            .take()
            .ok_or_else(|| metal_error("bytecode cycle host tail was already consumed"))?;
        if elements > self.cutoff_elements {
            return Err(metal_error(
                "bytecode cycle readback exceeds the preallocated host tail",
            ));
        }
        let [combined, fused_combined, fused_inc, ra0, ra1] = &mut tables;
        sequence
            .read_current_tables(BytecodeCycleTablesMut {
                combined: &mut combined[..elements],
                fused_combined: &mut fused_combined[..elements],
                fused_inc: &mut fused_inc[..elements],
                ra0: &mut ra0[..elements],
                ra1: &mut ra1[..elements],
            })
            .map_err(|error| metal_error(error.to_string()))?;
        for table in &mut tables {
            table.truncate(elements);
        }
        let [combined, fused_combined, fused_inc, ra0, ra1] = tables;
        self.cpu.metal_restore_dense(BytecodeCycleDenseState {
            combined,
            fused_combined,
            fused_inc,
            ra0,
            ra1,
        })
    }
}

impl ProveRounds<AkitaField> for MetalBytecodeReadRafKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.sequence.as_ref().is_some_and(|sequence| {
            !sequence.is_dense() || sequence.current_elements() > self.cutoff_elements
        }) {
            RoundExecutionDomain::Accelerator
        } else {
            RoundExecutionDomain::Host
        }
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.sequence.as_ref().is_some_and(|sequence| {
            sequence.is_dense() && sequence.current_elements() <= self.cutoff_elements
        }) {
            self.restore_cpu_tail()?;
            let _span = tracing::info_span!("MetalBytecodeReadRafCycle::cpu_tail").entered();
            return self.cpu.prove_round(bind, round, previous_claim);
        }

        if let Some(sequence) = self.sequence.as_mut() {
            let span = match (bind.is_some(), sequence.is_dense()) {
                (false, false) => {
                    tracing::info_span!("MetalBytecodeReadRafCycle::first_message")
                }
                (true, false) => tracing::info_span!("MetalBytecodeReadRafCycle::first_bind"),
                (true, true) => tracing::info_span!("MetalBytecodeReadRafCycle::dense_round"),
                (false, true) => tracing::info_span!("MetalBytecodeReadRafCycle::invalid_round"),
            };
            let _span = span.enter();
            let evals = match bind {
                None => sequence
                    .message()
                    .map_err(|error| metal_error(error.to_string()))?,
                Some(challenge) => {
                    let evals = sequence
                        .bind_and_message(challenge)
                        .map_err(|error| metal_error(error.to_string()))?;
                    self.cpu.metal_commit_bind(sequence.current_elements())?;
                    evals
                }
            };
            return self.cpu.metal_message(evals, previous_claim);
        }

        let _span = tracing::info_span!("MetalBytecodeReadRafCycle::cpu_tail").entered();
        self.cpu.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        let _span = tracing::info_span!("MetalBytecodeReadRafCycle::cpu_tail").entered();
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalBytecodeReadRafKernel {
    type Relation = BytecodeReadRafCycle<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

fn bytecode_prepare_can_fallback(error: &MetalError) -> bool {
    matches!(
        error,
        MetalError::InputTooLong(_)
            | MetalError::BufferTooLong { .. }
            | MetalError::WorkingSetTooLarge { .. }
            | MetalError::FunctionLookup { .. }
            | MetalError::PipelineCompilation { .. }
            | MetalError::UnsupportedBytecodeCycleExecutionWidth { .. }
            | MetalError::InvalidThreadgroupWidth { .. }
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal bytecode parity setup")]
mod tests {
    use std::num::NonZeroUsize;

    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionReadRafChallenges, InstructionReadRafInputClaims,
    };
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_verifier::stages::stage5::InstructionReadRaf;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeStagePoints;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, READ_RAF_CYCLE_STAGES,
    };
    use jolt_witness::testing::with_sample_backend_at_geometry;
    use jolt_witness::witnesses::FusedInc;
    use jolt_witness::ProgramSource;

    use super::*;
    use crate::metal::solinas::bytecode_read_raf_address::BytecodeAddressStage1TopologyOwner;
    use crate::metal::solinas::InstructionReadRafStage1Owner;
    use crate::metal::spartan_product::SpartanProductRemainderMetalConfig;
    use crate::optimized::harness::{probe_input_claim, run_lockstep};
    use crate::optimized::instruction_read_raf::{
        collect_instruction_cycle_rows, InstructionCycleRow,
    };
    use crate::uniskip::UniskipKernel;
    use crate::ReferenceBackend;

    #[test]
    fn bytecode_address_metal_is_opt_in() {
        assert_eq!(
            BytecodeReadRafAddressMetalConfig::default().implementation,
            BytecodeReadRafAddressImplementation::Cpu
        );
    }

    #[test]
    fn unsupported_bytecode_domain_skips_the_stage1_carrier() {
        let log_t = 15;
        with_sample_backend_at_geometry(log_t, 14, 8, |witness| {
            assert!(!bytecode_address_major_supported(witness));
            let backend = MetalBackend::new(super::super::MetalConfig {
                instruction_read_raf: super::super::InstructionReadRafMetalConfig {
                    address_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig {
                    implementation: BytecodeReadRafAddressImplementation::AddressMajor,
                    trace_cutoff_elements: 1 << log_t,
                },
                ..Default::default()
            })
            .unwrap();
            let mut session = ProofSession::default();
            <MetalBackend as UniskipKernel<AkitaField, OuterRemainder<AkitaField>>>::prepare_witness(
                &backend,
                &mut session,
                log_t,
                witness,
            )
            .unwrap();

            assert!(session.state::<InstructionReadRafStage1Owner>().is_some());
            assert!(session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_none());
        });
    }

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn direct_input_claim_rejects_pc_outside_relation_domain() {
        let rows = (0..4)
            .map(|index| {
                InstructionCycleRow::new(
                    0,
                    None,
                    false,
                    Some(if index == 0 { 4 } else { index }),
                    None,
                    FusedInc(0),
                )
            })
            .collect::<Vec<_>>();
        let inputs = MetalBytecodeCycleInputs {
            stage_points: (0..9).map(|stage| point(2, stage)).collect(),
            stage_weights: vec![AkitaField::one(); 9],
            entry_weight: AkitaField::one(),
            ra0: vec![AkitaField::one(); 256],
            ra1: vec![AkitaField::one(); 256],
        };

        assert!(matches!(
            exact_bytecode_cycle_input_claim(&rows, &inputs, 4),
            Err(KernelError::InvariantViolation {
                reason: "Bytecode evaluator mapped PC exceeds the address domain"
            })
        ));
    }

    #[test]
    fn bytecode_address_production_route_matches_the_optimized_host_shell() {
        let log_t = 15;
        with_sample_backend_at_geometry(log_t, 13, 8, |witness| {
            assert_eq!(
                witness.program_preprocessing().bytecode.bytecode.len(),
                1 << 13
            );
            let dimensions = BytecodeReadRafDimensions::new(log_t, 13, 2);
            let relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                true,
                BytecodeStagePoints {
                    stage_cycle_points: std::array::from_fn(|stage| {
                        point(log_t, 11 + stage as u64)
                    }),
                    register_read_write_point: point(7 + log_t, 31),
                    register_val_evaluation_point: point(7 + log_t, 37),
                    fused_inc_cycle_points: (0..4)
                        .map(|stage| point(log_t, 43 + stage as u64))
                        .collect(),
                },
                0,
            );
            let challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: AkitaField::from_u64(3),
                stage1_gamma: AkitaField::from_u64(5),
                stage2_gamma: AkitaField::from_u64(7),
                stage3_gamma: AkitaField::from_u64(11),
                stage4_gamma: AkitaField::from_u64(13),
                stage5_gamma: AkitaField::from_u64(17),
            };
            let claims = LatticeReadRafAddressPhaseInputClaims::<AkitaField>::default();
            let points = LatticeReadRafAddressPhaseInputClaims::<Vec<AkitaField>>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut reference = <ReferenceBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafAddressPhase<AkitaField>,
            >>::prepare(
                &ReferenceBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            let claim = probe_input_claim(reference.as_mut());
            let round_challenges = point(dimensions.log_k(), 101);

            let production = MetalBackend::new(super::super::MetalConfig {
                spartan_product_remainder: SpartanProductRemainderMetalConfig {
                    trace_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                instruction_read_raf: super::super::InstructionReadRafMetalConfig {
                    address_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig {
                    implementation: BytecodeReadRafAddressImplementation::AddressMajor,
                    trace_cutoff_elements: 1 << log_t,
                },
                ..Default::default()
            })
            .unwrap();
            let mut session = ProofSession::default();
            <MetalBackend as UniskipKernel<AkitaField, OuterRemainder<AkitaField>>>::prepare_witness(
                &production,
                &mut session,
                log_t,
                witness,
            )
            .unwrap();
            let source_receipt = session
                .state::<InstructionReadRafStage1Owner>()
                .unwrap()
                .receipt();
            assert_eq!(source_receipt.row_bytes(), 32 * (1 << log_t));
            assert!(session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_some());
            assert!(session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_none());
            let instruction_dimensions = InstructionReadRafDimensions::new(
                log_t,
                2 * RISCV_XLEN,
                NonZeroUsize::new(4).unwrap(),
            );
            let instruction_relation = InstructionReadRaf::new(instruction_dimensions);
            let instruction_claims = InstructionReadRafInputClaims::<AkitaField>::default();
            let instruction_points = InstructionReadRafInputClaims {
                lookup_output: point(log_t, 151),
                left_lookup_operand: point(log_t, 157),
                right_lookup_operand: point(log_t, 163),
            };
            let instruction_challenges = InstructionReadRafChallenges {
                gamma: AkitaField::from_u64(167),
            };
            let instruction_inputs = ProverInputs {
                relation: &instruction_relation,
                claims: &instruction_claims,
                points: &instruction_points,
                challenges: &instruction_challenges,
            };
            let instruction_kernel = <MetalBackend as PrepareKernel<
                AkitaField,
                InstructionReadRaf<AkitaField>,
            >>::prepare(
                &production, &mut session, witness, instruction_inputs
            )
            .unwrap();
            assert!(session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_none());
            assert!(session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_some());
            drop(instruction_kernel);
            let mut production_actual =
                <MetalBackend as PrepareKernel<
                    AkitaField,
                    BytecodeReadRafAddressPhase<AkitaField>,
                >>::prepare(&production, &mut session, witness, inputs())
                .unwrap();
            assert!(session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_none());
            let mut production_expected =
                prepare_bytecode_read_raf_address(&mut ProofSession::default(), witness, inputs())
                    .unwrap();
            run_lockstep(
                &mut production_expected,
                production_actual.as_mut(),
                claim,
                &round_challenges,
            );
            assert_eq!(
                production_expected.output_claims(&claims).unwrap(),
                production_actual.output_claims(&claims).unwrap()
            );
        });
    }

    #[test]
    fn production_kernel_matches_optimized_cpu_through_handoff() {
        let log_t = 10;
        with_sample_backend_at_geometry(log_t, 13, 8, |witness| {
            let dimensions = BytecodeReadRafDimensions::new(log_t, 13, 2);
            let relation = BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                dimensions,
                r_address: point(13, 19),
                stage_cycle_points: std::array::from_fn(|stage| point(log_t, 41 + stage as u64)),
                entry_bytecode_index: 17,
                committed_chunk_bits: 8,
                val_stages: (0..NUM_BYTECODE_VAL_STAGES)
                    .map(|stage| AkitaField::from_u64(101 + stage as u64))
                    .collect(),
            });
            assert_eq!(relation.stage_cycle_points().len(), READ_RAF_CYCLE_STAGES);
            let claims = BytecodeReadRafInputClaims::<AkitaField>::default();
            let points = BytecodeReadRafInputClaims::<Vec<AkitaField>>::default();
            let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let (mut shell, metadata) =
                prepare_metal_bytecode_cycle_shell(inputs(), BytecodeCycleAlgebra::Q10Accum)
                    .unwrap();
            assert_eq!(shell.metal_elements().unwrap(), 1 << log_t);
            assert_eq!(shell.metal_rounds_bound(), 0);
            let _ = shell
                .metal_message(
                    std::array::from_fn(|index| AkitaField::from_u64(7 + index as u64)),
                    AkitaField::zero(),
                )
                .unwrap();
            assert!(shell.metal_commit_bind((1 << log_t) / 4).is_err());
            shell.metal_commit_bind((1 << log_t) / 2).unwrap();
            assert_eq!(shell.metal_rounds_bound(), 1);
            assert_eq!(shell.metal_elements().unwrap(), (1 << log_t) / 2);

            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let claim = exact_bytecode_cycle_input_claim(&packed, &metadata, 1 << 13).unwrap();
            let mut reference = <ReferenceBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafCycle<AkitaField>,
            >>::prepare(
                &ReferenceBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            assert_eq!(claim, probe_input_claim(reference.as_mut()));
            assert_ne!(claim, AkitaField::zero());

            let mut expected = OptimizedBytecodeReadRafCycle::new(BytecodeCycleAlgebra::Q10Accum)
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(super::super::MetalConfig {
                bytecode_read_raf_cycle: BytecodeReadRafMetalConfig {
                    trace_cutoff_elements: 2,
                    cutoff_elements: 4,
                    dispatch: BytecodeCycleSequenceConfig {
                        message_threads_per_threadgroup: Some(32),
                        transition_threads_per_threadgroup: Some(32),
                        max_threadgroups: 1 << 13,
                    },
                    cpu_tail_algebra: BytecodeCycleAlgebra::Q10Accum,
                },
                ..Default::default()
            })
            .unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let mut session = ProofSession::default();
            session.park(resident);
            let mut actual = <MetalBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafCycle<AkitaField>,
            >>::prepare(&metal, &mut session, witness, inputs())
            .unwrap();
            assert!(session.state::<BooleanityRows>().is_some());

            let mut round_challenges = point(log_t, 211);
            round_challenges[0] = AkitaField::zero();
            round_challenges[1] = AkitaField::one();
            round_challenges[2] = -AkitaField::one();
            let mut claim = claim;
            let mut nonzero_round = false;
            for round in 0..log_t {
                let bind = round
                    .checked_sub(1)
                    .map(|previous| round_challenges[previous]);
                let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                assert_eq!(actual_poly, expected_poly, "round {round}");
                nonzero_round |= expected_poly
                    .coefficients()
                    .iter()
                    .any(|coefficient| *coefficient != AkitaField::zero());
                claim = expected_poly.evaluate(round_challenges[round]);
            }
            assert!(nonzero_round, "all round polynomials were zero");
            let final_bind = round_challenges[log_t - 1];
            expected.finish_rounds(final_bind).unwrap();
            actual.finish_rounds(final_bind).unwrap();
            assert_eq!(
                actual.output_claims(&claims).unwrap(),
                expected.output_claims(&claims).unwrap()
            );
        });
    }
}
