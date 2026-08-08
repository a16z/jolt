use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6b::bytecode_read_raf::{
    BytecodeReadRafCycle, BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
};
use jolt_witness::JoltWitnessPlane;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::backend::MetalBackend;
use super::solinas::bytecode_read_raf::{
    split_stage_eq_tables, BytecodeReadRafConfig, BytecodeReadRafFusedProductPath,
    BytecodeReadRafShape,
};
use super::solinas::bytecode_read_raf_address::{
    BytecodeAddressMajorConfig, BytecodeAddressMajorResidentCarrier,
};
use super::solinas::{
    BooleanityRows, BytecodeCycleRowInputs, BytecodeCycleRowSequence, BytecodeCycleSequenceConfig,
    BytecodeCycleTablesMut, MetalError,
};
use crate::optimized::bytecode_read_raf::{
    prepare_bytecode_read_raf_address, prepare_bytecode_read_raf_address_from_pushforwards,
    prepare_metal_bytecode_cycle_shell, AddressKernel, BytecodeCycleAlgebra,
    BytecodeCycleDenseState, CycleKernel, MetalBytecodeCycleInputs, OptimizedBytecodeReadRafCycle,
};
use crate::optimized::instruction_read_raf::{collect_instruction_cycle_rows, InstructionCycleRow};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BytecodeReadRafAddressImplementation {
    Cpu,
    CsrShadow,
    AddressMajorShadow,
    AddressMajor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafAddressMetalConfig {
    pub implementation: BytecodeReadRafAddressImplementation,
    pub dispatch: BytecodeReadRafConfig,
    pub product_path: BytecodeReadRafFusedProductPath,
    pub address_major: BytecodeAddressMajorConfig,
}

impl Default for BytecodeReadRafAddressMetalConfig {
    fn default() -> Self {
        Self {
            implementation: BytecodeReadRafAddressImplementation::Cpu,
            dispatch: BytecodeReadRafConfig::default(),
            product_path: BytecodeReadRafFusedProductPath::FullWidth,
            address_major: BytecodeAddressMajorConfig::default(),
        }
    }
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

#[doc(hidden)]
#[derive(Clone)]
pub struct BytecodeReadRafResidentRows(BooleanityRows);

impl BytecodeReadRafResidentRows {
    #[doc(hidden)]
    pub fn install(&self, session: &mut ProofSession) {
        session.park(self.0.clone());
    }
}

impl MetalBackend {
    #[doc(hidden)]
    pub fn bytecode_read_raf_input_claim(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        relation: &BytecodeReadRafCycle<AkitaField>,
        challenges: &BytecodeReadRafCyclePhaseCommittedChallenges<AkitaField>,
    ) -> Result<AkitaField, KernelError<AkitaField>> {
        let trace_elements = 1usize
            .checked_shl(relation.dimensions().log_t() as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Bytecode evaluator trace domain overflows usize",
            })?;
        let rows = collect_instruction_cycle_rows(witness, trace_elements)?;
        let claims = BytecodeReadRafInputClaims::default();
        let points = BytecodeReadRafInputClaims::default();
        let (_, metadata) = prepare_metal_bytecode_cycle_shell(
            ProverInputs {
                relation,
                claims: &claims,
                points: &points,
                challenges,
            },
            BytecodeCycleAlgebra::Q10,
        )?;
        let address_elements = 1usize
            .checked_shl(relation.dimensions().log_k() as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Bytecode evaluator address domain overflows usize",
            })?;
        exact_bytecode_cycle_input_claim(&rows, &metadata, address_elements)
    }

    #[doc(hidden)]
    pub fn prepare_bytecode_read_raf_resident_rows(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        trace_elements: usize,
    ) -> Result<BytecodeReadRafResidentRows, KernelError<AkitaField>> {
        if trace_elements < 4 || !trace_elements.is_power_of_two() {
            return Err(KernelError::InvariantViolation {
                reason: "Bytecode evaluator trace length must be a power of two of at least four",
            });
        }
        let rows = collect_instruction_cycle_rows(witness, trace_elements)?;
        let resident_rows = self
            .context
            .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&rows))
            .map_err(|error| KernelError::from(metal_error(error.to_string())))?;
        Ok(BytecodeReadRafResidentRows(resident_rows))
    }
}

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
        if config.implementation == BytecodeReadRafAddressImplementation::Cpu {
            return Ok(Box::new(cpu(session)?));
        }

        let dimensions = relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let address_elements = 1usize << dimensions.log_k();
        if trace_elements < config.dispatch.trace_cutoff {
            return Ok(Box::new(cpu(session)?));
        }
        let stage_points = relation
            .stage_cycle_points()
            .iter()
            .chain(relation.fused_inc_cycle_points())
            .cloned()
            .collect::<Vec<_>>();
        let prepare_span =
            if config.implementation == BytecodeReadRafAddressImplementation::AddressMajor {
                tracing::info_span!("MetalBytecodeReadRafAddress::address_major_prepare")
            } else {
                tracing::info_span!("MetalBytecodeReadRafAddress::shadow_prepare")
            }
            .entered();
        let shape = if config.implementation == BytecodeReadRafAddressImplementation::AddressMajor {
            BytecodeReadRafShape::new(trace_elements, address_elements).map_err(|error| {
                KernelError::Sumcheck(metal_error(format!(
                    "bytecode address-major shape is invalid: {error}"
                )))
            })?
        } else {
            match config.dispatch.validate(trace_elements, address_elements) {
                Ok(shape) => shape,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "bytecode address CSR shape unavailable; using the optimized CPU kernel"
                    );
                    return Ok(Box::new(cpu(session)?));
                }
            }
        };
        let tables = split_stage_eq_tables(&stage_points, shape).map_err(|error| {
            KernelError::Sumcheck(metal_error(format!(
                "bytecode address equality-table preparation failed: {error}"
            )))
        })?;
        if config.implementation == BytecodeReadRafAddressImplementation::AddressMajor {
            let carrier = session
                .take::<BytecodeAddressMajorResidentCarrier>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "bytecode address-major carrier is missing",
                })?;
            let receipt = carrier.receipt();
            let storage = self
                .context
                .prepare_bytecode_address_major_resident_carrier(
                    carrier,
                    &tables.e_lo,
                    &tables.e_hi,
                    config.address_major,
                )
                .map_err(|error| {
                    KernelError::Sumcheck(metal_error(format!(
                        "bytecode address-major carrier preparation failed: {error}"
                    )))
                })?;
            let storage_stats = storage.storage();
            let pending = storage.submit().map_err(|error| {
                KernelError::Sumcheck(metal_error(format!(
                    "bytecode address-major submission failed: {error}"
                )))
            })?;
            drop(prepare_span);

            let _join_span =
                tracing::info_span!("MetalBytecodeReadRafAddress::address_major_join").entered();
            let (_, observation) = pending.join().map_err(|error| {
                KernelError::Sumcheck(metal_error(format!(
                    "bytecode address-major completion failed: {error}"
                )))
            })?;
            let producer = receipt.producer();
            if observation.source_rows_storage_id != Some(producer.source_allocation_identity())
                || observation.source_rows_device_registry_id != Some(producer.device_registry_id())
            {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode address-major source identity changed",
                });
            }
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
            tracing::info!(
                target: "jolt::metal",
                submit_ns = observation.submit_wall.as_nanos() as u64,
                overlap_ns = observation.overlap_wall.as_nanos() as u64,
                join_ns = observation.join_wall.as_nanos() as u64,
                total_ns = observation.total_wall.as_nanos() as u64,
                gpu_active_ns = observation.gpu_active.as_nanos() as u64,
                completed_before_join = observation.completed_before_join,
                source_generation = producer.generation(),
                source_rows_storage_id = producer.source_allocation_identity(),
                source_device_registry_id = producer.device_registry_id(),
                carrier_cells_storage_id = receipt.cells().allocation_identity(),
                carrier_inner_sign_storage_id = receipt.inner_sign().allocation_identity(),
                carrier_magnitude_storage_id = receipt.magnitude().allocation_identity(),
                carrier_bytes = storage_stats.carrier_bytes as u64,
                member_carrier_owned_bytes = 0u64,
                member_source_scans = 0u64,
                member_source_upload_bytes = 0u64,
                "Metal bytecode address-major carrier produced authoritative pushforwards"
            );
            return Ok(Box::new(prepared));
        }

        let Some(rows) = session.state::<BooleanityRows>().cloned() else {
            return Ok(Box::new(cpu(session)?));
        };
        if rows.len() != trace_elements || self.context.validate_booleanity_rows(&rows).is_err() {
            return Ok(Box::new(cpu(session)?));
        }
        if config.implementation == BytecodeReadRafAddressImplementation::AddressMajorShadow {
            let invocation = match self.context.prepare_bytecode_address_major_resident_shadow(
                rows.clone(),
                &tables.e_lo,
                &tables.e_hi,
                config.address_major,
            ) {
                Ok(invocation) => invocation,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "bytecode address-major preparation unavailable; using the optimized CPU kernel"
                    );
                    return Ok(Box::new(cpu(session)?));
                }
            };
            let producer_support_bytes = invocation.storage().producer_support_bytes;
            let resident_rows_storage_id = rows.allocation_identity();
            let resident_rows_device_registry_id = rows.device_registry_id();
            let pending = match invocation.submit() {
                Ok(pending) => pending,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "bytecode address-major submission unavailable; using the optimized CPU kernel"
                    );
                    return Ok(Box::new(cpu(session)?));
                }
            };
            drop(prepare_span);

            let prepared_cpu = cpu(session)?;
            let _join_span =
                tracing::info_span!("MetalBytecodeReadRafAddress::address_major_join").entered();
            let (_, observation) = match pending.join() {
                Ok(completed) => completed,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "bytecode address-major completion failed; retaining optimized CPU output"
                    );
                    return Ok(Box::new(prepared_cpu));
                }
            };
            if observation.source_rows_storage_id != Some(resident_rows_storage_id)
                || observation.source_rows_device_registry_id
                    != Some(resident_rows_device_registry_id)
            {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode address-major source identity changed",
                });
            }
            validate_bytecode_address_pushforwards(
                &prepared_cpu,
                address_elements,
                &observation.output,
                "bytecode address-major",
            )?;
            let producer_status =
                observation
                    .producer_status
                    .ok_or(KernelError::InvariantViolation {
                        reason: "bytecode address-major producer status is missing",
                    })?;
            tracing::info!(
                target: "jolt::metal",
                submit_ns = observation.submit_wall.as_nanos() as u64,
                overlap_ns = observation.overlap_wall.as_nanos() as u64,
                join_ns = observation.join_wall.as_nanos() as u64,
                total_ns = observation.total_wall.as_nanos() as u64,
                gpu_active_ns = observation.gpu_active.as_nanos() as u64,
                completed_before_join = observation.completed_before_join,
                completed_outer_blocks = producer_status.completed_outer_blocks,
                emitted_rows = producer_status.emitted_rows,
                max_active_addresses = observation.max_active_addresses.unwrap_or(0) as u64,
                producer_threadgroup_bytes =
                    observation.producer_threadgroup_bytes.unwrap_or(0) as u64,
                resident_rows_storage_id,
                row_allocations = 0u64,
                row_upload_bytes = 0u64,
                carrier_upload_bytes = producer_support_bytes as u64,
                "Metal bytecode address-major resident shadow matched the optimized CPU tables"
            );
            return Ok(Box::new(prepared_cpu));
        }
        let invocation = match self.context.prepare_bytecode_read_raf_csr(
            rows.clone(),
            &tables,
            config.dispatch,
            config.product_path,
        ) {
            Ok(invocation) => invocation,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "bytecode address CSR preparation unavailable; using the optimized CPU kernel"
                );
                return Ok(Box::new(cpu(session)?));
            }
        };
        let resident_rows_storage_id = rows.allocation_identity();
        let pending = match invocation.submit() {
            Ok(pending) => pending,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "bytecode address CSR submission unavailable; using the optimized CPU kernel"
                );
                return Ok(Box::new(cpu(session)?));
            }
        };
        drop(prepare_span);

        let prepared_cpu = cpu(session)?;
        let _join_span = tracing::info_span!("MetalBytecodeReadRafAddress::shadow_join").entered();
        let (_, observation) = match pending.join() {
            Ok(completed) => completed,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "bytecode address CSR completion failed; retaining optimized CPU output"
                );
                return Ok(Box::new(prepared_cpu));
            }
        };
        if observation.source_rows_storage_id != resident_rows_storage_id
            || observation.source_rows_device_registry_id != rows.device_registry_id()
        {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address CSR source identity changed",
            });
        }
        validate_bytecode_address_pushforwards(
            &prepared_cpu,
            address_elements,
            &observation.output,
            "bytecode address CSR",
        )?;
        tracing::info!(
            target: "jolt::metal",
            submit_ns = observation.submit_wall.as_nanos() as u64,
            overlap_ns = observation.overlap_wall.as_nanos() as u64,
            join_ns = observation.join_wall.as_nanos() as u64,
            total_ns = observation.total_wall.as_nanos() as u64,
            gpu_active_ns = observation.gpu_active.as_nanos() as u64,
            completed_before_join = observation.completed_before_join,
            short_runs = observation.telemetry.status.short_runs,
            long_runs = observation.telemetry.status.long_runs,
            short_occurrences = observation.telemetry.diagnostics.short_occurrences,
            long_occurrences = observation.telemetry.diagnostics.long_occurrences,
            maximum_run = observation.telemetry.diagnostics.maximum_run,
            resident_rows_storage_id,
            row_allocations = 0u64,
            row_upload_bytes = 0u64,
            "Metal bytecode address CSR shadow matched the optimized CPU tables"
        );
        Ok(Box::new(prepared_cpu))
    }
}

fn validate_bytecode_address_pushforwards(
    cpu: &AddressKernel<AkitaField>,
    address_elements: usize,
    gpu_output: &[AkitaField],
    implementation: &'static str,
) -> Result<(), KernelError<AkitaField>> {
    let expected_output_elements = cpu
        .pushforward_tables()
        .len()
        .checked_mul(address_elements)
        .ok_or(KernelError::InvariantViolation {
            reason: "bytecode address pushforward size overflow",
        })?;
    if gpu_output.len() != expected_output_elements {
        return Err(KernelError::TableSizeMismatch {
            table: "Metal bytecode address pushforwards".to_owned(),
            expected: expected_output_elements,
            got: gpu_output.len(),
        });
    }
    for (stage, cpu_table) in cpu.pushforward_tables().enumerate() {
        let start = stage * address_elements;
        let gpu_table = &gpu_output[start..start + address_elements];
        if let Some(index) = cpu_table
            .iter()
            .zip(gpu_table)
            .position(|(cpu, gpu)| cpu != gpu)
        {
            tracing::error!(
                stage,
                index,
                implementation,
                "bytecode address parity mismatch"
            );
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address pushforward mismatch",
            });
        }
    }
    Ok(())
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
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
    use jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeStagePoints;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, READ_RAF_CYCLE_STAGES,
    };
    use jolt_witness::testing::with_sample_backend_at_geometry;
    use jolt_witness::witnesses::FusedInc;
    use jolt_witness::ProgramSource;

    use super::*;
    use crate::optimized::harness::{probe_input_claim, run_lockstep};
    use crate::optimized::instruction_read_raf::{
        collect_instruction_cycle_rows, InstructionCycleRow,
    };
    use crate::uniskip::UniskipKernel;
    use crate::ReferenceBackend;

    #[test]
    fn bytecode_address_shadow_is_opt_in() {
        assert_eq!(
            BytecodeReadRafAddressMetalConfig::default().implementation,
            BytecodeReadRafAddressImplementation::Cpu
        );
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
    fn bytecode_address_metal_routes_match_the_optimized_host_shell() {
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
            let mut expected =
                prepare_bytecode_read_raf_address(&mut ProofSession::default(), witness, inputs())
                    .unwrap();
            let metal = MetalBackend::new(super::super::MetalConfig {
                bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig {
                    implementation: BytecodeReadRafAddressImplementation::AddressMajorShadow,
                    dispatch: BytecodeReadRafConfig {
                        trace_cutoff: 1 << log_t,
                        ..Default::default()
                    },
                    address_major: BytecodeAddressMajorConfig { outer_tiles: 1 },
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let mut active_addresses = packed
                .iter()
                .map(|row| row.mapped_pc().unwrap_or(0) as u32)
                .collect::<Vec<_>>();
            active_addresses.sort_unstable();
            active_addresses.dedup();
            let support_offsets = [0, active_addresses.len() as u32];
            let resident = metal
                .context
                .prepare_booleanity_rows_with_bytecode_support(
                    InstructionCycleRow::metal_booleanity_rows(&packed),
                    &support_offsets,
                    &active_addresses,
                )
                .unwrap();
            let mut session = ProofSession::default();
            session.park(resident);
            let mut actual = <MetalBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafAddressPhase<AkitaField>,
            >>::prepare(&metal, &mut session, witness, inputs())
            .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let round_challenges = point(dimensions.log_k(), 101);
            run_lockstep(&mut expected, actual.as_mut(), claim, &round_challenges);
            assert_eq!(
                expected.output_claims(&claims).unwrap(),
                actual.output_claims(&claims).unwrap()
            );

            let production = MetalBackend::new(super::super::MetalConfig {
                instruction_read_raf: super::super::InstructionReadRafMetalConfig {
                    address_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig {
                    implementation: BytecodeReadRafAddressImplementation::AddressMajor,
                    dispatch: BytecodeReadRafConfig {
                        trace_cutoff: 1 << log_t,
                        ..Default::default()
                    },
                    address_major: BytecodeAddressMajorConfig { outer_tiles: 1 },
                    ..Default::default()
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
            assert!(session
                .state::<BytecodeAddressMajorResidentCarrier>()
                .is_some());
            let mut production_actual =
                <MetalBackend as PrepareKernel<
                    AkitaField,
                    BytecodeReadRafAddressPhase<AkitaField>,
                >>::prepare(&production, &mut session, witness, inputs())
                .unwrap();
            assert!(session
                .state::<BytecodeAddressMajorResidentCarrier>()
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
