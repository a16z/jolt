use std::{mem::size_of, slice, time::Duration};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::super::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, Fp128, InstructionInputRow,
    MetalError, PipelineLimits, SolinasMetal, SpartanRawRow,
};
use super::{
    RegistersClaimGeometry, RegistersClaimKernelConfig, RegistersClaimPlanError,
    ALIAS_FOLD_COMPACT_ROWS_SLOT, ALIAS_FOLD_EQ_PREFIX_SLOT, ALIAS_FOLD_OUTPUT_SLOT,
    ALIAS_FOLD_PARAMS_SLOT, ALIAS_FOLD_PIPELINE, ALIAS_FOLD_RAW_ROWS_SLOT, ALIAS_FOLD_RD_POST_SLOT,
    ALIAS_FOLD_THREADGROUP_SLOT, REGISTERS_CLAIM_AKITA_OFFSET, REGISTERS_CLAIM_SIMD_WIDTH,
};

#[derive(Debug, Error)]
pub enum RegistersClaimError {
    #[error(transparent)]
    Plan(#[from] RegistersClaimPlanError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("registers claim reduction requires Akita offset {expected:#x}, got {got:#x}")]
    UnsupportedOffset { expected: u32, got: u32 },
    #[error("registers claim prefix has {actual} challenges, expected {expected}")]
    WrongPrefixChallengeCount { expected: usize, actual: usize },
    #[error("{name} buffer belongs to Metal device {got}, expected {expected}")]
    BufferDevice {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("{name} buffer has {actual} bytes, expected {expected}")]
    BufferLength {
        name: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("registers claim buffers alias across read and write bindings")]
    AliasedInvocationBuffers,
    #[error(
        "{pipeline} has execution width {got}, expected {expected} for the registers claim ABI"
    )]
    UnsupportedExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "registers claim alias fold needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    ThreadgroupMemory { requested: u64, maximum: u64 },
    #[error("invalid registers claim state: {0}")]
    InvalidState(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentRdMetadata {
    geometry: RegistersClaimGeometry,
    plane_bytes: u64,
    device_registry_id: u64,
    allocation_identity: usize,
    source_generation: u64,
    completion_serial: u64,
}

#[derive(Clone)]
pub(crate) struct RegistersClaimResidentRdPlane {
    buffer: Buffer,
    metadata: ResidentRdMetadata,
}

impl RegistersClaimResidentRdPlane {
    pub(crate) const fn geometry(&self) -> RegistersClaimGeometry {
        self.metadata.geometry
    }

    pub(crate) const fn device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id
    }

    pub(crate) const fn allocation_identity(&self) -> usize {
        self.metadata.allocation_identity
    }

    pub(crate) const fn source_generation(&self) -> u64 {
        self.metadata.source_generation
    }

    pub(crate) const fn resident_bytes(&self) -> u64 {
        self.metadata.plane_bytes
    }

    pub(crate) const fn completion_serial(&self) -> u64 {
        self.metadata.completion_serial
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub(crate) fn validate_for(
        &self,
        context: &SolinasMetal,
        geometry: RegistersClaimGeometry,
    ) -> Result<(), RegistersClaimError> {
        if self.metadata.geometry != geometry
            || self.metadata.source_generation == 0
            || self.metadata.completion_serial == 0
        {
            return Err(RegistersClaimError::InvalidState(
                "resident rd receipt differs from the invocation",
            ));
        }
        validate_buffer_binding(
            &self.buffer,
            "resident rd_write_value",
            self.metadata.plane_bytes,
            context.device_registry_id(),
            self.metadata.allocation_identity,
        )
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RegistersClaimResidentRdPlane {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple(
            allocative::Key::new("device_rows"),
            self.metadata.plane_bytes as usize,
        );
    }
}

/// The registers-claim view of the resident Stage-1 rows: RdWriteValue
/// decodes from the compact flags and the raw memory words
/// (`spartan_row_rd_write_value`); rows at or past `explicit_rows` are zero.
#[derive(Clone)]
pub(crate) struct RegistersClaimResidentRows {
    compact: Buffer,
    raw: Buffer,
    geometry: RegistersClaimGeometry,
    explicit_rows: usize,
    source_generation: u64,
}

impl RegistersClaimResidentRows {
    pub(crate) const fn geometry(&self) -> RegistersClaimGeometry {
        self.geometry
    }

    pub(crate) const fn source_generation(&self) -> u64 {
        self.source_generation
    }

    pub(crate) fn compact_allocation_identity(&self) -> usize {
        self.compact.as_ptr() as usize
    }

    fn validate_for(&self, context: &SolinasMetal) -> Result<(), RegistersClaimError> {
        let rows = self.geometry.rows();
        for (name, buffer, row_bytes) in [
            (
                "resident compact rows",
                &self.compact,
                size_of::<InstructionInputRow>(),
            ),
            ("resident raw rows", &self.raw, size_of::<SpartanRawRow>()),
        ] {
            let bytes = rows
                .checked_mul(row_bytes)
                .ok_or(MetalError::InputTooLong(rows))?;
            validate_buffer_shape(buffer, name, to_u64(bytes)?, context.device_registry_id())?;
        }
        Ok(())
    }
}

struct AliasFoldBuffers {
    eq_prefix: Buffer,
    rd_dense: Buffer,
}

pub(crate) struct RegistersClaimAliasFoldInvocation {
    context: SolinasMetal,
    rows: RegistersClaimResidentRows,
    rd_post: Option<RegistersClaimResidentRdPlane>,
    pipeline: ComputePipelineState,
    limits: PipelineLimits,
    buffers: AliasFoldBuffers,
    buffer_identities: [usize; 2],
    geometry: RegistersClaimGeometry,
    params: super::RegistersClaimParams,
    threads_per_threadgroup: usize,
    dynamic_threadgroup_bytes: usize,
}

pub(crate) struct RegistersClaimAliasFoldObservation {
    pub(crate) rd_write_value: Vec<AkitaField>,
    pub(crate) gpu_active: Duration,
}

impl SolinasMetal {
    #[cfg(feature = "test-utils")]
    pub(crate) fn prepare_test_registers_claim_resident_rd_plane(
        &self,
        rows: usize,
        physical_rows: usize,
        mut value: impl FnMut(usize) -> u64,
    ) -> Result<RegistersClaimResidentRdPlane, RegistersClaimError> {
        if physical_rows == 0 || physical_rows > rows {
            return Err(RegistersClaimError::InvalidState(
                "test resident rd plane has invalid physical rows",
            ));
        }
        let bytes = to_u64(
            rows.checked_mul(size_of::<u64>())
                .ok_or(MetalError::InputTooLong(rows))?,
        )?;
        self.validate_buffer_length(bytes)?;
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        // SAFETY: the new shared buffer is exclusively owned and has `rows`
        // contiguous u64 elements.
        let values = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<u64>(), rows) };
        for (row, output) in values.iter_mut().take(physical_rows).enumerate() {
            *output = value(row);
        }
        values[physical_rows..].fill(0);
        self.attach_registers_claim_resident_rd_plane(buffer, rows, 1, 1)
    }

    pub(crate) fn attach_registers_claim_resident_rd_plane(
        &self,
        buffer: Buffer,
        rows: usize,
        source_generation: u64,
        completion_serial: u64,
    ) -> Result<RegistersClaimResidentRdPlane, RegistersClaimError> {
        let geometry = RegistersClaimGeometry::new(rows)?;
        let plane_bytes = to_u64(
            rows.checked_mul(size_of::<u64>())
                .ok_or(MetalError::InputTooLong(rows))?,
        )?;
        let allocation_identity = buffer.as_ptr() as usize;
        if source_generation == 0 || completion_serial == 0 || allocation_identity == 0 {
            return Err(RegistersClaimError::InvalidState(
                "resident rd receipt is incomplete",
            ));
        }
        validate_buffer_binding(
            &buffer,
            "resident rd_write_value",
            plane_bytes,
            self.device_registry_id(),
            allocation_identity,
        )?;
        Ok(RegistersClaimResidentRdPlane {
            buffer,
            metadata: ResidentRdMetadata {
                geometry,
                plane_bytes,
                device_registry_id: self.device_registry_id(),
                allocation_identity,
                source_generation,
                completion_serial,
            },
        })
    }

    pub(crate) fn attach_registers_claim_resident_rows(
        &self,
        compact: Buffer,
        raw: Buffer,
        rows: usize,
        explicit_rows: usize,
        source_generation: u64,
    ) -> Result<RegistersClaimResidentRows, RegistersClaimError> {
        let geometry = RegistersClaimGeometry::new(rows)?;
        let _ = geometry.params(explicit_rows, false)?;
        if source_generation == 0 {
            return Err(RegistersClaimError::InvalidState(
                "resident rows receipt is incomplete",
            ));
        }
        let rows = RegistersClaimResidentRows {
            compact,
            raw,
            geometry,
            explicit_rows,
            source_generation,
        };
        rows.validate_for(self)?;
        Ok(rows)
    }

    /// Allocates the RdWriteValue plane the Stage-4 registers read-write
    /// source binds; the alias fold over `rows` writes it.
    pub(crate) fn prepare_registers_claim_rd_post(
        &self,
        rows: &RegistersClaimResidentRows,
        completion_serial: u64,
    ) -> Result<RegistersClaimResidentRdPlane, RegistersClaimError> {
        let row_count = rows.geometry.rows();
        let bytes = to_u64(
            row_count
                .checked_mul(size_of::<u64>())
                .ok_or(MetalError::InputTooLong(row_count))?,
        )?;
        self.validate_buffer_length(bytes)?;
        self.validate_additional_working_set(bytes)?;
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        self.attach_registers_claim_resident_rd_plane(
            buffer,
            row_count,
            rows.source_generation,
            completion_serial,
        )
    }

    pub(crate) fn prepare_registers_claim_alias_fold(
        &self,
        rows: &RegistersClaimResidentRows,
        rd_post: Option<&RegistersClaimResidentRdPlane>,
        prefix_challenges: &[AkitaField],
        config: RegistersClaimKernelConfig,
    ) -> Result<RegistersClaimAliasFoldInvocation, RegistersClaimError> {
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let geometry = rows.geometry();
        rows.validate_for(self)?;
        if let Some(rd_post) = rd_post {
            rd_post.validate_for(self, geometry)?;
        }
        if prefix_challenges.len() != geometry.prefix_vars() {
            return Err(RegistersClaimError::WrongPrefixChallengeCount {
                expected: geometry.prefix_vars(),
                actual: prefix_challenges.len(),
            });
        }
        let config = config.validate()?;
        let prefix_point = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
        let eq_prefix = encode_fields(&EqPolynomial::<AkitaField>::evals(&prefix_point, None));
        self.validate_inputs("registers claim alias eq_prefix", &eq_prefix)?;
        let eq_bytes = geometry
            .prefix_elements()
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(geometry.rows()))?;
        let output_bytes = geometry
            .suffix_elements()
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(geometry.rows()))?;
        self.validate_buffer_length(to_u64(eq_bytes)?)?;
        self.validate_buffer_length(to_u64(output_bytes)?)?;
        self.validate_additional_working_set(to_u64(eq_bytes + output_bytes)?)?;

        let pipeline = self.compile_named_pipeline(ALIAS_FOLD_PIPELINE)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH {
            return Err(RegistersClaimError::UnsupportedExecutionWidth {
                pipeline: ALIAS_FOLD_PIPELINE,
                expected: REGISTERS_CLAIM_SIMD_WIDTH,
                got: limits.thread_execution_width,
            });
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(Some(config.fold_threads_per_threadgroup), limits)?;
        let dynamic_threadgroup_bytes = config.alias_fold_threadgroup_bytes()?;
        let requested = to_u64(dynamic_threadgroup_bytes)?
            .checked_add(limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(dynamic_threadgroup_bytes))?;
        let maximum = self.device.max_threadgroup_memory_length();
        if requested > maximum {
            return Err(RegistersClaimError::ThreadgroupMemory { requested, maximum });
        }

        let buffers = AliasFoldBuffers {
            eq_prefix: buffer_from_slice(&self.device, &eq_prefix),
            rd_dense: self
                .device
                .new_buffer(to_u64(output_bytes)?, MTLResourceOptions::StorageModeShared),
        };
        let buffer_identities = [
            buffers.eq_prefix.as_ptr() as usize,
            buffers.rd_dense.as_ptr() as usize,
        ];
        if buffer_identities[0] == buffer_identities[1]
            || rd_post.is_some_and(|rd_post| {
                [
                    rows.compact_allocation_identity(),
                    rows.raw.as_ptr() as usize,
                    buffer_identities[0],
                    buffer_identities[1],
                ]
                .contains(&rd_post.allocation_identity())
            })
        {
            return Err(RegistersClaimError::AliasedInvocationBuffers);
        }
        Ok(RegistersClaimAliasFoldInvocation {
            context: self.clone(),
            rows: rows.clone(),
            rd_post: rd_post.cloned(),
            pipeline,
            limits,
            buffers,
            buffer_identities,
            geometry,
            params: geometry.params(rows.explicit_rows, rd_post.is_some())?,
            threads_per_threadgroup,
            dynamic_threadgroup_bytes,
        })
    }
}

impl RegistersClaimAliasFoldInvocation {
    pub(crate) fn execute_timed(
        &self,
    ) -> Result<RegistersClaimAliasFoldObservation, RegistersClaimError> {
        self.validate_state()?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(ALIAS_FOLD_COMPACT_ROWS_SLOT, Some(&self.rows.compact), 0);
            encoder.set_buffer(ALIAS_FOLD_RAW_ROWS_SLOT, Some(&self.rows.raw), 0);
            encoder.set_buffer(ALIAS_FOLD_EQ_PREFIX_SLOT, Some(&self.buffers.eq_prefix), 0);
            encoder.set_buffer(ALIAS_FOLD_OUTPUT_SLOT, Some(&self.buffers.rd_dense), 0);
            set_inline_bytes(encoder, ALIAS_FOLD_PARAMS_SLOT, &self.params);
            // Without a plane `write_rd_post` is zero and the kernel never
            // writes this slot; it is bound so no argument is nil.
            let rd_post = self
                .rd_post
                .as_ref()
                .map_or(&self.buffers.rd_dense, |rd_post| &rd_post.buffer);
            encoder.set_buffer(ALIAS_FOLD_RD_POST_SLOT, Some(rd_post), 0);
            encoder.set_threadgroup_memory_length(
                ALIAS_FOLD_THREADGROUP_SLOT,
                to_u64(self.dynamic_threadgroup_bytes)?,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.geometry.suffix_elements() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = completed_command_gpu_time(command_buffer)?;
            // SAFETY: command completion initializes exactly one field per
            // suffix row in the shared output allocation.
            let fields = unsafe {
                slice::from_raw_parts(
                    self.buffers.rd_dense.contents().cast::<Fp128>(),
                    self.geometry.suffix_elements(),
                )
            };
            self.context
                .validate_inputs("registers claim alias rd output", fields)?;
            Ok(RegistersClaimAliasFoldObservation {
                rd_write_value: fields
                    .iter()
                    .map(|&value| value.into_jolt_field())
                    .collect(),
                gpu_active,
            })
        })
    }

    fn validate_state(&self) -> Result<(), RegistersClaimError> {
        self.rows.validate_for(&self.context)?;
        if let Some(rd_post) = &self.rd_post {
            rd_post.validate_for(&self.context, self.geometry)?;
        }
        if self.rows.geometry != self.geometry
            || self.params
                != self
                    .geometry
                    .params(self.rows.explicit_rows, self.rd_post.is_some())?
            || self.limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH
            || self.threads_per_threadgroup > self.limits.max_total_threads_per_threadgroup
            || !self
                .threads_per_threadgroup
                .is_multiple_of(self.limits.thread_execution_width)
        {
            return Err(RegistersClaimError::InvalidState(
                "alias-fold invocation differs from its checked plan",
            ));
        }
        for (name, buffer, expected_bytes, identity) in [
            (
                "alias eq_prefix",
                &self.buffers.eq_prefix,
                self.geometry.prefix_elements() * size_of::<Fp128>(),
                self.buffer_identities[0],
            ),
            (
                "alias rd output",
                &self.buffers.rd_dense,
                self.geometry.suffix_elements() * size_of::<Fp128>(),
                self.buffer_identities[1],
            ),
        ] {
            validate_buffer_binding(
                buffer,
                name,
                to_u64(expected_bytes)?,
                self.context.device_registry_id(),
                identity,
            )?;
        }
        Ok(())
    }
}

fn validate_buffer_binding(
    buffer: &Buffer,
    name: &'static str,
    expected_bytes: u64,
    expected_device: u64,
    expected_identity: usize,
) -> Result<(), RegistersClaimError> {
    validate_buffer_shape(buffer, name, expected_bytes, expected_device)?;
    if buffer.as_ptr() as usize != expected_identity {
        return Err(RegistersClaimError::InvalidState(
            "buffer allocation identity changed",
        ));
    }
    Ok(())
}

fn validate_buffer_shape(
    buffer: &Buffer,
    name: &'static str,
    expected_bytes: u64,
    expected_device: u64,
) -> Result<(), RegistersClaimError> {
    let got_device = buffer.device().registry_id();
    if got_device != expected_device {
        return Err(RegistersClaimError::BufferDevice {
            name,
            expected: expected_device,
            got: got_device,
        });
    }
    if buffer.length() != expected_bytes {
        return Err(RegistersClaimError::BufferLength {
            name,
            expected: expected_bytes,
            actual: buffer.length(),
        });
    }
    Ok(())
}

fn encode_fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn to_u64(value: usize) -> Result<u64, MetalError> {
    u64::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::FromPrimitiveInt;
    use jolt_witness::witnesses::SpartanOuterRow;

    use super::super::super::spartan_outer_uniskip::test_rows::{splitmix, witness};
    use super::super::super::SpartanOuterUniskipRow;
    use super::*;

    #[test]
    fn alias_fold_decodes_rd_write_value_from_the_resident_rows() {
        let Ok(context) = SolinasMetal::for_akita() else {
            return;
        };
        let witness = witness::<SpartanOuterRow>(15);
        let explicit_rows = witness.len() - 1;
        let mut stage1 = witness
            .iter()
            .map(SpartanOuterUniskipRow::from_spartan_outer)
            .collect::<Vec<_>>();
        let writer = witness
            .iter()
            .position(|row| row.rd_write_value.0 != 0)
            .unwrap();
        stage1[explicit_rows] = stage1[writer];
        let resident = context.prepare_spartan_outer_uniskip_rows(&stage1).unwrap();
        let rows = context
            .attach_registers_claim_resident_rows(
                resident.instruction_input_buffer().clone(),
                resident.raw_buffer().clone(),
                resident.len(),
                explicit_rows,
                resident.key().generation,
            )
            .unwrap();
        let geometry = rows.geometry();
        let expected_rd = witness
            .iter()
            .take(explicit_rows)
            .map(|row| row.rd_write_value.0)
            .chain([0])
            .collect::<Vec<_>>();
        let challenges = (0..geometry.prefix_vars() as u64)
            .map(|index| AkitaField::from_u64(splitmix(index)))
            .collect::<Vec<_>>();
        let prefix_point = challenges.iter().rev().copied().collect::<Vec<_>>();
        let eq_prefix = EqPolynomial::<AkitaField>::evals(&prefix_point, None);
        let expected_fold = expected_rd
            .chunks(geometry.prefix_elements())
            .map(|block| {
                block
                    .iter()
                    .zip(&eq_prefix)
                    .map(|(&rd, &weight)| weight * AkitaField::from_u64(rd))
                    .sum::<AkitaField>()
            })
            .collect::<Vec<_>>();

        let rd_post = context.prepare_registers_claim_rd_post(&rows, 1).unwrap();
        for plane in [None, Some(&rd_post)] {
            let observation = context
                .prepare_registers_claim_alias_fold(
                    &rows,
                    plane,
                    &challenges,
                    RegistersClaimKernelConfig::default(),
                )
                .unwrap()
                .execute_timed()
                .unwrap();
            assert_eq!(observation.rd_write_value, expected_fold);
        }
        // SAFETY: the completed fold wrote one u64 per row into the shared
        // plane and no command buffer still references it.
        let plane = unsafe {
            slice::from_raw_parts(rd_post.buffer().contents().cast::<u64>(), geometry.rows())
        };
        assert_eq!(plane, expected_rd);
    }
}
