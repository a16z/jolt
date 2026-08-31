use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::super::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};
use super::{
    RegistersClaimGeometry, RegistersClaimKernelConfig, RegistersClaimPlanError,
    ALIAS_FOLD_EQ_PREFIX_SLOT, ALIAS_FOLD_OUTPUT_SLOT, ALIAS_FOLD_PARAMS_SLOT, ALIAS_FOLD_PIPELINE,
    ALIAS_FOLD_RD_WRITE_VALUE_SLOT, ALIAS_FOLD_THREADGROUP_SLOT, REGISTERS_CLAIM_AKITA_OFFSET,
    REGISTERS_CLAIM_SIMD_WIDTH,
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

struct AliasFoldBuffers {
    eq_prefix: Buffer,
    rd_dense: Buffer,
}

pub(crate) struct RegistersClaimAliasFoldInvocation {
    context: SolinasMetal,
    rd: RegistersClaimResidentRdPlane,
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
    #[cfg(any(test, feature = "test-utils"))]
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

    pub(crate) fn prepare_registers_claim_alias_fold(
        &self,
        rd: &RegistersClaimResidentRdPlane,
        prefix_challenges: &[AkitaField],
        config: RegistersClaimKernelConfig,
    ) -> Result<RegistersClaimAliasFoldInvocation, RegistersClaimError> {
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let geometry = rd.geometry();
        rd.validate_for(self, geometry)?;
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
            || buffer_identities.contains(&rd.allocation_identity())
        {
            return Err(RegistersClaimError::AliasedInvocationBuffers);
        }
        Ok(RegistersClaimAliasFoldInvocation {
            context: self.clone(),
            rd: rd.clone(),
            pipeline,
            limits,
            buffers,
            buffer_identities,
            geometry,
            params: geometry.params()?,
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
            encoder.set_buffer(ALIAS_FOLD_RD_WRITE_VALUE_SLOT, Some(&self.rd.buffer), 0);
            encoder.set_buffer(ALIAS_FOLD_EQ_PREFIX_SLOT, Some(&self.buffers.eq_prefix), 0);
            encoder.set_buffer(ALIAS_FOLD_OUTPUT_SLOT, Some(&self.buffers.rd_dense), 0);
            set_inline_bytes(encoder, ALIAS_FOLD_PARAMS_SLOT, &self.params);
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
        self.rd.validate_for(&self.context, self.geometry)?;
        if self.params != self.geometry.params()?
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
    if buffer.as_ptr() as usize != expected_identity {
        return Err(RegistersClaimError::InvalidState(
            "buffer allocation identity changed",
        ));
    }
    Ok(())
}

fn encode_fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn to_u64(value: usize) -> Result<u64, MetalError> {
    u64::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}
