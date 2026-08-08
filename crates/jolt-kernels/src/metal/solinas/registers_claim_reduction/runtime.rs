use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use super::{
    RegistersClaimGeometry, RegistersClaimKernelConfig, RegistersClaimLinearQPlan,
    RegistersClaimPlan, RegistersClaimPlanError, RegistersClaimStrategy, ALIAS_FOLD_EQ_PREFIX_SLOT,
    ALIAS_FOLD_OUTPUT_SLOT, ALIAS_FOLD_PARAMS_SLOT, ALIAS_FOLD_PIPELINE,
    ALIAS_FOLD_RD_WRITE_VALUE_SLOT, ALIAS_FOLD_THREADGROUP_SLOT, DIRECT_FOLD_EQ_PREFIX_SLOT,
    DIRECT_FOLD_OUTPUT_SLOT, DIRECT_FOLD_PARAMS_SLOT, DIRECT_FOLD_PIPELINE,
    DIRECT_FOLD_RD_WRITE_VALUE_SLOT, DIRECT_FOLD_RS1_VALUE_SLOT, DIRECT_FOLD_RS2_VALUE_SLOT,
    DIRECT_FOLD_THREADGROUP_SLOT, LINEAR_Q_EQ_SUFFIX_SLOT, LINEAR_Q_GAMMA_POWERS_SLOT,
    LINEAR_Q_OUTPUT_SLOT, LINEAR_Q_PARAMS_SLOT, LINEAR_Q_RD_WRITE_VALUE_SLOT,
    LINEAR_Q_RS1_VALUE_SLOT, LINEAR_Q_RS2_VALUE_SLOT, REGISTERS_CLAIM_AKITA_OFFSET,
    REGISTERS_CLAIM_OUTPUT_COLUMNS, REGISTERS_CLAIM_SIMD_WIDTH,
};

#[derive(Debug, Error)]
pub enum RegistersClaimLinearQError {
    #[error(transparent)]
    Plan(#[from] RegistersClaimPlanError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("registers claim linear-q requires Akita offset {expected:#x}, got {got:#x}")]
    UnsupportedOffset { expected: u32, got: u32 },
    #[error("registers claim linear-q point has length {actual}, expected {expected}")]
    WrongPointLength { expected: usize, actual: usize },
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
    #[error("registers claim resident planes must use three distinct allocations")]
    AliasedResidentPlanes,
    #[error("registers claim linear-q buffers alias across read and write bindings")]
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
        "registers claim direct fold needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    ThreadgroupMemory { requested: u64, maximum: u64 },
    #[error("invalid registers claim linear-q state: {0}")]
    InvalidState(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentPlaneMetadata {
    geometry: RegistersClaimGeometry,
    plane_bytes: u64,
    device_registry_id: u64,
    allocation_identities: [usize; 3],
}

/// Three native register-value planes owned by one Metal device.
///
/// The wrapper records byte lengths, allocation identities, and device identity
/// when attached. Every invocation checks those values again before dispatch.
#[derive(Clone)]
pub struct RegistersClaimResidentPlanes {
    rd_write_value: Buffer,
    rs1_value: Buffer,
    rs2_value: Buffer,
    metadata: ResidentPlaneMetadata,
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

    pub(crate) const fn resident_bytes(&self) -> u64 {
        self.metadata.plane_bytes
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

    fn validate_for(
        &self,
        context: &SolinasMetal,
        geometry: RegistersClaimGeometry,
    ) -> Result<(), RegistersClaimLinearQError> {
        if self.metadata.geometry != geometry
            || self.metadata.source_generation == 0
            || self.metadata.completion_serial == 0
        {
            return Err(RegistersClaimLinearQError::InvalidState(
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

impl RegistersClaimResidentPlanes {
    pub const fn rows(&self) -> usize {
        self.metadata.geometry.rows()
    }

    pub const fn geometry(&self) -> RegistersClaimGeometry {
        self.metadata.geometry
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.metadata.plane_bytes * 3
    }

    pub const fn device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id
    }

    pub fn allocation_identities(&self) -> [usize; 3] {
        [
            self.rd_write_value.as_ptr() as usize,
            self.rs1_value.as_ptr() as usize,
            self.rs2_value.as_ptr() as usize,
        ]
    }

    fn validate_for(
        &self,
        context: &SolinasMetal,
        geometry: RegistersClaimGeometry,
    ) -> Result<(), RegistersClaimLinearQError> {
        if self.metadata.geometry != geometry {
            return Err(RegistersClaimLinearQError::InvalidState(
                "resident geometry differs from the invocation plan",
            ));
        }
        let expected_device = context.device_registry_id();
        if self.metadata.device_registry_id != expected_device {
            return Err(RegistersClaimLinearQError::BufferDevice {
                name: "resident metadata",
                expected: expected_device,
                got: self.metadata.device_registry_id,
            });
        }
        let expected_identities = self.metadata.allocation_identities;
        let actual_identities = self.allocation_identities();
        if actual_identities != expected_identities {
            return Err(RegistersClaimLinearQError::InvalidState(
                "resident allocation identity changed",
            ));
        }
        validate_distinct_three(actual_identities)?;

        for (name, buffer) in [
            ("rd_write_value", &self.rd_write_value),
            ("rs1_value", &self.rs1_value),
            ("rs2_value", &self.rs2_value),
        ] {
            validate_buffer_binding(
                buffer,
                name,
                self.metadata.plane_bytes,
                expected_device,
                buffer.as_ptr() as usize,
            )?;
        }
        Ok(())
    }
}

struct LinearQBuffers {
    gamma_powers: Buffer,
    eq_suffix: Buffer,
    output: Buffer,
}

/// Prepared resident-buffer invocation for `solinas_registers_claim_build_linear_q`.
///
/// Preparation owns all allocation, equality-table generation, field encoding,
/// and pipeline compilation. `execute_timed` performs no device allocations.
pub struct RegistersClaimLinearQInvocation {
    context: SolinasMetal,
    rows: RegistersClaimResidentPlanes,
    pipeline: ComputePipelineState,
    limits: PipelineLimits,
    buffers: LinearQBuffers,
    buffer_identities: [usize; 3],
    plan: RegistersClaimLinearQPlan,
    threads_per_threadgroup: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimLinearQObservation {
    pub q: Vec<AkitaField>,
    pub checksum: u64,
    pub useful_half_width_terms: u64,
    pub full_products: u64,
    pub useful_threads: usize,
    pub dispatched_threads: usize,
    pub gpu_active: Duration,
    pub resident_wall: Duration,
}

struct DirectFoldBuffers {
    eq_prefix: Buffer,
    dense_outputs: Buffer,
}

/// Prepared midpoint projection for the three canonical register openings.
pub struct RegistersClaimDirectFoldInvocation {
    context: SolinasMetal,
    rows: RegistersClaimResidentPlanes,
    pipeline: ComputePipelineState,
    limits: PipelineLimits,
    buffers: DirectFoldBuffers,
    buffer_identities: [usize; 2],
    plan: RegistersClaimPlan,
    threads_per_threadgroup: usize,
    dynamic_threadgroup_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimDirectFoldObservation {
    pub outputs: super::RegistersClaimDenseOutputs<AkitaField>,
    pub useful_half_width_terms: u64,
    pub threadgroups: usize,
    pub gpu_active: Duration,
    pub resident_wall: Duration,
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
    pub(crate) useful_half_width_terms: u64,
    pub(crate) gpu_active: Duration,
    pub(crate) resident_wall: Duration,
}

impl SolinasMetal {
    pub fn prepare_registers_claim_resident_planes(
        &self,
        rd_write_value: &[u64],
        rs1_value: &[u64],
        rs2_value: &[u64],
    ) -> Result<RegistersClaimResidentPlanes, RegistersClaimLinearQError> {
        let geometry = RegistersClaimGeometry::new(rd_write_value.len())?;
        for (name, actual) in [
            ("rd_write_value", rd_write_value.len()),
            ("rs1_value", rs1_value.len()),
            ("rs2_value", rs2_value.len()),
        ] {
            if actual != geometry.rows() {
                return Err(RegistersClaimPlanError::WrongPlaneLength {
                    name,
                    expected: geometry.rows(),
                    actual,
                }
                .into());
            }
        }

        self.prepare_registers_claim_resident_planes_with_fill(
            geometry.rows(),
            |rd_destination, rs1_destination, rs2_destination| {
                rd_destination.copy_from_slice(rd_write_value);
                rs1_destination.copy_from_slice(rs1_value);
                rs2_destination.copy_from_slice(rs2_value);
            },
        )
    }

    pub(crate) fn prepare_registers_claim_resident_planes_with_fill(
        &self,
        rows: usize,
        fill: impl FnOnce(&mut [u64], &mut [u64], &mut [u64]),
    ) -> Result<RegistersClaimResidentPlanes, RegistersClaimLinearQError> {
        let geometry = RegistersClaimGeometry::new(rows)?;
        let storage = geometry.linear_q_storage()?;
        let plane_bytes = to_u64(storage.native_plane_bytes)?;
        self.validate_buffer_length(plane_bytes)?;
        self.validate_additional_working_set(to_u64(storage.resident_input_bytes)?)?;

        let rd_write_value = self
            .device
            .new_buffer(plane_bytes, MTLResourceOptions::StorageModeShared);
        let rs1_value = self
            .device
            .new_buffer(plane_bytes, MTLResourceOptions::StorageModeShared);
        let rs2_value = self
            .device
            .new_buffer(plane_bytes, MTLResourceOptions::StorageModeShared);
        // SAFETY: the three shared allocations each own exactly `rows` u64s
        // and no command can observe them until the fill callback returns.
        let (rd_destination, rs1_destination, rs2_destination) = unsafe {
            (
                slice::from_raw_parts_mut(rd_write_value.contents().cast::<u64>(), rows),
                slice::from_raw_parts_mut(rs1_value.contents().cast::<u64>(), rows),
                slice::from_raw_parts_mut(rs2_value.contents().cast::<u64>(), rows),
            )
        };
        fill(rd_destination, rs1_destination, rs2_destination);

        self.attach_registers_claim_resident_planes(
            rd_write_value,
            rs1_value,
            rs2_value,
            geometry.rows(),
        )
    }

    pub fn attach_registers_claim_resident_planes(
        &self,
        rd_write_value: Buffer,
        rs1_value: Buffer,
        rs2_value: Buffer,
        rows: usize,
    ) -> Result<RegistersClaimResidentPlanes, RegistersClaimLinearQError> {
        let geometry = RegistersClaimGeometry::new(rows)?;
        let plane_bytes = to_u64(geometry.linear_q_storage()?.native_plane_bytes)?;
        let expected_device = self.device_registry_id();
        let buffers = [&rd_write_value, &rs1_value, &rs2_value];
        let names = ["rd_write_value", "rs1_value", "rs2_value"];
        let identities = std::array::from_fn(|index| buffers[index].as_ptr() as usize);
        validate_distinct_three(identities)?;
        for (name, buffer) in names.into_iter().zip(buffers) {
            self.validate_buffer_length(buffer.length())?;
            validate_buffer_binding(
                buffer,
                name,
                plane_bytes,
                expected_device,
                buffer.as_ptr() as usize,
            )?;
        }

        Ok(RegistersClaimResidentPlanes {
            rd_write_value,
            rs1_value,
            rs2_value,
            metadata: ResidentPlaneMetadata {
                geometry,
                plane_bytes,
                device_registry_id: expected_device,
                allocation_identities: identities,
            },
        })
    }

    pub(crate) fn attach_registers_claim_resident_rd_plane(
        &self,
        buffer: Buffer,
        rows: usize,
        source_generation: u64,
        completion_serial: u64,
    ) -> Result<RegistersClaimResidentRdPlane, RegistersClaimLinearQError> {
        let geometry = RegistersClaimGeometry::new(rows)?;
        let plane_bytes = to_u64(
            rows.checked_mul(size_of::<u64>())
                .ok_or(MetalError::InputTooLong(rows))?,
        )?;
        let allocation_identity = buffer.as_ptr() as usize;
        if source_generation == 0 || completion_serial == 0 || allocation_identity == 0 {
            return Err(RegistersClaimLinearQError::InvalidState(
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

    pub fn prepare_registers_claim_linear_q(
        &self,
        rows: &RegistersClaimResidentPlanes,
        tau: &[AkitaField],
        gamma: AkitaField,
        config: RegistersClaimKernelConfig,
    ) -> Result<RegistersClaimLinearQInvocation, RegistersClaimLinearQError> {
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimLinearQError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let max_buffer_length = usize::try_from(self.device.max_buffer_length())
            .map_err(|_| MetalError::InputTooLong(rows.rows()))?;
        let plan = RegistersClaimLinearQPlan::new(rows.rows(), max_buffer_length, config)?;
        rows.validate_for(self, plan.geometry)?;
        if tau.len() != plan.geometry.log_t() {
            return Err(RegistersClaimLinearQError::WrongPointLength {
                expected: plan.geometry.log_t(),
                actual: tau.len(),
            });
        }

        let tau_hi = &tau[..plan.geometry.suffix_vars()];
        let eq_suffix = EqPolynomial::<AkitaField>::evals(tau_hi, None);
        let gamma_powers = [gamma, gamma * gamma];
        let eq_suffix = encode_fields(&eq_suffix);
        let gamma_powers = encode_fields(&gamma_powers);
        self.validate_inputs("registers claim eq_suffix", &eq_suffix)?;
        self.validate_inputs("registers claim gamma powers", &gamma_powers)?;

        self.validate_additional_working_set(to_u64(plan.storage.private_bytes)?)?;
        for bytes in [
            plan.storage.gamma_powers_bytes,
            plan.storage.eq_suffix_bytes,
            plan.storage.output_bytes,
        ] {
            self.validate_buffer_length(to_u64(bytes)?)?;
        }

        let pipeline_name = plan.config.accumulator.pipeline();
        let pipeline = self.compile_named_pipeline(pipeline_name)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH {
            return Err(RegistersClaimLinearQError::UnsupportedExecutionWidth {
                pipeline: pipeline_name,
                expected: REGISTERS_CLAIM_SIMD_WIDTH,
                got: limits.thread_execution_width,
            });
        }
        let threads_per_threadgroup = Self::resolve_threadgroup_width(
            Some(plan.config.build_threads_per_threadgroup),
            limits,
        )?;
        if threads_per_threadgroup != plan.threads_per_threadgroup() {
            return Err(RegistersClaimLinearQError::InvalidState(
                "resolved threadgroup width differs from the checked plan",
            ));
        }

        let buffers = LinearQBuffers {
            gamma_powers: buffer_from_slice(&self.device, &gamma_powers),
            eq_suffix: buffer_from_slice(&self.device, &eq_suffix),
            output: self.device.new_buffer(
                to_u64(plan.storage.output_bytes)?,
                MTLResourceOptions::StorageModeShared,
            ),
        };
        let buffer_identities = [
            buffers.gamma_powers.as_ptr() as usize,
            buffers.eq_suffix.as_ptr() as usize,
            buffers.output.as_ptr() as usize,
        ];
        validate_invocation_aliases(rows.allocation_identities(), buffer_identities)?;

        Ok(RegistersClaimLinearQInvocation {
            context: self.clone(),
            rows: rows.clone(),
            pipeline,
            limits,
            buffers,
            buffer_identities,
            plan,
            threads_per_threadgroup,
        })
    }

    pub fn prepare_registers_claim_direct_fold(
        &self,
        rows: &RegistersClaimResidentPlanes,
        prefix_challenges: &[AkitaField],
        config: RegistersClaimKernelConfig,
    ) -> Result<RegistersClaimDirectFoldInvocation, RegistersClaimLinearQError> {
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimLinearQError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let max_buffer_length = usize::try_from(self.device.max_buffer_length())
            .map_err(|_| MetalError::InputTooLong(rows.rows()))?;
        let plan = RegistersClaimPlan::new(
            rows.rows(),
            max_buffer_length,
            config,
            RegistersClaimStrategy::DirectLinear,
        )?;
        rows.validate_for(self, plan.geometry)?;
        if prefix_challenges.len() != plan.geometry.prefix_vars() {
            return Err(RegistersClaimLinearQError::WrongPrefixChallengeCount {
                expected: plan.geometry.prefix_vars(),
                actual: prefix_challenges.len(),
            });
        }

        let prefix_point = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
        let eq_prefix = encode_fields(&EqPolynomial::<AkitaField>::evals(&prefix_point, None));
        self.validate_inputs("registers claim eq_prefix", &eq_prefix)?;
        let private_bytes = plan
            .storage
            .prefix_field_bytes
            .checked_add(plan.storage.direct_dense_bytes)
            .ok_or(MetalError::InputTooLong(rows.rows()))?;
        self.validate_additional_working_set(to_u64(private_bytes)?)?;
        for bytes in [
            plan.storage.prefix_field_bytes,
            plan.storage.direct_dense_bytes,
        ] {
            self.validate_buffer_length(to_u64(bytes)?)?;
        }

        let pipeline = self.compile_named_pipeline(DIRECT_FOLD_PIPELINE)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH {
            return Err(RegistersClaimLinearQError::UnsupportedExecutionWidth {
                pipeline: DIRECT_FOLD_PIPELINE,
                expected: REGISTERS_CLAIM_SIMD_WIDTH,
                got: limits.thread_execution_width,
            });
        }
        let threads_per_threadgroup = Self::resolve_threadgroup_width(
            Some(plan.config.fold_threads_per_threadgroup),
            limits,
        )?;
        if threads_per_threadgroup != plan.config.fold_threads_per_threadgroup {
            return Err(RegistersClaimLinearQError::InvalidState(
                "resolved direct-fold width differs from the checked plan",
            ));
        }
        let dynamic_threadgroup_bytes = plan.fold_threadgroup_bytes()?;
        let total_threadgroup_bytes = to_u64(dynamic_threadgroup_bytes)?
            .checked_add(limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(dynamic_threadgroup_bytes))?;
        let maximum = self.device.max_threadgroup_memory_length();
        if total_threadgroup_bytes > maximum {
            return Err(RegistersClaimLinearQError::ThreadgroupMemory {
                requested: total_threadgroup_bytes,
                maximum,
            });
        }

        let buffers = DirectFoldBuffers {
            eq_prefix: buffer_from_slice(&self.device, &eq_prefix),
            dense_outputs: self.device.new_buffer(
                to_u64(plan.storage.direct_dense_bytes)?,
                MTLResourceOptions::StorageModeShared,
            ),
        };
        let buffer_identities = [
            buffers.eq_prefix.as_ptr() as usize,
            buffers.dense_outputs.as_ptr() as usize,
        ];
        validate_direct_fold_aliases(rows.allocation_identities(), buffer_identities)?;

        Ok(RegistersClaimDirectFoldInvocation {
            context: self.clone(),
            rows: rows.clone(),
            pipeline,
            limits,
            buffers,
            buffer_identities,
            plan,
            threads_per_threadgroup,
            dynamic_threadgroup_bytes,
        })
    }

    pub(crate) fn prepare_registers_claim_alias_fold(
        &self,
        rd: &RegistersClaimResidentRdPlane,
        prefix_challenges: &[AkitaField],
        config: RegistersClaimKernelConfig,
    ) -> Result<RegistersClaimAliasFoldInvocation, RegistersClaimLinearQError> {
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimLinearQError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let geometry = rd.geometry();
        rd.validate_for(self, geometry)?;
        if prefix_challenges.len() != geometry.prefix_vars() {
            return Err(RegistersClaimLinearQError::WrongPrefixChallengeCount {
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
            return Err(RegistersClaimLinearQError::UnsupportedExecutionWidth {
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
            return Err(RegistersClaimLinearQError::ThreadgroupMemory { requested, maximum });
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
            return Err(RegistersClaimLinearQError::AliasedInvocationBuffers);
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

impl RegistersClaimLinearQInvocation {
    pub fn execute(&self) -> Result<Vec<AkitaField>, RegistersClaimLinearQError> {
        self.execute_timed().map(|observation| observation.q)
    }

    /// Times a prepared resident invocation.
    ///
    /// `resident_wall` starts before validation and ends after canonical output
    /// conversion and checksum. It excludes plane upload, equality generation,
    /// allocation, and pipeline compilation. `gpu_active` uses Metal command
    /// timestamps over the single q-build dispatch.
    pub fn execute_timed(
        &self,
    ) -> Result<RegistersClaimLinearQObservation, RegistersClaimLinearQError> {
        let wall_started = Instant::now();
        self.validate_state()?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(
                LINEAR_Q_RD_WRITE_VALUE_SLOT,
                Some(&self.rows.rd_write_value),
                0,
            );
            encoder.set_buffer(LINEAR_Q_RS1_VALUE_SLOT, Some(&self.rows.rs1_value), 0);
            encoder.set_buffer(LINEAR_Q_RS2_VALUE_SLOT, Some(&self.rows.rs2_value), 0);
            encoder.set_buffer(
                LINEAR_Q_GAMMA_POWERS_SLOT,
                Some(&self.buffers.gamma_powers),
                0,
            );
            encoder.set_buffer(LINEAR_Q_EQ_SUFFIX_SLOT, Some(&self.buffers.eq_suffix), 0);
            encoder.set_buffer(LINEAR_Q_OUTPUT_SLOT, Some(&self.buffers.output), 0);
            set_inline_bytes(encoder, LINEAR_Q_PARAMS_SLOT, &self.plan.params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.threadgroups() as u64,
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
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()).into());
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }

            let q = self.read_q()?;
            let work = self.plan.work()?;
            Ok(RegistersClaimLinearQObservation {
                checksum: registers_claim_q_checksum(&q),
                q,
                useful_half_width_terms: work.half_width_terms,
                full_products: work.full_products,
                useful_threads: self.plan.useful_threads(),
                dispatched_threads: self.plan.dispatched_threads()?,
                gpu_active: Duration::from_secs_f64(end - start),
                resident_wall: wall_started.elapsed(),
            })
        })
    }

    pub const fn plan(&self) -> RegistersClaimLinearQPlan {
        self.plan
    }

    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn source_allocation_identities(&self) -> [usize; 3] {
        self.rows.allocation_identities()
    }

    pub fn output_allocation_identity(&self) -> usize {
        self.buffers.output.as_ptr() as usize
    }

    fn validate_state(&self) -> Result<(), RegistersClaimLinearQError> {
        if self.context.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimLinearQError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.context.offset,
            });
        }
        self.rows.validate_for(&self.context, self.plan.geometry)?;
        if self.plan.params != self.plan.geometry.params()? || self.plan.params.reserved != 0 {
            return Err(RegistersClaimLinearQError::InvalidState(
                "shader parameters differ from checked geometry",
            ));
        }
        if self.limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH
            || self.threads_per_threadgroup != self.plan.threads_per_threadgroup()
            || self.threads_per_threadgroup > self.limits.max_total_threads_per_threadgroup
            || !self
                .threads_per_threadgroup
                .is_multiple_of(self.limits.thread_execution_width)
        {
            return Err(RegistersClaimLinearQError::InvalidState(
                "pipeline limits differ from the prepared dispatch",
            ));
        }

        let expected_device = self.context.device_registry_id();
        for (name, buffer, expected_bytes, expected_identity) in [
            (
                "gamma powers",
                &self.buffers.gamma_powers,
                self.plan.storage.gamma_powers_bytes,
                self.buffer_identities[0],
            ),
            (
                "eq_suffix",
                &self.buffers.eq_suffix,
                self.plan.storage.eq_suffix_bytes,
                self.buffer_identities[1],
            ),
            (
                "q output",
                &self.buffers.output,
                self.plan.storage.output_bytes,
                self.buffer_identities[2],
            ),
        ] {
            validate_buffer_binding(
                buffer,
                name,
                to_u64(expected_bytes)?,
                expected_device,
                expected_identity,
            )?;
        }
        validate_invocation_aliases(self.rows.allocation_identities(), self.buffer_identities)
    }

    fn read_q(&self) -> Result<Vec<AkitaField>, RegistersClaimLinearQError> {
        // SAFETY: the shared output buffer owns exactly `prefix_elements`
        // fields, and command completion precedes this host read.
        let fields = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.plan.geometry.prefix_elements(),
            )
        };
        self.context
            .validate_inputs("registers claim linear-q output", fields)?;
        Ok(fields
            .iter()
            .map(|&value| value.into_jolt_field())
            .collect())
    }
}

impl RegistersClaimDirectFoldInvocation {
    pub fn execute(
        &self,
    ) -> Result<super::RegistersClaimDenseOutputs<AkitaField>, RegistersClaimLinearQError> {
        self.execute_timed().map(|observation| observation.outputs)
    }

    pub fn execute_timed(
        &self,
    ) -> Result<RegistersClaimDirectFoldObservation, RegistersClaimLinearQError> {
        let wall_started = Instant::now();
        self.validate_state()?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(
                DIRECT_FOLD_RD_WRITE_VALUE_SLOT,
                Some(&self.rows.rd_write_value),
                0,
            );
            encoder.set_buffer(DIRECT_FOLD_RS1_VALUE_SLOT, Some(&self.rows.rs1_value), 0);
            encoder.set_buffer(DIRECT_FOLD_RS2_VALUE_SLOT, Some(&self.rows.rs2_value), 0);
            encoder.set_buffer(DIRECT_FOLD_EQ_PREFIX_SLOT, Some(&self.buffers.eq_prefix), 0);
            encoder.set_buffer(
                DIRECT_FOLD_OUTPUT_SLOT,
                Some(&self.buffers.dense_outputs),
                0,
            );
            set_inline_bytes(encoder, DIRECT_FOLD_PARAMS_SLOT, &self.plan.params);
            encoder.set_threadgroup_memory_length(
                DIRECT_FOLD_THREADGROUP_SLOT,
                to_u64(self.dynamic_threadgroup_bytes)?,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.fold_threadgroups() as u64,
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
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()).into());
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }

            let outputs = self.read_outputs()?;
            let work = self
                .plan
                .geometry
                .work(RegistersClaimStrategy::DirectLinear)?
                .fold;
            Ok(RegistersClaimDirectFoldObservation {
                outputs,
                useful_half_width_terms: work.half_width_terms,
                threadgroups: self.plan.fold_threadgroups(),
                gpu_active: Duration::from_secs_f64(end - start),
                resident_wall: wall_started.elapsed(),
            })
        })
    }

    pub const fn plan(&self) -> RegistersClaimPlan {
        self.plan
    }

    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn dynamic_threadgroup_bytes(&self) -> usize {
        self.dynamic_threadgroup_bytes
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    fn validate_state(&self) -> Result<(), RegistersClaimLinearQError> {
        if self.context.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimLinearQError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.context.offset,
            });
        }
        self.rows.validate_for(&self.context, self.plan.geometry)?;
        if self.plan.strategy != RegistersClaimStrategy::DirectLinear
            || self.plan.params != self.plan.geometry.params()?
            || self.plan.params.reserved != 0
        {
            return Err(RegistersClaimLinearQError::InvalidState(
                "direct-fold plan differs from the checked geometry",
            ));
        }
        if self.limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH
            || self.threads_per_threadgroup != self.plan.config.fold_threads_per_threadgroup
            || self.threads_per_threadgroup > self.limits.max_total_threads_per_threadgroup
            || !self
                .threads_per_threadgroup
                .is_multiple_of(self.limits.thread_execution_width)
        {
            return Err(RegistersClaimLinearQError::InvalidState(
                "direct-fold pipeline limits differ from the prepared dispatch",
            ));
        }

        let expected_device = self.context.device_registry_id();
        for (name, buffer, expected_bytes, expected_identity) in [
            (
                "eq_prefix",
                &self.buffers.eq_prefix,
                self.plan.storage.prefix_field_bytes,
                self.buffer_identities[0],
            ),
            (
                "direct dense outputs",
                &self.buffers.dense_outputs,
                self.plan.storage.direct_dense_bytes,
                self.buffer_identities[1],
            ),
        ] {
            validate_buffer_binding(
                buffer,
                name,
                to_u64(expected_bytes)?,
                expected_device,
                expected_identity,
            )?;
        }
        validate_direct_fold_aliases(self.rows.allocation_identities(), self.buffer_identities)
    }

    fn read_outputs(
        &self,
    ) -> Result<super::RegistersClaimDenseOutputs<AkitaField>, RegistersClaimLinearQError> {
        let column_elements = self.plan.geometry.suffix_elements();
        let elements = column_elements
            .checked_mul(REGISTERS_CLAIM_OUTPUT_COLUMNS)
            .ok_or(MetalError::InputTooLong(column_elements))?;
        // SAFETY: the output buffer owns exactly three dense columns and the
        // command is complete before this shared-buffer read.
        let fields = unsafe {
            slice::from_raw_parts(
                self.buffers.dense_outputs.contents().cast::<Fp128>(),
                elements,
            )
        };
        self.context
            .validate_inputs("registers claim direct-fold output", fields)?;
        let column = |index: usize| {
            fields[index * column_elements..(index + 1) * column_elements]
                .iter()
                .map(|&value| value.into_jolt_field())
                .collect()
        };
        Ok(super::RegistersClaimDenseOutputs {
            rd_write_value: column(0),
            rs1_value: column(1),
            rs2_value: column(2),
        })
    }
}

impl RegistersClaimAliasFoldInvocation {
    pub(crate) fn execute_timed(
        &self,
    ) -> Result<RegistersClaimAliasFoldObservation, RegistersClaimLinearQError> {
        let wall_started = Instant::now();
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
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()).into());
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }
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
                useful_half_width_terms: self.geometry.rows() as u64,
                gpu_active: Duration::from_secs_f64(end - start),
                resident_wall: wall_started.elapsed(),
            })
        })
    }

    fn validate_state(&self) -> Result<(), RegistersClaimLinearQError> {
        self.rd.validate_for(&self.context, self.geometry)?;
        if self.params != self.geometry.params()?
            || self.limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH
            || self.threads_per_threadgroup > self.limits.max_total_threads_per_threadgroup
            || !self
                .threads_per_threadgroup
                .is_multiple_of(self.limits.thread_execution_width)
        {
            return Err(RegistersClaimLinearQError::InvalidState(
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

pub fn registers_claim_q_checksum(values: &[AkitaField]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    values.iter().fold(OFFSET_BASIS, |mut checksum, value| {
        let bytes = Fp128::from_jolt_field(value).to_u128().to_le_bytes();
        for byte in bytes {
            checksum ^= u64::from(byte);
            checksum = checksum.wrapping_mul(PRIME);
        }
        checksum
    })
}

fn validate_distinct_three(identities: [usize; 3]) -> Result<(), RegistersClaimLinearQError> {
    if identities[0] == identities[1]
        || identities[0] == identities[2]
        || identities[1] == identities[2]
    {
        Err(RegistersClaimLinearQError::AliasedResidentPlanes)
    } else {
        Ok(())
    }
}

fn validate_invocation_aliases(
    resident: [usize; 3],
    invocation: [usize; 3],
) -> Result<(), RegistersClaimLinearQError> {
    let mut identities = [0usize; 6];
    identities[..3].copy_from_slice(&resident);
    identities[3..].copy_from_slice(&invocation);
    for left in 0..identities.len() {
        for right in left + 1..identities.len() {
            if identities[left] == identities[right] {
                return Err(RegistersClaimLinearQError::AliasedInvocationBuffers);
            }
        }
    }
    Ok(())
}

fn validate_direct_fold_aliases(
    resident: [usize; 3],
    invocation: [usize; 2],
) -> Result<(), RegistersClaimLinearQError> {
    let identities = [
        resident[0],
        resident[1],
        resident[2],
        invocation[0],
        invocation[1],
    ];
    for left in 0..identities.len() {
        for right in left + 1..identities.len() {
            if identities[left] == identities[right] {
                return Err(RegistersClaimLinearQError::AliasedInvocationBuffers);
            }
        }
    }
    Ok(())
}

fn validate_buffer_binding(
    buffer: &Buffer,
    name: &'static str,
    expected_bytes: u64,
    expected_device: u64,
    expected_identity: usize,
) -> Result<(), RegistersClaimLinearQError> {
    let got_device = buffer.device().registry_id();
    if got_device != expected_device {
        return Err(RegistersClaimLinearQError::BufferDevice {
            name,
            expected: expected_device,
            got: got_device,
        });
    }
    if buffer.length() != expected_bytes {
        return Err(RegistersClaimLinearQError::BufferLength {
            name,
            expected: expected_bytes,
            actual: buffer.length(),
        });
    }
    if buffer.as_ptr() as usize != expected_identity {
        return Err(RegistersClaimLinearQError::InvalidState(
            "buffer allocation identity changed",
        ));
    }
    Ok(())
}

fn encode_fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

fn to_u64(value: usize) -> Result<u64, MetalError> {
    u64::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}
