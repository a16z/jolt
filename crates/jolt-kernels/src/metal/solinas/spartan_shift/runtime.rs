use std::{mem::size_of, slice, time::Duration};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLResourceOptions, MTLSize,
};

use super::super::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};
use super::{
    mixed_gamma_multipliers, mixed_high_weights, prefix_fold_weights,
    ResidentSpartanShiftBufferMetadata, ResidentSpartanShiftMetadata, SpartanShiftFlagWord,
    SpartanShiftGeometry, SpartanShiftKernelConfig, SpartanShiftOutputs, SpartanShiftPlan,
    BUILD_MIXED_PIPELINE, FOLD_NATIVE_PIPELINE, REDUCE_PREFIX_PIPELINE,
    SPARTAN_SHIFT_OUTPUT_COLUMNS, SPARTAN_SHIFT_PREFIX_PAIRS, SPARTAN_SHIFT_SIMD_WIDTH,
};

#[derive(Clone)]
pub struct SpartanShiftResidentRows {
    unexpanded_pc: Buffer,
    pc: Buffer,
    flags: Buffer,
    metadata: ResidentSpartanShiftMetadata,
}

impl SpartanShiftResidentRows {
    pub const fn len(&self) -> usize {
        self.metadata.rows
    }

    pub const fn is_empty(&self) -> bool {
        self.metadata.rows == 0
    }

    pub const fn device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id
    }

    pub const fn resident_bytes(&self) -> usize {
        self.metadata.unexpanded_pc.byte_len
            + self.metadata.pc.byte_len
            + self.metadata.flags.byte_len
    }

    pub fn allocation_identities(&self) -> [usize; 3] {
        [
            self.unexpanded_pc.as_ptr() as usize,
            self.pc.as_ptr() as usize,
            self.flags.as_ptr() as usize,
        ]
    }

    pub(crate) fn source_buffers(&self) -> [&Buffer; 3] {
        [&self.unexpanded_pc, &self.pc, &self.flags]
    }

    fn validate_for(
        &self,
        context: &SolinasMetal,
        plan: &SpartanShiftPlan,
    ) -> Result<(), MetalError> {
        let metadata = self
            .metadata
            .validate(plan.geometry, context.device_registry_id())?;
        if self.allocation_identities()
            != [
                metadata.unexpanded_pc.allocation_identity,
                metadata.pc.allocation_identity,
                metadata.flags.allocation_identity,
            ]
        {
            return Err(MetalError::InvalidSpartanShiftState(
                "resident allocation identity changed",
            ));
        }
        Ok(())
    }
}

struct SpartanShiftPrefixBuffers {
    gamma_powers: Buffer,
    high_weights: Buffer,
    partials: Buffer,
    q: Buffer,
}

pub struct SpartanShiftPrefixInvocation {
    context: SolinasMetal,
    rows: SpartanShiftResidentRows,
    build_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    reduce_threads: usize,
    buffers: SpartanShiftPrefixBuffers,
    plan: SpartanShiftPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanShiftPrefixObservation {
    pub q: [Vec<AkitaField>; SPARTAN_SHIFT_PREFIX_PAIRS],
    pub gpu_active: Duration,
}

struct SpartanShiftFoldBuffers {
    low_weights: Buffer,
    dense_outputs: Buffer,
}

pub struct SpartanShiftFoldInvocation {
    context: SolinasMetal,
    rows: SpartanShiftResidentRows,
    pipeline: ComputePipelineState,
    limits: PipelineLimits,
    buffers: SpartanShiftFoldBuffers,
    plan: SpartanShiftPlan,
    dynamic_threadgroup_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanShiftFoldObservation {
    pub outputs: SpartanShiftOutputs<Vec<AkitaField>>,
    pub gpu_active: Duration,
}

struct SpartanShiftSubmittedCommand {
    command_buffer: CommandBuffer,
    source_allocation_identities: [usize; 3],
    output_allocation_identity: usize,
}

#[must_use = "a submitted Spartan shift prefix command must be joined"]
pub struct PendingSpartanShiftPrefix {
    invocation: Option<SpartanShiftPrefixInvocation>,
    command: Option<SpartanShiftSubmittedCommand>,
}

#[must_use = "a submitted Spartan shift fold command must be joined"]
pub struct PendingSpartanShiftFold {
    invocation: Option<SpartanShiftFoldInvocation>,
    command: Option<SpartanShiftSubmittedCommand>,
}

impl Drop for PendingSpartanShiftPrefix {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

impl Drop for PendingSpartanShiftFold {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

impl PendingSpartanShiftPrefix {
    pub fn join(
        mut self,
    ) -> Result<(SpartanShiftPrefixInvocation, SpartanShiftPrefixObservation), MetalError> {
        let invocation = self
            .invocation
            .take()
            .ok_or(MetalError::InvalidSpartanShiftState(
                "submitted prefix lost its invocation",
            ))?;
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidSpartanShiftState(
                "submitted prefix lost its command",
            ))?;
        let observation = invocation.complete(command)?;
        Ok((invocation, observation))
    }
}

impl PendingSpartanShiftFold {
    pub fn join(
        mut self,
    ) -> Result<(SpartanShiftFoldInvocation, SpartanShiftFoldObservation), MetalError> {
        let invocation = self
            .invocation
            .take()
            .ok_or(MetalError::InvalidSpartanShiftState(
                "submitted fold lost its invocation",
            ))?;
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidSpartanShiftState(
                "submitted fold lost its command",
            ))?;
        let observation = invocation.complete(command)?;
        Ok((invocation, observation))
    }
}

impl SolinasMetal {
    pub fn prepare_spartan_shift_rows(
        &self,
        unexpanded_pc: &[u64],
        pc: &[u64],
        flags: &[SpartanShiftFlagWord],
        exact_current_flags: bool,
    ) -> Result<SpartanShiftResidentRows, MetalError> {
        let geometry = SpartanShiftGeometry::new(unexpanded_pc.len())?;
        for (name, actual, expected) in [
            ("unexpanded PC", unexpanded_pc.len(), geometry.rows()),
            ("PC", pc.len(), geometry.rows()),
            ("flag words", flags.len(), geometry.flag_words()),
        ] {
            if actual != expected {
                return Err(super::SpartanShiftPlanError::WrongLength {
                    name,
                    expected,
                    actual,
                }
                .into());
            }
        }

        self.prepare_spartan_shift_rows_with_fill(
            geometry.rows(),
            exact_current_flags,
            |destination_unexpanded_pc, destination_pc, destination_flags| {
                destination_unexpanded_pc.copy_from_slice(unexpanded_pc);
                destination_pc.copy_from_slice(pc);
                destination_flags.copy_from_slice(flags);
                Ok(())
            },
        )
    }

    pub(crate) fn prepare_spartan_shift_rows_with_fill(
        &self,
        rows: usize,
        exact_current_flags: bool,
        fill: impl FnOnce(&mut [u64], &mut [u64], &mut [SpartanShiftFlagWord]) -> Result<(), MetalError>,
    ) -> Result<SpartanShiftResidentRows, MetalError> {
        let geometry = SpartanShiftGeometry::new(rows)?;

        let value_bytes = geometry
            .rows()
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::InputTooLong(geometry.rows()))?;
        let flag_bytes = geometry
            .flag_words()
            .checked_mul(size_of::<SpartanShiftFlagWord>())
            .ok_or(MetalError::InputTooLong(geometry.rows()))?;
        let value_bytes_u64 =
            u64::try_from(value_bytes).map_err(|_| MetalError::InputTooLong(value_bytes))?;
        let flag_bytes_u64 =
            u64::try_from(flag_bytes).map_err(|_| MetalError::InputTooLong(flag_bytes))?;
        for bytes in [value_bytes_u64, flag_bytes_u64] {
            self.validate_buffer_length(bytes)?;
        }
        let resident_bytes = value_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(flag_bytes))
            .ok_or(MetalError::InputTooLong(geometry.rows()))?;
        self.validate_additional_working_set(
            u64::try_from(resident_bytes).map_err(|_| MetalError::InputTooLong(resident_bytes))?,
        )?;

        let unexpanded_pc = self
            .device
            .new_buffer(value_bytes_u64, MTLResourceOptions::StorageModeShared);
        let pc = self
            .device
            .new_buffer(value_bytes_u64, MTLResourceOptions::StorageModeShared);
        let flags = self
            .device
            .new_buffer(flag_bytes_u64, MTLResourceOptions::StorageModeShared);

        // SAFETY: the shared buffers above have exactly the element counts used
        // below and are not submitted to Metal until after `fill` returns.
        let destination_unexpanded_pc = unsafe {
            slice::from_raw_parts_mut(unexpanded_pc.contents().cast::<u64>(), geometry.rows())
        };
        // SAFETY: see the allocation and submission invariant above.
        let destination_pc =
            unsafe { slice::from_raw_parts_mut(pc.contents().cast::<u64>(), geometry.rows()) };
        // SAFETY: see the allocation and submission invariant above.
        let destination_flags = unsafe {
            slice::from_raw_parts_mut(
                flags.contents().cast::<SpartanShiftFlagWord>(),
                geometry.flag_words(),
            )
        };
        fill(destination_unexpanded_pc, destination_pc, destination_flags)?;

        self.attach_spartan_shift_rows(
            unexpanded_pc,
            pc,
            flags,
            geometry.rows(),
            exact_current_flags,
        )
    }

    pub fn attach_spartan_shift_rows(
        &self,
        unexpanded_pc: Buffer,
        pc: Buffer,
        flags: Buffer,
        rows: usize,
        exact_current_flags: bool,
    ) -> Result<SpartanShiftResidentRows, MetalError> {
        let geometry = SpartanShiftGeometry::new(rows)?;
        let value_bytes = rows
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::InputTooLong(rows))?;
        let flag_bytes = geometry
            .flag_words()
            .checked_mul(size_of::<SpartanShiftFlagWord>())
            .ok_or(MetalError::InputTooLong(rows))?;
        let expected_device = self.device_registry_id();
        for (name, buffer, expected_bytes) in [
            ("unexpanded PC", &unexpanded_pc, value_bytes),
            ("PC", &pc, value_bytes),
            ("flags", &flags, flag_bytes),
        ] {
            let got_device = buffer.device().registry_id();
            if got_device != expected_device {
                return Err(MetalError::SpartanShiftBufferDevice {
                    name,
                    expected: expected_device,
                    got: got_device,
                });
            }
            self.validate_buffer_length(buffer.length())?;
            let expected_length = u64::try_from(expected_bytes)
                .map_err(|_| MetalError::InputTooLong(expected_bytes))?;
            if buffer.length() != expected_length {
                return Err(super::SpartanShiftPlanError::WrongLength {
                    name,
                    expected: expected_bytes,
                    actual: usize::try_from(buffer.length()).unwrap_or(usize::MAX),
                }
                .into());
            }
        }
        let metadata = ResidentSpartanShiftMetadata {
            rows: geometry.rows(),
            unexpanded_pc: ResidentSpartanShiftBufferMetadata {
                allocation_identity: unexpanded_pc.as_ptr() as usize,
                byte_len: value_bytes,
            },
            pc: ResidentSpartanShiftBufferMetadata {
                allocation_identity: pc.as_ptr() as usize,
                byte_len: value_bytes,
            },
            flags: ResidentSpartanShiftBufferMetadata {
                allocation_identity: flags.as_ptr() as usize,
                byte_len: flag_bytes,
            },
            device_registry_id: self.device_registry_id(),
            exact_current_flags,
        }
        .validate(geometry, self.device_registry_id())?;

        Ok(SpartanShiftResidentRows {
            unexpanded_pc,
            pc,
            flags,
            metadata,
        })
    }

    pub fn prepare_spartan_shift_prefix(
        &self,
        rows: &SpartanShiftResidentRows,
        r_outer: &[AkitaField],
        r_product: &[AkitaField],
        gamma: AkitaField,
        config: SpartanShiftKernelConfig,
    ) -> Result<SpartanShiftPrefixInvocation, MetalError> {
        let plan = SpartanShiftPlan::new(rows.len(), config)?;
        rows.validate_for(self, &plan)?;
        let gamma_powers = encode_fields(&mixed_gamma_multipliers(gamma));
        let high_weights = encode_fields(&mixed_high_weights(
            plan.geometry,
            r_outer,
            r_product,
            gamma,
        )?);

        let private_bytes = plan
            .storage
            .high_weight_bytes
            .checked_add(plan.storage.partial_bytes)
            .and_then(|bytes| bytes.checked_add(plan.storage.q_bytes))
            .and_then(|bytes| bytes.checked_add(size_of_val(&gamma_powers[..])))
            .ok_or(MetalError::InputTooLong(rows.len()))?;
        validate_allocations(
            self,
            private_bytes,
            &[
                plan.storage.high_weight_bytes,
                plan.storage.partial_bytes,
                plan.storage.q_bytes,
            ],
        )?;

        let build_name = BUILD_MIXED_PIPELINE;
        let build_pipeline = self.compile_named_pipeline(build_name)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PREFIX_PIPELINE)?;
        let build_limits = Self::limits(&build_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        validate_execution_width(build_name, build_limits)?;
        validate_execution_width(REDUCE_PREFIX_PIPELINE, reduce_limits)?;
        let build_threads = Self::resolve_threadgroup_width(
            Some(plan.config.build_threads_per_threadgroup),
            build_limits,
        )?;
        if build_threads != plan.config.build_threads_per_threadgroup {
            return Err(MetalError::InvalidSpartanShiftState(
                "resolved build width differs from checked plan",
            ));
        }
        let reduce_threads = Self::resolve_threadgroup_width(None, reduce_limits)?;

        Ok(SpartanShiftPrefixInvocation {
            context: self.clone(),
            rows: rows.clone(),
            build_pipeline,
            reduce_pipeline,
            reduce_threads,
            buffers: SpartanShiftPrefixBuffers {
                gamma_powers: buffer_from_slice(&self.device, &gamma_powers),
                high_weights: buffer_from_slice(&self.device, &high_weights),
                partials: self.device.new_buffer(
                    plan.storage.partial_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                q: self.device.new_buffer(
                    plan.storage.q_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            plan,
        })
    }

    pub fn prepare_spartan_shift_fold(
        &self,
        rows: &SpartanShiftResidentRows,
        prefix_challenges: &[AkitaField],
        config: SpartanShiftKernelConfig,
    ) -> Result<SpartanShiftFoldInvocation, MetalError> {
        let plan = SpartanShiftPlan::new(rows.len(), config)?;
        rows.validate_for(self, &plan)?;
        let low_weights = encode_fields(&prefix_fold_weights(plan.geometry, prefix_challenges)?);
        self.validate_inputs("Spartan shift low weights", &low_weights)?;
        let private_bytes = plan
            .storage
            .low_weight_bytes
            .checked_add(plan.storage.dense_output_bytes)
            .ok_or(MetalError::InputTooLong(rows.len()))?;
        validate_allocations(
            self,
            private_bytes,
            &[
                plan.storage.low_weight_bytes,
                plan.storage.dense_output_bytes,
            ],
        )?;

        let pipeline = self.compile_named_pipeline(FOLD_NATIVE_PIPELINE)?;
        let limits = Self::limits(&pipeline);
        validate_execution_width(FOLD_NATIVE_PIPELINE, limits)?;
        let threads = Self::resolve_threadgroup_width(
            Some(plan.config.fold_threads_per_threadgroup),
            limits,
        )?;
        if threads != plan.config.fold_threads_per_threadgroup {
            return Err(MetalError::InvalidSpartanShiftState(
                "resolved fold width differs from checked plan",
            ));
        }
        let dynamic_threadgroup_bytes = plan.config.fold_threadgroup_bytes()?;
        let total_threadgroup_bytes = u64::try_from(dynamic_threadgroup_bytes)
            .ok()
            .and_then(|dynamic| dynamic.checked_add(limits.static_threadgroup_memory_length))
            .ok_or(MetalError::InputTooLong(dynamic_threadgroup_bytes))?;
        if total_threadgroup_bytes > self.device.max_threadgroup_memory_length() {
            return Err(MetalError::SpartanShiftThreadgroupMemory {
                requested: total_threadgroup_bytes,
                maximum: self.device.max_threadgroup_memory_length(),
            });
        }

        Ok(SpartanShiftFoldInvocation {
            context: self.clone(),
            rows: rows.clone(),
            pipeline,
            limits,
            buffers: SpartanShiftFoldBuffers {
                low_weights: buffer_from_slice(&self.device, &low_weights),
                dense_outputs: self.device.new_buffer(
                    plan.storage.dense_output_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            plan,
            dynamic_threadgroup_bytes,
        })
    }
}

impl SpartanShiftPrefixInvocation {
    pub fn execute(&self) -> Result<SpartanShiftPrefixObservation, MetalError> {
        let command = self.submit_command()?;
        self.complete(command)
    }

    pub fn submit(self) -> Result<PendingSpartanShiftPrefix, MetalError> {
        let command = self.submit_command()?;
        Ok(PendingSpartanShiftPrefix {
            invocation: Some(self),
            command: Some(command),
        })
    }

    fn submit_command(&self) -> Result<SpartanShiftSubmittedCommand, MetalError> {
        self.rows.validate_for(&self.context, &self.plan)?;
        let command_buffer = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| -> Result<(), MetalError> {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.build_pipeline);
            encoder.set_buffer(0, Some(&self.rows.unexpanded_pc), 0);
            encoder.set_buffer(1, Some(&self.rows.pc), 0);
            encoder.set_buffer(2, Some(&self.rows.flags), 0);
            encoder.set_buffer(3, Some(&self.buffers.gamma_powers), 0);
            encoder.set_buffer(4, Some(&self.buffers.high_weights), 0);
            encoder.set_buffer(5, Some(&self.buffers.partials), 0);
            set_inline_bytes(encoder, 6, &self.plan.params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.build_threadgroups() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.plan.config.build_threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );

            encoder.set_compute_pipeline_state(&self.reduce_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.partials), 0);
            encoder.set_buffer(1, Some(&self.buffers.q), 0);
            set_inline_bytes(encoder, 2, &self.plan.params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self
                        .plan
                        .geometry
                        .prefix_elements()
                        .div_ceil(self.reduce_threads) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.reduce_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            Ok(())
        })?;
        Ok(SpartanShiftSubmittedCommand {
            command_buffer,
            source_allocation_identities: self.rows.allocation_identities(),
            output_allocation_identity: self.buffers.q.as_ptr() as usize,
        })
    }

    fn complete(
        &self,
        command: SpartanShiftSubmittedCommand,
    ) -> Result<SpartanShiftPrefixObservation, MetalError> {
        validate_command_resources(
            &self.rows,
            &self.buffers.q,
            &command,
            "prefix resources changed before completion",
        )?;
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        let q = read_field_columns(
            &self.context,
            &self.buffers.q,
            self.plan.geometry.prefix_elements(),
            "Spartan shift prefix Q",
        )?;
        Ok(SpartanShiftPrefixObservation { q, gpu_active })
    }

    copy_field_getters! { pub, { plan: SpartanShiftPlan }}

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn source_allocation_identities(&self) -> [usize; 3] {
        self.rows.allocation_identities()
    }
}

impl SpartanShiftFoldInvocation {
    pub fn execute(&self) -> Result<SpartanShiftFoldObservation, MetalError> {
        let command = self.submit_command()?;
        self.complete(command)
    }

    pub fn submit(self) -> Result<PendingSpartanShiftFold, MetalError> {
        let command = self.submit_command()?;
        Ok(PendingSpartanShiftFold {
            invocation: Some(self),
            command: Some(command),
        })
    }

    fn submit_command(&self) -> Result<SpartanShiftSubmittedCommand, MetalError> {
        self.rows.validate_for(&self.context, &self.plan)?;
        let command_buffer = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&self.rows.unexpanded_pc), 0);
            encoder.set_buffer(1, Some(&self.rows.pc), 0);
            encoder.set_buffer(2, Some(&self.rows.flags), 0);
            encoder.set_buffer(3, Some(&self.buffers.low_weights), 0);
            encoder.set_buffer(4, Some(&self.buffers.dense_outputs), 0);
            set_inline_bytes(encoder, 5, &self.plan.params);
            encoder.set_threadgroup_memory_length(0, self.dynamic_threadgroup_bytes as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.fold_threadgroups() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.plan.config.fold_threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
        });
        Ok(SpartanShiftSubmittedCommand {
            command_buffer,
            source_allocation_identities: self.rows.allocation_identities(),
            output_allocation_identity: self.buffers.dense_outputs.as_ptr() as usize,
        })
    }

    fn complete(
        &self,
        command: SpartanShiftSubmittedCommand,
    ) -> Result<SpartanShiftFoldObservation, MetalError> {
        validate_command_resources(
            &self.rows,
            &self.buffers.dense_outputs,
            &command,
            "fold resources changed before completion",
        )?;
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        let columns: [Vec<AkitaField>; SPARTAN_SHIFT_OUTPUT_COLUMNS] = read_field_columns(
            &self.context,
            &self.buffers.dense_outputs,
            self.plan.geometry.suffix_elements(),
            "Spartan shift dense outputs",
        )?;
        let [unexpanded_pc, pc, is_virtual, is_first_in_sequence, is_noop] = columns;
        Ok(SpartanShiftFoldObservation {
            outputs: SpartanShiftOutputs {
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
            },
            gpu_active,
        })
    }

    copy_field_getters! { pub, {
        plan: SpartanShiftPlan,
        pipeline_limits => limits: PipelineLimits,
        dynamic_threadgroup_bytes: usize,
    }}

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn source_allocation_identities(&self) -> [usize; 3] {
        self.rows.allocation_identities()
    }
}

fn encode_fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn validate_allocations(
    context: &SolinasMetal,
    total_bytes: usize,
    buffer_bytes: &[usize],
) -> Result<(), MetalError> {
    context.validate_additional_working_set(
        u64::try_from(total_bytes).map_err(|_| MetalError::InputTooLong(total_bytes))?,
    )?;
    for &bytes in buffer_bytes {
        context.validate_buffer_length(
            u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(bytes))?,
        )?;
    }
    Ok(())
}

fn validate_execution_width(
    pipeline: &'static str,
    limits: PipelineLimits,
) -> Result<(), MetalError> {
    if limits.thread_execution_width != SPARTAN_SHIFT_SIMD_WIDTH {
        return Err(MetalError::UnsupportedSpartanShiftExecutionWidth {
            pipeline,
            expected: SPARTAN_SHIFT_SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    Ok(())
}

fn validate_command_resources(
    rows: &SpartanShiftResidentRows,
    output: &Buffer,
    command: &SpartanShiftSubmittedCommand,
    error: &'static str,
) -> Result<(), MetalError> {
    if rows.allocation_identities() != command.source_allocation_identities
        || output.as_ptr() as usize != command.output_allocation_identity
    {
        return Err(MetalError::InvalidSpartanShiftState(error));
    }
    Ok(())
}

fn read_field_columns<const COLUMNS: usize>(
    context: &SolinasMetal,
    buffer: &Buffer,
    elements: usize,
    side: &'static str,
) -> Result<[Vec<AkitaField>; COLUMNS], MetalError> {
    let total = elements
        .checked_mul(COLUMNS)
        .ok_or(MetalError::InputTooLong(elements))?;
    // SAFETY: the shared buffer owns exactly `total` fields, and the command
    // completed before this immutable host read.
    let output = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), total) };
    context.validate_inputs(side, output)?;
    Ok(std::array::from_fn(|column| {
        output[column * elements..(column + 1) * elements]
            .iter()
            .map(|&value| value.into_jolt_field())
            .collect()
    }))
}
