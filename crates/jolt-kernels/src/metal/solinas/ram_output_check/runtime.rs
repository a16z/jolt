use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use super::{
    RamOutputCheckFoldParams, RamOutputCheckHybridPlan, RamOutputCheckPlanError,
    RamOutputCheckStorage, ResidentRamFinalMetadata, RAM_OUTPUT_CHECK_FOLD_PIPELINE,
    RAM_OUTPUT_CHECK_REDUCE_PIPELINE, RAM_OUTPUT_CHECK_SIMD_WIDTH,
};

#[derive(Clone)]
pub struct ResidentRamFinalValues {
    buffer: Buffer,
    metadata: ResidentRamFinalMetadata,
}

impl ResidentRamFinalValues {
    pub const fn len(&self) -> usize {
        self.metadata.elements
    }

    pub const fn is_empty(&self) -> bool {
        self.metadata.elements == 0
    }

    pub const fn resident_bytes(&self) -> usize {
        self.metadata.bytes
    }

    pub const fn device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id
    }

    pub fn allocation_identity(&self) -> usize {
        self.buffer.as_ptr() as usize
    }

    pub const fn public_io_certified(&self) -> bool {
        self.metadata.public_io_certified
    }

    pub fn as_slice(&self) -> &[u64] {
        // SAFETY: the shared buffer owns `elements` initialized `u64` values
        // and this immutable owner submits no device writes to it.
        unsafe {
            slice::from_raw_parts(self.buffer.contents().cast::<u64>(), self.metadata.elements)
        }
    }

    fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

struct RamOutputCheckBuffers {
    weights: Buffer,
    partials: Buffer,
    output: Buffer,
}

pub struct RamOutputCheckFold {
    context: SolinasMetal,
    partial_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    partial_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    values: ResidentRamFinalValues,
    buffers: RamOutputCheckBuffers,
    params: RamOutputCheckFoldParams,
    plan: RamOutputCheckHybridPlan,
    storage: RamOutputCheckStorage,
}

impl SolinasMetal {
    pub fn prepare_ram_output_check_values(
        &self,
        values: &[u64],
        public_io_certified: bool,
        plan: RamOutputCheckHybridPlan,
    ) -> Result<ResidentRamFinalValues, MetalError> {
        if values.len() != plan.addresses() {
            return Err(RamOutputCheckPlanError::Length {
                name: "resident native RamValFinal",
                expected: plan.addresses(),
                got: values.len(),
            }
            .into());
        }
        let storage = plan.storage()?;
        let bytes = u64::try_from(storage.borrowed_input_bytes)
            .map_err(|_| MetalError::InputTooLong(storage.borrowed_input_bytes))?;
        self.validate_buffer_length(bytes)?;
        self.validate_additional_working_set(bytes)?;
        let buffer = buffer_from_slice(&self.device, values);
        let metadata = ResidentRamFinalMetadata {
            elements: values.len(),
            bytes: storage.borrowed_input_bytes,
            device_registry_id: self.device_registry_id(),
            allocation_identity: buffer.as_ptr() as usize,
            public_io_certified,
        }
        .validate(plan)?;
        Ok(ResidentRamFinalValues { buffer, metadata })
    }

    pub fn prepare_ram_output_check_fold(
        &self,
        values: &ResidentRamFinalValues,
        prefix_challenges: &[AkitaField],
        plan: RamOutputCheckHybridPlan,
    ) -> Result<RamOutputCheckFold, MetalError> {
        let _ = values.metadata.validate(plan)?;
        if values.device_registry_id() != self.device_registry_id() {
            return Err(MetalError::RamOutputCheckValuesDevice {
                expected: self.device_registry_id(),
                got: values.device_registry_id(),
            });
        }
        if values.allocation_identity() != values.metadata.allocation_identity {
            return Err(MetalError::InvalidRamOutputCheckState(
                "resident allocation identity changed",
            ));
        }

        let weights = plan
            .low_weights(prefix_challenges)?
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        self.validate_inputs("RAM output-check low weights", &weights)?;
        let storage = plan.storage()?;
        let private_bytes = u64::try_from(storage.private_bytes)
            .map_err(|_| MetalError::InputTooLong(storage.private_bytes))?;
        self.validate_additional_working_set(private_bytes)?;
        for bytes in [
            storage.weight_bytes,
            storage.partial_bytes,
            storage.output_bytes,
        ] {
            self.validate_buffer_length(
                u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(bytes))?,
            )?;
        }

        let partial_pipeline = self.compile_named_pipeline(RAM_OUTPUT_CHECK_FOLD_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(RAM_OUTPUT_CHECK_REDUCE_PIPELINE)?;
        let partial_limits = Self::limits(&partial_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        for (pipeline, limits) in [
            (RAM_OUTPUT_CHECK_FOLD_PIPELINE, partial_limits),
            (RAM_OUTPUT_CHECK_REDUCE_PIPELINE, reduce_limits),
        ] {
            if limits.thread_execution_width != RAM_OUTPUT_CHECK_SIMD_WIDTH {
                return Err(MetalError::UnsupportedRamOutputCheckExecutionWidth {
                    pipeline,
                    expected: RAM_OUTPUT_CHECK_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads =
            Self::resolve_threadgroup_width(Some(plan.threads_per_threadgroup()), partial_limits)?;
        if threads != plan.threads_per_threadgroup() {
            return Err(MetalError::InvalidRamOutputCheckState(
                "resolved thread width differs from checked plan",
            ));
        }
        let dynamic_threadgroup_bytes = threads / RAM_OUTPUT_CHECK_SIMD_WIDTH * size_of::<Fp128>();
        let total_threadgroup_bytes = u64::try_from(dynamic_threadgroup_bytes)
            .ok()
            .and_then(|dynamic| {
                dynamic.checked_add(partial_limits.static_threadgroup_memory_length)
            })
            .ok_or(MetalError::InputTooLong(dynamic_threadgroup_bytes))?;
        if total_threadgroup_bytes > self.device.max_threadgroup_memory_length() {
            return Err(MetalError::RamOutputCheckThreadgroupMemory {
                requested: total_threadgroup_bytes,
                maximum: self.device.max_threadgroup_memory_length(),
            });
        }

        let params = plan.shader_params()?;
        Ok(RamOutputCheckFold {
            context: self.clone(),
            partial_pipeline,
            reduce_pipeline,
            partial_limits,
            reduce_limits,
            values: values.clone(),
            buffers: RamOutputCheckBuffers {
                weights: buffer_from_slice(&self.device, &weights),
                partials: self.device.new_buffer(
                    storage.partial_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                output: self.device.new_buffer(
                    storage.output_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            params,
            plan,
            storage,
        })
    }
}

impl RamOutputCheckFold {
    pub fn execute(&self) -> Result<Vec<AkitaField>, MetalError> {
        self.execute_timed().map(|(values, _)| values)
    }

    pub fn execute_timed(&self) -> Result<(Vec<AkitaField>, Duration), MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.partial_pipeline);
            encoder.set_buffer(0, Some(self.values.buffer()), 0);
            encoder.set_buffer(1, Some(&self.buffers.weights), 0);
            encoder.set_buffer(2, Some(&self.buffers.partials), 0);
            set_inline_bytes(encoder, 3, &self.params);
            encoder.set_threadgroup_memory_length(0, self.dynamic_threadgroup_bytes() as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.partial_count() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.plan.threads_per_threadgroup() as u64,
                    height: 1,
                    depth: 1,
                },
            );

            encoder.set_compute_pipeline_state(&self.reduce_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.partials), 0);
            encoder.set_buffer(1, Some(&self.buffers.output), 0);
            set_inline_bytes(encoder, 2, &self.params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self
                        .plan
                        .tail_elements()
                        .div_ceil(RAM_OUTPUT_CHECK_SIMD_WIDTH) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: RAM_OUTPUT_CHECK_SIMD_WIDTH as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            Ok((self.read_output()?, Duration::from_secs_f64(end - start)))
        })
    }

    pub const fn plan(&self) -> RamOutputCheckHybridPlan {
        self.plan
    }

    pub const fn storage(&self) -> RamOutputCheckStorage {
        self.storage
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn source_allocation_identity(&self) -> usize {
        self.values.allocation_identity()
    }

    pub const fn partial_pipeline_limits(&self) -> PipelineLimits {
        self.partial_limits
    }

    pub const fn reduce_pipeline_limits(&self) -> PipelineLimits {
        self.reduce_limits
    }

    pub const fn dynamic_threadgroup_bytes(&self) -> usize {
        self.plan.threads_per_threadgroup() / RAM_OUTPUT_CHECK_SIMD_WIDTH * size_of::<Fp128>()
    }

    fn read_output(&self) -> Result<Vec<AkitaField>, MetalError> {
        // SAFETY: the shared output buffer owns exactly `tail_elements` fields,
        // and the command completed before this host read.
        let output = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.plan.tail_elements(),
            )
        };
        self.context
            .validate_inputs("RAM output-check folded values", output)?;
        Ok(output
            .iter()
            .map(|&value| value.into_jolt_field())
            .collect())
    }
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}
