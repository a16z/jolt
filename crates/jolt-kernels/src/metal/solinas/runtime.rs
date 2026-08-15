use std::{ffi::c_void, mem::size_of, time::Duration};

use super::{source::library_source, Fp128, MetalError, AKITA_OFFSET_FFFFA7F7, OFFSET_275};
use metal::{
    objc::{runtime::Sel, Message},
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(crate) fn set_inline_bytes<T>(
    encoder: &metal::ComputeCommandEncoderRef,
    index: u64,
    value: &T,
) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<c_void>(),
    );
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ColumnReductionParams {
    input_count: u32,
    output_count: u32,
    columns: u32,
    reserved: u32,
}

const _: [(); 16] = [(); size_of::<ColumnReductionParams>()];

pub(crate) trait ReductionBuffer {
    fn bind_reduction(&self, encoder: &ComputeCommandEncoderRef, index: u64);
}

impl ReductionBuffer for Buffer {
    fn bind_reduction(&self, encoder: &ComputeCommandEncoderRef, index: u64) {
        encoder.set_buffer(index, Some(self), 0);
    }
}

pub(crate) fn encode_column_reductions<B: ReductionBuffer>(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    partial_a: &B,
    partial_b: &B,
    mut input_count: usize,
    columns: usize,
    width: usize,
) -> Result<bool, MetalError> {
    let columns = u32::try_from(columns).map_err(|_| MetalError::InputTooLong(columns))?;
    let mut input_a = true;
    while input_count > 1 {
        let output_count = input_count.div_ceil(width);
        let params = ColumnReductionParams {
            input_count: u32::try_from(input_count)
                .map_err(|_| MetalError::InputTooLong(input_count))?,
            output_count: u32::try_from(output_count)
                .map_err(|_| MetalError::InputTooLong(output_count))?,
            columns,
            reserved: 0,
        };
        encoder.set_compute_pipeline_state(pipeline);
        let (input, output) = if input_a {
            (partial_a, partial_b)
        } else {
            (partial_b, partial_a)
        };
        input.bind_reduction(encoder, 0);
        output.bind_reduction(encoder, 1);
        set_inline_bytes(encoder, 2, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: output_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: width as u64,
                height: 1,
                depth: 1,
            },
        );
        input_count = output_count;
        input_a = !input_a;
    }
    Ok(input_a)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineLimits {
    pub thread_execution_width: usize,
    pub max_total_threads_per_threadgroup: usize,
    pub static_threadgroup_memory_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub max_buffer_length: u64,
    pub max_threadgroup_memory_length: u64,
    pub recommended_max_working_set_size: u64,
    pub current_allocated_size: u64,
    pub offset: u32,
}

#[derive(Clone)]
pub struct SolinasMetal {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    pub(super) library: Library,
    pub(super) offset: u32,
}

impl SolinasMetal {
    pub fn for_akita() -> Result<Self, MetalError> {
        Self::new(AKITA_OFFSET_FFFFA7F7)
    }

    pub(crate) fn for_akita_production() -> Result<Self, MetalError> {
        Self::new_with_source(AKITA_OFFSET_FFFFA7F7, library_source(AKITA_OFFSET_FFFFA7F7))
    }

    pub fn for_offset_275() -> Result<Self, MetalError> {
        Self::new(OFFSET_275)
    }

    pub(crate) fn device_registry_id(&self) -> u64 {
        self.device.registry_id()
    }

    pub fn new(offset: u32) -> Result<Self, MetalError> {
        Self::new_with_source(offset, library_source(offset))
    }

    fn new_with_source(offset: u32, source: String) -> Result<Self, MetalError> {
        if offset == 0 {
            return Err(MetalError::InvalidOffset);
        }
        let device = Device::system_default().ok_or(MetalError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = {
            let _span = tracing::info_span!(
                "MetalSolinas::library_compile",
                source_bytes = source.len(),
                offset
            )
            .entered();
            device
                .new_library_with_source(&source, &options)
                .map_err(MetalError::LibraryCompilation)?
        };
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            library,
            offset,
        })
    }

    pub fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name().to_owned(),
            max_buffer_length: self.device.max_buffer_length(),
            max_threadgroup_memory_length: self.device.max_threadgroup_memory_length(),
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            current_allocated_size: self.device.current_allocated_size(),
            offset: self.offset,
        }
    }

    pub(crate) fn validate_additional_working_set(
        &self,
        additional: u64,
    ) -> Result<(), MetalError> {
        super::validate_working_set(
            self.device.current_allocated_size(),
            additional,
            self.device.recommended_max_working_set_size(),
        )
    }

    pub(super) fn compile_named_pipeline(
        &self,
        name: &'static str,
    ) -> Result<ComputePipelineState, MetalError> {
        let _span = tracing::info_span!(
            "MetalSolinas::pipeline_compile",
            pipeline = name,
            specialized = false
        )
        .entered();
        let function = self
            .library
            .get_function(name, None)
            .map_err(|message| MetalError::FunctionLookup { name, message })?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation { name, message })
    }

    pub(super) fn validate_inputs(
        &self,
        side: &'static str,
        values: &[Fp128],
    ) -> Result<(), MetalError> {
        #[cfg(feature = "parallel")]
        let invalid = values
            .par_iter()
            .enumerate()
            .find_first(|(_, value)| !value.is_canonical(self.offset));
        #[cfg(not(feature = "parallel"))]
        let invalid = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_canonical(self.offset));
        if let Some((index, _)) = invalid {
            return Err(MetalError::NonCanonicalInput {
                side,
                index,
                offset: self.offset,
            });
        }
        Ok(())
    }

    pub(super) fn limits(pipeline: &ComputePipelineState) -> PipelineLimits {
        PipelineLimits {
            thread_execution_width: pipeline.thread_execution_width() as usize,
            max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup()
                as usize,
            static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
        }
    }

    pub(super) fn resolve_threadgroup_width(
        requested: Option<usize>,
        limits: PipelineLimits,
    ) -> Result<usize, MetalError> {
        let execution_width = limits.thread_execution_width;
        let maximum = limits.max_total_threads_per_threadgroup;
        let default = (execution_width * 8).min(maximum);
        let width = requested.unwrap_or(default);
        if width == 0 || width > maximum || !width.is_multiple_of(execution_width) {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: width,
                execution_width,
                maximum,
            });
        }
        Ok(width)
    }

    pub(super) fn validate_buffer_length(&self, requested: u64) -> Result<(), MetalError> {
        let maximum = self.device.max_buffer_length();
        if requested > maximum {
            return Err(MetalError::BufferTooLong { requested, maximum });
        }
        Ok(())
    }
}

pub(crate) fn validate_working_set(
    current: u64,
    additional: u64,
    maximum: u64,
) -> Result<(), MetalError> {
    if current
        .checked_add(additional)
        .is_none_or(|total| total > maximum)
    {
        return Err(MetalError::WorkingSetTooLarge {
            current,
            additional,
            maximum,
        });
    }
    Ok(())
}

pub(super) fn buffer_from_slice<T>(device: &Device, values: &[T]) -> Buffer {
    debug_assert!(!values.is_empty());
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub(super) fn command_buffer_timestamp(
    command_buffer: &metal::CommandBufferRef,
    name: &'static str,
) -> Result<f64, MetalError> {
    // SAFETY: both selectors are required, argument-free MTLCommandBuffer
    // properties returning CFTimeInterval, which is an f64.
    unsafe { command_buffer.send_message::<(), f64>(Sel::register(name), ()) }.map_err(|error| {
        MetalError::GpuTimestampLookup {
            name,
            message: error.to_string(),
        }
    })
}

pub(super) fn completed_command_gpu_time(
    command_buffer: &metal::CommandBufferRef,
) -> Result<Duration, MetalError> {
    validate_completed_command(command_buffer)?;
    let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
    let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return Err(MetalError::InvalidGpuTimestamps { start, end });
    }
    Ok(Duration::from_secs_f64(end - start))
}

pub(super) fn validate_completed_command(
    command_buffer: &metal::CommandBufferRef,
) -> Result<(), MetalError> {
    let status = command_buffer.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(MetalError::CommandFailed(status));
    }
    Ok(())
}
