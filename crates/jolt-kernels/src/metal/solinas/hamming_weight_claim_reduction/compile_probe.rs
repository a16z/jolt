use std::time::{Duration, Instant};

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, Library};

use super::{
    FINALIZE_PIPELINE, HAMMING_WEIGHT_BINS, HAMMING_WEIGHT_SIMD_WIDTH,
    HAMMING_WEIGHT_THREADGROUP_BYTES, HAMMING_WEIGHT_THREADS, HISTOGRAM_PIPELINE,
};
use crate::metal::solinas::{
    source::hamming_weight_claim_reduction_probe_source, DeviceInfo, MetalError, PipelineLimits,
    AKITA_OFFSET_FFFFA7F7,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HammingWeightCompileReport {
    pub device: DeviceInfo,
    pub histogram: PipelineLimits,
    pub finalize: PipelineLimits,
    pub dynamic_threadgroup_memory_bytes: u64,
    pub source_bytes: usize,
    pub library_compile_wall: Duration,
}

impl HammingWeightCompileReport {
    pub fn histogram_admitted(&self) -> bool {
        let memory_admitted = self
            .histogram
            .static_threadgroup_memory_length
            .checked_add(self.dynamic_threadgroup_memory_bytes)
            .is_some_and(|requested| requested <= self.device.max_threadgroup_memory_length);
        self.histogram.thread_execution_width == HAMMING_WEIGHT_SIMD_WIDTH
            && self.histogram.max_total_threads_per_threadgroup >= HAMMING_WEIGHT_THREADS
            && memory_admitted
    }

    pub fn finalize_admitted(&self) -> bool {
        self.finalize.thread_execution_width == HAMMING_WEIGHT_SIMD_WIDTH
            && self.finalize.max_total_threads_per_threadgroup >= HAMMING_WEIGHT_BINS
            && self.finalize.static_threadgroup_memory_length
                <= self.device.max_threadgroup_memory_length
    }

    pub fn admitted(&self) -> bool {
        self.histogram_admitted() && self.finalize_admitted()
    }
}

pub fn compile_hamming_weight_claim_reduction_probe(
) -> Result<HammingWeightCompileReport, MetalError> {
    Ok(HammingWeightProbeContext::new()?
        .compile_pipelines()?
        .report)
}

pub(super) struct HammingWeightProbeContext {
    pub device: Device,
    pub queue: CommandQueue,
    library: Library,
    source_bytes: usize,
    library_compile_wall: Duration,
}

pub(super) struct HammingWeightProbePipelines {
    pub histogram: ComputePipelineState,
    pub finalize: ComputePipelineState,
    pub report: HammingWeightCompileReport,
}

impl HammingWeightProbeContext {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::DeviceUnavailable)?;
        let source = hamming_weight_claim_reduction_probe_source(AKITA_OFFSET_FFFFA7F7);
        let options = CompileOptions::new();
        let compile_started = Instant::now();
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(MetalError::LibraryCompilation)?;
        let library_compile_wall = compile_started.elapsed();
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            library,
            source_bytes: source.len(),
            library_compile_wall,
        })
    }

    pub fn compile_pipelines(&self) -> Result<HammingWeightProbePipelines, MetalError> {
        let histogram = compile_pipeline(&self.device, &self.library, HISTOGRAM_PIPELINE)?;
        let finalize = compile_pipeline(&self.device, &self.library, FINALIZE_PIPELINE)?;
        let report = HammingWeightCompileReport {
            device: self.device_info(),
            histogram: pipeline_limits(&histogram),
            finalize: pipeline_limits(&finalize),
            dynamic_threadgroup_memory_bytes: HAMMING_WEIGHT_THREADGROUP_BYTES as u64,
            source_bytes: self.source_bytes,
            library_compile_wall: self.library_compile_wall,
        };

        Ok(HammingWeightProbePipelines {
            histogram,
            finalize,
            report,
        })
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name().to_owned(),
            max_buffer_length: self.device.max_buffer_length(),
            max_threadgroup_memory_length: self.device.max_threadgroup_memory_length(),
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            current_allocated_size: self.device.current_allocated_size(),
            offset: AKITA_OFFSET_FFFFA7F7,
        }
    }
}

fn compile_pipeline(
    device: &Device,
    library: &metal::LibraryRef,
    name: &'static str,
) -> Result<ComputePipelineState, MetalError> {
    let function = library
        .get_function(name, None)
        .map_err(|message| MetalError::FunctionLookup { name, message })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::PipelineCompilation { name, message })
}

fn pipeline_limits(pipeline: &ComputePipelineState) -> PipelineLimits {
    PipelineLimits {
        thread_execution_width: pipeline.thread_execution_width() as usize,
        max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup() as usize,
        static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
    }
}
