use std::time::{Duration, Instant};

use metal::{CompileOptions, Device};

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
    let device = Device::system_default().ok_or(MetalError::DeviceUnavailable)?;
    let source = hamming_weight_claim_reduction_probe_source(AKITA_OFFSET_FFFFA7F7);
    let options = CompileOptions::new();
    let compile_started = Instant::now();
    let library = device
        .new_library_with_source(&source, &options)
        .map_err(MetalError::LibraryCompilation)?;
    let library_compile_wall = compile_started.elapsed();

    let histogram = pipeline_limits(&device, &library, HISTOGRAM_PIPELINE)?;
    let finalize = pipeline_limits(&device, &library, FINALIZE_PIPELINE)?;
    let device_info = DeviceInfo {
        name: device.name().to_owned(),
        max_buffer_length: device.max_buffer_length(),
        max_threadgroup_memory_length: device.max_threadgroup_memory_length(),
        recommended_max_working_set_size: device.recommended_max_working_set_size(),
        current_allocated_size: device.current_allocated_size(),
        offset: AKITA_OFFSET_FFFFA7F7,
    };

    Ok(HammingWeightCompileReport {
        device: device_info,
        histogram,
        finalize,
        dynamic_threadgroup_memory_bytes: HAMMING_WEIGHT_THREADGROUP_BYTES as u64,
        source_bytes: source.len(),
        library_compile_wall,
    })
}

fn pipeline_limits(
    device: &Device,
    library: &metal::LibraryRef,
    name: &'static str,
) -> Result<PipelineLimits, MetalError> {
    let function = library
        .get_function(name, None)
        .map_err(|message| MetalError::FunctionLookup { name, message })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::PipelineCompilation { name, message })?;
    Ok(PipelineLimits {
        thread_execution_width: pipeline.thread_execution_width() as usize,
        max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup() as usize,
        static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
    })
}
