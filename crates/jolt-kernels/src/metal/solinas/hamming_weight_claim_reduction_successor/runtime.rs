use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::super::{
    command_buffer_timestamp, Fp128, HammingHotRows, MetalError, PipelineLimits, SolinasMetal,
    AKITA_OFFSET_FFFFA7F7,
};
use super::{
    HammingWeightRetainedConfig, HammingWeightRetainedError, HammingWeightRetainedGeometry,
    HammingWeightRetainedParams, HAMMING_RETAINED_FINALIZE_BUFFER_OUTPUT,
    HAMMING_RETAINED_FINALIZE_BUFFER_PARAMS, HAMMING_RETAINED_FINALIZE_BUFFER_PARTIALS,
    HAMMING_RETAINED_FINALIZE_PIPELINE, HAMMING_RETAINED_FINALIZE_THREADGROUP_BYTES,
    HAMMING_RETAINED_SIMD_WIDTH, HAMMING_RETAINED_TILE_BUFFER_E_IN,
    HAMMING_RETAINED_TILE_BUFFER_E_OUT, HAMMING_RETAINED_TILE_BUFFER_HOT,
    HAMMING_RETAINED_TILE_BUFFER_PARAMS, HAMMING_RETAINED_TILE_BUFFER_PARTIALS,
    HAMMING_RETAINED_TILE_PIPELINES, HAMMING_RETAINED_TILE_WIDTHS,
};

#[derive(Debug, thiserror::Error)]
pub enum HammingWeightRetainedRuntimeError {
    #[error("invalid retained-Hamming plan: {0}")]
    Plan(#[from] HammingWeightRetainedError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("retained-Hamming reference point has length {got}, expected {expected}")]
    ReferencePointLength { expected: usize, got: usize },
    #[error("retained-Hamming lease has {got} rows, expected {expected}")]
    LeaseLength { expected: usize, got: usize },
    #[error("retained-Hamming lease belongs to device {got}, expected {expected}")]
    LeaseDevice { expected: u64, got: u64 },
    #[error("retained-Hamming lease has {got} bytes, expected {expected}")]
    LeaseBytes { expected: u64, got: u64 },
    #[error("retained-Hamming pipeline `{pipeline}` requires SIMD width {expected}, got {got}")]
    ExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "retained-Hamming pipeline `{pipeline}` needs {requested} threads, maximum is {maximum}"
    )]
    ThreadLimit {
        pipeline: &'static str,
        requested: usize,
        maximum: usize,
    },
    #[error(
        "retained-Hamming pipeline `{pipeline}` needs {requested} threadgroup bytes, device maximum is {maximum}"
    )]
    ThreadgroupMemory {
        pipeline: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("invalid retained-Hamming state: {0}")]
    InvalidState(&'static str),
}

struct Buffers {
    hot_rows: HammingHotRows,
    e_in: Buffer,
    e_out: Buffer,
    partials: Buffer,
    output: Buffer,
}

pub struct HammingWeightRetainedInvocation {
    context: SolinasMetal,
    tile_pipelines: Vec<ComputePipelineState>,
    finalize_pipeline: ComputePipelineState,
    tile_limits: Vec<PipelineLimits>,
    finalize_limits: PipelineLimits,
    buffers: Buffers,
    geometry: HammingWeightRetainedGeometry,
    params: Vec<HammingWeightRetainedParams>,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_hamming_weight_retained(
        &self,
        hot_rows: HammingHotRows,
        reference_cycle: &[AkitaField],
        config: HammingWeightRetainedConfig,
    ) -> Result<HammingWeightRetainedInvocation, HammingWeightRetainedRuntimeError> {
        if self.offset != AKITA_OFFSET_FFFFA7F7 {
            return Err(MetalError::UnexpectedSolinasOffset {
                expected: AKITA_OFFSET_FFFFA7F7,
                got: self.offset,
            }
            .into());
        }
        let geometry = HammingWeightRetainedGeometry::new(hot_rows.len(), config)?;
        let expected_point = hot_rows.len().ilog2() as usize;
        if reference_cycle.len() != expected_point {
            return Err(HammingWeightRetainedRuntimeError::ReferencePointLength {
                expected: expected_point,
                got: reference_cycle.len(),
            });
        }
        if hot_rows.len() != geometry.rows() {
            return Err(HammingWeightRetainedRuntimeError::LeaseLength {
                expected: geometry.rows(),
                got: hot_rows.len(),
            });
        }
        let expected_device = self.device_registry_id();
        if hot_rows.device_registry_id() != expected_device {
            return Err(HammingWeightRetainedRuntimeError::LeaseDevice {
                expected: expected_device,
                got: hot_rows.device_registry_id(),
            });
        }
        let lengths = geometry.buffer_lengths()?;
        if hot_rows.buffer().length() != lengths.hot_bytes {
            return Err(HammingWeightRetainedRuntimeError::LeaseBytes {
                expected: lengths.hot_bytes,
                got: hot_rows.buffer().length(),
            });
        }

        let split = reference_cycle.len() - config.inner_log2;
        let (out_point, in_point) = reference_cycle.split_at(split);
        let e_in = EqPolynomial::evals(in_point, None);
        let e_out = EqPolynomial::evals(out_point, None);

        let tile_pipelines = HAMMING_RETAINED_TILE_PIPELINES
            .iter()
            .map(|pipeline| self.compile_named_pipeline(pipeline))
            .collect::<Result<Vec<_>, _>>()?;
        let finalize_pipeline = self.compile_named_pipeline(HAMMING_RETAINED_FINALIZE_PIPELINE)?;
        let tile_limits = tile_pipelines.iter().map(Self::limits).collect::<Vec<_>>();
        let finalize_limits = Self::limits(&finalize_pipeline);
        for ((pipeline, limits), width) in HAMMING_RETAINED_TILE_PIPELINES
            .iter()
            .copied()
            .zip(tile_limits.iter().copied())
            .zip(HAMMING_RETAINED_TILE_WIDTHS)
        {
            validate_pipeline(
                pipeline,
                limits,
                config.accumulator_threads_per_threadgroup,
                tile_threadgroup_bytes(width)?,
                self.device.max_threadgroup_memory_length(),
            )?;
        }
        validate_pipeline(
            HAMMING_RETAINED_FINALIZE_PIPELINE,
            finalize_limits,
            config.finalize_threads_per_threadgroup,
            HAMMING_RETAINED_FINALIZE_THREADGROUP_BYTES as u64,
            self.device.max_threadgroup_memory_length(),
        )?;

        for bytes in [
            field_bytes(lengths.e_in_fields)?,
            field_bytes(lengths.e_out_fields)?,
            field_bytes(lengths.partial_fields)?,
            field_bytes(lengths.output_fields)?,
        ] {
            self.validate_buffer_length(bytes)?;
        }
        self.validate_additional_working_set(lengths.owned_bytes()?)?;

        let e_in_buffer = self.device.new_buffer(
            field_bytes(lengths.e_in_fields)?,
            MTLResourceOptions::StorageModeShared,
        );
        let e_out_buffer = self.device.new_buffer(
            field_bytes(lengths.e_out_fields)?,
            MTLResourceOptions::StorageModeShared,
        );
        super::super::booleanity::write_fields(&e_in_buffer, e_in.len(), &e_in)?;
        super::super::booleanity::write_fields(&e_out_buffer, e_out.len(), &e_out)?;
        let params = (0..HAMMING_RETAINED_TILE_WIDTHS.len())
            .map(|tile| geometry.params(tile))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(HammingWeightRetainedInvocation {
            context: self.clone(),
            tile_pipelines,
            finalize_pipeline,
            tile_limits,
            finalize_limits,
            buffers: Buffers {
                hot_rows,
                e_in: e_in_buffer,
                e_out: e_out_buffer,
                partials: self.device.new_buffer(
                    field_bytes(lengths.partial_fields)?,
                    MTLResourceOptions::StorageModePrivate,
                ),
                output: self.device.new_buffer(
                    field_bytes(lengths.output_fields)?,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            geometry,
            params,
            completed: Cell::new(false),
        })
    }
}

impl HammingWeightRetainedInvocation {
    pub fn tile_pipeline_limits(&self) -> &[PipelineLimits] {
        &self.tile_limits
    }

    pub const fn finalize_pipeline_limits(&self) -> PipelineLimits {
        self.finalize_limits
    }

    pub const fn output_elements(&self) -> usize {
        super::HAMMING_RETAINED_SELECTORS * super::HAMMING_RETAINED_BINS
    }

    pub fn hot_rows_storage_id(&self) -> usize {
        self.buffers.hot_rows.allocation_identity()
    }

    pub fn source_rows_storage_id(&self) -> usize {
        self.buffers.hot_rows.source_rows_storage_id()
    }

    pub fn execute_timed(&self) -> Result<Duration, HammingWeightRetainedRuntimeError> {
        self.completed.set(false);
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            for (tile_index, ((pipeline, params), width)) in self
                .tile_pipelines
                .iter()
                .zip(&self.params)
                .zip(HAMMING_RETAINED_TILE_WIDTHS)
                .enumerate()
            {
                debug_assert_eq!(tile_index, params.selector_offset as usize / 6);
                let tile = command_buffer.new_compute_command_encoder();
                tile.set_compute_pipeline_state(pipeline);
                tile.set_buffer(
                    HAMMING_RETAINED_TILE_BUFFER_HOT,
                    Some(self.buffers.hot_rows.buffer()),
                    0,
                );
                tile.set_buffer(
                    HAMMING_RETAINED_TILE_BUFFER_E_IN,
                    Some(&self.buffers.e_in),
                    0,
                );
                tile.set_buffer(
                    HAMMING_RETAINED_TILE_BUFFER_E_OUT,
                    Some(&self.buffers.e_out),
                    0,
                );
                tile.set_buffer(
                    HAMMING_RETAINED_TILE_BUFFER_PARTIALS,
                    Some(&self.buffers.partials),
                    0,
                );
                set_inline_bytes(tile, HAMMING_RETAINED_TILE_BUFFER_PARAMS, params);
                tile.set_threadgroup_memory_length(0, tile_threadgroup_bytes(width)?);
                tile.dispatch_thread_groups(
                    MTLSize {
                        width: self.geometry.e_out_length() as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.geometry.accumulator_threads_per_threadgroup() as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                tile.end_encoding();

                let finalize = command_buffer.new_compute_command_encoder();
                finalize.set_compute_pipeline_state(&self.finalize_pipeline);
                finalize.set_buffer(
                    HAMMING_RETAINED_FINALIZE_BUFFER_PARTIALS,
                    Some(&self.buffers.partials),
                    0,
                );
                finalize.set_buffer(
                    HAMMING_RETAINED_FINALIZE_BUFFER_OUTPUT,
                    Some(&self.buffers.output),
                    0,
                );
                set_inline_bytes(finalize, HAMMING_RETAINED_FINALIZE_BUFFER_PARAMS, params);
                finalize.set_threadgroup_memory_length(
                    0,
                    HAMMING_RETAINED_FINALIZE_THREADGROUP_BYTES as u64,
                );
                finalize.dispatch_thread_groups(
                    MTLSize {
                        width: width as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.geometry.finalize_threads_per_threadgroup() as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                finalize.end_encoding();
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status).into());
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
    }

    pub fn read_masses(&self) -> Result<Vec<AkitaField>, HammingWeightRetainedRuntimeError> {
        if !self.completed.get() {
            return Err(HammingWeightRetainedRuntimeError::InvalidState(
                "readback before command completion",
            ));
        }
        // SAFETY: the shared allocation contains output_elements fields and
        // the command buffer completed the only writer before this read.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.output_elements(),
            )
        };
        self.context
            .validate_inputs("retained Hamming output", values)?;
        Ok(values
            .iter()
            .map(|value| (*value).into_jolt_field())
            .collect())
    }
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
    threads: usize,
    dynamic_threadgroup_bytes: u64,
    maximum_threadgroup_bytes: u64,
) -> Result<(), HammingWeightRetainedRuntimeError> {
    if limits.thread_execution_width != HAMMING_RETAINED_SIMD_WIDTH {
        return Err(HammingWeightRetainedRuntimeError::ExecutionWidth {
            pipeline,
            expected: HAMMING_RETAINED_SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    if threads > limits.max_total_threads_per_threadgroup {
        return Err(HammingWeightRetainedRuntimeError::ThreadLimit {
            pipeline,
            requested: threads,
            maximum: limits.max_total_threads_per_threadgroup,
        });
    }
    let requested = limits
        .static_threadgroup_memory_length
        .checked_add(dynamic_threadgroup_bytes)
        .ok_or(HammingWeightRetainedError::ArithmeticOverflow)?;
    if requested > maximum_threadgroup_bytes {
        return Err(HammingWeightRetainedRuntimeError::ThreadgroupMemory {
            pipeline,
            requested,
            maximum: maximum_threadgroup_bytes,
        });
    }
    Ok(())
}

fn tile_threadgroup_bytes(width: usize) -> Result<u64, HammingWeightRetainedError> {
    width
        .checked_mul(super::HAMMING_RETAINED_BINS)
        .and_then(|value| value.checked_mul(super::HAMMING_RETAINED_DEFERRED_WORDS))
        .and_then(|value| value.checked_mul(size_of::<u32>()))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(HammingWeightRetainedError::ArithmeticOverflow)
}

fn field_bytes(fields: u64) -> Result<u64, HammingWeightRetainedError> {
    fields
        .checked_mul(size_of::<Fp128>() as u64)
        .ok_or(HammingWeightRetainedError::ArithmeticOverflow)
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}
