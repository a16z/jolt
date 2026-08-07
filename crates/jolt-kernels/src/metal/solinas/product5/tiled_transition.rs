//! Tiled five-factor bind-and-message transition.

use core::mem::size_of;
use std::{cell::Cell, slice, time::Duration, time::Instant};

use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

const FACTORS: usize = 5;
const MAIN_THREADS: usize = 160;
const SIMD_WIDTH: usize = 32;
const REDUCE_PIPELINE: &str = "solinas_product5_reduce";
const WEIGHT_TILES_PIPELINE: &str = "solinas_product5_tiled_weight_tiles";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseTransitionTile {
    Pairs32,
    Pairs64,
    Pairs128,
}

impl DenseTransitionTile {
    pub const fn pairs(self) -> usize {
        match self {
            Self::Pairs32 => 32,
            Self::Pairs64 => 64,
            Self::Pairs128 => 128,
        }
    }

    const fn pipeline(self) -> &'static str {
        match self {
            Self::Pairs32 => "solinas_product5_tiled_factor_sample_32",
            Self::Pairs64 => "solinas_product5_tiled_factor_sample_64",
            Self::Pairs128 => "solinas_product5_tiled_factor_sample_128",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseTransitionParams {
    pub source_elements: u32,
    pub destination_elements: u32,
    pub e_in_length: u32,
    pub e_out_length: u32,
    pub tile_pairs: u32,
    pub tiles_per_out: u32,
    pub total_tiles: u32,
    pub reserved: u32,
}

const _: [(); 32] = [(); size_of::<DenseTransitionParams>()];

#[repr(C)]
#[derive(Clone, Copy)]
struct ReductionParams {
    input_count: u32,
    output_count: u32,
    reserved: [u32; 2],
}

const _: [(); 16] = [(); size_of::<ReductionParams>()];

#[derive(Debug, thiserror::Error)]
pub enum DenseTransitionError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(
        "tiled Product5 transition needs a power-of-two source length of at least four, got {0}"
    )]
    InvalidSourceElements(usize),
    #[error("tiled Product5 transition source has {got} fields, expected {expected}")]
    SourceLength { expected: usize, got: usize },
    #[error("tiled Product5 transition weights cover {covered} pairs, expected {expected}")]
    WeightShape { expected: usize, covered: usize },
    #[error(
        "tiled Product5 transition inner length {inner} is not divisible by tile width {tile}"
    )]
    TileShape { inner: usize, tile: usize },
    #[error("tiled Product5 pipeline `{pipeline}` requires SIMD width 32, got {got}")]
    ExecutionWidth { pipeline: &'static str, got: usize },
    #[error(
        "tiled Product5 pipeline `{pipeline}` needs {requested} threads, maximum is {maximum}"
    )]
    ThreadLimit {
        pipeline: &'static str,
        requested: usize,
        maximum: usize,
    },
    #[error(
        "tiled Product5 transition needs {requested} threadgroup bytes, device maximum is {maximum}"
    )]
    ThreadgroupMemory { requested: u64, maximum: u64 },
    #[error("tiled Product5 transition size arithmetic overflowed")]
    SizeOverflow,
}

struct ReductionStep {
    input_outer: bool,
    input_count: usize,
    output_count: usize,
    params: Buffer,
}

struct DenseTransitionBuffers {
    source: Buffer,
    destination: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    challenge: Buffer,
    params: Buffer,
    tile_partials: Buffer,
    outer_partials: Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseTransitionObservation {
    pub tile: DenseTransitionTile,
    pub main_threadgroups: usize,
    pub dynamic_threadgroup_bytes: usize,
    pub dispatches: usize,
    pub useful_products: u64,
    pub logical_bytes: u64,
    pub allocated_bytes: u64,
    pub gpu_active: Duration,
    pub resident_wall: Duration,
}

pub struct DenseTransitionInvocation {
    context: SolinasMetal,
    main_pipeline: ComputePipelineState,
    weight_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    main_limits: PipelineLimits,
    weight_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    buffers: DenseTransitionBuffers,
    reduction_steps: Vec<ReductionStep>,
    params: DenseTransitionParams,
    tile: DenseTransitionTile,
    source_elements: usize,
    destination_elements: usize,
    e_out_length: usize,
    final_in_outer: bool,
    allocated_bytes: u64,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_product5_tiled_transition(
        &self,
        source: &[Fp128],
        source_elements: usize,
        challenge: Fp128,
        e_in: &[Fp128],
        e_out: &[Fp128],
        tile: DenseTransitionTile,
    ) -> Result<DenseTransitionInvocation, DenseTransitionError> {
        if source_elements < 4 || !source_elements.is_power_of_two() {
            return Err(DenseTransitionError::InvalidSourceElements(source_elements));
        }
        let expected_source = FACTORS
            .checked_mul(source_elements)
            .ok_or(DenseTransitionError::SizeOverflow)?;
        if source.len() != expected_source {
            return Err(DenseTransitionError::SourceLength {
                expected: expected_source,
                got: source.len(),
            });
        }
        let message_pairs = source_elements / 4;
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(DenseTransitionError::SizeOverflow)?;
        if e_in.is_empty() || e_out.is_empty() || covered != message_pairs {
            return Err(DenseTransitionError::WeightShape {
                expected: message_pairs,
                covered,
            });
        }
        let tile_pairs = tile.pairs();
        if !e_in.len().is_multiple_of(tile_pairs) {
            return Err(DenseTransitionError::TileShape {
                inner: e_in.len(),
                tile: tile_pairs,
            });
        }
        let destination_elements = source_elements / 2;
        let tiles_per_out = e_in.len() / tile_pairs;
        let total_tiles = e_out
            .len()
            .checked_mul(tiles_per_out)
            .ok_or(DenseTransitionError::SizeOverflow)?;
        let params = DenseTransitionParams {
            source_elements: abi_count(source_elements)?,
            destination_elements: abi_count(destination_elements)?,
            e_in_length: abi_count(e_in.len())?,
            e_out_length: abi_count(e_out.len())?,
            tile_pairs: abi_count(tile_pairs)?,
            tiles_per_out: abi_count(tiles_per_out)?,
            total_tiles: abi_count(total_tiles)?,
            reserved: 0,
        };
        self.validate_inputs("tiled Product5 source", source)?;
        self.validate_inputs("tiled Product5 e_in", e_in)?;
        self.validate_inputs("tiled Product5 e_out", e_out)?;
        self.validate_inputs("tiled Product5 challenge", slice::from_ref(&challenge))?;

        let main_pipeline = self.compile_named_pipeline(tile.pipeline())?;
        let weight_pipeline = self.compile_named_pipeline(WEIGHT_TILES_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let main_limits = Self::limits(&main_pipeline);
        let weight_limits = Self::limits(&weight_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        for (pipeline, limits) in [
            (tile.pipeline(), main_limits),
            (WEIGHT_TILES_PIPELINE, weight_limits),
            (REDUCE_PIPELINE, reduce_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(DenseTransitionError::ExecutionWidth {
                    pipeline,
                    got: limits.thread_execution_width,
                });
            }
        }
        for (pipeline, requested, maximum) in [
            (
                tile.pipeline(),
                MAIN_THREADS,
                main_limits.max_total_threads_per_threadgroup,
            ),
            (
                WEIGHT_TILES_PIPELINE,
                SIMD_WIDTH,
                weight_limits.max_total_threads_per_threadgroup,
            ),
            (
                REDUCE_PIPELINE,
                SIMD_WIDTH,
                reduce_limits.max_total_threads_per_threadgroup,
            ),
        ] {
            if requested > maximum {
                return Err(DenseTransitionError::ThreadLimit {
                    pipeline,
                    requested,
                    maximum,
                });
            }
        }
        let dynamic_threadgroup_bytes = FACTORS
            .checked_mul(2)
            .and_then(|value| value.checked_mul(tile_pairs))
            .and_then(|value| value.checked_mul(size_of::<Fp128>()))
            .ok_or(DenseTransitionError::SizeOverflow)?;
        let requested_threadgroup = u64::try_from(dynamic_threadgroup_bytes)
            .map_err(|_| DenseTransitionError::SizeOverflow)?
            .checked_add(main_limits.static_threadgroup_memory_length)
            .ok_or(DenseTransitionError::SizeOverflow)?;
        let maximum_threadgroup = self.device.max_threadgroup_memory_length();
        if requested_threadgroup > maximum_threadgroup {
            return Err(DenseTransitionError::ThreadgroupMemory {
                requested: requested_threadgroup,
                maximum: maximum_threadgroup,
            });
        }

        let source_bytes = field_bytes(expected_source)?;
        let destination_bytes = field_bytes(FACTORS * destination_elements)?;
        let e_in_bytes = field_bytes(e_in.len())?;
        let e_out_bytes = field_bytes(e_out.len())?;
        let tile_bytes = field_bytes(FACTORS * total_tiles)?;
        let outer_bytes = field_bytes(FACTORS * e_out.len())?;
        for bytes in [
            source_bytes,
            destination_bytes,
            e_in_bytes,
            e_out_bytes,
            tile_bytes,
            outer_bytes,
        ] {
            self.validate_buffer_length(bytes)?;
        }

        let mut reduction_steps = Vec::new();
        let mut input_count = e_out.len();
        let mut input_outer = true;
        while input_count > 1 {
            let output_count = input_count.div_ceil(SIMD_WIDTH);
            let reduction = ReductionParams {
                input_count: abi_count(input_count)?,
                output_count: abi_count(output_count)?,
                reserved: [0; 2],
            };
            reduction_steps.push(ReductionStep {
                input_outer,
                input_count,
                output_count,
                params: buffer_from_slice(&self.device, slice::from_ref(&reduction)),
            });
            input_count = output_count;
            input_outer = !input_outer;
        }

        let tiny_bytes = size_of::<Fp128>()
            .checked_add(size_of::<DenseTransitionParams>())
            .and_then(|value| {
                value.checked_add(reduction_steps.len() * size_of::<ReductionParams>())
            })
            .ok_or(DenseTransitionError::SizeOverflow)?;
        let allocated_bytes = [
            source_bytes,
            destination_bytes,
            e_in_bytes,
            e_out_bytes,
            tile_bytes,
            outer_bytes,
            u64::try_from(tiny_bytes).map_err(|_| DenseTransitionError::SizeOverflow)?,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or(DenseTransitionError::SizeOverflow)?;
        self.validate_additional_working_set(allocated_bytes)?;

        Ok(DenseTransitionInvocation {
            context: self.clone(),
            main_pipeline,
            weight_pipeline,
            reduce_pipeline,
            main_limits,
            weight_limits,
            reduce_limits,
            buffers: DenseTransitionBuffers {
                source: buffer_from_slice(&self.device, source),
                destination: self
                    .device
                    .new_buffer(destination_bytes, MTLResourceOptions::StorageModeShared),
                e_in: buffer_from_slice(&self.device, e_in),
                e_out: buffer_from_slice(&self.device, e_out),
                challenge: buffer_from_slice(&self.device, slice::from_ref(&challenge)),
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
                tile_partials: self
                    .device
                    .new_buffer(tile_bytes, MTLResourceOptions::StorageModeShared),
                outer_partials: self
                    .device
                    .new_buffer(outer_bytes, MTLResourceOptions::StorageModeShared),
            },
            reduction_steps,
            params,
            tile,
            source_elements,
            destination_elements,
            e_out_length: e_out.len(),
            final_in_outer: input_outer,
            allocated_bytes,
            completed: Cell::new(false),
        })
    }
}

impl DenseTransitionInvocation {
    pub const fn params(&self) -> DenseTransitionParams {
        self.params
    }

    pub const fn main_pipeline_limits(&self) -> PipelineLimits {
        self.main_limits
    }

    pub const fn weight_pipeline_limits(&self) -> PipelineLimits {
        self.weight_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduce_limits
    }

    pub const fn dynamic_threadgroup_bytes(&self) -> usize {
        FACTORS * 2 * self.tile.pairs() * size_of::<Fp128>()
    }

    pub const fn useful_products(&self) -> u64 {
        8 * self.source_elements as u64 + FACTORS as u64 * self.e_out_length as u64
    }

    pub fn logical_bytes(&self) -> Result<u64, DenseTransitionError> {
        let mut bytes = 120u64
            .checked_mul(self.source_elements as u64)
            .and_then(|bytes| {
                bytes.checked_add(2 * FACTORS as u64 * self.params.total_tiles as u64 * 16)
            })
            .and_then(|bytes| bytes.checked_add(FACTORS as u64 * self.e_out_length as u64 * 16))
            .ok_or(DenseTransitionError::SizeOverflow)?;
        for step in &self.reduction_steps {
            let traffic_fields = FACTORS
                .checked_mul(step.input_count)
                .and_then(|value| value.checked_add(FACTORS * step.output_count))
                .ok_or(DenseTransitionError::SizeOverflow)?;
            bytes = bytes
                .checked_add(
                    u64::try_from(traffic_fields)
                        .map_err(|_| DenseTransitionError::SizeOverflow)?
                        * 16,
                )
                .ok_or(DenseTransitionError::SizeOverflow)?;
        }
        Ok(bytes)
    }

    pub fn execute_timed(&self) -> Result<DenseTransitionObservation, DenseTransitionError> {
        let wall_started = Instant::now();
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.main_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.source), 0);
            encoder.set_buffer(1, Some(&self.buffers.destination), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.tile_partials), 0);
            encoder.set_buffer(4, Some(&self.buffers.challenge), 0);
            encoder.set_buffer(5, Some(&self.buffers.params), 0);
            encoder.set_threadgroup_memory_length(0, self.dynamic_threadgroup_bytes() as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.params.total_tiles as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: MAIN_THREADS as u64,
                    height: 1,
                    depth: 1,
                },
            );

            encoder.set_compute_pipeline_state(&self.weight_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.tile_partials), 0);
            encoder.set_buffer(1, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(2, Some(&self.buffers.outer_partials), 0);
            encoder.set_buffer(3, Some(&self.buffers.params), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.e_out_length as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SIMD_WIDTH as u64,
                    height: 1,
                    depth: 1,
                },
            );

            for step in &self.reduction_steps {
                encoder.set_compute_pipeline_state(&self.reduce_pipeline);
                let (input, output) = if step.input_outer {
                    (&self.buffers.outer_partials, &self.buffers.tile_partials)
                } else {
                    (&self.buffers.tile_partials, &self.buffers.outer_partials)
                };
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_buffer(2, Some(&step.params), 0);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: step.output_count as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: SIMD_WIDTH as u64,
                        height: 1,
                        depth: 1,
                    },
                );
            }
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
            self.completed.set(true);
            Ok(DenseTransitionObservation {
                tile: self.tile,
                main_threadgroups: self.params.total_tiles as usize,
                dynamic_threadgroup_bytes: self.dynamic_threadgroup_bytes(),
                dispatches: 2 + self.reduction_steps.len(),
                useful_products: self.useful_products(),
                logical_bytes: self.logical_bytes()?,
                allocated_bytes: self.allocated_bytes,
                gpu_active: Duration::from_secs_f64(end - start),
                resident_wall: wall_started.elapsed(),
            })
        })
    }

    pub fn read_bound_tables(&self) -> Result<Vec<Fp128>, DenseTransitionError> {
        self.require_completed()?;
        let fields = FACTORS * self.destination_elements;
        // SAFETY: the destination buffer owns exactly `fields` initialized
        // values after the command buffer completes.
        let values = unsafe {
            slice::from_raw_parts(self.buffers.destination.contents().cast::<Fp128>(), fields)
        };
        self.context
            .validate_inputs("tiled Product5 destination", values)?;
        Ok(values.to_vec())
    }

    pub fn read_message(&self) -> Result<[Fp128; FACTORS], DenseTransitionError> {
        self.require_completed()?;
        let buffer = if self.final_in_outer {
            &self.buffers.outer_partials
        } else {
            &self.buffers.tile_partials
        };
        // SAFETY: the final reduction writes five contiguous fields.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), FACTORS) };
        self.context
            .validate_inputs("tiled Product5 message", values)?;
        Ok(std::array::from_fn(|index| values[index]))
    }

    fn require_completed(&self) -> Result<(), DenseTransitionError> {
        if self.completed.get() {
            Ok(())
        } else {
            Err(MetalError::NotExecuted.into())
        }
    }
}

fn abi_count(value: usize) -> Result<u32, DenseTransitionError> {
    u32::try_from(value).map_err(|_| DenseTransitionError::SizeOverflow)
}

fn field_bytes(fields: usize) -> Result<u64, DenseTransitionError> {
    fields
        .checked_mul(size_of::<Fp128>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(DenseTransitionError::SizeOverflow)
}
