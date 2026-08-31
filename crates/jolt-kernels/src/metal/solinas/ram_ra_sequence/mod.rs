use std::{ffi::c_void, mem::size_of, slice, sync::Arc};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, FunctionConstantValues, MTLDataType,
    MTLResourceOptions, MTLSize,
};

use super::{
    encode_column_reductions, set_inline_bytes, validate_completed_command, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};
use crate::optimized::ram_trace::RamAccessColumns;

const MAX_FACTORS: usize = 3;
const BINS: usize = 256;
const REDUCTION_COLUMNS: usize = 4;
const MESSAGE_SAMPLES: usize = 3;
const SIMD_WIDTH: usize = 32;
const MESSAGE_THREADS: usize = 128;
const MATERIALIZE_THREADS: usize = 64;
const BRANCH_THREADS: usize = 256;
const MATERIALIZE_WIDTH: usize = 16;

const LAZY_PIPELINE: &str = "solinas_ram_ra_lazy_message";
const DOUBLE_PIPELINE: &str = "solinas_ram_ra_double_branches";
const MATERIALIZE_PIPELINE: &str = "solinas_ram_ra_materialize_width_16";
const DENSE_PIPELINE: &str = "solinas_ram_ra_dense_transition";
const REDUCE_PIPELINE: &str = "solinas_instruction_ra_reduce";

#[repr(C)]
#[derive(Clone, Copy)]
struct MessageParams {
    e_in_length: u32,
    e_out_length: u32,
    factor_count: u32,
    _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BranchParams {
    branch_width: u32,
    factor_count: u32,
    _reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaterializeParams {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    factor_count: u32,
}

struct Pipelines {
    lazy: Vec<(usize, ComputePipelineState)>,
    double: ComputePipelineState,
    materialize: ComputePipelineState,
    dense: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct Buffers {
    addresses: Buffer,
    branches_a: Buffer,
    branches_b: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub(crate) struct RamRaSequence {
    context: SolinasMetal,
    _columns: Arc<RamAccessColumns>,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    factors: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    message_threads: usize,
    materialize_threads: usize,
    branch_threads: usize,
    branch_width: usize,
    branches_in_a: bool,
    dense: bool,
    dense_in_a: bool,
    dense_elements: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamRaSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            device_storage_bytes(&self.buffers),
        );
        visitor.visit_simple(
            allocative::Key::new("aliased_addresses"),
            self.rows.saturating_mul(size_of::<u32>()),
        );
        visitor.exit();
    }
}

impl SolinasMetal {
    fn compile_ram_ra_width_pipeline(
        &self,
        width: usize,
    ) -> Result<ComputePipelineState, MetalError> {
        let key = (LAZY_PIPELINE, Some(width as u32));
        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| MetalError::PipelineCachePoisoned)?;
        if let Some(pipeline) = cache.get(&key) {
            return Ok(pipeline.clone());
        }
        let width = u32::try_from(width).map_err(|_| MetalError::InputTooLong(width))?;
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            std::ptr::from_ref(&width).cast::<c_void>(),
            MTLDataType::UInt,
            20,
        );
        let function = self
            .library
            .get_function(LAZY_PIPELINE, Some(constants))
            .map_err(|message| MetalError::FunctionLookup {
                name: LAZY_PIPELINE,
                message,
            })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation {
                name: LAZY_PIPELINE,
                message,
            })?;
        let _ = cache.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    pub(crate) fn prepare_ram_ra_sequence(
        &self,
        columns: Arc<RamAccessColumns>,
        chunk_tables: &[AkitaField],
        factors: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
    ) -> Result<RamRaSequence, MetalError> {
        let rows = columns.addresses.len();
        if rows < 2 * MATERIALIZE_WIDTH || !rows.is_power_of_two() {
            return Err(MetalError::InvalidRamRaRows(rows));
        }
        if !(2..=MAX_FACTORS).contains(&factors) {
            return Err(MetalError::InvalidRamRaState(
                "the direct sequence supports two or three factors",
            ));
        }
        if chunk_tables.len() != factors * BINS {
            return Err(MetalError::RamRaStorageLength {
                expected: factors * BINS,
                got: chunk_tables.len(),
            });
        }
        let covered = e_in_capacity
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(rows))?;
        if e_in_capacity == 0 || e_out_capacity == 0 || covered != rows / 2 {
            return Err(MetalError::RamRaWeightShape {
                expected: rows / 2,
                covered,
            });
        }

        let lazy = [1, 2, 4, 8]
            .into_iter()
            .map(|width| {
                self.compile_ram_ra_width_pipeline(width)
                    .map(|pipeline| (width, pipeline))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pipelines = Pipelines {
            lazy,
            double: self.compile_named_pipeline(DOUBLE_PIPELINE)?,
            materialize: self.compile_named_pipeline(MATERIALIZE_PIPELINE)?,
            dense: self.compile_named_pipeline(DENSE_PIPELINE)?,
            reduce: self.compile_named_pipeline(REDUCE_PIPELINE)?,
        };
        let message_limits = Self::limits(&pipelines.lazy[0].1);
        let materialize_limits = Self::limits(&pipelines.materialize);
        let reduction_limits = Self::limits(&pipelines.reduce);
        for (name, limits) in [
            (LAZY_PIPELINE, message_limits),
            (DOUBLE_PIPELINE, Self::limits(&pipelines.double)),
            (MATERIALIZE_PIPELINE, materialize_limits),
            (DENSE_PIPELINE, Self::limits(&pipelines.dense)),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedRamRaExecutionWidth {
                    pipeline: name,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        for (_, pipeline) in &pipelines.lazy {
            let limits = Self::limits(pipeline);
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedRamRaExecutionWidth {
                    pipeline: LAZY_PIPELINE,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let message_threads =
            Self::resolve_threadgroup_width(Some(MESSAGE_THREADS), message_limits)?;
        let materialize_threads =
            Self::resolve_threadgroup_width(Some(MATERIALIZE_THREADS), materialize_limits)?;
        let branch_threads =
            Self::resolve_threadgroup_width(Some(BRANCH_THREADS), Self::limits(&pipelines.double))?;

        let branch_capacity = factors * MATERIALIZE_WIDTH * BINS;
        let dense_a_capacity = factors
            .checked_mul(rows / MATERIALIZE_WIDTH)
            .ok_or(MetalError::InputTooLong(rows))?;
        let dense_b_capacity = dense_a_capacity / 2;
        let partial_capacity = REDUCTION_COLUMNS
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(e_out_capacity))?;
        let source_bytes = byte_length::<u32>(rows)?;
        let owned_bytes = [
            byte_length::<Fp128>(branch_capacity)?,
            byte_length::<Fp128>(branch_capacity)?,
            byte_length::<Fp128>(dense_a_capacity)?,
            byte_length::<Fp128>(dense_b_capacity)?,
            byte_length::<Fp128>(e_in_capacity)?,
            byte_length::<Fp128>(e_out_capacity)?,
            byte_length::<Fp128>(partial_capacity)?,
            byte_length::<Fp128>(partial_capacity)?,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InputTooLong(rows))?;
        self.validate_buffer_length(source_bytes)?;
        self.validate_additional_working_set(
            source_bytes
                .checked_add(owned_bytes)
                .ok_or(MetalError::InputTooLong(rows))?,
        )?;

        let (addresses, _) = self.shared_no_copy_buffer(
            Arc::clone(&columns),
            columns.addresses.as_ptr().cast_mut().cast::<c_void>(),
            source_bytes,
        )?;
        let buffers = Buffers {
            addresses,
            branches_a: new_field_buffer(self, branch_capacity)?,
            branches_b: new_field_buffer(self, branch_capacity)?,
            dense_a: new_field_buffer(self, dense_a_capacity)?,
            dense_b: new_field_buffer(self, dense_b_capacity)?,
            e_in: new_field_buffer(self, e_in_capacity)?,
            e_out: new_field_buffer(self, e_out_capacity)?,
            partial_a: new_field_buffer(self, partial_capacity)?,
            partial_b: new_field_buffer(self, partial_capacity)?,
        };
        write_fields(&buffers.branches_a, branch_capacity, chunk_tables)?;

        Ok(RamRaSequence {
            context: self.clone(),
            _columns: columns,
            pipelines,
            reduction_limits,
            buffers,
            rows,
            factors,
            e_in_capacity,
            e_out_capacity,
            message_threads,
            materialize_threads,
            branch_threads,
            branch_width: 1,
            branches_in_a: true,
            dense: false,
            dense_in_a: true,
            dense_elements: 0,
        })
    }
}

impl RamRaSequence {
    pub(crate) fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        self.execute_lazy(None, e_in, e_out)
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        if self.dense {
            self.execute_dense(challenge, e_in, e_out)
        } else {
            self.execute_lazy(Some(challenge), e_in, e_out)
        }
    }

    pub(crate) fn finish_bind(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; MAX_FACTORS], MetalError> {
        if !self.dense || self.dense_elements != 2 {
            return Err(MetalError::InvalidRamRaState(
                "terminal bind requires two dense values per factor",
            ));
        }
        // SAFETY: the last dense command completed synchronously and the active
        // source contains exactly two initialized fields per factor.
        let values = unsafe {
            slice::from_raw_parts(
                self.dense_source_buffer().contents().cast::<Fp128>(),
                self.factors * 2,
            )
        };
        self.context
            .validate_inputs("RAM RA terminal tables", values)?;
        let mut output = [AkitaField::zero(); MAX_FACTORS];
        for (factor, output) in output.iter_mut().enumerate().take(self.factors) {
            let lo = values[2 * factor].into_jolt_field::<AkitaField>();
            let hi = values[2 * factor + 1].into_jolt_field::<AkitaField>();
            *output = lo + challenge * (hi - lo);
        }
        self.dense_elements = 1;
        Ok(output)
    }

    pub(crate) const fn current_elements(&self) -> usize {
        if self.dense {
            self.dense_elements
        } else {
            self.rows / self.branch_width
        }
    }

    pub(crate) const fn is_dense(&self) -> bool {
        self.dense
    }

    pub(crate) const fn branch_width(&self) -> usize {
        self.branch_width
    }

    fn execute_lazy(
        &mut self,
        challenge: Option<AkitaField>,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        if self.dense {
            return Err(MetalError::InvalidRamRaState(
                "the lazy prefix has already materialized",
            ));
        }
        let next_width = if challenge.is_some() {
            self.branch_width * 2
        } else {
            self.branch_width
        };
        if next_width > MATERIALIZE_WIDTH {
            return Err(MetalError::InvalidRamRaState(
                "branch width exceeds the materialization point",
            ));
        }
        let source_elements = self.rows / next_width;
        self.validate_weights(source_elements / 2, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let materialize = next_width == MATERIALIZE_WIDTH;
        let message_params = MessageParams {
            e_in_length: u32::try_from(e_in.len())
                .map_err(|_| MetalError::InputTooLong(e_in.len()))?,
            e_out_length: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            factor_count: self.factors as u32,
            _reserved: 0,
        };
        let materialize_params = MaterializeParams {
            source_elements: u32::try_from(source_elements)
                .map_err(|_| MetalError::InputTooLong(source_elements))?,
            e_in_length: message_params.e_in_length,
            e_out_length: message_params.e_out_length,
            factor_count: self.factors as u32,
        };
        let message_pipeline = if materialize {
            None
        } else {
            Some(self.lazy_pipeline(next_width)?.clone())
        };

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            let mut message_branches_in_a = self.branches_in_a;
            if let Some(challenge) = challenge {
                let params = BranchParams {
                    branch_width: self.branch_width as u32,
                    factor_count: self.factors as u32,
                    _reserved: [0; 2],
                };
                encoder.set_compute_pipeline_state(&self.pipelines.double);
                encoder.set_buffer(0, Some(self.branch_source_buffer()), 0);
                encoder.set_buffer(1, Some(self.branch_destination_buffer()), 0);
                set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(encoder, 3, &params);
                let elements = self.factors * self.branch_width * BINS;
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: elements.div_ceil(self.branch_threads) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.branch_threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                message_branches_in_a = !message_branches_in_a;
            }

            if materialize {
                encoder.set_compute_pipeline_state(&self.pipelines.materialize);
                encoder.set_buffer(0, Some(&self.buffers.addresses), 0);
                encoder.set_buffer(1, Some(self.branch_buffer(message_branches_in_a)), 0);
                encoder.set_buffer(2, Some(&self.buffers.dense_a), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 6, &materialize_params);
                Self::encode_message_dispatch(encoder, e_out.len(), self.materialize_threads);
            } else {
                encoder.set_compute_pipeline_state(message_pipeline.as_ref().ok_or(
                    MetalError::InvalidRamRaState("the lazy message pipeline is missing"),
                )?);
                encoder.set_buffer(0, Some(&self.buffers.addresses), 0);
                encoder.set_buffer(1, Some(self.branch_buffer(message_branches_in_a)), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 5, &message_params);
                Self::encode_message_dispatch(encoder, e_out.len(), self.message_threads);
            }
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                REDUCTION_COLUMNS,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;

        let message = self.finish_command(command_buffer, final_in_a)?;
        if challenge.is_some() {
            self.branch_width = next_width;
            self.branches_in_a = !self.branches_in_a;
        }
        if materialize {
            self.dense = true;
            self.dense_in_a = true;
            self.dense_elements = source_elements;
        }
        Ok(message)
    }

    fn execute_dense(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        if !self.dense || self.dense_elements < 4 {
            return Err(MetalError::InvalidRamRaState(
                "dense transition needs at least four elements per factor",
            ));
        }
        self.validate_weights(self.dense_elements / 4, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let params = MaterializeParams {
            source_elements: u32::try_from(self.dense_elements)
                .map_err(|_| MetalError::InputTooLong(self.dense_elements))?,
            e_in_length: u32::try_from(e_in.len())
                .map_err(|_| MetalError::InputTooLong(e_in.len()))?,
            e_out_length: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            factor_count: self.factors as u32,
        };

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.dense);
            encoder.set_buffer(0, Some(self.dense_source_buffer()), 0);
            encoder.set_buffer(1, Some(self.dense_destination_buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 5, &Fp128::from_jolt_field(&challenge));
            set_inline_bytes(encoder, 6, &params);
            Self::encode_message_dispatch(encoder, e_out.len(), self.message_threads);
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                REDUCTION_COLUMNS,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;

        let message = self.finish_command(command_buffer, final_in_a)?;
        self.dense_elements /= 2;
        self.dense_in_a = !self.dense_in_a;
        Ok(message)
    }

    fn validate_weights(
        &self,
        expected: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(), MetalError> {
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(expected))?;
        if e_in.is_empty()
            || e_out.is_empty()
            || e_in.len() > self.e_in_capacity
            || e_out.len() > self.e_out_capacity
            || covered != expected
        {
            return Err(MetalError::RamRaWeightShape { expected, covered });
        }
        Ok(())
    }

    fn write_weights(&self, e_in: &[AkitaField], e_out: &[AkitaField]) -> Result<(), MetalError> {
        write_fields(&self.buffers.e_in, self.e_in_capacity, e_in)?;
        write_fields(&self.buffers.e_out, self.e_out_capacity, e_out)
    }

    fn lazy_pipeline(&self, width: usize) -> Result<&ComputePipelineState, MetalError> {
        self.pipelines
            .lazy
            .iter()
            .find_map(|(pipeline_width, pipeline)| (*pipeline_width == width).then_some(pipeline))
            .ok_or(MetalError::InvalidRamRaState(
                "no lazy message pipeline for this branch width",
            ))
    }

    fn encode_message_dispatch(
        encoder: &metal::ComputeCommandEncoderRef,
        groups: usize,
        threads: usize,
    ) {
        let simdgroups = threads / SIMD_WIDTH;
        encoder.set_threadgroup_memory_length(
            0,
            (REDUCTION_COLUMNS * simdgroups * size_of::<Fp128>()) as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn finish_command(
        &self,
        command_buffer: &metal::CommandBufferRef,
        final_in_a: bool,
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        validate_completed_command(command_buffer)?;
        let buffer = if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the completed reduction initialized the leading value of
        // every reduction column in this shared buffer.
        let values =
            unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), REDUCTION_COLUMNS) };
        self.context.validate_inputs("RAM RA message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    fn branch_source_buffer(&self) -> &Buffer {
        self.branch_buffer(self.branches_in_a)
    }

    fn branch_destination_buffer(&self) -> &Buffer {
        self.branch_buffer(!self.branches_in_a)
    }

    fn branch_buffer(&self, in_a: bool) -> &Buffer {
        if in_a {
            &self.buffers.branches_a
        } else {
            &self.buffers.branches_b
        }
    }

    fn dense_source_buffer(&self) -> &Buffer {
        if self.dense_in_a {
            &self.buffers.dense_a
        } else {
            &self.buffers.dense_b
        }
    }

    fn dense_destination_buffer(&self) -> &Buffer {
        if self.dense_in_a {
            &self.buffers.dense_b
        } else {
            &self.buffers.dense_a
        }
    }
}

#[cfg(feature = "allocative")]
fn device_storage_bytes(buffers: &Buffers) -> usize {
    [
        &buffers.branches_a,
        &buffers.branches_b,
        &buffers.dense_a,
        &buffers.dense_b,
        &buffers.e_in,
        &buffers.e_out,
        &buffers.partial_a,
        &buffers.partial_b,
    ]
    .into_iter()
    .fold(0usize, |bytes, buffer| {
        bytes.saturating_add(buffer.length() as usize)
    })
}

fn new_field_buffer(context: &SolinasMetal, elements: usize) -> Result<Buffer, MetalError> {
    let bytes = byte_length::<Fp128>(elements)?;
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}

fn write_fields(buffer: &Buffer, capacity: usize, values: &[AkitaField]) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::RamRaStorageLength {
            expected: capacity,
            got: values.len(),
        });
    }
    // SAFETY: the shared buffer was allocated for `capacity` fields and no
    // command can observe it while the initial table prefix is written.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<MessageParams>() == 16);
const _: () = assert!(size_of::<BranchParams>() == 16);
const _: () = assert!(size_of::<MaterializeParams>() == 16);
