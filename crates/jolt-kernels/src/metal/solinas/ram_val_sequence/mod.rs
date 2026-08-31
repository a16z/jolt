use std::{
    ffi::c_void,
    mem::size_of,
    slice,
    sync::Arc,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use jolt_poly::{EqPolynomial, LtPolynomial};
use metal::{
    objc::rc::autoreleasepool, Buffer, CommandBuffer, CommandQueue, ComputePipelineState,
    FunctionConstantValues, MTLDataType, MTLResourceOptions, MTLSize,
};

use super::{
    completed_command_gpu_time, encode_column_reductions, set_inline_bytes,
    validate_completed_command, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use crate::optimized::ram_trace::{RamAccessColumns, RamIncrementActivity};

const FACTORS: usize = 3;
const REDUCTION_COLUMNS: usize = 4;
const MESSAGE_SAMPLES: usize = 3;
const SIMD_WIDTH: usize = 32;
const MESSAGE_THREADS: usize = 128;
const BRANCH_THREADS: usize = 256;
const MATERIALIZE_WIDTH: usize = 32;
const FIRST_MESSAGE_PREFETCH_MIN_ROWS: usize = 1 << 28;

const PREFIX_PIPELINE: &str = "solinas_ram_val_sparse_prefix";
const DOUBLE_PIPELINE: &str = "solinas_ram_val_double_branches";
const MATERIALIZE_PIPELINE: &str = "solinas_ram_val_materialize_width_32";
const DENSE_PIPELINE: &str = "solinas_ram_val_dense_transition";
const REDUCE_PIPELINE: &str = "solinas_instruction_ra_reduce";

#[repr(C)]
#[derive(Clone, Copy)]
struct PrefixParams {
    increment_count: u32,
    address_domain: u32,
    branch_width: u32,
    lt_lo_length: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaterializeParams {
    increment_count: u32,
    source_elements: u32,
    address_domain: u32,
    lt_lo_length: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DenseParams {
    source_elements: u32,
    _reserved_0: u32,
    _reserved_1: [u32; 2],
}

struct Pipelines {
    prefix: Vec<(usize, ComputePipelineState)>,
    double: ComputePipelineState,
    materialize: ComputePipelineState,
    dense: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct Buffers {
    addresses: Buffer,
    increment_cycles: Buffer,
    increments: Buffer,
    branches_a: Buffer,
    branches_b: Buffer,
    cycle_weights: Buffer,
    lt_lo: Buffer,
    lt_hi: Buffer,
    eq_hi: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

struct PendingFirstMessage {
    command_buffer: Option<CommandBuffer>,
    _queue: CommandQueue,
    final_in_a: bool,
    submitted_at: Instant,
}

impl Drop for PendingFirstMessage {
    fn drop(&mut self) {
        if let Some(command_buffer) = &self.command_buffer {
            command_buffer.wait_until_completed();
        }
    }
}

pub(crate) struct RamValSequence {
    context: SolinasMetal,
    _columns: Arc<RamAccessColumns>,
    _increments: Arc<RamIncrementActivity>,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    address_domain: usize,
    increment_count: usize,
    partial_groups: usize,
    message_threads: usize,
    branch_threads: usize,
    branch_width: usize,
    branches_in_a: bool,
    cycle_weights: Vec<AkitaField>,
    lt_lo: Vec<AkitaField>,
    lt_lo_capacity: usize,
    dense: bool,
    dense_in_a: bool,
    dense_elements: usize,
    pending_first_message: Option<PendingFirstMessage>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamValSequence {
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
        visitor.visit_simple(
            allocative::Key::new("aliased_increment_cycles"),
            self.increment_count.saturating_mul(size_of::<u64>()),
        );
        visitor.visit_simple(
            allocative::Key::new("aliased_increments"),
            self.increment_count.saturating_mul(size_of::<i128>()),
        );
        visitor.exit();
    }
}

impl SolinasMetal {
    fn compile_ram_val_prefix_pipeline(
        &self,
        width: usize,
    ) -> Result<ComputePipelineState, MetalError> {
        let key = (PREFIX_PIPELINE, Some(width as u32));
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
            21,
        );
        let function = self
            .library
            .get_function(PREFIX_PIPELINE, Some(constants))
            .map_err(|message| MetalError::FunctionLookup {
                name: PREFIX_PIPELINE,
                message,
            })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation {
                name: PREFIX_PIPELINE,
                message,
            })?;
        let _ = cache.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    pub(crate) fn prepare_ram_val_sequence(
        &self,
        columns: Arc<RamAccessColumns>,
        increments: Arc<RamIncrementActivity>,
        r_address: &[AkitaField],
        r_cycle: &[AkitaField],
        gamma: AkitaField,
    ) -> Result<RamValSequence, MetalError> {
        let rows = columns.addresses.len();
        if rows < 2 * MATERIALIZE_WIDTH || !rows.is_power_of_two() {
            return Err(MetalError::InvalidRamRaRows(rows));
        }
        if r_cycle.len() != rows.ilog2() as usize {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check cycle point does not match the row domain",
            ));
        }
        if increments.len() == 0 {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check sparse prefix requires a nonempty increment stream",
            ));
        }
        if r_address.len() >= usize::BITS as usize {
            return Err(MetalError::InputTooLong(r_address.len()));
        }
        let address_domain = 1usize << r_address.len();
        if address_domain > u32::MAX as usize {
            return Err(MetalError::InputTooLong(address_domain));
        }

        let prefix = [1, 2, 4, 8, 16]
            .into_iter()
            .map(|width| {
                self.compile_ram_val_prefix_pipeline(width)
                    .map(|pipeline| (width, pipeline))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pipelines = Pipelines {
            prefix,
            double: self.compile_named_pipeline(DOUBLE_PIPELINE)?,
            materialize: self.compile_named_pipeline(MATERIALIZE_PIPELINE)?,
            dense: self.compile_named_pipeline(DENSE_PIPELINE)?,
            reduce: self.compile_named_pipeline(REDUCE_PIPELINE)?,
        };
        let message_limits = Self::limits(&pipelines.prefix[0].1);
        let reduction_limits = Self::limits(&pipelines.reduce);
        for (name, limits) in [
            (PREFIX_PIPELINE, message_limits),
            (DOUBLE_PIPELINE, Self::limits(&pipelines.double)),
            (MATERIALIZE_PIPELINE, Self::limits(&pipelines.materialize)),
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
        for (_, pipeline) in &pipelines.prefix {
            let limits = Self::limits(pipeline);
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedRamRaExecutionWidth {
                    pipeline: PREFIX_PIPELINE,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let message_threads =
            Self::resolve_threadgroup_width(Some(MESSAGE_THREADS), message_limits)?;
        let branch_threads =
            Self::resolve_threadgroup_width(Some(BRANCH_THREADS), Self::limits(&pipelines.double))?;

        let eq_address = EqPolynomial::<AkitaField>::evals(r_address, None);
        if eq_address.len() != address_domain {
            return Err(MetalError::RamRaStorageLength {
                expected: address_domain,
                got: eq_address.len(),
            });
        }
        let low_variables = r_cycle.len() / 2;
        let high_variables = r_cycle.len() - low_variables;
        let (r_hi, r_lo) = r_cycle.split_at(high_variables);
        if r_lo.len() < MATERIALIZE_WIDTH.ilog2() as usize {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check split LT exhausts before width-32 materialization",
            ));
        }
        let lt_lo = LtPolynomial::<AkitaField>::evaluations(r_lo);
        let lt_hi = LtPolynomial::<AkitaField>::evaluations(r_hi)
            .into_iter()
            .map(|value| value + gamma)
            .collect::<Vec<_>>();
        let eq_hi = EqPolynomial::<AkitaField>::evals(r_hi, None);
        for (name, values) in [
            ("RAM value-check address equality", eq_address.as_slice()),
            ("RAM value-check LT low", lt_lo.as_slice()),
            ("RAM value-check LT high", lt_hi.as_slice()),
            ("RAM value-check equality high", eq_hi.as_slice()),
        ] {
            self.validate_inputs(
                name,
                &values
                    .iter()
                    .map(Fp128::from_jolt_field)
                    .collect::<Vec<_>>(),
            )?;
        }

        let increment_count = increments.len();
        let lt_lo_capacity = lt_lo.len();
        let source_elements = rows / MATERIALIZE_WIDTH;
        let materialize_pairs = source_elements / 2;
        let prefix_groups = increments.len().div_ceil(message_threads);
        let materialize_groups = materialize_pairs.div_ceil(message_threads);
        let partial_groups = prefix_groups.max(materialize_groups);
        let partial_capacity = REDUCTION_COLUMNS
            .checked_mul(partial_groups)
            .ok_or(MetalError::InputTooLong(partial_groups))?;
        let branches_a_capacity = 16usize
            .checked_mul(address_domain)
            .ok_or(MetalError::InputTooLong(address_domain))?;
        let branches_b_capacity = MATERIALIZE_WIDTH
            .checked_mul(address_domain)
            .ok_or(MetalError::InputTooLong(address_domain))?;
        let dense_a_capacity = FACTORS
            .checked_mul(source_elements)
            .ok_or(MetalError::InputTooLong(source_elements))?;
        let dense_b_capacity = dense_a_capacity / 2;

        let address_bytes = byte_length::<u32>(rows)?;
        let increment_cycle_bytes = byte_length::<u64>(increments.len())?;
        let increment_bytes = byte_length::<i128>(increments.len())?;
        let owned_bytes = [
            byte_length::<Fp128>(branches_a_capacity)?,
            byte_length::<Fp128>(branches_b_capacity)?,
            byte_length::<Fp128>(MATERIALIZE_WIDTH)?,
            byte_length::<Fp128>(lt_lo.len())?,
            byte_length::<Fp128>(lt_hi.len())?,
            byte_length::<Fp128>(eq_hi.len())?,
            byte_length::<Fp128>(dense_a_capacity)?,
            byte_length::<Fp128>(dense_b_capacity)?,
            byte_length::<Fp128>(partial_capacity)?,
            byte_length::<Fp128>(partial_capacity)?,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InputTooLong(rows))?;
        for bytes in [address_bytes, increment_cycle_bytes, increment_bytes] {
            self.validate_buffer_length(bytes)?;
        }
        self.validate_additional_working_set(
            address_bytes
                .checked_add(increment_cycle_bytes)
                .and_then(|bytes| bytes.checked_add(increment_bytes))
                .and_then(|bytes| bytes.checked_add(owned_bytes))
                .ok_or(MetalError::InputTooLong(rows))?,
        )?;

        let (addresses, _) = self.shared_no_copy_buffer(
            Arc::clone(&columns),
            columns.addresses.as_ptr().cast_mut().cast::<c_void>(),
            address_bytes,
        )?;
        let increment_cycles = self.device.new_buffer_with_bytes_no_copy(
            increments
                .cycle_slice()
                .as_ptr()
                .cast_mut()
                .cast::<c_void>(),
            increment_cycle_bytes,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let increment_buffer = self.device.new_buffer_with_bytes_no_copy(
            increments
                .increment_slice()
                .as_ptr()
                .cast_mut()
                .cast::<c_void>(),
            increment_bytes,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let buffers = Buffers {
            addresses,
            increment_cycles,
            increments: increment_buffer,
            branches_a: new_field_buffer(self, branches_a_capacity)?,
            branches_b: new_field_buffer(self, branches_b_capacity)?,
            cycle_weights: new_field_buffer(self, MATERIALIZE_WIDTH)?,
            lt_lo: new_field_buffer(self, lt_lo.len())?,
            lt_hi: new_field_buffer(self, lt_hi.len())?,
            eq_hi: new_field_buffer(self, eq_hi.len())?,
            dense_a: new_field_buffer(self, dense_a_capacity)?,
            dense_b: new_field_buffer(self, dense_b_capacity)?,
            partial_a: new_field_buffer(self, partial_capacity)?,
            partial_b: new_field_buffer(self, partial_capacity)?,
        };
        write_fields(&buffers.branches_a, branches_a_capacity, &eq_address)?;
        write_fields(
            &buffers.cycle_weights,
            MATERIALIZE_WIDTH,
            &[AkitaField::one()],
        )?;
        write_fields(&buffers.lt_lo, lt_lo.len(), &lt_lo)?;
        write_fields(&buffers.lt_hi, lt_hi.len(), &lt_hi)?;
        write_fields(&buffers.eq_hi, eq_hi.len(), &eq_hi)?;

        let mut sequence = RamValSequence {
            context: self.clone(),
            _columns: columns,
            _increments: increments,
            pipelines,
            reduction_limits,
            buffers,
            rows,
            address_domain,
            increment_count,
            partial_groups,
            message_threads,
            branch_threads,
            branch_width: 1,
            branches_in_a: true,
            cycle_weights: vec![AkitaField::one()],
            lt_lo,
            lt_lo_capacity,
            dense: false,
            dense_in_a: true,
            dense_elements: 0,
            pending_first_message: None,
        };
        if rows >= FIRST_MESSAGE_PREFETCH_MIN_ROWS {
            sequence.submit_first_message()?;
        }
        Ok(sequence)
    }
}

impl RamValSequence {
    pub(crate) fn message(&mut self) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        if let Some(mut pending) = self.pending_first_message.take() {
            let command_buffer =
                pending
                    .command_buffer
                    .take()
                    .ok_or(MetalError::InvalidRamRaState(
                        "RAM value-check prefetch lost its command",
                    ))?;
            let join_started = Instant::now();
            command_buffer.wait_until_completed();
            let join = join_started.elapsed();
            let gpu_active = completed_command_gpu_time(&command_buffer)?;
            let total = pending.submitted_at.elapsed();
            tracing::info!(
                target: "jolt::metal",
                join_ns = duration_nanos(join),
                gpu_active_ns = duration_nanos(gpu_active),
                total_ns = duration_nanos(total),
                "joined prefetched RAM value-check first message"
            );
            return self.finish_command(&command_buffer, pending.final_in_a);
        }
        self.execute_prefix(None)
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        if self.dense {
            self.execute_dense(challenge)
        } else {
            self.execute_prefix(Some(challenge))
        }
    }

    pub(crate) fn finish_bind(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; FACTORS], MetalError> {
        if !self.dense || self.dense_elements != 2 {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check terminal bind requires two dense values per factor",
            ));
        }
        // SAFETY: the last dense command completed synchronously and the active
        // source contains exactly two initialized fields per factor.
        let values = unsafe {
            slice::from_raw_parts(
                self.dense_source_buffer().contents().cast::<Fp128>(),
                FACTORS * 2,
            )
        };
        self.context
            .validate_inputs("RAM value-check terminal tables", values)?;
        let mut output = [AkitaField::zero(); FACTORS];
        for (factor, output) in output.iter_mut().enumerate() {
            let low = values[2 * factor].into_jolt_field::<AkitaField>();
            let high = values[2 * factor + 1].into_jolt_field::<AkitaField>();
            *output = low + challenge * (high - low);
        }
        self.dense_elements = 1;
        Ok(output)
    }

    fn execute_prefix(
        &mut self,
        challenge: Option<AkitaField>,
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        if self.dense {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check sparse prefix has already materialized",
            ));
        }
        let next_width = if challenge.is_some() {
            self.branch_width * 2
        } else {
            self.branch_width
        };
        if next_width > MATERIALIZE_WIDTH {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check branch width exceeds the materialization point",
            ));
        }

        let next_cycle_weights = challenge
            .map(|challenge| bind_branch_weights(&self.cycle_weights, challenge))
            .transpose()?
            .unwrap_or_else(|| self.cycle_weights.clone());
        let next_lt_lo = challenge
            .map(|challenge| bind_adjacent(&self.lt_lo, challenge))
            .transpose()?
            .unwrap_or_else(|| self.lt_lo.clone());
        write_fields(
            &self.buffers.cycle_weights,
            MATERIALIZE_WIDTH,
            &next_cycle_weights,
        )?;
        write_fields(&self.buffers.lt_lo, self.lt_lo_capacity, &next_lt_lo)?;

        let materialize = next_width == MATERIALIZE_WIDTH;
        let groups = if materialize {
            (self.rows / MATERIALIZE_WIDTH / 2).div_ceil(self.message_threads)
        } else {
            self.increment_count.div_ceil(self.message_threads)
        };
        if groups == 0 || groups > self.partial_groups {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check message dispatch exceeds its prepared reduction storage",
            ));
        }
        let prefix_params = PrefixParams {
            increment_count: u32::try_from(self.increment_count)
                .map_err(|_| MetalError::InputTooLong(self.increment_count))?,
            address_domain: u32::try_from(self.address_domain)
                .map_err(|_| MetalError::InputTooLong(self.address_domain))?,
            branch_width: u32::try_from(next_width)
                .map_err(|_| MetalError::InputTooLong(next_width))?,
            lt_lo_length: u32::try_from(next_lt_lo.len())
                .map_err(|_| MetalError::InputTooLong(next_lt_lo.len()))?,
        };
        let materialize_params = MaterializeParams {
            increment_count: prefix_params.increment_count,
            source_elements: u32::try_from(self.rows / MATERIALIZE_WIDTH)
                .map_err(|_| MetalError::InputTooLong(self.rows / MATERIALIZE_WIDTH))?,
            address_domain: prefix_params.address_domain,
            lt_lo_length: prefix_params.lt_lo_length,
        };

        let queue = if self.rows >= FIRST_MESSAGE_PREFETCH_MIN_ROWS
            && challenge.is_some()
            && self.branch_width == 1
        {
            self.context.device.new_command_queue()
        } else {
            self.context.queue.clone()
        };
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            let mut message_branches_in_a = self.branches_in_a;
            if let Some(challenge) = challenge {
                let double_params = PrefixParams {
                    branch_width: u32::try_from(self.branch_width)
                        .map_err(|_| MetalError::InputTooLong(self.branch_width))?,
                    ..prefix_params
                };
                encoder.set_compute_pipeline_state(&self.pipelines.double);
                encoder.set_buffer(0, Some(self.branch_source_buffer()), 0);
                encoder.set_buffer(1, Some(self.branch_destination_buffer()), 0);
                set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(encoder, 3, &double_params);
                let elements = self.branch_width * self.address_domain;
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
                self.encode_prefix_buffers(encoder, message_branches_in_a);
                encoder.set_buffer(8, Some(&self.buffers.dense_a), 0);
                encoder.set_buffer(9, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 10, &materialize_params);
            } else {
                encoder.set_compute_pipeline_state(self.prefix_pipeline(next_width)?);
                self.encode_prefix_buffers(encoder, message_branches_in_a);
                encoder.set_buffer(8, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 9, &prefix_params);
            }
            self.encode_message_dispatch(encoder, groups);
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                groups,
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
            self.cycle_weights = next_cycle_weights;
            self.lt_lo = next_lt_lo;
        }
        if materialize {
            self.dense = true;
            self.dense_in_a = true;
            self.dense_elements = self.rows / MATERIALIZE_WIDTH;
        }
        Ok(message)
    }

    fn submit_first_message(&mut self) -> Result<(), MetalError> {
        if self.dense || self.branch_width != 1 || self.pending_first_message.is_some() {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check first-message prefetch has an invalid state",
            ));
        }
        write_fields(
            &self.buffers.cycle_weights,
            MATERIALIZE_WIDTH,
            &self.cycle_weights,
        )?;
        write_fields(&self.buffers.lt_lo, self.lt_lo_capacity, &self.lt_lo)?;
        let groups = self.increment_count.div_ceil(self.message_threads);
        if groups == 0 || groups > self.partial_groups {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check prefetched message exceeds its reduction storage",
            ));
        }
        let params = PrefixParams {
            increment_count: u32::try_from(self.increment_count)
                .map_err(|_| MetalError::InputTooLong(self.increment_count))?,
            address_domain: u32::try_from(self.address_domain)
                .map_err(|_| MetalError::InputTooLong(self.address_domain))?,
            branch_width: 1,
            lt_lo_length: u32::try_from(self.lt_lo.len())
                .map_err(|_| MetalError::InputTooLong(self.lt_lo.len()))?,
        };
        let queue = self.context.device.new_command_queue();
        let command_buffer = queue.new_command_buffer().to_owned();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(self.prefix_pipeline(1)?);
            self.encode_prefix_buffers(encoder, self.branches_in_a);
            encoder.set_buffer(8, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 9, &params);
            self.encode_message_dispatch(encoder, groups);
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                groups,
                REDUCTION_COLUMNS,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            Ok::<bool, MetalError>(final_in_a)
        })?;
        self.pending_first_message = Some(PendingFirstMessage {
            command_buffer: Some(command_buffer),
            _queue: queue,
            final_in_a,
            submitted_at: Instant::now(),
        });
        tracing::info!(
            target: "jolt::metal",
            groups,
            "submitted RAM value-check first-message prefetch"
        );
        Ok(())
    }

    fn execute_dense(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; MESSAGE_SAMPLES], MetalError> {
        if !self.dense || self.dense_elements < 4 {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check dense transition needs at least four elements per factor",
            ));
        }
        let pairs = self.dense_elements / 4;
        let groups = pairs.div_ceil(self.message_threads);
        if groups == 0 || groups > self.partial_groups {
            return Err(MetalError::InvalidRamRaState(
                "RAM value-check dense dispatch exceeds its prepared reduction storage",
            ));
        }
        let params = DenseParams {
            source_elements: u32::try_from(self.dense_elements)
                .map_err(|_| MetalError::InputTooLong(self.dense_elements))?,
            _reserved_0: 0,
            _reserved_1: [0; 2],
        };

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.dense);
            encoder.set_buffer(0, Some(self.dense_source_buffer()), 0);
            encoder.set_buffer(1, Some(self.dense_destination_buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 3, &Fp128::from_jolt_field(&challenge));
            set_inline_bytes(encoder, 4, &params);
            self.encode_message_dispatch(encoder, groups);
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                groups,
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

    fn encode_prefix_buffers(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        branches_in_a: bool,
    ) {
        encoder.set_buffer(0, Some(&self.buffers.addresses), 0);
        encoder.set_buffer(1, Some(&self.buffers.increment_cycles), 0);
        encoder.set_buffer(2, Some(&self.buffers.increments), 0);
        encoder.set_buffer(3, Some(self.branch_buffer(branches_in_a)), 0);
        encoder.set_buffer(4, Some(&self.buffers.cycle_weights), 0);
        encoder.set_buffer(5, Some(&self.buffers.lt_lo), 0);
        encoder.set_buffer(6, Some(&self.buffers.lt_hi), 0);
        encoder.set_buffer(7, Some(&self.buffers.eq_hi), 0);
    }

    fn encode_message_dispatch(&self, encoder: &metal::ComputeCommandEncoderRef, groups: usize) {
        let simdgroups = self.message_threads / SIMD_WIDTH;
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
                width: self.message_threads as u64,
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
        self.context
            .validate_inputs("RAM value-check message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    fn prefix_pipeline(&self, width: usize) -> Result<&ComputePipelineState, MetalError> {
        self.pipelines
            .prefix
            .iter()
            .find_map(|(pipeline_width, pipeline)| (*pipeline_width == width).then_some(pipeline))
            .ok_or(MetalError::InvalidRamRaState(
                "RAM value-check sparse prefix pipeline is missing",
            ))
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

fn bind_branch_weights(
    weights: &[AkitaField],
    challenge: AkitaField,
) -> Result<Vec<AkitaField>, MetalError> {
    let next_len = weights
        .len()
        .checked_mul(2)
        .ok_or(MetalError::InputTooLong(weights.len()))?;
    if next_len > MATERIALIZE_WIDTH {
        return Err(MetalError::InvalidRamRaState(
            "RAM value-check cycle branch weights exceed width 32",
        ));
    }
    let mut next = Vec::with_capacity(next_len);
    let complement = AkitaField::one() - challenge;
    next.extend(weights.iter().map(|weight| *weight * complement));
    next.extend(weights.iter().map(|weight| *weight * challenge));
    Ok(next)
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn bind_adjacent(
    values: &[AkitaField],
    challenge: AkitaField,
) -> Result<Vec<AkitaField>, MetalError> {
    if values.len() < 2 || !values.len().is_multiple_of(2) {
        return Err(MetalError::InvalidRamRaState(
            "RAM value-check LT-low table cannot bind adjacent entries",
        ));
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
        .collect())
}

#[cfg(feature = "allocative")]
fn device_storage_bytes(buffers: &Buffers) -> usize {
    [
        &buffers.branches_a,
        &buffers.branches_b,
        &buffers.cycle_weights,
        &buffers.lt_lo,
        &buffers.lt_hi,
        &buffers.eq_hi,
        &buffers.dense_a,
        &buffers.dense_b,
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
    // SAFETY: the shared buffer was allocated for `capacity` fields and every
    // caller waits for the previous command before replacing its active prefix.
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

const _: () = assert!(size_of::<PrefixParams>() == 16);
const _: () = assert!(size_of::<MaterializeParams>() == 16);
const _: () = assert!(size_of::<DenseParams>() == 16);
