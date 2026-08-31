use std::{ffi::c_void, mem::size_of, slice, sync::Arc};

use jolt_field::AkitaField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, FunctionConstantValues, MTLDataType,
    MTLResourceOptions, MTLSize,
};

use super::{
    encode_column_reductions, set_inline_bytes, validate_completed_command, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};
use crate::optimized::ram_trace::RamAccessColumns;

const SIMD_WIDTH: usize = 32;
const MESSAGE_THREADS: usize = 128;
const PACK_THREADS: usize = 256;
const MATERIALIZE_WIDTH: usize = 32;
const REDUCTION_COLUMNS: usize = 2;
const VALUE_TABLE_CAPACITY: usize = 4 * 256;
const Q_TABLE_PATTERNS_MAX: usize = 1 << 16;
const Q_TABLE_CAPACITY: usize = 2 * Q_TABLE_PATTERNS_MAX;

const PACK_PIPELINE: &str = "solinas_ram_hamming_pack";
const PREFIX_PIPELINE: &str = "solinas_ram_hamming_prefix";
const DENSE_PIPELINE: &str = "solinas_ram_hamming_dense_transition";
const REDUCE_PIPELINE: &str = "solinas_booleanity_reduce";

#[repr(C)]
#[derive(Clone, Copy)]
struct PackParams {
    words: u32,
    _reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PrefixParams {
    e_in_length: u32,
    e_out_length: u32,
    q_patterns: u32,
    materialize: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DenseParams {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    _reserved: u32,
}

struct Pipelines {
    pack: ComputePipelineState,
    prefix: Vec<(usize, ComputePipelineState)>,
    dense: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct Buffers {
    addresses: Buffer,
    access_bits: Buffer,
    value_table: Buffer,
    q_table: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RamHammingTerminal {
    hamming: AkitaField,
    eq_cycle: AkitaField,
}

impl RamHammingTerminal {
    copy_field_getters! { pub(crate), {
        hamming: AkitaField,
        eq_cycle: AkitaField,
    }}
}

pub(crate) struct RamHammingSequence {
    context: SolinasMetal,
    _columns: Arc<RamAccessColumns>,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    eq: GruenSplitEqPolynomial<AkitaField>,
    rows: usize,
    message_threads: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    branch_weights: Vec<AkitaField>,
    branch_width: usize,
    packed: bool,
    dense: bool,
    dense_in_a: bool,
    dense_elements: usize,
    round: usize,
    rounds: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamHammingSequence {
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
            allocative::Key::new("branch_weights"),
            self.branch_weights
                .capacity()
                .saturating_mul(size_of::<AkitaField>()),
        );
        visitor.exit();
    }
}

impl SolinasMetal {
    fn compile_ram_hamming_prefix_pipeline(
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
            22,
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

    pub(crate) fn prepare_ram_hamming_sequence(
        &self,
        columns: Arc<RamAccessColumns>,
        stage1_cycle_binding: &[AkitaField],
    ) -> Result<RamHammingSequence, MetalError> {
        let rows = columns.addresses.len();
        if rows < 2 * MATERIALIZE_WIDTH || !rows.is_power_of_two() {
            return Err(MetalError::InvalidRamRaRows(rows));
        }
        let rounds = rows.ilog2() as usize;
        if stage1_cycle_binding.len() != rounds {
            return Err(MetalError::InvalidRamRaState(
                "RAM Hamming cycle point does not match the row domain",
            ));
        }
        let eq_point = stage1_cycle_binding
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let eq = GruenSplitEqPolynomial::new(&eq_point, BindingOrder::LowToHigh);
        let e_in_capacity = eq.e_in_current_len();
        let e_out_capacity = eq.e_out_current_len();

        let pipelines = Pipelines {
            pack: self.compile_named_pipeline(PACK_PIPELINE)?,
            prefix: [1, 2, 4, 8, 16, 32]
                .into_iter()
                .map(|width| {
                    self.compile_ram_hamming_prefix_pipeline(width)
                        .map(|pipeline| (width, pipeline))
                })
                .collect::<Result<Vec<_>, _>>()?,
            dense: self.compile_named_pipeline(DENSE_PIPELINE)?,
            reduce: self.compile_named_pipeline(REDUCE_PIPELINE)?,
        };
        let reduction_limits = Self::limits(&pipelines.reduce);
        for (name, limits) in [
            (PACK_PIPELINE, Self::limits(&pipelines.pack)),
            (PREFIX_PIPELINE, Self::limits(&pipelines.prefix[0].1)),
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
        let message_threads = Self::resolve_threadgroup_width(
            Some(MESSAGE_THREADS),
            Self::limits(&pipelines.prefix[0].1),
        )?;

        let address_bytes = byte_length::<u32>(rows)?;
        let access_words = rows / u32::BITS as usize;
        let dense_elements = rows / MATERIALIZE_WIDTH;
        let partial_capacity = REDUCTION_COLUMNS
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(e_out_capacity))?;
        let owned_bytes = [
            byte_length::<u32>(access_words)?,
            byte_length::<Fp128>(VALUE_TABLE_CAPACITY)?,
            byte_length::<Fp128>(Q_TABLE_CAPACITY)?,
            byte_length::<Fp128>(e_in_capacity)?,
            byte_length::<Fp128>(e_out_capacity)?,
            byte_length::<Fp128>(dense_elements)?,
            byte_length::<Fp128>(dense_elements / 2)?,
            byte_length::<Fp128>(partial_capacity)?,
            byte_length::<Fp128>(partial_capacity)?,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InputTooLong(rows))?;
        self.validate_buffer_length(address_bytes)?;
        self.validate_additional_working_set(
            address_bytes
                .checked_add(owned_bytes)
                .ok_or(MetalError::InputTooLong(rows))?,
        )?;

        let (addresses, _) = self.shared_no_copy_buffer(
            Arc::clone(&columns),
            columns.addresses.as_ptr().cast_mut().cast::<c_void>(),
            address_bytes,
        )?;
        let buffers = Buffers {
            addresses,
            access_bits: self.new_ram_hamming_buffer::<u32>(access_words)?,
            value_table: self.new_ram_hamming_buffer::<Fp128>(VALUE_TABLE_CAPACITY)?,
            q_table: self.new_ram_hamming_buffer::<Fp128>(Q_TABLE_CAPACITY)?,
            e_in: self.new_ram_hamming_buffer::<Fp128>(e_in_capacity)?,
            e_out: self.new_ram_hamming_buffer::<Fp128>(e_out_capacity)?,
            dense_a: self.new_ram_hamming_buffer::<Fp128>(dense_elements)?,
            dense_b: self.new_ram_hamming_buffer::<Fp128>(dense_elements / 2)?,
            partial_a: self.new_ram_hamming_buffer::<Fp128>(partial_capacity)?,
            partial_b: self.new_ram_hamming_buffer::<Fp128>(partial_capacity)?,
        };
        Ok(RamHammingSequence {
            context: self.clone(),
            _columns: columns,
            pipelines,
            reduction_limits,
            buffers,
            eq,
            rows,
            message_threads,
            e_in_capacity,
            e_out_capacity,
            branch_weights: vec![AkitaField::one()],
            branch_width: 1,
            packed: false,
            dense: false,
            dense_in_a: true,
            dense_elements: 0,
            round: 0,
            rounds,
        })
    }

    fn new_ram_hamming_buffer<T>(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = byte_length::<T>(elements)?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl RamHammingSequence {
    pub(crate) fn message(
        &mut self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, MetalError> {
        if self.round != 0 || self.dense {
            return Err(MetalError::InvalidRamRaState(
                "RAM Hamming initial message is out of order",
            ));
        }
        self.execute_prefix(previous_claim)
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, MetalError> {
        if self.round == 0 || self.round >= self.rounds {
            return Err(MetalError::InvalidRamRaState(
                "RAM Hamming bind/message is out of order",
            ));
        }
        self.eq.bind(challenge);
        if self.dense {
            self.execute_dense(challenge, previous_claim)
        } else {
            self.branch_weights = bind_branch_weights(&self.branch_weights, challenge)?;
            self.branch_width *= 2;
            self.execute_prefix(previous_claim)
        }
    }

    pub(crate) fn finish_bind(
        &mut self,
        challenge: AkitaField,
    ) -> Result<RamHammingTerminal, MetalError> {
        if self.round != self.rounds || !self.dense || self.dense_elements != 2 {
            return Err(MetalError::InvalidRamRaState(
                "RAM Hamming terminal bind has the wrong sequence state",
            ));
        }
        self.eq.bind(challenge);
        let source = self.dense_source_buffer();
        // SAFETY: the final dense command completed synchronously and the
        // active buffer contains exactly two initialized leading fields.
        let values = unsafe { slice::from_raw_parts(source.contents().cast::<Fp128>(), 2) };
        self.context
            .validate_inputs("RAM Hamming terminal", values)?;
        let low = values[0].into_jolt_field::<AkitaField>();
        let high = values[1].into_jolt_field::<AkitaField>();
        self.dense_elements = 1;
        Ok(RamHammingTerminal {
            hamming: low + challenge * (high - low),
            eq_cycle: self.eq.current_scalar(),
        })
    }

    fn execute_prefix(
        &mut self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, MetalError> {
        let width = self.branch_width;
        if ![1, 2, 4, 8, 16, 32].contains(&width) {
            return Err(MetalError::InvalidRamRaState(
                "RAM Hamming prefix width is unsupported",
            ));
        }
        let pairs = self.rows / (2 * width);
        self.validate_eq_shape(pairs)?;
        self.write_eq_tables()?;
        let q_patterns = if (2..=8).contains(&width) {
            let q_table = build_q_table(&self.branch_weights)?;
            let patterns = q_table.len() / REDUCTION_COLUMNS;
            write_fields(&self.buffers.q_table, Q_TABLE_CAPACITY, &q_table)?;
            patterns
        } else {
            0
        };
        if width >= 16 {
            let value_table = build_value_table(&self.branch_weights)?;
            write_fields(
                &self.buffers.value_table,
                VALUE_TABLE_CAPACITY,
                &value_table,
            )?;
        }
        let materialize = width == MATERIALIZE_WIDTH;
        let params = PrefixParams {
            e_in_length: u32::try_from(self.eq.e_in_current_len())
                .map_err(|_| MetalError::InputTooLong(self.eq.e_in_current_len()))?,
            e_out_length: u32::try_from(self.eq.e_out_current_len())
                .map_err(|_| MetalError::InputTooLong(self.eq.e_out_current_len()))?,
            q_patterns: u32::try_from(q_patterns)
                .map_err(|_| MetalError::InputTooLong(q_patterns))?,
            materialize: u32::from(materialize),
        };
        let pack_params = PackParams {
            words: u32::try_from(self.rows / u32::BITS as usize)
                .map_err(|_| MetalError::InputTooLong(self.rows))?,
            _reserved: [0; 3],
        };
        let groups = self.eq.e_out_current_len();
        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            if !self.packed {
                encoder.set_compute_pipeline_state(&self.pipelines.pack);
                encoder.set_buffer(0, Some(&self.buffers.addresses), 0);
                encoder.set_buffer(1, Some(&self.buffers.access_bits), 0);
                set_inline_bytes(encoder, 2, &pack_params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: (pack_params.words as usize).div_ceil(PACK_THREADS) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: PACK_THREADS as u64,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.set_compute_pipeline_state(self.prefix_pipeline(width)?);
            encoder.set_buffer(0, Some(&self.buffers.access_bits), 0);
            encoder.set_buffer(1, Some(&self.buffers.value_table), 0);
            encoder.set_buffer(2, Some(&self.buffers.q_table), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(5, Some(&self.buffers.dense_a), 0);
            encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 7, &params);
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
        let [constant, leading] = self.finish_command(command_buffer, final_in_a)?;
        self.packed = true;
        if materialize {
            self.dense = true;
            self.dense_in_a = true;
            self.dense_elements = self.rows / MATERIALIZE_WIDTH;
        }
        self.round += 1;
        Ok(self.eq.gruen_poly_deg_3(constant, leading, previous_claim))
    }

    fn execute_dense(
        &mut self,
        challenge: AkitaField,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, MetalError> {
        if self.dense_elements < 4 {
            return Err(MetalError::InvalidRamRaState(
                "RAM Hamming dense transition is exhausted",
            ));
        }
        let pairs = self.dense_elements / 4;
        self.validate_eq_shape(pairs)?;
        self.write_eq_tables()?;
        let params = DenseParams {
            source_elements: u32::try_from(self.dense_elements)
                .map_err(|_| MetalError::InputTooLong(self.dense_elements))?,
            e_in_length: u32::try_from(self.eq.e_in_current_len())
                .map_err(|_| MetalError::InputTooLong(self.eq.e_in_current_len()))?,
            e_out_length: u32::try_from(self.eq.e_out_current_len())
                .map_err(|_| MetalError::InputTooLong(self.eq.e_out_current_len()))?,
            _reserved: 0,
        };
        let challenge = Fp128::from_jolt_field(&challenge);
        let groups = self.eq.e_out_current_len();
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
            set_inline_bytes(encoder, 5, &challenge);
            set_inline_bytes(encoder, 6, &params);
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
        let [constant, leading] = self.finish_command(command_buffer, final_in_a)?;
        self.dense_elements /= 2;
        self.dense_in_a = !self.dense_in_a;
        self.round += 1;
        Ok(self.eq.gruen_poly_deg_3(constant, leading, previous_claim))
    }

    fn validate_eq_shape(&self, pairs: usize) -> Result<(), MetalError> {
        let covered = self
            .eq
            .e_in_current_len()
            .checked_mul(self.eq.e_out_current_len())
            .ok_or(MetalError::InputTooLong(pairs))?;
        if covered != pairs {
            return Err(MetalError::BooleanityWeightShape {
                expected: pairs,
                covered,
            });
        }
        Ok(())
    }

    fn write_eq_tables(&self) -> Result<(), MetalError> {
        write_fields(
            &self.buffers.e_in,
            self.e_in_capacity,
            self.eq.e_in_current(),
        )?;
        write_fields(
            &self.buffers.e_out,
            self.e_out_capacity,
            self.eq.e_out_current(),
        )
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
    ) -> Result<[AkitaField; REDUCTION_COLUMNS], MetalError> {
        validate_completed_command(command_buffer)?;
        let buffer = if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the completed reduction initialized the first field in each
        // reduction column contiguously at the front of the final buffer.
        let values =
            unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), REDUCTION_COLUMNS) };
        self.context
            .validate_inputs("RAM Hamming message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    fn prefix_pipeline(&self, width: usize) -> Result<&ComputePipelineState, MetalError> {
        self.pipelines
            .prefix
            .iter()
            .find_map(|(candidate, pipeline)| (*candidate == width).then_some(pipeline))
            .ok_or(MetalError::InvalidRamRaState(
                "RAM Hamming prefix pipeline is missing",
            ))
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
            "RAM Hamming branch weights exceed width 32",
        ));
    }
    let complement = AkitaField::one() - challenge;
    let mut next = Vec::with_capacity(next_len);
    next.extend(weights.iter().map(|weight| *weight * complement));
    next.extend(weights.iter().map(|weight| *weight * challenge));
    Ok(next)
}

fn build_q_table(weights: &[AkitaField]) -> Result<Vec<AkitaField>, MetalError> {
    let width = weights.len();
    if ![2, 4, 8].contains(&width) {
        return Err(MetalError::InvalidRamRaState(
            "RAM Hamming quadratic table width is unsupported",
        ));
    }
    let patterns = 1usize
        .checked_shl(u32::try_from(2 * width).map_err(|_| MetalError::InputTooLong(width))?)
        .ok_or(MetalError::InputTooLong(width))?;
    let mut table = vec![AkitaField::zero(); REDUCTION_COLUMNS * patterns];
    let child_mask = (1usize << width) - 1;
    for pattern in 0..patterns {
        let low = bound_value(weights, pattern & child_mask);
        let high = bound_value(weights, pattern >> width);
        let delta = high - low;
        table[pattern] = low * (low - AkitaField::one());
        table[patterns + pattern] = delta * delta;
    }
    Ok(table)
}

fn build_value_table(weights: &[AkitaField]) -> Result<Vec<AkitaField>, MetalError> {
    if ![16, 32].contains(&weights.len()) {
        return Err(MetalError::InvalidRamRaState(
            "RAM Hamming byte table width is unsupported",
        ));
    }
    let segments = weights.len() / u8::BITS as usize;
    let mut table = vec![AkitaField::zero(); segments * 256];
    for segment in 0..segments {
        let segment_weights = &weights[segment * 8..(segment + 1) * 8];
        for mask in 0..256usize {
            table[segment * 256 + mask] = bound_value(segment_weights, mask);
        }
    }
    Ok(table)
}

fn bound_value(weights: &[AkitaField], mask: usize) -> AkitaField {
    weights
        .iter()
        .enumerate()
        .filter(|(bit, _)| mask & (1usize << bit) != 0)
        .fold(AkitaField::zero(), |sum, (_, weight)| sum + *weight)
}

fn write_fields(buffer: &Buffer, capacity: usize, values: &[AkitaField]) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::RamRaStorageLength {
            expected: capacity,
            got: values.len(),
        });
    }
    // SAFETY: the shared buffer owns `capacity` field slots and callers wait
    // for the preceding command before updating the active prefix.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

#[cfg(feature = "allocative")]
fn device_storage_bytes(buffers: &Buffers) -> usize {
    [
        &buffers.access_bits,
        &buffers.value_table,
        &buffers.q_table,
        &buffers.e_in,
        &buffers.e_out,
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

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<PackParams>() == 16);
const _: () = assert!(size_of::<PrefixParams>() == 16);
const _: () = assert!(size_of::<DenseParams>() == 16);
