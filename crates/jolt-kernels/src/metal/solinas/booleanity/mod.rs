use std::{mem::size_of, slice, sync::Arc};

use super::{
    buffer_from_slice, encode_column_reductions, set_inline_bytes, validate_completed_command,
    Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};

const SIMD_WIDTH: usize = 32;
const MAX_MATERIALIZE_WIDTH: usize = 32;
const MESSAGE_LANES: usize = 2;
const LAZY_PIPELINE: &str = "solinas_booleanity_lazy_message";
const DOUBLE_PIPELINE: &str = "solinas_booleanity_double_branches";
const DENSE_PIPELINE: &str = "solinas_booleanity_dense_transition";
const REDUCE_PIPELINE: &str = "solinas_booleanity_reduce";

const PACKED_PC_MASK: u64 = (1 << 56) - 1;
const PACKED_BYTECODE_CHUNK_RANK_LOW_SHIFT: u32 = 56;
const PACKED_BYTECODE_CHUNK_RANK_LOW_MASK: u64 = 0x7f << PACKED_BYTECODE_CHUNK_RANK_LOW_SHIFT;
const PACKED_INC_SIGN_SHIFT: u32 = 63;
const INSTRUCTION_SOURCE_RAM_MASK: u64 = u32::MAX as u64;
const INSTRUCTION_SOURCE_PC_SHIFT: u32 = 32;
const INSTRUCTION_SOURCE_PC_MASK: u64 = (1 << 14) - 1;
const INSTRUCTION_SOURCE_RD_SHIFT: u32 = 46;
const INSTRUCTION_SOURCE_RANK_SHIFT: u32 = 54;
const INSTRUCTION_SOURCE_FUSED_SIGN_SHIFT: u32 = 61;
const INSTRUCTION_SOURCE_RD_SIGN_SHIFT: u32 = 62;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityRow {
    lookup_lo: u64,
    lookup_hi: u64,
    ram_address_plus_one: u64,
    fused_inc_magnitude: u64,
    packed_pc_and_flags: u64,
}

pub(crate) const BOOLEANITY_SOURCE_WORDS: usize = 4;
pub(crate) const BOOLEANITY_SOURCE_ROW_BYTES: usize = BOOLEANITY_SOURCE_WORDS * size_of::<u64>();

struct BooleanityRowsInner {
    buffer: Buffer,
    len: usize,
    device_registry_id: u64,
}

#[derive(Clone)]
pub struct BooleanityRows(Arc<BooleanityRowsInner>);

impl BooleanityRows {
    pub(super) fn from_initialized_buffer(
        buffer: Buffer,
        len: usize,
        expected_device_registry_id: u64,
    ) -> Result<Self, MetalError> {
        if len == 0 {
            return Err(MetalError::EmptyInput);
        }
        let expected_bytes = instruction_source_byte_length(len)?;
        if buffer.length() != expected_bytes {
            return Err(MetalError::BooleanityStorageLength {
                name: "resident rows",
                expected: expected_bytes as usize,
                got: buffer.length() as usize,
            });
        }
        let got = buffer.device().registry_id();
        if got != expected_device_registry_id {
            return Err(MetalError::BooleanityRowsDevice {
                expected: expected_device_registry_id,
                got,
            });
        }
        Ok(Self(Arc::new(BooleanityRowsInner {
            buffer,
            len,
            device_registry_id: expected_device_registry_id,
        })))
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.0.buffer
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn device_registry_id(&self) -> u64 {
        self.0.device_registry_id
    }

    pub fn allocation_identity(&self) -> usize {
        self.0.buffer.as_ptr() as usize
    }

    #[cfg(test)]
    pub(crate) fn shares_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for BooleanityRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(mut shared) = visitor.enter_shared(
            allocative::Key::new("rows"),
            size_of::<*const BooleanityRowsInner>(),
            Arc::as_ptr(&self.0).cast(),
        ) {
            shared.visit_simple(
                allocative::Key::new("ArcInner"),
                2 * size_of::<usize>() + size_of::<BooleanityRowsInner>(),
            );
            shared.visit_simple(
                allocative::Key::new("device_rows"),
                self.len() * BOOLEANITY_SOURCE_ROW_BYTES,
            );
            shared.exit();
        }
        visitor.exit();
    }
}

impl BooleanityRow {
    pub fn new(
        lookup_index: u128,
        mapped_pc: Option<u64>,
        remapped_ram_address: Option<u64>,
        fused_inc: i128,
    ) -> Result<Self, MetalError> {
        let pc_plus_one = mapped_pc
            .map(|pc| pc.checked_add(1).ok_or(MetalError::InvalidBooleanityRow))
            .transpose()?
            .unwrap_or(0);
        let ram_address_plus_one = remapped_ram_address
            .map(|address| {
                address
                    .checked_add(1)
                    .ok_or(MetalError::InvalidBooleanityRow)
            })
            .transpose()?
            .unwrap_or(0);
        if pc_plus_one > PACKED_PC_MASK || fused_inc.unsigned_abs() > u64::MAX as u128 {
            return Err(MetalError::InvalidBooleanityRow);
        }
        Ok(Self {
            lookup_lo: lookup_index as u64,
            lookup_hi: (lookup_index >> 64) as u64,
            ram_address_plus_one,
            fused_inc_magnitude: fused_inc.unsigned_abs() as u64,
            packed_pc_and_flags: pc_plus_one | (u64::from(fused_inc < 0) << PACKED_INC_SIGN_SHIFT),
        })
    }

    pub const fn from_words(words: [u64; 5]) -> Self {
        Self {
            lookup_lo: words[0],
            lookup_hi: words[1],
            ram_address_plus_one: words[2],
            fused_inc_magnitude: words[3],
            packed_pc_and_flags: words[4],
        }
    }

    pub const fn words(self) -> [u64; 5] {
        [
            self.lookup_lo,
            self.lookup_hi,
            self.ram_address_plus_one,
            self.fused_inc_magnitude,
            self.packed_pc_and_flags,
        ]
    }

    pub(crate) fn instruction_source_words(
        self,
        rd_write: Option<(u8, u64, u64)>,
    ) -> Result<[u64; 4], MetalError> {
        let [fused_inc_magnitude, packed_metadata] = self.instruction_source_aux_words(rd_write)?;
        Ok([
            self.lookup_lo,
            self.lookup_hi,
            fused_inc_magnitude,
            packed_metadata,
        ])
    }

    pub(crate) fn instruction_source_aux_words(
        self,
        rd_write: Option<(u8, u64, u64)>,
    ) -> Result<[u64; 2], MetalError> {
        let pc_plus_one = self.packed_pc_and_flags & PACKED_PC_MASK;
        if pc_plus_one > INSTRUCTION_SOURCE_PC_MASK
            || self.ram_address_plus_one > INSTRUCTION_SOURCE_RAM_MASK
        {
            return Err(MetalError::InvalidBooleanityRow);
        }
        let fused_negative = self.packed_pc_and_flags >> PACKED_INC_SIGN_SHIFT;
        let (rd_plus_one, rd_negative) = match rd_write {
            Some((register, pre_value, post_value)) if register < 128 => {
                let increment = i128::from(post_value) - i128::from(pre_value);
                if increment.unsigned_abs() != u128::from(self.fused_inc_magnitude)
                    || u64::from(increment < 0) != fused_negative
                {
                    return Err(MetalError::InvalidBooleanityRow);
                }
                (u64::from(register) + 1, increment < 0)
            }
            Some(_) => return Err(MetalError::InvalidBooleanityRow),
            None => (0, false),
        };
        let rank = (self.packed_pc_and_flags & PACKED_BYTECODE_CHUNK_RANK_LOW_MASK)
            >> PACKED_BYTECODE_CHUNK_RANK_LOW_SHIFT;
        let metadata = self.ram_address_plus_one
            | (pc_plus_one << INSTRUCTION_SOURCE_PC_SHIFT)
            | (rd_plus_one << INSTRUCTION_SOURCE_RD_SHIFT)
            | (rank << INSTRUCTION_SOURCE_RANK_SHIFT)
            | (fused_negative << INSTRUCTION_SOURCE_FUSED_SIGN_SHIFT)
            | (u64::from(rd_negative) << INSTRUCTION_SOURCE_RD_SIGN_SHIFT);
        Ok([self.fused_inc_magnitude, metadata])
    }

    #[cfg(test)]
    pub(crate) const fn from_instruction_source_words(words: [u64; 4]) -> Self {
        let metadata = words[3];
        let pc_plus_one = (metadata >> INSTRUCTION_SOURCE_PC_SHIFT) & INSTRUCTION_SOURCE_PC_MASK;
        let rank = (metadata >> INSTRUCTION_SOURCE_RANK_SHIFT) & 0x7f;
        let fused_negative = (metadata >> INSTRUCTION_SOURCE_FUSED_SIGN_SHIFT) & 1;
        Self {
            lookup_lo: words[0],
            lookup_hi: words[1],
            ram_address_plus_one: metadata & INSTRUCTION_SOURCE_RAM_MASK,
            fused_inc_magnitude: words[2],
            packed_pc_and_flags: pc_plus_one
                | (rank << PACKED_BYTECODE_CHUNK_RANK_LOW_SHIFT)
                | (fused_negative << PACKED_INC_SIGN_SHIFT),
        }
    }

    #[cfg(test)]
    pub(crate) const fn mapped_pc(self) -> Option<usize> {
        let plus_one = self.packed_pc_and_flags & PACKED_PC_MASK;
        if plus_one == 0 {
            None
        } else {
            Some((plus_one - 1) as usize)
        }
    }

    pub(crate) const fn with_bytecode_chunk_rank_low7(mut self, rank: u8) -> Self {
        self.packed_pc_and_flags = (self.packed_pc_and_flags
            & !PACKED_BYTECODE_CHUNK_RANK_LOW_MASK)
            | (((rank & 0x7f) as u64) << PACKED_BYTECODE_CHUNK_RANK_LOW_SHIFT);
        self
    }

    #[cfg(test)]
    pub(crate) const fn bytecode_chunk_rank_low7(self) -> u8 {
        ((self.packed_pc_and_flags & PACKED_BYTECODE_CHUNK_RANK_LOW_MASK)
            >> PACKED_BYTECODE_CHUNK_RANK_LOW_SHIFT) as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BooleanitySelector {
    Lookup { shift: u32 },
    Bytecode { shift: u32 },
    Ram { shift: u32 },
    FusedInc { shift: u32 },
    FusedIncMsb,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanitySequenceConfig {
    pub threads_per_threadgroup: Option<usize>,
    pub dense_threads_per_threadgroup: Option<usize>,
    pub materialize_width: usize,
}

impl Default for BooleanitySequenceConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(256),
            dense_threads_per_threadgroup: Some(128),
            materialize_width: 8,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SelectorAbi {
    kind: u32,
    shift: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    polys: u32,
    k: u32,
    branch_width: u32,
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    materialize: u32,
    inc_bias: u64,
    chunk_bits: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BranchParams {
    polys: u32,
    k: u32,
    branch_width: u32,
    reserved: u32,
}

struct Buffers {
    rows: BooleanityRows,
    selectors: Buffer,
    rho: Buffer,
    initial_constant: Buffer,
    branches_a: Buffer,
    branches_b: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct BooleanitySequence {
    context: SolinasMetal,
    lazy_pipeline: ComputePipelineState,
    double_pipeline: ComputePipelineState,
    dense_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    polys: usize,
    k: usize,
    chunk_bits: usize,
    inc_bias: u64,
    threads_per_threadgroup: usize,
    dense_threads_per_threadgroup: usize,
    double_threads_per_threadgroup: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    branch_width: usize,
    materialize_width: usize,
    branches_in_a: bool,
    dense: bool,
    dense_elements: usize,
    dense_source_in_a: bool,
    rho_values: Vec<AkitaField>,
}

impl SolinasMetal {
    #[expect(
        clippy::too_many_arguments,
        reason = "the sequence has row, selector, relation, and split-equality inputs"
    )]
    pub fn prepare_booleanity_sequence(
        &self,
        rows: &[BooleanityRow],
        selectors: &[BooleanitySelector],
        base_tables: &[AkitaField],
        rho: &[AkitaField],
        k: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: BooleanitySequenceConfig,
    ) -> Result<BooleanitySequence, MetalError> {
        let resident_rows = self.prepare_booleanity_rows(rows)?;
        self.prepare_booleanity_sequence_with_rows(
            resident_rows,
            selectors,
            base_tables,
            rho,
            k,
            e_in_capacity,
            e_out_capacity,
            config,
        )
    }

    pub fn prepare_booleanity_rows(
        &self,
        rows: &[BooleanityRow],
    ) -> Result<BooleanityRows, MetalError> {
        if rows.is_empty() {
            return Err(MetalError::EmptyInput);
        }
        let len = rows.len();
        let bytes = instruction_source_byte_length(len)?;
        self.validate_buffer_length(bytes)?;
        self.validate_additional_working_set(bytes)?;
        let mut column_major = vec![0u64; BOOLEANITY_SOURCE_WORDS * len];
        for (index, row) in rows.iter().copied().enumerate() {
            for (word, value) in row.instruction_source_words(None)?.into_iter().enumerate() {
                column_major[word * len + index] = value;
            }
        }
        Ok(BooleanityRows(Arc::new(BooleanityRowsInner {
            buffer: buffer_from_slice(&self.device, &column_major),
            len,
            device_registry_id: self.device.registry_id(),
        })))
    }

    pub(crate) fn validate_booleanity_rows(&self, rows: &BooleanityRows) -> Result<(), MetalError> {
        let expected = self.device.registry_id();
        let got = rows.device_registry_id();
        if got != expected {
            return Err(MetalError::BooleanityRowsDevice { expected, got });
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the sequence has resident rows, relation tables, and split-equality inputs"
    )]
    pub(crate) fn prepare_booleanity_sequence_with_rows(
        &self,
        resident_rows: BooleanityRows,
        selectors: &[BooleanitySelector],
        base_tables: &[AkitaField],
        rho: &[AkitaField],
        k: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: BooleanitySequenceConfig,
    ) -> Result<BooleanitySequence, MetalError> {
        self.validate_booleanity_rows(&resident_rows)?;
        let rows_len = resident_rows.len();
        if rows_len < 4 || !rows_len.is_power_of_two() {
            return Err(MetalError::InvalidBooleanityRows(rows_len));
        }
        if selectors.is_empty() || selectors.len() != rho.len() {
            return Err(MetalError::BooleanityStorageLength {
                name: "rho",
                expected: selectors.len(),
                got: rho.len(),
            });
        }
        if !(2..=256).contains(&k) || !k.is_power_of_two() {
            return Err(MetalError::InvalidBooleanityK(k));
        }
        if !(1..=MAX_MATERIALIZE_WIDTH).contains(&config.materialize_width)
            || !config.materialize_width.is_power_of_two()
            || rows_len < 2 * config.materialize_width
        {
            return Err(MetalError::InvalidBooleanityMaterializeWidth(
                config.materialize_width,
            ));
        }
        let chunk_bits = k.ilog2() as usize;
        let expected_tables = selectors
            .len()
            .checked_mul(k)
            .ok_or(MetalError::InputTooLong(selectors.len()))?;
        if base_tables.len() != expected_tables {
            return Err(MetalError::BooleanityStorageLength {
                name: "base tables",
                expected: expected_tables,
                got: base_tables.len(),
            });
        }
        if e_in_capacity == 0
            || e_out_capacity == 0
            || e_in_capacity
                .checked_mul(e_out_capacity)
                .ok_or(MetalError::InputTooLong(rows_len))?
                != rows_len / 2
        {
            return Err(MetalError::BooleanityWeightShape {
                expected: rows_len / 2,
                covered: e_in_capacity.saturating_mul(e_out_capacity),
            });
        }
        let selector_abi = selectors
            .iter()
            .copied()
            .map(|selector| selector_abi(selector, chunk_bits))
            .collect::<Result<Vec<_>, _>>()?;

        let lazy_pipeline = self.compile_named_pipeline(LAZY_PIPELINE)?;
        let double_pipeline = self.compile_named_pipeline(DOUBLE_PIPELINE)?;
        let dense_pipeline = self.compile_named_pipeline(DENSE_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let lazy_limits = Self::limits(&lazy_pipeline);
        let double_limits = Self::limits(&double_pipeline);
        let dense_limits = Self::limits(&dense_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (LAZY_PIPELINE, lazy_limits),
            (DOUBLE_PIPELINE, double_limits),
            (DENSE_PIPELINE, dense_limits),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedBooleanityExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, lazy_limits)?;
        let dense_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config
                .dense_threads_per_threadgroup
                .or(config.threads_per_threadgroup),
            dense_limits,
        )?;
        let double_threads_per_threadgroup =
            Self::resolve_threadgroup_width(Some(256), double_limits)?;

        let branch_capacity = selectors
            .len()
            .checked_mul(config.materialize_width)
            .and_then(|value| value.checked_mul(k))
            .ok_or(MetalError::InputTooLong(selectors.len()))?;
        let initial_stride = k + 1;
        let initial_constant_elements = selectors
            .len()
            .checked_mul(initial_stride)
            .ok_or(MetalError::InputTooLong(selectors.len()))?;
        let initial_dense_elements = rows_len / config.materialize_width;
        let dense_a_elements = selectors
            .len()
            .checked_mul(initial_dense_elements)
            .ok_or(MetalError::InputTooLong(initial_dense_elements))?;
        let dense_b_elements = dense_a_elements / 2;
        let partial_elements = MESSAGE_LANES
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(e_out_capacity))?;
        for bytes in [
            byte_length::<BooleanityRow>(rows_len)?,
            byte_length::<SelectorAbi>(selector_abi.len())?,
            byte_length::<Fp128>(rho.len())?,
            byte_length::<Fp128>(initial_constant_elements)?,
            byte_length::<Fp128>(branch_capacity)?,
            byte_length::<Fp128>(dense_a_elements)?,
            byte_length::<Fp128>(dense_b_elements)?,
            byte_length::<Fp128>(e_in_capacity)?,
            byte_length::<Fp128>(e_out_capacity)?,
            byte_length::<Fp128>(partial_elements)?,
        ] {
            self.validate_buffer_length(bytes)?;
        }

        let selectors_buffer = self.device.new_buffer_with_data(
            selector_abi.as_ptr().cast(),
            byte_length::<SelectorAbi>(selector_abi.len())?,
            MTLResourceOptions::StorageModeShared,
        );
        let rho_buffer = self.new_booleanity_buffer(rho.len())?;
        write_fields(&rho_buffer, rho.len(), rho)?;
        let initial_constant = self.new_booleanity_buffer(initial_constant_elements)?;
        write_initial_constant_table(&initial_constant, base_tables, rho, k)?;
        let branches_a = self.new_booleanity_buffer(branch_capacity)?;
        write_fields(&branches_a, branch_capacity, base_tables)?;

        Ok(BooleanitySequence {
            context: self.clone(),
            lazy_pipeline,
            double_pipeline,
            dense_pipeline,
            reduction_pipeline,
            reduction_limits,
            buffers: Buffers {
                rows: resident_rows,
                selectors: selectors_buffer,
                rho: rho_buffer,
                initial_constant,
                branches_a,
                branches_b: self.new_booleanity_buffer(branch_capacity)?,
                dense_a: self.new_booleanity_buffer(dense_a_elements)?,
                dense_b: self.new_booleanity_buffer(dense_b_elements)?,
                e_in: self.new_booleanity_buffer(e_in_capacity)?,
                e_out: self.new_booleanity_buffer(e_out_capacity)?,
                partial_a: self.new_booleanity_buffer(partial_elements)?,
                partial_b: self.new_booleanity_buffer(partial_elements)?,
            },
            rows: rows_len,
            polys: selectors.len(),
            k,
            chunk_bits,
            inc_bias: balanced_bias(chunk_bits),
            threads_per_threadgroup,
            dense_threads_per_threadgroup,
            double_threads_per_threadgroup,
            e_in_capacity,
            e_out_capacity,
            branch_width: 1,
            materialize_width: config.materialize_width,
            branches_in_a: true,
            dense: false,
            dense_elements: 0,
            dense_source_in_a: true,
            rho_values: rho.to_vec(),
        })
    }

    fn new_booleanity_buffer(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = byte_length::<Fp128>(elements)?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl BooleanitySequence {
    pub fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_LANES], MetalError> {
        if self.dense {
            return Err(MetalError::InvalidBooleanityState(
                "a dense sequence has no unbound message entry",
            ));
        }
        self.execute_lazy(None, e_in, e_out)
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_LANES], MetalError> {
        if self.dense {
            self.execute_dense(challenge, e_in, e_out)
        } else {
            self.execute_lazy(Some(challenge), e_in, e_out)
        }
    }

    pub fn reset(&mut self, base_tables: &[AkitaField]) -> Result<(), MetalError> {
        let expected = self.polys * self.k;
        if base_tables.len() != expected {
            return Err(MetalError::BooleanityStorageLength {
                name: "base tables",
                expected,
                got: base_tables.len(),
            });
        }
        let branch_capacity = self.polys * self.materialize_width * self.k;
        write_fields(&self.buffers.branches_a, branch_capacity, base_tables)?;
        write_initial_constant_table(
            &self.buffers.initial_constant,
            base_tables,
            &self.rho_values,
            self.k,
        )?;
        self.branch_width = 1;
        self.branches_in_a = true;
        self.dense = false;
        self.dense_elements = 0;
        self.dense_source_in_a = true;
        Ok(())
    }

    pub fn read_current_tables(&self, output: &mut [AkitaField]) -> Result<(), MetalError> {
        if !self.dense {
            return Err(MetalError::InvalidBooleanityState(
                "lazy tables cannot be read as dense tables",
            ));
        }
        let elements = self.polys * self.dense_elements;
        if output.len() != elements {
            return Err(MetalError::BooleanityStorageLength {
                name: "dense output",
                expected: elements,
                got: output.len(),
            });
        }
        // SAFETY: the resident dense buffer has `elements` initialized slots,
        // and every preceding transition completed synchronously.
        let values = unsafe {
            slice::from_raw_parts(
                self.dense_source_buffer().contents().cast::<Fp128>(),
                elements,
            )
        };
        self.context
            .validate_inputs("booleanity dense output", values)?;
        for (output, value) in output.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(())
    }

    pub const fn current_elements(&self) -> usize {
        if self.dense {
            self.dense_elements
        } else {
            self.rows / self.branch_width
        }
    }

    copy_field_getters! { pub, {
        branch_width: usize,
        is_dense => dense: bool,
    }}

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    fn execute_lazy(
        &mut self,
        challenge: Option<AkitaField>,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_LANES], MetalError> {
        let next_width = if challenge.is_some() {
            self.branch_width * 2
        } else {
            self.branch_width
        };
        if next_width > self.materialize_width {
            return Err(MetalError::InvalidBooleanityState(
                "lazy branch width exceeds the materialization point",
            ));
        }
        let source_elements = self.rows / next_width;
        self.validate_weights(source_elements / 2, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let materialize = next_width == self.materialize_width;
        let params = self.params(
            source_elements,
            next_width,
            e_in.len(),
            e_out.len(),
            materialize,
        )?;

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            let mut message_branches_in_a = self.branches_in_a;
            if let Some(challenge) = challenge {
                let branch_params = BranchParams {
                    polys: self.polys as u32,
                    k: self.k as u32,
                    branch_width: self.branch_width as u32,
                    reserved: 0,
                };
                encoder.set_compute_pipeline_state(&self.double_pipeline);
                encoder.set_buffer(0, Some(self.branch_source_buffer()), 0);
                encoder.set_buffer(1, Some(self.branch_destination_buffer()), 0);
                set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(encoder, 3, &branch_params);
                let elements = self.polys * self.branch_width * self.k;
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: elements.div_ceil(self.double_threads_per_threadgroup) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.double_threads_per_threadgroup as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                message_branches_in_a = !message_branches_in_a;
            }

            encoder.set_compute_pipeline_state(&self.lazy_pipeline);
            encoder.set_buffer(0, Some(self.buffers.rows.buffer()), 0);
            encoder.set_buffer(1, Some(&self.buffers.selectors), 0);
            encoder.set_buffer(2, Some(self.branch_buffer(message_branches_in_a)), 0);
            encoder.set_buffer(3, Some(&self.buffers.rho), 0);
            encoder.set_buffer(4, Some(&self.buffers.dense_a), 0);
            encoder.set_buffer(5, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(6, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(7, Some(&self.buffers.partial_a), 0);
            encoder.set_buffer(8, Some(&self.buffers.initial_constant), 0);
            set_inline_bytes(encoder, 9, &params);
            Self::encode_message_dispatch(encoder, e_out.len(), self.threads_per_threadgroup);
            let final_in_a = encode_column_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                MESSAGE_LANES,
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
            self.dense_elements = source_elements;
            self.dense_source_in_a = true;
        }
        Ok(message)
    }

    fn execute_dense(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; MESSAGE_LANES], MetalError> {
        if self.dense_elements < 4 {
            return Err(MetalError::InvalidBooleanityState(
                "dense transition needs at least four elements",
            ));
        }
        self.validate_weights(self.dense_elements / 4, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let params = self.params(
            self.dense_elements,
            self.branch_width,
            e_in.len(),
            e_out.len(),
            false,
        )?;
        let challenge = Fp128::from_jolt_field(&challenge);

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.dense_pipeline);
            encoder.set_buffer(0, Some(self.dense_source_buffer()), 0);
            encoder.set_buffer(1, Some(self.dense_destination_buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.rho), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 6, &challenge);
            set_inline_bytes(encoder, 7, &params);
            Self::encode_message_dispatch(encoder, e_out.len(), self.dense_threads_per_threadgroup);
            let final_in_a = encode_column_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                MESSAGE_LANES,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;
        let message = self.finish_command(command_buffer, final_in_a)?;
        self.dense_elements /= 2;
        self.dense_source_in_a = !self.dense_source_in_a;
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
            return Err(MetalError::BooleanityWeightShape { expected, covered });
        }
        Ok(())
    }

    fn write_weights(&self, e_in: &[AkitaField], e_out: &[AkitaField]) -> Result<(), MetalError> {
        write_fields(&self.buffers.e_in, self.e_in_capacity, e_in)?;
        write_fields(&self.buffers.e_out, self.e_out_capacity, e_out)
    }

    fn params(
        &self,
        source_elements: usize,
        branch_width: usize,
        e_in_length: usize,
        e_out_length: usize,
        materialize: bool,
    ) -> Result<Params, MetalError> {
        Ok(Params {
            rows: u32::try_from(self.rows).map_err(|_| MetalError::InputTooLong(self.rows))?,
            polys: u32::try_from(self.polys).map_err(|_| MetalError::InputTooLong(self.polys))?,
            k: self.k as u32,
            branch_width: branch_width as u32,
            source_elements: u32::try_from(source_elements)
                .map_err(|_| MetalError::InputTooLong(source_elements))?,
            e_in_length: u32::try_from(e_in_length)
                .map_err(|_| MetalError::InputTooLong(e_in_length))?,
            e_out_length: u32::try_from(e_out_length)
                .map_err(|_| MetalError::InputTooLong(e_out_length))?,
            materialize: u32::from(materialize),
            inc_bias: self.inc_bias,
            chunk_bits: self.chunk_bits as u32,
            reserved: 0,
        })
    }

    fn encode_message_dispatch(
        encoder: &metal::ComputeCommandEncoderRef,
        groups: usize,
        threads_per_threadgroup: usize,
    ) {
        let simdgroups = threads_per_threadgroup / SIMD_WIDTH;
        encoder.set_threadgroup_memory_length(
            0,
            (MESSAGE_LANES * simdgroups * size_of::<Fp128>()) as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn finish_command(
        &mut self,
        command_buffer: &metal::CommandBufferRef,
        final_in_a: bool,
    ) -> Result<[AkitaField; MESSAGE_LANES], MetalError> {
        validate_completed_command(command_buffer)?;
        let buffer = if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the completed reduction leaves exactly two message fields at
        // the front of the selected shared buffer.
        let values =
            unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), MESSAGE_LANES) };
        self.context.validate_inputs("booleanity message", values)?;
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
        if self.dense_source_in_a {
            &self.buffers.dense_a
        } else {
            &self.buffers.dense_b
        }
    }

    fn dense_destination_buffer(&self) -> &Buffer {
        if self.dense_source_in_a {
            &self.buffers.dense_b
        } else {
            &self.buffers.dense_a
        }
    }
}

pub(super) fn selector_abi(
    selector: BooleanitySelector,
    chunk_bits: usize,
) -> Result<SelectorAbi, MetalError> {
    let (kind, shift, limit) = match selector {
        BooleanitySelector::Lookup { shift } => (0, shift, 128),
        BooleanitySelector::Bytecode { shift } => (1, shift, 56),
        BooleanitySelector::Ram { shift } => (2, shift, 64),
        BooleanitySelector::FusedInc { shift } => (3, shift, 64),
        BooleanitySelector::FusedIncMsb => (4, 0, chunk_bits),
    };
    if shift as usize + chunk_bits > limit
        || (kind == 0 && shift < 64 && shift as usize + chunk_bits > 64)
    {
        return Err(MetalError::InvalidBooleanitySelector);
    }
    Ok(SelectorAbi { kind, shift })
}

pub(super) fn balanced_bias(chunk_bits: usize) -> u64 {
    let radix = 1u128 << chunk_bits;
    let bias = (radix / 2) * (u128::from(u64::MAX) / (radix - 1));
    bias as u64
}

pub(super) fn write_fields(
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::BooleanityStorageLength {
            name: "field buffer",
            expected: capacity,
            got: values.len(),
        });
    }
    // SAFETY: callers allocate the shared Metal buffer for `capacity` fields
    // and no GPU command uses it while host values are copied in.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn write_initial_constant_table(
    constant_buffer: &Buffer,
    base_tables: &[AkitaField],
    rho: &[AkitaField],
    k: usize,
) -> Result<(), MetalError> {
    let expected = rho
        .len()
        .checked_mul(k)
        .ok_or(MetalError::InputTooLong(rho.len()))?;
    if base_tables.len() != expected {
        return Err(MetalError::BooleanityStorageLength {
            name: "initial constant table",
            expected,
            got: base_tables.len(),
        });
    }
    let stride = k + 1;
    let constant_elements = rho
        .len()
        .checked_mul(stride)
        .ok_or(MetalError::InputTooLong(rho.len()))?;
    // SAFETY: the fresh shared buffer has the checked capacity below and no
    // command buffer uses it while the host refreshes a sequence.
    let constants = unsafe {
        slice::from_raw_parts_mut(
            constant_buffer.contents().cast::<Fp128>(),
            constant_elements,
        )
    };
    for (poly, (table, rho)) in base_tables.chunks_exact(k).zip(rho).enumerate() {
        for first in 0..stride {
            let h_0 = table.get(first).copied().unwrap_or_else(AkitaField::zero);
            constants[poly * stride + first] = Fp128::from_jolt_field(&(h_0 * (h_0 - *rho)));
        }
    }
    Ok(())
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

fn instruction_source_byte_length(rows: usize) -> Result<u64, MetalError> {
    byte_length::<[u64; BOOLEANITY_SOURCE_WORDS]>(rows)
}

const _: () = assert!(size_of::<BooleanityRow>() == 40);
const _: () = assert!(BOOLEANITY_SOURCE_ROW_BYTES == 32);
const _: () = assert!(size_of::<SelectorAbi>() == 8);
const _: () = assert!(size_of::<Params>() == 48);
const _: () = assert!(size_of::<BranchParams>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal resident-row validation setup")]
mod tests {
    use super::*;

    #[test]
    fn resident_rows_reject_a_different_device_registry() {
        let context = SolinasMetal::for_akita().unwrap();
        let rows = context
            .prepare_booleanity_rows(&[BooleanityRow::default(); 4])
            .unwrap();
        let expected = rows.device_registry_id();
        let got = expected.wrapping_add(1);
        let wrong_device_rows = BooleanityRows(Arc::new(BooleanityRowsInner {
            buffer: rows.buffer().clone(),
            len: rows.len(),
            device_registry_id: got,
        }));

        assert!(matches!(
            context.prepare_booleanity_sequence_with_rows(
                wrong_device_rows,
                &[],
                &[],
                &[],
                2,
                1,
                2,
                BooleanitySequenceConfig::default(),
            ),
            Err(MetalError::BooleanityRowsDevice {
                expected: error_expected,
                got: error_got,
            }) if error_expected == expected && error_got == got
        ));
    }

    #[test]
    fn bytecode_chunk_rank_preserves_pc_and_increment_sign() {
        let positive = BooleanityRow::new(7, Some((1u64 << 55) + 9), Some(3), 17).unwrap();
        let negative = BooleanityRow::new(7, Some(41), Some(3), -17).unwrap();

        for (row, rank) in [(positive, 127), (negative, 95)] {
            let encoded = row.with_bytecode_chunk_rank_low7(rank);
            assert_eq!(encoded.mapped_pc(), row.mapped_pc());
            assert_eq!(encoded.bytecode_chunk_rank_low7(), rank);
            assert_eq!(
                encoded.words()[4] >> PACKED_INC_SIGN_SHIFT,
                row.words()[4] >> 63
            );
        }
    }

    #[test]
    fn instruction_source_layout_round_trips_logical_row_and_derives_register_delta() {
        let row = BooleanityRow::new(7, Some(8191), Some(u32::MAX as u64 - 1), -17)
            .unwrap()
            .with_bytecode_chunk_rank_low7(0x7f);
        let words = row.instruction_source_words(Some((127, 17, 0))).unwrap();

        assert_eq!(BooleanityRow::from_instruction_source_words(words), row);
        assert_eq!(words.len(), 4);
        assert_eq!(words[2], 17);
        assert_eq!((words[3] >> INSTRUCTION_SOURCE_RD_SHIFT) & 0xff, 128);
        assert_eq!((words[3] >> INSTRUCTION_SOURCE_RD_SIGN_SHIFT) & 1, 1);
        assert!(matches!(
            row.instruction_source_words(Some((127, 18, 0))),
            Err(MetalError::InvalidBooleanityRow)
        ));
        assert!(matches!(
            row.instruction_source_words(Some((127, 0, 17))),
            Err(MetalError::InvalidBooleanityRow)
        ));
        assert!(matches!(
            row.instruction_source_words(Some((128, 0, 1))),
            Err(MetalError::InvalidBooleanityRow)
        ));

        let store = BooleanityRow::new(11, Some(7), Some(9), u64::MAX as i128).unwrap();
        let store_words = store.instruction_source_words(None).unwrap();
        assert_eq!(
            BooleanityRow::from_instruction_source_words(store_words),
            store
        );
        assert_eq!(store_words[2], u64::MAX);
        assert_eq!((store_words[3] >> INSTRUCTION_SOURCE_RD_SHIFT) & 0xff, 0);

        let extreme = BooleanityRow::new(13, None, None, -(u64::MAX as i128)).unwrap();
        let extreme_words = extreme
            .instruction_source_words(Some((0, u64::MAX, 0)))
            .unwrap();
        assert_eq!(
            BooleanityRow::from_instruction_source_words(extreme_words),
            extreme
        );
        assert_eq!(extreme_words[2], u64::MAX);
    }
}
