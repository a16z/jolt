use std::{
    cell::Cell,
    mem::size_of,
    slice,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

#[cfg(feature = "allocative")]
use allocative::{Allocative, Key, Visitor};
use jolt_field::signed::{S192, S256, S64};
use jolt_field::Zero as _;
use jolt_field::{Accumulator as _, Prime128OffsetA7F7 as AkitaField, WithAccumulator};
use jolt_witness::witnesses::SpartanOuterRow;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBufferRef,
    ComputePipelineState, MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::spartan_shift::SpartanShiftResidentRows;
use super::{
    buffer_from_slice, completed_command_gpu_time,
    residency::{self, ResidencyPrefetch},
    set_inline_bytes, Fp128, InstructionInputRow, InstructionInputRows, MetalError, SolinasMetal,
};

pub const SPARTAN_OUTER_EXTENDED_NODES: usize = 9;
/// Pipeline depth of [`SpartanOuterUniskipInvocation::execute_filling`].
const FILL_CHUNKS: u32 = 32;
const ROW_WORDS: usize = 20;
const RAW_ROW_WORDS: usize = 4;
const SIMD_WIDTH: usize = 32;
const BLOCKS_PIPELINE: &str = "solinas_spartan_outer_uniskip_blocks";
const REDUCE_PIPELINE: &str = "solinas_spartan_outer_uniskip_reduce";
static NEXT_STAGE1_ROWS_GENERATION: AtomicU64 = AtomicU64::new(1);

const EXTENSION_COEFFICIENTS: [[i64; 10]; SPARTAN_OUTER_EXTENDED_NODES] = [
    [
        2002, -15015, 51480, -105_105, 140_140, -126_126, 76440, -30030, 6930, -715,
    ],
    [
        715, -5148, 17160, -34320, 45045, -40040, 24024, -9360, 2145, -220,
    ],
    [
        220, -1485, 4752, -9240, 11880, -10395, 6160, -2376, 540, -55,
    ],
    [55, -330, 990, -1848, 2310, -1980, 1155, -440, 99, -10],
    [10, -45, 120, -210, 252, -210, 120, -45, 10, -1],
    [-1, 10, -45, 120, -210, 252, -210, 120, -45, 10],
    [-10, 99, -440, 1155, -1980, 2310, -1848, 990, -330, 55],
    [
        -55, 540, -2376, 6160, -10395, 11880, -9240, 4752, -1485, 220,
    ],
    [
        -220, 2145, -9360, 24024, -40040, 45045, -34320, 17160, -5148, 715,
    ],
];

const FLAG_LOAD: u32 = 0;
const FLAG_STORE: u32 = 1;
const FLAG_ADD: u32 = 2;
const FLAG_SUB: u32 = 3;
const FLAG_MUL: u32 = 4;
const FLAG_JUMP: u32 = 5;
const FLAG_SHOULD_BRANCH: u32 = 6;
const FLAG_ASSERT: u32 = 7;
const FLAG_SHOULD_JUMP: u32 = 8;
pub(crate) const FLAG_VIRTUAL: u32 = 9;
const FLAG_IS_LAST: u32 = 10;
const FLAG_NEXT_VIRTUAL: u32 = 11;
const FLAG_NEXT_FIRST: u32 = 12;
const FLAG_ADVICE: u32 = 13;
const FLAG_WRITE_LOOKUP: u32 = 14;
const FLAG_DO_NOT_UPDATE: u32 = 15;
const FLAG_COMPRESSED: u32 = 16;
const FLAG_RIGHT_INPUT_POSITIVE: u32 = 17;
const FLAG_IMM_POSITIVE: u32 = 18;
const FLAG_PRODUCT_POSITIVE: u32 = 19;
const FLAG_LEFT_OPERAND_IS_RS1: u32 = 20;
const FLAG_LEFT_OPERAND_IS_PC: u32 = 21;
const FLAG_RIGHT_OPERAND_IS_RS2: u32 = 22;
const FLAG_RIGHT_OPERAND_IS_IMM: u32 = 23;
pub(crate) const FLAG_IS_FIRST: u32 = 24;
const FLAG_BRANCH: u32 = 25;
const FLAG_NEXT_IS_NOOP: u32 = 26;
pub(crate) const FLAG_IS_NOOP: u32 = 27;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpartanOuterUniskipRow {
    words: [u64; ROW_WORDS],
}

/// The Stage-1 words no resident kernel can derive from the compact row:
/// `[pc, memory_0, memory_1, lookup_output]`, memory slots as in
/// [`SpartanOuterUniskipRow::split`]. `spartan_outer_decode_residual` in
/// `spartan_outer_common.metal` rebuilds the other residual words.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SpartanRawRow {
    words: [u64; RAW_ROW_WORDS],
}

impl SpartanRawRow {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn pc(self) -> u64 {
        self.words[0]
    }

    pub(crate) const fn stage1_ram_pre_value(self) -> u64 {
        self.words[2]
    }
}

/// Provenance of one resident Stage-1 row allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SpartanStage1RowsKey {
    pub(crate) generation: u64,
    pub(crate) rows: usize,
    pub(crate) device_registry_id: u64,
    pub(crate) raw_storage_id: usize,
    pub(crate) compact_storage_id: usize,
}

/// Parked beside Stage-1 rows from
/// [`SolinasMetal::allocate_spartan_outer_uniskip_rows`]: Stage 1's uni-skip
/// writes them while it dispatches, and nothing reads them before.
pub(crate) struct UnwrittenStage1Rows;

#[derive(Clone)]
pub struct SpartanOuterUniskipRows {
    instruction_input_rows: InstructionInputRows,
    raw_buffer: Buffer,
    len: usize,
    explicit_rows: usize,
    device_registry_id: u64,
    generation: u64,
    accounts_instruction_input_rows: bool,
}

impl SpartanOuterUniskipRows {
    copy_field_getters! { pub, {
        len: usize,
        explicit_rows: usize,
        device_registry_id: u64,
    }}

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn with_explicit_rows(mut self, explicit_rows: usize) -> Result<Self, MetalError> {
        if explicit_rows > self.len {
            return Err(MetalError::OuterRemainderExplicitRows {
                explicit: explicit_rows,
                logical: self.len,
            });
        }
        self.explicit_rows = explicit_rows;
        Ok(self)
    }

    pub(crate) fn instruction_input_buffer(&self) -> &Buffer {
        self.instruction_input_rows.buffer()
    }

    pub(crate) fn raw_buffer(&self) -> &Buffer {
        &self.raw_buffer
    }

    pub fn allocation_identity(&self) -> usize {
        self.raw_buffer.as_ptr() as usize
    }

    pub(crate) fn prefetch_residency(&self) -> ResidencyPrefetch {
        residency::prefetch(vec![
            self.instruction_input_buffer().clone(),
            self.raw_buffer.clone(),
        ])
    }

    pub fn instruction_input_allocation_identity(&self) -> usize {
        self.instruction_input_rows.allocation_identity()
    }

    pub(crate) fn key(&self) -> SpartanStage1RowsKey {
        SpartanStage1RowsKey {
            generation: self.generation,
            rows: self.len,
            device_registry_id: self.device_registry_id,
            raw_storage_id: self.allocation_identity(),
            compact_storage_id: self.instruction_input_allocation_identity(),
        }
    }

    pub(crate) fn share_instruction_input_rows(&mut self) -> InstructionInputRows {
        self.accounts_instruction_input_rows = false;
        self.instruction_input_rows.clone()
    }

    pub(crate) fn clone_instruction_input_rows(&self) -> InstructionInputRows {
        self.instruction_input_rows.clone()
    }

    pub(crate) fn restore_instruction_input_accounting(&mut self) {
        self.accounts_instruction_input_rows = true;
    }

    pub(crate) fn share_shift_rows(
        &self,
        context: &SolinasMetal,
    ) -> Result<SpartanShiftResidentRows, MetalError> {
        context.attach_spartan_shift_rows(
            self.instruction_input_buffer().clone(),
            self.raw_buffer.clone(),
            self.len,
        )
    }

    pub(crate) fn share_product_remainder_rows(
        &self,
    ) -> Result<super::ProductRemainderRows, MetalError> {
        super::ProductRemainderRows::from_spartan_stage1(
            self.instruction_input_buffer().clone(),
            self.raw_buffer.clone(),
            self.len,
            self.device_registry_id,
            self.generation,
        )
    }
}

#[cfg(feature = "allocative")]
impl Allocative for SpartanOuterUniskipRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            Key::new("device_raw_rows"),
            self.len * size_of::<SpartanRawRow>(),
        );
        if self.accounts_instruction_input_rows {
            visitor.visit_simple(
                Key::new("device_instruction_input_rows"),
                self.len * size_of::<InstructionInputRow>(),
            );
        }
        visitor.exit();
    }
}

impl SpartanOuterUniskipRow {
    pub const fn from_words(words: [u64; ROW_WORDS]) -> Self {
        Self { words }
    }

    copy_field_getters! { pub, { words: [u64; ROW_WORDS] }}

    pub fn from_spartan_outer(row: &SpartanOuterRow) -> Self {
        let load = row.load.0;
        let store = row.store.0;
        let slot1 = if load {
            row.ram_address.0
        } else {
            row.rs2_value.0
        };
        let slot2 = if store { row.ram_read_value.0 } else { 0 };
        let slot3 = if store {
            row.ram_address.0
        } else {
            row.rd_write_value.0
        };
        let right_input = row.right_instruction_input.0.unsigned_abs();
        let imm = row.imm.0.unsigned_abs();
        let product_limbs = row.product.0.magnitude_limbs();
        let mut flags = 0u64;
        let mut set = |bit: u32, value: bool| flags |= u64::from(value) << bit;
        set(FLAG_LOAD, load);
        set(FLAG_STORE, store);
        set(FLAG_ADD, row.add_operands.0);
        set(FLAG_SUB, row.subtract_operands.0);
        set(FLAG_MUL, row.multiply_operands.0);
        set(FLAG_JUMP, row.jump.0);
        set(FLAG_SHOULD_BRANCH, row.should_branch.0);
        set(FLAG_ASSERT, row.assert_flag.0);
        set(FLAG_SHOULD_JUMP, row.should_jump.0);
        set(FLAG_VIRTUAL, row.virtual_instruction.0);
        set(FLAG_IS_LAST, row.is_last_in_sequence.0);
        set(FLAG_NEXT_VIRTUAL, row.next_is_virtual.0);
        set(FLAG_NEXT_FIRST, row.next_is_first_in_sequence.0);
        set(FLAG_ADVICE, row.advice.0);
        set(FLAG_WRITE_LOOKUP, row.write_lookup_output_to_rd.0);
        set(FLAG_DO_NOT_UPDATE, row.do_not_update_unexpanded_pc.0);
        set(FLAG_COMPRESSED, row.is_compressed.0);
        set(
            FLAG_RIGHT_INPUT_POSITIVE,
            row.right_instruction_input.0 >= 0,
        );
        set(FLAG_IMM_POSITIVE, row.imm.0 >= 0);
        set(FLAG_PRODUCT_POSITIVE, row.product.0.is_positive);
        set(FLAG_LEFT_OPERAND_IS_RS1, row.left_operand_is_rs1.0);
        set(FLAG_LEFT_OPERAND_IS_PC, row.left_operand_is_pc.0);
        set(FLAG_RIGHT_OPERAND_IS_RS2, row.right_operand_is_rs2.0);
        set(FLAG_RIGHT_OPERAND_IS_IMM, row.right_operand_is_imm.0);
        set(FLAG_IS_FIRST, row.is_first_in_sequence.0);
        set(FLAG_BRANCH, row.branch_flag.0);
        set(FLAG_NEXT_IS_NOOP, row.next_is_noop.0);
        set(FLAG_IS_NOOP, row.is_noop.0);
        Self {
            words: [
                row.left_instruction_input.0,
                right_input as u64,
                (right_input >> 64) as u64,
                product_limbs[0],
                product_limbs[1],
                row.pc.0,
                row.unexpanded_pc.0,
                imm as u64,
                (imm >> 64) as u64,
                row.rs1_value.0,
                slot1,
                slot2,
                slot3,
                row.left_lookup_operand.0,
                row.right_lookup_operand.0 as u64,
                (row.right_lookup_operand.0 >> 64) as u64,
                row.next_unexpanded_pc.0,
                row.next_pc.0,
                row.lookup_output.0,
                flags,
            ],
        }
    }

    pub(crate) fn split(self) -> (InstructionInputRow, SpartanRawRow) {
        let words = self.words;
        let flags = words[19];
        let load = flags & (1 << FLAG_LOAD) != 0;
        let slot1 = words[10];
        let slot2 = words[11];
        let slot3 = words[12];
        let memory_0 = if load { slot1 } else { slot3 };
        let memory_1 = if load {
            slot3
        } else if flags & (1 << FLAG_STORE) != 0 {
            slot2
        } else {
            0
        };
        (
            InstructionInputRow::from_full_words(words),
            SpartanRawRow {
                words: [words[5], memory_0, memory_1, words[18]],
            },
        )
    }

    /// Decodes the eight InstructionInput columns in output-claim order.
    pub fn instruction_input_fields<F: jolt_field::Field>(&self) -> [F; 8] {
        let flags = self.words[19];
        let flag = |bit| F::from_u64((flags >> bit) & 1);
        let load = ((flags >> FLAG_LOAD) & 1) != 0;
        let rs2 = if load { 0 } else { self.words[10] };
        let imm_magnitude = u128::from(self.words[7]) | (u128::from(self.words[8]) << 64);
        let imm = F::from_u128(imm_magnitude);
        let imm = if ((flags >> FLAG_IMM_POSITIVE) & 1) != 0 {
            imm
        } else {
            -imm
        };
        [
            flag(FLAG_LEFT_OPERAND_IS_RS1),
            F::from_u64(self.words[9]),
            flag(FLAG_LEFT_OPERAND_IS_PC),
            F::from_u64(self.words[6]),
            flag(FLAG_RIGHT_OPERAND_IS_RS2),
            F::from_u64(rs2),
            flag(FLAG_RIGHT_OPERAND_IS_IMM),
            imm,
        ]
    }

    #[cfg(test)]
    fn spartan_outer_fields<F: jolt_field::Field>(&self) -> [F; 35] {
        let words = self.words;
        let flags = words[19];
        let flag = |bit| F::from_u64((flags >> bit) & 1);
        let enabled = |bit| ((flags >> bit) & 1) != 0;
        let signed = |low: u64, high: u64, positive: bool| {
            let magnitude = u128::from(low) | (u128::from(high) << 64);
            let value = F::from_u128(magnitude);
            if positive {
                value
            } else {
                -value
            }
        };
        let load = enabled(FLAG_LOAD);
        let store = enabled(FLAG_STORE);
        let slot1 = words[10];
        let slot2 = words[11];
        let slot3 = words[12];
        let ram_address = if load {
            slot1
        } else if store {
            slot3
        } else {
            0
        };
        let rs2 = if load { 0 } else { slot1 };
        let rd_write = if store { 0 } else { slot3 };
        let ram_read = if load {
            slot3
        } else if store {
            slot2
        } else {
            0
        };
        let ram_write = if load {
            slot3
        } else if store {
            slot1
        } else {
            0
        };
        [
            F::from_u64(words[0]),
            signed(words[1], words[2], enabled(FLAG_RIGHT_INPUT_POSITIVE)),
            signed(words[3], words[4], enabled(FLAG_PRODUCT_POSITIVE)),
            flag(FLAG_SHOULD_BRANCH),
            F::from_u64(words[5]),
            F::from_u64(words[6]),
            signed(words[7], words[8], enabled(FLAG_IMM_POSITIVE)),
            F::from_u64(ram_address),
            F::from_u64(words[9]),
            F::from_u64(rs2),
            F::from_u64(rd_write),
            F::from_u64(ram_read),
            F::from_u64(ram_write),
            F::from_u64(words[13]),
            F::from_u128(u128::from(words[14]) | (u128::from(words[15]) << 64)),
            F::from_u64(words[16]),
            F::from_u64(words[17]),
            flag(FLAG_NEXT_VIRTUAL),
            flag(FLAG_NEXT_FIRST),
            F::from_u64(words[18]),
            flag(FLAG_SHOULD_JUMP),
            flag(FLAG_ADD),
            flag(FLAG_SUB),
            flag(FLAG_MUL),
            flag(FLAG_LOAD),
            flag(FLAG_STORE),
            flag(FLAG_JUMP),
            flag(FLAG_WRITE_LOOKUP),
            flag(FLAG_VIRTUAL),
            flag(FLAG_ASSERT),
            flag(FLAG_DO_NOT_UPDATE),
            flag(FLAG_ADVICE),
            flag(FLAG_COMPRESSED),
            flag(FLAG_IS_FIRST),
            flag(FLAG_IS_LAST),
        ]
    }
}

struct CpuRowGroups {
    a_first: [i64; 10],
    a_second: [i64; 9],
    b_first: [S192; 10],
    b_second: [S192; 9],
}

fn cpu_row_groups(row: SpartanOuterUniskipRow) -> CpuRowGroups {
    let words = row.words;
    let flags = words[19];
    let flag = |bit: u32| ((flags >> bit) & 1) as i64;
    let load = flag(FLAG_LOAD);
    let store = flag(FLAG_STORE);
    let add = flag(FLAG_ADD);
    let sub = flag(FLAG_SUB);
    let mul = flag(FLAG_MUL);
    let jump = flag(FLAG_JUMP);
    let should_branch = flag(FLAG_SHOULD_BRANCH);
    let slot1 = words[10];
    let slot2 = words[11];
    let slot3 = words[12];
    let ram_address = if load != 0 {
        slot1
    } else if store != 0 {
        slot3
    } else {
        0
    };
    let rs2 = if load != 0 { 0 } else { slot1 };
    let rd_write = if store != 0 { 0 } else { slot3 };
    let ram_read = if load != 0 {
        slot3
    } else if store != 0 {
        slot2
    } else {
        0
    };
    let ram_write = if load != 0 {
        slot3
    } else if store != 0 {
        slot1
    } else {
        0
    };
    let a_first = [
        1 - load - store,
        load,
        load,
        store,
        add + sub + mul,
        1 - add - sub - mul,
        flag(FLAG_ASSERT),
        flag(FLAG_SHOULD_JUMP),
        flag(FLAG_VIRTUAL) - flag(FLAG_IS_LAST),
        flag(FLAG_NEXT_VIRTUAL) - flag(FLAG_NEXT_FIRST),
    ];
    let a_second = [
        load + store,
        add,
        sub,
        mul,
        1 - add - sub - mul - flag(FLAG_ADVICE),
        flag(FLAG_WRITE_LOOKUP),
        jump,
        should_branch,
        1 - should_branch - jump,
    ];
    let signed = |low: u64, high: u64, positive: bool| S192::new([low, high, 0], positive);
    let diff = |left: u64, right: u64| S192::from_i128(i128::from(left) - i128::from(right));
    let right_lookup = signed(words[14], words[15], true);
    let right_input = signed(words[1], words[2], flag(FLAG_RIGHT_INPUT_POSITIVE) != 0);
    let product = signed(words[3], words[4], flag(FLAG_PRODUCT_POSITIVE) != 0);
    let imm = signed(words[7], words[8], flag(FLAG_IMM_POSITIVE) != 0);
    let b_first = [
        S192::from_u64(ram_address),
        diff(ram_read, ram_write),
        diff(ram_read, rd_write),
        diff(rs2, ram_write),
        S192::from_u64(words[13]),
        diff(words[13], words[0]),
        S192::from_i128(i128::from(words[18]) - 1),
        diff(words[16], words[18]),
        S192::from_i128(i128::from(words[17]) - i128::from(words[5]) - 1),
        S192::from_i64(1 - flag(FLAG_DO_NOT_UPDATE)),
    ];
    let two_pow_64 = S192::new([0, 1, 0], true);
    let b_second = [
        S192::from_i128(i128::from(ram_address) - i128::from(words[9])) - imm,
        right_lookup - S192::from_u64(words[0]) - right_input,
        right_lookup - S192::from_u64(words[0]) + right_input - two_pow_64,
        right_lookup - product,
        right_lookup - right_input,
        diff(rd_write, words[18]),
        S192::from_i128(
            i128::from(rd_write) - i128::from(words[6]) - 4 + 2 * i128::from(flag(FLAG_COMPRESSED)),
        ),
        S192::from_i128(i128::from(words[16]) - i128::from(words[6])) - imm,
        S192::from_i128(
            i128::from(words[16]) - i128::from(words[6]) - 4
                + 4 * i128::from(flag(FLAG_DO_NOT_UPDATE))
                + 2 * i128::from(flag(FLAG_COMPRESSED)),
        ),
    ];
    CpuRowGroups {
        a_first,
        a_second,
        b_first,
        b_second,
    }
}

fn cpu_extended_products(values: &CpuRowGroups) -> [(S256, S256); SPARTAN_OUTER_EXTENDED_NODES] {
    let mut products = [(S256::zero(), S256::zero()); SPARTAN_OUTER_EXTENDED_NODES];
    for (node, coefficients) in EXTENSION_COEFFICIENTS.iter().enumerate() {
        let mut az_first = 0i64;
        let mut az_second = 0i64;
        let mut bz_first = S192::zero();
        let mut bz_second = S192::zero();
        for (index, &coefficient) in coefficients.iter().enumerate() {
            az_first += coefficient * values.a_first[index];
            S64::from_i64(coefficient).fmadd_trunc::<3, 3>(&values.b_first[index], &mut bz_first);
            if index < 9 {
                az_second += coefficient * values.a_second[index];
                S64::from_i64(coefficient)
                    .fmadd_trunc::<3, 3>(&values.b_second[index], &mut bz_second);
            }
        }
        products[node] = (
            S64::from_i64(az_first).mul_trunc::<3, 4>(&bz_first),
            S64::from_i64(az_second).mul_trunc::<3, 4>(&bz_second),
        );
    }
    products
}

pub fn evaluate_spartan_outer_uniskip_cpu(
    rows: &[SpartanOuterUniskipRow],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> Result<[AkitaField; SPARTAN_OUTER_EXTENDED_NODES], MetalError> {
    let pairs_per_block = e_in.len() / 2;
    if e_in.is_empty()
        || !e_in.len().is_multiple_of(2)
        || e_out.is_empty()
        || pairs_per_block.checked_mul(e_out.len()) != Some(rows.len())
    {
        return Err(MetalError::SpartanOuterUniskipShape {
            rows: rows.len(),
            e_in: e_in.len(),
            e_out: e_out.len(),
        });
    }
    let block = |block: usize| {
        let mut accumulators: [<AkitaField as WithAccumulator>::SignedProductAccumulator;
            SPARTAN_OUTER_EXTENDED_NODES] = Default::default();
        for pair in 0..pairs_per_block {
            let values = cpu_row_groups(rows[block * pairs_per_block + pair]);
            let products = cpu_extended_products(&values);
            for (accumulator, (first, second)) in accumulators.iter_mut().zip(&products) {
                accumulator.fmadd_s256(e_in[2 * pair], first);
                accumulator.fmadd_s256(e_in[2 * pair + 1], second);
            }
        }
        std::array::from_fn(|node| e_out[block] * accumulators[node].reduce())
    };
    let merge = |mut left: [AkitaField; SPARTAN_OUTER_EXTENDED_NODES], right| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    #[cfg(feature = "parallel")]
    let output = (0..e_out.len())
        .into_par_iter()
        .map(block)
        .reduce(|| [AkitaField::zero(); SPARTAN_OUTER_EXTENDED_NODES], merge);
    #[cfg(not(feature = "parallel"))]
    let output = (0..e_out.len())
        .map(block)
        .fold([AkitaField::zero(); SPARTAN_OUTER_EXTENDED_NODES], merge);
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanOuterUniskipConfig {
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for SpartanOuterUniskipConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(256),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    pairs_per_block: u32,
    blocks: u32,
    first_block: u32,
}

struct Buffers {
    instruction_input_rows: Buffer,
    raw_rows: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    block_sums: Buffer,
    output: Buffer,
}

pub struct SpartanOuterUniskipInvocation<'a> {
    context: &'a SolinasMetal,
    blocks_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    buffers: Buffers,
    params: Params,
    threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_spartan_outer_uniskip(
        &self,
        rows: &[SpartanOuterUniskipRow],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: SpartanOuterUniskipConfig,
    ) -> Result<SpartanOuterUniskipInvocation<'_>, MetalError> {
        let resident = self.prepare_spartan_outer_uniskip_rows(rows)?;
        self.prepare_spartan_outer_uniskip_with_rows(&resident, e_in, e_out, config)
    }

    pub fn prepare_spartan_outer_uniskip_rows(
        &self,
        rows: &[SpartanOuterUniskipRow],
    ) -> Result<SpartanOuterUniskipRows, MetalError> {
        self.prepare_spartan_outer_uniskip_rows_with_fill(rows.len(), |instruction_input, raw| {
            for ((source, instruction_input), raw) in
                rows.iter().copied().zip(instruction_input).zip(raw)
            {
                (*instruction_input, *raw) = source.split();
            }
            Ok(())
        })
    }

    pub(crate) fn prepare_spartan_outer_uniskip_rows_with_fill(
        &self,
        rows: usize,
        fill: impl FnOnce(&mut [InstructionInputRow], &mut [SpartanRawRow]) -> Result<(), MetalError>,
    ) -> Result<SpartanOuterUniskipRows, MetalError> {
        let prepared = self.allocate_spartan_outer_uniskip_rows(rows)?;
        let _residency = prepared.prefetch_residency();
        // SAFETY: the shared buffers have exactly `rows` elements and no command
        // buffer can observe an allocation until `fill` returns.
        let instruction_input = unsafe {
            slice::from_raw_parts_mut(
                prepared
                    .instruction_input_buffer()
                    .contents()
                    .cast::<InstructionInputRow>(),
                rows,
            )
        };
        // SAFETY: see the instruction-input slice above.
        let raw = unsafe {
            slice::from_raw_parts_mut(prepared.raw_buffer.contents().cast::<SpartanRawRow>(), rows)
        };
        fill(instruction_input, raw)?;
        Ok(prepared)
    }

    /// Allocates the Stage-1 rows without writing them: the uni-skip writes
    /// them through [`SpartanOuterUniskipInvocation::execute_filling`].
    pub(crate) fn allocate_spartan_outer_uniskip_rows(
        &self,
        rows: usize,
    ) -> Result<SpartanOuterUniskipRows, MetalError> {
        if rows == 0 {
            return Err(MetalError::EmptyInput);
        }
        let instruction_input_bytes = byte_length::<InstructionInputRow>(rows)?;
        let raw_bytes = byte_length::<SpartanRawRow>(rows)?;
        for bytes in [instruction_input_bytes, raw_bytes] {
            self.validate_buffer_length(bytes)?;
        }
        self.validate_additional_working_set(spartan_outer_uniskip_row_bytes(rows)?)?;
        let instruction_input_buffer = self.device.new_buffer(
            instruction_input_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let raw_buffer = self
            .device
            .new_buffer(raw_bytes, MTLResourceOptions::StorageModeShared);
        let device_registry_id = self.device_registry_id();
        let generation = NEXT_STAGE1_ROWS_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                MetalError::InvalidInstructionInputState("Stage-1 row generation counter exhausted")
            })?;
        Ok(SpartanOuterUniskipRows {
            instruction_input_rows: InstructionInputRows::from_buffer(
                instruction_input_buffer,
                rows,
                device_registry_id,
            ),
            raw_buffer,
            len: rows,
            explicit_rows: rows,
            device_registry_id,
            generation,
            accounts_instruction_input_rows: true,
        })
    }

    pub fn prepare_spartan_outer_uniskip_with_rows(
        &self,
        rows: &SpartanOuterUniskipRows,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: SpartanOuterUniskipConfig,
    ) -> Result<SpartanOuterUniskipInvocation<'_>, MetalError> {
        self.prepare_spartan_outer_uniskip_from_buffers(
            rows.instruction_input_buffer().clone(),
            rows.raw_buffer().clone(),
            rows.len,
            e_in,
            e_out,
            config,
        )
    }

    fn prepare_spartan_outer_uniskip_from_buffers(
        &self,
        instruction_input_rows_buffer: Buffer,
        raw_rows_buffer: Buffer,
        rows: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: SpartanOuterUniskipConfig,
    ) -> Result<SpartanOuterUniskipInvocation<'_>, MetalError> {
        let Some(pairs_per_block) = e_in.len().checked_div(2) else {
            return Err(MetalError::SpartanOuterUniskipShape {
                rows,
                e_in: e_in.len(),
                e_out: e_out.len(),
            });
        };
        if e_in.is_empty()
            || !e_in.len().is_multiple_of(2)
            || e_out.is_empty()
            || pairs_per_block.checked_mul(e_out.len()) != Some(rows)
        {
            return Err(MetalError::SpartanOuterUniskipShape {
                rows,
                e_in: e_in.len(),
                e_out: e_out.len(),
            });
        }

        let blocks_pipeline = self.compile_named_pipeline(BLOCKS_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let blocks_limits = Self::limits(&blocks_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        for (pipeline, limits) in [
            (BLOCKS_PIPELINE, blocks_limits),
            (REDUCE_PIPELINE, reduce_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedSpartanOuterExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, blocks_limits)?;
        if threads_per_threadgroup > reduce_limits.max_total_threads_per_threadgroup {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: threads_per_threadgroup,
                execution_width: reduce_limits.thread_execution_width,
                maximum: reduce_limits.max_total_threads_per_threadgroup,
            });
        }
        let block_scratch = byte_length::<Fp128>(threads_per_threadgroup)?;
        let reduce_scratch = byte_length::<Fp128>(threads_per_threadgroup / SIMD_WIDTH)?;
        for requested in [block_scratch, reduce_scratch] {
            if requested > self.device.max_threadgroup_memory_length() {
                return Err(MetalError::SpartanOuterThreadgroupMemory {
                    requested,
                    maximum: self.device.max_threadgroup_memory_length(),
                });
            }
        }

        let e_in_fp = e_in.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let e_out_fp = e_out.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let block_elements = e_out
            .len()
            .checked_mul(SPARTAN_OUTER_EXTENDED_NODES)
            .ok_or(MetalError::InputTooLong(e_out.len()))?;
        let e_in_bytes = byte_length::<Fp128>(e_in_fp.len())?;
        let e_out_bytes = byte_length::<Fp128>(e_out_fp.len())?;
        let block_bytes = byte_length::<Fp128>(block_elements)?;
        let output_bytes = byte_length::<Fp128>(SPARTAN_OUTER_EXTENDED_NODES)?;
        let params_bytes = byte_length::<Params>(1)?;
        for requested in [
            e_in_bytes,
            e_out_bytes,
            block_bytes,
            output_bytes,
            params_bytes,
        ] {
            self.validate_buffer_length(requested)?;
        }
        let invocation_bytes = [
            e_in_bytes,
            e_out_bytes,
            block_bytes,
            output_bytes,
            params_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(MetalError::InputTooLong(rows))
        })?;
        self.validate_additional_working_set(invocation_bytes)?;
        let params = Params {
            rows: u32::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?,
            pairs_per_block: u32::try_from(pairs_per_block)
                .map_err(|_| MetalError::InputTooLong(pairs_per_block))?,
            blocks: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            first_block: 0,
        };

        Ok(SpartanOuterUniskipInvocation {
            context: self,
            blocks_pipeline,
            reduce_pipeline,
            buffers: Buffers {
                instruction_input_rows: instruction_input_rows_buffer,
                raw_rows: raw_rows_buffer,
                e_in: buffer_from_slice(&self.device, &e_in_fp),
                e_out: buffer_from_slice(&self.device, &e_out_fp),
                block_sums: self
                    .device
                    .new_buffer(block_bytes, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared),
            },
            params,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }
}

impl SpartanOuterUniskipInvocation<'_> {
    copy_field_getters! { pub, { threads_per_threadgroup: usize }}

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            self.encode_blocks(command_buffer, 0, self.params.blocks)?;
            self.encode_reduce(command_buffer)?;
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = completed_command_gpu_time(command_buffer)?;
            self.completed.set(true);
            Ok(gpu_active)
        })
    }

    /// Writes the rows with `fill(first_row, compact, raw)` in
    /// [`FILL_CHUNKS`] chunks and dispatches each chunk's blocks once the next
    /// chunk is written, so the GPU reduces behind the producer.
    ///
    /// The rows must come from
    /// [`SolinasMetal::allocate_spartan_outer_uniskip_rows`] with nothing
    /// reading them before this call.
    pub(crate) fn execute_filling(
        &self,
        mut fill: impl FnMut(
            usize,
            &mut [InstructionInputRow],
            &mut [SpartanRawRow],
        ) -> Result<(), MetalError>,
    ) -> Result<Duration, MetalError> {
        let rows = self.params.rows as usize;
        let rows_per_block = self.params.pairs_per_block as usize;
        // SAFETY: the row buffers hold exactly `rows` elements; a command
        // buffer binding them is committed only after every row its blocks
        // read is written, and no other reader exists before this call.
        let instruction_input = unsafe {
            slice::from_raw_parts_mut(
                self.buffers
                    .instruction_input_rows
                    .contents()
                    .cast::<InstructionInputRow>(),
                rows,
            )
        };
        // SAFETY: see the instruction-input slice above.
        let raw = unsafe {
            slice::from_raw_parts_mut(
                self.buffers.raw_rows.contents().cast::<SpartanRawRow>(),
                rows,
            )
        };
        let chunk_blocks = self.params.blocks.div_ceil(FILL_CHUNKS);
        let chunk_rows = chunk_blocks as usize * rows_per_block;
        autoreleasepool(|| {
            let mut commands = Vec::with_capacity(FILL_CHUNKS as usize);
            let mut pending_block = None;
            for (chunk, (instruction_input, raw)) in instruction_input
                .chunks_mut(chunk_rows)
                .zip(raw.chunks_mut(chunk_rows))
                .enumerate()
            {
                let first_row = chunk * chunk_rows;
                fill(first_row, instruction_input, raw)?;
                // A block's last row reads its successor's compact and raw
                // words (`spartan_outer_decode_residual`), so the previous
                // chunk goes out once this one is written.
                if let Some(first_block) = pending_block.replace(first_row / rows_per_block) {
                    let command_buffer = self.context.queue.new_command_buffer();
                    self.encode_blocks(command_buffer, first_block as u32, chunk_blocks)?;
                    command_buffer.commit();
                    commands.push(command_buffer.to_owned());
                }
            }
            let first_block = pending_block.ok_or(MetalError::EmptyInput)? as u32;
            let command_buffer = self.context.queue.new_command_buffer();
            self.encode_blocks(
                command_buffer,
                first_block,
                self.params.blocks - first_block,
            )?;
            self.encode_reduce(command_buffer)?;
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = commands.iter().try_fold(
                completed_command_gpu_time(command_buffer)?,
                |total, command| Ok::<_, MetalError>(total + completed_command_gpu_time(command)?),
            )?;
            self.completed.set(true);
            Ok(gpu_active)
        })
    }

    fn encode_blocks(
        &self,
        command_buffer: &CommandBufferRef,
        first_block: u32,
        blocks: u32,
    ) -> Result<(), MetalError> {
        let params = Params {
            first_block,
            ..self.params
        };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.blocks_pipeline);
        encoder.set_buffer(0, Some(&self.buffers.instruction_input_rows), 0);
        encoder.set_buffer(1, Some(&self.buffers.raw_rows), 0);
        encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(4, Some(&self.buffers.block_sums), 0);
        set_inline_bytes(encoder, 5, &params);
        encoder
            .set_threadgroup_memory_length(0, byte_length::<Fp128>(self.threads_per_threadgroup)?);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(blocks),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        Ok(())
    }

    fn encode_reduce(&self, command_buffer: &CommandBufferRef) -> Result<(), MetalError> {
        let reduce = command_buffer.new_compute_command_encoder();
        reduce.set_compute_pipeline_state(&self.reduce_pipeline);
        reduce.set_buffer(0, Some(&self.buffers.block_sums), 0);
        reduce.set_buffer(1, Some(&self.buffers.output), 0);
        set_inline_bytes(reduce, 2, &self.params);
        reduce.set_threadgroup_memory_length(
            0,
            byte_length::<Fp128>(self.threads_per_threadgroup / SIMD_WIDTH)?,
        );
        reduce.dispatch_thread_groups(
            MTLSize {
                width: SPARTAN_OUTER_EXTENDED_NODES as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
        reduce.end_encoding();
        Ok(())
    }

    pub fn read_output(&self) -> Result<[AkitaField; SPARTAN_OUTER_EXTENDED_NODES], MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        // SAFETY: the shared output buffer has exactly nine field elements and
        // the command buffer has completed before this read.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                SPARTAN_OUTER_EXTENDED_NODES,
            )
        };
        self.context
            .validate_inputs("Spartan outer uni-skip output", values)?;
        Ok(std::array::from_fn(|index| {
            values[index].into_jolt_field::<AkitaField>()
        }))
    }
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

/// Bytes of the resident Stage-1 rows: compact plus raw words.
pub(crate) fn spartan_outer_uniskip_row_bytes(rows: usize) -> Result<u64, MetalError> {
    byte_length::<InstructionInputRow>(rows)?
        .checked_add(byte_length::<SpartanRawRow>(rows)?)
        .ok_or(MetalError::InputTooLong(rows))
}

pub(crate) fn spartan_outer_uniskip_invocation_bytes(rows: usize) -> Result<u64, MetalError> {
    if rows == 0 || !rows.is_power_of_two() {
        return Err(MetalError::SpartanOuterUniskipShape {
            rows,
            e_in: 0,
            e_out: 0,
        });
    }
    let split = (rows.ilog2() as usize).div_ceil(2);
    let e_out = 1usize << split;
    let e_in = rows
        .checked_mul(2)
        .and_then(|elements| elements.checked_div(e_out))
        .ok_or(MetalError::InputTooLong(rows))?;
    let block_elements = e_out
        .checked_mul(SPARTAN_OUTER_EXTENDED_NODES)
        .ok_or(MetalError::InputTooLong(e_out))?;
    [
        byte_length::<Fp128>(e_in)?,
        byte_length::<Fp128>(e_out)?,
        byte_length::<Fp128>(block_elements)?,
        byte_length::<Fp128>(SPARTAN_OUTER_EXTENDED_NODES)?,
        byte_length::<Params>(1)?,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or(MetalError::InputTooLong(rows))
    })
}

const _: () = assert!(size_of::<SpartanOuterUniskipRow>() == 160);
const _: () = assert!(size_of::<SpartanRawRow>() == 32);
const _: () = assert!(size_of::<InstructionInputRow>() + size_of::<SpartanRawRow>() == 80);
const _: () = assert!(size_of::<Params>() == 16);

/// Witness Stage-1 rows over every expanded instruction kind, the ground truth
/// for the resident-row decoders.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test fixture")]
pub(crate) mod test_rows {
    use jolt_program::execution::{
        RamAccess, RamRead, RamWrite, RegisterRead, RegisterState, RegisterWrite, TraceRow,
    };
    use jolt_program::expand::{expand_instruction, ExpansionAllocator};
    use jolt_program::image::decode::decode_instruction;
    use jolt_program::preprocess::BytecodePreprocessing;
    use jolt_riscv::{
        CircuitFlags, Flags as _, JoltInstruction, JoltInstructionKind, JoltInstructionRow,
        NormalizedOperands, RV64IMAC_JOLT,
    };
    use jolt_witness::testing::with_trace_backend;
    use jolt_witness::witnesses::SpartanOuterRow;
    use jolt_witness::{BundleSource, WitnessBundle};

    use super::SpartanOuterUniskipRow;

    pub(crate) const ADVICE_KINDS: [&str; 3] =
        ["VirtualAdvice", "VirtualAdviceLoad", "VirtualAdviceLen"];

    pub(crate) struct ExpandedTrace {
        pub(crate) instructions: Vec<JoltInstructionRow>,
        pub(crate) rows: Vec<TraceRow>,
        pub(crate) sources: usize,
        /// The fixed edge rows closing `rows`.
        pub(crate) edge_rows: usize,
    }

    pub(crate) fn splitmix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Random RV64 words through the production decoder and expansion (plus
    /// the inline-only and advice kinds as standalone rows) with random
    /// register state. Row 0 is a Noop; the other rows are shuffled so every
    /// prefix mixes instruction kinds.
    pub(crate) fn expanded_trace() -> ExpandedTrace {
        const OPCODES: [u32; 15] = [
            0b011_0111, 0b001_0111, 0b110_1111, 0b110_0111, 0b110_0011, 0b000_0011, 0b010_0011,
            0b001_0011, 0b001_1011, 0b011_0011, 0b011_1011, 0b000_1111, 0b010_1111, 0b111_0011,
            0b101_1011,
        ];
        const VALUES: [u64; 8] = [
            0,
            1,
            u64::MAX,
            1 << 63,
            (1 << 63) - 1,
            0xffff_ffff,
            0x8000_0000,
            0x1234_5678_9abc_def0,
        ];
        let base = 0x8000_0000usize;
        let mut instructions = vec![JoltInstructionRow {
            address: base,
            ..Default::default()
        }];
        let mut seed = 0x0dec_0de5_u64;
        let mut sources = 0usize;
        for opcode in OPCODES {
            for _ in 0..1500 {
                seed = splitmix(seed);
                let word = opcode | ((seed as u32) & !0x7f);
                let address = base + 4 * (1 + instructions.len());
                let Ok(source) =
                    decode_instruction(word, address as u64, seed >> 63 == 1, RV64IMAC_JOLT)
                else {
                    continue;
                };
                let Ok(expanded) =
                    expand_instruction(&source, &mut ExpansionAllocator::new(), RV64IMAC_JOLT)
                else {
                    continue;
                };
                let expanded = expanded
                    .into_iter()
                    .map(JoltInstructionRow::from)
                    .collect::<Vec<_>>();
                if BytecodePreprocessing::preprocess(
                    expanded.clone(),
                    address as u64,
                    RV64IMAC_JOLT,
                )
                .is_err()
                {
                    continue;
                }
                sources += 1;
                instructions.extend(expanded);
            }
        }
        // Inline-only virtual kinds (and the advice kinds) as standalone rows;
        // the immediate is a shift mask with seven trailing zeros, valid for
        // every rotate/pow2/bitmask lowering.
        let expanded_kinds = instructions
            .iter()
            .map(|row| row.instruction_kind)
            .collect::<Vec<_>>();
        for &kind in JoltInstructionKind::ALL {
            if expanded_kinds.contains(&kind) && !ADVICE_KINDS.contains(&kind.name()) {
                continue;
            }
            let row = JoltInstructionRow {
                instruction_kind: kind,
                address: base + 4 * (1 + instructions.len()),
                operands: NormalizedOperands {
                    rd: Some(9),
                    rs1: Some(6),
                    rs2: Some(7),
                    imm: 0xffff_ff80,
                },
                ..Default::default()
            };
            if BytecodePreprocessing::preprocess(vec![row], row.address as u64, RV64IMAC_JOLT)
                .is_ok()
            {
                instructions.push(row);
            }
        }
        let mut value = 0x5eed_u64;
        let mut next_value = || {
            value = splitmix(value);
            if value.is_multiple_of(3) {
                VALUES[(value >> 8) as usize % VALUES.len()]
            } else {
                value
            }
        };
        let mut rows = instructions
            .iter()
            .filter_map(|&instruction| {
                let operands = instruction.operands;
                let read = |register: Option<u8>, value: u64| {
                    register.map(|register| RegisterRead { register, value })
                };
                // Odd rs2 keeps the virtual shifts' trailing-zero shift amounts in range.
                let mut registers = RegisterState {
                    rs1: read(operands.rs1, next_value()),
                    rs2: read(operands.rs2, next_value() | 1),
                    rd: operands.rd.map(|register| RegisterWrite {
                        register,
                        pre_value: next_value(),
                        post_value: next_value(),
                    }),
                };
                let flags = JoltInstruction::try_from(instruction).ok()?.circuit_flags();
                let ram = if flags.get(CircuitFlags::Load) {
                    registers.rs2 = None;
                    RamAccess::Read(RamRead {
                        address: next_value(),
                        value: registers.rd.map_or(0, |rd| rd.post_value),
                    })
                } else if flags.get(CircuitFlags::Store) {
                    RamAccess::Write(RamWrite {
                        address: next_value(),
                        pre_value: next_value(),
                        post_value: registers.rs2.map_or(0, |rs2| rs2.value),
                    })
                } else {
                    RamAccess::NoOp
                };
                TraceRow::new(instruction, registers, ram).ok()
            })
            .collect::<Vec<_>>();
        rows[1..].sort_by_cached_key(|row| splitmix(row.instruction().address as u64));
        // Fixed edges, appended after the shuffle so every shorter prefix keeps
        // its rows: AUIPC and both alignment asserts with immediates negative
        // in 64-bit two's complement, a taken branch with a sign-magnitude
        // negative immediate, and multiplies of maximal operands.
        let read = |register, value| Some(RegisterRead { register, value });
        let write = |register, post_value| {
            Some(RegisterWrite {
                register,
                pre_value: 0,
                post_value,
            })
        };
        let operands = |rd, rs1, rs2, imm| NormalizedOperands { rd, rs1, rs2, imm };
        let auipc_pc = (base + 4 * (1 + instructions.len())) as u64;
        let auipc_imm = -4096i64 as u64;
        let edges = [
            (
                JoltInstructionKind::AUIPC,
                operands(Some(9), None, None, i128::from(auipc_imm)),
                RegisterState {
                    rs1: None,
                    rs2: None,
                    rd: write(9, auipc_pc.wrapping_add(auipc_imm)),
                },
            ),
            (
                JoltInstructionKind::VirtualAssertHalfwordAlignment,
                operands(None, Some(6), None, i128::from(-2i64 as u64)),
                RegisterState {
                    rs1: read(6, 0x1000),
                    rs2: None,
                    rd: None,
                },
            ),
            (
                JoltInstructionKind::VirtualAssertWordAlignment,
                operands(None, Some(6), None, i128::from(-4i64 as u64)),
                RegisterState {
                    rs1: read(6, 0x1000),
                    rs2: None,
                    rd: None,
                },
            ),
            (
                JoltInstructionKind::BEQ,
                operands(None, Some(6), Some(7), -8),
                RegisterState {
                    rs1: read(6, 5),
                    rs2: read(7, 5),
                    rd: None,
                },
            ),
            (
                JoltInstructionKind::MUL,
                operands(Some(9), Some(6), Some(7), 0),
                RegisterState {
                    rs1: read(6, u64::MAX),
                    rs2: read(7, u64::MAX),
                    rd: write(9, 1),
                },
            ),
            (
                JoltInstructionKind::MULHU,
                operands(Some(9), Some(6), Some(7), 0),
                RegisterState {
                    rs1: read(6, u64::MAX),
                    rs2: read(7, u64::MAX),
                    rd: write(9, u64::MAX - 1),
                },
            ),
        ];
        let edge_rows = edges.len();
        for (instruction_kind, operands, registers) in edges {
            let instruction = JoltInstructionRow {
                instruction_kind,
                address: base + 4 * (1 + instructions.len()),
                operands,
                ..Default::default()
            };
            let _bytecode = BytecodePreprocessing::preprocess(
                vec![instruction],
                instruction.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap();
            instructions.push(instruction);
            rows.push(TraceRow::new(instruction, registers, RamAccess::NoOp).unwrap());
        }
        ExpandedTrace {
            instructions,
            rows,
            sources,
            edge_rows,
        }
    }

    /// The witness of the first `2^log_t - 1` rows of [`expanded_trace`],
    /// padded to `2^log_t`.
    pub(crate) fn witness<B: WitnessBundle + Clone + Send + Sync>(log_t: usize) -> Vec<B> {
        let ExpandedTrace {
            instructions,
            mut rows,
            ..
        } = expanded_trace();
        rows.truncate((1 << log_t) - 1);
        let log_k = instructions.len().next_power_of_two().ilog2() as usize;
        with_trace_backend(instructions, rows, log_t, log_k, 4, |backend| {
            backend.bundles().unwrap()
        })
    }

    pub(crate) fn stage1_rows(log_t: usize) -> Vec<SpartanOuterUniskipRow> {
        witness::<SpartanOuterRow>(log_t)
            .iter()
            .map(SpartanOuterUniskipRow::from_spartan_outer)
            .collect()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::{FromPrimitiveInt, One as _};
    use jolt_poly::EqPolynomial;
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::OpFlag;
    use jolt_witness::BundleSource;

    use jolt_riscv::JoltInstructionKind;

    use super::test_rows::{expanded_trace, splitmix, stage1_rows, witness, ADVICE_KINDS};
    use super::*;
    use crate::metal::solinas::{
        OuterRemainderSequenceConfig, OuterRemainderStorageInitialization,
    };

    #[test]
    fn resident_row_bytes_match_the_production_geometry() {
        assert_eq!(
            spartan_outer_uniskip_row_bytes(1 << 26).unwrap(),
            5_368_709_120
        );
    }

    #[test]
    fn packed_row_preserves_is_first_in_sequence() {
        with_sample_backend(|backend| {
            let mut rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
            rows[0].is_first_in_sequence = OpFlag(true);
            rows[1].is_first_in_sequence = OpFlag(false);
            for row in rows.into_iter().take(2) {
                let packed = SpartanOuterUniskipRow::from_spartan_outer(&row);
                assert_eq!(
                    (packed.words()[19] >> FLAG_IS_FIRST) & 1,
                    u64::from(row.is_first_in_sequence.0)
                );
            }
        });
    }

    #[test]
    fn canonical_padding_has_one_nonzero_outer_opening() {
        with_sample_backend(|backend| {
            let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
            let mut expected = [AkitaField::zero(); 35];
            expected[30] = AkitaField::one();
            for row in &rows[2..] {
                let packed = SpartanOuterUniskipRow::from_spartan_outer(row);
                assert_eq!(packed.spartan_outer_fields::<AkitaField>(), expected);
            }
        });
    }

    #[test]
    fn invocation_bytes_match_the_split_eq_geometry() {
        assert_eq!(
            spartan_outer_uniskip_invocation_bytes(1 << 26).unwrap(),
            1_573_024
        );
        assert_eq!(
            spartan_outer_uniskip_invocation_bytes(1 << 28).unwrap(),
            3_145_888
        );
    }

    fn field_signed_magnitude(low: u64, high: u64, positive: bool) -> AkitaField {
        let magnitude = u128::from(low) | (u128::from(high) << 64);
        let value = AkitaField::from_u128(magnitude);
        if positive {
            value
        } else {
            -value
        }
    }

    fn flag(flags: u64, bit: u32) -> i64 {
        ((flags >> bit) & 1) as i64
    }

    fn row_groups(
        row: SpartanOuterUniskipRow,
    ) -> ([i64; 10], [i64; 9], [AkitaField; 10], [AkitaField; 9]) {
        let words = row.words();
        let flags = words[19];
        let load = flag(flags, FLAG_LOAD);
        let store = flag(flags, FLAG_STORE);
        let add = flag(flags, FLAG_ADD);
        let sub = flag(flags, FLAG_SUB);
        let mul = flag(flags, FLAG_MUL);
        let jump = flag(flags, FLAG_JUMP);
        let should_branch = flag(flags, FLAG_SHOULD_BRANCH);
        let slot1 = words[10];
        let slot2 = words[11];
        let slot3 = words[12];
        let ram_address = if load != 0 {
            slot1
        } else if store != 0 {
            slot3
        } else {
            0
        };
        let rs2 = if load != 0 { 0 } else { slot1 };
        let rd_write = if store != 0 { 0 } else { slot3 };
        let ram_read = if load != 0 {
            slot3
        } else if store != 0 {
            slot2
        } else {
            0
        };
        let ram_write = if load != 0 {
            slot3
        } else if store != 0 {
            slot1
        } else {
            0
        };
        let a_first = [
            1 - load - store,
            load,
            load,
            store,
            add + sub + mul,
            1 - add - sub - mul,
            flag(flags, FLAG_ASSERT),
            flag(flags, FLAG_SHOULD_JUMP),
            flag(flags, FLAG_VIRTUAL) - flag(flags, FLAG_IS_LAST),
            flag(flags, FLAG_NEXT_VIRTUAL) - flag(flags, FLAG_NEXT_FIRST),
        ];
        let a_second = [
            load + store,
            add,
            sub,
            mul,
            1 - add - sub - mul - flag(flags, FLAG_ADVICE),
            flag(flags, FLAG_WRITE_LOOKUP),
            jump,
            should_branch,
            1 - should_branch - jump,
        ];
        let u = AkitaField::from_u64;
        let signed = |value: i128| AkitaField::from_i128(value);
        let right_lookup =
            AkitaField::from_u128(u128::from(words[14]) | (u128::from(words[15]) << 64));
        let right_input = field_signed_magnitude(
            words[1],
            words[2],
            flag(flags, FLAG_RIGHT_INPUT_POSITIVE) != 0,
        );
        let product =
            field_signed_magnitude(words[3], words[4], flag(flags, FLAG_PRODUCT_POSITIVE) != 0);
        let imm = field_signed_magnitude(words[7], words[8], flag(flags, FLAG_IMM_POSITIVE) != 0);
        let b_first = [
            u(ram_address),
            signed(i128::from(ram_read) - i128::from(ram_write)),
            signed(i128::from(ram_read) - i128::from(rd_write)),
            signed(i128::from(rs2) - i128::from(ram_write)),
            u(words[13]),
            signed(i128::from(words[13]) - i128::from(words[0])),
            signed(i128::from(words[18]) - 1),
            signed(i128::from(words[16]) - i128::from(words[18])),
            signed(i128::from(words[17]) - i128::from(words[5]) - 1),
            signed(1 - i128::from(flag(flags, FLAG_DO_NOT_UPDATE))),
        ];
        let two_pow_64 = AkitaField::from_u128(1u128 << 64);
        let b_second = [
            signed(i128::from(ram_address) - i128::from(words[9])) - imm,
            right_lookup - u(words[0]) - right_input,
            right_lookup - u(words[0]) + right_input - two_pow_64,
            right_lookup - product,
            right_lookup - right_input,
            signed(i128::from(rd_write) - i128::from(words[18])),
            signed(i128::from(rd_write) - i128::from(words[6]) - 4)
                + u64::try_from(2 * flag(flags, FLAG_COMPRESSED))
                    .map_or_else(|_| AkitaField::zero(), AkitaField::from_u64),
            signed(i128::from(words[16]) - i128::from(words[6])) - imm,
            signed(i128::from(words[16]) - i128::from(words[6]) - 4)
                + AkitaField::from_i64(4 * flag(flags, FLAG_DO_NOT_UPDATE))
                + AkitaField::from_i64(2 * flag(flags, FLAG_COMPRESSED)),
        ];
        (a_first, a_second, b_first, b_second)
    }

    fn reference(
        rows: &[SpartanOuterUniskipRow],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> [AkitaField; SPARTAN_OUTER_EXTENDED_NODES] {
        let pairs = e_in.len() / 2;
        let mut output = [AkitaField::zero(); SPARTAN_OUTER_EXTENDED_NODES];
        for (block, &outer) in e_out.iter().enumerate() {
            let mut block_sum = [AkitaField::zero(); SPARTAN_OUTER_EXTENDED_NODES];
            for pair in 0..pairs {
                let (a_first, a_second, b_first, b_second) = row_groups(rows[block * pairs + pair]);
                for (node, coefficients) in EXTENSION_COEFFICIENTS.iter().enumerate() {
                    let mut az_first = AkitaField::zero();
                    let mut az_second = AkitaField::zero();
                    let mut bz_first = AkitaField::zero();
                    let mut bz_second = AkitaField::zero();
                    for (index, &coefficient) in coefficients.iter().enumerate() {
                        let coefficient = AkitaField::from_i64(coefficient);
                        az_first += coefficient * AkitaField::from_i64(a_first[index]);
                        bz_first += coefficient * b_first[index];
                        if index < 9 {
                            az_second += coefficient * AkitaField::from_i64(a_second[index]);
                            bz_second += coefficient * b_second[index];
                        }
                    }
                    block_sum[node] += e_in[2 * pair] * az_first * bz_first
                        + e_in[2 * pair + 1] * az_second * bz_second;
                }
            }
            for (sum, block_sum) in output.iter_mut().zip(block_sum) {
                *sum += outer * block_sum;
            }
        }
        output
    }

    fn outer_tables(
        rows: &[SpartanOuterUniskipRow],
        lagrange: &[AkitaField; 10],
    ) -> (Vec<AkitaField>, Vec<AkitaField>) {
        let mut az = Vec::with_capacity(2 * rows.len());
        let mut bz = Vec::with_capacity(2 * rows.len());
        for &row in rows {
            let (a_first, a_second, b_first, b_second) = row_groups(row);
            az.push(
                lagrange
                    .iter()
                    .zip(a_first)
                    .map(|(&weight, value)| weight * AkitaField::from_i64(value))
                    .sum(),
            );
            bz.push(
                lagrange
                    .iter()
                    .zip(b_first)
                    .map(|(&weight, value)| weight * value)
                    .sum(),
            );
            az.push(
                lagrange[..9]
                    .iter()
                    .zip(a_second)
                    .map(|(&weight, value)| weight * AkitaField::from_i64(value))
                    .sum(),
            );
            bz.push(
                lagrange[..9]
                    .iter()
                    .zip(b_second)
                    .map(|(&weight, value)| weight * value)
                    .sum(),
            );
        }
        (az, bz)
    }

    fn outer_endpoints(
        az: &[AkitaField],
        bz: &[AkitaField],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> [AkitaField; 2] {
        let mut output = [AkitaField::zero(); 2];
        for (x_out, &outer) in e_out.iter().enumerate() {
            for (x_in, &inner) in e_in.iter().enumerate() {
                let pair = 2 * (x_out * e_in.len() + x_in);
                output[0] += outer * inner * az[pair] * bz[pair];
                output[1] += outer * inner * (az[pair + 1] - az[pair]) * (bz[pair + 1] - bz[pair]);
            }
        }
        output
    }

    fn bind_table(values: &[AkitaField], challenge: AkitaField) -> Vec<AkitaField> {
        values
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect()
    }

    fn outer_opening_values(row: SpartanOuterUniskipRow) -> [AkitaField; 35] {
        let words = row.words();
        let flags = words[19];
        let load = flag(flags, FLAG_LOAD) != 0;
        let store = flag(flags, FLAG_STORE) != 0;
        let ram_address = if load {
            words[10]
        } else if store {
            words[12]
        } else {
            0
        };
        let rs2 = if load { 0 } else { words[10] };
        let rd_write = if store { 0 } else { words[12] };
        let ram_read = if load {
            words[12]
        } else if store {
            words[11]
        } else {
            0
        };
        let ram_write = if load {
            words[12]
        } else if store {
            words[10]
        } else {
            0
        };
        let bit = |bit| AkitaField::from_i64(flag(flags, bit));
        let unsigned128 =
            |low, high| AkitaField::from_u128(u128::from(low) | (u128::from(high) << 64));
        [
            AkitaField::from_u64(words[0]),
            field_signed_magnitude(
                words[1],
                words[2],
                flag(flags, FLAG_RIGHT_INPUT_POSITIVE) != 0,
            ),
            field_signed_magnitude(words[3], words[4], flag(flags, FLAG_PRODUCT_POSITIVE) != 0),
            bit(FLAG_SHOULD_BRANCH),
            AkitaField::from_u64(words[5]),
            AkitaField::from_u64(words[6]),
            field_signed_magnitude(words[7], words[8], flag(flags, FLAG_IMM_POSITIVE) != 0),
            AkitaField::from_u64(ram_address),
            AkitaField::from_u64(words[9]),
            AkitaField::from_u64(rs2),
            AkitaField::from_u64(rd_write),
            AkitaField::from_u64(ram_read),
            AkitaField::from_u64(ram_write),
            AkitaField::from_u64(words[13]),
            unsigned128(words[14], words[15]),
            AkitaField::from_u64(words[16]),
            AkitaField::from_u64(words[17]),
            bit(FLAG_NEXT_VIRTUAL),
            bit(FLAG_NEXT_FIRST),
            AkitaField::from_u64(words[18]),
            bit(FLAG_SHOULD_JUMP),
            bit(FLAG_ADD),
            bit(FLAG_SUB),
            bit(FLAG_MUL),
            bit(FLAG_LOAD),
            bit(FLAG_STORE),
            bit(FLAG_JUMP),
            bit(FLAG_WRITE_LOOKUP),
            bit(FLAG_VIRTUAL),
            bit(FLAG_ASSERT),
            bit(FLAG_DO_NOT_UPDATE),
            bit(FLAG_ADVICE),
            bit(FLAG_COMPRESSED),
            bit(FLAG_IS_FIRST),
            bit(FLAG_IS_LAST),
        ]
    }

    fn outer_openings(
        rows: &[SpartanOuterUniskipRow],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> [AkitaField; 35] {
        let mut output = [AkitaField::zero(); 35];
        for (x_out, &outer) in e_out.iter().enumerate() {
            for (x_in, &inner) in e_in.iter().enumerate() {
                let values = outer_opening_values(rows[x_out * e_in.len() + x_in]);
                for (output, value) in output.iter_mut().zip(values) {
                    *output += outer * inner * value;
                }
            }
        }
        output
    }

    fn product_uniskip_endpoints(
        rows: &[SpartanOuterUniskipRow],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> [AkitaField; 2] {
        let mut output = [AkitaField::zero(); 2];
        for (x_out, &outer) in e_out.iter().enumerate() {
            for (x_in, &inner) in e_in.iter().enumerate() {
                let words = rows[x_out * e_in.len() + x_in].words();
                let flags = words[19];
                let left = AkitaField::from_u64(words[0]);
                let right = field_signed_magnitude(
                    words[1],
                    words[2],
                    flag(flags, FLAG_RIGHT_INPUT_POSITIVE) != 0,
                );
                let lookup = AkitaField::from_u64(words[18]);
                let jump = AkitaField::from_u64(flag(flags, FLAG_JUMP) as u64);
                let branch = AkitaField::from_u64(flag(flags, FLAG_BRANCH) as u64);
                let not_next_noop =
                    AkitaField::one() - AkitaField::from_u64(flag(flags, FLAG_NEXT_IS_NOOP) as u64);
                let weight = outer * inner;
                let three = AkitaField::from_u64(3);
                output[0] += weight
                    * (three * left - three * lookup + jump)
                    * (three * right - three * branch + not_next_noop);
                output[1] += weight
                    * (left - three * lookup + three * jump)
                    * (right - three * branch + three * not_next_noop);
            }
        }
        output
    }

    /// `e_in`, `e_out` weights over `pairs` table pairs, split as the host
    /// splits its eq tables.
    fn round_weights(seed: u64, pairs: usize) -> (Vec<AkitaField>, Vec<AkitaField>) {
        let inner = 1 << (pairs.ilog2() / 2);
        let weights = |seed: u64, len: usize| {
            (0..len)
                .map(|index| {
                    AkitaField::from_u64(splitmix(seed ^ ((index as u64) << 32)) & ((1 << 48) - 1))
                })
                .collect::<Vec<_>>()
        };
        (weights(seed, inner), weights(!seed, pairs / inner))
    }

    fn assert_outer_sequence_matches_field_oracle(
        packed: &[SpartanOuterUniskipRow],
        explicit_rows: usize,
    ) {
        const CPU_TAIL: usize = 4;
        let mut padding_openings = [AkitaField::zero(); 35];
        padding_openings[30] = AkitaField::one();
        assert_eq!(
            outer_opening_values(packed[explicit_rows]),
            padding_openings
        );
        let lagrange = std::array::from_fn(|index| {
            AkitaField::from_u64(splitmix(0x600d_f00d ^ index as u64) & ((1 << 48) - 1))
        });
        let (initial_in, initial_out) = round_weights(1, packed.len());
        let (mut az, mut bz) = outer_tables(packed, &lagrange);
        let expected_first = outer_endpoints(&az, &bz, &initial_in, &initial_out);

        let context = SolinasMetal::for_akita().unwrap();
        let resident = context
            .prepare_spartan_outer_uniskip_rows(packed)
            .unwrap()
            .with_explicit_rows(explicit_rows)
            .unwrap();
        assert_eq!(resident.explicit_rows(), explicit_rows);
        let compact_id = resident.instruction_input_allocation_identity();
        let raw_id = resident.allocation_identity();
        let config = OuterRemainderSequenceConfig {
            max_threadgroups: 2,
            cpu_tail_elements: CPU_TAIL,
            storage_initialization: OuterRemainderStorageInitialization::Lazy,
            product_uniskip_carrier: true,
            ..OuterRemainderSequenceConfig::default()
        };
        let storage = context
            .prepare_outer_remainder_sequence_storage(resident.len(), config)
            .unwrap();
        #[cfg(feature = "test-utils")]
        {
            let initialization = storage.eval_stats().unwrap();
            assert_eq!(initialization.initialized_bytes, 0);
            assert_eq!(initialization.initialization_device_buffers, 0);
            assert_eq!(initialization.initialization_gpu_active, Duration::ZERO);
        }
        let mut sequence = storage.attach(resident).unwrap();
        let storage_before_export = sequence.storage_stats().unwrap();
        assert!(storage_before_export
            .buffer_identities
            .iter()
            .enumerate()
            .all(|(index, identity)| *identity != 0
                && !storage_before_export.buffer_identities[..index].contains(identity)));
        assert_eq!(
            sequence
                .materialize_and_first_message(&lagrange, &initial_in, &initial_out)
                .unwrap(),
            expected_first
        );

        let stream_challenge = AkitaField::from_u64(101);
        az = bind_table(&az, stream_challenge);
        bz = bind_table(&bz, stream_challenge);
        let (stream_in, stream_out) = round_weights(2, az.len() / 2);
        assert_eq!(
            sequence
                .bind_stream_and_message(stream_challenge, &lagrange, &stream_in, &stream_out)
                .unwrap(),
            outer_endpoints(&az, &bz, &stream_in, &stream_out)
        );
        for round in 3u64.. {
            if az.len() == CPU_TAIL {
                break;
            }
            let challenge = AkitaField::from_u64(100 + round);
            az = bind_table(&az, challenge);
            bz = bind_table(&bz, challenge);
            let (e_in, e_out) = round_weights(round, az.len() / 2);
            assert_eq!(
                sequence.bind_and_message(challenge, &e_in, &e_out).unwrap(),
                outer_endpoints(&az, &bz, &e_in, &e_out)
            );
        }

        let mut actual_az = vec![AkitaField::zero(); CPU_TAIL];
        let mut actual_bz = vec![AkitaField::zero(); CPU_TAIL];
        sequence
            .export_cpu_tail(&mut actual_az, &mut actual_bz)
            .unwrap();
        assert_eq!(actual_az, az);
        assert_eq!(actual_bz, bz);
        let storage_after_export = sequence.storage_stats().unwrap();
        assert_eq!(storage_after_export.buffer_identities[..2], [0, 0]);
        assert_eq!(
            storage_before_export.owned_bytes - storage_after_export.owned_bytes,
            (3 * packed.len() * size_of::<Fp128>()) as u64
        );

        let (opening_in, opening_out) = round_weights(0x0be5, packed.len());
        assert_eq!(
            sequence
                .evaluate_openings(&opening_in, &opening_out)
                .unwrap(),
            outer_openings(packed, &opening_in, &opening_out)
        );
        assert_eq!(
            sequence.take_product_uniskip_endpoints(),
            Some(product_uniskip_endpoints(packed, &opening_in, &opening_out))
        );
        let stats = sequence.storage_stats().unwrap();
        assert_eq!(stats.compact_row_identity, compact_id);
        assert_eq!(stats.raw_row_identity, raw_id);
        let compact = sequence.into_instruction_input_rows().unwrap();
        assert_eq!(compact.allocation_identity(), compact_id);
    }

    #[test]
    fn outer_remainder_sequence_matches_field_oracle() {
        let packed = stage1_rows(4);
        assert_outer_sequence_matches_field_oracle(&packed, packed.len() - 1);
    }

    /// The Metal decoder against the witness: the uni-skip and the 35 Outer
    /// openings over resident rows of every expanded instruction kind match
    /// the field oracle over the witness rows.
    #[test]
    fn stage1_rows_cover_every_expanded_instruction() {
        let trace = expanded_trace();
        assert!(trace.sources > 5000, "decoded {} sources", trace.sources);
        assert!(
            trace.rows.len() * 10 > trace.instructions.len() * 9,
            "rows rejected"
        );
        let kinds = trace
            .rows
            .iter()
            .map(|row| row.instruction().instruction_kind)
            .collect::<Vec<_>>();
        let missing = JoltInstructionKind::ALL
            .iter()
            .filter(|kind| !kinds.contains(kind))
            .map(|kind| kind.name())
            .collect::<Vec<_>>();
        assert!(missing.is_empty(), "uncovered kinds: {missing:?}");
        for name in ADVICE_KINDS {
            assert!(
                kinds.iter().any(|kind| kind.name() == name),
                "{name} not covered"
            );
        }

        let log_t = (trace.rows.len() + 1).next_power_of_two().ilog2() as usize;
        let witness = witness::<SpartanOuterRow>(log_t);
        let rows = witness
            .iter()
            .map(SpartanOuterUniskipRow::from_spartan_outer)
            .collect::<Vec<_>>();
        for (index, (row, witness)) in rows.iter().zip(&witness).enumerate() {
            assert_eq!(
                (row.split().0.words()[5] >> FLAG_IS_NOOP) & 1,
                u64::from(witness.is_noop.0),
                "row {index}"
            );
        }
        assert!(witness[0].is_noop.0, "row-0 Noop");
        let explicit_rows = trace.rows.len();
        assert!(!witness[explicit_rows - 1].is_noop.0, "last explicit row");
        assert!(
            witness[explicit_rows..].iter().all(|row| row.is_noop.0),
            "padding"
        );
        let edge = explicit_rows - trace.edge_rows;
        assert_eq!(
            kinds[edge..],
            [
                JoltInstructionKind::AUIPC,
                JoltInstructionKind::VirtualAssertHalfwordAlignment,
                JoltInstructionKind::VirtualAssertWordAlignment,
                JoltInstructionKind::BEQ,
                JoltInstructionKind::MUL,
                JoltInstructionKind::MULHU,
            ]
        );
        let edge_words = |index: usize| rows[edge + index].words();
        // Upper immediates and expanded alignment offsets are 64-bit two's
        // complement: positive, with bit 63 set when negative.
        for index in 0..3 {
            let words = edge_words(index);
            assert!(
                flag(words[19], FLAG_IMM_POSITIVE) != 0 && words[7] >> 63 == 1,
                "{} with a negative 64-bit immediate",
                kinds[edge + index].name()
            );
        }
        let branch = edge_words(3);
        assert!(flag(branch[19], FLAG_BRANCH) != 0);
        assert!(flag(branch[19], FLAG_IMM_POSITIVE) == 0 && branch[7] == 8);
        for index in 4..6 {
            let words = edge_words(index);
            assert!(flag(words[19], FLAG_MUL) != 0);
            assert_eq!([words[3], words[4]], [1, u64::MAX - 1], "maximal product");
        }

        let point = (0..=log_t)
            .map(|index| AkitaField::from_u64(splitmix(index as u64) & ((1 << 48) - 1)))
            .collect::<Vec<_>>();
        let split = point.len() / 2;
        let e_out = EqPolynomial::<AkitaField>::evals(&point[..split], None);
        let e_in = EqPolynomial::<AkitaField>::evals(&point[split..], None);
        let expected = reference(&rows, &e_in, &e_out);
        assert_eq!(
            evaluate_spartan_outer_uniskip_cpu(&rows, &e_in, &e_out).unwrap(),
            expected
        );
        let context = SolinasMetal::for_akita().unwrap();
        let invocation = context
            .prepare_spartan_outer_uniskip(
                &rows,
                &e_in,
                &e_out,
                SpartanOuterUniskipConfig::default(),
            )
            .unwrap();
        invocation.execute().unwrap();
        assert_eq!(invocation.read_output().unwrap(), expected);
        let unwritten = context
            .allocate_spartan_outer_uniskip_rows(rows.len())
            .unwrap();
        let filling = context
            .prepare_spartan_outer_uniskip_with_rows(
                &unwritten,
                &e_in,
                &e_out,
                SpartanOuterUniskipConfig::default(),
            )
            .unwrap();
        let _gpu_active = filling
            .execute_filling(|first_row, instruction_input, raw| {
                for ((source, instruction_input), raw) in
                    rows[first_row..].iter().zip(instruction_input).zip(raw)
                {
                    (*instruction_input, *raw) = source.split();
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(filling.read_output().unwrap(), expected);
        assert_outer_sequence_matches_field_oracle(&rows, explicit_rows);
    }

    #[test]
    fn spartan_outer_uniskip_rejects_shape_mismatch() {
        let context = SolinasMetal::for_akita().unwrap();
        let error = context
            .prepare_spartan_outer_uniskip(
                &[SpartanOuterUniskipRow::default(); 8],
                &[AkitaField::one(); 4],
                &[AkitaField::one(); 3],
                SpartanOuterUniskipConfig::default(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, MetalError::SpartanOuterUniskipShape { .. }));
    }

    #[test]
    fn packed_row_recovers_instruction_input_values() {
        let mut words = [0u64; ROW_WORDS];
        words[6] = 17;
        words[7] = 23;
        words[9] = 29;
        words[10] = 31;
        words[19] = (1 << FLAG_IMM_POSITIVE) | (1 << 20) | (1 << 23);
        let row = SpartanOuterUniskipRow::from_words(words);
        assert_eq!(
            row.instruction_input_fields::<AkitaField>(),
            [
                AkitaField::one(),
                AkitaField::from_u64(29),
                AkitaField::zero(),
                AkitaField::from_u64(17),
                AkitaField::zero(),
                AkitaField::from_u64(31),
                AkitaField::one(),
                AkitaField::from_u64(23),
            ]
        );

        words[10] = 0xfeed_cafe;
        words[19] = (1 << FLAG_LOAD) | (1 << FLAG_IMM_POSITIVE) | (1 << 21) | (1 << 23);
        let load = SpartanOuterUniskipRow::from_words(words);
        assert_eq!(
            load.instruction_input_fields::<AkitaField>()[5],
            AkitaField::zero(),
            "loads canonically have no rs2"
        );
    }
}
