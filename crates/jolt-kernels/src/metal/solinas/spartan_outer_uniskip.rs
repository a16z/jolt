use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::signed::{S192, S256, S64};
use jolt_field::{AkitaField, SignedProductAccumulator as _, WithSignedProductAccumulator};
use jolt_witness::witnesses::SpartanOuterRow;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

pub const SPARTAN_OUTER_EXTENDED_NODES: usize = 9;
const ROW_WORDS: usize = 20;
const SIMD_WIDTH: usize = 32;
const BLOCKS_PIPELINE: &str = "solinas_spartan_outer_uniskip_blocks";
const REDUCE_PIPELINE: &str = "solinas_spartan_outer_uniskip_reduce";

const EXTENSION_COEFFICIENTS: [[i64; 10]; SPARTAN_OUTER_EXTENDED_NODES] = [
    [
        2002, -15015, 51480, -105105, 140140, -126126, 76440, -30030, 6930, -715,
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
const FLAG_VIRTUAL: u32 = 9;
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

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpartanOuterUniskipRow {
    words: [u64; ROW_WORDS],
}

#[derive(Clone)]
pub struct SpartanOuterUniskipRows {
    buffer: Buffer,
    len: usize,
}

impl SpartanOuterUniskipRows {
    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for SpartanOuterUniskipRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_rows"),
            self.len * size_of::<SpartanOuterUniskipRow>(),
        );
        visitor.exit();
    }
}

impl SpartanOuterUniskipRow {
    pub const fn from_words(words: [u64; ROW_WORDS]) -> Self {
        Self { words }
    }

    pub const fn words(self) -> [u64; ROW_WORDS] {
        self.words
    }

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
        let mut accumulators: [
            <AkitaField as WithSignedProductAccumulator>::SignedProductAccumulator;
            SPARTAN_OUTER_EXTENDED_NODES
        ] = Default::default();
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
    reserved: u32,
}

struct Buffers {
    rows: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    block_sums: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct SpartanOuterUniskipInvocation<'a> {
    context: &'a SolinasMetal,
    blocks_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    blocks_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    buffers: Buffers,
    blocks: usize,
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
        self.prepare_spartan_outer_uniskip_rows_with_fill(rows.len(), |destination| {
            destination.copy_from_slice(rows);
            Ok(())
        })
    }

    pub(crate) fn prepare_spartan_outer_uniskip_rows_with_fill(
        &self,
        rows: usize,
        fill: impl FnOnce(&mut [SpartanOuterUniskipRow]) -> Result<(), MetalError>,
    ) -> Result<SpartanOuterUniskipRows, MetalError> {
        if rows == 0 {
            return Err(MetalError::EmptyInput);
        }
        let row_bytes = byte_length::<SpartanOuterUniskipRow>(rows)?;
        self.validate_buffer_length(row_bytes)?;
        let rows_buffer = self
            .device
            .new_buffer(row_bytes, MTLResourceOptions::StorageModeShared);
        // SAFETY: the shared buffer has exactly `rows` elements and no command
        // buffer can observe it until `fill` returns and the invocation executes.
        let destination = unsafe {
            slice::from_raw_parts_mut(
                rows_buffer.contents().cast::<SpartanOuterUniskipRow>(),
                rows,
            )
        };
        fill(destination)?;
        Ok(SpartanOuterUniskipRows {
            buffer: rows_buffer,
            len: rows,
        })
    }

    pub fn prepare_spartan_outer_uniskip_with_rows(
        &self,
        rows: &SpartanOuterUniskipRows,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: SpartanOuterUniskipConfig,
    ) -> Result<SpartanOuterUniskipInvocation<'_>, MetalError> {
        self.prepare_spartan_outer_uniskip_from_buffer(
            rows.buffer.to_owned(),
            rows.len,
            e_in,
            e_out,
            config,
        )
    }

    fn prepare_spartan_outer_uniskip_from_buffer(
        &self,
        rows_buffer: Buffer,
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
        let block_bytes = byte_length::<Fp128>(block_elements)?;
        let output_bytes = byte_length::<Fp128>(SPARTAN_OUTER_EXTENDED_NODES)?;
        for requested in [
            byte_length::<Fp128>(e_in_fp.len())?,
            byte_length::<Fp128>(e_out_fp.len())?,
            block_bytes,
            output_bytes,
        ] {
            self.validate_buffer_length(requested)?;
        }
        let params = Params {
            rows: u32::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?,
            pairs_per_block: u32::try_from(pairs_per_block)
                .map_err(|_| MetalError::InputTooLong(pairs_per_block))?,
            blocks: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            reserved: 0,
        };

        Ok(SpartanOuterUniskipInvocation {
            context: self,
            blocks_pipeline,
            reduce_pipeline,
            blocks_limits,
            reduce_limits,
            buffers: Buffers {
                rows: rows_buffer,
                e_in: buffer_from_slice(&self.device, &e_in_fp),
                e_out: buffer_from_slice(&self.device, &e_out_fp),
                block_sums: self
                    .device
                    .new_buffer(block_bytes, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared),
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
            },
            blocks: e_out.len(),
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }
}

impl SpartanOuterUniskipInvocation<'_> {
    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.blocks_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduce_limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn block_count(&self) -> usize {
        self.blocks
    }

    pub const fn execution_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let blocks = command_buffer.new_compute_command_encoder();
            blocks.set_compute_pipeline_state(&self.blocks_pipeline);
            blocks.set_buffer(0, Some(&self.buffers.rows), 0);
            blocks.set_buffer(1, Some(&self.buffers.e_in), 0);
            blocks.set_buffer(2, Some(&self.buffers.e_out), 0);
            blocks.set_buffer(3, Some(&self.buffers.block_sums), 0);
            blocks.set_buffer(4, Some(&self.buffers.params), 0);
            blocks.set_threadgroup_memory_length(
                0,
                byte_length::<Fp128>(self.threads_per_threadgroup)?,
            );
            blocks.dispatch_thread_groups(
                MTLSize {
                    width: self.blocks as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            blocks.end_encoding();

            let reduce = command_buffer.new_compute_command_encoder();
            reduce.set_compute_pipeline_state(&self.reduce_pipeline);
            reduce.set_buffer(0, Some(&self.buffers.block_sums), 0);
            reduce.set_buffer(1, Some(&self.buffers.output), 0);
            reduce.set_buffer(2, Some(&self.buffers.params), 0);
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

            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
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

const _: () = assert!(size_of::<SpartanOuterUniskipRow>() == 160);
const _: () = assert!(size_of::<Params>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::FromPrimitiveInt;
    use jolt_poly::EqPolynomial;

    use super::*;

    fn splitmix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn rows(count: usize) -> Vec<SpartanOuterUniskipRow> {
        (0..count)
            .map(|index| {
                let mut words = [0u64; ROW_WORDS];
                for (word, value) in words[..19].iter_mut().enumerate() {
                    *value = splitmix(index as u64 ^ (word as u64).wrapping_mul(0x1000_0001));
                }
                words[2] &= (1 << 24) - 1;
                words[4] &= (1 << 24) - 1;
                words[8] = 0;
                words[15] &= (1 << 24) - 1;
                let selector = splitmix(index as u64 ^ 0xa5a5_5a5a);
                let mut flags = 0u64;
                match selector % 3 {
                    1 => flags |= 1 << FLAG_LOAD,
                    2 => flags |= 1 << FLAG_STORE,
                    _ => {}
                }
                match (selector >> 2) % 4 {
                    1 => flags |= 1 << FLAG_ADD,
                    2 => flags |= 1 << FLAG_SUB,
                    3 => flags |= 1 << FLAG_MUL,
                    _ => {}
                }
                for bit in FLAG_JUMP..=FLAG_COMPRESSED {
                    flags |= ((selector >> (bit + 7)) & 1) << bit;
                }
                flags |= ((selector >> 40) & 1) << FLAG_RIGHT_INPUT_POSITIVE;
                flags |= ((selector >> 41) & 1) << FLAG_IMM_POSITIVE;
                flags |= ((selector >> 42) & 1) << FLAG_PRODUCT_POSITIVE;
                words[19] = flags;
                SpartanOuterUniskipRow::from_words(words)
            })
            .collect()
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

    #[test]
    fn spartan_outer_uniskip_matches_field_oracle() {
        let rows = rows(1 << 10);
        let point = (0..11)
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
    }

    #[test]
    fn spartan_outer_uniskip_rejects_shape_mismatch() {
        let context = SolinasMetal::for_akita().unwrap();
        let error = context
            .prepare_spartan_outer_uniskip(
                &rows(8),
                &[AkitaField::one(); 4],
                &[AkitaField::one(); 3],
                SpartanOuterUniskipConfig::default(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, MetalError::SpartanOuterUniskipShape { .. }));
    }
}
