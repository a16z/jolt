//! The shared trace record: ONE walk over the execution trace materializing
//! compact branch-resolved lanes for every st1-4 trace consumer (and the
//! shared RAM access columns), parked in the [`ProofSession`] at the first
//! consumer's `prepare`.
//!
//! Before this record, each kernel's `prepare` re-walked the 160-byte trace
//! rows through the atomic extractors — 8 `stream_witnesses`/`collect_rows`
//! passes plus 3 increment `oracle_table` walks across stages 1-4, each
//! re-running the per-row instruction decode dozens of times
//! ([`jolt_witness::TraceRecordRow`] pays it once). Consumers read the lanes
//! they need instead; the scalars are bit-identical to the atomic extractors'
//! (pinned in jolt-witness), so every downstream field computation is
//! byte-identical.
//!
//! Memory: the lanes total ~151 B/cycle — strictly below the ~230 B/cycle
//! `Vec<SpartanOuterRow>` they replace at the stage-1 peak. The RAM
//! pre/post/remapped columns live in the session's [`RamAccessColumns`]
//! (built by the same walk, no second RAM pass), which stage 6b's
//! virtualization kernel drops early as before; the record itself is taken
//! by its last consumer (stage 4).
//!
//! The walk also co-produces the two downstream shared scans — the 48 B/cycle
//! [`SharedInstructionRows`] (stage 5's walk today, reused by 6a/6b) and the
//! 8 B/cycle bytecode [`PcRow`] scan (stage 6a's walk) — extending their
//! session lifetime from their first consumer back to stage 1: +56 B/cycle
//! across stages 1-4, well under the ~150 B/cycle the record conversions
//! freed there, and nothing new at the stage-5/6b peaks where they already
//! lived.

use std::sync::Arc;

use jolt_field::signed::S128;
use jolt_field::Field;
use jolt_riscv::{
    CircuitFlagSet, CircuitFlags, InstructionFlagSet, InstructionFlags, InterleavedBitsMarker as _,
};
use jolt_witness::witnesses::{
    Imm, InstructionFlag, LeftInstructionInput, LeftLookupOperand, LookupOutput,
    NextIsFirstInSequence, NextIsNoop, NextIsVirtual, NextPc, NextUnexpandedPc, OpFlag, Pc,
    Product, RamAddress, RamReadValue, RamWriteValue, RdWriteValue, RightInstructionInput,
    RightLookupOperand, Rs1Value, Rs2Value, ShouldBranch, ShouldJump, UnexpandedPc,
};
#[cfg(feature = "parallel")]
use jolt_witness::RandomAccessRows;
use jolt_witness::{stream_witnesses, RowSource, StreamConsumer, TraceRecordRow};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::bytecode_read_raf::{park_pc_rows, PcRow};
use super::instruction_claim_reduction::InstructionOperandRow;
use super::instruction_read_raf::{InstructionCycleRow, SharedInstructionRows};
use super::ram_trace::{RamAccessColumns, NO_ACCESS};
use super::spartan_outer::SpartanOuterRow;
use super::spartan_product::SpartanProductRow;
use crate::{KernelError, ProofSession};

/// [`InstructionFlags::IsNoop`]'s bit inside the packed flags lane.
const IS_NOOP_MASK: u32 = 1 << (INSTRUCTION_FLAGS_SHIFT + InstructionFlags::IsNoop as u32);

/// Register-index lane sentinel for cycles where the operand is absent.
pub(crate) const NO_REGISTER: u8 = u8::MAX;

/// The walk's streaming chunk — matches `support::collect_rows` (chunk size
/// never changes the collected values, only the walk shape).
const RECORD_CHUNK: usize = 1 << 16;

/// Packed per-cycle flags lane: the instruction's two flag sets verbatim
/// plus the record's resolved booleans.
const INSTRUCTION_FLAGS_SHIFT: u32 = 16;
const SHOULD_BRANCH_BIT: u32 = 1 << 24;
const SHOULD_JUMP_BIT: u32 = 1 << 25;
const NEXT_IS_NOOP_BIT: u32 = 1 << 26;
const PRODUCT_POSITIVE_BIT: u32 = 1 << 27;

/// The per-cycle register lanes, in their own [`Arc`] so stage 4's kernel
/// can hold JUST these (~27 B/cycle) while the released record's remaining
/// ~124 B/cycle free BEFORE its sparse-entry reservation — the stage-4
/// coexistence window is the proof's RSS high-water mark otherwise. Absent
/// operands store 0 values and [`NO_REGISTER`] indices.
pub(crate) struct RegisterLanes {
    pub rs1_value: Vec<u64>,
    pub rs2_value: Vec<u64>,
    pub rd_pre_value: Vec<u64>,
    pub rd_post_value: Vec<u64>,
    pub rs1_index: Vec<u8>,
    pub rs2_index: Vec<u8>,
    pub rd_index: Vec<u8>,
}

/// Column-major branch-resolved witness lanes over the padded cycle domain.
/// Lane semantics are [`TraceRecordRow`]'s field semantics, position `t` per
/// cycle.
pub(crate) struct TraceRecord {
    pub pc: Vec<u64>,
    pub unexpanded_pc: Vec<u64>,
    pub imm: Vec<i128>,
    pub registers: Arc<RegisterLanes>,
    /// Raw (unremapped) RAM address; 0 when the cycle makes no access.
    pub ram_address: Vec<u64>,
    pub left_lookup_operand: Vec<u64>,
    pub right_lookup_operand: Vec<u128>,
    pub left_instruction_input: Vec<u64>,
    pub right_instruction_input: Vec<i128>,
    pub product_magnitude_lo: Vec<u64>,
    pub product_magnitude_hi: Vec<u64>,
    pub lookup_output: Vec<u64>,
    pub flags: Vec<u32>,
    /// Packed lookup/bytecode/RAM address rows produced by the same walk.
    /// Stage 0's Metal commit extractor reads this in place; later stages
    /// reclaim the identical allocation through [`SharedInstructionRows`].
    #[cfg(feature = "metal")]
    pub instruction_rows: Arc<Vec<InstructionCycleRow>>,
    /// The shared RAM access columns (remapped address + pre/post values),
    /// built by the same walk and also parked in the session for the RAM
    /// kernels' [`RamAccessColumns::shared`].
    pub ram: Arc<RamAccessColumns>,
}

/// The walk's lane-scattering consumer.
struct CollectRecord {
    record: LaneBuffers,
}

struct LaneBuffers {
    pc: Vec<u64>,
    unexpanded_pc: Vec<u64>,
    imm: Vec<i128>,
    rs1_value: Vec<u64>,
    rs2_value: Vec<u64>,
    rd_pre_value: Vec<u64>,
    rd_post_value: Vec<u64>,
    rs1_index: Vec<u8>,
    rs2_index: Vec<u8>,
    rd_index: Vec<u8>,
    ram_address: Vec<u64>,
    left_lookup_operand: Vec<u64>,
    right_lookup_operand: Vec<u128>,
    left_instruction_input: Vec<u64>,
    right_instruction_input: Vec<i128>,
    product_magnitude_lo: Vec<u64>,
    product_magnitude_hi: Vec<u64>,
    lookup_output: Vec<u64>,
    flags: Vec<u32>,
    addresses: Vec<u64>,
    pre_values: Vec<u64>,
    post_values: Vec<u64>,
    /// The stage-5/6 shared packed rows, co-produced by the same walk
    /// (their extra sources — lookup index, table index — live only here).
    instruction_rows: Vec<InstructionCycleRow>,
}

struct PackedRecordRow {
    pc: u64,
    unexpanded_pc: u64,
    imm: i128,
    rs1_value: u64,
    rs2_value: u64,
    rd_pre_value: u64,
    rd_post_value: u64,
    rs1_index: u8,
    rs2_index: u8,
    rd_index: u8,
    ram_address: u64,
    left_lookup_operand: u64,
    right_lookup_operand: u128,
    left_instruction_input: u64,
    right_instruction_input: i128,
    product_magnitude_lo: u64,
    product_magnitude_hi: u64,
    lookup_output: u64,
    flags: u32,
    address: u64,
    pre_value: u64,
    post_value: u64,
    instruction_row: InstructionCycleRow,
}

impl From<&TraceRecordRow> for PackedRecordRow {
    fn from(row: &TraceRecordRow) -> Self {
        let (rs1_index, rs1_value) = row
            .rs1
            .map_or((NO_REGISTER, 0), |(index, value)| (index, value));
        let (rs2_index, rs2_value) = row
            .rs2
            .map_or((NO_REGISTER, 0), |(index, value)| (index, value));
        let (rd_index, rd_pre_value, rd_post_value) = row
            .rd
            .map_or((NO_REGISTER, 0, 0), |(index, pre, post)| (index, pre, post));
        let product_limbs = row.product.magnitude_limbs();
        let mut flags = u32::from(row.circuit_flags.bits())
            | (u32::from(row.instruction_flags.bits()) << INSTRUCTION_FLAGS_SHIFT);
        if row.should_branch {
            flags |= SHOULD_BRANCH_BIT;
        }
        if row.should_jump {
            flags |= SHOULD_JUMP_BIT;
        }
        if row.next_is_noop {
            flags |= NEXT_IS_NOOP_BIT;
        }
        if row.product.is_positive {
            flags |= PRODUCT_POSITIVE_BIT;
        }

        Self {
            pc: row.pc,
            unexpanded_pc: row.unexpanded_pc,
            imm: row.imm,
            rs1_value,
            rs2_value,
            rd_pre_value,
            rd_post_value,
            rs1_index,
            rs2_index,
            rd_index,
            ram_address: row.ram_address,
            left_lookup_operand: row.left_lookup_operand,
            right_lookup_operand: row.right_lookup_operand,
            left_instruction_input: row.left_instruction_input,
            right_instruction_input: row.right_instruction_input,
            product_magnitude_lo: product_limbs[0],
            product_magnitude_hi: product_limbs[1],
            lookup_output: row.lookup_output,
            flags,
            address: row.remapped_ram_address.unwrap_or(NO_ACCESS),
            pre_value: row.ram_read_value,
            post_value: row.ram_write_value,
            instruction_row: InstructionCycleRow::new(
                row.lookup_index,
                row.table_index,
                !row.circuit_flags.is_interleaved_operands(),
                Some(row.pc as usize),
                row.remapped_ram_address,
            ),
        }
    }
}

impl LaneBuffers {
    fn with_capacity(cycles: usize) -> Self {
        Self {
            pc: Vec::with_capacity(cycles),
            unexpanded_pc: Vec::with_capacity(cycles),
            imm: Vec::with_capacity(cycles),
            rs1_value: Vec::with_capacity(cycles),
            rs2_value: Vec::with_capacity(cycles),
            rd_pre_value: Vec::with_capacity(cycles),
            rd_post_value: Vec::with_capacity(cycles),
            rs1_index: Vec::with_capacity(cycles),
            rs2_index: Vec::with_capacity(cycles),
            rd_index: Vec::with_capacity(cycles),
            ram_address: Vec::with_capacity(cycles),
            left_lookup_operand: Vec::with_capacity(cycles),
            right_lookup_operand: Vec::with_capacity(cycles),
            left_instruction_input: Vec::with_capacity(cycles),
            right_instruction_input: Vec::with_capacity(cycles),
            product_magnitude_lo: Vec::with_capacity(cycles),
            product_magnitude_hi: Vec::with_capacity(cycles),
            lookup_output: Vec::with_capacity(cycles),
            flags: Vec::with_capacity(cycles),
            addresses: Vec::with_capacity(cycles),
            pre_values: Vec::with_capacity(cycles),
            post_values: Vec::with_capacity(cycles),
            instruction_rows: Vec::with_capacity(cycles),
        }
    }

    fn push(&mut self, row: &TraceRecordRow) {
        let row = PackedRecordRow::from(row);
        self.pc.push(row.pc);
        self.unexpanded_pc.push(row.unexpanded_pc);
        self.imm.push(row.imm);
        self.rs1_value.push(row.rs1_value);
        self.rs2_value.push(row.rs2_value);
        self.rd_pre_value.push(row.rd_pre_value);
        self.rd_post_value.push(row.rd_post_value);
        self.rs1_index.push(row.rs1_index);
        self.rs2_index.push(row.rs2_index);
        self.rd_index.push(row.rd_index);
        self.ram_address.push(row.ram_address);
        self.left_lookup_operand.push(row.left_lookup_operand);
        self.right_lookup_operand.push(row.right_lookup_operand);
        self.left_instruction_input.push(row.left_instruction_input);
        self.right_instruction_input
            .push(row.right_instruction_input);
        self.product_magnitude_lo.push(row.product_magnitude_lo);
        self.product_magnitude_hi.push(row.product_magnitude_hi);
        self.lookup_output.push(row.lookup_output);
        self.flags.push(row.flags);
        self.addresses.push(row.address);
        self.pre_values.push(row.pre_value);
        self.post_values.push(row.post_value);
        self.instruction_rows.push(row.instruction_row);
    }

    #[cfg(feature = "parallel")]
    fn collect_parallel(
        access: &RandomAccessRows<'_>,
        cycles: usize,
    ) -> Result<Self, jolt_witness::WitnessError> {
        let empty_instruction = InstructionCycleRow::new(0, None, false, None, None);
        let mut lanes = Self {
            pc: vec![0; cycles],
            unexpanded_pc: vec![0; cycles],
            imm: vec![0; cycles],
            rs1_value: vec![0; cycles],
            rs2_value: vec![0; cycles],
            rd_pre_value: vec![0; cycles],
            rd_post_value: vec![0; cycles],
            rs1_index: vec![NO_REGISTER; cycles],
            rs2_index: vec![NO_REGISTER; cycles],
            rd_index: vec![NO_REGISTER; cycles],
            ram_address: vec![0; cycles],
            left_lookup_operand: vec![0; cycles],
            right_lookup_operand: vec![0; cycles],
            left_instruction_input: vec![0; cycles],
            right_instruction_input: vec![0; cycles],
            product_magnitude_lo: vec![0; cycles],
            product_magnitude_hi: vec![0; cycles],
            lookup_output: vec![0; cycles],
            flags: vec![0; cycles],
            addresses: vec![NO_ACCESS; cycles],
            pre_values: vec![0; cycles],
            post_values: vec![0; cycles],
            instruction_rows: vec![empty_instruction; cycles],
        };
        fill_record_lanes(access, lanes.as_slices_mut(), 0)?;
        Ok(lanes)
    }

    #[cfg(feature = "parallel")]
    fn as_slices_mut(&mut self) -> LaneSlices<'_> {
        LaneSlices {
            pc: &mut self.pc,
            unexpanded_pc: &mut self.unexpanded_pc,
            imm: &mut self.imm,
            rs1_value: &mut self.rs1_value,
            rs2_value: &mut self.rs2_value,
            rd_pre_value: &mut self.rd_pre_value,
            rd_post_value: &mut self.rd_post_value,
            rs1_index: &mut self.rs1_index,
            rs2_index: &mut self.rs2_index,
            rd_index: &mut self.rd_index,
            ram_address: &mut self.ram_address,
            left_lookup_operand: &mut self.left_lookup_operand,
            right_lookup_operand: &mut self.right_lookup_operand,
            left_instruction_input: &mut self.left_instruction_input,
            right_instruction_input: &mut self.right_instruction_input,
            product_magnitude_lo: &mut self.product_magnitude_lo,
            product_magnitude_hi: &mut self.product_magnitude_hi,
            lookup_output: &mut self.lookup_output,
            flags: &mut self.flags,
            addresses: &mut self.addresses,
            pre_values: &mut self.pre_values,
            post_values: &mut self.post_values,
            instruction_rows: &mut self.instruction_rows,
        }
    }
}

#[cfg(feature = "parallel")]
struct LaneSlices<'a> {
    pc: &'a mut [u64],
    unexpanded_pc: &'a mut [u64],
    imm: &'a mut [i128],
    rs1_value: &'a mut [u64],
    rs2_value: &'a mut [u64],
    rd_pre_value: &'a mut [u64],
    rd_post_value: &'a mut [u64],
    rs1_index: &'a mut [u8],
    rs2_index: &'a mut [u8],
    rd_index: &'a mut [u8],
    ram_address: &'a mut [u64],
    left_lookup_operand: &'a mut [u64],
    right_lookup_operand: &'a mut [u128],
    left_instruction_input: &'a mut [u64],
    right_instruction_input: &'a mut [i128],
    product_magnitude_lo: &'a mut [u64],
    product_magnitude_hi: &'a mut [u64],
    lookup_output: &'a mut [u64],
    flags: &'a mut [u32],
    addresses: &'a mut [u64],
    pre_values: &'a mut [u64],
    post_values: &'a mut [u64],
    instruction_rows: &'a mut [InstructionCycleRow],
}

#[cfg(feature = "parallel")]
impl LaneSlices<'_> {
    fn len(&self) -> usize {
        self.pc.len()
    }

    fn split_at_mut(self, mid: usize) -> (Self, Self) {
        let (pc_l, pc_r) = self.pc.split_at_mut(mid);
        let (unexpanded_pc_l, unexpanded_pc_r) = self.unexpanded_pc.split_at_mut(mid);
        let (imm_l, imm_r) = self.imm.split_at_mut(mid);
        let (rs1_value_l, rs1_value_r) = self.rs1_value.split_at_mut(mid);
        let (rs2_value_l, rs2_value_r) = self.rs2_value.split_at_mut(mid);
        let (rd_pre_value_l, rd_pre_value_r) = self.rd_pre_value.split_at_mut(mid);
        let (rd_post_value_l, rd_post_value_r) = self.rd_post_value.split_at_mut(mid);
        let (rs1_index_l, rs1_index_r) = self.rs1_index.split_at_mut(mid);
        let (rs2_index_l, rs2_index_r) = self.rs2_index.split_at_mut(mid);
        let (rd_index_l, rd_index_r) = self.rd_index.split_at_mut(mid);
        let (ram_address_l, ram_address_r) = self.ram_address.split_at_mut(mid);
        let (left_lookup_operand_l, left_lookup_operand_r) =
            self.left_lookup_operand.split_at_mut(mid);
        let (right_lookup_operand_l, right_lookup_operand_r) =
            self.right_lookup_operand.split_at_mut(mid);
        let (left_instruction_input_l, left_instruction_input_r) =
            self.left_instruction_input.split_at_mut(mid);
        let (right_instruction_input_l, right_instruction_input_r) =
            self.right_instruction_input.split_at_mut(mid);
        let (product_magnitude_lo_l, product_magnitude_lo_r) =
            self.product_magnitude_lo.split_at_mut(mid);
        let (product_magnitude_hi_l, product_magnitude_hi_r) =
            self.product_magnitude_hi.split_at_mut(mid);
        let (lookup_output_l, lookup_output_r) = self.lookup_output.split_at_mut(mid);
        let (flags_l, flags_r) = self.flags.split_at_mut(mid);
        let (addresses_l, addresses_r) = self.addresses.split_at_mut(mid);
        let (pre_values_l, pre_values_r) = self.pre_values.split_at_mut(mid);
        let (post_values_l, post_values_r) = self.post_values.split_at_mut(mid);
        let (instruction_rows_l, instruction_rows_r) = self.instruction_rows.split_at_mut(mid);
        (
            Self {
                pc: pc_l,
                unexpanded_pc: unexpanded_pc_l,
                imm: imm_l,
                rs1_value: rs1_value_l,
                rs2_value: rs2_value_l,
                rd_pre_value: rd_pre_value_l,
                rd_post_value: rd_post_value_l,
                rs1_index: rs1_index_l,
                rs2_index: rs2_index_l,
                rd_index: rd_index_l,
                ram_address: ram_address_l,
                left_lookup_operand: left_lookup_operand_l,
                right_lookup_operand: right_lookup_operand_l,
                left_instruction_input: left_instruction_input_l,
                right_instruction_input: right_instruction_input_l,
                product_magnitude_lo: product_magnitude_lo_l,
                product_magnitude_hi: product_magnitude_hi_l,
                lookup_output: lookup_output_l,
                flags: flags_l,
                addresses: addresses_l,
                pre_values: pre_values_l,
                post_values: post_values_l,
                instruction_rows: instruction_rows_l,
            },
            Self {
                pc: pc_r,
                unexpanded_pc: unexpanded_pc_r,
                imm: imm_r,
                rs1_value: rs1_value_r,
                rs2_value: rs2_value_r,
                rd_pre_value: rd_pre_value_r,
                rd_post_value: rd_post_value_r,
                rs1_index: rs1_index_r,
                rs2_index: rs2_index_r,
                rd_index: rd_index_r,
                ram_address: ram_address_r,
                left_lookup_operand: left_lookup_operand_r,
                right_lookup_operand: right_lookup_operand_r,
                left_instruction_input: left_instruction_input_r,
                right_instruction_input: right_instruction_input_r,
                product_magnitude_lo: product_magnitude_lo_r,
                product_magnitude_hi: product_magnitude_hi_r,
                lookup_output: lookup_output_r,
                flags: flags_r,
                addresses: addresses_r,
                pre_values: pre_values_r,
                post_values: post_values_r,
                instruction_rows: instruction_rows_r,
            },
        )
    }

    fn write(&mut self, index: usize, row: &TraceRecordRow) {
        let row = PackedRecordRow::from(row);
        self.pc[index] = row.pc;
        self.unexpanded_pc[index] = row.unexpanded_pc;
        self.imm[index] = row.imm;
        self.rs1_value[index] = row.rs1_value;
        self.rs2_value[index] = row.rs2_value;
        self.rd_pre_value[index] = row.rd_pre_value;
        self.rd_post_value[index] = row.rd_post_value;
        self.rs1_index[index] = row.rs1_index;
        self.rs2_index[index] = row.rs2_index;
        self.rd_index[index] = row.rd_index;
        self.ram_address[index] = row.ram_address;
        self.left_lookup_operand[index] = row.left_lookup_operand;
        self.right_lookup_operand[index] = row.right_lookup_operand;
        self.left_instruction_input[index] = row.left_instruction_input;
        self.right_instruction_input[index] = row.right_instruction_input;
        self.product_magnitude_lo[index] = row.product_magnitude_lo;
        self.product_magnitude_hi[index] = row.product_magnitude_hi;
        self.lookup_output[index] = row.lookup_output;
        self.flags[index] = row.flags;
        self.addresses[index] = row.address;
        self.pre_values[index] = row.pre_value;
        self.post_values[index] = row.post_value;
        self.instruction_rows[index] = row.instruction_row;
    }
}

#[cfg(feature = "parallel")]
fn fill_record_lanes(
    access: &RandomAccessRows<'_>,
    mut lanes: LaneSlices<'_>,
    base: usize,
) -> Result<(), jolt_witness::WitnessError> {
    const GRAIN: usize = 1 << 12;
    if lanes.len() <= GRAIN {
        for index in 0..lanes.len() {
            let row = access.bundle_at::<TraceRecordRow>(base + index)?;
            lanes.write(index, &row);
        }
        return Ok(());
    }
    let mid = lanes.len() / 2;
    let (left, right) = lanes.split_at_mut(mid);
    let (left_result, right_result) = rayon::join(
        || fill_record_lanes(access, left, base),
        || fill_record_lanes(access, right, base + mid),
    );
    left_result?;
    right_result
}

impl StreamConsumer for CollectRecord {
    type Witness = TraceRecordRow;

    fn consume(&mut self, chunk: &[TraceRecordRow]) {
        for row in chunk {
            self.record.push(row);
        }
    }
}

impl TraceRecord {
    /// The session-shared record: collected on first request (stage 0's
    /// Metal commitment extractor or stage 1's Spartan outer kernel), cloned
    /// out as an [`Arc`] afterwards.
    /// Also parks the [`RamAccessColumns`] the walk co-produces, so the RAM
    /// kernels' `shared` finds them without a second pass.
    #[expect(
        clippy::expect_used,
        reason = "the entry is parked by this function right above the read"
    )]
    #[tracing::instrument(skip_all, name = "TraceRecord::collect", fields(cycles = 1usize << log_t))]
    pub(crate) fn shared<F: Field>(
        session: &mut ProofSession,
        source: &dyn RowSource,
        log_t: usize,
    ) -> Result<Arc<Self>, KernelError<F>> {
        if session.state::<Arc<Self>>().is_none() {
            let cycles = 1usize << log_t;
            let collect_streaming = || {
                let mut consumers = (CollectRecord {
                    record: LaneBuffers::with_capacity(cycles),
                },);
                stream_witnesses(source, 0..cycles, RECORD_CHUNK, &mut consumers)?;
                Ok::<_, jolt_witness::WitnessError>(consumers.0.record)
            };
            #[cfg(feature = "parallel")]
            // The record is the shared SoA substrate for stages 1-6. Fill
            // its final lanes directly; staging chunk-local AoS rows made
            // the serial lane scatter the collection wall.
            let lanes = match source
                .random_access()
                .filter(|access| cycles <= access.cycles)
            {
                Some(access) => LaneBuffers::collect_parallel(&access, cycles)?,
                None => collect_streaming()?,
            };
            #[cfg(not(feature = "parallel"))]
            let lanes = collect_streaming()?;
            let ram = Arc::new(RamAccessColumns {
                addresses: lanes.addresses,
                pre_values: lanes.pre_values,
                post_values: lanes.post_values,
            });
            if session.state::<Arc<RamAccessColumns>>().is_none() {
                session.park(Arc::clone(&ram));
            }
            let instruction_rows = Arc::new(lanes.instruction_rows);
            session.park(SharedInstructionRows(Arc::clone(&instruction_rows)));
            // The bytecode PC scan, packed from the pc/noop lanes (fallible
            // u32-range guards, so it runs after the walk).
            let pack_pc =
                |(&pc, &flags): (&u64, &u32)| PcRow::from_lanes::<F>(pc, flags & IS_NOOP_MASK != 0);
            #[cfg(feature = "parallel")]
            let pc_rows: Vec<PcRow> = lanes
                .pc
                .par_iter()
                .zip(&lanes.flags)
                .map(pack_pc)
                .collect::<Result<_, _>>()?;
            #[cfg(not(feature = "parallel"))]
            let pc_rows: Vec<PcRow> = lanes
                .pc
                .iter()
                .zip(&lanes.flags)
                .map(pack_pc)
                .collect::<Result<_, _>>()?;
            park_pc_rows(session, pc_rows);
            let record = Arc::new(Self {
                pc: lanes.pc,
                unexpanded_pc: lanes.unexpanded_pc,
                imm: lanes.imm,
                registers: Arc::new(RegisterLanes {
                    rs1_value: lanes.rs1_value,
                    rs2_value: lanes.rs2_value,
                    rd_pre_value: lanes.rd_pre_value,
                    rd_post_value: lanes.rd_post_value,
                    rs1_index: lanes.rs1_index,
                    rs2_index: lanes.rs2_index,
                    rd_index: lanes.rd_index,
                }),
                ram_address: lanes.ram_address,
                left_lookup_operand: lanes.left_lookup_operand,
                right_lookup_operand: lanes.right_lookup_operand,
                left_instruction_input: lanes.left_instruction_input,
                right_instruction_input: lanes.right_instruction_input,
                product_magnitude_lo: lanes.product_magnitude_lo,
                product_magnitude_hi: lanes.product_magnitude_hi,
                lookup_output: lanes.lookup_output,
                flags: lanes.flags,
                #[cfg(feature = "metal")]
                instruction_rows,
                ram,
            });
            session.park(record);
        }
        Ok(Arc::clone(
            session
                .state::<Arc<Self>>()
                .expect("trace record parked above"),
        ))
    }

    /// Drop the session's record — called by the LAST record consumer's
    /// `prepare` (stage 4's registers read-write checking today) so the
    /// lanes free before the stage-5 peak. The RAM access columns survive
    /// under their own session `Arc` for stages 4-6b, exactly as before.
    pub(crate) fn release(session: &mut ProofSession) {
        let _ = session.take::<Arc<Self>>();
    }

    pub(crate) fn len(&self) -> usize {
        self.pc.len()
    }

    #[inline]
    pub(crate) fn circuit_flags(&self, t: usize) -> CircuitFlagSet {
        CircuitFlagSet::from_bits(self.flags[t] as u16)
    }

    #[inline]
    pub(crate) fn circuit_flag(&self, t: usize, flag: CircuitFlags) -> bool {
        self.circuit_flags(t).get(flag)
    }

    #[inline]
    pub(crate) fn instruction_flag(&self, t: usize, flag: InstructionFlags) -> bool {
        InstructionFlagSet::from_bits((self.flags[t] >> INSTRUCTION_FLAGS_SHIFT) as u8).get(flag)
    }

    /// `flag`'s bit position inside the packed [`Self::flags`] lane — for
    /// consumers (the Metal instruction-input slot) that read the lane raw.
    #[cfg(feature = "metal")]
    #[inline]
    pub(crate) fn instruction_flag_bit(flag: InstructionFlags) -> u32 {
        INSTRUCTION_FLAGS_SHIFT + flag as u32
    }

    #[inline]
    pub(crate) fn should_branch(&self, t: usize) -> bool {
        self.flags[t] & SHOULD_BRANCH_BIT != 0
    }

    #[inline]
    pub(crate) fn should_jump(&self, t: usize) -> bool {
        self.flags[t] & SHOULD_JUMP_BIT != 0
    }

    #[inline]
    pub(crate) fn next_is_noop(&self, t: usize) -> bool {
        self.flags[t] & NEXT_IS_NOOP_BIT != 0
    }

    #[inline]
    pub(crate) fn product(&self, t: usize) -> S128 {
        S128::new(
            [self.product_magnitude_lo[t], self.product_magnitude_hi[t]],
            self.flags[t] & PRODUCT_POSITIVE_BIT != 0,
        )
    }
}

/// A per-cycle typed bundle reconstructible from the record's lanes. Views
/// rebuild the exact bundle the kernel's own `collect_rows` walk would have
/// extracted — same scalars, so downstream computation is byte-identical.
pub(crate) trait RecordView: Copy {
    fn from_record(record: &TraceRecord, t: usize) -> Self;
}

/// Row access for kernels that keep typed rows across their sumcheck:
/// directly collected bundles (the parity tests' seam — their synthetic rows
/// encode `Next*` values no shifted lane read could — and the post-use empty
/// state) or the session-shared record's lanes (production).
pub(crate) enum RecordRows<R> {
    Collected(Vec<R>),
    Record(Arc<TraceRecord>),
}

impl<R: RecordView> RecordRows<R> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Collected(rows) => rows.len(),
            Self::Record(record) => record.len(),
        }
    }

    #[inline]
    pub(crate) fn row(&self, t: usize) -> R {
        match self {
            Self::Collected(rows) => rows[t],
            Self::Record(record) => R::from_record(record, t),
        }
    }
}

/// The stage-1 Spartan outer bundle: the `Next*` fields follow the
/// extractors' lookahead semantics exactly — the successor row's own lanes,
/// zero/false at `t = T - 1`.
impl RecordView for SpartanOuterRow {
    #[inline]
    fn from_record(record: &TraceRecord, t: usize) -> Self {
        let next = t + 1 < record.len();
        let circuit_flags = record.circuit_flags(t);
        let flag = |flag: CircuitFlags| OpFlag(circuit_flags.get(flag));
        SpartanOuterRow {
            left_instruction_input: LeftInstructionInput(record.left_instruction_input[t]),
            right_instruction_input: RightInstructionInput(record.right_instruction_input[t]),
            product: Product(record.product(t)),
            should_branch: ShouldBranch(record.should_branch(t)),
            pc: Pc(record.pc[t]),
            unexpanded_pc: UnexpandedPc(record.unexpanded_pc[t]),
            imm: Imm(record.imm[t]),
            ram_address: RamAddress(record.ram_address[t]),
            rs1_value: Rs1Value(record.registers.rs1_value[t]),
            rs2_value: Rs2Value(record.registers.rs2_value[t]),
            rd_write_value: RdWriteValue(record.registers.rd_post_value[t]),
            ram_read_value: RamReadValue(record.ram.pre_values[t]),
            ram_write_value: RamWriteValue(record.ram.post_values[t]),
            left_lookup_operand: LeftLookupOperand(record.left_lookup_operand[t]),
            right_lookup_operand: RightLookupOperand(record.right_lookup_operand[t]),
            next_unexpanded_pc: NextUnexpandedPc(if next {
                record.unexpanded_pc[t + 1]
            } else {
                0
            }),
            next_pc: NextPc(if next { record.pc[t + 1] } else { 0 }),
            next_is_virtual: NextIsVirtual(
                next && record.circuit_flag(t + 1, CircuitFlags::VirtualInstruction),
            ),
            next_is_first_in_sequence: NextIsFirstInSequence(
                next && record.circuit_flag(t + 1, CircuitFlags::IsFirstInSequence),
            ),
            lookup_output: LookupOutput(record.lookup_output[t]),
            should_jump: ShouldJump(record.should_jump(t)),
            add_operands: flag(CircuitFlags::AddOperands),
            subtract_operands: flag(CircuitFlags::SubtractOperands),
            multiply_operands: flag(CircuitFlags::MultiplyOperands),
            load: flag(CircuitFlags::Load),
            store: flag(CircuitFlags::Store),
            jump: flag(CircuitFlags::Jump),
            write_lookup_output_to_rd: flag(CircuitFlags::WriteLookupOutputToRD),
            virtual_instruction: flag(CircuitFlags::VirtualInstruction),
            assert_flag: flag(CircuitFlags::Assert),
            do_not_update_unexpanded_pc: flag(CircuitFlags::DoNotUpdateUnexpandedPC),
            advice: flag(CircuitFlags::Advice),
            is_compressed: flag(CircuitFlags::IsCompressed),
            is_first_in_sequence: flag(CircuitFlags::IsFirstInSequence),
            is_last_in_sequence: flag(CircuitFlags::IsLastInSequence),
        }
    }
}

/// The stage-2 product-virtualization bundle ([`NextIsNoop`]'s
/// missing-successor-is-noop convention is stored, not shifted).
impl RecordView for SpartanProductRow {
    #[inline]
    fn from_record(record: &TraceRecord, t: usize) -> Self {
        SpartanProductRow {
            left_instruction_input: LeftInstructionInput(record.left_instruction_input[t]),
            right_instruction_input: RightInstructionInput(record.right_instruction_input[t]),
            jump_flag: OpFlag(record.circuit_flag(t, CircuitFlags::Jump)),
            write_lookup_output_to_rd: OpFlag(
                record.circuit_flag(t, CircuitFlags::WriteLookupOutputToRD),
            ),
            lookup_output: LookupOutput(record.lookup_output[t]),
            branch_flag: InstructionFlag(record.instruction_flag(t, InstructionFlags::Branch)),
            next_is_noop: NextIsNoop(record.next_is_noop(t)),
            virtual_instruction: OpFlag(record.circuit_flag(t, CircuitFlags::VirtualInstruction)),
        }
    }
}

/// The stage-2 instruction claim-reduction bundle.
impl RecordView for InstructionOperandRow {
    #[inline]
    fn from_record(record: &TraceRecord, t: usize) -> Self {
        InstructionOperandRow {
            lookup_output: LookupOutput(record.lookup_output[t]),
            left_lookup_operand: LeftLookupOperand(record.left_lookup_operand[t]),
            right_lookup_operand: RightLookupOperand(record.right_lookup_operand[t]),
            left_instruction_input: LeftInstructionInput(record.left_instruction_input[t]),
            right_instruction_input: RightInstructionInput(record.right_instruction_input[t]),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::super::registers_read_write::RegisterCycleRow;
    use super::super::support::collect_rows;
    use super::super::testing::{with_ram_fixture, FixtureShape, RamOp};
    use super::*;
    use crate::ProofSession;

    /// The record's `outer_row` view is indistinguishable from the collected
    /// bundle walk, padding rows and the `T - 1` lookahead edge included; the
    /// co-produced RAM columns match the RAM kernels' own walk.
    #[test]
    fn record_lockstep_with_collected_bundles() {
        let shape = FixtureShape { log_t: 4, ram_k: 8 };
        let ops = vec![
            RamOp::Write { word: 3, post: 7 },
            RamOp::Read { word: 3 },
            RamOp::None,
            RamOp::Write { word: 2, post: 9 },
            RamOp::Read { word: 2 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let mut session = ProofSession::default();
            let record =
                TraceRecord::shared::<jolt_field::Fr>(&mut session, witness, shape.log_t).unwrap();
            let rows: Vec<SpartanOuterRow> = collect_rows(witness, 1 << shape.log_t).unwrap();
            assert_eq!(record.len(), rows.len());
            for (t, row) in rows.iter().enumerate() {
                assert_eq!(&SpartanOuterRow::from_record(&record, t), row, "cycle {t}");
            }

            let product_rows: Vec<SpartanProductRow> =
                collect_rows(witness, 1 << shape.log_t).unwrap();
            for (t, row) in product_rows.iter().enumerate() {
                assert_eq!(
                    &SpartanProductRow::from_record(&record, t),
                    row,
                    "cycle {t}"
                );
            }
            let operand_rows: Vec<InstructionOperandRow> =
                collect_rows(witness, 1 << shape.log_t).unwrap();
            for (t, row) in operand_rows.iter().enumerate() {
                assert_eq!(
                    &InstructionOperandRow::from_record(&record, t),
                    row,
                    "cycle {t}"
                );
            }
            let inc: Vec<jolt_field::Fr> = record.ram.inc_column();
            let oracle_inc: Vec<jolt_field::Fr> = witness
                .oracle_table(
                    jolt_claims::protocols::jolt::geometry::ram::ram_inc().polynomial_id(),
                )
                .unwrap();
            assert_eq!(inc, oracle_inc);

            // The co-produced stage-5 rows and bytecode PC scan match their
            // own walks' packing (fresh sessions force the collect paths).
            let walked = super::super::instruction_read_raf::collect_instruction_cycle_rows::<
                jolt_field::Fr,
            >(witness, 1 << shape.log_t)
            .unwrap();
            let parked = super::super::instruction_read_raf::shared_instruction_rows::<
                jolt_field::Fr,
            >(&mut session, witness, 1 << shape.log_t)
            .unwrap();
            assert_eq!(parked.len(), walked.len());
            for (t, (packed, fresh)) in parked.iter().zip(&walked).enumerate() {
                assert_eq!(packed.lookup_index, fresh.lookup_index, "cycle {t}");
                assert_eq!(packed.table_index(), fresh.table_index(), "cycle {t}");
                assert_eq!(packed.raf_flag, fresh.raf_flag, "cycle {t}");
                assert_eq!(packed.mapped_pc(), fresh.mapped_pc(), "cycle {t}");
                assert_eq!(
                    packed.remapped_ram_address(),
                    fresh.remapped_ram_address(),
                    "cycle {t}"
                );
            }

            let parked_pc = super::super::bytecode_read_raf::pc_rows::<jolt_field::Fr>(
                &mut session,
                witness,
                1 << shape.log_t,
            )
            .unwrap();
            let mut fresh_session = ProofSession::default();
            let walked_pc = super::super::bytecode_read_raf::pc_rows::<jolt_field::Fr>(
                &mut fresh_session,
                witness,
                1 << shape.log_t,
            )
            .unwrap();
            assert_eq!(*parked_pc, *walked_pc);

            let register_rows: Vec<RegisterCycleRow> =
                collect_rows(witness, 1 << shape.log_t).unwrap();
            for (t, row) in register_rows.iter().enumerate() {
                let (rs1_index, rs1_value) = row.rs1.map_or((NO_REGISTER, 0), |(i, v)| (i, v));
                let (rs2_index, rs2_value) = row.rs2.map_or((NO_REGISTER, 0), |(i, v)| (i, v));
                let (rd_index, rd_pre, rd_post) = row
                    .rd
                    .map_or((NO_REGISTER, 0, 0), |(i, pre, post)| (i, pre, post));
                assert_eq!(record.registers.rs1_index[t], rs1_index, "cycle {t}");
                assert_eq!(record.registers.rs1_value[t], rs1_value, "cycle {t}");
                assert_eq!(record.registers.rs2_index[t], rs2_index, "cycle {t}");
                assert_eq!(record.registers.rs2_value[t], rs2_value, "cycle {t}");
                assert_eq!(record.registers.rd_index[t], rd_index, "cycle {t}");
                assert_eq!(record.registers.rd_pre_value[t], rd_pre, "cycle {t}");
                assert_eq!(record.registers.rd_post_value[t], rd_post, "cycle {t}");
            }

            let mut fresh = ProofSession::default();
            let columns = RamAccessColumns::shared(&mut fresh, witness, shape.log_t).unwrap();
            assert_eq!(record.ram.addresses, columns.addresses);
            assert_eq!(record.ram.pre_values, columns.pre_values);
            assert_eq!(record.ram.post_values, columns.post_values);

            // The walk parked the RAM columns: a same-session `shared` reuses
            // them (pointer identity) instead of re-walking.
            let parked = RamAccessColumns::shared(&mut session, witness, shape.log_t).unwrap();
            assert!(Arc::ptr_eq(&parked, &record.ram));
        });
    }
}
