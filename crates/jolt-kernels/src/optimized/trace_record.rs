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
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, TraceRecordRow};
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
        let (rs1_index, rs1_value) = row
            .rs1
            .map_or((NO_REGISTER, 0), |(index, value)| (index, value));
        let (rs2_index, rs2_value) = row
            .rs2
            .map_or((NO_REGISTER, 0), |(index, value)| (index, value));
        let (rd_index, rd_pre, rd_post) = row
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

        self.pc.push(row.pc);
        self.unexpanded_pc.push(row.unexpanded_pc);
        self.imm.push(row.imm);
        self.rs1_value.push(rs1_value);
        self.rs2_value.push(rs2_value);
        self.rd_pre_value.push(rd_pre);
        self.rd_post_value.push(rd_post);
        self.rs1_index.push(rs1_index);
        self.rs2_index.push(rs2_index);
        self.rd_index.push(rd_index);
        self.ram_address.push(row.ram_address);
        self.left_lookup_operand.push(row.left_lookup_operand);
        self.right_lookup_operand.push(row.right_lookup_operand);
        self.left_instruction_input.push(row.left_instruction_input);
        self.right_instruction_input
            .push(row.right_instruction_input);
        self.product_magnitude_lo.push(product_limbs[0]);
        self.product_magnitude_hi.push(product_limbs[1]);
        self.lookup_output.push(row.lookup_output);
        self.flags.push(flags);
        self.addresses
            .push(row.remapped_ram_address.unwrap_or(NO_ACCESS));
        self.pre_values.push(row.ram_read_value);
        self.post_values.push(row.ram_write_value);
        // The stage-5 packed row: `raf_flag` is `InstructionRafFlag`'s
        // formula over the same flag set; `mapped_pc = Some(pc)` because the
        // record's own `pc` extraction (which requires the mapping) succeeded
        // for this row.
        self.instruction_rows.push(InstructionCycleRow::new(
            row.lookup_index,
            row.table_index,
            !row.circuit_flags.is_interleaved_operands(),
            Some(row.pc as usize),
            row.remapped_ram_address,
        ));
    }
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
    /// The session-shared record: collected on first request (stage 1's
    /// Spartan outer kernel today), cloned out as an [`Arc`] afterwards.
    /// Also parks the [`RamAccessColumns`] the walk co-produces, so the RAM
    /// kernels' `shared` finds them without a second pass.
    #[expect(
        clippy::expect_used,
        reason = "the entry is parked by this function right above the read"
    )]
    #[tracing::instrument(skip_all, name = "TraceRecord::collect", fields(cycles = 1usize << log_t))]
    pub(crate) fn shared<F: Field>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Arc<Self>, KernelError<F>> {
        if session.state::<Arc<Self>>().is_none() {
            let cycles = 1usize << log_t;
            let mut consumers = (CollectRecord {
                record: LaneBuffers::with_capacity(cycles),
            },);
            stream_witnesses(witness, 0..cycles, RECORD_CHUNK, &mut consumers)?;
            let lanes = consumers.0.record;
            let ram = Arc::new(RamAccessColumns {
                addresses: lanes.addresses,
                pre_values: lanes.pre_values,
                post_values: lanes.post_values,
            });
            if session.state::<Arc<RamAccessColumns>>().is_none() {
                session.park(Arc::clone(&ram));
            }
            session.park(SharedInstructionRows(Arc::new(lanes.instruction_rows)));
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
            let record = TraceRecord::shared(&mut session, witness, shape.log_t).unwrap();
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
