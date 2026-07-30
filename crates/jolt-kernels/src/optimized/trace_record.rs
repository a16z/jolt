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
//! by its last consumer.

use std::sync::Arc;

use jolt_field::signed::S128;
use jolt_field::Field;
use jolt_riscv::{CircuitFlagSet, CircuitFlags};
use jolt_witness::witnesses::{
    Imm, LeftInstructionInput, LeftLookupOperand, LookupOutput, NextIsFirstInSequence,
    NextIsVirtual, NextPc, NextUnexpandedPc, OpFlag, Pc, Product, RamAddress, RamReadValue,
    RamWriteValue, RdWriteValue, RightInstructionInput, RightLookupOperand, Rs1Value, Rs2Value,
    ShouldBranch, ShouldJump, UnexpandedPc,
};
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, TraceRecordRow};

use super::ram_trace::{RamAccessColumns, NO_ACCESS};
use super::spartan_outer::SpartanOuterRow;
use crate::{KernelError, ProofSession};

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

/// Column-major branch-resolved witness lanes over the padded cycle domain.
/// Lane semantics are [`TraceRecordRow`]'s field semantics, position `t` per
/// cycle; absent register operands store 0 values and [`NO_REGISTER`]
/// indices.
pub(crate) struct TraceRecord {
    pub pc: Vec<u64>,
    pub unexpanded_pc: Vec<u64>,
    pub imm: Vec<i128>,
    pub rs1_value: Vec<u64>,
    pub rs2_value: Vec<u64>,
    #[cfg_attr(not(test), expect(dead_code, reason = "stage-4 consumers land next"))]
    pub rd_pre_value: Vec<u64>,
    pub rd_post_value: Vec<u64>,
    #[cfg_attr(not(test), expect(dead_code, reason = "stage-4 consumers land next"))]
    pub rs1_index: Vec<u8>,
    #[cfg_attr(not(test), expect(dead_code, reason = "stage-4 consumers land next"))]
    pub rs2_index: Vec<u8>,
    #[cfg_attr(not(test), expect(dead_code, reason = "stage-4 consumers land next"))]
    pub rd_index: Vec<u8>,
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
            let record = Arc::new(Self {
                pc: lanes.pc,
                unexpanded_pc: lanes.unexpanded_pc,
                imm: lanes.imm,
                rs1_value: lanes.rs1_value,
                rs2_value: lanes.rs2_value,
                rd_pre_value: lanes.rd_pre_value,
                rd_post_value: lanes.rd_post_value,
                rs1_index: lanes.rs1_index,
                rs2_index: lanes.rs2_index,
                rd_index: lanes.rd_index,
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
    pub(crate) fn should_branch(&self, t: usize) -> bool {
        self.flags[t] & SHOULD_BRANCH_BIT != 0
    }

    #[inline]
    pub(crate) fn should_jump(&self, t: usize) -> bool {
        self.flags[t] & SHOULD_JUMP_BIT != 0
    }

    #[inline]
    pub(crate) fn product(&self, t: usize) -> S128 {
        S128::new(
            [self.product_magnitude_lo[t], self.product_magnitude_hi[t]],
            self.flags[t] & PRODUCT_POSITIVE_BIT != 0,
        )
    }

    /// The stage-1 Spartan outer bundle at cycle `t`, rebuilt from the lanes.
    /// The `Next*` fields follow the extractors' lookahead semantics exactly:
    /// the successor row's own lanes, zero/false at `t = T - 1`.
    #[inline]
    pub(crate) fn outer_row(&self, t: usize) -> SpartanOuterRow {
        let next = t + 1 < self.len();
        let circuit_flags = self.circuit_flags(t);
        let flag = |flag: CircuitFlags| OpFlag(circuit_flags.get(flag));
        SpartanOuterRow {
            left_instruction_input: LeftInstructionInput(self.left_instruction_input[t]),
            right_instruction_input: RightInstructionInput(self.right_instruction_input[t]),
            product: Product(self.product(t)),
            should_branch: ShouldBranch(self.should_branch(t)),
            pc: Pc(self.pc[t]),
            unexpanded_pc: UnexpandedPc(self.unexpanded_pc[t]),
            imm: Imm(self.imm[t]),
            ram_address: RamAddress(self.ram_address[t]),
            rs1_value: Rs1Value(self.rs1_value[t]),
            rs2_value: Rs2Value(self.rs2_value[t]),
            rd_write_value: RdWriteValue(self.rd_post_value[t]),
            ram_read_value: RamReadValue(self.ram.pre_values[t]),
            ram_write_value: RamWriteValue(self.ram.post_values[t]),
            left_lookup_operand: LeftLookupOperand(self.left_lookup_operand[t]),
            right_lookup_operand: RightLookupOperand(self.right_lookup_operand[t]),
            next_unexpanded_pc: NextUnexpandedPc(if next { self.unexpanded_pc[t + 1] } else { 0 }),
            next_pc: NextPc(if next { self.pc[t + 1] } else { 0 }),
            next_is_virtual: NextIsVirtual(
                next && self.circuit_flag(t + 1, CircuitFlags::VirtualInstruction),
            ),
            next_is_first_in_sequence: NextIsFirstInSequence(
                next && self.circuit_flag(t + 1, CircuitFlags::IsFirstInSequence),
            ),
            lookup_output: LookupOutput(self.lookup_output[t]),
            should_jump: ShouldJump(self.should_jump(t)),
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
                assert_eq!(&record.outer_row(t), row, "cycle {t}");
            }

            let register_rows: Vec<RegisterCycleRow> =
                collect_rows(witness, 1 << shape.log_t).unwrap();
            for (t, row) in register_rows.iter().enumerate() {
                let (rs1_index, rs1_value) = row.rs1.map_or((NO_REGISTER, 0), |(i, v)| (i, v));
                let (rs2_index, rs2_value) = row.rs2.map_or((NO_REGISTER, 0), |(i, v)| (i, v));
                let (rd_index, rd_pre, rd_post) = row
                    .rd
                    .map_or((NO_REGISTER, 0, 0), |(i, pre, post)| (i, pre, post));
                assert_eq!(record.rs1_index[t], rs1_index, "cycle {t}");
                assert_eq!(record.rs1_value[t], rs1_value, "cycle {t}");
                assert_eq!(record.rs2_index[t], rs2_index, "cycle {t}");
                assert_eq!(record.rs2_value[t], rs2_value, "cycle {t}");
                assert_eq!(record.rd_index[t], rd_index, "cycle {t}");
                assert_eq!(record.rd_pre_value[t], rd_pre, "cycle {t}");
                assert_eq!(record.rd_post_value[t], rd_post, "cycle {t}");
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
