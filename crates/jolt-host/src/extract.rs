//! Single-pass trace extraction: witness inputs + R1CS + instruction flags.
//!
//! [`extract_trace`] iterates the trace once, producing all three artifacts
//! the prover needs. R1CS witness construction delegates to
//! [`r1cs_cycle_witness`](crate::r1cs_witness::r1cs_cycle_witness).

use common::jolt_device::MemoryLayout;
use jolt_field::Field;
use jolt_instructions::flags::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use jolt_r1cs::constraints::rv64::*;
use jolt_witness::CycleInput;

use crate::bytecode::BytecodePreprocessing;
use crate::r1cs_witness::r1cs_cycle_witness;
use crate::CycleRow;
use jolt_witness::{replay_field_regs, FieldRegEvent, FrCycleData};

/// Per-cycle instruction flag polynomials for sumcheck instances.
pub struct InstructionFlagData<F> {
    pub is_noop: Vec<F>,
    pub left_is_rs1: Vec<F>,
    pub left_is_pc: Vec<F>,
    pub right_is_rs2: Vec<F>,
    pub right_is_imm: Vec<F>,
}

/// Extract witness inputs, R1CS witness, and instruction flags in one pass.
///
/// Produces `size`-length outputs, padding beyond `trace.len()` with defaults.
///
/// `fr_events` is the BN254 Fr coprocessor event stream. Pass an empty slice
/// for traces with no FR cycles. When non-empty, this populates
/// V_FIELD_RS1/RS2/RD_VALUE in each cycle's R1CS witness from the per-cycle
/// FR replay snapshot, so the Spartan witness matches FR Twist state.
pub fn extract_trace<C: CycleRow, F: Field>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    num_vars_padded: usize,
    fr_events: &[FieldRegEvent],
) -> (Vec<CycleInput>, Vec<F>, InstructionFlagData<F>) {
    let mut inputs = Vec::with_capacity(size);
    let mut r1cs = vec![F::from_u64(0); size * num_vars_padded];
    let mut flags = InstructionFlagData::new(size);

    // Build FR snapshots once (linear pass over events) when needed.
    let fr_snapshots: Option<Vec<FrCycleData>> = if fr_events.is_empty() {
        None
    } else {
        let bytecode_snaps: Vec<_> = trace.iter().map(|c| c.fr_meta()).collect();
        Some(replay_field_regs(trace.len(), &bytecode_snaps, fr_events))
    };

    for t in 0..size {
        let offset = t * num_vars_padded;

        if t < trace.len() {
            let cycle = &trace[t];
            let snap = fr_snapshots.as_ref().and_then(|s| s.get(t));

            if cycle.is_noop() {
                inputs.push(CycleInput::PADDING);
            } else {
                inputs.push(cycle_input(cycle, bytecode, memory_layout, snap));
            }

            let row = r1cs_cycle_witness::<C, F>(trace, t, bytecode, snap);
            r1cs[offset..offset + NUM_VARS_PER_CYCLE].copy_from_slice(&row);

            let iflags = cycle.instruction_flags();
            flags.set(t, &iflags);
        } else {
            inputs.push(CycleInput::PADDING);
            flags.is_noop[t] = F::from_u64(1);
            r1cs[offset + V_CONST] = F::from_u64(1);
            r1cs[offset + V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = F::from_u64(1);
            r1cs[offset + V_NEXT_IS_NOOP] = F::from_u64(1);
        }
    }

    (inputs, r1cs, flags)
}

fn cycle_input(
    cycle: &impl CycleRow,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    fr_snapshot: Option<&FrCycleData>,
) -> CycleInput {
    let rd_inc = match cycle.rd_write() {
        Some((_, pre, post)) => post as i128 - pre as i128,
        None => 0,
    };
    let ram_inc = match (cycle.ram_read_value(), cycle.ram_write_value()) {
        (Some(pre), Some(post)) => post as i128 - pre as i128,
        _ => 0,
    };
    let lowest = memory_layout.get_lowest_address();
    let ram_address = cycle.ram_access_address().map(|addr| {
        debug_assert!(
            addr >= lowest,
            "RAM address {addr:#x} below lowest {lowest:#x}"
        );
        ((addr - lowest) / 8) as u128
    });

    // FieldRegRa one-hot index: encode the FR write slot for this cycle.
    // None for non-writing cycles (preserves all-zero one-hot for non-FR
    // traces). FieldRegInc dense is left at 0 here — Fr deltas are 256-bit
    // and don't fit in i128, so the prover-side caller MUST overwrite the
    // FieldRegInc buffer for any FR-active program by calling
    // `polys.insert(PolynomialId::FieldRegInc,
    //               jolt_witness::field_reg_inc_polynomial(events, T))`
    // after `polys.finish()`. Forgetting this silently commits all-zero
    // deltas while FieldRegRa is non-zero — see `specs/fr-v2-audit.md` C11.
    let fr_write_slot = fr_snapshot
        .and_then(|s| s.write_slot)
        .map(|k| k as u128);

    CycleInput {
        dense: [rd_inc, ram_inc, 0],
        one_hot: [
            Some(cycle.lookup_index()),
            Some(bytecode.get_pc(cycle) as u128),
            ram_address,
            fr_write_slot,
        ],
    }
}

impl<F: Field> InstructionFlagData<F> {
    fn new(size: usize) -> Self {
        Self {
            is_noop: vec![F::from_u64(0); size],
            left_is_rs1: vec![F::from_u64(0); size],
            left_is_pc: vec![F::from_u64(0); size],
            right_is_rs2: vec![F::from_u64(0); size],
            right_is_imm: vec![F::from_u64(0); size],
        }
    }

    fn set(&mut self, t: usize, iflags: &[bool; NUM_INSTRUCTION_FLAGS]) {
        self.is_noop[t] = F::from_u64(iflags[InstructionFlags::IsNoop] as u64);
        self.left_is_rs1[t] = F::from_u64(iflags[InstructionFlags::LeftOperandIsRs1Value] as u64);
        self.left_is_pc[t] = F::from_u64(iflags[InstructionFlags::LeftOperandIsPC] as u64);
        self.right_is_rs2[t] = F::from_u64(iflags[InstructionFlags::RightOperandIsRs2Value] as u64);
        self.right_is_imm[t] = F::from_u64(iflags[InstructionFlags::RightOperandIsImm] as u64);
    }
}
