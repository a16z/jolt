//! Trace-to-kernel witness views for Bolt-generated Jolt stages.

use jolt_field::signed::{S128, S64};
use jolt_field::Field;
use jolt_program::{
    execution::{RamAccess, TraceRow},
    preprocess::BytecodePreprocessing,
};
use jolt_riscv::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags,
    InterleavedBitsMarker, JoltInstructionKind, LookupInstruction, NormalizedInstruction,
    NUM_CIRCUIT_FLAGS,
};
use jolt_witness::Stage6BytecodeEntry;

use crate::stage1::Stage1Rv64Cycle;
use crate::stage2::{Stage2InstructionLookupCycle, Stage2ProductVirtualCycle, Stage2RamAccess};
use crate::stage3::Stage3Cycle;
use crate::stage4::{Stage4RegisterAccess, Stage4RegisterRead, Stage4RegisterWrite};

pub trait CycleRow: Copy {
    fn is_noop(&self) -> bool;
    fn instruction(&self) -> NormalizedInstruction;
    fn unexpanded_pc(&self) -> u64;
    fn rs1_read(&self) -> Option<(u8, u64)>;
    fn rs2_read(&self) -> Option<(u8, u64)>;
    fn rd_write(&self) -> Option<(u8, u64, u64)>;
    fn ram_access_address(&self) -> Option<u64>;
    fn ram_read_value(&self) -> Option<u64>;
    fn ram_write_value(&self) -> Option<u64>;
    fn imm(&self) -> i128;
    fn circuit_flags(&self) -> CircuitFlagSet;
    fn instruction_flags(&self) -> InstructionFlagSet;
    fn lookup_index(&self) -> u128;
    fn lookup_output(&self) -> u64;
}

impl CycleRow for TraceRow {
    fn is_noop(&self) -> bool {
        self.instruction_flags()[InstructionFlags::IsNoop]
    }

    fn instruction(&self) -> NormalizedInstruction {
        self.instruction
    }

    fn unexpanded_pc(&self) -> u64 {
        if self.is_noop() {
            0
        } else {
            self.instruction.address as u64
        }
    }

    fn rs1_read(&self) -> Option<(u8, u64)> {
        self.registers.rs1.map(|read| (read.register, read.value))
    }

    fn rs2_read(&self) -> Option<(u8, u64)> {
        self.registers.rs2.map(|read| (read.register, read.value))
    }

    fn rd_write(&self) -> Option<(u8, u64, u64)> {
        self.registers
            .rd
            .map(|write| (write.register, write.pre_value, write.post_value))
    }

    fn ram_access_address(&self) -> Option<u64> {
        match self.ram_access {
            RamAccess::Read(read) => Some(read.address),
            RamAccess::Write(write) => Some(write.address),
            RamAccess::NoOp => None,
        }
    }

    fn ram_read_value(&self) -> Option<u64> {
        match self.ram_access {
            RamAccess::Read(read) => Some(read.value),
            RamAccess::Write(write) => Some(write.pre_value),
            RamAccess::NoOp => None,
        }
    }

    fn ram_write_value(&self) -> Option<u64> {
        match self.ram_access {
            RamAccess::Read(read) => Some(read.value),
            RamAccess::Write(write) => Some(write.post_value),
            RamAccess::NoOp => None,
        }
    }

    fn imm(&self) -> i128 {
        if self.is_noop() {
            0
        } else {
            self.instruction.operands.imm
        }
    }

    fn circuit_flags(&self) -> CircuitFlagSet {
        instruction_circuit_flags(&self.instruction)
    }

    fn instruction_flags(&self) -> InstructionFlagSet {
        instruction_instruction_flags(&self.instruction)
    }

    fn lookup_index(&self) -> u128 {
        let cflags = self.circuit_flags();
        let iflags = self.instruction_flags();
        let (left, right) = instruction_inputs_with_flags(self, iflags);

        if cflags[CircuitFlags::AddOperands] {
            (left as u128).wrapping_add(right)
        } else if cflags[CircuitFlags::SubtractOperands] {
            (1u128 << 64).wrapping_sub(right).wrapping_add(left as u128)
        } else if cflags[CircuitFlags::MultiplyOperands] {
            (left as u128).wrapping_mul(right)
        } else if cflags[CircuitFlags::Advice] {
            self.rd_write().map_or(0, |(_, _, post)| post as u128)
        } else if self.is_noop() {
            0
        } else {
            interleave_bits(left, right as u64)
        }
    }

    fn lookup_output(&self) -> u64 {
        if self.is_noop() {
            return 0;
        }
        let cflags = self.circuit_flags();
        let iflags = self.instruction_flags();

        if cflags[CircuitFlags::Jump] {
            let left = if iflags[InstructionFlags::LeftOperandIsPC] {
                self.unexpanded_pc()
            } else {
                self.rs1_read().map_or(0, |(_, value)| value)
            };
            let target = (left as i64).wrapping_add(self.imm() as i64) as u64;
            if iflags[InstructionFlags::LeftOperandIsRs1Value] {
                target & !1
            } else {
                target
            }
        } else if cflags[CircuitFlags::Assert] {
            1
        } else if iflags[InstructionFlags::Branch] {
            let rs1 = self.rs1_read().map_or(0, |(_, value)| value);
            let rs2 = self.rs2_read().map_or(0, |(_, value)| value);
            branch_result(self.instruction.instruction_kind, rs1, rs2)
        } else if cflags[CircuitFlags::WriteLookupOutputToRD] {
            self.rd_write().map_or(0, |(_, _, post)| post)
        } else {
            0
        }
    }
}

pub fn stage1_rv64_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
) -> Vec<Stage1Rv64Cycle> {
    (0..size)
        .map(|cycle| stage1_rv64_cycle(trace, cycle, bytecode))
        .collect()
}

pub fn stage2_product_virtual_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
) -> Vec<Stage2ProductVirtualCycle> {
    (0..size)
        .map(|index| stage2_product_virtual_cycle(trace, index))
        .collect()
}

pub fn stage2_instruction_lookup_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
) -> Vec<Stage2InstructionLookupCycle> {
    (0..size)
        .map(|index| stage2_instruction_lookup_cycle(trace.get(index).copied()))
        .collect()
}

pub fn stage2_ram_accesses<C, R>(
    trace: &[C],
    size: usize,
    mut remap_address: R,
) -> Vec<Stage2RamAccess>
where
    C: CycleRow,
    R: FnMut(u64) -> Option<usize>,
{
    (0..size)
        .map(|index| {
            let Some(cycle) = trace.get(index) else {
                return Stage2RamAccess::noop();
            };
            let Some(address) = cycle.ram_access_address() else {
                return Stage2RamAccess::noop();
            };
            Stage2RamAccess {
                remapped_address: remap_address(address),
                read_value: cycle.ram_read_value().unwrap_or(0),
                write_value: cycle.ram_write_value().unwrap_or(0),
            }
        })
        .collect()
}

pub fn stage3_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
) -> Vec<Stage3Cycle> {
    (0..size)
        .map(|cycle| stage3_cycle(trace.get(cycle).copied(), bytecode))
        .collect()
}

pub fn stage4_register_accesses<C: CycleRow>(
    trace: &[C],
    size: usize,
) -> Vec<Stage4RegisterAccess> {
    (0..size)
        .map(|cycle| stage4_register_access(trace.get(cycle).copied()))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5LookupTrace {
    pub lookup_indices: Vec<u128>,
    pub lookup_table_indices: Vec<Option<usize>>,
    pub is_interleaved_operands: Vec<bool>,
}

pub fn stage5_lookup_trace<C, L>(
    trace: &[C],
    size: usize,
    mut lookup_table_index: L,
) -> Stage5LookupTrace
where
    C: CycleRow,
    L: FnMut(&C) -> Option<usize>,
{
    let mut lookup_indices = Vec::with_capacity(size);
    let mut lookup_table_indices = Vec::with_capacity(size);
    let mut is_interleaved_operands = Vec::with_capacity(size);
    for index in 0..size {
        let Some(cycle) = trace.get(index) else {
            lookup_indices.push(0);
            lookup_table_indices.push(None);
            is_interleaved_operands.push(false);
            continue;
        };
        lookup_indices.push(cycle.lookup_index());
        lookup_table_indices.push(lookup_table_index(cycle));
        is_interleaved_operands.push(cycle.circuit_flags().is_interleaved_operands());
    }
    Stage5LookupTrace {
        lookup_indices,
        lookup_table_indices,
        is_interleaved_operands,
    }
}

pub fn stage6_bytecode_entries<F, L>(
    bytecode: &BytecodePreprocessing,
    mut lookup_table_index: L,
) -> Vec<Stage6BytecodeEntry<F>>
where
    F: Field,
    L: FnMut(&NormalizedInstruction) -> Option<usize>,
{
    bytecode
        .bytecode
        .iter()
        .map(|instruction| {
            let instr = *instruction;
            let circuit_flags = instruction_circuit_flags(instruction);
            let instruction_flags = instruction_instruction_flags(instruction);
            Stage6BytecodeEntry {
                address: F::from_u64(instr.address as u64),
                imm: F::from_i128(instr.operands.imm),
                circuit_flags: stage6_circuit_flags(circuit_flags),
                rd: instr.operands.rd.map(usize::from),
                rs1: instr.operands.rs1.map(usize::from),
                rs2: instr.operands.rs2.map(usize::from),
                lookup_table: lookup_table_index(instruction),
                is_interleaved: circuit_flags.is_interleaved_operands(),
                is_branch: instruction_flags[InstructionFlags::Branch],
                left_is_rs1: instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
                left_is_pc: instruction_flags[InstructionFlags::LeftOperandIsPC],
                right_is_rs2: instruction_flags[InstructionFlags::RightOperandIsRs2Value],
                right_is_imm: instruction_flags[InstructionFlags::RightOperandIsImm],
                is_noop: instruction_flags[InstructionFlags::IsNoop],
            }
        })
        .collect()
}

fn stage2_product_virtual_cycle<C: CycleRow>(
    trace: &[C],
    index: usize,
) -> Stage2ProductVirtualCycle {
    let Some(cycle) = trace.get(index) else {
        return Stage2ProductVirtualCycle::padding();
    };
    let (instruction_left_input, instruction_right_input) = instruction_inputs(cycle);
    let flags = cycle.circuit_flags();
    let instruction_flags = cycle.instruction_flags();
    let not_next_noop = trace.get(index + 1).is_some_and(|next| !next.is_noop());
    Stage2ProductVirtualCycle {
        instruction_left_input,
        instruction_right_input,
        should_branch_lookup_output: cycle.lookup_output(),
        write_lookup_output_to_rd_flag: flags[CircuitFlags::WriteLookupOutputToRD],
        jump_flag: flags[CircuitFlags::Jump],
        should_branch_flag: instruction_flags[InstructionFlags::Branch],
        not_next_noop,
        virtual_instruction_flag: flags[CircuitFlags::VirtualInstruction],
    }
}

fn stage2_instruction_lookup_cycle<C: CycleRow>(cycle: Option<C>) -> Stage2InstructionLookupCycle {
    let Some(cycle) = cycle else {
        return Stage2InstructionLookupCycle::padding();
    };
    let (left_instruction_input, right_instruction_input) = instruction_inputs(&cycle);
    let product = instruction_product(left_instruction_input, right_instruction_input);
    let (left_lookup_operand, right_lookup_operand) = lookup_operands_raw(
        left_instruction_input,
        right_instruction_input,
        product,
        cycle.circuit_flags(),
        cycle.lookup_output(),
    );
    Stage2InstructionLookupCycle {
        lookup_output: cycle.lookup_output(),
        left_lookup_operand,
        right_lookup_operand,
        left_instruction_input,
        right_instruction_input,
    }
}

fn stage4_register_access<C: CycleRow>(cycle: Option<C>) -> Stage4RegisterAccess {
    let Some(cycle) = cycle else {
        return Stage4RegisterAccess::default();
    };
    Stage4RegisterAccess {
        rs1: cycle.rs1_read().map(|(address, value)| Stage4RegisterRead {
            address: address as usize,
            value,
        }),
        rs2: cycle.rs2_read().map(|(address, value)| Stage4RegisterRead {
            address: address as usize,
            value,
        }),
        rd: cycle
            .rd_write()
            .map(|(address, pre_value, post_value)| Stage4RegisterWrite {
                address: address as usize,
                pre_value,
                post_value,
            }),
    }
}

fn stage3_cycle<C: CycleRow>(cycle: Option<C>, bytecode: &BytecodePreprocessing) -> Stage3Cycle {
    let Some(cycle) = cycle else {
        return Stage3Cycle::padding();
    };
    let circuit_flags = cycle.circuit_flags();
    let instruction_flags = cycle.instruction_flags();
    Stage3Cycle {
        unexpanded_pc: cycle.unexpanded_pc(),
        pc: bytecode_pc(bytecode, &cycle.instruction()) as u64,
        is_virtual: circuit_flags[CircuitFlags::VirtualInstruction],
        is_first_in_sequence: circuit_flags[CircuitFlags::IsFirstInSequence],
        is_noop: instruction_flags[InstructionFlags::IsNoop],
        left_operand_is_rs1: instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
        rs1_value: cycle.rs1_read().map_or(0, |(_, value)| value),
        left_operand_is_pc: instruction_flags[InstructionFlags::LeftOperandIsPC],
        right_operand_is_rs2: instruction_flags[InstructionFlags::RightOperandIsRs2Value],
        rs2_value: cycle.rs2_read().map_or(0, |(_, value)| value),
        right_operand_is_imm: instruction_flags[InstructionFlags::RightOperandIsImm],
        imm: cycle.imm(),
        rd_write_value: cycle.rd_write().map_or(0, |(_, _, post)| post),
    }
}

fn stage1_rv64_cycle<C: CycleRow>(
    trace: &[C],
    cycle_index: usize,
    bytecode: &BytecodePreprocessing,
) -> Stage1Rv64Cycle {
    let Some(cycle) = trace.get(cycle_index) else {
        return Stage1Rv64Cycle::padding();
    };
    let next = trace.get(cycle_index + 1);
    if cycle.is_noop() {
        let mut row = Stage1Rv64Cycle::padding();
        fill_next_rv64_fields(&mut row, next, bytecode);
        return row;
    }

    let flags_set = cycle.circuit_flags();
    let instruction_flags = cycle.instruction_flags();
    let (left_input, right_i128) = instruction_inputs(cycle);
    let right_input = s64_from_i128(right_i128);
    let product = instruction_product(left_input, right_i128);
    let lookup_output = cycle.lookup_output();
    let (left_lookup, right_lookup) =
        lookup_operands_raw(left_input, right_i128, product, flags_set, lookup_output);
    let next_is_noop = next.is_none_or(CycleRow::is_noop);
    let flags = stage1_rv64_flags(flags_set);

    let mut row = Stage1Rv64Cycle {
        left_input,
        right_input,
        product,
        left_lookup,
        right_lookup,
        lookup_output,
        rs1_read_value: cycle.rs1_read().map_or(0, |(_, value)| value),
        rs2_read_value: cycle.rs2_read().map_or(0, |(_, value)| value),
        rd_write_value: cycle.rd_write().map_or(0, |(_, _, post)| post),
        ram_addr: cycle.ram_access_address().unwrap_or(0),
        ram_read_value: cycle.ram_read_value().unwrap_or(0),
        ram_write_value: cycle.ram_write_value().unwrap_or(0),
        pc: bytecode_pc(bytecode, &cycle.instruction()) as u64,
        next_pc: 0,
        unexpanded_pc: cycle.unexpanded_pc(),
        next_unexpanded_pc: 0,
        imm: s64_from_i128(cycle.imm()),
        flags,
        should_jump: flags_set[CircuitFlags::Jump] && !next_is_noop,
        should_branch: instruction_flags[InstructionFlags::Branch] && lookup_output == 1,
        next_is_virtual: false,
        next_is_first_in_sequence: false,
    };
    fill_next_rv64_fields(&mut row, next, bytecode);
    row
}

fn fill_next_rv64_fields<C: CycleRow>(
    row: &mut Stage1Rv64Cycle,
    next: Option<&C>,
    bytecode: &BytecodePreprocessing,
) {
    if let Some(next_cycle) = next {
        row.next_pc = bytecode_pc(bytecode, &next_cycle.instruction()) as u64;
        row.next_unexpanded_pc = next_cycle.unexpanded_pc();
        let next_flags = next_cycle.circuit_flags();
        row.next_is_virtual = next_flags[CircuitFlags::VirtualInstruction];
        row.next_is_first_in_sequence = next_flags[CircuitFlags::IsFirstInSequence];
    }
}

fn instruction_inputs(cycle: &impl CycleRow) -> (u64, i128) {
    let instruction_flags = cycle.instruction_flags();
    instruction_inputs_i128_with_flags(cycle, instruction_flags)
}

fn instruction_inputs_i128_with_flags(
    cycle: &impl CycleRow,
    instruction_flags: InstructionFlagSet,
) -> (u64, i128) {
    let left_input = if instruction_flags[InstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if instruction_flags[InstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0, |(_, value)| value)
    } else {
        0
    };
    let right_input = if instruction_flags[InstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if instruction_flags[InstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0, |(_, value)| value as i128)
    } else {
        0
    };
    (left_input, right_input)
}

fn instruction_inputs_with_flags(
    cycle: &impl CycleRow,
    instruction_flags: InstructionFlagSet,
) -> (u64, u128) {
    let (left, right) = instruction_inputs_i128_with_flags(cycle, instruction_flags);
    (left, right as u64 as u128)
}

fn instruction_circuit_flags(instruction: &NormalizedInstruction) -> CircuitFlagSet {
    lookup_instruction(instruction).map_or_else(CircuitFlagSet::default, |instruction| {
        instruction.circuit_flags()
    })
}

fn instruction_instruction_flags(instruction: &NormalizedInstruction) -> InstructionFlagSet {
    lookup_instruction(instruction).map_or_else(InstructionFlagSet::default, |instruction| {
        instruction.instruction_flags()
    })
}

fn lookup_instruction(instruction: &NormalizedInstruction) -> Option<LookupInstruction> {
    LookupInstruction::try_from(*instruction).ok()
}

fn bytecode_pc(bytecode: &BytecodePreprocessing, instruction: &NormalizedInstruction) -> usize {
    bytecode.get_pc(instruction).unwrap_or(0)
}

fn instruction_product(left: u64, right: i128) -> S128 {
    S64::from_u64(left).mul_trunc::<2, 2>(&S128::from_i128(right))
}

fn stage1_rv64_flags(flags: CircuitFlagSet) -> [bool; NUM_CIRCUIT_FLAGS] {
    [
        flags[CircuitFlags::AddOperands],
        flags[CircuitFlags::SubtractOperands],
        flags[CircuitFlags::MultiplyOperands],
        flags[CircuitFlags::Load],
        flags[CircuitFlags::Store],
        flags[CircuitFlags::Jump],
        flags[CircuitFlags::WriteLookupOutputToRD],
        flags[CircuitFlags::VirtualInstruction],
        flags[CircuitFlags::Assert],
        flags[CircuitFlags::DoNotUpdateUnexpandedPC],
        flags[CircuitFlags::Advice],
        flags[CircuitFlags::IsCompressed],
        flags[CircuitFlags::IsFirstInSequence],
        flags[CircuitFlags::IsLastInSequence],
    ]
}

fn stage6_circuit_flags(flags: CircuitFlagSet) -> [bool; NUM_CIRCUIT_FLAGS] {
    stage1_rv64_flags(flags)
}

fn s64_from_i128(value: i128) -> S64 {
    let magnitude = value.unsigned_abs();
    assert!(magnitude <= u64::MAX as u128, "S64 input overflow");
    S64::from_u64_with_sign(magnitude as u64, value >= 0)
}

fn lookup_operands_raw(
    left: u64,
    right: i128,
    product: S128,
    flags: CircuitFlagSet,
    lookup_output: u64,
) -> (u64, u128) {
    if flags[CircuitFlags::AddOperands] {
        (0, (left as i128 + right) as u128)
    } else if flags[CircuitFlags::SubtractOperands] {
        (0, (left as i128 - right + (1i128 << 64)) as u128)
    } else if flags[CircuitFlags::MultiplyOperands] {
        (0, product.magnitude_as_u128())
    } else if flags[CircuitFlags::Advice] {
        (0, lookup_output as u128)
    } else {
        (left, right as u128)
    }
}

#[inline]
fn interleave_bits(x: u64, y: u64) -> u128 {
    let mut x_bits = x as u128;
    x_bits = (x_bits | (x_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    let mut y_bits = y as u128;
    y_bits = (y_bits | (y_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    (x_bits << 1) | y_bits
}

fn branch_result(kind: JoltInstructionKind, rs1: u64, rs2: u64) -> u64 {
    let taken = match kind {
        JoltInstructionKind::BEQ => rs1 == rs2,
        JoltInstructionKind::BNE => rs1 != rs2,
        JoltInstructionKind::BLT => (rs1 as i64) < (rs2 as i64),
        JoltInstructionKind::BGE => (rs1 as i64) >= (rs2 as i64),
        JoltInstructionKind::BLTU => rs1 < rs2,
        JoltInstructionKind::BGEU => rs1 >= rs2,
        _ => false,
    };
    taken as u64
}
