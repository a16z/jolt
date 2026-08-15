use jolt_riscv::CircuitFlags;
use jolt_witness::witnesses::{
    Imm, LeftInstructionInput, LeftLookupOperand, LookupOutput, OpFlag, Pc, Product, RamAddress,
    RamReadValue, RamWriteValue, RdWriteValue, RightInstructionInput, RightLookupOperand, Rs1Value,
    Rs2Value, ShouldBranch, ShouldJump, UnexpandedPc,
};
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub const VARIABLES: usize = 35;

pub const NARROW: usize = 13;
pub const WIDE: usize = 4;
pub const SIGNED_WIDE: usize = 3;

const _: () = assert!(SIGNED_WIDE < WIDE);

pub const KIND_NARROW: u32 = 0;
pub const KIND_WIDE: u32 = 1;
pub const KIND_FLAG: u32 = 2;

pub const PC_SLOT: usize = 1;
pub const UNEXPANDED_PC_SLOT: usize = 2;
pub const NEXT_UNEXPANDED_PC_SLOT: usize = 11;
pub const NEXT_PC_SLOT: usize = 12;

pub const VIRTUAL_INSTRUCTION_BIT: u32 = 9;
pub const FIRST_IN_SEQUENCE_BIT: u32 = 14;
pub const NEXT_IS_VIRTUAL_BIT: u32 = 16;
pub const NEXT_IS_FIRST_IN_SEQUENCE_BIT: u32 = 17;
pub const SIGN_BIT_BASE: u32 = 24;

const fn code(kind: u32, slot: u32) -> u32 {
    (kind << 8) | slot
}

pub const LAYOUT: [u32; VARIABLES] = [
    code(KIND_NARROW, 0),
    code(KIND_WIDE, 0),
    code(KIND_WIDE, 1),
    code(KIND_FLAG, 0),
    code(KIND_NARROW, PC_SLOT as u32),
    code(KIND_NARROW, UNEXPANDED_PC_SLOT as u32),
    code(KIND_WIDE, 2),
    code(KIND_NARROW, 3),
    code(KIND_NARROW, 4),
    code(KIND_NARROW, 5),
    code(KIND_NARROW, 6),
    code(KIND_NARROW, 7),
    code(KIND_NARROW, 8),
    code(KIND_NARROW, 9),
    code(KIND_WIDE, 3),
    code(KIND_NARROW, NEXT_UNEXPANDED_PC_SLOT as u32),
    code(KIND_NARROW, NEXT_PC_SLOT as u32),
    code(KIND_FLAG, NEXT_IS_VIRTUAL_BIT),
    code(KIND_FLAG, NEXT_IS_FIRST_IN_SEQUENCE_BIT),
    code(KIND_NARROW, 10),
    code(KIND_FLAG, 1),
    code(KIND_FLAG, 2),
    code(KIND_FLAG, 3),
    code(KIND_FLAG, 4),
    code(KIND_FLAG, 5),
    code(KIND_FLAG, 6),
    code(KIND_FLAG, 7),
    code(KIND_FLAG, 8),
    code(KIND_FLAG, VIRTUAL_INSTRUCTION_BIT),
    code(KIND_FLAG, 10),
    code(KIND_FLAG, 11),
    code(KIND_FLAG, 12),
    code(KIND_FLAG, 13),
    code(KIND_FLAG, FIRST_IN_SEQUENCE_BIT),
    code(KIND_FLAG, 15),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub struct SpartanOuterWitness {
    pub left_instruction_input: LeftInstructionInput,
    pub right_instruction_input: RightInstructionInput,
    pub product: Product,
    pub imm: Imm,
    pub right_lookup_operand: RightLookupOperand,
    pub pc: Pc,
    pub unexpanded_pc: UnexpandedPc,
    pub ram_address: RamAddress,
    pub rs1_value: Rs1Value,
    pub rs2_value: Rs2Value,
    pub rd_write_value: RdWriteValue,
    pub ram_read_value: RamReadValue,
    pub ram_write_value: RamWriteValue,
    pub left_lookup_operand: LeftLookupOperand,
    pub lookup_output: LookupOutput,
    pub should_branch: ShouldBranch,
    pub should_jump: ShouldJump,
    #[opening(OpFlags(CircuitFlags::AddOperands))]
    pub add_operands: OpFlag,
    #[opening(OpFlags(CircuitFlags::SubtractOperands))]
    pub subtract_operands: OpFlag,
    #[opening(OpFlags(CircuitFlags::MultiplyOperands))]
    pub multiply_operands: OpFlag,
    #[opening(OpFlags(CircuitFlags::Load))]
    pub load: OpFlag,
    #[opening(OpFlags(CircuitFlags::Store))]
    pub store: OpFlag,
    #[opening(OpFlags(CircuitFlags::Jump))]
    pub jump: OpFlag,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD))]
    pub write_lookup_output_to_rd: OpFlag,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub virtual_instruction: OpFlag,
    #[opening(OpFlags(CircuitFlags::Assert))]
    pub assert_flag: OpFlag,
    #[opening(OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC))]
    pub do_not_update_unexpanded_pc: OpFlag,
    #[opening(OpFlags(CircuitFlags::Advice))]
    pub advice: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsCompressed))]
    pub is_compressed: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence))]
    pub is_first_in_sequence: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsLastInSequence))]
    pub is_last_in_sequence: OpFlag,
}

pub struct Packed {
    pub narrow: Vec<u64>,
    pub wide: Vec<u64>,
    pub flags: Vec<u32>,
}

const PACK_CHUNK: usize = 1 << 13;

pub fn pack(rows: &[SpartanOuterWitness]) -> Packed {
    let cycles = rows.len();
    let mut narrow = vec![0u64; cycles * NARROW];
    let mut wide = vec![0u64; cycles * WIDE * 2];
    let mut flags = vec![0u32; cycles];

    #[cfg(feature = "parallel")]
    narrow
        .par_chunks_mut(PACK_CHUNK * NARROW)
        .zip(wide.par_chunks_mut(PACK_CHUNK * WIDE * 2))
        .zip(flags.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .for_each(|(((narrow, wide), flags), rows)| fill(narrow, wide, flags, rows));
    #[cfg(not(feature = "parallel"))]
    narrow
        .chunks_mut(PACK_CHUNK * NARROW)
        .zip(wide.chunks_mut(PACK_CHUNK * WIDE * 2))
        .zip(flags.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .for_each(|(((narrow, wide), flags), rows)| fill(narrow, wide, flags, rows));

    Packed {
        narrow,
        wide,
        flags,
    }
}

fn fill(narrow: &mut [u64], wide: &mut [u64], flags: &mut [u32], rows: &[SpartanOuterWitness]) {
    for (index, row) in rows.iter().enumerate() {
        let slots = &mut narrow[index * NARROW..(index + 1) * NARROW];
        slots[0] = row.left_instruction_input.0;
        slots[PC_SLOT] = row.pc.0;
        slots[UNEXPANDED_PC_SLOT] = row.unexpanded_pc.0;
        slots[3] = row.ram_address.0;
        slots[4] = row.rs1_value.0;
        slots[5] = row.rs2_value.0;
        slots[6] = row.rd_write_value.0;
        slots[7] = row.ram_read_value.0;
        slots[8] = row.ram_write_value.0;
        slots[9] = row.left_lookup_operand.0;
        slots[10] = row.lookup_output.0;

        let mut mask = 0u32;
        let magnitudes = [
            signed_i128(row.right_instruction_input.0),
            product_magnitude(row.product),
            signed_i128(row.imm.0),
            (false, row.right_lookup_operand.0),
        ];
        let limbs = &mut wide[index * WIDE * 2..(index + 1) * WIDE * 2];
        for (slot, (negative, magnitude)) in magnitudes.into_iter().enumerate() {
            limbs[2 * slot] = magnitude as u64;
            limbs[2 * slot + 1] = (magnitude >> 64) as u64;
            if negative {
                mask |= 1 << (SIGN_BIT_BASE + slot as u32);
            }
        }

        let bits = [
            row.should_branch.0,
            row.should_jump.0,
            row.add_operands.0,
            row.subtract_operands.0,
            row.multiply_operands.0,
            row.load.0,
            row.store.0,
            row.jump.0,
            row.write_lookup_output_to_rd.0,
            row.virtual_instruction.0,
            row.assert_flag.0,
            row.do_not_update_unexpanded_pc.0,
            row.advice.0,
            row.is_compressed.0,
            row.is_first_in_sequence.0,
            row.is_last_in_sequence.0,
        ];
        for (bit, set) in bits.into_iter().enumerate() {
            if set {
                mask |= 1 << bit as u32;
            }
        }
        flags[index] = mask;
    }
}

const fn signed_i128(value: i128) -> (bool, u128) {
    if value < 0 {
        (true, value.unsigned_abs())
    } else {
        (false, value as u128)
    }
}

fn product_magnitude(product: Product) -> (bool, u128) {
    let magnitude = product.0.magnitude_as_u128();
    (!product.0.is_positive && magnitude != 0, magnitude)
}
