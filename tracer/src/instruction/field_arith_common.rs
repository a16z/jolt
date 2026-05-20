//! Shared helpers for the BN254 Fr coprocessor arithmetic instructions
//! (`FieldMul`, `FieldAdd`, `FieldSub`, `FieldInv`).
//!
//! Each variant lives in its own file (one struct per `JoltInstructionKind`
//! so the per-PC bytecode commits the right static `IsFieldX` circuit flag),
//! but they all share:
//!   - the same operand format (R-type: `frd, frs1, frs2`),
//!   - the same RAM-access type (`()` — FR ops don't touch RAM),
//!   - the natural-form 256-bit limb representation in `cpu.field_regs`,
//!   - the same `FieldRegEvent` emission shape on trace.

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};

use crate::emulator::cpu::{Cpu, FieldRegEvent};
use crate::instruction::{
    format::{format_r::FormatR, InstructionFormat},
    Cycle, RISCVCycle, RISCVInstruction,
};

/// Native-field coprocessor opcode (custom-0).
pub const FIELD_OP_OPCODE: u32 = 0x0B;
/// BN254 Fr funct7 family selector (FMUL/FADD/FSUB/FINV/FAssertEq/FMov live here).
pub const BN254_FR_FUNCT7: u32 = 0x40;

pub const FUNCT3_FMUL: u32 = 0x02;
pub const FUNCT3_FADD: u32 = 0x03;
pub const FUNCT3_FINV: u32 = 0x04;
pub const FUNCT3_FSUB: u32 = 0x05;

/// MASK that pins opcode + funct3 + funct7. Each FR-arithmetic variant
/// uses this mask with its own MATCH bit-pattern.
pub const FIELD_ARITH_MASK: u32 = 0xfe00707f;

/// Build the MATCH constant for a given FR-arithmetic funct3.
pub const fn field_arith_match(funct3: u32) -> u32 {
    (BN254_FR_FUNCT7 << 25) | (funct3 << 12) | FIELD_OP_OPCODE
}

/// Convert natural-form little-endian u64 limbs to an Fr element.
/// Values `>= p` are reduced modulo the BN254 Fr prime.
pub fn fr_from_limbs(limbs: &[u64; 4]) -> Fr {
    let mut bytes = [0u8; 32];
    for (i, &limb) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    <Fr as PrimeField>::from_le_bytes_mod_order(&bytes)
}

/// Convert Fr to natural-form little-endian u64 limbs.
pub fn fr_to_limbs(fr: &Fr) -> [u64; 4] {
    let bytes: Vec<u8> = fr.into_bigint().to_bytes_le();
    let mut limbs = [0u64; 4];
    debug_assert!(bytes.len() <= 32, "Fr bytes must fit in 32");
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        let end = core::cmp::min(start + 8, bytes.len());
        if start < bytes.len() {
            let mut buf = [0u8; 8];
            buf[..end - start].copy_from_slice(&bytes[start..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}

/// Common trace body for the FR-arithmetic instructions.
///
/// Snapshots the pre-execution operands, runs `execute`, and emits one
/// `FieldRegEvent` for the frd slot. The variant-specific arithmetic is
/// driven by the caller's `Instr::execute()` impl.
pub fn trace_field_arith_cycle<I>(
    instr: &I,
    operands: &FormatR,
    cpu: &mut Cpu,
    trace: Option<&mut Vec<Cycle>>,
) where
    I: RISCVInstruction<Format = FormatR, RAMAccess = ()> + Copy,
    Cycle: From<RISCVCycle<I>>,
{
    let frd = operands.rd;
    let frs1 = operands.rs1;
    let frs2 = operands.rs2;
    let frd_pre = cpu.field_regs[frd as usize];
    let _frs1_val = cpu.field_regs[frs1 as usize];
    let _frs2_val = cpu.field_regs[frs2 as usize];

    let mut cycle = RISCVCycle::<I> {
        instruction: *instr,
        register_state: Default::default(),
        ram_access: (),
    };
    instr
        .operands()
        .capture_pre_execution_state(&mut cycle.register_state, cpu);
    instr.execute(cpu, &mut cycle.ram_access);
    instr
        .operands()
        .capture_post_execution_state(&mut cycle.register_state, cpu);

    let frd_post = cpu.field_regs[frd as usize];

    if let Some(trace_vec) = trace {
        let cycle_index = cpu.trace_len + trace_vec.len();
        cpu.field_reg_events.push(FieldRegEvent {
            cycle_index,
            slot: frd,
            old: frd_pre,
            new: frd_post,
        });
        trace_vec.push(cycle.into());
    }
}
