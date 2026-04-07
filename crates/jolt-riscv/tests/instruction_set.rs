//! Integration tests for the jolt-riscv crate.
//!
//! Exercises the JoltInstructionSet registry, instruction execution semantics,
//! flag consistency, and bit-interleaving utilities.

use jolt_riscv::{CircuitFlags, Flags, Instruction, JoltInstructionSet};

// Import instruction structs directly
use jolt_riscv::rv::arithmetic::*;
use jolt_riscv::rv::arithmetic_w::*;
use jolt_riscv::rv::branch::*;
use jolt_riscv::rv::compare::*;
use jolt_riscv::rv::jump::*;
use jolt_riscv::rv::load::*;
use jolt_riscv::rv::logic::*;
use jolt_riscv::rv::shift::*;
use jolt_riscv::rv::store::*;

// Registry completeness

/// Every instruction has a unique name and the registry has the expected count.
#[test]
fn registry_completeness() {
    let set = JoltInstructionSet::new();
    assert_eq!(set.len(), JoltInstructionSet::COUNT);

    let mut names: Vec<&str> = set.iter().map(|i| i.name()).collect();
    let original_len = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        original_len,
        "duplicate instruction names found"
    );
}

/// Out-of-range opcode returns None.
#[test]
fn registry_out_of_range() {
    let set = JoltInstructionSet::new();
    assert!(set.instruction(JoltInstructionSet::COUNT as u32).is_none());
    assert!(set.instruction(u32::MAX).is_none());
}

// Arithmetic instruction semantics

#[test]
fn add_basic_and_overflow() {
    assert_eq!(Add.execute(3, 5), 8);
    assert_eq!(Add.execute(u64::MAX, 1), 0);
    assert_eq!(Add.execute(0, 0), 0);
}

#[test]
fn sub_basic_and_underflow() {
    assert_eq!(Sub.execute(10, 3), 7);
    assert_eq!(Sub.execute(0, 1), u64::MAX);
}

#[test]
fn mul_basic_and_overflow() {
    assert_eq!(Mul.execute(6, 7), 42);
    assert_eq!(Mul.execute(u64::MAX, 2), u64::MAX.wrapping_mul(2));
    assert_eq!(Mul.execute(0, u64::MAX), 0);
}

// Division edge cases

#[test]
fn div_by_zero_returns_max() {
    assert_eq!(Div.execute(42, 0), u64::MAX);
}

#[test]
fn divu_by_zero_returns_max() {
    assert_eq!(DivU.execute(42, 0), u64::MAX);
}

#[test]
fn rem_by_zero_returns_dividend() {
    assert_eq!(Rem.execute(42, 0), 42);
}

#[test]
fn remu_by_zero_returns_dividend() {
    assert_eq!(RemU.execute(42, 0), 42);
}

#[test]
fn div_normal() {
    let neg5 = (-5i64) as u64;
    let result = Div.execute(neg5, 2);
    assert_eq!(result as i64, -2);
}

// Shift edge cases

#[test]
fn sll_large_shift() {
    assert_eq!(Sll.execute(1, 0), 1);
    assert_eq!(Sll.execute(1, 63), 1u64 << 63);
    assert_eq!(Sll.execute(1, 64), 1); // 64 mod 64 = 0
}

#[test]
fn srl_basic() {
    assert_eq!(Srl.execute(0xFF00, 8), 0xFF);
    assert_eq!(Srl.execute(u64::MAX, 63), 1);
}

#[test]
fn sra_sign_extension() {
    let neg = (-16i64) as u64;
    let result = Sra.execute(neg, 2);
    assert_eq!(result as i64, -4);
}

// Branch instruction semantics

#[test]
fn branch_eq_ne() {
    assert_eq!(Beq.execute(5, 5), 1);
    assert_eq!(Beq.execute(5, 6), 0);
    assert_eq!(Bne.execute(5, 5), 0);
    assert_eq!(Bne.execute(5, 6), 1);
}

#[test]
fn branch_signed_comparison() {
    let neg1 = (-1i64) as u64;
    assert_eq!(Blt.execute(neg1, 0), 1);
    assert_eq!(Bge.execute(neg1, 0), 0);
    assert_eq!(Blt.execute(0, neg1), 0);
    assert_eq!(Bge.execute(0, neg1), 1);
}

#[test]
fn branch_unsigned_comparison() {
    let large = u64::MAX;
    assert_eq!(BltU.execute(large, 0), 0);
    assert_eq!(BgeU.execute(large, 0), 1);
    assert_eq!(BltU.execute(0, large), 1);
    assert_eq!(BgeU.execute(0, large), 0);
}

// W-suffix sign extension

#[test]
fn addw_sign_extends() {
    let result = AddW.execute(0x7FFF_FFFF, 1);
    assert_eq!(result as i64, -2_147_483_648_i64);
    assert_eq!(AddW.execute(3, 5), 8);
}

#[test]
fn subw_sign_extends() {
    let result = SubW.execute(0, 1);
    assert_eq!(result as i64, -1);
}

// Load/store masking

#[test]
fn load_byte_sign_extend() {
    assert_eq!(Lb.execute(0xFF, 0) as i64, -1);
    assert_eq!(Lbu.execute(0xFF, 0), 0xFF);
}

#[test]
fn store_masking() {
    assert_eq!(Sb.execute(0xDEAD_BEEF, 0), 0xEF);
    assert_eq!(Sh.execute(0xDEAD_BEEF, 0), 0xBEEF);
    assert_eq!(Sw.execute(0xDEAD_BEEF, 0), 0xDEAD_BEEF);
}

// Compare instructions

#[test]
fn slt_signed() {
    let neg1 = (-1i64) as u64;
    assert_eq!(Slt.execute(neg1, 0), 1);
    assert_eq!(Slt.execute(0, neg1), 0);
    assert_eq!(Slt.execute(5, 5), 0);
}

#[test]
fn sltu_unsigned() {
    assert_eq!(SltU.execute(0, 1), 1);
    assert_eq!(SltU.execute(1, 0), 0);
    assert_eq!(SltU.execute(u64::MAX, 0), 0);
}

// Logic instructions

#[test]
fn bitwise_operations() {
    assert_eq!(And.execute(0xFF, 0x0F), 0x0F);
    assert_eq!(Or.execute(0xF0, 0x0F), 0xFF);
    assert_eq!(Xor.execute(0xFF, 0xFF), 0x00);
    assert_eq!(Xor.execute(0xFF, 0x00), 0xFF);
}

// Flag consistency

/// ADD has AddOperands and WriteLookupOutputToRD flags.
#[test]
fn add_has_expected_flags() {
    let cf = Add.circuit_flags();
    assert!(cf[CircuitFlags::AddOperands]);
    assert!(cf[CircuitFlags::WriteLookupOutputToRD]);
    assert!(!cf[CircuitFlags::Load]);
    assert!(!cf[CircuitFlags::Store]);
}

/// Loads have Load flag, stores have Store flag.
#[test]
fn load_store_flags() {
    assert!(Lw.circuit_flags()[CircuitFlags::Load]);
    assert!(!Lw.circuit_flags()[CircuitFlags::Store]);

    assert!(Sw.circuit_flags()[CircuitFlags::Store]);
    assert!(!Sw.circuit_flags()[CircuitFlags::Load]);
}

/// Jump instructions have Jump flag.
#[test]
fn jump_flags() {
    assert!(Jal.circuit_flags()[CircuitFlags::Jump]);
}
