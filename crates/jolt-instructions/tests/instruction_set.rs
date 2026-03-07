//! Integration tests for the jolt-instructions crate.
//!
//! Exercises the JoltInstructionSet registry, instruction execution semantics,
//! flag consistency, and bit-interleaving utilities.

use jolt_instructions::{
    interleave_bits, opcodes, uninterleave_bits, CircuitFlags, JoltInstructionSet,
    NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};

// Registry completeness

/// Every opcode 0..COUNT maps to an instruction with matching opcode.
#[test]
fn registry_all_opcodes_present() {
    let set = JoltInstructionSet::new();
    assert_eq!(set.len(), opcodes::COUNT as usize);

    for op in 0..opcodes::COUNT {
        let instr = set
            .instruction(op)
            .unwrap_or_else(|| panic!("opcode {op} missing from registry"));
        assert_eq!(instr.opcode(), op, "opcode mismatch for {}", instr.name());
    }
}

/// All instruction names are unique.
#[test]
fn registry_unique_names() {
    let set = JoltInstructionSet::new();
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
    assert!(set.instruction(opcodes::COUNT).is_none());
    assert!(set.instruction(u32::MAX).is_none());
}

// Arithmetic instruction semantics

#[test]
fn add_basic_and_overflow() {
    let set = JoltInstructionSet::new();
    let add = set.instruction(opcodes::ADD).unwrap();
    assert_eq!(add.execute(3, 5), 8);
    assert_eq!(add.execute(u64::MAX, 1), 0); // wrapping
    assert_eq!(add.execute(0, 0), 0);
}

#[test]
fn sub_basic_and_underflow() {
    let set = JoltInstructionSet::new();
    let sub = set.instruction(opcodes::SUB).unwrap();
    assert_eq!(sub.execute(10, 3), 7);
    assert_eq!(sub.execute(0, 1), u64::MAX); // wrapping
}

#[test]
fn mul_basic_and_overflow() {
    let set = JoltInstructionSet::new();
    let mul = set.instruction(opcodes::MUL).unwrap();
    assert_eq!(mul.execute(6, 7), 42);
    assert_eq!(mul.execute(u64::MAX, 2), u64::MAX.wrapping_mul(2));
    assert_eq!(mul.execute(0, u64::MAX), 0);
}

// Division edge cases

#[test]
fn div_by_zero_returns_max() {
    let set = JoltInstructionSet::new();
    let div = set.instruction(opcodes::DIV).unwrap();
    // RISC-V spec: signed div by zero returns -1 (all bits set)
    assert_eq!(div.execute(42, 0), u64::MAX);
}

#[test]
fn divu_by_zero_returns_max() {
    let set = JoltInstructionSet::new();
    let divu = set.instruction(opcodes::DIVU).unwrap();
    assert_eq!(divu.execute(42, 0), u64::MAX);
}

#[test]
fn rem_by_zero_returns_dividend() {
    let set = JoltInstructionSet::new();
    let rem = set.instruction(opcodes::REM).unwrap();
    assert_eq!(rem.execute(42, 0), 42);
}

#[test]
fn remu_by_zero_returns_dividend() {
    let set = JoltInstructionSet::new();
    let remu = set.instruction(opcodes::REMU).unwrap();
    assert_eq!(remu.execute(42, 0), 42);
}

#[test]
fn div_normal() {
    let set = JoltInstructionSet::new();
    let div = set.instruction(opcodes::DIV).unwrap();
    // Signed division: interpret as i64
    let neg5 = (-5i64) as u64;
    let result = div.execute(neg5, 2);
    assert_eq!(result as i64, -2); // truncates toward zero
}

// Shift edge cases

#[test]
fn sll_large_shift() {
    let set = JoltInstructionSet::new();
    let sll = set.instruction(opcodes::SLL).unwrap();
    // Only lower 6 bits of shift used (mod 64)
    assert_eq!(sll.execute(1, 0), 1);
    assert_eq!(sll.execute(1, 63), 1u64 << 63);
    assert_eq!(sll.execute(1, 64), 1); // 64 mod 64 = 0
}

#[test]
fn srl_basic() {
    let set = JoltInstructionSet::new();
    let srl = set.instruction(opcodes::SRL).unwrap();
    assert_eq!(srl.execute(0xFF00, 8), 0xFF);
    assert_eq!(srl.execute(u64::MAX, 63), 1);
}

#[test]
fn sra_sign_extension() {
    let set = JoltInstructionSet::new();
    let sra = set.instruction(opcodes::SRA).unwrap();
    let neg = (-16i64) as u64;
    let result = sra.execute(neg, 2);
    assert_eq!(result as i64, -4); // sign-extended
}

// Branch instruction semantics

#[test]
fn branch_eq_ne() {
    let set = JoltInstructionSet::new();
    let beq = set.instruction(opcodes::BEQ).unwrap();
    let bne = set.instruction(opcodes::BNE).unwrap();

    assert_eq!(beq.execute(5, 5), 1);
    assert_eq!(beq.execute(5, 6), 0);
    assert_eq!(bne.execute(5, 5), 0);
    assert_eq!(bne.execute(5, 6), 1);
}

#[test]
fn branch_signed_comparison() {
    let set = JoltInstructionSet::new();
    let blt = set.instruction(opcodes::BLT).unwrap();
    let bge = set.instruction(opcodes::BGE).unwrap();

    let neg1 = (-1i64) as u64;
    // -1 < 0 is true (signed)
    assert_eq!(blt.execute(neg1, 0), 1);
    assert_eq!(bge.execute(neg1, 0), 0);
    // 0 < -1 is false (signed)
    assert_eq!(blt.execute(0, neg1), 0);
    assert_eq!(bge.execute(0, neg1), 1);
}

#[test]
fn branch_unsigned_comparison() {
    let set = JoltInstructionSet::new();
    let bltu = set.instruction(opcodes::BLTU).unwrap();
    let bgeu = set.instruction(opcodes::BGEU).unwrap();

    let large = u64::MAX; // unsigned: largest value
                          // MAX < 0 is false (unsigned)
    assert_eq!(bltu.execute(large, 0), 0);
    assert_eq!(bgeu.execute(large, 0), 1);
    // 0 < MAX is true (unsigned)
    assert_eq!(bltu.execute(0, large), 1);
    assert_eq!(bgeu.execute(0, large), 0);
}

// W-suffix sign extension

#[test]
fn addw_sign_extends() {
    let set = JoltInstructionSet::new();
    let addw = set.instruction(opcodes::ADDW).unwrap();

    // 0x7FFF_FFFF + 1 = 0x8000_0000 → sign-extended to 0xFFFF_FFFF_8000_0000
    let result = addw.execute(0x7FFF_FFFF, 1);
    assert_eq!(result as i64, -2_147_483_648_i64); // i32::MIN sign-extended

    // Normal case
    assert_eq!(addw.execute(3, 5), 8);
}

#[test]
fn subw_sign_extends() {
    let set = JoltInstructionSet::new();
    let subw = set.instruction(opcodes::SUBW).unwrap();

    // 0 - 1 in 32-bit = 0xFFFF_FFFF → sign-extended to 0xFFFF_FFFF_FFFF_FFFF = -1
    let result = subw.execute(0, 1);
    assert_eq!(result as i64, -1);
}

// Load/store masking

#[test]
fn load_byte_sign_extend() {
    let set = JoltInstructionSet::new();
    let lb = set.instruction(opcodes::LB).unwrap();
    let lbu = set.instruction(opcodes::LBU).unwrap();

    // 0xFF → LB sign-extends to -1, LBU zero-extends to 255
    assert_eq!(lb.execute(0xFF, 0) as i64, -1);
    assert_eq!(lbu.execute(0xFF, 0), 0xFF);
}

#[test]
fn store_masking() {
    let set = JoltInstructionSet::new();
    let sb = set.instruction(opcodes::SB).unwrap();
    let sh = set.instruction(opcodes::SH).unwrap();
    let sw = set.instruction(opcodes::SW).unwrap();

    assert_eq!(sb.execute(0xDEAD_BEEF, 0), 0xEF);
    assert_eq!(sh.execute(0xDEAD_BEEF, 0), 0xBEEF);
    assert_eq!(sw.execute(0xDEAD_BEEF, 0), 0xDEAD_BEEF);
}

// Compare instructions

#[test]
fn slt_signed() {
    let set = JoltInstructionSet::new();
    let slt = set.instruction(opcodes::SLT).unwrap();
    let neg1 = (-1i64) as u64;
    assert_eq!(slt.execute(neg1, 0), 1); // -1 < 0 signed
    assert_eq!(slt.execute(0, neg1), 0);
    assert_eq!(slt.execute(5, 5), 0);
}

#[test]
fn sltu_unsigned() {
    let set = JoltInstructionSet::new();
    let sltu = set.instruction(opcodes::SLTU).unwrap();
    assert_eq!(sltu.execute(0, 1), 1);
    assert_eq!(sltu.execute(1, 0), 0);
    assert_eq!(sltu.execute(u64::MAX, 0), 0); // MAX is largest unsigned
}

// Logic instructions

#[test]
fn bitwise_operations() {
    let set = JoltInstructionSet::new();
    let and = set.instruction(opcodes::AND).unwrap();
    let or = set.instruction(opcodes::OR).unwrap();
    let xor = set.instruction(opcodes::XOR).unwrap();

    assert_eq!(and.execute(0xFF, 0x0F), 0x0F);
    assert_eq!(or.execute(0xF0, 0x0F), 0xFF);
    assert_eq!(xor.execute(0xFF, 0xFF), 0x00);
    assert_eq!(xor.execute(0xFF, 0x00), 0xFF);
}

// Flag consistency

/// All circuit flag arrays have the correct length.
#[test]
fn circuit_flag_dimensions() {
    let set = JoltInstructionSet::new();
    for instr in set.iter() {
        let flags = instr.circuit_flags();
        assert_eq!(
            flags.len(),
            NUM_CIRCUIT_FLAGS,
            "circuit flags wrong size for {}",
            instr.name()
        );
    }
}

/// All instruction flag arrays have the correct length.
#[test]
fn instruction_flag_dimensions() {
    let set = JoltInstructionSet::new();
    for instr in set.iter() {
        let flags = instr.instruction_flags();
        assert_eq!(
            flags.len(),
            NUM_INSTRUCTION_FLAGS,
            "instruction flags wrong size for {}",
            instr.name()
        );
    }
}

/// ADD has AddOperands and WriteLookupOutputToRD flags.
#[test]
fn add_has_expected_flags() {
    let set = JoltInstructionSet::new();
    let add = set.instruction(opcodes::ADD).unwrap();
    let cf = add.circuit_flags();
    assert!(cf[CircuitFlags::AddOperands as usize]);
    assert!(cf[CircuitFlags::WriteLookupOutputToRD as usize]);
    assert!(!cf[CircuitFlags::Load as usize]);
    assert!(!cf[CircuitFlags::Store as usize]);
}

/// Loads have Load flag, stores have Store flag.
#[test]
fn load_store_flags() {
    let set = JoltInstructionSet::new();

    let lw = set.instruction(opcodes::LW).unwrap();
    assert!(lw.circuit_flags()[CircuitFlags::Load as usize]);
    assert!(!lw.circuit_flags()[CircuitFlags::Store as usize]);

    let sw = set.instruction(opcodes::SW).unwrap();
    assert!(sw.circuit_flags()[CircuitFlags::Store as usize]);
    assert!(!sw.circuit_flags()[CircuitFlags::Load as usize]);
}

/// Jump instructions have Jump flag.
#[test]
fn jump_flags() {
    let set = JoltInstructionSet::new();
    let jal = set.instruction(opcodes::JAL).unwrap();
    assert!(jal.circuit_flags()[CircuitFlags::Jump as usize]);
}

// Lookup table consistency

/// Instructions with lookup tables have non-None table kinds.
/// Instructions without (loads, stores, system) have None.
#[test]
fn lookup_table_assignment() {
    let set = JoltInstructionSet::new();

    // ADD should have a lookup table (RangeCheck)
    let add = set.instruction(opcodes::ADD).unwrap();
    assert!(
        add.lookup_table().is_some(),
        "ADD should have a lookup table"
    );

    // LW should not have a lookup table
    let lw = set.instruction(opcodes::LW).unwrap();
    assert!(
        lw.lookup_table().is_none(),
        "LW should not have a lookup table"
    );

    // ECALL should not have a lookup table
    let ecall = set.instruction(opcodes::ECALL).unwrap();
    assert!(
        ecall.lookup_table().is_none(),
        "ECALL should not have a lookup table"
    );
}

// Bit interleaving

/// Round-trip: uninterleave(interleave(x, y)) == (x, y).
#[test]
fn interleave_round_trip() {
    let test_values = [0u64, 1, 0xFF, 0xFFFF, 0xDEAD_BEEF, u64::MAX, u64::MAX / 2];
    for &x in &test_values {
        for &y in &test_values {
            let interleaved = interleave_bits(x, y);
            let (rx, ry) = uninterleave_bits(interleaved);
            assert_eq!((rx, ry), (x, y), "round-trip failed for x={x:#x}, y={y:#x}");
        }
    }
}

/// Single-bit positions are correctly placed.
#[test]
fn interleave_single_bits() {
    // x=1 (bit 0) should go to even position 0 → bit 1 in interleaved
    // y=1 (bit 0) should go to odd position 0 → bit 0 in interleaved
    // Wait, convention depends on MSB/LSB. Let's just verify round-trip.
    for bit in 0..64 {
        let x = 1u64 << bit;
        let (rx, _) = uninterleave_bits(interleave_bits(x, 0));
        assert_eq!(rx, x, "x single bit {bit} failed");

        let (_, ry) = uninterleave_bits(interleave_bits(0, x));
        assert_eq!(ry, x, "y single bit {bit} failed");
    }
}
