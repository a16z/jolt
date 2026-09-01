//! Row-count ratchet for the keccak256 inline expansion.
//!
//! Every instruction in the expanded sequence is one trace row, so this count
//! is the per-permutation prover cost. If a change reduces it, lower the
//! constant; if it regresses, that is a real cost increase to justify.
#![cfg(feature = "host")]

use jolt_inlines_keccak256::{
    INLINE_OPCODE, KECCAK256_ABSORB_PERMUTE_FUNCT3, KECCAK256_FUNCT3, KECCAK256_FUNCT7,
    KECCAK256_NAME,
};
use tracer::utils::inline_test_harness::InlineTestHarness;
use tracer::utils::virtual_registers::VirtualRegisterAllocator;

#[test]
fn count_keccak256_rows() {
    // Force inline registration (inventory) by touching the crate.
    let _ = KECCAK256_NAME;

    let instr = InlineTestHarness::create_default_instruction(
        INLINE_OPCODE,
        KECCAK256_FUNCT3,
        KECCAK256_FUNCT7,
    );
    let sequence = instr.inline_sequence(&VirtualRegisterAllocator::default());
    // 24 rounds x 125 rows (70 XOR + 5 XORROTL1 + 24 ROTRI + 25 ANDN + 1 XORI)
    // + 25 LD + 25 SD + 37 register resets.
    assert_eq!(
        sequence.len(),
        3087,
        "keccak256 inline row count changed; update this ratchet deliberately"
    );

    let absorb_instr = InlineTestHarness::create_default_instruction(
        INLINE_OPCODE,
        KECCAK256_ABSORB_PERMUTE_FUNCT3,
        KECCAK256_FUNCT7,
    );
    let absorb_sequence = absorb_instr.inline_sequence(&VirtualRegisterAllocator::default());
    // The fused entry adds 17 block LD + 17 XOR rows to the permutation.
    assert_eq!(
        absorb_sequence.len(),
        3121,
        "keccak256 absorb-permute row count changed; update this ratchet deliberately"
    );
}
