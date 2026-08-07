//! Row-count ratchet for the keccak256 inline expansion.
//!
//! Every instruction in the expanded sequence is one trace row, so this count
//! is the per-permutation prover cost. If a change reduces it, lower the
//! constant; if it regresses, that is a real cost increase to justify.
#![cfg(feature = "host")]

use std::collections::BTreeMap;

use tracer::utils::inline_test_harness::InlineTestHarness;
use tracer::utils::virtual_registers::VirtualRegisterAllocator;

#[test]
fn count_keccak256_rows() {
    // Force inline registration (inventory) by touching the crate.
    let _ = jolt_inlines_keccak256::KECCAK256_NAME;

    let instr = InlineTestHarness::create_default_instruction(
        jolt_inlines_keccak256::INLINE_OPCODE,
        jolt_inlines_keccak256::KECCAK256_FUNCT3,
        jolt_inlines_keccak256::KECCAK256_FUNCT7,
    );
    let sequence = instr.inline_sequence(&VirtualRegisterAllocator::default());

    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    for i in &sequence {
        let dbg = format!("{i:?}");
        let name = dbg
            .split([' ', '(', '{'])
            .next()
            .unwrap_or("unknown")
            .to_string();
        *histogram.entry(name).or_default() += 1;
    }

    println!("TOTAL ROWS: {}", sequence.len());
    for (name, count) in &histogram {
        println!("{count:6}  {name}");
    }

    // 24 rounds x 125 rows (70 XOR + 5 XORROTL1 + 24 ROTRI + 25 ANDN + 1 XORI)
    // + 25 LD + 25 SD + 37 register resets.
    assert_eq!(
        sequence.len(),
        3087,
        "keccak256 inline row count changed; update this ratchet deliberately"
    );
}
