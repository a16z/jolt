//! Row-count probe for the keccak256 inline expansion (measurement tool, not a correctness gate).
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
}
