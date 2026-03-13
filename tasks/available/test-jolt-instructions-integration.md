# test-jolt-instructions-integration: Integration tests for jolt-instructions

**Scope:** crates/jolt-instructions/tests/

**Depends:** impl-jolt-instructions, test-jolt-instructions

**Verifier:** ./verifiers/scoped.sh /workdir jolt-instructions

**Context:**

Write integration tests for the `jolt-instructions` crate that verify the complete instruction set and lookup table functionality from an external perspective.

### Integration Test Files

Create the following test files in `crates/jolt-instructions/tests/`:

#### 1. `instruction_set.rs` - Test complete instruction set

Test the full instruction set as a cohesive unit:
- Verify all RISC-V base instructions are present
- Test instruction lookup by opcode
- Verify virtual instructions are accessible
- Test instruction categorization (arithmetic, logic, branch, etc.)
- Ensure no opcode conflicts

#### 2. `lookup_tables.rs` - Test lookup table consistency

Verify lookup tables are consistent with instruction execution:
- For each instruction, verify `lookups()` followed by table evaluation reconstructs `execute()` result
- Test that all referenced tables exist in the instruction set
- Verify table sizes match expected domains
- Test prefix/suffix decomposition correctness

#### 3. `decomposition.rs` - Test instruction decomposition

Test the decomposition of instructions into lookup queries:
- Verify arithmetic instructions decompose correctly
- Test boundary cases (overflow, underflow)
- Verify shift amounts are handled correctly
- Test sign extension for signed operations
- Verify branch condition evaluation

### Test Structure

Each test file should:
- Import the crate as external: `use jolt_instructions::*;`
- Test the public API comprehensively
- Focus on consistency between components
- Verify invariants that must hold

### Specific Test Cases

**Instruction Set Coverage:**
```rust
#[test]
fn test_all_rv64i_instructions_present() {
    let instruction_set = JoltInstructionSet::new();

    // List of all RV64I opcodes
    let rv64i_opcodes = vec![
        /* ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, ... */
    ];

    for opcode in rv64i_opcodes {
        assert!(
            instruction_set.instruction(opcode).is_some(),
            "Missing RV64I instruction with opcode {:#x}", opcode
        );
    }
}
```

**Lookup Consistency:**
```rust
#[test]
fn test_add_instruction_lookup_consistency() {
    let instruction_set = JoltInstructionSet::new();
    let add = instruction_set.instruction(ADD_OPCODE).unwrap();

    // Test various operand combinations
    let test_cases = vec![
        (1, 2),
        (u64::MAX, 1), // overflow
        (0, 0),
        // ... more cases
    ];

    for (a, b) in test_cases {
        let result = add.execute(&[a, b]);
        let lookups = add.lookups(&[a, b]);

        // Reconstruct result from lookups
        let reconstructed = reconstruct_from_lookups(
            &lookups,
            &instruction_set.tables()
        );

        assert_eq!(result, reconstructed,
            "Lookup reconstruction failed for ADD({}, {})", a, b
        );
    }
}
```

**Decomposition Properties:**
```rust
#[test]
fn test_shift_decomposition_bounds() {
    let instruction_set = JoltInstructionSet::new();

    // Test all shift instructions
    for opcode in [SLL_OPCODE, SRL_OPCODE, SRA_OPCODE] {
        let shift_instr = instruction_set.instruction(opcode).unwrap();

        // Shift amount should be masked to 6 bits for RV64
        let value = 0x123456789ABCDEF0u64;
        let shift = 100; // > 63, should be masked

        let result = shift_instr.execute(&[value, shift]);
        let result_masked = shift_instr.execute(&[value, shift & 0x3F]);

        assert_eq!(result, result_masked,
            "Shift instruction should mask shift amount"
        );
    }
}
```

**Lookup Table Properties:**
```rust
#[test]
fn test_lookup_table_domains() {
    let instruction_set = JoltInstructionSet::new();

    for table in instruction_set.tables() {
        let size = table.size();

        // Verify table can be evaluated at all indices in domain
        for i in 0..size {
            let _ = table.evaluate(i as u64); // Should not panic
        }

        // Verify materialized table has correct size
        let materialized = table.materialize();
        assert_eq!(materialized.len(), size,
            "Table {} has inconsistent size", table.name()
        );
    }
}
```

### Edge Cases to Test

- Arithmetic overflow/underflow
- Division by zero behavior
- Sign extension edge cases
- Maximum shift amounts
- Branch conditions at boundaries
- Memory operation alignment

### Acceptance Criteria

- All three integration test files created and passing
- Comprehensive coverage of instruction set
- Lookup table consistency verified
- Decomposition correctness tested
- Edge cases covered
- Tests well-documented
- All tests pass with `cargo nextest run -p jolt-instructions`
- No source code modifications