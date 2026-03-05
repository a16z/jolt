# test-jolt-instructions: Comprehensive tests for jolt-instructions

**Scope:** crates/jolt-instructions/

**Depends:** impl-jolt-instructions

**Verifier:** ./verifiers/scoped.sh /workdir jolt-instructions

**Context:**

Write comprehensive tests for the `jolt-instructions` crate. The implementation task includes basic inline tests. This task adds exhaustive correctness tests, property-based tests for the lookup decomposition, and edge case coverage.

**Do not modify source logic — test-only changes.**

### Test categories

#### 1. Arithmetic instruction correctness

For each arithmetic instruction (ADD, SUB, MUL, etc.), verify that `execute` matches Rust's wrapping arithmetic:

```rust
// Example structure
#[test]
fn test_add_wrapping() {
    let add = Add;
    assert_eq!(add.execute(&[u64::MAX, 1]), 0); // wrapping
    assert_eq!(add.execute(&[0, 0]), 0);
    assert_eq!(add.execute(&[123, 456]), 579);
}
```

Test each instruction with:
- Zero operands
- Max operands (overflow/wrapping)
- Random operands (at least 100 random pairs)
- Boundary values (1, -1 as two's complement, powers of 2)

#### 2. Logic instruction correctness

AND, OR, XOR with:
- All zeros
- All ones
- One operand zero
- Random operands

#### 3. Shift instruction correctness

SLL, SRL, SRA with:
- Shift by 0
- Shift by 63 (max for 64-bit)
- Shift by 1
- SRA with negative (high bit set) values — verify sign extension

#### 4. Branch instruction correctness

BEQ, BNE, BLT, BGE, BLTU, BGEU:
- Equal operands
- First < second (signed and unsigned)
- First > second (signed and unsigned)
- Boundary: 0 vs u64::MAX (unsigned), i64::MIN vs i64::MAX (signed)

#### 5. Lookup decomposition round-trip (property-based)

For every instruction, verify that the lookup decomposition reconstructs correctly:

```rust
// For each instruction `instr` and random operands:
let result = instr.execute(&operands);
let queries = instr.lookups(&operands);
let reconstructed = reconstruct_from_lookups(&queries, &tables);
assert_eq!(result, reconstructed);
```

This is the most critical test — it verifies that the Jolt lookup argument will be sound.

Use `proptest` to generate random operands for each instruction.

#### 6. Lookup table exhaustive tests

For every lookup table with a small domain (≤ 2^16 entries), exhaustively verify every entry:
- Table is deterministic (same input → same output)
- Table covers the full domain
- No panics on any valid input

#### 7. Instruction set completeness

- Verify `JoltInstructionSet` contains all expected opcodes
- Verify no duplicate opcodes
- Verify every instruction is reachable via `instruction(opcode)`

**Acceptance:**

- Every instruction tested with zero/max/boundary/random operands
- Lookup decomposition round-trip tested for every instruction (proptest)
- Small lookup tables exhaustively verified
- Instruction set completeness and uniqueness verified
- At least 200 test cases across all categories
- All tests pass
- No modifications to non-test source code
