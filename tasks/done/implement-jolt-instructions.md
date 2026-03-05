# impl-jolt-instructions: Clean-room implementation of jolt-instructions

**Scope:** crates/jolt-instructions/

**Depends:** scaffold-workspace

**Verifier:** ./verifiers/scoped.sh /workdir jolt-instructions

**Context:**

Implement the `jolt-instructions` crate — the Jolt instruction set (RISC-V base + virtual instructions) and their decomposition into lookup tables. This crate is Jolt-specific.

**This is a clean-room rewrite.** Study `jolt-core/src/zkvm/instruction/` and `jolt-core/src/zkvm/lookup_table/` for algorithmic reference, but design the API and write the code from scratch.

**Dependency:** `jolt-field` only.

### Reference material

The old code lives in:
- `jolt-core/src/zkvm/instruction/` — 65 instruction files
- `jolt-core/src/zkvm/lookup_table/` — 100+ prefix/suffix/virtual table files

Also read the Jolt Book: https://jolt.a16zcrypto.com/ — sections on instruction execution and lookup arguments.

### Public API contract

```rust
/// A Jolt instruction: a function from operands to a result,
/// decomposed into lookup table queries for the prover.
pub trait Instruction<F: Field>: Send + Sync + 'static {
    fn opcode(&self) -> u32;
    fn name(&self) -> &'static str;
    fn execute(&self, operands: &[u64]) -> u64;
    fn lookups(&self, operands: &[u64]) -> Vec<LookupQuery>;
}

/// A single lookup table query.
pub struct LookupQuery {
    pub table: TableId,
    pub input: u64,
}

/// Identifies a lookup table.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TableId(pub u16);

/// A lookup table: a function from a small input domain to field elements.
pub trait LookupTable<F: Field>: Send + Sync {
    fn id(&self) -> TableId;
    fn name(&self) -> &'static str;
    fn size(&self) -> usize;
    fn evaluate(&self, input: u64) -> F;
    fn materialize(&self) -> Vec<F>;
}

/// The complete Jolt instruction set.
pub struct JoltInstructionSet { ... }

impl JoltInstructionSet {
    pub fn new() -> Self;
    pub fn instruction(&self, opcode: u32) -> Option<&dyn Instruction<F>>;
    pub fn tables(&self) -> &[Box<dyn LookupTable<F>>];
}
```

### Instruction categories

**Standard RISC-V (RV32I/RV64I):**
- Arithmetic: ADD, ADDI, SUB, LUI, AUIPC
- Logic: AND, ANDI, OR, ORI, XOR, XORI
- Shift: SLL, SLLI, SRL, SRLI, SRA, SRAI
- Compare: SLT, SLTI, SLTU, SLTIU
- Branch: BEQ, BNE, BLT, BGE, BLTU, BGEU
- Jump: JAL, JALR
- Load: LB, LBU, LH, LHU, LW, LWU, LD
- Store: SB, SH, SW, SD
- System: ECALL, EBREAK, FENCE

**Virtual (Jolt-specific):**
- ASSERT_EQ, ASSERT_LTE
- POW2, MOVSIGN
- ROTRIW, XOR_ROT, etc.

**Lookup table categories:**
- Prefix tables: operate on upper bits (AND, OR, XOR, LT, GT, shifts, etc.)
- Suffix tables: operate on lower bits
- Virtual tables: composite operations

### Implementation notes

- Each instruction is a unit struct implementing `Instruction<F>`. Group related instructions into modules (arithmetic.rs, logic.rs, etc.).
- `execute` must match Rust's wrapping arithmetic exactly — this is the correctness oracle.
- `lookups` decomposes the operation into small-domain table queries. Study the old decomposition logic carefully — this is where Jolt's efficiency comes from.
- Lookup tables can be implemented as simple functions or precomputed arrays depending on the domain size.
- All instruction structs and lookup table structs should derive `Serialize, Deserialize`.

### File structure

```
jolt-instructions/src/
├── lib.rs
├── traits.rs           # Instruction, LookupTable traits
├── instruction_set.rs  # JoltInstructionSet
├── rv/                 # Standard RISC-V instructions
│   ├── mod.rs
│   ├── arithmetic.rs   # ADD, SUB, etc.
│   ├── logic.rs        # AND, OR, XOR
│   ├── shift.rs        # SLL, SRL, SRA
│   ├── compare.rs      # SLT, SLTU
│   ├── branch.rs       # BEQ, BNE, etc.
│   ├── jump.rs         # JAL, JALR
│   ├── load.rs         # LB, LW, LD, etc.
│   ├── store.rs        # SB, SW, SD
│   └── system.rs       # ECALL, EBREAK, FENCE
├── virtual_/           # Virtual instructions
│   ├── mod.rs
│   ├── assert.rs
│   ├── bitwise.rs
│   └── arithmetic.rs
└── tables/             # Lookup tables
    ├── mod.rs
    ├── prefix/
    ├── suffix/
    └── virtual_/
```

### Documentation standard

- Rustdoc on every public item
- Each instruction's doc comment explains the RISC-V semantics and the lookup decomposition strategy
- Reference the Jolt Book and RISC-V ISA spec

**Acceptance:**

- All 65 instructions implemented with correct `execute` behavior
- `execute` matches wrapping native Rust arithmetic for all arithmetic/logic/shift operations
- All lookup tables implemented and materialized correctly
- `lookups` decomposition for each instruction reconstructs to the correct `execute` result
- `JoltInstructionSet` provides lookup by opcode
- All types `Serialize + Deserialize`
- No file exceeds 500 lines — split large instruction categories across files
- Rustdoc on all public items
- `cargo clippy` clean
- Basic unit tests for each instruction category inline
