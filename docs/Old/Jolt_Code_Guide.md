# Jolt Code Guide: From Theory to Implementation

**Version**: v0.2.0+ (Twist and Shout era)
**Audience**: Developers who've read the theory ([Jolt.md](Jolt.md)) and want to contribute to the codebase
**Goal**: Bridge theory and practice by showing *where* concepts live in code and *how* they connect

---

## How to Use This Guide

This guide assumes you understand the theory from [Jolt.md](Jolt.md):
- Sumcheck protocol and why it's used
- Multilinear extensions (MLEs) and polynomial commitments
- Twist (memory checking) and Shout (lookup arguments)
- The five Jolt components (R1CS, RAM, Registers, Instructions, Bytecode)

**What we add here**: Concrete code locations, data structures, and execution flow.

**Learning Path**:
1. **First time?** Read sections 1-3 (big picture + code layout)
2. **Adding a feature?** Jump to section 12 (Contributing Guide)
3. **Debugging?** See section 12.7 for workflows
4. **Understanding a component?** Pick from section 8

**This guide is NOT**:
- ❌ A Rust tutorial or ZK primer
- ❌ Line-by-line code walkthrough
- ❌ A replacement for reading theory

**This guide IS**:
- ✅ A map connecting theory to implementation
- ✅ Guidance on data flow and file structure
- ✅ Practical advice with code examples

---

## Table of Contents

1. [The Big Picture: Mental Model](#1-the-big-picture-mental-model)
2. [How a Proof Happens: Following the Flow](#2-how-a-proof-happens-following-the-flow)
3. [Understanding the Codebase Layout](#3-understanding-the-codebase-layout)
4. [Guest Programs: What Gets Proven](#4-guest-programs-what-gets-proven)
5. [Trace Generation: From Execution to Data](#5-trace-generation-from-execution-to-data)
6. [Witness Construction: Data to Polynomials](#6-witness-construction-data-to-polynomials)
7. [The Proof DAG: How Components Interact](#7-the-proof-dag-how-components-interact)
8. [The Five Components Explained](#8-the-five-components-explained)
9. [Key Abstractions and Patterns](#9-key-abstractions-and-patterns)
10. [State Management: The Glue](#10-state-management-the-glue)
11. [Common Misconceptions](#11-common-misconceptions)
12. [Contributing: Practical Guide](#12-contributing-practical-guide)

---

## 1. The Big Picture: Mental Model

### 1.1 What Jolt Actually Does

**Think of Jolt as a CPU emulator + proof generator**:

```
┌─────────────────────────────────────────────────┐
│  YOU WRITE: Rust function                      │
│  fn fib(n: u32) -> u128 { ... }                │
└────────────┬────────────────────────────────────┘
             │
             ↓ Compilation
┌─────────────────────────────────────────────────┐
│  JOLT SEES: RISC-V machine code                │
│  [ADD, MUL, LOAD, STORE, ...]                  │
└────────────┬────────────────────────────────────┘
             │
             ↓ Emulation (tracer/src/emulator/)
┌─────────────────────────────────────────────────┐
│  JOLT RECORDS: Execution trace                 │
│  Cycle 1: ADD r1, r2, r3                       │
│  Cycle 2: LOAD r4, [r1+8]                      │
│  ...                                            │
└────────────┬────────────────────────────────────┘
             │
             ↓ Witness gen (jolt-core/src/zkvm/witness.rs)
┌─────────────────────────────────────────────────┐
│  JOLT PRODUCES: zkSNARK proof                  │
│  "This trace came from executing bytecode B    │
│   with inputs I, producing outputs O"          │
└─────────────────────────────────────────────────┘
```

**Key Insight**: You never write "proof code" directly. You write normal Rust, and Jolt's machinery converts execution into a proof.

### 1.2 The Three Worlds of Jolt

As a contributor, you'll work in three distinct "worlds":

| World | What Lives Here | File Patterns | Key Files |
|-------|----------------|---------------|-----------|
| **Guest** | Code to be proven | `examples/*/guest/` | `guest/src/lib.rs` |
| **Host** | Proof orchestration | `examples/*/host/`, `jolt-core/src/zkvm/` | `jolt-core/src/zkvm/dag/jolt_dag.rs` |
| **Tracer** | RISC-V emulation | `tracer/src/` | `tracer/src/emulator/cpu.rs` |

**Mental Model**:
- **Guest** = "Inside the zkVM" (what gets proven)
- **Host** = "Outside the zkVM" (who orchestrates the proof)
- **Tracer** = "The zkVM itself" (how we execute and record)

### 1.3 The Macro Magic: Entry Point for Contributors

When you write:
```rust
#[jolt::provable]
fn my_function(x: u32) -> u64 { x as u64 * 2 }
```

**Location**: `jolt-sdk/macros/src/lib.rs`

The macro generates **11 functions** (see lines 404-830):

**Core functions you'll use**:
```rust
// 1. Compilation
compile_my_function(target_dir: &str) -> Program
  // -> jolt-core/src/host/program.rs:build_with_channel()

// 2. Preprocessing
preprocess_prover_my_function(&mut program) -> JoltProverPreprocessing
  // -> jolt-core/src/zkvm/mod.rs:prover_preprocess()

// 3. Proving
prove_my_function(input) -> (output, proof, program_io)
  // -> jolt-core/src/zkvm/dag/jolt_dag.rs:prove()

// 4. Verification
verify_my_function(input, output, proof) -> bool
  // -> jolt-core/src/zkvm/dag/jolt_dag.rs:verify()
```

**As a contributor**: You rarely touch macro internals. You interact with the *generated* functions in host programs or modify the proof system itself.

---

## 2. How a Proof Happens: Following the Flow

Let's trace what happens when you prove `fibonacci(10)`:

### 2.1 The Six Stages

```
Stage 0: COMPILATION
┌──────────────────────────────────────────┐
│ You: cargo run --release                 │
│ Macro: Compiles guest to RISC-V ELF     │
│ Output: Binary file with bytecode       │
└──────────────────────────────────────────┘
Files: jolt-sdk/macros/src/lib.rs:404-432
       jolt-core/src/host/program.rs:build_with_channel()

Stage 1: PREPROCESSING (once per program)
┌──────────────────────────────────────────┐
│ Parse ELF → Extract bytecode            │
│ Generate Dory SRS (crypto keys)         │
│ Commit to bytecode                       │
│ Output: Preprocessing data for P and V  │
└──────────────────────────────────────────┘
Files: jolt-core/src/zkvm/mod.rs:prover_preprocess()
       jolt-core/src/zkvm/bytecode/mod.rs:preprocess()
Why: Verifier needs to know what program was executed

Stage 2: EXECUTION
┌──────────────────────────────────────────┐
│ RISC-V emulator runs bytecode           │
│ Records every cycle's state              │
│ Output: Execution trace (Vec<Cycle>)    │
└──────────────────────────────────────────┘
Files: tracer/src/emulator/cpu.rs:step()
       tracer/src/instruction/*.rs (per-instruction execute())
Think of it as: Detailed flight recorder for CPU

Stage 3: WITNESS GENERATION
┌──────────────────────────────────────────┐
│ Trace → Multilinear polynomials         │
│ Parallel construction of ~30 MLEs       │
│ Commit to all polynomials               │
│ Output: Polynomial commitments          │
└──────────────────────────────────────────┘
Files: jolt-core/src/zkvm/witness.rs:generate_witness_batch()
       jolt-core/src/poly/commitment/dory.rs:batch_commit()
Why: Proofs work over polynomials, not raw data

Stage 4: PROOF GENERATION (The DAG)
┌──────────────────────────────────────────┐
│ 5 stages of batched sumchecks            │
│ Each component proves its correctness    │
│ Final batched opening proof             │
│ Output: Proof object                     │
└──────────────────────────────────────────┘
Files: jolt-core/src/zkvm/dag/jolt_dag.rs:prove()
       jolt-core/src/subprotocols/sumcheck.rs:BatchedSumcheck::prove()
This is where the magic happens (80% of proving time)

Stage 5: VERIFICATION
┌──────────────────────────────────────────┐
│ Replay sumcheck verification            │
│ Check batched opening proof             │
│ Output: Accept or Reject                │
└──────────────────────────────────────────┘
Files: jolt-core/src/zkvm/dag/jolt_dag.rs:verify()
       jolt-core/src/subprotocols/sumcheck.rs:BatchedSumcheck::verify()
Fast (~milliseconds for small programs)
```

### 2.2 Key Data Transformations

Understanding these transformations is crucial:

**Rust function → RISC-V bytecode**
- Why: We prove RISC-V execution, not Rust directly
- Tool: `rustc` with riscv64gc target
- Code: `jolt-core/src/host/program.rs:decode()`
- Output: `Program { bytecode: Vec<ELFInstruction>, ... }`

**RISC-V bytecode → Execution trace**
- Why: Need concrete values to build polynomials
- Tool: Tracer emulator
- Code: `tracer/src/emulator/cpu.rs:execute_program()`
- Output: `Vec<Cycle>` where `Cycle` contains:
  ```rust
  pub struct RISCVCycle<I: RISCVInstruction> {
      pub instruction: I,
      pub register_state: RegisterState,
      pub ram_access: I::RAMAccess,
      pub pc: u64,
  }
  ```

**Execution trace → Multilinear polynomials**
- Why: Sumcheck operates over polynomials
- Tool: Witness generation (parallel via Rayon)
- Code: `jolt-core/src/zkvm/witness.rs:CommittedPolynomial::generate_witness_batch()`
- Output: `HashMap<CommittedPolynomial, DenseMultilinearExtension<F>>`
  - ~30 committed polynomials representing all trace data
  - Each is an MLE: `DenseMultilinearExtension { evaluations: Vec<F>, num_vars: usize }`

**Polynomials → Proof**
- Why: This is what gets sent to the verifier
- Tool: DAG execution (5 stages of sumchecks)
- Code: `jolt-core/src/zkvm/dag/jolt_dag.rs:prove()`
- Output: `JoltProof` containing:
  ```rust
  pub struct JoltProof<F, PCS, ProofTranscript> {
      pub commitments: Vec<PCS::Commitment>,
      pub stage_proofs: HashMap<ProofKeys, ProofData>,
      pub opening_proof: PCS::Proof,
      // ... auxiliary data
  }
  ```
  - Size: ~100KB - 1MB depending on trace length

---

## 3. Understanding the Codebase Layout

### 3.1 Crate Structure with Key Files

```
jolt/
├── jolt-core/           ← The heart of the prover
│   ├── src/zkvm/        ← Components (R1CS, RAM, Registers, etc.)
│   │   ├── dag/
│   │   │   ├── jolt_dag.rs        ← Main proof orchestration
│   │   │   ├── state_manager.rs   ← Data coordination hub
│   │   │   └── stage.rs            ← SumcheckStages trait
│   │   ├── r1cs/
│   │   │   └── constraints.rs      ← ~30 R1CS constraint definitions
│   │   ├── ram/
│   │   │   └── mod.rs              ← Twist for memory
│   │   ├── registers/
│   │   │   └── mod.rs              ← Twist for registers
│   │   ├── instruction_lookups/
│   │   │   └── read_raf_checking.rs ← Shout for instructions
│   │   ├── bytecode/
│   │   │   └── mod.rs              ← Shout for bytecode
│   │   └── witness.rs              ← Trace → polynomial conversion
│   ├── src/poly/        ← Polynomial operations, commitments
│   │   ├── dense_mlpoly.rs         ← DenseMultilinearExtension
│   │   └── commitment/
│   │       └── dory.rs             ← Dory PCS integration
│   └── src/subprotocols/← Sumcheck, Twist, Shout implementations
│       ├── sumcheck.rs             ← Core sumcheck engine
│       ├── twist.rs                ← Memory checking protocol
│       └── shout/                  ← Lookup argument
│
├── jolt-sdk/            ← What guest programs import
│   └── macros/src/lib.rs           ← The #[jolt::provable] macro
│
├── tracer/              ← RISC-V emulator
│   ├── src/emulator/
│   │   └── cpu.rs                  ← CPU state and execution loop
│   └── src/instruction/
│       ├── mod.rs                  ← Instruction enum and decode
│       ├── add.rs                  ← ADD instruction (example)
│       └── virtual_*.rs            ← Virtual sequences
│
├── examples/            ← Example programs
│   └── fibonacci/
│       ├── guest/src/lib.rs        ← Code to be proven
│       └── host/src/main.rs        ← Proof orchestration
│
└── common/              ← Shared types and utilities
```

### 3.2 Navigation Tips with File References

**"I want to understand how X is proven..."**

| X | Theory Location | Code Location |
|---|-----------------|---------------|
| Instruction execution | Jolt.md: Lookup-centric architecture | `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` |
| Memory correctness | Jolt.md: Twist protocol | `jolt-core/src/zkvm/ram/mod.rs` + `registers/mod.rs` |
| PC updates | Jolt.md: R1CS constraints | `jolt-core/src/zkvm/r1cs/constraints.rs:UNIFORM_R1CS` |
| Bytecode decoding | Jolt.md: Offline memory checking | `jolt-core/src/zkvm/bytecode/mod.rs` |
| Sumcheck itself | Theory/The Sum-check Protocol.md | `jolt-core/src/subprotocols/sumcheck.rs` |

**"I want to add/modify an instruction..."**

1. **Tracer side** (execution): `tracer/src/instruction/your_instr.rs`
   - Implement `RISCVInstruction::execute()` trait
   - Example: `tracer/src/instruction/add.rs`

2. **Prover side** (proving): `jolt-core/src/zkvm/instruction/your_instr.rs`
   - Implement `InstructionLookup` trait
   - Example: `jolt-core/src/zkvm/instruction/add.rs`

3. **Lookup table** (if needed): `jolt-core/src/zkvm/lookup_table/your_table.rs`
   - Implement `JoltLookupTable::evaluate_mle()`
   - Example: `jolt-core/src/zkvm/lookup_table/range_check.rs`

**"I'm debugging a proof failure..."**

1. Check `StateManager` state: `jolt-core/src/zkvm/dag/state_manager.rs`
2. Enable logging: `RUST_LOG=debug cargo test`
3. Look at profiling: `cargo run --release --features pprof -p jolt-core profile --name sha3`

---

## 4. Guest Programs: What Gets Proven

### 4.1 The Guest Perspective

**Mental Model**: You're writing code for a special computer (RISC-V) that records everything it does.

**Example**: `examples/fibonacci/guest/src/lib.rs`

```rust
#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable]
fn add_numbers(a: u32, b: u32) -> u32 {
    a + b  // This becomes RISC-V ADD instruction
}
```

**What happens behind the scenes**:

1. **Macro generates RISC-V entry point**
   - Location: `jolt-sdk/macros/src/lib.rs:710-830`
   - Creates `main()` that reads from memory-mapped I/O

2. **rustc compiles to RISC-V machine code**
   - Target: `riscv64gc-unknown-none-elf`
   - Output: ELF binary with `.text` section

3. **Tracer executes and records**
   - Location: `tracer/src/emulator/cpu.rs:execute_program()`
   - Returns: `Vec<Cycle>` with complete execution history

4. **Prover generates proof**
   - Location: `jolt-core/src/zkvm/dag/jolt_dag.rs:prove()`
   - Proves execution was correct

### 4.2 Guest Constraints

**Why `no_std`?**
- Guest runs in minimal environment
- No file system, no threads, no networking
- Only CPU + memory

**How to use heap allocation**:
```rust
extern crate alloc;
use alloc::vec::Vec;  // ✅ Works!

#[jolt::provable]
fn use_heap(n: u32) -> u32 {
    let v = Vec::from_iter(0..n);  // Allocates on heap
    v.iter().sum()
}
```

**Code**: Heap allocation via custom allocator defined in macro-generated guest code (`jolt-sdk/macros/src/lib.rs:720-750`)

**Macro parameters you should know**:
```rust
#[jolt::provable(
    stack_size = 10000,        // If you get stack overflow
    memory_size = 10000000,    // If you run out of heap
    max_trace_length = 100000, // If execution is very long
)]
```

**Location**: Parsed in `jolt-sdk/macros/src/lib.rs:80-110`

### 4.3 Debugging Guest Code

**Problem**: Guest code runs in RISC-V emulator, not natively

**Solutions**:

**1. Print statements** (appear during tracing):
```rust
use jolt::{jolt_println};

#[jolt::provable]
fn debug_me(x: u32) -> u32 {
    jolt_println!("x is: {}", x);  // Shows during emulation
    x * 2
}
```

**Code**: `jolt_println!` macro in `jolt-sdk/src/lib.rs` → outputs via tracer's print buffer

**2. Cycle tracking** (measure performance):
```rust
use jolt::{start_cycle_tracking, end_cycle_tracking};

#[jolt::provable]
fn profile_me(n: u32) -> u32 {
    start_cycle_tracking("loop");
    let result = (0..n).sum();
    end_cycle_tracking("loop");  // Prints cycle count
    result
}
```

**Code**: Tracking implemented in `tracer/src/emulator/cpu.rs:cycle_tracker`

**3. Test outside zkVM first**:
```rust
// Regular Rust test (runs natively)
#[test]
fn test_logic() {
    assert_eq!(add_numbers(2, 3), 5);
}
```

---

## 5. Trace Generation: From Execution to Data

### 5.1 What is "The Trace"?

**Mental Model**: Imagine a debugger that records *every single thing* the CPU does.

For this simple program:
```rust
fn add(a: u32, b: u32) -> u32 { a + b }
```

The trace might look like:
```
Cycle 0: LOAD r1, [0x1000]    # Load 'a' from memory
         PC=0x80000000, rs1=?, rd=r1
         RAM read: addr=0x1000, value=5

Cycle 1: LOAD r2, [0x1008]    # Load 'b' from memory
         PC=0x80000004, rs1=?, rd=r2
         RAM read: addr=0x1008, value=3

Cycle 2: ADD r3, r1, r2        # Actually add them
         PC=0x80000008, rs1=r1, rs2=r2, rd=r3
         Register read: r1=5, r2=3
         Register write: r3=8

Cycle 3: STORE [0x2000], r3   # Store result
         PC=0x8000000C, rs1=r3
         RAM write: addr=0x2000, value=8
```

**Key Insight**: Every cycle captures:
- Which instruction executed
- What values were read/written
- Where PC moved to

**Data structure**: `tracer/src/instruction/mod.rs`
```rust
pub struct RISCVCycle<I: RISCVInstruction> {
    pub instruction: I,              // The instruction that executed
    pub register_state: RegisterState, // All 32 registers after execution
    pub ram_access: I::RAMAccess,     // Memory reads/writes
    pub pc: u64,                      // Program counter
}

pub struct RegisterState {
    pub x: [u64; 32],  // x0-x31 (RISC-V general purpose registers)
    pub pc: u64,       // Program counter
}
```

### 5.2 How Tracer Works

**Location**: `tracer/src/emulator/cpu.rs`

**The core loop** (simplified):
```rust
pub fn execute_program(&mut self) -> Vec<Cycle> {
    let mut trace = Vec::new();

    loop {
        // 1. Fetch: Read instruction bytes at PC
        let instr_bytes = self.read_memory(self.pc, 4);

        // 2. Decode: Match bytes to instruction type
        let instruction = Instruction::decode(instr_bytes)?;

        // 3. Execute: Update CPU state
        let cycle = instruction.trace(self, &mut self.memory);
        trace.push(cycle);

        // 4. Check termination
        if self.should_terminate() {
            break;
        }
    }

    trace
}
```

**Actual implementation**: `tracer/src/emulator/cpu.rs:execute_program()`

**Dual representation**:
- **Tracer knows**: How to *execute* (e.g., `cpu.x[rd] = x + y`)
  - File: `tracer/src/instruction/add.rs:exec()`
- **Prover knows**: How to *prove* (e.g., "lookup in ADD table")
  - File: `jolt-core/src/zkvm/instruction/add.rs:lookup_table()`

### 5.3 Instruction Implementation Pattern

Every RISC-V instruction has **two implementations**:

**In tracer** (`tracer/src/instruction/add.rs`):
```rust
// Defines HOW to execute
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

declare_riscv_instr!(
    name   = ADD,
    mask   = 0xfe00707f,
    match  = 0x00000033,
    format = FormatR,
    ram    = ()
);

impl ADD {
    fn exec(&self, cpu: &mut Cpu, _: &mut RAMAccess) {
        cpu.x[self.operands.rd] =
            cpu.x[self.operands.rs1]
                .wrapping_add(cpu.x[self.operands.rs2]);
    }
}

impl RISCVTrace for ADD {}  // Use default trace() impl
```

**Macro location**: `tracer/src/instruction/macros.rs:declare_riscv_instr!`

**In prover** (`jolt-core/src/zkvm/instruction/add.rs`):
```rust
// Defines HOW to prove
use tracer::instruction::add::ADD;
use crate::zkvm::lookup_table::RangeCheckTable;

impl<const XLEN: usize> InstructionLookup<XLEN> for ADD {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(RangeCheckTable.into())  // Proves no overflow
    }
}

impl InstructionFlags for ADD {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value] = true;
        flags[CircuitFlags::RightOperandIsRs2Value] = true;
        flags[CircuitFlags::AddOperands] = true;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ADD> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = self.to_instruction_inputs();
        x.overflowing_add(y as u64).0  // Expected result
    }
}
```

**Traits defined in**: `jolt-core/src/zkvm/instruction/mod.rs`

**Why the split?**
- Tracer = "fast execution" (just runs the code)
- Prover = "correct proof" (needs mathematical structure for lookups/sumchecks)

### 5.4 Virtual Instructions

**Problem**: Some operations don't fit Jolt's lookup structure well (e.g., division)

**Solution**: Expand complex instructions into sequences of simpler ones

**Example**: Division becomes:
```
1. VIRTUAL_ADVICE: Store quotient/remainder in virtual registers
2. VIRTUAL_ASSERT_VALID_DIV: Check q*divisor + r == dividend
3. VIRTUAL_ASSERT_LTE: Check r < divisor
4. MOVE: Copy quotient to destination register
```

**Code example**: `tracer/src/instruction/virtual_assert_valid_div.rs`

```rust
pub struct VirtualDivisionSequence {
    pub dividend: u64,
    pub divisor: u64,
    pub quotient: u64,  // Stored in virtual register
    pub remainder: u64, // Stored in virtual register
}

impl VirtualInstructionSequence for VirtualDivisionSequence {
    fn execute(&mut self, cpu: &mut Cpu, _: &mut RAMAccess) {
        // Single operation for testing
        self.quotient = self.dividend / self.divisor;
        self.remainder = self.dividend % self.divisor;
    }

    fn trace(&mut self, cpu: &mut Cpu, ram: &mut RAMAccess) -> Vec<Cycle> {
        vec![
            // Step 1: Advice (store q, r)
            VIRTUAL_ADVICE::trace(self.quotient, self.remainder),

            // Step 2: Assert q*divisor + r == dividend
            VIRTUAL_ASSERT_EQ::trace(
                q * divisor + r,
                dividend
            ),

            // Step 3: Assert r < divisor
            VIRTUAL_ASSERT_LTE::trace(r, divisor),

            // Step 4: Move quotient to destination
            VIRTUAL_MOVE::trace(quotient_vreg, rd),
        ]
    }
}
```

**Trait location**: `tracer/src/instruction/mod.rs:VirtualInstructionSequence`

**Virtual registers**: Registers 32-63 (beyond standard 32 RISC-V registers)
- Only exist within virtual sequences
- Never visible to guest program
- Defined in: `tracer/src/emulator/cpu.rs` (extends register array to 64)

**As a contributor**: Look at `tracer/src/instruction/virtual_*.rs` for examples

---

## 6. Witness Construction: Data to Polynomials

### 6.1 Why Polynomials?

**Theory recap** (from Jolt.md): Sumcheck works over multilinear polynomials

**The transformation**:
```
Execution trace (concrete values)
    ↓ jolt-core/src/zkvm/witness.rs
Multilinear Extensions (polynomial representation)
    ↓ jolt-core/src/poly/commitment/dory.rs
Committed polynomials (cryptographic commitment)
```

### 6.2 What Gets Committed?

**Location**: `jolt-core/src/zkvm/witness.rs:47-80`

```rust
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum CommittedPolynomial {
    // R1CS aux variables
    LeftInstructionInput,
    RightInstructionInput,
    WriteLookupOutputToRD,
    WritePCtoRD,
    ShouldBranch,
    ShouldJump,

    // Twist/Shout witnesses
    RdInc,                      // Registers increment
    RamInc,                     // RAM increment
    InstructionRa(usize),       // d=16 for instructions
    BytecodeRa(usize),          // d varies (computed)
    RamRa(usize),               // d varies (computed)
}
```

**~30 polynomials total**, grouped by purpose:

| Category | Examples | What They Represent | Code Location |
|----------|----------|---------------------|---------------|
| R1CS inputs | `LeftInstructionInput`, `RightInstructionInput` | Operands for arithmetic | `witness.rs:generate_r1cs_witness()` |
| R1CS flags | `ShouldJump`, `ShouldBranch` | Control flow decisions | `witness.rs:generate_r1cs_witness()` |
| Registers | `RdInc` | Value changes in registers | `registers/mod.rs:compute_increment_witness()` |
| RAM | `RamInc` | Value changes in memory | `ram/mod.rs:compute_increment_witness()` |
| Lookups | `InstructionRa[0..15]` | Which instruction executed | `instruction_lookups/mod.rs:compute_ra_evals()` |

**Key Insight**: Each polynomial encodes one "column" of the trace

**Example**: `LeftInstructionInput` polynomial
- Input: `[5, 3, 8, 2, ...]` (left operand for each cycle)
- Code: `jolt-core/src/zkvm/witness.rs:generate_left_input()`
```rust
fn generate_left_input(trace: &[Cycle]) -> Vec<F> {
    trace.iter().map(|cycle| {
        F::from(cycle.to_instruction_inputs().0)  // Extract left operand
    }).collect()
}
```
- Output: `DenseMultilinearExtension` that interpolates these values over Boolean hypercube

**MLE struct**: `jolt-core/src/poly/dense_mlpoly.rs`
```rust
pub struct DenseMultilinearExtension<F: JoltField> {
    pub evaluations: Vec<F>,  // Evaluations at all points in {0,1}^n
    pub num_vars: usize,      // n (number of variables)
}
```

### 6.3 One-Hot Encoding Pattern

**Problem**: How do we encode "instruction at cycle 7 was ADD"?

**Solution**: One-hot polynomials (sparse representation)

**Code**: `jolt-core/src/zkvm/witness.rs:compute_one_hot_polynomial()`

```rust
// Stored efficiently as Vec<Option<u8>>
fn compute_one_hot_polynomial(
    trace: &[Cycle],
    extract_index: impl Fn(&Cycle) -> Option<usize>
) -> Vec<Option<u8>> {
    trace.iter().map(|cycle| {
        extract_index(cycle).map(|idx| idx as u8)
    }).collect()
}

// Example usage for instruction lookups:
let instruction_ra = compute_one_hot_polynomial(trace, |cycle| {
    Some(cycle.instruction.lookup_index())
});

// Result looks like:
instruction_ra[0] = [None, Some(3), None, Some(1), ...]
//                   ^^^^  ^^^^^^^  ^^^^  ^^^^^^^
//                    |       |      |       |
//                    No access here | Position 1 is hot
//                          Position 3 is hot
```

**Why this matters**:
- Most cycles: no access to most memory/registers
- Sparse storage saves memory during proof generation
- Converted to dense MLE only when needed for sumcheck

### 6.4 Committed vs Virtual Polynomials

**Two types of polynomials**:

**1. Committed** (most of them)
- Explicitly committed using Dory PCS
- Opening proven in stage 5
- Examples: `LeftInstructionInput`, `RamInc`

**Code**: `jolt-core/src/poly/commitment/dory.rs:batch_commit()`
```rust
pub fn batch_commit(
    polys: &[DenseMultilinearExtension<F>],
    setup: &ProverSetup
) -> Vec<(Commitment, OpeningProofHint)> {
    polys.par_iter()  // Parallel via Rayon!
        .map(|poly| commit_single(poly, setup))
        .collect()
}
```

**2. Virtual** (a few special ones)
- Never committed directly
- Evaluation proven by subsequent sumcheck
- Examples: `RegistersVal`, `RamRaf`

**Enum**: `jolt-core/src/zkvm/witness.rs:89-102`
```rust
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum VirtualPolynomial {
    UniformConstraintEq,  // R1CS
    RegistersVal,         // Registers
    RamVal,               // RAM
    RamRaf,               // RAM
    InstructionsRaf,      // Instructions
    BytecodeRd,           // Bytecode
}
```

**Mental Model**:
- Committed = "I promise this polynomial has these values" → Proven via PCS in Stage 5
- Virtual = "Let me prove this polynomial evaluates to X at point r" → Proven via sumcheck in next stage

**Why have virtual polynomials?**
- Optimization: Avoids committing to intermediate values
- They're only needed as inputs to other sumchecks
- Creates the DAG structure (one sumcheck's output = next sumcheck's input)

---

## 7. The Proof DAG: How Components Interact

### 7.1 What is "The DAG"?

**DAG = Directed Acyclic Graph**

**Mental Model**: Think of the proof as a flowchart where:
- **Nodes** = Sumcheck instances ("prove this claim")
- **Edges** = Polynomial evaluations ("output of one sumcheck feeds into another")
- **Stages** = Layers that can run in parallel

**Code structure**: `jolt-core/src/zkvm/dag/jolt_dag.rs`

```rust
pub struct JoltDAG;

impl JoltDAG {
    pub fn prove<F, ProofTranscript, PCS>(
        state_manager: StateManager<F, ProofTranscript, PCS>,
    ) -> Result<JoltProof<F, PCS, ProofTranscript>> {
        // Stage 1: Spartan outer
        spartan_dag.stage1_prove(&mut state_manager)?;

        // Stage 2: Batched sumchecks (4 components)
        let stage2_instances = collect_stage2_instances(...);
        BatchedSumcheck::prove(stage2_instances, ...)?;

        // Stage 3: Batched sumchecks (4 components)
        let stage3_instances = collect_stage3_instances(...);
        BatchedSumcheck::prove(stage3_instances, ...)?;

        // Stage 4: Batched sumchecks (3 components)
        let stage4_instances = collect_stage4_instances(...);
        BatchedSumcheck::prove(stage4_instances, ...)?;

        // Stage 5: Batched opening proof
        let opening_proof = prove_all_openings(...)?;

        Ok(JoltProof { /* ... */ })
    }
}
```

**File**: `jolt-core/src/zkvm/dag/jolt_dag.rs:68-380`

### 7.2 The Five Stages (Simplified)

```
Stage 1: Kick things off
┌─────────────────────────┐
│ Spartan: R1CS outer     │  "Prove constraints are satisfied"
└────────┬────────────────┘
         │ Outputs random point r₁
         │ Code: jolt-core/src/zkvm/spartan/mod.rs:stage1_prove()
         ↓

Stage 2: Component read/write checking
┌─────────────────────────────────────┐
│ 4 sumchecks running in parallel:    │
│ • Spartan middle                    │ → spartan/product.rs
│ • Registers read/write (Twist)      │ → registers/read_write_checking.rs
│ • RAM read/write (Twist)            │ → ram/read_write_checking.rs
│ • Instructions read (Shout)         │ → instruction_lookups/booleanity.rs
└────────┬────────────────────────────┘
         │ All use same random point r₂
         │ Code: jolt-core/src/zkvm/dag/jolt_dag.rs:138-186
         ↓

Stage 3: Evaluate constructions
┌─────────────────────────────────────┐
│ 4 sumchecks running in parallel:    │
│ • Spartan inner                     │ → spartan/inner.rs
│ • Registers Val evaluation          │ → registers/mod.rs:stage3_prover_instances()
│ • Instructions Raf evaluation       │ → instruction_lookups/ra_virtual.rs
│ • RAM Raf evaluation                │ → ram/raf_evaluation.rs
└────────┬────────────────────────────┘
         │ All use same random point r₃
         │ Code: jolt-core/src/zkvm/dag/jolt_dag.rs:188-240
         ↓

Stage 4: Final checks
┌─────────────────────────────────────┐
│ 3 sumchecks running in parallel:    │
│ • RAM output check                  │ → ram/output_check.rs
│ • Bytecode read checking (Shout)    │ → bytecode/read_raf_checking.rs
│ • Instructions finalization         │ → instruction_lookups/read_raf_checking.rs
└────────┬────────────────────────────┘
         │ All use same random point r₄
         │ Code: jolt-core/src/zkvm/dag/jolt_dag.rs:242-293
         ↓

Stage 5: Batched opening proof
┌─────────────────────────────────────┐
│ Single batched proof via Dory:      │
│ "All polynomial openings from       │
│  stages 1-4 are correct"            │
└─────────────────────────────────────┘
Code: jolt-core/src/zkvm/dag/jolt_dag.rs:295-347
      jolt-core/src/poly/opening_proof.rs:reduce_and_prove()
```

### 7.3 Why This Structure?

**Key Insight**: Sumchecks can run in parallel *if they don't depend on each other*

**Batching benefits**:
- **Proof size**: One random challenge instead of N challenges
- **Verifier time**: Check N sumchecks with same challenge
- **Prover parallelism**: Modern CPUs have multiple cores

**Code**: `jolt-core/src/subprotocols/sumcheck.rs:178-250`

```rust
pub struct BatchedSumcheck;

impl BatchedSumcheck {
    pub fn prove<F, ProofTranscript>(
        instances: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>>,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof, Vec<F>) {
        // 1. Sample batching coefficients α₁, α₂, ... from transcript
        let coeffs = transcript.challenge_scalars(instances.len());

        // 2. For each round:
        for round in 0..num_rounds {
            // Combine all instances: α₁·P₁(x) + α₂·P₂(x) + ...
            let combined_poly = combine_with_coefficients(&instances, &coeffs, round);

            // Send single univariate polynomial
            transcript.append_univariate(&combined_poly);

            // Receive single challenge
            let challenge = transcript.challenge_scalar();

            // All instances bind same challenge
            for instance in &mut instances {
                instance.bind(challenge, round);
            }
        }

        // 3. Each instance caches its output claim
        for instance in instances {
            let claim = instance.final_evaluation();
            accumulator.append_virtual(instance.poly_id(), claim);
        }

        (proof, challenges)
    }
}
```

**As a contributor**: Each component implements `SumcheckStages` trait

**Trait location**: `jolt-core/src/zkvm/dag/stage.rs`

```rust
pub trait SumcheckStages<F: JoltField, ProofTranscript: Transcript> {
    fn stage2_prover_instances(&mut self, state: &mut StateManager<F, ProofTranscript, PCS>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>;

    fn stage3_prover_instances(&mut self, state: &mut StateManager<F, ProofTranscript, PCS>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>;

    fn stage4_prover_instances(&mut self, state: &mut StateManager<F, ProofTranscript, PCS>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>;
}
```

**Example implementation**: `jolt-core/src/zkvm/registers/mod.rs`

```rust
impl<F: JoltField, ProofTranscript: Transcript> SumcheckStages<F, ProofTranscript>
    for RegistersDag<F>
{
    fn stage2_prover_instances(&mut self, state: &mut StateManager<...>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>
    {
        vec![
            Box::new(self.read_checking_rs1),  // Twist for rs1
            Box::new(self.read_checking_rs2),  // Twist for rs2
            Box::new(self.write_checking_rd),  // Twist for rd
        ]
    }

    fn stage3_prover_instances(&mut self, state: &mut StateManager<...>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>
    {
        vec![
            Box::new(self.hamming_weight),     // Part of Twist
            Box::new(self.val_evaluation),     // Prove Val(r)
        ]
    }
}
```

### 7.4 Following Data Through the DAG

**Example: RAM value at address 0x1000**

**Stage 2**: Prove "reads match most recent writes"
- File: `jolt-core/src/zkvm/ram/read_write_checking.rs:prove()`
- Sumcheck outputs: "RamVal polynomial evaluates to V at point r"
- This is a **virtual polynomial opening** (not proven yet)
- Code:
  ```rust
  // After sumcheck completes:
  accumulator.append_virtual_opening(
      VirtualPolynomial::RamVal,
      point_r,
      claimed_eval_V
  );
  ```

**Stage 3**: Prove that opening claim
- File: `jolt-core/src/zkvm/ram/raf_evaluation.rs:prove()`
- Sumcheck takes V as input claim
- Proves V is correct evaluation by sumchecking over RamVal construction:
  ```rust
  fn final_check(&self, claim: F, r: &[F]) -> Result<(), Error> {
      let expected = self.compute_ram_val_at(r);
      if claim != expected {
          return Err(Error::FinalCheckFailed);
      }
      Ok(())
  }
  ```
- Outputs: Several **committed polynomial openings** at r

**Stage 5**: Prove all committed openings
- File: `jolt-core/src/poly/opening_proof.rs:reduce_and_prove()`
- Batched Dory proof for all openings from stages 1-4
- Code:
  ```rust
  pub fn reduce_and_prove<F, PCS>(
      &mut self,
      polynomials: HashMap<CommittedPolynomial, DenseMultilinearExtension<F>>,
      hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
      generators: &PCS::ProverSetup,
      transcript: &mut ProofTranscript,
  ) -> PCS::Proof {
      // Collect all opening claims
      let claims: Vec<_> = self.appended_openings.iter().collect();

      // Batch them via random linear combination
      let coeffs = transcript.challenge_scalars(claims.len());
      let combined_claim = claims.iter().zip(&coeffs)
          .map(|((_, (point, eval)), coeff)| coeff * eval)
          .sum();

      // Single Dory opening proof
      PCS::prove_batch(&polynomials, &claims, &hints, generators, transcript)
  }
  ```

**Key Insight**: Virtual polynomials connect stages together, creating the DAG structure!

---

## 8. The Five Components Explained

### 8.1 Overview: What Each Component Does

| Component | Purpose | Key Challenge | Solution | Code Location |
|-----------|---------|---------------|----------|---------------|
| **R1CS** | Link components together | ~30 constraints per cycle | Uniform constraints (Spartan optimization) | `jolt-core/src/zkvm/r1cs/` |
| **Registers** | Prove register reads/writes correct | 64 registers, 3 accesses/cycle | Twist with fixed K=64, d=1 | `jolt-core/src/zkvm/registers/` |
| **RAM** | Prove memory reads/writes correct | Large, dynamic address space | Twist with dynamic K, d | `jolt-core/src/zkvm/ram/` |
| **Instructions** | Prove instruction outputs correct | 2^128 lookup table | Prefix-suffix sumcheck | `jolt-core/src/zkvm/instruction_lookups/` |
| **Bytecode** | Prove correct instruction fetch | Decode all instructions | Offline memory checking (Shout) | `jolt-core/src/zkvm/bytecode/` |

### 8.2 Component 1: R1CS (The Glue)

**Location**: `jolt-core/src/zkvm/r1cs/`

**Mental Model**: R1CS is like the "wiring diagram" that connects components

**What R1CS constraints do**:
1. **Link components**: "Value from RAM equals value written to register"
2. **PC updates**: "Next PC is current PC + 4, unless jumping"
3. **Arithmetic**: "left_input + right_input = lookup_output (for ADD)"

**Example constraint** (`jolt-core/src/zkvm/r1cs/constraints.rs:150-160`):
```rust
NamedConstraint {
    name: ConstraintName::RamReadEqRdWriteIfLoad,
    cons: constraint_eq_conditional_lc(
        lc!(JoltR1CSInputs::OpFlags_LoadFlag),  // When load instruction
        lc!(JoltR1CSInputs::RamRead),           // RAM read value
        lc!(JoltR1CSInputs::RdWrite),           // Must equal rd write value
    ),
    cz: CzKind::Zero,
}

// Mathematical form:
// load_flag * (ram_read - rd_write) = 0
```

**All 27 constraints defined in**: `jolt-core/src/zkvm/r1cs/constraints.rs:104-260`

```rust
pub const NUM_R1CS_CONSTRAINTS: usize = 27;

pub static UNIFORM_R1CS: [NamedConstraint; NUM_R1CS_CONSTRAINTS] = [
    NamedConstraint {
        name: ConstraintName::LeftInputEqRs1,
        cons: constraint_eq_conditional_lc(
            lc!(JoltR1CSInputs::OpFlags_LeftOperandIsRs1Value),
            lc!(JoltR1CSInputs::LeftInput),
            lc!(JoltR1CSInputs::Rs1),
        ),
        cz: CzKind::Zero,
    },
    // ... 26 more constraints
];
```

**Why ~30 constraints?**
- Cover all possible instruction behaviors
- Not all active every cycle (controlled by flags)
- Same constraints every cycle (uniformity → faster proving)

**Spartan implementation**: `jolt-core/src/zkvm/spartan/`

Three sumchecks across stages 1-3:
- **Stage 1** (`spartan/mod.rs:stage1_prove()`): Outer sumcheck over constraints
- **Stage 2** (`spartan/product.rs`): Product sumcheck for Az, Bz, Cz
- **Stage 3** (`spartan/inner.rs`): Inner sumcheck for MLE evaluations

**As a contributor**:
- See `constraints.rs` for full list with comments
- Each constraint has descriptive name
- To add constraint: Add to enum + UNIFORM_R1CS array + increment NUM_R1CS_CONSTRAINTS

### 8.3 Component 2: Registers (Fixed Size, Fast)

**Location**: `jolt-core/src/zkvm/registers/mod.rs`

**What it proves**: "Every register read returns the last value written"

**Key parameters**:
```rust
pub const REGISTER_K: usize = 64;  // 32 RISC-V + 32 virtual
pub const REGISTER_D: usize = 1;   // No chunking needed
```

**Why d=1?** K=64 is small enough for single polynomial (no need to chunk)

**Three polynomials** (`registers/mod.rs:30-50`):
```rust
pub struct RegistersPolynomials<F: JoltField> {
    // Read address polynomials (one-hot)
    pub ra_rs1: DenseMultilinearExtension<F>,  // 1 if instr reads rs1
    pub ra_rs2: DenseMultilinearExtension<F>,  // 1 if instr reads rs2

    // Write address polynomial (one-hot)
    pub wa_rd: DenseMultilinearExtension<F>,   // 1 if instr writes rd

    // Value polynomials
    pub val: DenseMultilinearExtension<F>,     // Register values
    pub inc: DenseMultilinearExtension<F>,     // Value changes (virtual)
}
```

**How Twist works** (simplified):

1. **Two memory traces**:
   - Time-ordered: registers as accessed during execution
   - Address-ordered: same accesses sorted by register number

2. **Prove they're the same via grand product**:
   ```rust
   // File: registers/read_write_checking.rs
   fn prove_consistency() {
       // Fingerprint = hash(address, timestamp, value)
       let time_ordered_product = compute_product(time_ordered_trace);
       let addr_ordered_product = compute_product(addr_ordered_trace);

       // Prove: time_ordered_product == addr_ordered_product
       // via sumcheck over their ratio
   }
   ```

3. **Three sumchecks** (batched in Stage 2):
   ```rust
   // File: registers/mod.rs:stage2_prover_instances()
   vec![
       Box::new(ReadCheckingInstance::new(self.ra_rs1, ...)),  // rs1
       Box::new(ReadCheckingInstance::new(self.ra_rs2, ...)),  // rs2
       Box::new(WriteCheckingInstance::new(self.wa_rd, ...)),  // rd
   ]
   ```

**Clever optimization**: No one-hot checks needed!
- Register addresses are hardcoded in bytecode
- Bytecode component already proves them correct
- Code comment in `registers/mod.rs:15-20`:
  ```rust
  // No explicit one-hot checks needed:
  // Register addresses come from bytecode (rs1, rs2, rd fields)
  // Bytecode component proves these are correctly decoded
  // Therefore, ra_rs1, ra_rs2, wa_rd are implicitly one-hot
  ```

### 8.4 Component 3: RAM (Dynamic Size, Complex)

**Location**: `jolt-core/src/zkvm/ram/mod.rs`

**What it proves**: "Every memory read returns the last value written"

**Key challenge**: RAM size is not fixed
- Different programs access different amounts of memory
- Can't commit to 2^64 memory cells!

**Solution: Dynamic parameters** (`ram/mod.rs:102-121`):
```rust
pub fn new_prover<F, ProofTranscript, PCS>(
    state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
) -> Self {
    // K = actual memory used (computed from trace)
    let K = state_manager.ram_K;

    // K computed earlier in StateManager::new_prover():
    // K = max(
    //   max_address_accessed,
    //   bytecode_end_address
    // ).next_power_of_two()

    // d = chunking factor: K^(1/d) = 2^8 (256 entries per chunk)
    let d = compute_d_parameter(K);

    RAMProver { K, d, /* ... */ }
}
```

**Example**:
```rust
// Trace accesses addresses: 0x80000000 - 0x80002000 (8KB)
// K = 8192 bytes / 8 bytes per word = 1024 words = 2^10
// d = compute_d_parameter(1024) = 1 (since 1024^(1/1) = 1024 < 256 threshold)
```

**Address remapping** (`ram/mod.rs:94-104`):
```rust
pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> Option<u64> {
    if address == 0 {
        return None;  // No-op cycle
    }

    if address >= memory_layout.trusted_advice_start {
        // Treat memory as doubleword-addressable (8-byte cells)
        Some((address - memory_layout.trusted_advice_start) / 8 + 1)
    } else {
        panic!("Unexpected address {address}")
    }
}
```

**Example**:
```
Guest access: LOAD from 0x80000010
Remapped: (0x80000010 - 0x80000000) / 8 + 1 = 3
Prover tracks: Memory cell 3
```

**Why division by 8?**
- RAM operations are 64-bit (doubleword)
- Reduces address space: 2^64 bytes → 2^61 doublewords
- Smaller polynomials = faster proving

**RAM Stages**:

**Stage 2** (`ram/read_write_checking.rs`): Read/write checking
```rust
// Proves: read values match most recent write (or init)
pub struct ReadWriteCheckingInstance {
    ra: DenseMultilinearExtension<F>,     // Read address polynomial
    val: DenseMultilinearExtension<F>,    // Value polynomial
    inc: DenseMultilinearExtension<F>,    // Increment polynomial
}

impl SumcheckInstance for ReadWriteCheckingInstance {
    fn combine(&self, point: &[F]) -> F {
        // Claim: ∑ ra(x) · (val(x) - inc(x)) = 0
        // Proves reads match writes
    }
}
```

**Stage 3** (`ram/raf_evaluation.rs`): Raf evaluation
```rust
// Evaluates address fingerprint polynomial at random point
```

**Stage 4** (`ram/output_check.rs`): Output check + Val final
```rust
// Verifies final memory state
// Checks program outputs in designated memory region
pub fn check_outputs(
    final_memory: &Memory,
    claimed_outputs: &[u64],
    memory_layout: &MemoryLayout,
) -> Result<(), Error> {
    for (i, expected) in claimed_outputs.iter().enumerate() {
        let addr = memory_layout.output_start + (i as u64 * 8);
        let actual = final_memory.read(addr);
        if actual != *expected {
            return Err(Error::OutputMismatch);
        }
    }
    Ok(())
}
```

### 8.5 Component 4: Instructions (Lookup-Centric)

**Location**: `jolt-core/src/zkvm/instruction_lookups/`

**What it proves**: "Instruction output matches lookup table"

**The lookup table** (conceptual):
```rust
// For 64-bit operands: 2^128 table size!
Table: (u64, u64) -> u64

Table[5, 3] = match instruction {
    ADD => 8,
    MUL => 15,
    LTU => 0,  // 5 < 3? No
    // ...
}
```

**Problem**: Table size is 2^128 (two 64-bit operands!)

**Solution: Prefix-suffix sumcheck** (`instruction_lookups/read_raf_checking.rs`)

**Key idea**: Split lookup index into prefix and suffix

```rust
// Lookup index: i = interleave(x, y)  (128 bits total)
//
// Split: i = (prefix, suffix)
//   prefix:  i[0:64]   (upper 64 bits)
//   suffix:  i[64:128] (lower 64 bits)
//
// Key property:
// Val(k_prefix, k_suffix) = Σ prefix(k_prefix) · suffix(k_suffix)
```

**Efficiently evaluable MLE**: Each instruction implements trait

**Trait**: `jolt-core/src/zkvm/lookup_table/mod.rs`

```rust
pub trait JoltLookupTable {
    fn evaluate_mle(&self, prefix: u64, suffix: u64) -> F;
}
```

**Example** (`jolt-core/src/zkvm/lookup_table/range_check.rs:40-50`):

```rust
pub struct RangeCheckTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for RangeCheckTable<XLEN> {
    fn evaluate_mle(&self, prefix: u64, suffix: u64) -> F {
        // Deinterleave bits to get original value
        let value = deinterleave_bits(prefix, suffix);

        // Check if value fits in XLEN bits
        if value < (1u128 << XLEN) {
            F::one()  // In range!
        } else {
            F::zero()  // Overflow detected
        }
    }
}
```

**Eight Phases, 16 Rounds Each**: Total 128 sumcheck rounds (log 2^128)

**Structure** (`instruction_lookups/read_raf_checking.rs:100-300`):
```rust
pub struct PrefixSuffixSumcheck {
    phase: usize,           // Current phase (0-7)
    round_in_phase: usize,  // Round within phase (0-15)
    // ...
}

impl SumcheckInstance for PrefixSuffixSumcheck {
    fn compute_prover_message(&mut self, point: &[F]) -> UnivariatePolynomial {
        // Special algorithm for prefix-suffix structure
        // Exploits efficient MLE evaluation
        // Phase i binds variables [16*i .. 16*(i+1)]
        // ...
    }
}
```

**Per-Instruction Tables** (`jolt-core/src/zkvm/lookup_table/`):

Each instruction type defines its lookup table:
```
range_check.rs  → Overflow checking
ltu.rs          → Unsigned less-than
lt.rs           → Signed less-than
xor.rs          → Bitwise XOR
and.rs          → Bitwise AND
or.rs           → Bitwise OR
// ... ~20 more tables
```

**Example: ADD instruction** (`jolt-core/src/zkvm/instruction/add.rs:30-50`):

```rust
impl<const XLEN: usize> InstructionLookup<XLEN> for ADD {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        // ADD just needs range check (detect overflow)
        Some(LookupTables::RangeCheck(RangeCheckTable))
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ADD> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = self.to_instruction_inputs();
        // Combined operand for lookup
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = self.to_instruction_inputs();
        // Expected result (with wrapping)
        x.overflowing_add(y as u64).0
    }
}
```

### 8.6 Component 5: Bytecode (Instruction Fetch)

**Location**: `jolt-core/src/zkvm/bytecode/mod.rs`

**What it proves**: "Trace instructions match program bytecode"

**How it works**:

1. **Preprocessing** (`bytecode/mod.rs:preprocess()`): Decode all instructions once
   ```rust
   pub fn preprocess(mut bytecode: Vec<Instruction>) -> Self {
       // Prepend no-op (PC starts at 0)
       bytecode.insert(0, Instruction::NoOp);

       // Pad to power of 2
       let code_size = bytecode.len().next_power_of_two();
       bytecode.resize(code_size, Instruction::NoOp);

       BytecodePreprocessing {
           code_size,
           bytecode,
           d: compute_d_parameter(code_size),
           // ...
       }
   }
   ```

2. **Execution**: Trace records which PC was accessed each cycle
   ```rust
   // In tracer:
   for cycle in execution {
       let pc = cycle.pc;  // Which bytecode instruction
       // ...
   }
   ```

3. **Proving** (Stage 4): Use Shout to prove trace reads match decoded bytecode
   ```rust
   // File: bytecode/read_raf_checking.rs
   pub struct BytecodeReadCheckingInstance {
       ra: DenseMultilinearExtension<F>,  // Read addresses (PCs from trace)
       rd: DenseMultilinearExtension<F>,  // Read values (instruction fields)
       // ...
   }

   // Sumcheck claim:
   // ∑_k ra(k) · (rd(k) - bytecode_field(k)) = 0
   // Proves: for each PC accessed, read value matches bytecode
   ```

**Read values include** (`bytecode/mod.rs:150-180`):
```rust
pub struct BytecodeRead {
    pub opcode: u8,
    pub rs1: u8,      // Source register 1
    pub rs2: u8,      // Source register 2
    pub rd: u8,       // Destination register
    pub imm: i32,     // Immediate value
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],  // Derived from opcode
}

// Each field gets its own read-checking sumcheck!
```

**Why "offline" memory checking?**
- Bytecode is read-only (no writes)
- Simpler than full Twist (don't need write-checking sumcheck)
- Uses Shout variant optimized for this case

**Clever trick: Virtualization** (from `bytecode/mod.rs:10-25`):
```rust
// Circuit flags and register addresses are virtualized:
//
// Problem: R1CS constraints use circuit flags (is_jump, is_load, etc.)
//          How do we prove these flags are correct?
//
// Solution: Flags are derived from opcode during bytecode preprocessing
//           Bytecode sumcheck proves opcodes are correct
//           Therefore, flags must be correct (no separate check needed!)
//
// This is "virtualization": bytecode correctness implies flag correctness
```

**Code**: Flags derived in `tracer/src/instruction/mod.rs:circuit_flags()`

---

## 9. Key Abstractions and Patterns

### 9.1 The `SumcheckStages` Trait

**Location**: `jolt-core/src/zkvm/dag/stage.rs:10-50`

**Purpose**: Each component declares which sumchecks belong to which stage

```rust
pub trait SumcheckStages<F: JoltField, ProofTranscript: Transcript> {
    // Stage 1 (Spartan only)
    fn stage1_prove(&mut self, state: &mut StateManager<F, ProofTranscript, PCS>)
        -> Result<(), Error> {
        Ok(())  // Default: no stage 1 sumchecks
    }

    // Stage 2
    fn stage2_prover_instances(&mut self, state: &mut StateManager<F, ProofTranscript, PCS>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>;

    // Stage 3
    fn stage3_prover_instances(&mut self, state: &mut StateManager<F, ProofTranscript, PCS>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>;

    // Stage 4
    fn stage4_prover_instances(&mut self, state: &mut StateManager<F, ProofTranscript, PCS>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>;

    // Mirror methods for verifier: stage2_verifier_instances, etc.
}
```

**Why this matters**: DAG orchestrator collects all stage 2 sumchecks from all components and batches them

**Example implementation**: `jolt-core/src/zkvm/ram/mod.rs:200-250`

```rust
impl<F: JoltField, ProofTranscript: Transcript> SumcheckStages<F, ProofTranscript>
    for RamDag<F>
{
    fn stage2_prover_instances(&mut self, state: &mut StateManager<...>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>
    {
        vec![
            Box::new(self.read_write_checking),  // Twist sumcheck
        ]
    }

    fn stage3_prover_instances(&mut self, state: &mut StateManager<...>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>
    {
        vec![
            Box::new(self.hamming_weight),       // Part of Twist
            Box::new(self.raf_evaluation),       // Evaluate fingerprint poly
        ]
    }

    fn stage4_prover_instances(&mut self, state: &mut StateManager<...>)
        -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>>
    {
        vec![
            Box::new(self.output_check),         // Verify outputs
            Box::new(self.val_final),            // Final value check
        ]
    }
}
```

### 9.2 The `RISCVInstruction` Trait (Tracer Side)

**Location**: `tracer/src/instruction/mod.rs:50-100`

**Purpose**: Define how to execute an instruction

```rust
pub trait RISCVInstruction: Sized {
    // Decoding
    const MASK: u32;   // Bit pattern to match
    const MATCH: u32;  // Expected value after masking

    // Associated types
    type Format: InstructionFormat;  // R-type, I-type, etc.
    type RAMAccess: RAMAccessTrait;  // What memory it touches

    // Execution
    fn execute(&mut self, cpu: &mut Cpu, ram: &mut Self::RAMAccess);

    // Parsing
    fn from_u32(bits: u32) -> Option<Self>;
}
```

**Helper macro**: `tracer/src/instruction/macros.rs`

```rust
macro_rules! declare_riscv_instr {
    (
        name = $name:ident,
        mask = $mask:expr,
        match = $match_val:expr,
        format = $format:ty,
        ram = $ram:ty
    ) => {
        pub struct $name {
            pub operands: <$format as InstructionFormat>::Operands,
            pub address: u64,
            // ...
        }

        impl RISCVInstruction for $name {
            const MASK: u32 = $mask;
            const MATCH: u32 = $match_val;
            type Format = $format;
            type RAMAccess = $ram;

            fn from_u32(bits: u32) -> Option<Self> {
                if (bits & Self::MASK) == Self::MATCH {
                    Some($name {
                        operands: <$format>::parse(bits),
                        // ...
                    })
                } else {
                    None
                }
            }

            // User implements execute() separately
        }
    };
}
```

**Example usage**: `tracer/src/instruction/add.rs`

```rust
declare_riscv_instr!(
    name = ADD,
    mask = 0xfe00707f,
    match = 0x00000033,
    format = FormatR,
    ram = ()
);

impl ADD {
    fn exec(&self, cpu: &mut Cpu, _: &mut ()) {
        cpu.x[self.operands.rd] =
            cpu.x[self.operands.rs1]
                .wrapping_add(cpu.x[self.operands.rs2]);
    }
}

impl RISCVTrace for ADD {}  // Use default trace() impl
```

### 9.3 The `InstructionLookup` Trait (Prover Side)

**Location**: `jolt-core/src/zkvm/instruction/mod.rs:30-80`

**Purpose**: Define how to prove an instruction

```rust
pub trait InstructionLookup<const XLEN: usize> {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>>;
}

pub trait InstructionFlags {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS];
}

pub trait LookupQuery<const XLEN: usize> {
    fn to_instruction_inputs(&self) -> (u64, i128);
    fn to_lookup_operands(&self) -> (u64, u128);
    fn to_lookup_index(&self) -> u128;
    fn to_lookup_output(&self) -> u64;
}
```

**Example**: `jolt-core/src/zkvm/instruction/mul.rs`

```rust
use tracer::instruction::mul::MUL;

impl<const XLEN: usize> InstructionLookup<XLEN> for MUL {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        // MUL uses product table
        Some(LookupTables::Product(ProductTable))
    }
}

impl InstructionFlags for MUL {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value] = true;
        flags[CircuitFlags::RightOperandIsRs2Value] = true;
        flags[CircuitFlags::MultiplyOperands] = true;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<MUL> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = self.to_instruction_inputs();
        x.wrapping_mul(y as u64)  // Expected result
    }
}
```

### 9.4 Common Patterns You'll See

**Pattern 1: Parallel iteration with Rayon** (`jolt-core/src/zkvm/witness.rs:200-220`):
```rust
use rayon::prelude::*;

// Parallel polynomial generation
let polynomials: Vec<_> = CommittedPolynomial::iter()
    .par_bridge()  // Parallel iterator
    .map(|poly_type| {
        generate_polynomial(poly_type, trace)
    })
    .collect();

// Parallel commitment
let commitments: Vec<_> = polynomials
    .par_iter()  // Parallel!
    .map(|poly| PCS::commit(poly, setup))
    .collect();
```

**Pattern 2: Shared state with `Rc<RefCell<T>>`** (`jolt-core/src/zkvm/dag/state_manager.rs:50-70`):
```rust
use std::cell::RefCell;
use std::rc::Rc;

pub struct StateManager<F, ProofTranscript, PCS> {
    // Shared transcript accessed by all components
    pub transcript: Rc<RefCell<ProofTranscript>>,

    // Shared accumulator for polynomial openings
    pub accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,

    // ...
}

// Usage:
let transcript = Rc::new(RefCell::new(Transcript::new()));
let transcript_clone = transcript.clone();

// Component A appends to transcript
transcript.borrow_mut().append(&value_a);

// Component B uses same transcript
let challenge = transcript_clone.borrow_mut().challenge();
```

**Why**: Multiple components need to append to same transcript

**Pattern 3: Dynamic dispatch for sumchecks** (`jolt-core/src/zkvm/dag/jolt_dag.rs:140-160`):
```rust
// Collect instances from all components
let mut instances: Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> = vec![];
instances.extend(spartan_dag.stage2_prover_instances(&mut state));
instances.extend(registers_dag.stage2_prover_instances(&mut state));
instances.extend(ram_dag.stage2_prover_instances(&mut state));

// Convert to mutable references
let instance_refs: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>> =
    instances.iter_mut()
        .map(|i| &mut **i as &mut dyn SumcheckInstance<_, _>)
        .collect();

// Batch prove them all
BatchedSumcheck::prove(instance_refs, accumulator, transcript)?;
```

**Why**: Different sumcheck types, same proving algorithm

---

## 10. State Management: The Glue

### 10.1 The `StateManager`

**Location**: `jolt-core/src/zkvm/dag/state_manager.rs:30-100`

**Mental Model**: StateManager is like a clipboard that all components share

```rust
pub struct StateManager<'a, F: JoltField, ProofTranscript, PCS> {
    // === Shared across all components ===

    // Fiat-Shamir transcript (random challenges)
    pub transcript: Rc<RefCell<ProofTranscript>>,

    // Proof data (sumcheck proofs from all stages)
    pub proofs: Rc<RefCell<HashMap<ProofKeys, ProofData>>>,

    // Polynomial commitments
    pub commitments: Rc<RefCell<Vec<PCS::Commitment>>>,

    // Opening accumulator (tracks polynomial evaluation claims)
    // This creates the DAG structure!
    accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,

    // === Prover-specific data ===

    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: &'a [Cycle],
    pub final_memory_state: &'a Memory,

    // === Configuration ===

    pub ram_K: usize,                      // RAM size (dynamic)
    pub twist_sumcheck_switch_index: usize, // Parallelization parameter
    pub program_io: JoltDevice,            // Inputs/outputs
}
```

**Why it exists**: Components need to:
- Append to shared transcript → ensures verifier samples same challenges
- Record polynomial evaluation claims → creates DAG edges
- Access preprocessing data → bytecode, memory layout, etc.
- Store proof pieces → collected into final proof object

**Key methods** (`state_manager.rs:150-250`):

```rust
impl<F, ProofTranscript, PCS> StateManager<F, ProofTranscript, PCS> {
    // Access prover data
    pub fn get_prover_data(&self) -> (
        &JoltProverPreprocessing<F, PCS>,
        &[Cycle],
        &Memory,
        &JoltDevice
    ) {
        (self.preprocessing, self.trace, self.final_memory_state, &self.program_io)
    }

    // Access accumulator
    pub fn get_prover_accumulator(&self) -> &Rc<RefCell<ProverOpeningAccumulator<F>>> {
        &self.accumulator
    }

    // Access transcript
    pub fn get_transcript(&self) -> &Rc<RefCell<ProofTranscript>> {
        &self.transcript
    }

    // Store commitments
    pub fn set_commitments(&mut self, commitments: Vec<PCS::Commitment>) {
        *self.commitments.borrow_mut() = commitments;
    }
}
```

### 10.2 The Opening Accumulator

**Location**: `jolt-core/src/poly/opening_proof.rs:30-150`

**Purpose**: Track all polynomial evaluation claims throughout proof

```rust
pub struct ProverOpeningAccumulator<F: JoltField> {
    // Committed polynomial openings (proven in stage 5 via PCS)
    pub appended_openings: RefCell<HashMap<
        OpeningId<CommittedPolynomial>,
        (OpeningPoint<F>, F)  // (point, evaluation)
    >>,

    // Virtual polynomial openings (proven in next sumcheck stage)
    pub appended_virtual_openings: RefCell<HashMap<
        OpeningId<VirtualPolynomial>,
        (OpeningPoint<F>, F)
    >>,
}
```

**Two types of claims**:

**1. Committed polynomial openings** (`opening_proof.rs:80-100`):
```rust
impl<F: JoltField> ProverOpeningAccumulator<F> {
    pub fn append_committed_opening(
        &self,
        poly: CommittedPolynomial,
        sumcheck: SumcheckId,
        point: Vec<F>,
        eval: F,
    ) {
        let id = OpeningId::new(poly, sumcheck);
        self.appended_openings.borrow_mut().insert(
            id,
            (OpeningPoint::new(point), eval)
        );
    }
}

// Example usage (from spartan/inner.rs):
accumulator.append_committed_opening(
    CommittedPolynomial::LeftInstructionInput,
    SumcheckId::SpartanInner,
    point_r,
    claimed_eval
);
// → Will be proven in stage 5 using Dory
```

**2. Virtual polynomial openings** (`opening_proof.rs:110-130`):
```rust
impl<F: JoltField> ProverOpeningAccumulator<F> {
    pub fn append_virtual_opening(
        &self,
        poly: VirtualPolynomial,
        sumcheck: SumcheckId,
        point: Vec<F>,
        eval: F,
    ) {
        let id = OpeningId::new(poly, sumcheck);
        self.appended_virtual_openings.borrow_mut().insert(
            id,
            (OpeningPoint::new(point), eval)
        );
    }
}

// Example usage (from ram/read_write_checking.rs):
accumulator.append_virtual_opening(
    VirtualPolynomial::RamVal,
    SumcheckId::RamReadWriteChecking,
    point_r,
    claimed_eval
);
// → Will be proven in next stage via sumcheck
```

**Workflow** (creates DAG structure!):

```
Stage 2: Sumcheck 1 finishes
    ↓
    accumulator.append_virtual_opening(
        VirtualPolynomial::RamVal,
        point = r,
        eval = V
    )
    ↓
Stage 3: Sumcheck 2 starts
    ↓
    let (point, claimed_eval) = accumulator.get_virtual_opening(RamVal);
    ↓
    "Prove RamVal(r) = V"
    ↓
    Sumcheck 2 finishes, outputs committed polynomial claims
    ↓
    accumulator.append_committed_opening(
        CommittedPolynomial::RamInc,
        point = r',
        eval = V'
    )
    ↓
Stage 5: Batch prove all committed polynomial claims
```

**Key insight**: Accumulator creates edges in the DAG by passing evaluation claims between sumchecks!

---

## 11. Common Misconceptions

### 11.1 "The Guest Program is Proven Directly"

**❌ Wrong**: Jolt doesn't prove Rust semantics

**✅ Right**: Jolt proves RISC-V execution
- Your Rust code is compiled to RISC-V (`jolt-core/src/host/program.rs`)
- The RISC-V execution is what gets proven (`tracer/src/emulator/cpu.rs`)
- If compiler has bugs, Jolt doesn't catch them

**Implication**: Trust in Jolt = Trust in (rustc + RISC-V spec)

**Code evidence**: `tracer/src/emulator/cpu.rs:execute_program()` operates on RISC-V bytecode, not Rust AST

### 11.2 "Polynomial Commitments Happen in Stage 5"

**❌ Wrong**: Commitments happen during witness generation (before stage 1)

**✅ Right**: Stage 5 is about *opening proofs*

**Timeline**:
1. **Witness generation** (`jolt-core/src/zkvm/dag/jolt_dag.rs:574-611`): Commit to all polynomials
   ```rust
   let (commitments, hints) = PCS::batch_commit(&polynomials, setup);
   state_manager.set_commitments(commitments);
   ```

2. **Stages 1-4**: Generate claims about polynomial evaluations
   ```rust
   // Example from Stage 2:
   accumulator.append_committed_opening(
       CommittedPolynomial::RamInc,
       point_r,
       claimed_eval
   );
   ```

3. **Stage 5** (`jolt-core/src/zkvm/dag/jolt_dag.rs:295-347`): Batch prove all those evaluation claims
   ```rust
   let opening_proof = accumulator.reduce_and_prove(
       polynomials_map,
       hints,
       generators,
       transcript
   );
   ```

### 11.3 "Each Sumcheck Operates on Different Data"

**❌ Wrong**: All sumchecks in a batch use the same random challenge

**✅ Right**: Batching means shared randomness

**Code**: `jolt-core/src/subprotocols/sumcheck.rs:BatchedSumcheck::prove()`

```rust
// NOT this:
for instance in instances {
    let challenge = transcript.challenge();  // Different challenges
    instance.bind(challenge);
}

// But this:
let challenge = transcript.challenge();  // SAME challenge for all!
for instance in instances {
    instance.bind(challenge);  // All use same challenge
}
```

**Why batch?** Reduces interaction rounds (proof size)
- Without batching: N proofs, N transcripts, N·k challenges (k = #rounds)
- With batching: 1 proof, 1 transcript, k challenges

### 11.4 "Virtual Registers Are Like Real Registers"

**❌ Wrong**: Virtual registers are accessible from guest code

**✅ Right**: Virtual registers only exist in virtual instruction sequences

**Code evidence**: `tracer/src/emulator/cpu.rs:40-60`

```rust
pub struct Cpu {
    pub x: [u64; 64],  // x[0..31]: RISC-V standard
                       // x[32..63]: Virtual (only for proof optimization)
    pub pc: u64,
}

// Guest code can only access x[0..31]:
// (enforced by RISC-V instruction encoding - only 5 bits for register address)

// Virtual sequences use x[32..63]:
impl VirtualMove {
    fn exec(&self, cpu: &mut Cpu) {
        cpu.x[self.rd] = cpu.x[self.virtual_reg_source];  // virtual_reg >= 32
    }
}
```

- Registers 0-31: RISC-V standard (guest can use)
- Registers 32-63: Virtual (only for proof optimization)
- Guest never sees virtual registers (no RISC-V instruction encodes register > 31)

### 11.5 "Preprocessing Must Run Every Time"

**❌ Wrong**: You need to preprocess for every proof

**✅ Right**: Preprocessing is per-program, not per-execution

**Code**: `jolt-sdk/macros/src/lib.rs:435-481` (generated `preprocess_prover_*` function)

```rust
// Preprocessing inputs:
// - bytecode: Vec<Instruction>        // SAME for all inputs
// - memory_layout: MemoryLayout       // SAME for all inputs
// - max_trace_length: usize           // SAME for all inputs
//
// NOT inputs:
// - actual program inputs              // Different each time!
// - execution trace                    // Different each time!

pub fn preprocess_prover_my_function(program: &mut Program)
    -> JoltProverPreprocessing<F, PCS>
{
    let (bytecode, memory_init, _) = program.decode();

    JoltRV64IMAC::prover_preprocess(
        bytecode,          // ← Only depends on program
        memory_layout,     // ← Only depends on program
        memory_init,       // ← Only depends on program
        max_trace_length,  // ← Configuration constant
    )
}
```

**Usage pattern**:
```rust
// Preprocess once:
let preprocessing = preprocess_prover_fibonacci(&mut program);

// Prove many times with different inputs:
for input in inputs {
    let (output, proof) = prove_fibonacci(input, &preprocessing);
}
```

**Future**: Caching preprocessing to disk across runs (not yet implemented, see GitHub issues)

---

## 12. Contributing: Practical Guide

### 12.1 Starting Points for Contributors

**"I want to add a new RISC-V instruction"** → Section 12.2

**"I want to optimize the prover"** → Start with profiling (section 12.6)

**"I want to understand how X works"** → Read relevant component (section 8)

**"I want to fix a bug"** → Debugging workflow (section 12.7)

**"I want to add tests"** → Testing strategies (section 12.5)

### 12.2 Adding a New Instruction (Step-by-Step)

**Goal**: Add support for a new RISC-V instruction

**Step 1: Tracer side** (how to execute)

**File**: `tracer/src/instruction/my_instr.rs`

```rust
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MYINSTR,
    mask   = 0x...,       // From RISC-V spec
    match  = 0x...,       // From RISC-V spec
    format = FormatR,     // or FormatI, FormatS, etc.
    ram    = ()           // or RAMAccess if touches memory
);

impl MYINSTR {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MYINSTR as RISCVInstruction>::RAMAccess) {
        // Implement execution logic
        cpu.x[self.operands.rd] = /* compute result */;
    }
}

impl RISCVTrace for MYINSTR {}  // Use default trace() impl
```

**Test**: Does it execute correctly?
```rust
#[cfg(test)]
mod test {
    #[test]
    fn test_myinstr_execution() {
        let mut cpu = Cpu::new();
        cpu.x[1] = 10;
        cpu.x[2] = 20;

        let instr = MYINSTR { /* ... */ };
        instr.execute(&mut cpu, &mut ());

        assert_eq!(cpu.x[3], expected_result);
    }
}
```

**Step 2: Add to instruction enum**

**File**: `tracer/src/instruction/mod.rs`

```rust
pub enum Instruction {
    // ... existing instructions
    MYINSTR(MYINSTR),
}

pub enum Cycle {
    // ... existing cycles
    MYINSTR(RISCVCycle<MYINSTR>),
}

// Add to decode function:
impl Instruction {
    pub fn decode(bits: u32, address: u64) -> Result<Self, &'static str> {
        // ... existing matches
        if (bits & MYINSTR::MASK) == MYINSTR::MATCH {
            return Ok(MYINSTR::from_u32(bits).into());
        }
        // ...
    }
}
```

**Step 3: Prover side** (how to prove)

**File**: `jolt-core/src/zkvm/instruction/my_instr.rs`

```rust
use tracer::instruction::my_instr::MYINSTR;
use crate::zkvm::lookup_table::LookupTables;
use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery};

impl<const XLEN: usize> InstructionLookup<XLEN> for MYINSTR {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        // Return appropriate lookup table
        Some(/* ... */)
    }
}

impl InstructionFlags for MYINSTR {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        // Set appropriate flags
        flags[CircuitFlags::LeftOperandIsRs1Value] = true;
        // ...
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<MYINSTR> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        // Compute expected output
    }
}
```

**Step 4: Add to prover instruction enum**

**File**: `jolt-core/src/zkvm/instruction/mod.rs`

```rust
define_rv32im_trait_impls! {
    instructions: [
        // ... existing
        MYINSTR,
    ]
}
```

**Step 5: Test end-to-end**

**File**: `examples/test_myinstr/guest/src/lib.rs`

```rust
#[jolt::provable]
fn test_myinstr(a: u32, b: u32) -> u32 {
    // Inline assembly to use your instruction
    unsafe {
        let result: u32;
        asm!(
            ".word 0x...",  // Your instruction encoding
            out("x10") result,
            in("x11") a,
            in("x12") b,
        );
        result
    }
}
```

**File**: `examples/test_myinstr/host/src/main.rs`

```rust
fn main() {
    let (output, proof, io) = guest::prove_test_myinstr(10, 20);
    assert_eq!(output, expected);
    assert!(guest::verify_test_myinstr(10, 20, output, io.panic, proof));
}
```

**Example workflow**: See `tracer/src/instruction/add.rs` + `jolt-core/src/zkvm/instruction/add.rs` as templates

### 12.3 Adding Virtual Instruction Sequences

**When to use**: Complex operation that doesn't fit lookup structure

**Steps**:

**1. Design sequence**: Break operation into simple steps

Example for division:
```
1. VIRTUAL_ADVICE: Store quotient/remainder in virtual registers
2. VIRTUAL_ASSERT_EQ: Check q*divisor + r == dividend
3. VIRTUAL_ASSERT_LTE: Check r < divisor
4. MOVE: Copy quotient to destination register
```

**2. Implement each virtual instruction**

**File**: `tracer/src/instruction/virtual_my_sequence.rs`

```rust
pub struct MyVirtualSequence {
    // Intermediate values
    pub intermediate1: u64,
    pub intermediate2: u64,
}

impl VirtualInstructionSequence for MyVirtualSequence {
    type Instruction = MyVirtualInstr;

    fn execute(&mut self, cpu: &mut Cpu, ram: &mut RAMAccess) {
        // Single operation for testing
        let result = compute_result();
        cpu.x[self.rd] = result;
    }

    fn trace(&mut self, cpu: &mut Cpu, ram: &mut RAMAccess) -> Vec<Cycle> {
        vec![
            // Step 1: Virtual instruction 1
            VIRTUAL_INSTR1::trace(cpu, ram, self.intermediate1),

            // Step 2: Virtual instruction 2
            VIRTUAL_INSTR2::trace(cpu, ram, self.intermediate2),

            // Step 3: Move to destination
            VIRTUAL_MOVE::trace(cpu, ram, self.rd),
        ]
    }
}
```

**3. Test both paths**:

```rust
#[test]
fn test_virtual_sequence() {
    let mut cpu = Cpu::new();
    let mut sequence = MyVirtualSequence { /* ... */ };

    // Test execute() gives correct result
    sequence.execute(&mut cpu, &mut ());
    assert_eq!(cpu.x[rd], expected);

    // Test trace() produces equivalent sequence
    let trace = sequence.trace(&mut cpu, &mut ());
    assert_eq!(trace.len(), 3);  // 3 virtual instructions

    // Verify final state matches
    assert_eq!(cpu.x[rd], expected);
}
```

**Location**: See `tracer/src/instruction/virtual_assert_valid_div.rs` for complete example

### 12.4 Adding R1CS Constraints

**When**: You need to link components in a new way

**Steps**:

**1. Identify the constraint**
- What invariant are you enforcing?
- Example: "If storing to RAM, rs2 value must equal RAM write value"
- Mathematical form: `store_flag * (rs2 - ram_write) = 0`

**2. Add to `ConstraintName` enum**

**File**: `jolt-core/src/zkvm/r1cs/constraints.rs:104-131`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum_macros::EnumIter)]
pub enum ConstraintName {
    // ... existing constraints
    MyNewConstraint,
}
```

**3. Define constraint in `UNIFORM_R1CS` array**

**File**: `jolt-core/src/zkvm/r1cs/constraints.rs:140-260`

```rust
pub static UNIFORM_R1CS: [NamedConstraint; NUM_R1CS_CONSTRAINTS] = [
    // ... existing constraints
    NamedConstraint {
        name: ConstraintName::MyNewConstraint,
        cons: constraint_eq_conditional_lc(
            lc!(JoltR1CSInputs::MyFlag),        // When is this active?
            lc!(JoltR1CSInputs::LeftValue),     // What should equal...
            lc!(JoltR1CSInputs::RightValue),    // ...what?
        ),
        cz: CzKind::Zero,
    },
];
```

**4. Update constraint count**

```rust
pub const NUM_R1CS_CONSTRAINTS: usize = 28;  // Increment from 27
```

**5. Test**:

```rust
#[test]
fn test_new_constraint() {
    // Create witness with constraint satisfied
    let satisfied_witness = /* ... */;
    assert!(verify_r1cs_constraints(&satisfied_witness).is_ok());

    // Create witness with constraint violated
    let violated_witness = /* ... */;
    assert!(verify_r1cs_constraints(&violated_witness).is_err());
}
```

### 12.5 Testing Strategies

**Three levels of testing**:

**Level 1: Unit tests** (in implementation files)

**Example**: `jolt-core/src/zkvm/registers/mod.rs:test`

```rust
#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_registers_consistency() {
        // Create simple trace
        let trace = create_test_trace();

        // Generate witness
        let witness = RegistersPolynomials::generate_witness(&trace);

        // Verify consistency
        assert!(verify_twist_consistency(&witness).is_ok());
    }
}
```

**Level 2: Integration tests** (in `jolt-core/src/zkvm/mod.rs`)

**Example**: `jolt-core/src/zkvm/mod.rs:test`

```rust
#[test]
fn test_full_proof_for_simple_program() {
    let bytecode = vec![
        Instruction::ADD(/* ... */),
        Instruction::SUB(/* ... */),
    ];

    let (proof, _) = prove_program(bytecode, inputs);
    assert!(verify_program(bytecode, inputs, outputs, proof).is_ok());
}
```

**Level 3: Example programs** (in `examples/`)

**Example**: `examples/fibonacci/`

```bash
cargo run --release -p fibonacci
# If it completes without panicking, proof verified!
```

**Debugging failed tests**:

1. **Enable logging**:
   ```bash
   RUST_LOG=debug cargo test test_name
   ```

2. **Check which sumcheck failed**:
   ```
   Error: Stage 2 sumcheck failed for component RAM
   ```

3. **Inspect StateManager state**:
   ```rust
   // Add to test:
   dbg!(state_manager.get_prover_accumulator().borrow());
   ```

4. **Compare expected vs actual**:
   ```rust
   let expected_eval = compute_expected_evaluation(point);
   let actual_eval = polynomial.evaluate(point);
   assert_eq!(expected_eval, actual_eval, "Mismatch at point {:?}", point);
   ```

### 12.6 Understanding Profiling Output

**Three profiling modes**:

**1. Execution profiling** (tracing_chrome):

```bash
cargo run --release -p jolt-core profile --name sha3 --format chrome
# Opens trace-<timestamp>.json at https://ui.perfetto.dev/
```

**What to look for**:
- Which stage takes longest? → Optimize that first
- Which component within stage is slowest? → Look at that component's code
- Is polynomial generation parallelized? → Check for `.par_iter()` usage

**Code**: Tracing instrumented throughout `jolt-core/src/zkvm/`
```rust
#[tracing::instrument(name = "Stage 2 sumchecks", skip_all)]
fn stage2_prove(...) {
    // This span shows up in Perfetto!
}
```

**2. CPU profiling** (pprof):

```bash
cargo run --release --features pprof -p jolt-core profile --name sha3
go tool pprof -http=:8080 target/release/jolt-core benchmark-runs/pprof/sha3_prove.pb
```

**What to look for**:
- Which functions are CPU-intensive? → Top of flamegraph
- Is serialization/deserialization slow? → Look for `serde` functions
- Are field operations optimized? → Check `ark_ff` usage

**Code**: pprof integration in `jolt-core/src/lib.rs:setup_pprof()`

**3. Memory profiling** (allocative):

```bash
RUST_LOG=debug cargo run --release --features allocative -p jolt-core profile --name sha3
# Generates SVG flamegraphs for each stage
```

**What to look for**:
- Which polynomials use most memory? → Shows in flamegraph
- Are there unexpected allocations? → Look for `vec!` or `HashMap::new()`
- Can we reduce peak memory usage? → Consider streaming or chunking

**Code**: Allocative annotations in `jolt-core/src/zkvm/witness.rs`
```rust
#[derive(Allocative)]
pub struct CommittedPolynomial {
    // Memory usage tracked automatically
}
```

### 12.7 Debugging Workflows

**Scenario 1: Proof generation fails**

```
Error message: "Stage 2 sumcheck failed: final check mismatch"

Step 1: Identify which component
    → Add RUST_LOG=debug
    → Output: "RAM read/write checking sumcheck failed"
    → File: jolt-core/src/zkvm/ram/read_write_checking.rs

Step 2: Identify the specific sumcheck
    → Look at error context
    → "Expected evaluation: 12345, actual: 12346"
    → Mismatch at final check

Step 3: Inspect polynomial evaluations
    → Add breakpoint or print in read_write_checking.rs:final_check()
    → Check claimed_eval vs computed_eval
    → Code:
      fn final_check(&self, claim: F, r: &[F]) -> Result<(), Error> {
          let computed = self.compute_expected_eval_at(r);
          dbg!(claim, computed);  // ← Add this
          if claim != computed {
              return Err(Error::FinalCheckMismatch);
          }
          Ok(())
      }

Step 4: Check trace correctness
    → Is trace itself valid?
    → Print relevant Cycle data:
      for cycle in trace.iter().filter(|c| involves_ram(c)) {
          dbg!(cycle);
      }
    → Verify RAM accesses manually
```

**Scenario 2: Prover is slow**

```
Observation: Proving takes 60 seconds, should be ~10 seconds

Step 1: Profile to find bottleneck
    → cargo run --features pprof -p jolt-core profile --name sha3
    → Look at flamegraph
    → Example: "compute_ra_evals" taking 40% of time

Step 2: Check polynomial sizes
    → Add logging:
      tracing::info!("RAM K = {}, d = {}", K, d);
    → Are K or d parameters too large?
    → Can we optimize address remapping to reduce K?

Step 3: Check parallelization
    → Search for .par_iter() in hot code path
    → If missing, add Rayon parallelization:
      // Before:
      let results: Vec<_> = items.iter().map(|item| process(item)).collect();

      // After:
      use rayon::prelude::*;
      let results: Vec<_> = items.par_iter().map(|item| process(item)).collect();

Step 4: Optimize hot paths
    → Use `#[inline]` on small frequently-called functions
    → Cache repeated computations
    → Example:
      // Before:
      for i in 0..n {
          let result = expensive_computation(i);  // Recomputed!
          use_result(result);
      }

      // After:
      let cached = (0..n).map(|i| expensive_computation(i)).collect::<Vec<_>>();
      for i in 0..n {
          use_result(cached[i]);
      }
```

**Scenario 3: Emulation produces wrong result**

```
Observation: Guest function returns 42, expected 43

Step 1: Verify instruction execution
    → Unit test the instruction in tracer
    → File: tracer/src/instruction/add.rs:test
    → Check against RISC-V spec

Step 2: Enable guest printing
    → Add to guest:
      use jolt::jolt_println;

      #[jolt::provable]
      fn debug_me(x: u32) -> u32 {
          jolt_println!("x = {}", x);
          let result = x + 1;
          jolt_println!("result = {}", result);
          result
      }
    → Run and check output

Step 3: Compare with native execution
    → Compile guest for host target:
      cargo test --package guest --lib
    → Run tests natively first
    → If native is correct but RISC-V is wrong:
      → Check for undefined behavior (unsafe code)
      → Check for endianness issues

Step 4: Examine execution trace
    → Enable trace output:
      let trace = program.trace(inputs);
      for (i, cycle) in trace.iter().enumerate() {
          println!("Cycle {}: {:?}", i, cycle);
      }
    → Look for unexpected instructions or register values
```

### 12.8 Common Pitfalls

**Pitfall 1: Forgetting to update enums**
- Add instruction to tracer → must also add to prover
- Update in `mod.rs` for both `tracer/` and `jolt-core/src/zkvm/instruction/`
- Easy to miss one!

**Pitfall 2: Off-by-one in memory addressing**
- Remember: addresses are remapped
  ```rust
  // Guest sees: 0x80000000 - 0x8FFFFFFF
  // Prover sees: 1 - K
  let witness_idx = (guest_addr - RAM_START) / 8 + 1;  // Don't forget +1!
  ```
- Remember: division by 8 for doubleword addressing

**Pitfall 3: Mixing committed and virtual polynomials**
- Virtual polynomials: don't commit, prove via sumcheck
  ```rust
  // WRONG:
  let commitment = PCS::commit(&virtual_poly);  // ❌

  // RIGHT:
  accumulator.append_virtual_opening(
      VirtualPolynomial::RamVal,
      point, eval
  );  // ✅
  ```
- Committed polynomials: commit, prove via Dory opening
  ```rust
  // WRONG:
  accumulator.append_virtual_opening(
      CommittedPolynomial::RamInc,  // ❌ This is committed!
      point, eval
  );

  // RIGHT:
  accumulator.append_committed_opening(
      CommittedPolynomial::RamInc,
      point, eval
  );  // ✅
  ```

**Pitfall 4: Incorrect uniformity assumptions**
- R1CS constraints must be truly uniform
  ```rust
  // WRONG: Different constraint per cycle
  if cycle_idx < 100 {
      enforce_constraint_a();
  } else {
      enforce_constraint_b();  // ❌ Not uniform!
  }

  // RIGHT: Same constraints, different witness values
  let flag = if cycle_idx < 100 { 1 } else { 0 };
  enforce_constraint_a_with_flag(flag);  // ✅ Uniform!
  ```

**Pitfall 5: Not testing edge cases**
- What happens on last cycle?
  ```rust
  // Test with trace_length = 1
  // Test with trace_length = max_trace_length
  ```
- What if no RAM accesses occur?
  ```rust
  // Test with program that never touches memory
  ```
- What if all instructions are NOOPs?
  ```rust
  // Test with program that just returns
  ```

---

## Quick Reference Tables

### File Location Quick Reference

| I want to... | Look here | Key functions/structs |
|--------------|-----------|----------------------|
| Understand proving flow | `jolt-core/src/zkvm/dag/jolt_dag.rs` | `JoltDAG::prove()` |
| See R1CS constraints | `jolt-core/src/zkvm/r1cs/constraints.rs` | `UNIFORM_R1CS` array |
| Add instruction (tracer) | `tracer/src/instruction/` | `declare_riscv_instr!` macro |
| Add instruction (prover) | `jolt-core/src/zkvm/instruction/` | `InstructionLookup` trait |
| Modify sumcheck | `jolt-core/src/subprotocols/sumcheck.rs` | `BatchedSumcheck::prove()` |
| Change memory checking | `jolt-core/src/zkvm/ram/` or `registers/` | `ReadWriteCheckingInstance` |
| Understand witness generation | `jolt-core/src/zkvm/witness.rs` | `CommittedPolynomial::generate_witness_batch()` |
| Work with polynomials | `jolt-core/src/poly/dense_mlpoly.rs` | `DenseMultilinearExtension` |
| Modify macro | `jolt-sdk/macros/src/lib.rs` | `#[proc_macro_attribute]` |
| Understand state flow | `jolt-core/src/zkvm/dag/state_manager.rs` | `StateManager` struct |

### Theory to Code Mapping

| Theory Concept | Primary Implementation | Key Files | Key Data Structures |
|----------------|------------------------|-----------|---------------------|
| Multilinear Extensions | `jolt-core/src/poly/dense_mlpoly.rs` | `dense_mlpoly.rs:30-100` | `DenseMultilinearExtension<F>` |
| Sumcheck Protocol | `jolt-core/src/subprotocols/sumcheck.rs` | `sumcheck.rs:40-300` | `SumcheckInstance` trait, `BatchedSumcheck` |
| Twist (memory checking) | `jolt-core/src/subprotocols/twist.rs` | `twist.rs`, `ram/mod.rs`, `registers/mod.rs` | `ReadWriteCheckingInstance` |
| Shout (lookup argument) | `jolt-core/src/subprotocols/shout/` | `instruction_lookups/`, `bytecode/` | `PrefixSuffixSumcheck` |
| Spartan R1CS | `jolt-core/src/zkvm/r1cs/` | `spartan/mod.rs`, `r1cs/constraints.rs` | `UNIFORM_R1CS`, `SpartanDag` |
| Dory PCS | `jolt-core/src/poly/commitment/dory.rs` | `dory.rs:30-200` | `DoryCommitmentScheme` |
| Prefix-suffix sumcheck | `jolt-core/src/zkvm/instruction_lookups/` | `read_raf_checking.rs:100-300` | `PrefixSuffixSumcheck` |
| Execution trace | `tracer/src/emulator/cpu.rs` | `cpu.rs:execute_program()` | `Vec<Cycle>`, `RISCVCycle<I>` |
| Program compilation | `jolt-core/src/host/program.rs` | `program.rs:build_with_channel()` | `Program` struct |
| Witness generation | `jolt-core/src/zkvm/witness.rs` | `witness.rs:generate_witness_batch()` | `CommittedPolynomial` enum |

### Common Commands

```bash
# Build entire workspace
cargo build --release

# Run specific example
cargo run --release -p fibonacci

# Run tests
cargo test -p jolt-core
cargo test -p tracer  # Test emulator

# Profile proving (execution trace)
cargo run --release -p jolt-core profile --name sha3 --format chrome
# View at https://ui.perfetto.dev/

# Profile proving (CPU sampling)
cargo run --release --features pprof -p jolt-core profile --name sha3
go tool pprof -http=:8080 target/release/jolt-core benchmark-runs/pprof/sha3_prove.pb

# Profile memory usage
RUST_LOG=debug cargo run --release --features allocative -p jolt-core profile --name sha3

# Lint
cargo clippy --no-deps

# Build with fast profile (dev iteration)
cargo build --profile build-fast -p jolt-core

# Clean build
cargo clean && cargo build --release
```

---

## Final Thoughts

### This Guide is a Map, Not the Territory

**Remember**:
- Reading code is how you truly learn
- This guide points you in the right direction
- Experiment, break things, ask questions
- Use `RUST_LOG=debug` liberally!

### Next Steps

1. **Pick a small task** from GitHub issues (label: "good first issue")
2. **Read the relevant component** (section 8)
3. **Try making a change** (section 12)
4. **Run tests** to verify your change
5. **Ask for help** when stuck (GitHub discussions)

### Contributing Back

Found this guide helpful? Consider:
- Improving explanations that confused you
- Adding code examples that helped you understand
- Documenting patterns you discovered
- Updating file references when code moves

**This is a living document** - as Jolt evolves, so should this guide.

---

**For theory**: Read [docs/Jolt.md](Jolt.md)
**For usage**: Read [CLAUDE.md](../CLAUDE.md)
**For code**: You're ready now! Start with section 12 and dive in.
