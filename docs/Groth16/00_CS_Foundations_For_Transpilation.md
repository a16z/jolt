# Computer Science Foundations for Transpilation

**Document Purpose**: Comprehensive technical foundations in computer architecture, compilation theory, and program analysis needed to understand and implement the Jolt-to-Groth16 transpilation pipeline.

**Audience**: Engineers with mathematical background who want to deeply understand the CS theory underlying code transformation, extraction, and circuit compilation.

**Scope**: This document covers fundamental CS concepts from first principles with rigorous definitions, small working examples, and connections to the transpilation challenge. It complements the cryptography-focused [01_Jolt_Theory_Enhanced.md](../01_Jolt_Theory_Enhanced.md) by providing the systems and compiler background.

**Date**: 2025-11-07
**Author**: Parti

---

## Table of Contents

### Part 0: Computer Architecture Fundamentals
0.1. [The Abstraction Hierarchy: From Silicon to Source](#01-the-abstraction-hierarchy)
0.2. [Memory Architecture and Addressing](#02-memory-architecture-and-addressing)
0.3. [Registers: The Fastest Memory](#03-registers-the-fastest-memory)
0.4. [Instruction Set Architectures](#04-instruction-set-architectures)
0.5. [The Fetch-Decode-Execute Cycle](#05-the-fetch-decode-execute-cycle)
0.6. [RISC vs CISC: Philosophical Differences](#06-risc-vs-cisc)

### Part 1: Compilation Pipeline Deep Dive
1.1. [The Compilation Phases](#11-the-compilation-phases)
1.2. [Lexical Analysis: From Text to Tokens](#12-lexical-analysis)
1.3. [Syntactic Analysis: Building the AST](#13-syntactic-analysis)
1.4. [Semantic Analysis: Type Checking and Scoping](#14-semantic-analysis)
1.5. [Intermediate Representations](#15-intermediate-representations)
1.6. [Optimization Passes](#16-optimization-passes)
1.7. [Code Generation](#17-code-generation)

### Part 2: Abstract Syntax Trees and IRs
2.1. [Abstract Syntax Trees: Structure and Properties](#21-abstract-syntax-trees)
2.2. [Three-Address Code](#22-three-address-code)
2.3. [Static Single Assignment (SSA) Form](#23-static-single-assignment-form)
2.4. [Control Flow Graphs](#24-control-flow-graphs)
2.5. [Data Flow Analysis](#25-data-flow-analysis)

### Part 3: Transpilation Theory
3.1. [Source-to-Source Compilation](#31-source-to-source-compilation)
3.2. [Semantic Preservation](#32-semantic-preservation)
3.3. [Type System Mapping](#33-type-system-mapping)
3.4. [Control Flow Translation](#34-control-flow-translation)
3.5. [Memory Model Translation](#35-memory-model-translation)

### Part 4: Circuit Representation
4.1. [Arithmetic Circuits](#41-arithmetic-circuits)
4.2. [R1CS Constraint Systems](#42-r1cs-constraint-systems)
4.3. [From Imperative Code to Circuits](#43-from-imperative-code-to-circuits)
4.4. [Witness Generation](#44-witness-generation)

### Part 5: The Extraction Problem
5.1. [Static Analysis vs Tracing](#51-static-analysis-vs-tracing)
5.2. [Operation Capture Techniques](#52-operation-capture-techniques)
5.3. [Memory Management for IR](#53-memory-management-for-ir)
5.4. [Optimization: Common Subexpression Elimination](#54-optimization-cse)

### Appendices
- [Appendix A: Notation Reference](#appendix-a-notation-reference)
- [Appendix B: Worked Examples](#appendix-b-worked-examples)
- [Appendix C: Connection to Jolt Transpilation](#appendix-c-connection-to-jolt)

---

# Part 0: Computer Architecture Fundamentals

> **Purpose**: Before understanding transpilation, we must understand what programs *are* at the machine level. This section builds the foundation from hardware to software abstraction layers.

## 0.1 The Abstraction Hierarchy: From Silicon to Source

### The Full Stack

Modern computing operates through layers of abstraction. Understanding transpilation requires clarity about which layer we're operating at and how information transforms between layers.

**The complete hierarchy** (bottom to top):

```
Level 0: Physics
├─ Transistors (switches: voltage high/low $\to$ binary 1/0)
├─ Logic gates (NAND, NOR, XOR built from transistors)
└─ Flip-flops and latches (memory cells)

Level 1: Digital Logic
├─ Combinational circuits (ALU components)
├─ Sequential circuits (registers, counters)
└─ Memory arrays (SRAM, DRAM cells)

Level 2: Microarchitecture
├─ Datapath (ALU, register file, buses)
├─ Control unit (FSM driving datapath)
└─ Cache hierarchy (L1, L2, L3)

Level 3: Instruction Set Architecture (ISA)
├─ Machine code (binary instruction encoding)
├─ Opcodes and operands (what operations exist)
└─ Architectural state (visible registers, memory model)

Level 4: Assembly Language
├─ Symbolic opcodes (ADD, SUB, JMP)
├─ Labels and symbolic addresses
└─ Assembler directives

Level 5: High-Level Languages
├─ Variables and types (int x = 5;)
├─ Control structures (if, while, for)
└─ Functions and procedures

Level 6: Domain-Specific Languages
├─ SQL (database queries)
├─ Gnark (arithmetic circuits)
└─ Lean4 (proof languages)
```

### Key Insight: Information Preservation and Loss

As we move **up** the hierarchy (silicon $\to$ source):
- Gain: Abstraction, expressiveness, human readability
- Lose: Precise control, timing guarantees, hardware awareness

As we move **down** the hierarchy (source $\to$ silicon):
- Gain: Precise semantics, deterministic execution
- Lose: High-level structure, intent

**Critical for transpilation**: When we transpile (source A $\to$ source B), we're typically staying at the same level but *changing representation*. The challenge is preserving semantics while adapting to the target's idioms.

### Machine Code vs Bytecode vs Source Code

These terms are often conflated. Precise definitions:

#### Machine Code

**Definition**: Binary-encoded instructions directly executable by hardware (Level 3).

**Properties**:
- Fixed-width instruction encoding (e.g., RISC-V: 32-bit instruction words)
- Decoded by CPU control unit
- Addresses refer to physical or virtual memory
- Architecture-specific (x86 machine code won't run on ARM)

**Example (RISC-V RV64I)**:
```
Address     Hex          Binary                              Disassembly
0x1000      0x00A58593   00000000 10100101 10000101 10010011  ADDI a1, a1, 10
```

Breaking down the encoding:
```
Bits [31:20]: 000000001010  $\to$ immediate value (10 in decimal)
Bits [19:15]: 01011        $\to$ rs1 (source register a1 = x11)
Bits [14:12]: 000          $\to$ funct3 (ADDI operation)
Bits [11:7]:  01011        $\to$ rd (destination register a1 = x11)
Bits [6:0]:   0010011      $\to$ opcode (I-type arithmetic)
```

#### Bytecode

**Definition**: Intermediate representation designed for a virtual machine, not hardware (Level 4.5).

**Properties**:
- Platform-independent
- Often variable-length encoding
- Executed by interpreter or JIT compiler
- Examples: JVM bytecode, Python bytecode, WebAssembly

**Example (JVM bytecode)**:
```
Bytecode       Mnemonic      Stack Effect
0x10 0x0A      BIPUSH 10     [] $\to$ [10]
0x3C           ISTORE_1      [10] $\to$ []    (store to local variable 1)
0x1B           ILOAD_1       [] $\to$ [10]    (load from local variable 1)
0x10 0x14      BIPUSH 20     [10] $\to$ [10, 20]
0x60           IADD          [10, 20] $\to$ [30]
```

**Why bytecode exists**: Allows "write once, run anywhere" by targeting a virtual machine specification rather than hardware.

#### Source Code

**Definition**: Human-readable text in a high-level language (Level 5+).

**Properties**:
- Uses keywords, identifiers, operators
- Structured by grammar rules (syntax)
- Meaning defined by language semantics
- Requires compilation or interpretation

**Example (Rust)**:
```rust
fn add_ten(x: i32) -> i32 {
    let y = x + 10;
    y
}
```

### The Role of ELF (Executable and Linkable Format)

**ELF is NOT bytecode or machine code—it's a container format.**

**What ELF contains**:

```
┌─────────────────────────────────┐
│ ELF Header                       │
│  ├─ Magic: 0x7F 'E' 'L' 'F'    │
│  ├─ Class: 64-bit               │
│  ├─ Data: Little-endian         │
│  ├─ Machine: RISC-V             │
│  └─ Entry point: 0x80000000     │
├─────────────────────────────────┤
│ Program Headers (segments)       │
│  ├─ LOAD: map .text to memory   │
│  ├─ LOAD: map .data to memory   │
│  └─ DYNAMIC: linker info         │
├─────────────────────────────────┤
│ Section: .text (executable)      │
│  ├─ Machine code instructions    │ $\leftarrow$ This is what CPU executes
│  └─ Size: 1024 bytes             │
├─────────────────────────────────┤
│ Section: .rodata (read-only)     │
│  ├─ String constants             │
│  └─ Jump tables                  │
├─────────────────────────────────┤
│ Section: .data (initialized)     │
│  ├─ Global variables             │
│  └─ Static data                  │
├─────────────────────────────────┤
│ Section: .bss (uninitialized)    │
│  └─ Zero-initialized globals     │
├─────────────────────────────────┤
│ Section: .symtab (symbol table)  │
│  ├─ Function names               │
│  └─ Variable names               │
├─────────────────────────────────┤
│ Section: .strtab (string table)  │
│  └─ Symbol name strings          │
└─────────────────────────────────┘
```

**Compilation pipeline to ELF**:

```
Rust source (.rs)
    ↓ [rustc frontend: parse, type check, borrow check]
MIR (Mid-level IR)
    ↓ [rustc backend: optimization]
LLVM IR
    ↓ [LLVM: target-specific optimization]
Assembly (.s)
    ↓ [Assembler: encode to machine code]
Object file (.o)
    ↓ [Linker: resolve symbols, layout sections]
ELF executable
```

**What Jolt does with ELF**:

```rust
// In jolt-core/src/host/program.rs
pub struct Program {
    pub elf: Vec<u8>,                    // Raw ELF bytes
    pub bytecode: Vec<ELFInstruction>,   // Extracted from .text section
    pub memory_layout: MemoryLayout,     // Address ranges
}

impl Program {
    pub fn from_elf(elf_bytes: &[u8]) -> Self {
        // Parse ELF structure
        let elf = goblin::elf::Elf::parse(elf_bytes)?;

        // Extract .text section (machine code)
        let text_section = elf.section_headers
            .iter()
            .find(|s| s.sh_name == ".text")?;

        // Decode machine code to ELFInstruction structs
        let bytecode = decode_instructions(
            &elf_bytes[text_section.sh_offset..],
            text_section.sh_size
        );

        // Set up memory layout
        let memory_layout = MemoryLayout::from_elf(&elf);

        Self { elf: elf_bytes.to_vec(), bytecode, memory_layout }
    }
}
```

**Three distinct concepts**:
1. **ELF file**: Container with metadata + multiple sections
2. **Machine code**: Binary instructions in `.text` section
3. **Bytecode** (Jolt's usage): Decoded representation of machine code as structs

**Why this matters for Jolt**:
- ELF provides structure (code vs data separation)
- Machine code is what gets proven (execution trace)
- Bytecode commitment is what verifier checks against

---

## 0.2 Memory Architecture and Addressing

### The Memory Hierarchy

Modern computers organize memory by speed-capacity trade-off:

```
Register File
├─ Size: 64-256 bytes (32-64 registers × 4-8 bytes each)
├─ Access time: 0 cycles (synchronous with ALU)
└─ Bandwidth: 10-20 reads/writes per cycle

L1 Cache
├─ Size: 32-64 KB per core
├─ Access time: 3-4 cycles
└─ Bandwidth: 1-2 cache lines (64 bytes) per cycle

L2 Cache
├─ Size: 256 KB - 1 MB per core
├─ Access time: 10-20 cycles
└─ Bandwidth: Shared with L1

L3 Cache
├─ Size: 8-32 MB shared across all cores
├─ Access time: 40-75 cycles
└─ Bandwidth: Shared across all cores

DRAM (Main Memory)
├─ Size: 8-128 GB
├─ Access time: 200-300 cycles
└─ Bandwidth: 50-100 GB/s

SSD/HDD (Storage)
├─ Size: 256 GB - 16 TB
├─ Access time: 10,000+ cycles (SSD), 10,000,000+ cycles (HDD)
└─ Bandwidth: 500 MB/s (SSD), 100 MB/s (HDD)
```

**Critical insight for zkVMs**: Jolt's memory model abstracts away caching. The prover must track *every* memory access to RAM and registers, but doesn't model cache behavior. This is sound because:
- Cache is transparent (doesn't change program semantics)
- We're proving *correctness*, not *performance*

### Memory Models: Byte-Addressable vs Word-Addressable

#### Byte-Addressable Memory (RISC-V, x86, ARM)

**Definition**: Each memory address refers to a single byte (8 bits).

**Addressing arithmetic**:
```
Address    Value (hex)    Interpretation
0x1000     0x12          Byte at address 0x1000
0x1001     0x34          Byte at address 0x1001
0x1002     0x56          Byte at address 0x1002
0x1003     0x78          Byte at address 0x1003

Reading 32-bit word from 0x1000 (little-endian):
    0x78563412

Reading 64-bit doubleword from 0x1000 (little-endian):
    0x????????78563412  (need 4 more bytes)
```

**Why little-endian**: Least significant byte at lowest address. Makes multi-precision arithmetic easier.

#### Word-Addressable Memory (Hypothetical)

**Definition**: Each memory address refers to a word (e.g., 32-bit or 64-bit).

**Addressing arithmetic**:
```
Address    Value (hex)        Interpretation
0x0        0x12345678         Word 0
0x1        0x9ABCDEF0         Word 1
0x2        0x11223344         Word 2

To address individual bytes: Need byte-within-word offset
```

**Why word-addressable is simpler for zkVMs**:
- Fewer memory cells to track (divide address space by word size)
- Most operations work on full words anyway
- Simplifies circuit size

#### Jolt's Hybrid Approach

**RISC-V is byte-addressable**, but Jolt optimizes by treating memory as **doubleword-addressable internally**:

**Address remapping formula**:
```
Jolt_index = (RISC-V_address - 0x80000000) / 8 + 1
```

**Example**:
```
RISC-V address    Jolt index    Contents
0x80000000   $\to$   1            Doubleword 1 (bytes 0-7)
0x80000008   $\to$   2            Doubleword 2 (bytes 8-15)
0x80000010   $\to$   3            Doubleword 3 (bytes 16-23)
```

**For sub-word operations** (LB, LH, LW, etc.), Jolt expands them to load/store full doubleword + masking:

**Example: LB (load byte)**:
```
RISC-V instruction: LB r1, 0x80000002(r0)

Expanded to virtual sequence:
1. LD r_temp, 0x80000000(r0)      # Load doubleword containing target byte
2. SRLI r_temp, r_temp, 16        # Shift right by 16 bits (byte offset 2)
3. ANDI r1, r_temp, 0xFF          # Mask to get lowest byte
4. SEXT r1, r1, 8                 # Sign-extend from 8 bits
```

**Why this works**:
- Fewer memory consistency checks (only track doubleword operations)
- Twist's K parameter (memory size) reduced by factor of 8
- Trade-off: More instructions per byte operation (but rare in practice)

### Memory Layout: Stack vs Heap

#### The Stack

**Definition**: Contiguous memory region managed via stack pointer, grows downward (by convention).

**Operations**:
- **PUSH**: Decrement SP, write value
- **POP**: Read value, increment SP

**Used for**:
- Function call frames (local variables, return address, saved registers)
- Temporary storage
- Expression evaluation in stack-based VMs

**Example (RISC-V ABI)**:
```
Stack pointer (SP = x2) initially at 0x7FFFFFFF

Function call sequence:
    ADDI sp, sp, -32      # Allocate 32-byte frame
    SD   ra, 24(sp)       # Save return address
    SD   s0, 16(sp)       # Save saved register s0
    SD   s1, 8(sp)        # Save saved register s1
    # ... function body ...
    LD   s1, 8(sp)        # Restore s1
    LD   s0, 16(sp)       # Restore s0
    LD   ra, 24(sp)       # Restore return address
    ADDI sp, sp, 32       # Deallocate frame
    RET                   # Return (jump to ra)
```

**Stack frame layout**:
```
Higher addresses
    ↑
0x7FFFFFFF  ┌───────────────┐  $\leftarrow$ SP before call
            │  Return addr  │
            ├───────────────┤
            │  Saved s0     │
            ├───────────────┤
            │  Saved s1     │
            ├───────────────┤
            │  Local var 1  │
            ├───────────────┤
            │  Local var 2  │
            ├───────────────┤
            │  ...          │
0x7FFFFFE0  └───────────────┘  $\leftarrow$ SP after prologue (32 bytes below)
    ↓
Lower addresses
```

#### The Heap

**Definition**: Dynamically allocated memory, managed by allocator (malloc/free or Vec/Box).

**Operations**:
- **Allocate**: Request N bytes, receive pointer
- **Free**: Return pointer, mark memory available

**Used for**:
- Dynamic data structures (vectors, trees, hash maps)
- Long-lived data
- Variable-size data

**Example (Rust Box::new)**:
```rust
let x = Box::new(42);  // Allocates 4 bytes on heap, x is pointer
```

**Heap management in Jolt**:
- Guest programs can use `alloc` crate (Vec, Box, etc.)
- Allocator built into guest runtime
- Jolt's memory tracking treats heap same as any RAM access

**Why stack vs heap matters for transpilation**:
- Stack: Local, structured, easy to analyze statically
- Heap: Global, unstructured, requires pointer aliasing analysis
- When extracting operations, stack operations often compile to simple register operations, while heap operations require memory load/store in target

---

## 0.3 Registers: The Fastest Memory

### What Registers Are

**Definition**: Small, fixed-size storage locations directly accessible by ALU.

**Physical implementation**: D flip-flops with address decoder, integrated into CPU datapath.

**Key properties**:
1. **Fixed number**: ISA defines how many (e.g., RISC-V: 32 general-purpose registers)
2. **Fixed width**: All registers same size (e.g., RISC-V RV64: 64-bit registers)
3. **Named access**: Instructions specify registers by number (e.g., `ADD x3, x1, x2`)
4. **Zero latency**: Register reads/writes happen in same clock cycle as ALU operation

### RISC-V Register File

**RV64I General-Purpose Registers** (32 total):

```
Register    ABI Name    Description                  Saved by
x0          zero        Hardwired zero               N/A (immutable)
x1          ra          Return address               Caller
x2          sp          Stack pointer                Callee
x3          gp          Global pointer               N/A (static)
x4          tp          Thread pointer               N/A (static)
x5-x7       t0-t2       Temporaries                  Caller
x8          s0/fp       Saved register/Frame ptr     Callee
x9          s1          Saved register               Callee
x10-x11     a0-a1       Function args/return vals    Caller
x12-x17     a2-a7       Function arguments           Caller
x18-x27     s2-s11      Saved registers              Callee
x28-x31     t3-t6       Temporaries                  Caller
```

**Calling convention** (RISC-V ABI):
- **a0-a7**: Function arguments (first 8 integer args)
- **a0-a1**: Return values (up to 128 bits)
- **t0-t6**: Temporaries (caller-saved, not preserved across calls)
- **s0-s11**: Saved registers (callee-saved, must be preserved)
- **ra**: Return address (where to jump back after function)
- **sp**: Stack pointer (top of current stack frame)

**Example: Function call**:
```rust
fn add(x: u64, y: u64) -> u64 {
    x + y
}

fn main() {
    let result = add(10, 20);
}
```

**Compiled to RISC-V**:
```asm
# Function: add(x: u64, y: u64) -> u64
add:
    # Arguments: a0 = x, a1 = y
    add a0, a0, a1      # a0 = x + y (return value in a0)
    ret                 # Return (jump to address in ra)

# Function: main()
main:
    addi sp, sp, -16    # Allocate stack frame
    sd   ra, 8(sp)      # Save return address

    # Call add(10, 20)
    li   a0, 10         # Load immediate 10 into a0 (first arg)
    li   a1, 20         # Load immediate 20 into a1 (second arg)
    call add            # Call function (stores PC+4 in ra, jumps to add)
    # Result is now in a0

    ld   ra, 8(sp)      # Restore return address
    addi sp, sp, 16     # Deallocate stack frame
    ret                 # Return
```

### Register Renaming (Microarchitectural Detail)

**Problem**: Instruction-level parallelism limited by register reuse.

**Example**:
```asm
add x3, x1, x2    # x3 = x1 + x2
sub x5, x3, x4    # x5 = x3 - x4  (depends on previous x3)
add x3, x6, x7    # x3 = x6 + x7  (reuses x3, creates false dependency)
mul x8, x3, x9    # x8 = x3 * x9  (depends on new x3)
```

**False dependency**: Third instruction doesn't actually depend on first, but reuses x3.

**Solution (microarchitectural)**: Rename x3 to different physical register:
```
add p10, p1, p2   # x3 $\to$ p10
sub p11, p10, p4  # x5 $\to$ p11, depends on p10
add p12, p6, p7   # x3 $\to$ p12 (different physical register, no dependency!)
mul p13, p12, p9  # x8 $\to$ p13, depends on p12
```

**Why Jolt doesn't model this**: Register renaming is invisible at ISA level. Jolt proves correctness of architectural state (x0-x31), not microarchitectural state (physical registers).

### Virtual Registers in Jolt

**Jolt extends RISC-V with 32 virtual registers** (x32-x63 conceptually) for use in virtual instruction sequences.

**Why virtual registers**:
- Virtual sequences (e.g., division) need scratch space
- Can't use architectural registers (would violate RISC-V semantics)
- Virtual registers only exist during multi-step operation expansion

**Example: Division using virtual registers**:
```asm
# RISC-V instruction: DIV x3, x10, x11  (x3 = x10 / x11)

# Expanded to virtual sequence:
# Step 1: Load advice (quotient guess from prover)
VLOAD v0, ADVICE_QUOT      # v0 = virtual register 0

# Step 2: Load advice (remainder guess)
VLOAD v1, ADVICE_REM       # v1 = virtual register 1

# Step 3: Verify: quotient * divisor + remainder = dividend
MUL   v2, v0, x11          # v2 = quotient * divisor
ADD   v3, v2, v1           # v3 = quotient * divisor + remainder
ASSERT_EQ v3, x10          # Assert v3 == dividend

# Step 4: Verify: remainder < divisor
SLTU  v4, v1, x11          # v4 = 1 if remainder < divisor, else 0
ASSERT_EQ v4, 1            # Assert remainder < divisor

# Step 5: Store quotient to destination
MOV   x3, v0               # x3 = quotient
```

**Key point**: Virtual registers (v0-v31) never appear in architectural state. They're internal to the virtual sequence expansion.

---

## 0.4 Instruction Set Architectures

### ISA: The Hardware-Software Contract

**Definition**: An Instruction Set Architecture (ISA) is the specification of:
1. **Instruction formats**: How instructions are encoded (bits $\to$ operation)
2. **Addressing modes**: How operands are specified (register, immediate, memory)
3. **Data types**: What sizes operations work on (byte, word, doubleword)
4. **Register set**: How many registers, what purpose
5. **Memory model**: Address space, alignment, endianness
6. **Exception/interrupt handling**: What happens on errors

**ISA is NOT**:
- A specific CPU implementation (e.g., Intel Core i9 implements x86-64)
- A programming language (though assembly is tied to ISA)
- A microarchitecture (e.g., out-of-order execution, branch prediction)

**ISA examples**:
- **x86-64**: Variable-length instructions, complex addressing modes, many instructions (CISC)
- **ARM**: Fixed-length instructions (32-bit or 16-bit Thumb), load-store architecture (RISC)
- **RISC-V**: Fixed-length base (32-bit), optional compressed (16-bit), modular extensions

### RISC-V: A Modular ISA

**Base ISAs**:
- **RV32I**: 32-bit integer base (32 instructions)
- **RV64I**: 64-bit integer base (extends RV32I with 64-bit ops)
- **RV128I**: 128-bit integer base (experimental)

**Standard extensions**:
- **M**: Integer multiplication and division (8 instructions)
- **A**: Atomic operations (11 instructions)
- **F**: Single-precision floating-point (26 instructions)
- **D**: Double-precision floating-point (26 instructions)
- **C**: Compressed instructions (16-bit encoding, ~40 instructions)

**Jolt targets RV64IMAC**:
- RV64I: Base 64-bit integer instructions
- M: Multiplication/division
- A: Atomic operations (for concurrent programming)
- C: Compressed instructions

### Instruction Formats

RISC-V uses six instruction formats, all derived from 32-bit words:

#### R-type (Register-register operations)

```
Format:
 31        25 24      20 19      15 14    12 11       7 6          0
┌───────────┬──────────┬──────────┬────────┬──────────┬────────────┐
│  funct7   │   rs2    │   rs1    │ funct3 │    rd    │   opcode   │
│  (7 bits) │ (5 bits) │ (5 bits) │(3 bits)│ (5 bits) │  (7 bits)  │
└───────────┴──────────┴──────────┴────────┴──────────┴────────────┘
```

**Example: ADD x3, x1, x2** (x3 = x1 + x2)
```
funct7 = 0000000   (ADD, not SUB)
rs2    = 00010     (x2)
rs1    = 00001     (x1)
funct3 = 000       (ADD operation)
rd     = 00011     (x3)
opcode = 0110011   (R-type ALU operation)

Full encoding: 0x002081B3
```

#### I-type (Immediate operations)

```
Format:
 31                    20 19      15 14    12 11       7 6          0
┌────────────────────────┬──────────┬────────┬──────────┬────────────┐
│      immediate         │   rs1    │ funct3 │    rd    │   opcode   │
│      (12 bits)         │ (5 bits) │(3 bits)│ (5 bits) │  (7 bits)  │
└────────────────────────┴──────────┴────────┴──────────┴────────────┘
```

**Example: ADDI x1, x1, 10** (x1 = x1 + 10)
```
immediate = 000000001010  (10 in binary, sign-extended)
rs1       = 00001         (x1)
funct3    = 000           (ADDI operation)
rd        = 00001         (x1, same as source)
opcode    = 0010011       (I-type ALU operation)

Full encoding: 0x00A08093
```

#### S-type (Store operations)

```
Format:
 31        25 24      20 19      15 14    12 11       7 6          0
┌───────────┬──────────┬──────────┬────────┬──────────┬────────────┐
│  imm[11:5]│   rs2    │   rs1    │ funct3 │ imm[4:0] │   opcode   │
│  (7 bits) │ (5 bits) │ (5 bits) │(3 bits)│ (5 bits) │  (7 bits)  │
└───────────┴──────────┴──────────┴────────┴──────────┴────────────┘
```

**Why split immediate?** Keeps register fields (rs1, rs2) in same positions across formats, simplifying decoding.

**Example: SD x5, 8(x2)** (store x5 to address x2+8)
```
imm[11:5] = 0000000  (upper 7 bits of offset 8)
rs2       = 00101    (x5, data to store)
rs1       = 00010    (x2, base address)
funct3    = 011      (SD, store doubleword)
imm[4:0]  = 01000    (lower 5 bits of offset 8)
opcode    = 0100011  (STORE)

Reconstructed offset: 0000000 || 01000 = 0x08 (8 in decimal)
```

#### B-type (Branch operations)

```
Format:
 31  30      25 24      20 19      15 14    12 11    8  7  6          0
┌────┬─────────┬──────────┬──────────┬────────┬──────┬───┬────────────┐
│[12]│ [10:5]  │   rs2    │   rs1    │ funct3 │ [4:1]│[11]│   opcode   │
│(1b)│ (6 bits)│ (5 bits) │ (5 bits) │(3 bits)│(4b)  │(1b)│  (7 bits)  │
└────┴─────────┴──────────┴──────────┴────────┴──────┴───┴────────────┘
```

**Example: BEQ x1, x2, 16** (if x1 == x2, PC = PC + 16)
```
imm[12]   = 0
imm[10:5] = 000000
rs2       = 00010    (x2)
rs1       = 00001    (x1)
funct3    = 000      (BEQ, branch if equal)
imm[4:1]  = 1000     (offset bits [4:1])
imm[11]   = 0
opcode    = 1100011  (BRANCH)

Reconstructed offset (sign-extended, always multiple of 2):
0 || 0 || 000000 || 1000 || 0 = 0x10 (16 in decimal)
```

#### U-type (Upper immediate)

```
Format:
 31                                  12 11       7 6          0
┌──────────────────────────────────────┬──────────┬────────────┐
│             immediate                │    rd    │   opcode   │
│            (20 bits)                 │ (5 bits) │  (7 bits)  │
└──────────────────────────────────────┴──────────┴────────────┘
```

**Example: LUI x1, 0x12345** (load upper immediate, x1 = 0x12345000)
```
immediate = 00010010001101000101  (0x12345 in binary)
rd        = 00001                 (x1)
opcode    = 0110111               (LUI)

Result: x1 = 0x0000000012345000  (immediate shifted left by 12 bits)
```

#### J-type (Jump operations)

```
Format:
 31  30       21  20  19         12 11       7 6          0
┌────┬───────────┬───┬────────────┬──────────┬────────────┐
│[20]│  [10:1]   │[11]│   [19:12]  │    rd    │   opcode   │
│(1b)│ (10 bits) │(1b)│  (8 bits)  │ (5 bits) │  (7 bits)  │
└────┴───────────┴───┴────────────┴──────────┴────────────┘
```

**Example: JAL x1, 100** (jump to PC+100, save PC+4 in x1)
```
imm[20]    = 0
imm[10:1]  = 0000110010  (bits [10:1] of offset 100)
imm[11]    = 0
imm[19:12] = 00000000
rd         = 00001       (x1, store return address)
opcode     = 1101111     (JAL)

Reconstructed offset: 0 || 00000000 || 0 || 0000110010 || 0 = 100
```

### Decoding Example: Complete Walkthrough

**Given machine code: 0x003100B3**

**Step 1: Extract opcode (bits [6:0])**
```
0x003100B3 in binary:
0000 0000 0011 0001 0000 0000 1011 0011
                              └──┬──┘
                              0110011 (R-type)
```

**Step 2: Identify format**
```
Opcode 0110011 $\to$ R-type arithmetic operation
```

**Step 3: Extract fields**
```
 31        25 24      20 19      15 14    12 11       7 6          0
┌───────────┬──────────┬──────────┬────────┬──────────┬────────────┐
│ 0000000   │  00011   │  00010   │  000   │  00001   │  0110011   │
└───────────┴──────────┴──────────┴────────┴──────────┴────────────┘
  funct7      rs2        rs1       funct3     rd         opcode
```

**Step 4: Decode operation**
```
funct7 = 0000000, funct3 = 000 $\to$ ADD
rs2 = 00011 $\to$ x3
rs1 = 00010 $\to$ x2
rd  = 00001 $\to$ x1

Instruction: ADD x1, x2, x3  (x1 = x2 + x3)
```

---

## 0.5 The Fetch-Decode-Execute Cycle

### The Classical Von Neumann Cycle

All stored-program computers follow this pattern:

```
┌──────────────────────────────────────────┐
│                                          │
│  ┌────────┐      ┌────────┐      ┌────┐ │
│  │ FETCH  │  $\to$   │ DECODE │  $\to$   │ EX │ │
│  └────────┘      └────────┘      └────┘ │
│       ↓                              ↓   │
│       └──────────── PC $\leftarrow$─────────────┘   │
│                                          │
└──────────────────────────────────────────┘
```

### Phase 1: FETCH

**Goal**: Retrieve next instruction from memory.

**Steps**:
1. Read PC (Program Counter) to get instruction address
2. Send address to memory/cache
3. Wait for memory to respond with instruction word
4. Store instruction in Instruction Register (IR)

**Example**:
```
PC = 0x80000000

Memory[0x80000000] = 0x002081B3  (ADD x3, x1, x2)

After fetch:
  IR = 0x002081B3
  PC = 0x80000000  (unchanged yet)
```

**In RISC-V**: Instructions are 32-bit aligned, so PC always increments by 4 (or 2 for compressed instructions).

### Phase 2: DECODE

**Goal**: Interpret instruction bit pattern to determine operation and operands.

**Steps**:
1. Extract opcode (bits [6:0])
2. Identify format based on opcode
3. Extract register specifiers (rs1, rs2, rd)
4. Extract immediate values (if any)
5. Generate control signals for datapath

**Example**:
```
IR = 0x002081B3

Decode:
  Opcode = 0110011 $\to$ R-type
  funct7 = 0000000, funct3 = 000 $\to$ ADD operation
  rs1 = 00001 $\to$ x1
  rs2 = 00010 $\to$ x2
  rd  = 00011 $\to$ x3

Control signals:
  - Read register x1
  - Read register x2
  - ALU operation: ADD
  - Write result to register x3
```

### Phase 3: EXECUTE

**Goal**: Perform the operation.

**Steps**:
1. Read operands from register file (if register instruction)
2. Compute result using ALU or other functional unit
3. Update PC (normally PC+4, or branch target if branch taken)

**Example**:
```
Before:
  x1 = 10
  x2 = 20
  x3 = 0  (uninitialized)
  PC = 0x80000000

Execute ADD x3, x1, x2:
  1. Read register x1 $\to$ value 10
  2. Read register x2 $\to$ value 20
  3. ALU: 10 + 20 $\to$ 30
  4. Write 30 to register x3
  5. Update PC: 0x80000000 + 4 = 0x80000004

After:
  x1 = 10
  x2 = 20
  x3 = 30  $\leftarrow$ Changed
  PC = 0x80000004  $\leftarrow$ Next instruction
```

### Memory Operations: Load/Store

**LOAD (LD)**: Read from memory into register

**Example: LD x5, 8(x2)** (load doubleword from address x2+8 into x5)

```
Before:
  x2 = 0x80001000  (base address)
  x5 = 0           (uninitialized)
  Memory[0x80001008] = 0x123456789ABCDEF0

Execute:
  1. Compute address: x2 + 8 = 0x80001000 + 8 = 0x80001008
  2. Send read request to memory at address 0x80001008
  3. Wait for memory to respond with data
  4. Write data to register x5

After:
  x5 = 0x123456789ABCDEF0  $\leftarrow$ Loaded from memory
  PC = PC + 4
```

**STORE (SD)**: Write from register to memory

**Example: SD x5, 8(x2)** (store doubleword from x5 to address x2+8)

```
Before:
  x2 = 0x80001000  (base address)
  x5 = 0xDEADBEEFCAFEBABE
  Memory[0x80001008] = 0x0000000000000000

Execute:
  1. Compute address: x2 + 8 = 0x80001008
  2. Read data from register x5: 0xDEADBEEFCAFEBABE
  3. Send write request to memory at address 0x80001008 with data

After:
  Memory[0x80001008] = 0xDEADBEEFCAFEBABE  $\leftarrow$ Stored
  PC = PC + 4
```

### Branch and Jump Instructions

**Branch (conditional)**: Update PC if condition holds

**Example: BEQ x1, x2, 16** (if x1 == x2, then PC = PC + 16)

```
Before:
  x1 = 42
  x2 = 42
  PC = 0x80000100

Execute:
  1. Read x1 and x2
  2. Compare: x1 == x2? $\to$ 42 == 42 $\to$ TRUE
  3. Compute target: PC + 16 = 0x80000100 + 16 = 0x80000110
  4. Update PC to target

After:
  PC = 0x80000110  $\leftarrow$ Branch taken
```

**If x1 ≠ x2**:
```
PC = 0x80000104  $\leftarrow$ PC + 4, branch not taken
```

**Jump (unconditional)**: Always update PC

**Example: JAL x1, 100** (jump to PC+100, save PC+4 in x1)

```
Before:
  x1 = 0
  PC = 0x80000200

Execute:
  1. Save return address: x1 = PC + 4 = 0x80000204
  2. Compute target: PC + 100 = 0x80000200 + 100 = 0x80000264
  3. Update PC to target

After:
  x1 = 0x80000204  $\leftarrow$ Return address saved
  PC = 0x80000264  $\leftarrow$ Jumped
```

### What Jolt Proves About Execution

For each cycle $t$ in the execution trace of length $T$:

**1. Correct instruction fetch** (Bytecode component):
$$\text{Instruction at } \text{PC}_t \text{ matches committed bytecode}$$

Proved via Shout lookup argument: Claim $\text{instr}_t = \text{Bytecode}[\text{PC}_t]$

**2. Correct instruction execution** (Instruction component):
$$\text{Output}_t = f(\text{Input1}_t, \text{Input2}_t)$$

where $f$ is the instruction's specification (e.g., ADD, XOR, MUL).

Proved via Shout lookup: Claim $(\text{Input1}, \text{Input2}) \to \text{Output}$ in lookup table.

**3. Correct state transitions** (R1CS, Registers, RAM):
- **Registers**: $\text{Reg}[r]_t = \text{last write to } r \text{ before cycle } t$
- **RAM**: $\text{RAM}[a]_t = \text{last write to } a \text{ before cycle } t$
- **PC**: $\text{PC}_{t+1} = \begin{cases} \text{PC}_t + 4 & \text{if not branch/jump} \\ \text{target} & \text{if branch taken or jump} \end{cases}$

Proved via Twist (Registers, RAM) and Spartan (PC updates, consistency checks).

---

## 0.6 RISC vs CISC: Philosophical Differences

### Definitions

**RISC (Reduced Instruction Set Computer)**:
- Philosophy: Simple instructions, executed in one cycle (ideally)
- Examples: RISC-V, ARM, MIPS, SPARC
- Design goals: Regular encoding, simple hardware, compiler-friendly

**CISC (Complex Instruction Set Computer)**:
- Philosophy: Complex instructions, multiple cycles, close to high-level constructs
- Examples: x86, x86-64, VAX, 68000
- Design goals: Reduce code size, simplify compilers (historically)

### Comparison

| Aspect | RISC | CISC |
|--------|------|------|
| **Instruction count** | ~100-200 instructions | ~1000+ instructions |
| **Instruction length** | Fixed (typically 32-bit) | Variable (1-15 bytes on x86) |
| **Addressing modes** | Simple (register, immediate, register+offset) | Complex (many modes) |
| **Execution time** | 1 cycle per instruction (ideal, pipelined) | Multiple cycles |
| **Registers** | Many (32+) | Few (8-16, historically) |
| **Memory access** | Load/store only | Many instructions can access memory |
| **Compiler complexity** | More complex (must schedule instructions) | Simpler (hardware does more) |

### Concrete Example: String Copy

**Task**: Copy 100 bytes from address in R1 to address in R2.

#### CISC Approach (x86)

```asm
; Single instruction does entire operation
MOV ECX, 100          ; Set count to 100
REP MOVSB             ; Repeat: copy byte [ESI] to [EDI], increment both, decrement ECX
                      ; This is ONE instruction that copies 100 bytes!
```

**Microarchitecture**: REP MOVSB is actually implemented as a microcoded loop inside the CPU. It takes ~100 cycles but looks like one instruction.

#### RISC Approach (RISC-V)

```asm
; Explicit loop required
    li   t0, 0        ; Initialize offset to 0
    li   t1, 100      ; Initialize count to 100
loop:
    lb   t2, 0(a0)    ; Load byte from source (a0 + 0)
    sb   t2, 0(a1)    ; Store byte to dest (a1 + 0)
    addi a0, a0, 1    ; Increment source pointer
    addi a1, a1, 1    ; Increment dest pointer
    addi t0, t0, 1    ; Increment counter
    bne  t0, t1, loop ; Branch if counter != count
```

**6 instructions per iteration × 100 iterations = 600 instructions** (vs 2 for CISC)

**But**: With pipelining and out-of-order execution, RISC can approach 1 cycle per iteration (after pipeline fill), achieving similar performance with simpler hardware.

### Why Jolt Uses RISC-V

**Advantages for zkVMs**:

1. **Regular instruction encoding**: All instructions are 32-bit (or 16-bit compressed), simplifying decoding in circuits
2. **Load-store architecture**: Memory operations are explicit (LD/SD only), making memory tracking cleaner
3. **Large register file**: 32 registers reduce memory traffic (easier to prove)
4. **Simple semantics**: Each instruction does one simple thing, easier to specify in lookup tables
5. **Open standard**: No licensing restrictions, good toolchain support

**Example: Why x86 would be harder**:

x86 instruction: `ADD [RAX + RBX*4 + 0x10], RCX`

This single instruction:
1. Reads RBX, multiplies by 4
2. Adds RAX
3. Adds immediate 0x10
4. Loads from computed address
5. Reads RCX
6. Adds loaded value and RCX
7. Stores result back to memory

**In RISC-V, this becomes**:
```asm
slli t0, rbx, 2       # t0 = rbx * 4
add  t0, rax, t0      # t0 = rax + (rbx * 4)
addi t0, t0, 0x10     # t0 = rax + (rbx * 4) + 0x10
ld   t1, 0(t0)        # t1 = memory[t0]
add  t1, t1, rcx      # t1 = t1 + rcx
sd   t1, 0(t0)        # memory[t0] = t1
```

6 simple operations vs 1 complex operation. For proving:
- RISC-V: 6 lookup table queries (one per instruction)
- x86: Would need to decompose the complex instruction into sub-operations anyway

**Conclusion**: RISC simplicity aligns with lookup-centric proving.

---

**End of Part 0**

---

# Part 1: Compilation Pipeline Deep Dive

> **Purpose**: Understanding transpilation requires understanding compilation. This section provides rigorous treatment of how source code transforms through lexical analysis, parsing, semantic analysis, IR generation, optimization, and code generation. We focus on the mathematical properties that enable automated transformation.

## 1.1 The Compilation Phases

### The Classical Compiler Pipeline

```
Source Code (text file)
    ↓
┌────────────────────────────────────────────────┐
│ FRONTEND                                       │
├────────────────────────────────────────────────┤
│                                                │
│  Lexical Analysis (Scanner)                   │
│    ├─ Input: Character stream                 │
│    └─ Output: Token stream                    │
│         ↓                                      │
│  Syntactic Analysis (Parser)                  │
│    ├─ Input: Token stream                     │
│    └─ Output: Abstract Syntax Tree (AST)      │
│         ↓                                      │
│  Semantic Analysis                             │
│    ├─ Input: AST                              │
│    ├─ Operations: Type checking, scoping      │
│    └─ Output: Annotated AST                   │
│                                                │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│ MIDDLE-END                                     │
├────────────────────────────────────────────────┤
│                                                │
│  IR Generation                                 │
│    ├─ Input: Annotated AST                    │
│    └─ Output: Intermediate Representation      │
│         ↓                                      │
│  IR Optimization                               │
│    ├─ Constant folding                        │
│    ├─ Dead code elimination                   │
│    ├─ Common subexpression elimination        │
│    ├─ Loop optimization                       │
│    └─ Output: Optimized IR                    │
│                                                │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│ BACKEND                                        │
├────────────────────────────────────────────────┤
│                                                │
│  Instruction Selection                         │
│    ├─ Input: Optimized IR                     │
│    └─ Output: Target instructions (abstract)  │
│         ↓                                      │
│  Register Allocation                           │
│    ├─ Assign variables to registers           │
│    └─ Insert spill code if needed             │
│         ↓                                      │
│  Instruction Scheduling                        │
│    ├─ Reorder for pipeline efficiency         │
│    └─ Output: Assembly                         │
│         ↓                                      │
│  Assembler                                     │
│    └─ Output: Machine code                    │
│                                                │
└────────────────────────────────────────────────┘
    ↓
Machine Code (ELF binary)
```

### Key Insight: Phase Independence

**Critical property**: Each phase can be developed and reasoned about independently, as long as interfaces are well-defined.

**Implication for transpilation**: We can extract IR from one compiler and feed it to another, enabling automatic code transformation.

**Example interfaces**:
- Scanner $\to$ Parser: Token stream (defined by token grammar)
- Parser $\to$ Semantic analyzer: AST (defined by AST grammar)
- Semantic analyzer $\to$ IR generator: Typed AST
- IR generator $\to$ Optimizer: IR (defined by IR specification)

---

## 1.2 Lexical Analysis: From Text to Tokens

### What is Lexical Analysis?

**Definition**: Lexical analysis (scanning) converts a character stream into a token stream by recognizing lexical patterns defined by regular expressions.

**Mathematical model**:
- Input: String $s \in \Sigma^*$ where $\Sigma$ is the alphabet (ASCII or Unicode characters)
- Output: Sequence of tokens $[t_1, t_2, \ldots, t_n]$ where each $t_i = (\text{type}, \text{value}, \text{position})$

**Token**: A tuple $(\text{TokenType}, \text{Lexeme}, \text{SourceLocation})$

### Token Types

**Examples from Rust-like language**:

```
TokenType:
├─ KEYWORD: fn, let, if, else, while, return, u32, i64
├─ IDENTIFIER: variable names, function names
├─ LITERAL:
│  ├─ INTEGER: 42, 0xFF, 0b1010
│  ├─ FLOAT: 3.14, 1.0e-5
│  ├─ STRING: "hello"
│  └─ CHAR: 'a'
├─ OPERATOR: +, -, *, /, ==, !=, <, >, &&, ||
├─ DELIMITER: (, ), {, }, [, ], ;, ,, :
└─ COMMENT: // single-line, /* multi-line */
```

### Concrete Example

**Input source code**:
```rust
fn add(x: u32, y: u32) -> u32 {
    x + y
}
```

**Character stream**:
```
['f', 'n', ' ', 'a', 'd', 'd', '(', 'x', ':', ' ', 'u', '3', '2', ',', ' ', ...]
```

**Token stream**:
```
Token 1: (KEYWORD, "fn", Line 1, Col 1)
Token 2: (IDENTIFIER, "add", Line 1, Col 4)
Token 3: (DELIMITER, "(", Line 1, Col 7)
Token 4: (IDENTIFIER, "x", Line 1, Col 8)
Token 5: (DELIMITER, ":", Line 1, Col 9)
Token 6: (KEYWORD, "u32", Line 1, Col 11)
Token 7: (DELIMITER, ",", Line 1, Col 14)
Token 8: (IDENTIFIER, "y", Line 1, Col 16)
Token 9: (DELIMITER, ":", Line 1, Col 17)
Token 10: (KEYWORD, "u32", Line 1, Col 19)
Token 11: (DELIMITER, ")", Line 1, Col 22)
Token 12: (OPERATOR, "->", Line 1, Col 24)
Token 13: (KEYWORD, "u32", Line 1, Col 27)
Token 14: (DELIMITER, "{", Line 1, Col 31)
Token 15: (IDENTIFIER, "x", Line 2, Col 5)
Token 16: (OPERATOR, "+", Line 2, Col 7)
Token 17: (IDENTIFIER, "y", Line 2, Col 9)
Token 18: (DELIMITER, "}", Line 3, Col 1)
Token 19: (EOF, "", Line 3, Col 2)
```

### Regular Expressions for Token Recognition

**Formal definition**: Token patterns defined by regular languages.

**Examples**:

| Token Type | Regular Expression | Examples |
|------------|-------------------|----------|
| IDENTIFIER | `[a-zA-Z_][a-zA-Z0-9_]*` | `x`, `add`, `my_var` |
| INTEGER | `[0-9]+` or `0x[0-9a-fA-F]+` | `42`, `0xFF` |
| FLOAT | `[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?` | `3.14`, `1e-5` |
| KEYWORD | Literal matches | `fn`, `let`, `if` |
| WHITESPACE | `[ \t\n\r]+` | Ignored |
| COMMENT | `//.*` or `/\*(.|\n)*\*/` | Ignored |

### Finite Automata Implementation

**Key insight**: Regular expressions compiled to Deterministic Finite Automaton (DFA) for efficient scanning.

**Example: Recognizing identifiers**

**Regular expression**: `[a-zA-Z_][a-zA-Z0-9_]*`

**DFA**:
```
States: {START, IDENT, ERROR}
Alphabet: {a-z, A-Z, _, 0-9, other}

START:
  [a-zA-Z_] $\to$ IDENT
  [0-9] $\to$ ERROR
  other $\to$ ERROR

IDENT (accepting state):
  [a-zA-Z0-9_] $\to$ IDENT
  other $\to$ END (emit IDENTIFIER token)

ERROR:
  * $\to$ ERROR
```

**Execution trace** for input `add`:
```
State      Input    Action
START      'a'      Transition to IDENT
IDENT      'd'      Stay in IDENT
IDENT      'd'      Stay in IDENT
IDENT      '('      End of identifier, emit token (IDENTIFIER, "add")
```

### Maximal Munch Principle

**Rule**: Scanner always produces the longest possible token.

**Example**:
```
Input: "return42"

Without maximal munch:
  Could tokenize as: (KEYWORD, "return"), (INTEGER, "42")
  Or: (IDENTIFIER, "return42")

With maximal munch:
  Always choose: (IDENTIFIER, "return42")
```

**Why this matters**: Prevents ambiguity. The scanner always prefers longer matches.

### Handling Keywords vs Identifiers

**Problem**: Keywords (like `fn`, `if`) match the identifier pattern `[a-zA-Z_][a-zA-Z0-9_]*`.

**Solution 1: Reserved word table**
```
Algorithm:
1. Scan identifier using identifier DFA
2. Look up lexeme in reserved word table
3. If found, emit KEYWORD token; otherwise emit IDENTIFIER token

Example:
  Input: "fn" $\to$ Scan as identifier $\to$ Lookup "fn" $\to$ Found $\to$ Emit (KEYWORD, "fn")
  Input: "foo" $\to$ Scan as identifier $\to$ Lookup "foo" $\to$ Not found $\to$ Emit (IDENTIFIER, "foo")
```

**Solution 2: Separate DFA for each keyword**
```
Build DFA with:
- Accepting states for "fn", "let", "if", etc.
- Lower priority than identifier pattern

Priority ordering ensures keywords matched before identifiers.
```

### Error Handling

**When lexical errors occur**:
```
Input: "let x = @42;"
                ^
                Unexpected character '@'
```

**Error recovery strategies**:
1. **Panic mode**: Skip characters until next valid token start (e.g., whitespace, semicolon)
2. **Error token**: Emit special ERROR token, let parser handle it
3. **Suggestion**: "Did you mean '=' instead of '@'?"

---

## 1.3 Syntactic Analysis: Building the AST

### What is Parsing?

**Definition**: Parsing (syntactic analysis) converts a token stream into an Abstract Syntax Tree by recognizing syntactic patterns defined by context-free grammars.

**Mathematical model**:
- Input: Token sequence $[t_1, t_2, \ldots, t_n]$
- Grammar: Context-free grammar $G = (N, T, P, S)$
  - $N$: Non-terminals (syntactic categories)
  - $T$: Terminals (tokens)
  - $P$: Production rules
  - $S$: Start symbol
- Output: Parse tree or Abstract Syntax Tree (AST)

### Context-Free Grammar (CFG)

**Formal definition**: A grammar $G = (N, T, P, S)$ where:
- $N$: Finite set of non-terminal symbols
- $T$: Finite set of terminal symbols (tokens)
- $P$: Finite set of production rules $A \to \alpha$ where $A \in N$ and $\alpha \in (N \cup T)^*$
- $S \in N$: Start symbol

**Example grammar for arithmetic expressions**:

```
Non-terminals: {Expr, Term, Factor}
Terminals: {INTEGER, +, -, *, /, (, )}
Start symbol: Expr

Productions:
  Expr   $\to$ Expr + Term       (E1)
         | Expr - Term       (E2)
         | Term              (E3)

  Term   $\to$ Term * Factor     (T1)
         | Term / Factor     (T2)
         | Factor            (T3)

  Factor $\to$ ( Expr )          (F1)
         | INTEGER           (F2)
```

**Derivation** for `2 + 3 * 4`:

```
Expr
  $\Rightarrow$ Expr + Term                [Apply E1]
  $\Rightarrow$ Term + Term                [Apply E3 on left Expr]
  $\Rightarrow$ Factor + Term              [Apply T3]
  $\Rightarrow$ INTEGER + Term             [Apply F2]
  $\Rightarrow$ 2 + Term                   [Substitute token]
  $\Rightarrow$ 2 + Term * Factor          [Apply T1]
  $\Rightarrow$ 2 + Factor * Factor        [Apply T3]
  $\Rightarrow$ 2 + INTEGER * Factor       [Apply F2]
  $\Rightarrow$ 2 + 3 * Factor             [Substitute token]
  $\Rightarrow$ 2 + 3 * INTEGER            [Apply F2]
  $\Rightarrow$ 2 + 3 * 4                  [Substitute token]
```

### Parse Tree vs Abstract Syntax Tree

**Parse tree**: Represents complete derivation, includes all grammar symbols.

**Abstract Syntax Tree (AST)**: Condenses parse tree, removing unnecessary nodes.

**Example: Parse tree for `2 + 3`**:

```
           Expr
            |
      ┌─────┼─────┐
      Expr  +   Term
      |         |
     Term      Factor
      |         |
    Factor    INTEGER
      |         |
   INTEGER      3
      |
      2
```

**Abstract Syntax Tree for `2 + 3`**:

```
     Add
    /   \
   2     3
```

**Why AST is better**:
- Smaller: No redundant nodes
- Easier to manipulate: Direct representation of program structure
- Language-independent: Same AST structure can represent expressions in different languages

### Recursive Descent Parsing

**Idea**: One recursive function per non-terminal, functions call each other to match production rules.

**Example: Parser for arithmetic expressions**

**Grammar (simplified)**:
```
Expr   $\to$ Term (('+' | '-') Term)*
Term   $\to$ Factor (('*' | '/') Factor)*
Factor $\to$ INTEGER | '(' Expr ')'
```

**Recursive descent implementation** (Rust-like pseudocode):

```rust
struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn parse_expr(&mut self) -> ASTNode {
        let mut left = self.parse_term();

        while self.match_tokens(&[TokenType::PLUS, TokenType::MINUS]) {
            let op = self.previous();
            let right = self.parse_term();
            left = ASTNode::BinaryOp { op, left: Box::new(left), right: Box::new(right) };
        }

        left
    }

    fn parse_term(&mut self) -> ASTNode {
        let mut left = self.parse_factor();

        while self.match_tokens(&[TokenType::STAR, TokenType::SLASH]) {
            let op = self.previous();
            let right = self.parse_factor();
            left = ASTNode::BinaryOp { op, left: Box::new(left), right: Box::new(right) };
        }

        left
    }

    fn parse_factor(&mut self) -> ASTNode {
        if self.match_tokens(&[TokenType::INTEGER]) {
            return ASTNode::Literal(self.previous().value);
        }

        if self.match_tokens(&[TokenType::LPAREN]) {
            let expr = self.parse_expr();
            self.expect(TokenType::RPAREN, "Expected ')' after expression");
            return expr;
        }

        panic!("Expected expression at token {}", self.current());
    }

    fn match_tokens(&mut self, types: &[TokenType]) -> bool {
        for t in types {
            if self.current().token_type == *t {
                self.advance();
                return true;
            }
        }
        false
    }

    fn current(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn advance(&mut self) {
        self.current += 1;
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn expect(&mut self, token_type: TokenType, message: &str) {
        if self.current().token_type != token_type {
            panic!("{}", message);
        }
        self.advance();
    }
}
```

**Execution trace** for `2 + 3 * 4`:

```
Call stack                          Token stream
─────────────────────────────────────────────────
parse_expr()                        [2, +, 3, *, 4]
  parse_term()                      [2, +, 3, *, 4]
    parse_factor()                  [2, +, 3, *, 4]
      Match INTEGER                 [2, +, 3, *, 4]
      Return Literal(2)                 ^
    Return Literal(2)
  Check for * or /                  [2, +, 3, *, 4]
  No match, return Literal(2)              ^
Check for + or -                    [2, +, 3, *, 4]
Match PLUS                                 ^
parse_term()                        [2, +, 3, *, 4]
  parse_factor()                               ^
    Match INTEGER                   [2, +, 3, *, 4]
    Return Literal(3)                     ^
  Check for * or /                  [2, +, 3, *, 4]
  Match STAR                                    ^
  parse_factor()                    [2, +, 3, *, 4]
    Match INTEGER                            ^
    Return Literal(4)
  Return BinaryOp(*, Literal(3), Literal(4))
Return BinaryOp(+, Literal(2), BinaryOp(*, Literal(3), Literal(4)))
```

**Resulting AST**:
```
        Add
       /   \
      2     Mul
           /   \
          3     4
```

### Operator Precedence

**Problem**: Without precedence rules, `2 + 3 * 4` could be `(2 + 3) * 4 = 20` or `2 + (3 * 4) = 14`.

**Solution**: Grammar structure encodes precedence.

**Precedence levels** (lowest to highest):
```
1. Addition, Subtraction       (+, -)
2. Multiplication, Division    (*, /)
3. Parentheses                 ( )
```

**Grammar reflects precedence**:
```
Expr   $\to$ Term (('+' | '-') Term)*      # Lowest precedence (parsed first)
Term   $\to$ Factor (('*' | '/') Factor)*  # Higher precedence (parsed deeper)
Factor $\to$ '(' Expr ')' | INTEGER        # Highest precedence (parsed deepest)
```

**Key insight**: The deeper in the grammar, the higher the precedence. Multiplication binds tighter because it's parsed in `parse_term()` which is called from `parse_expr()`.

### Left Recursion and LL(1) Parsing

**Problem**: Left-recursive grammars cause infinite recursion in recursive descent.

**Left-recursive grammar**:
```
Expr $\to$ Expr + Term    # Left recursion: Expr on left side of production
     | Term
```

**Why it fails**:
```
parse_expr():
  To parse Expr, first parse Expr  (recursive call)
    To parse Expr, first parse Expr  (recursive call)
      ... infinite recursion!
```

**Solution: Left factoring**

**Original (left-recursive)**:
```
Expr $\to$ Expr + Term
     | Term
```

**Transformed (right-recursive with iteration)**:
```
Expr $\to$ Term (('+' | '-') Term)*
```

**Implementation uses loop instead of recursion for repetition** (as shown in `parse_expr()` above).

---

## 1.4 Semantic Analysis: Type Checking and Scoping

### What is Semantic Analysis?

**Definition**: Semantic analysis verifies that the program follows language-specific rules beyond syntax.

**Key tasks**:
1. **Type checking**: Ensure operations apply to compatible types
2. **Scope resolution**: Ensure variables are declared before use
3. **Name resolution**: Bind identifiers to declarations
4. **Semantic error detection**: Catch errors like duplicate declarations

**Mathematical model**:
- Input: AST $T$
- Symbol table: $\Gamma : \text{Identifier} \to \text{Type}$
- Type rules: $\Gamma \vdash e : \tau$ (expression $e$ has type $\tau$ in context $\Gamma$)
- Output: Typed AST $T'$ with annotations

### Type Systems

**Purpose**: Prevent undefined behavior by rejecting ill-typed programs.

**Type**: A set of values and operations on those values.

**Examples**:
- `u32`: {0, 1, 2, ..., 2^32-1}, operations: {+, -, *, /, &, |, ^, <<, >>}
- `bool`: {true, false}, operations: {&&, ||, !}
- `u32 -> u32`: Functions from u32 to u32

### Type Checking Rules (Formal)

**Notation**: $\Gamma \vdash e : \tau$ means "In context $\Gamma$, expression $e$ has type $\tau$"

**Type rules for arithmetic expressions**:

**Integer literal**:
$$\frac{}{\Gamma \vdash n : \text{u32}} \quad \text{(T-INT)}$$

**Variable**:
$$\frac{x : \tau \in \Gamma}{\Gamma \vdash x : \tau} \quad \text{(T-VAR)}$$

**Addition**:
$$\frac{\Gamma \vdash e_1 : \text{u32} \quad \Gamma \vdash e_2 : \text{u32}}{\Gamma \vdash e_1 + e_2 : \text{u32}} \quad \text{(T-ADD)}$$

**Function application**:
$$\frac{\Gamma \vdash f : \tau_1 \to \tau_2 \quad \Gamma \vdash e : \tau_1}{\Gamma \vdash f(e) : \tau_2} \quad \text{(T-APP)}$$

### Worked Example: Type Checking

**Program**:
```rust
fn add(x: u32, y: u32) -> u32 {
    x + y
}

fn main() {
    let result = add(10, 20);
}
```

**Type checking derivation**:

**Step 1: Build initial context $\Gamma_0$**:
```
$\Gamma$₀ = {add : (u32, u32) -> u32}
```

**Step 2: Check body of `add`**:
```
$\Gamma$_add = $\Gamma$₀ ∪ {x : u32, y : u32}

Derivation for `x + y`:
  $\Gamma$_add ⊢ x : u32       (T-VAR, x : u32 ∈ $\Gamma$_add)
  $\Gamma$_add ⊢ y : u32       (T-VAR, y : u32 ∈ $\Gamma$_add)
  ─────────────────────────────────────────────  (T-ADD)
  $\Gamma$_add ⊢ x + y : u32

Return type matches declared return type u32 ✓
```

**Step 3: Check body of `main`**:
```
$\Gamma$_main = $\Gamma$₀

Derivation for `add(10, 20)`:
  $\Gamma$_main ⊢ add : (u32, u32) -> u32    (T-VAR, add ∈ $\Gamma$₀)
  $\Gamma$_main ⊢ 10 : u32                   (T-INT)
  $\Gamma$_main ⊢ 20 : u32                   (T-INT)
  ──────────────────────────────────────────────  (T-APP twice)
  $\Gamma$_main ⊢ add(10, 20) : u32

Binding result:
  $\Gamma$_main' = $\Gamma$_main ∪ {result : u32}
```

**Type checking succeeds ✓**

### Example: Type Error

**Program**:
```rust
fn add(x: u32, y: u32) -> u32 {
    x + y
}

fn main() {
    let result = add(10, true);  // Error: second argument has wrong type
}
```

**Type checking derivation**:
```
$\Gamma$_main ⊢ add : (u32, u32) -> u32
$\Gamma$_main ⊢ 10 : u32
$\Gamma$_main ⊢ true : bool          $\leftarrow$ Type mismatch!

Expected: u32
Found:    bool
Error: Cannot apply function of type (u32, u32) -> u32 to arguments (u32, bool)
```

### Scoping Rules

**Scope**: The region of code where a variable binding is valid.

**Lexical scoping** (most common): Variable bindings determined by textual structure.

**Example**:
```rust
fn outer() {
    let x = 10;        // Scope: outer function body

    {
        let y = 20;    // Scope: inner block
        println!("{} {}", x, y);  // x and y both in scope
    }

    println!("{}", x);   // Only x in scope
    println!("{}", y);   // Error: y not in scope
}
```

**Symbol table implementation**: Stack of scopes

```
Structure:
  Scope stack: [global, function_outer, block_inner]

Operations:
  - enter_scope(): Push new scope onto stack
  - exit_scope(): Pop scope from stack
  - declare(name, type): Add binding to current scope
  - lookup(name): Search stack from top to bottom
```

**Execution trace for example above**:

```
Action                   Scope stack            Symbol table
─────────────────────────────────────────────────────────────
enter_scope(outer)       [global, outer]        {}
declare(x, u32)          [global, outer]        {x: u32}
enter_scope(block)       [global, outer, block] {x: u32}
declare(y, u32)          [global, outer, block] {x: u32, y: u32}
lookup(x)                [global, outer, block] Found in outer
lookup(y)                [global, outer, block] Found in block
exit_scope()             [global, outer]        {x: u32}  (y removed)
lookup(x)                [global, outer]        Found in outer
lookup(y)                [global, outer]        Not found $\to$ Error!
exit_scope()             [global]               {}
```

### Name Resolution

**Problem**: Same identifier may refer to different entities in different contexts.

**Example**:
```rust
let x = 10;        // Global x

fn foo(x: u32) {   // Parameter x (shadows global x)
    let x = x + 1; // Local x (shadows parameter x)
    println!("{}", x);  // Which x?
}
```

**Resolution**: Each use of `x` is annotated with its declaration site.

**Annotated AST**:
```
GlobalDecl(x, 10)                         // x₀ = 10

FunctionDecl(foo, [Param(x, u32)],        // x₁: u32
  Block([
    LetDecl(x, Add(Var(x), 1))            // x₂ = x₁ + 1
    FunctionCall(println, [Var(x)])       // Refers to x₂
  ])
)
```

**Each variable use is uniquely resolved to its declaration**.

---

## 1.5 Intermediate Representations

### Why IR?

**Problem**: Compiling N languages to M architectures requires N×M compilers.

**Solution**: Use intermediate representation (IR) as common format.

```
N source languages $\to$ IR $\to$ M target architectures
Requires: N frontends + M backends = N + M components (not N×M)
```

**Example: LLVM ecosystem**
```
Source languages:        IR:       Target architectures:
  Rust     ─┐           LLVM IR   ┌─ x86-64
  C/C++    ─┼──────$\to$   (SSA-based) ──┼─ ARM
  Swift    ─┘                      └─ RISC-V
```

### Desirable IR Properties

**1. Language-independent**: Not tied to any source language syntax
**2. Target-independent**: Not tied to any machine architecture
**3. Compact**: Easy to manipulate and analyze
**4. Easy to generate**: From AST
**5. Easy to optimize**: Support standard optimizations
**6. Easy to translate**: To target code

### Three-Address Code (TAC)

**Definition**: IR where each instruction has at most three operands (two sources, one destination).

**General form**:
```
x = y op z    (binary operation)
x = op y      (unary operation)
x = y         (copy)
goto L        (unconditional jump)
if x goto L   (conditional jump)
```

**Example: Source code**:
```rust
let a = 2 + 3 * 4;
```

**Three-address code**:
```
t1 = 3 * 4
t2 = 2 + t1
a = t2
```

**Characteristics**:
- Each operation produces one result
- Intermediate values stored in temporaries ($t_1, t_2, \ldots$)
- Close to assembly but architecture-independent

### Static Single Assignment (SSA) Form

**Definition**: IR where each variable is assigned exactly once.

**Key idea**: When a variable is reassigned, create a new variable.

**Example: Non-SSA**:
```
x = 10
y = x + 5
x = 20        // Reassignment
z = x + y
```

**SSA form**:
```
x₁ = 10
y₁ = x₁ + 5
x₂ = 20       // New variable x₂
z₁ = x₂ + y₁
```

**Why SSA?**
- Simplifies data flow analysis
- Enables powerful optimizations (constant propagation, dead code elimination)
- Each use of a variable has unique definition (def-use chain is trivial)

**Challenge: Control flow merges**

**Problem**: What value does variable have when multiple paths merge?

**Example**:
```rust
let x;
if condition {
    x = 10;
} else {
    x = 20;
}
let y = x + 5;  // Which x? 10 or 20?
```

**Solution: $\Phi$ (phi) functions**

**SSA with phi function**:
```
       if condition
      /            \
    x₁ = 10      x₂ = 20
      \            /
       x₃ = $\phi$(x₁, x₂)    // x₃ is x₁ if came from left, x₂ if came from right
       y₁ = x₃ + 5
```

**Formal definition of $\phi$ function**:
$$x_3 = \phi(x_1, x_2) = \begin{cases} x_1 & \text{if control flow came from left branch} \\ x_2 & \text{if control flow came from right branch} \end{cases}$$

**Phi functions are not real instructions**—they're analysis constructs. Code generation eliminates them by inserting copies on each incoming edge.

### Control Flow Graph (CFG)

**Definition**: Directed graph $G = (V, E)$ where:
- $V$: Set of **basic blocks** (maximal sequences of straight-line code)
- $E$: Set of **control flow edges** (possible execution paths)

**Basic block**: Sequence of instructions with:
- One entry point (first instruction)
- One exit point (last instruction)
- No internal branches

**Example: Source code**:
```rust
fn abs_diff(x: i32, y: i32) -> i32 {
    if x > y {
        x - y
    } else {
        y - x
    }
}
```

**Control Flow Graph**:
```
          ┌──────────────────┐
          │ Entry            │
          │ t1 = x > y       │
          └────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         v (true)            v (false)
  ┌──────────┐        ┌──────────┐
  │ Block 2  │        │ Block 3  │
  │ t2 = x-y │        │ t3 = y-x │
  │ return t2│        │ return t3│
  └────┬─────┘        └────┬─────┘
       │                   │
       └─────────┬─────────┘
                 v
          ┌──────────────┐
          │ Exit         │
          └──────────────┘
```

**Each basic block**:
```
Block 1 (Entry):
  Instructions: [t1 = x > y]
  Successors: [Block 2, Block 3]

Block 2:
  Instructions: [t2 = x - y, return t2]
  Successors: [Exit]

Block 3:
  Instructions: [t3 = y - x, return t3]
  Successors: [Exit]
```

### LLVM IR Example

LLVM uses SSA-based IR with typed instructions.

**Source (C-like)**:
```c
int add(int x, int y) {
    return x + y;
}
```

**LLVM IR**:
```llvm
define i32 @add(i32 %x, i32 %y) {
entry:
  %result = add i32 %x, %y
  ret i32 %result
}
```

**Characteristics**:
- `i32`: 32-bit integer type
- `%x, %y`: Virtual registers (infinite supply)
- `@add`: Global function name
- `entry`: Basic block label
- SSA form: Each register assigned once

**More complex example with control flow**:
```c
int abs_diff(int x, int y) {
    int result;
    if (x > y) {
        result = x - y;
    } else {
        result = y - x;
    }
    return result;
}
```

**LLVM IR**:
```llvm
define i32 @abs_diff(i32 %x, i32 %y) {
entry:
  %cmp = icmp sgt i32 %x, %y    ; %cmp = (x > y)
  br i1 %cmp, label %then, label %else

then:
  %diff1 = sub i32 %x, %y       ; diff1 = x - y
  br label %merge

else:
  %diff2 = sub i32 %y, %x       ; diff2 = y - x
  br label %merge

merge:
  %result = phi i32 [%diff1, %then], [%diff2, %else]  ; result = $\phi$(diff1, diff2)
  ret i32 %result
}
```

**Note the phi function at merge point**: Selects `%diff1` if came from `%then`, `%diff2` if came from `%else`.

---

## 1.6 Optimization Passes

### What is Optimization?

**Definition**: Transformation that preserves program semantics while improving some metric (speed, code size, power consumption).

**Key insight**: IR enables optimizations independent of source language and target architecture.

**Optimization categories**:
1. **Local**: Within a single basic block
2. **Global**: Across basic blocks within a function
3. **Interprocedural**: Across function boundaries

### Constant Folding

**Idea**: Evaluate constant expressions at compile time.

**Example**:
```
Before:
  x = 2 + 3
  y = x * 4

After:
  x = 5
  y = 20
```

**Why this works**: Compiler can evaluate arithmetic on constants, no need to emit instructions.

**Implementation**:
```rust
fn constant_fold(expr: &mut Expr) {
    match expr {
        Expr::BinaryOp { op: Op::Add, left, right } => {
            constant_fold(left);
            constant_fold(right);
            if let (Expr::Literal(a), Expr::Literal(b)) = (&**left, &**right) {
                *expr = Expr::Literal(a + b);  // Replace with constant
            }
        }
        // Similar for other operators
        _ => {}
    }
}
```

### Constant Propagation

**Idea**: Replace variable uses with their known constant values.

**Example**:
```
Before:
  x = 5
  y = x + 3
  z = x * 2

After:
  x = 5
  y = 5 + 3    // Propagated x = 5
  z = 5 * 2    // Propagated x = 5

After constant folding:
  x = 5
  y = 8
  z = 10
```

**Why this enables more optimization**: After propagation, we can constant fold.

### Dead Code Elimination (DCE)

**Idea**: Remove computations whose results are never used.

**Example**:
```
Before:
  x = 10
  y = x + 5    // y is never used!
  z = 20
  return z

After:
  x = 10       // x might be removed if also unused
  z = 20
  return z
```

**Why this works**: If a variable is never read, its definition is useless.

**Implementation** (requires liveness analysis):
```rust
fn dead_code_elimination(cfg: &mut CFG) {
    let live_vars = liveness_analysis(cfg);

    for block in cfg.blocks.iter_mut() {
        block.instructions.retain(|instr| {
            match instr {
                Instr::Assign { target, .. } => {
                    // Keep instruction if target is live after this point
                    live_vars.get(instr.id()).contains(target)
                }
                _ => true  // Keep all other instructions
            }
        });
    }
}
```

### Common Subexpression Elimination (CSE)

**Idea**: If an expression is computed multiple times with the same operands, compute once and reuse.

**Example**:
```
Before:
  a = x + y
  b = x + y    // Same expression as a
  c = a + b

After:
  a = x + y
  b = a        // Reuse a instead of recomputing
  c = a + b
```

**Why this works**: If `x` and `y` haven't changed, `x + y` produces same result.

**Implementation** (requires available expressions analysis):
```rust
fn cse(block: &mut BasicBlock) {
    let mut available: HashMap<Expr, Variable> = HashMap::new();

    for instr in block.instructions.iter_mut() {
        match instr {
            Instr::Assign { target, expr } => {
                // Check if expression already computed
                if let Some(&var) = available.get(expr) {
                    // Replace with copy
                    *instr = Instr::Copy { target: *target, source: var };
                } else {
                    // Record this expression as available
                    available.insert(expr.clone(), *target);
                }
            }
            // Handle other instructions (invalidate available expressions if needed)
            _ => {}
        }
    }
}
```

### Loop Optimizations

**Loop-invariant code motion**: Move computations that don't change inside loop to before loop.

**Example**:
```
Before:
  for i in 0..n {
      x = y * 2    // y * 2 doesn't depend on i
      a[i] = x + i
  }

After:
  x = y * 2        // Moved outside loop
  for i in 0..n {
      a[i] = x + i
  }
```

**Why this works**: Avoid recomputing `y * 2` on every iteration.

---

## 1.7 Code Generation

### What is Code Generation?

**Definition**: Translate IR to target machine code or assembly.

**Key tasks**:
1. **Instruction selection**: Map IR operations to target instructions
2. **Register allocation**: Assign variables to physical registers
3. **Instruction scheduling**: Reorder instructions for pipeline efficiency

### Instruction Selection

**Problem**: Map high-level IR operations to sequences of target instructions.

**Example: IR to RISC-V**:

**IR**:
```
t1 = a + b
t2 = t1 * 4
c = t2
```

**RISC-V assembly**:
```
add  t0, a0, a1      # t0 = a + b  (a in a0, b in a1)
slli t1, t0, 2       # t1 = t0 << 2 (multiply by 4 via shift left by 2)
mv   a2, t1          # c = t1 (result in a2)
```

**Challenge: Multiple options**

Multiply by 4 can be done as:
- `slli t1, t0, 2` (shift left by 2, 1 instruction)
- `mul t1, t0, 4` (multiply with immediate if supported, 1-2 instructions)
- `add t1, t0, t0; add t1, t1, t1` (two adds, 2 instructions)

**Heuristic**: Choose fastest/smallest code sequence for target architecture.

### Register Allocation

**Problem**: IR has infinite virtual registers, hardware has finite physical registers (32 for RISC-V).

**Task**: Assign virtual registers to physical registers, spill to memory when necessary.

**Example: Before register allocation**:
```
v1 = 10
v2 = 20
v3 = v1 + v2
v4 = v3 * 2
return v4
```

**After register allocation** (assuming 3 physical registers: t0, t1, t2):
```
li   t0, 10          # v1 $\to$ t0
li   t1, 20          # v2 $\to$ t1
add  t2, t0, t1      # v3 $\to$ t2
slli t2, t2, 1       # v4 $\to$ t2 (reuse t2, v3 dead)
mv   a0, t2          # return value in a0
ret
```

**Graph coloring algorithm**:

**Step 1: Build interference graph**

Nodes: Virtual registers
Edges: Two registers interfere if they're live at the same time

**Example**:
```
v1 = ...
v2 = ...
v3 = v1 + v2    # v1, v2, v3 all live
v4 = v3 * 2     # v3, v4 live
return v4       # v4 live
```

**Interference graph**:
```
v1 ─── v2       v1 and v2 live simultaneously
│      │
└──v3──┘        v1, v2, v3 form triangle
   │
   v4           v3 and v4 interfere
```

**Step 2: Color graph**

Assign colors (physical registers) such that adjacent nodes have different colors.

```
v1 $\to$ t0  (color 0)
v2 $\to$ t1  (color 1)
v3 $\to$ t2  (color 2)
v4 $\to$ t2  (color 2, v3 dead when v4 assigned, so can reuse)
```

**If not enough colors**: Spill least-frequently-used variables to memory.

**Spilling example**:
```
If only 2 registers available (t0, t1):
v1 $\to$ t0
v2 $\to$ t1
v3 $\to$ spill to stack (store v3 to memory)
Later: Load v3 from stack when needed
```

### Instruction Scheduling

**Problem**: Modern CPUs use pipelining, want to avoid pipeline stalls.

**Example: Pipeline hazard**:

```
Original:
  ld   t0, 0(a0)     # Load from memory (3-cycle latency)
  add  t1, t0, t2    # Uses t0 immediately $\to$ stall!
```

**Optimized**:
```
  ld   t0, 0(a0)     # Load from memory
  sub  t3, t4, t5    # Independent instruction (fill delay slot)
  add  t1, t0, t2    # By now, t0 is ready
```

**Key insight**: Reorder instructions to fill pipeline bubbles, without changing semantics.

**Constraint**: Cannot reorder if data dependency exists.

---

**End of Part 1**

---

# Part 2: Abstract Syntax Trees and Intermediate Representations

> **Purpose**: Deep dive into the data structures that enable program transformation. Understanding ASTs and IRs is critical for extraction—we need to capture program semantics in a form amenable to automated translation.

## 2.1 Abstract Syntax Trees: Structure and Properties

### AST as a Mathematical Object

**Definition**: An Abstract Syntax Tree (AST) is a rooted, labeled tree $T = (V, E, \lambda, r)$ where:
- $V$: Finite set of nodes (vertices)
- $E \subseteq V \times V$: Set of directed edges (parent-child relationships)
- $\lambda: V \to \mathcal{L}$: Labeling function mapping nodes to syntactic constructs
- $r \in V$: Root node (entry point of the program structure)

**Properties**:
1. **Rooted tree**: Exactly one path from root to any node
2. **Ordered children**: For each node, children have a fixed order (e.g., left operand before right operand)
3. **Labeled**: Each node represents a syntactic construct (operator, literal, identifier, etc.)
4. **No redundant information**: Unlike parse trees, only semantically relevant structure preserved

### AST Node Types

**For expression-oriented language** (Rust-like):

```
ASTNode =
  | Literal(value: i64)
  | Identifier(name: String)
  | BinaryOp(op: BinOp, left: Box<ASTNode>, right: Box<ASTNode>)
  | UnaryOp(op: UnOp, operand: Box<ASTNode>)
  | FunctionCall(func: String, args: Vec<ASTNode>)
  | Let(name: String, type: Type, init: Box<ASTNode>)
  | If(condition: Box<ASTNode>, then_branch: Box<ASTNode>, else_branch: Option<Box<ASTNode>>)
  | While(condition: Box<ASTNode>, body: Box<ASTNode>)
  | Block(statements: Vec<ASTNode>)
  | Return(value: Option<Box<ASTNode>>)
  | Assignment(target: String, value: Box<ASTNode>)

BinOp = Add | Sub | Mul | Div | Eq | Ne | Lt | Gt | And | Or
UnOp = Neg | Not
```

### Small Concrete Example

**Source code**:
```rust
let x = 2 + 3;
let y = x * 4;
```

**AST (algebraic representation)**:
```
Block([
  Let("x", u32, BinaryOp(Add, Literal(2), Literal(3))),
  Let("y", u32, BinaryOp(Mul, Identifier("x"), Literal(4)))
])
```

**AST (tree visualization)**:
```
                Block
                  |
         ┌────────┴────────┐
         |                 |
        Let               Let
     ┌───┼───┐         ┌───┼───┐
    "x"  u32 Add       "y"  u32 Mul
            / \               / \
           2   3             x   4
```

### AST for Control Flow

**Source code**:
```rust
if x > 0 {
    y = x * 2;
} else {
    y = 0;
}
```

**AST**:
```
If(
  condition: BinaryOp(Gt, Identifier("x"), Literal(0)),
  then_branch: Block([
    Assignment("y", BinaryOp(Mul, Identifier("x"), Literal(2)))
  ]),
  else_branch: Some(Block([
    Assignment("y", Literal(0))
  ]))
)
```

**Tree visualization**:
```
                    If
         ┌───────────┼───────────┐
         |           |           |
        Gt        Block       Block
       /  \          |           |
      x    0    Assignment   Assignment
                /    \        /    \
               y    Mul      y      0
                   /  \
                  x    2
```

### Traversal Algorithms

**Why traversal matters**: To extract or transform an AST, we must visit nodes systematically.

#### Pre-order Traversal

**Definition**: Visit node, then recursively visit children left-to-right.

**Algorithm**:
```rust
fn preorder<F>(node: &ASTNode, visit: &mut F)
where F: FnMut(&ASTNode)
{
    visit(node);  // Visit current node first

    match node {
        ASTNode::BinaryOp { left, right, .. } => {
            preorder(left, visit);
            preorder(right, visit);
        }
        ASTNode::Block(statements) => {
            for stmt in statements {
                preorder(stmt, visit);
            }
        }
        // ... other node types
        _ => {}
    }
}
```

**Example**: For AST of `2 + 3 * 4`
```
        Add
       /   \
      2    Mul
          /   \
         3     4

Pre-order: Add, 2, Mul, 3, 4
```

**Use case**: Code generation (emit operator before operands for prefix notation)

#### Post-order Traversal

**Definition**: Recursively visit children first, then visit node.

**Algorithm**:
```rust
fn postorder<F>(node: &ASTNode, visit: &mut F)
where F: FnMut(&ASTNode)
{
    match node {
        ASTNode::BinaryOp { left, right, .. } => {
            postorder(left, visit);
            postorder(right, visit);
        }
        ASTNode::Block(statements) => {
            for stmt in statements {
                postorder(stmt, visit);
            }
        }
        // ... other node types
        _ => {}
    }

    visit(node);  // Visit current node last
}
```

**Example**: For AST of `2 + 3 * 4`
```
Post-order: 2, 3, 4, Mul, Add
```

**Use case**: Evaluation (compute operands before applying operator), stack-based code generation

#### In-order Traversal

**Definition**: Visit left child, then node, then right child.

**Only meaningful for binary trees**.

**Example**: For AST of `2 + 3 * 4`
```
In-order: 2, Add, 3, Mul, 4  (produces infix expression)
```

### AST Transformations

**Key insight**: Optimization and transpilation are AST transformations.

#### Constant Folding (AST Transformation)

**Transform**: Replace subtrees representing constant expressions with literal nodes.

**Example**:
```
Before:
    BinaryOp(Add, Literal(2), Literal(3))

After:
    Literal(5)
```

**Implementation**:
```rust
fn constant_fold(node: ASTNode) -> ASTNode {
    match node {
        ASTNode::BinaryOp { op: BinOp::Add, left, right } => {
            let left = constant_fold(*left);
            let right = constant_fold(*right);

            if let (ASTNode::Literal(a), ASTNode::Literal(b)) = (&left, &right) {
                return ASTNode::Literal(a + b);
            }

            ASTNode::BinaryOp {
                op: BinOp::Add,
                left: Box::new(left),
                right: Box::new(right)
            }
        }
        // Similar for other operators
        _ => node
    }
}
```

#### Dead Branch Elimination

**Transform**: Remove branches that can never execute.

**Example**:
```
Before:
    If(Literal(true), then_branch, else_branch)

After:
    then_branch  (else_branch is unreachable)
```

### Visitor Pattern for AST Processing

**Problem**: Want to perform operations on AST without modifying node types.

**Solution**: Visitor pattern separates traversal from operation.

**Trait definition**:
```rust
trait ASTVisitor {
    type Result;

    fn visit_literal(&mut self, value: i64) -> Self::Result;
    fn visit_identifier(&mut self, name: &str) -> Self::Result;
    fn visit_binary_op(&mut self, op: BinOp, left: Self::Result, right: Self::Result) -> Self::Result;
    // ... other visit methods
}

impl ASTNode {
    fn accept<V: ASTVisitor>(&self, visitor: &mut V) -> V::Result {
        match self {
            ASTNode::Literal(v) => visitor.visit_literal(*v),
            ASTNode::Identifier(name) => visitor.visit_identifier(name),
            ASTNode::BinaryOp { op, left, right } => {
                let left_result = left.accept(visitor);
                let right_result = right.accept(visitor);
                visitor.visit_binary_op(*op, left_result, right_result)
            }
            // ... other cases
        }
    }
}
```

**Example visitor: Expression evaluator**:
```rust
struct Evaluator {
    env: HashMap<String, i64>
}

impl ASTVisitor for Evaluator {
    type Result = i64;

    fn visit_literal(&mut self, value: i64) -> i64 {
        value
    }

    fn visit_identifier(&mut self, name: &str) -> i64 {
        *self.env.get(name).expect("Undefined variable")
    }

    fn visit_binary_op(&mut self, op: BinOp, left: i64, right: i64) -> i64 {
        match op {
            BinOp::Add => left + right,
            BinOp::Sub => left - right,
            BinOp::Mul => left * right,
            BinOp::Div => left / right,
            // ... other ops
        }
    }
}
```

**Usage**:
```rust
let ast = BinaryOp(Add, Literal(2), Literal(3));
let mut evaluator = Evaluator { env: HashMap::new() };
let result = ast.accept(&mut evaluator);
assert_eq!(result, 5);
```

---

## 2.2 Three-Address Code

### Definition and Properties

**Three-Address Code (TAC)**: Linear IR where each instruction has at most one operator and three addresses (operands).

**General form**:
```
x = y op z      (binary operation)
x = op y        (unary operation)
x = y           (copy)
x = y[i]        (indexed load)
x[i] = y        (indexed store)
goto L          (unconditional jump)
if x relop y goto L  (conditional jump)
param x         (pass parameter)
call f, n       (call function f with n parameters)
return x        (return from function)
```

**Properties**:
1. **Linear structure**: Sequence of instructions (not tree)
2. **Explicit temporaries**: Intermediate results stored in temporary variables
3. **Simple operations**: Each instruction performs one operation
4. **Named values**: All operands are variables or constants

### Example Translation

**Source code**:
```rust
let a = 2 + 3 * 4;
let b = a - 5;
return b;
```

**Three-address code**:
```
t1 = 3 * 4        // Compute 3 * 4
t2 = 2 + t1       // Compute 2 + (3 * 4)
a = t2            // Assign to a
t3 = a - 5        // Compute a - 5
b = t3            // Assign to b
return b          // Return b
```

**Note**: Temporaries $t_1, t_2, t_3$ are introduced to hold intermediate values.

### Control Flow in TAC

**Source code**:
```rust
if x > 0 {
    y = x * 2;
} else {
    y = 0;
}
z = y + 1;
```

**Three-address code with labels**:
```
      if x <= 0 goto L1     // If condition false, jump to else
      t1 = x * 2            // Then branch
      y = t1
      goto L2               // Skip else branch
L1:   y = 0                 // Else branch
L2:   t2 = y + 1            // Merge point
      z = t2
```

**Key insight**: Control flow represented by labeled instructions and jumps, not tree structure.

### Quadruples Representation

**Quadruple**: 4-tuple $(op, arg_1, arg_2, result)$ representing one TAC instruction.

**Example**: For `t1 = a + b`
```
Quadruple: (ADD, a, b, t1)
```

**Complete example**:
```
Source: a = 2 + 3 * 4

Quadruples:
(MUL, 3, 4, t1)
(ADD, 2, t1, t2)
(COPY, t2, -, a)
```

**Storage**: Array of quadruples
```
Index  Op     Arg1   Arg2   Result
0      MUL    3      4      t1
1      ADD    2      t1     t2
2      COPY   t2     -      a
```

**Advantage**: Compact, easy to manipulate (reorder, delete, insert instructions)

### Triples Representation

**Triple**: 3-tuple $(op, arg_1, arg_2)$ where result is implicit (reference by position).

**Example**: For `a = 2 + 3 * 4`
```
Index  Op     Arg1   Arg2
(0)    MUL    3      4
(1)    ADD    2      (0)      // Refers to result of instruction 0
(2)    COPY   (1)    a
```

**Advantage**: More compact than quadruples (no explicit result field)
**Disadvantage**: Harder to reorder (references by position, not name)

---

## 2.3 Static Single Assignment (SSA) Form

### Motivation and Definition

**Problem with non-SSA**: Variables assigned multiple times complicate analysis.

**Example**:
```
x = 1
x = x + 2     // Which x is being read? The one from line 1.
x = x * 3     // Which x? The one from line 2.
y = x + 4     // Which x? The one from line 3.
```

**Data flow question**: To know the value of $x$ at line 4, must trace back through all assignments.

**SSA Solution**: Each variable assigned exactly once.

**SSA form**:
```
x₁ = 1
x₂ = x₁ + 2
x₃ = x₂ * 3
y₁ = x₃ + 4
```

**Definition**: An IR is in **Static Single Assignment (SSA) form** if every variable has exactly one definition (assignment) in the program text.

**Key property**: For any use of variable $x_i$, there is a unique definition of $x_i$ that reaches it.

### SSA Construction Algorithm

**Input**: Non-SSA TAC
**Output**: SSA TAC

**Algorithm overview**:
1. Insert $\phi$-functions at merge points (where control flow joins)
2. Rename variables to ensure single assignment
3. Update uses to refer to correct versioned variable

#### Step 1: Dominance and Dominance Frontier

**Definition**: Block $A$ **dominates** block $B$ (written $A \text{ dom } B$) if every path from entry to $B$ passes through $A$.

**Dominance tree**: Tree where parent-child relationship represents immediate dominance.

**Dominance frontier**: Set of blocks where $A$'s dominance "ends".

Formally, $\text{DF}(A)$ = {$B$ : $A$ dominates a predecessor of $B$ but does not strictly dominate $B$}

**Why this matters**: $\phi$-functions needed at dominance frontier of definition blocks.

#### Step 2: Phi-Function Placement

**Rule**: For variable $x$, place $\phi$-function in block $B$ if:
- $x$ is defined in some block $A$
- $B$ is in dominance frontier of $A$

**Example CFG**:
```
Source:
    x = 1
    if (condition) {
        x = 2
    }
    y = x + 3

CFG:
    Block 1: x = 1
             if condition goto Block 2 else goto Block 3
    Block 2: x = 2
             goto Block 3
    Block 3: y = x + 3

Dominance:
    Block 1 dominates {Block 1, Block 2, Block 3}
    Block 2 dominates {Block 2}
    Block 3 dominates {Block 3}

Dominance frontier:
    DF(Block 1) = {}
    DF(Block 2) = {Block 3}

Phi-function placement:
    Variable x defined in Block 1 and Block 2
    Block 3 is in DF(Block 2), so place φ-function in Block 3
```

**After phi placement**:
```
Block 1: x = 1
         if condition goto Block 2 else goto Block 3
Block 2: x = 2
         goto Block 3
Block 3: x = φ(x, x)    // x₃ = φ(x₁, x₂)
         y = x + 3
```

#### Step 3: Variable Renaming

**Algorithm**: Depth-first traversal of dominator tree, maintaining stack of versions for each variable.

**Example**:
```
After phi placement:
    Block 1: x = 1
             if condition goto Block 2 else goto Block 3
    Block 2: x = 2
             goto Block 3
    Block 3: x = φ(x from Block 1, x from Block 2)
             y = x + 3

After renaming:
    Block 1: x₁ = 1
             if condition goto Block 2 else goto Block 3
    Block 2: x₂ = 2
             goto Block 3
    Block 3: x₃ = φ(x₁, x₂)
             y₁ = x₃ + 3
```

### Phi Functions: Formal Semantics

**Notation**: $x_3 = \phi(x_1, x_2)$ at block $B$ with predecessors $P_1, P_2$

**Semantics**:
$$x_3 = \begin{cases}
x_1 & \text{if control flow came from predecessor } P_1 \\
x_2 & \text{if control flow came from predecessor } P_2
\end{cases}$$

**Important**: $\phi$-function is not a real instruction—it's a notational device for analysis. During code generation, it's eliminated by inserting copies on predecessor edges.

### Example: Loop in SSA

**Source code**:
```rust
let mut sum = 0;
let mut i = 0;
while i < 10 {
    sum = sum + i;
    i = i + 1;
}
```

**Non-SSA TAC**:
```
      sum = 0
      i = 0
L1:   if i >= 10 goto L2
      sum = sum + i
      i = i + 1
      goto L1
L2:   // exit
```

**SSA form**:
```
      sum₁ = 0
      i₁ = 0
L1:   sum₂ = φ(sum₁, sum₃)
      i₂ = φ(i₁, i₃)
      if i₂ >= 10 goto L2
      sum₃ = sum₂ + i₂
      i₃ = i₂ + 1
      goto L1
L2:   // exit
```

**Key insight**: $\phi$-functions at loop header merge values from loop entry and loop back edge.

### SSA Benefits for Optimization

#### Constant Propagation in SSA

**Non-SSA (difficult)**:
```
x = 5
y = x + 3       // Can we replace x with 5? Need to check no intervening assignment.
x = 10          // Intervening assignment!
z = x * 2       // Here x is 10, not 5.
```

**SSA (trivial)**:
```
x₁ = 5
y₁ = x₁ + 3     // x₁ is always 5 (single definition)
x₂ = 10
z₁ = x₂ * 2     // x₂ is always 10

After constant propagation:
x₁ = 5
y₁ = 5 + 3      // Replaced x₁ with its constant value
x₂ = 10
z₁ = 10 * 2     // Replaced x₂ with its constant value

After constant folding:
x₁ = 5
y₁ = 8
x₂ = 10
z₁ = 20
```

#### Dead Code Elimination in SSA

**Key property**: If variable $x_i$ is never used, its definition is dead.

**SSA makes this trivial**: Count uses of each variable. If use count = 0, delete definition.

**Example**:
```
x₁ = 5
y₁ = x₁ + 3
z₁ = 10         // z₁ never used
w₁ = y₁ * 2

Use counts:
  x₁: 1 use (in y₁ = x₁ + 3)
  y₁: 1 use (in w₁ = y₁ * 2)
  z₁: 0 uses  ← dead code
  w₁: 0 uses  ← dead code (assuming w₁ not used later)

After DCE:
x₁ = 5
y₁ = x₁ + 3
```

---

## 2.4 Control Flow Graphs

### Formal Definition

**Control Flow Graph (CFG)**: Directed graph $G = (V, E, v_{\text{entry}}, v_{\text{exit}})$ where:
- $V$: Set of **basic blocks** (nodes)
- $E \subseteq V \times V$: Set of **control flow edges**
- $v_{\text{entry}} \in V$: Unique entry block
- $v_{\text{exit}} \in V$: Unique exit block

**Basic block**: Maximal sequence of instructions with:
1. **One entry**: Control enters only at the first instruction
2. **One exit**: Control leaves only at the last instruction
3. **No internal branches**: No jumps into or out of the middle of the block

### Basic Block Identification Algorithm

**Input**: Linear sequence of TAC instructions
**Output**: Set of basic blocks

**Algorithm**:
```
1. Identify leaders (first instruction of each basic block):
   - First instruction of program
   - Target of any jump (labeled instruction)
   - Instruction immediately following a jump

2. For each leader, its basic block consists of:
   - The leader
   - All instructions up to (but not including) the next leader
```

**Example**:
```
TAC:
1:    x = 1
2:    if x > 0 goto 5
3:    y = 2
4:    goto 6
5:    y = 3
6:    z = x + y

Leaders: 1 (start), 3 (after branch), 5 (branch target), 6 (after goto)

Basic blocks:
  Block 1: [1, 2]
  Block 2: [3, 4]
  Block 3: [5]
  Block 4: [6]
```

### CFG Construction

**After identifying basic blocks, build edges**:

**Rule 1**: If block $B_1$ ends with `goto L` and $L$ is first instruction of $B_2$, add edge $B_1 \to B_2$

**Rule 2**: If block $B_1$ ends with conditional `if ... goto L`:
- Add edge $B_1 \to B_2$ where $L$ is first instruction of $B_2$ (branch taken)
- Add edge $B_1 \to B_3$ where $B_3$ immediately follows $B_1$ (branch not taken)

**Rule 3**: If block $B_1$ ends with `return`, add edge $B_1 \to v_{\text{exit}}$

**Example CFG**:
```
TAC:
1:    x = 1
2:    if x > 0 goto 5
3:    y = 2
4:    goto 6
5:    y = 3
6:    z = x + y

Blocks:
  Block 1: [1: x=1, 2: if x>0 goto 5]
  Block 2: [3: y=2, 4: goto 6]
  Block 3: [5: y=3]
  Block 4: [6: z=x+y]

Edges:
  Block 1 → Block 3 (branch taken)
  Block 1 → Block 2 (branch not taken)
  Block 2 → Block 4 (goto)
  Block 3 → Block 4 (fall through)

CFG:
         Block 1
         /     \
        /       \
    Block 3   Block 2
        \       /
         \     /
         Block 4
```

### Special Blocks: Entry and Exit

**Entry block**: Artificial block with no instructions, single edge to first block.

**Exit block**: Artificial block with no instructions, edges from all return/exit points.

**Purpose**: Simplify analysis algorithms (unique source and sink in graph).

**Example**:
```
         Entry
           |
        Block 1
         /   \
    Block 2  Block 3
      |        |
    return   return
      \      /
        Exit
```

### Dominance Relations

**Definition**: Block $A$ **dominates** block $B$ (written $A \text{ dom } B$) if every path from entry to $B$ passes through $A$.

**Properties**:
- Reflexive: $A \text{ dom } A$
- Transitive: If $A \text{ dom } B$ and $B \text{ dom } C$, then $A \text{ dom } C$
- Entry block dominates all blocks

**Immediate dominator**: Block $A$ **immediately dominates** block $B$ (written $A \text{ idom } B$) if:
- $A \text{ dom } B$
- $A \neq B$
- For any block $C$ where $C \text{ dom } B$ and $C \neq B$, we have $C \text{ dom } A$

Informally: $A$ is the closest dominator of $B$ (other than $B$ itself).

**Dominator tree**: Tree where parent of $B$ is $\text{idom}(B)$.

**Example**:
```
CFG:
    Entry
      |
    Block 1
     /  \
Block 2  Block 3
     \  /
    Block 4

Dominance:
  Entry dominates {Entry, 1, 2, 3, 4}
  Block 1 dominates {1, 2, 3, 4}
  Block 2 dominates {2}
  Block 3 dominates {3}
  Block 4 dominates {4}

Immediate dominators:
  idom(Block 1) = Entry
  idom(Block 2) = Block 1
  idom(Block 3) = Block 1
  idom(Block 4) = Block 1

Dominator tree:
       Entry
         |
      Block 1
       / | \
      2  3  4
```

### Post-Dominance

**Definition**: Block $A$ **post-dominates** block $B$ if every path from $B$ to exit passes through $A$.

**Symmetric to dominance**, but direction reversed.

**Use case**: Identifying where values are "dead" (no longer used on any path to exit).

---

## 2.5 Data Flow Analysis

### What is Data Flow Analysis?

**Definition**: Static analysis technique to compute properties of program values at each point in the program.

**Purpose**: Answer questions like:
- Which variables are live at this point?
- Which expressions have already been computed?
- Which definitions reach this use?

**Framework**: Iterative computation over CFG, propagating information along edges.

### Reaching Definitions

**Problem**: For each program point, which variable definitions might reach it?

**Definition**: A definition $d$ of variable $x$ **reaches** a point $p$ if there exists a path from $d$ to $p$ with no intervening definition of $x$.

**Example**:
```
1:  x = 1        (definition d₁)
2:  y = 2
3:  if (y > 0) goto 6
4:  x = 3        (definition d₂)
5:  goto 7
6:  z = x        (which x? d₁ reaches here)
7:  w = x        (which x? d₁ and d₂ both reach here)
```

**At point 6**: Only $d_1$ reaches (path: 1 $\to$ 2 $\to$ 3 $\to$ 6)
**At point 7**: Both $d_1$ and $d_2$ reach
- Path via $d_1$: 1 $\to$ 2 $\to$ 3 $\to$ 6 $\to$ 7
- Path via $d_2$: 1 $\to$ 2 $\to$ 3 $\to$ 4 $\to$ 5 $\to$ 7

#### Reaching Definitions Data Flow Equations

**For each basic block $B$**:

**Gen set**: Definitions generated in $B$
$$\text{Gen}(B) = \{ d : d \text{ is a definition in } B \}$$

**Kill set**: Definitions killed by $B$
$$\text{Kill}(B) = \{ d : d \text{ defines variable } x \text{ and some statement in } B \text{ also defines } x \}$$

**Transfer function**:
$$\text{Out}(B) = \text{Gen}(B) \cup (\text{In}(B) - \text{Kill}(B))$$

**Meet operator** (join information from predecessors):
$$\text{In}(B) = \bigcup_{P \in \text{pred}(B)} \text{Out}(P)$$

**Algorithm**:
```
1. Initialize Out(B) = {} for all blocks B
2. Repeat until no changes:
     For each block B:
       In(B) = ⋃_{P ∈ pred(B)} Out(P)
       Out(B) = Gen(B) ∪ (In(B) - Kill(B))
```

**Convergence**: Guaranteed because sets only grow (monotone framework), and there are finitely many definitions.

### Live Variable Analysis

**Problem**: For each program point, which variables are **live** (will be used before being redefined)?

**Definition**: Variable $x$ is **live at point $p$** if there exists a path from $p$ to a use of $x$ with no intervening definition of $x$.

**Example**:
```
1:  x = 1
2:  y = 2
3:  z = x + y    // x and y are live at point 2
4:  w = z * 2    // z is live at point 3, but x and y are dead
```

**Use case**: Dead code elimination, register allocation.

#### Live Variable Data Flow Equations

**Direction**: Backward (from use to definition)

**For each basic block $B$**:

**Use set**: Variables used in $B$ before any definition in $B$
$$\text{Use}(B) = \{ x : x \text{ is used in } B \text{ before being defined in } B \}$$

**Def set**: Variables defined in $B$
$$\text{Def}(B) = \{ x : x \text{ is defined in } B \}$$

**Transfer function**:
$$\text{In}(B) = \text{Use}(B) \cup (\text{Out}(B) - \text{Def}(B))$$

**Meet operator** (join information from successors):
$$\text{Out}(B) = \bigcup_{S \in \text{succ}(B)} \text{In}(S)$$

**Algorithm**:
```
1. Initialize In(B) = {} for all blocks B
2. Repeat until no changes:
     For each block B (in reverse topological order):
       Out(B) = ⋃_{S ∈ succ(B)} In(S)
       In(B) = Use(B) ∪ (Out(B) - Def(B))
```

### Available Expressions

**Problem**: Which expressions have already been computed on all paths to this point?

**Definition**: Expression $e$ is **available at point $p$** if on every path from entry to $p$:
- $e$ has been computed
- None of $e$'s operands have been redefined since

**Use case**: Common subexpression elimination.

**Example**:
```
1:  a = x + y
2:  if (condition) goto 4
3:  b = x + y    // x + y is available (computed at line 1, no redefinition)
4:  c = x + y    // x + y NOT available (might not execute line 1 on all paths)
```

#### Available Expressions Data Flow Equations

**For each basic block $B$**:

**Gen set**: Expressions computed in $B$
$$\text{Gen}(B) = \{ e : e \text{ is computed in } B \text{ and operands not redefined after} \}$$

**Kill set**: Expressions invalidated in $B$
$$\text{Kill}(B) = \{ e : \text{operand of } e \text{ is defined in } B \}$$

**Transfer function**:
$$\text{Out}(B) = \text{Gen}(B) \cup (\text{In}(B) - \text{Kill}(B))$$

**Meet operator** (intersection—must be available on ALL paths):
$$\text{In}(B) = \bigcap_{P \in \text{pred}(B)} \text{Out}(P)$$

**Note**: Intersection (not union) because expression must be available on all incoming paths.

### Small Worked Example: Live Variables

**Source code**:
```
x = 1       // Block 1
y = 2
if (y > 0) goto L1
z = x + 3   // Block 2
goto L2
L1: z = 5   // Block 3
L2: w = z   // Block 4
```

**CFG**:
```
    Block 1
     /  \
Block 2  Block 3
     \  /
    Block 4
```

**Compute Use and Def sets**:
```
Block 1: Use = {}, Def = {x, y}
Block 2: Use = {x}, Def = {z}
Block 3: Use = {}, Def = {z}
Block 4: Use = {z}, Def = {w}
```

**Initialize In sets**:
```
In(Block 1) = {}
In(Block 2) = {}
In(Block 3) = {}
In(Block 4) = {}
```

**Iteration 1**:
```
Block 4 (no successors):
  Out(Block 4) = {}
  In(Block 4) = Use(Block 4) ∪ (Out(Block 4) - Def(Block 4))
              = {z} ∪ ({} - {w}) = {z}

Block 3:
  Out(Block 3) = In(Block 4) = {z}
  In(Block 3) = Use(Block 3) ∪ (Out(Block 3) - Def(Block 3))
              = {} ∪ ({z} - {z}) = {}

Block 2:
  Out(Block 2) = In(Block 4) = {z}
  In(Block 2) = Use(Block 2) ∪ (Out(Block 2) - Def(Block 2))
              = {x} ∪ ({z} - {z}) = {x}

Block 1:
  Out(Block 1) = In(Block 2) ∪ In(Block 3) = {x} ∪ {} = {x}
  In(Block 1) = Use(Block 1) ∪ (Out(Block 1) - Def(Block 1))
              = {} ∪ ({x} - {x, y}) = {}
```

**Iteration 2**: No changes, algorithm terminates.

**Result**:
```
Block 1: In = {},    Out = {x}      (x is live after Block 1)
Block 2: In = {x},   Out = {z}      (x is live entering Block 2)
Block 3: In = {},    Out = {z}
Block 4: In = {z},   Out = {}       (z is live entering Block 4)
```

**Interpretation**: Variable $y$ is dead after its definition (never used on any path to exit).

---

**End of Part 2**

---

# Part 3: Transpilation Theory

> **Purpose**: Transpilation (source-to-source compilation) is the core technique for the Jolt-to-Groth16 conversion. This section provides theoretical foundations for automated code translation, semantic preservation, and cross-language mapping.

## 3.1 Source-to-Source Compilation

### What is Transpilation?

**Definition**: **Transpilation** (or source-to-source compilation) is the process of translating a program from one high-level language to another while preserving semantics.

**Formal model**:
$$\text{Transpiler}: L_{\text{source}} \to L_{\text{target}}$$

Where:
- $L_{\text{source}}$: Source language with syntax $S_s$ and semantics $[\![\cdot]\!]_s$
- $L_{\text{target}}$: Target language with syntax $S_t$ and semantics $[\![\cdot]\!]_t$
- **Correctness**: For program $P \in L_{\text{source}}$, transpiled program $P' \in L_{\text{target}}$:
  $$[\![P]\!]_s = [\![P']\!]_t$$

**Contrast with compilation**:
- **Compilation**: High-level $\to$ Low-level (e.g., C $\to$ assembly)
- **Transpilation**: High-level $\to$ High-level (e.g., TypeScript $\to$ JavaScript, Rust $\to$ Gnark)

### Transpilation Pipeline

**Classical approach**:
```
Source Language
    ↓ [Parse]
Source AST
    ↓ [Semantic Analysis]
Typed Source AST
    ↓ [Lower to IR]
Intermediate Representation
    ↓ [Transform IR]
Target IR
    ↓ [Raise to Target AST]
Target AST
    ↓ [Pretty Print]
Target Language
```

**Key stages**:

1. **Frontend (Source Language)**:
   - Lexical analysis $\to$ tokens
   - Parsing $\to$ AST
   - Type checking $\to$ typed AST

2. **IR Translation**:
   - Lower source AST to IR (e.g., SSA form)
   - This captures program semantics independent of syntax

3. **IR Transformation**:
   - Map source IR constructs to target IR constructs
   - Handle language-specific idioms

4. **Backend (Target Language)**:
   - Raise IR to target AST
   - Pretty print to target syntax

### Example: Small Transpilation

**Source (Rust-like)**:
```rust
let x = 2 + 3;
let y = x * 4;
return y;
```

**Target (Go-like)**:
```go
x := 2 + 3
y := x * 4
return y
```

**IR (SSA)**:
```
x₁ = 2 + 3
y₁ = x₁ * 4
return y₁
```

**Key insight**: IR is language-neutral. Both source and target map to same IR, ensuring semantic equivalence.

---

## 3.2 Semantic Preservation

### What is Semantic Equivalence?

**Intuition**: Two programs are semantically equivalent if they produce the same observable behavior.

**Formal definition**: Programs $P_1$ and $P_2$ are **semantically equivalent** (written $P_1 \equiv P_2$) if for all inputs $I$:
$$[\![P_1]\!](I) = [\![P_2]\!](I)$$

Where $[\![P]\!](I)$ is the semantic function: "behavior of program $P$ on input $I$".

**Observable behavior** may include:
- **Output values**: Return value, printed output
- **Side effects**: File writes, network communication
- **Termination**: Does program halt or loop forever?
- **Exceptions**: Does program raise errors?

**For pure functions** (no side effects, always terminates):
$$[\![P]\!](I) = \text{output value on input } I$$

This is the case we focus on for Jolt verifier transpilation.

### Semantic Preservation Theorem

**Theorem**: Let $T: L_s \to L_t$ be a transpiler. $T$ is **semantics-preserving** if:
$$\forall P \in L_s. \forall I. [\![P]\!]_s(I) = [\![T(P)]\!]_t(I)$$

**Proof strategy** (compositional):
1. Define semantics for each language construct
2. Prove that translation of each construct preserves semantics
3. Compose to show whole-program preservation

**Example: Binary operation translation**

Source language construct: `x + y` with semantics $[\![x + y]\!]_s = [\![x]\!]_s + [\![y]\!]_s$

Target language construct: `x + y` with semantics $[\![x + y]\!]_t = [\![x]\!]_t + [\![y]\!]_t$

Translation: $T(x + y) = x + y$

Proof:
$$[\![T(x + y)]\!]_t = [\![x + y]\!]_t = [\![x]\!]_t + [\![y]\!]_t$$

By induction hypothesis: $[\![x]\!]_t = [\![x]\!]_s$ and $[\![y]\!]_t = [\![y]\!]_s$

Therefore: $[\![T(x + y)]\!]_t = [\![x]\!]_s + [\![y]\!]_s = [\![x + y]\!]_s$

### Challenges to Semantic Preservation

#### Challenge 1: Different Type Systems

**Problem**: Source and target may have incompatible types.

**Example**:
```rust
// Rust: Has Option<T> type
fn safe_div(x: i32, y: i32) -> Option<i32> {
    if y == 0 { None } else { Some(x / y) }
}
```

Target language (Go) doesn't have `Option<T>`. Must encode:
```go
// Go: Use multiple return values
func safeDiv(x int, y int) (int, bool) {
    if y == 0 {
        return 0, false
    }
    return x / y, true
}
```

**Semantic equivalence**:
- Rust `None` $\leftrightarrow$ Go `(_, false)`
- Rust `Some(v)` $\leftrightarrow$ Go `(v, true)`

#### Challenge 2: Different Control Flow

**Problem**: Source has constructs not present in target.

**Example**: Rust's pattern matching
```rust
match x {
    0 => println!("zero"),
    1 => println!("one"),
    _ => println!("many")
}
```

Must lower to if-else chain:
```go
if x == 0 {
    fmt.Println("zero")
} else if x == 1 {
    fmt.Println("one")
} else {
    fmt.Println("many")
}
```

**Semantic equivalence**: Both evaluate conditions top-to-bottom, execute first matching branch.

#### Challenge 3: Different Memory Models

**Problem**: Ownership semantics (Rust) vs garbage collection (Go).

Rust:
```rust
let x = Box::new(42);  // Ownership: x owns the box
let y = x;             // Ownership transferred to y
// x is now invalid
```

Go:
```go
x := new(int)
*x = 42
y := x  // Both x and y point to same location
// Both x and y valid
```

**Transpilation strategy**: Since Jolt verifier is pure (no mutation), this is less problematic. All values can be treated as immutable.

### Verification of Semantic Preservation

**Approach 1: Differential testing**

Generate random inputs, run both versions, compare outputs:
```rust
#[test]
fn test_equivalence() {
    for _ in 0..10000 {
        let input = generate_random_input();
        let rust_output = rust_verifier(input);
        let go_output = go_verifier(input);
        assert_eq!(rust_output, go_output);
    }
}
```

**Approach 2: Formal verification** (zkLean approach)

Extract both as formal specifications, prove equivalence:
```lean
theorem verifier_equivalence :
  ∀ (input : ProofInput),
    rust_verifier_spec input = go_verifier_spec input
```

---

## 3.3 Type System Mapping

### Type Systems: Source vs Target

**Type system**: Set of rules assigning types to expressions and enforcing type compatibility.

**Source (Rust)**:
- **Algebraic data types**: `enum Option<T> { Some(T), None }`
- **Generics**: `Vec<T>`
- **Lifetime annotations**: `&'a T`
- **Trait bounds**: `T: Clone`

**Target (Gnark/Go)**:
- **Structs**: `type Point struct { X, Y int }`
- **Interfaces**: `type Reader interface { Read() }`
- **No generics** (pre-Go 1.18)
- **No lifetimes** (garbage collected)

### Type Translation Strategies

#### Strategy 1: Direct Mapping

When types have natural correspondence:

| Rust | Gnark (Go) |
|------|------------|
| `i32` | `int32` |
| `u64` | `uint64` |
| `bool` | `bool` |
| `[T; N]` (array) | `[N]T` |
| `Vec<T>` (vector) | `[]T` (slice) |

**Example**:
```rust
// Rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

Transpiles to:
```go
// Go
func add(x int32, y int32) int32 {
    return x + y
}
```

#### Strategy 2: Encoding Complex Types

When types don't have direct correspondence:

**Rust `Option<T>`**:
```rust
enum Option<T> {
    Some(T),
    None
}
```

**Encoding in Go**:
```go
type Option_T struct {
    IsSome bool
    Value  T
}

func Some(val T) Option_T {
    return Option_T{IsSome: true, Value: val}
}

func None() Option_T {
    return Option_T{IsSome: false}
}
```

**Rust `Result<T, E>`**:
```rust
enum Result<T, E> {
    Ok(T),
    Err(E)
}
```

**Encoding in Go**:
```go
type Result_T_E struct {
    IsOk  bool
    Ok    T
    Err   E
}
```

#### Strategy 3: Monomorphization

**Problem**: Rust has generics, target may not (or different generics).

**Solution**: Generate specialized version for each concrete type.

**Rust generic**:
```rust
fn max<T: Ord>(x: T, y: T) -> T {
    if x > y { x } else { y }
}

let a = max(1, 2);        // T = i32
let b = max(1.0, 2.0);    // T = f64
```

**Monomorphized**:
```rust
fn max_i32(x: i32, y: i32) -> i32 {
    if x > y { x } else { y }
}

fn max_f64(x: f64, y: f64) -> f64 {
    if x > y { x } else { y }
}

let a = max_i32(1, 2);
let b = max_f64(1.0, 2.0);
```

**Transpile each monomorphized version separately**.

### Field Element Types (Critical for Jolt)

**Problem**: Jolt operates over finite field $\mathbb{F}_p$ where $p$ is a large prime (BN254 scalar field).

**Rust representation** (using arkworks):
```rust
use ark_bn254::Fr;  // Field element type

fn add_field_elements(x: Fr, y: Fr) -> Fr {
    x + y
}
```

**Gnark representation**:
```go
import "github.com/consensys/gnark/frontend"

func addFieldElements(api frontend.API, x frontend.Variable, y frontend.Variable) frontend.Variable {
    return api.Add(x, y)
}
```

**Key differences**:
1. **Rust**: Field elements are concrete types, arithmetic is direct (`x + y`)
2. **Gnark**: Field elements are `Variable`, arithmetic goes through `API` (`api.Add(x, y)`)

**Why this matters**: Gnark's API tracks constraints. Every operation creates R1CS constraints for the circuit.

### Type Preservation Theorem

**Theorem**: If source program $P$ is well-typed in $L_s$, and transpiler $T$ correctly maps types, then $T(P)$ is well-typed in $L_t$.

**Proof sketch**:
1. Type translation function $\mathcal{T}: \text{Type}_s \to \text{Type}_t$
2. For each typing rule in $L_s$:
   $$\frac{\Gamma \vdash_s e_1 : \tau_1 \quad \Gamma \vdash_s e_2 : \tau_2}{\Gamma \vdash_s e_1 \oplus e_2 : \tau}$$

   Construct corresponding rule in $L_t$:
   $$\frac{\mathcal{T}(\Gamma) \vdash_t T(e_1) : \mathcal{T}(\tau_1) \quad \mathcal{T}(\Gamma) \vdash_t T(e_2) : \mathcal{T}(\tau_2)}{\mathcal{T}(\Gamma) \vdash_t T(e_1 \oplus e_2) : \mathcal{T}(\tau)}$$

3. By structural induction on derivation of $\Gamma \vdash_s P : \tau$, show $\mathcal{T}(\Gamma) \vdash_t T(P) : \mathcal{T}(\tau)$

---

## 3.4 Control Flow Translation

### Control Flow Constructs

Different languages offer different control flow abstractions. Transpiler must map source constructs to target.

#### Sequential Composition

**Simplest case**: Both languages have statement sequences.

Source:
```rust
stmt1;
stmt2;
stmt3;
```

Target:
```go
stmt1
stmt2
stmt3
```

**Semantic preservation**: Execute statements in order.

#### Conditional Branching

**If-then-else**: Universal construct.

Source:
```rust
if condition {
    then_branch
} else {
    else_branch
}
```

Target (same structure):
```go
if condition {
    then_branch
} else {
    else_branch
}
```

**Semantics**:
$$[\![\text{if } c \text{ then } t \text{ else } e]\!] = \begin{cases} [\![t]\!] & \text{if } [\![c]\!] = \text{true} \\ [\![e]\!] & \text{if } [\![c]\!] = \text{false} \end{cases}$$

#### Pattern Matching to If-Else Chain

**Source (Rust)**:
```rust
match x {
    Pattern1 => expr1,
    Pattern2 => expr2,
    _ => expr3
}
```

**Target (Go) - via if-else chain**:
```go
var result T
if matches(x, Pattern1) {
    result = expr1
} else if matches(x, Pattern2) {
    result = expr2
} else {
    result = expr3
}
```

**Semantic equivalence**: Top-to-bottom matching, first match executes.

#### Loops

**While loop** (universal):

Source:
```rust
while condition {
    body
}
```

Target:
```go
for condition {
    body
}
```

**Semantics** (recursive definition):
$$[\![\text{while } c \text{ do } b]\!] = \begin{cases} [\![b]\!]; [\![\text{while } c \text{ do } b]\!] & \text{if } [\![c]\!] = \text{true} \\ \text{skip} & \text{if } [\![c]\!] = \text{false} \end{cases}$$

**For loop** (syntactic sugar):

Source:
```rust
for i in 0..n {
    body
}
```

Desugar to while:
```rust
let mut i = 0;
while i < n {
    body;
    i += 1;
}
```

Then translate to target.

### Control Flow Graph Isomorphism

**Key insight**: Source and target programs should have isomorphic CFGs.

**Definition**: Two CFGs $G_s$ and $G_t$ are **isomorphic** if there exists a bijection $f: V_s \to V_t$ such that:
- $(u, v) \in E_s \Leftrightarrow (f(u), f(v)) \in E_t$

**Transpilation principle**: Construct $G_t = T(G_s)$ such that $G_t \cong G_s$.

**Example**:

Source CFG (Rust):
```
        Entry
          |
    [if x > 0]
       /    \
  [y = x]  [y = 0]
       \    /
     [return y]
```

Target CFG (Go):
```
        Entry
          |
    [if x > 0]
       /    \
  [y = x]  [y = 0]
       \    /
     [return y]
```

**Same structure** $\implies$ same control flow $\implies$ semantic equivalence (modulo expression semantics).

### Structured vs Unstructured Control Flow

**Structured**: All control flow expressible with if/while/for (no arbitrary gotos).

**Unstructured**: Arbitrary jumps via `goto`, `break`, `continue`.

**Most modern languages are structured**. If source has `goto`, must reconstruct structured control flow.

**Example: goto elimination**

Source (with goto):
```
      x = 1
      if condition goto L1
      x = 2
L1:   y = x
```

Reconstructed (structured):
```
x = 1
if condition {
    // skip x = 2
} else {
    x = 2
}
y = x
```

**Algorithm**: Build CFG, identify loop headers and conditionals, reconstruct structured constructs.

---

## 3.5 Memory Model Translation

### Memory Models: Concepts

**Memory model**: Rules governing how programs access shared memory.

**Key concepts**:
1. **Aliasing**: Can two references point to same location?
2. **Mutability**: Can values be modified after creation?
3. **Lifetime**: When is memory allocated and deallocated?
4. **Ownership**: Who is responsible for freeing memory?

### Rust's Ownership Model

**Core rules**:
1. Each value has exactly one owner
2. When owner goes out of scope, value is dropped
3. References can be borrowed (shared `&T` or mutable `&mut T`)
4. At most one mutable reference, or any number of shared references

**Example**:
```rust
fn main() {
    let x = Box::new(42);  // x owns heap allocation
    let y = x;             // Ownership moved to y
    // println!("{}", x);  // Error: x no longer valid
    println!("{}", y);     // OK: y is owner
}  // y goes out of scope, memory freed
```

### Go's Garbage Collection Model

**Rules**:
1. Memory allocated with `new` or composite literals
2. No explicit deallocation
3. Garbage collector frees unreachable memory
4. Multiple references can point to same object
5. All references remain valid until GC

**Example**:
```go
func main() {
    x := new(int)
    *x = 42
    y := x            // y points to same location as x
    fmt.Println(*x)   // OK: x still valid
    fmt.Println(*y)   // OK: y also valid
}  // GC will free when both x and y unreachable
```

### Transpilation Strategy for Memory

**Key insight**: For pure functional code (Jolt verifier case), memory model differences don't matter.

**Why**:
- No mutation: All bindings are `let`, not `let mut`
- No aliasing issues: No shared mutable state
- No explicit memory management: Compiler handles all allocations

**Rust pure function**:
```rust
fn verify(proof: Proof) -> bool {
    let commitment = compute_commitment(&proof);
    let challenge = compute_challenge(&commitment);
    check_opening(&proof, &commitment, challenge)
}
```

**Go translation** (straightforward):
```go
func verify(proof Proof) bool {
    commitment := computeCommitment(proof)
    challenge := computeChallenge(commitment)
    return checkOpening(proof, commitment, challenge)
}
```

**No ownership transfer, no mutation $\implies$ memory models don't interfere**.

### Handling Mutable State (If Necessary)

**If source has mutation**, translate to functional style:

**Rust (imperative)**:
```rust
let mut sum = 0;
for i in 0..n {
    sum += i;
}
```

**Rust (functional)**:
```rust
let sum = (0..n).fold(0, |acc, i| acc + i);
```

**Go translation**:
```go
sum := 0
for i := 0; i < n; i++ {
    sum = foldStep(sum, i)
}

func foldStep(acc int, i int) int {
    return acc + i
}
```

**Or, use recursion**:
```go
func sumRange(n int) int {
    if n == 0 {
        return 0
    }
    return n - 1 + sumRange(n - 1)
}
```

### Circuit Memory Model (Gnark)

**Critical difference**: Gnark circuits are **static** (all "memory" is fixed-size arrays allocated at circuit definition time).

**No dynamic allocation**: Cannot create `Vec` or `HashMap` at runtime.

**Workaround**: Pre-allocate maximum size.

**Example**:

**Rust (dynamic)**:
```rust
fn sum_elements(elements: Vec<Field>) -> Field {
    elements.iter().fold(Field::zero(), |acc, &x| acc + x)
}
```

**Gnark (static, bounded)**:
```go
const MAX_ELEMENTS = 1000

type Circuit struct {
    Elements [MAX_ELEMENTS]frontend.Variable
    Length   frontend.Variable  // Actual length (≤ MAX_ELEMENTS)
}

func (circuit *Circuit) Define(api frontend.API) error {
    sum := frontend.Variable(0)
    for i := 0; i < MAX_ELEMENTS; i++ {
        // Conditionally add: if i < Length, add Elements[i], else add 0
        isInRange := api.IsZero(api.Sub(i, circuit.Length))
        term := api.Select(isInRange, circuit.Elements[i], 0)
        sum = api.Add(sum, term)
    }
    // ... rest of circuit
}
```

**Key insight**: Loops must be **unrolled** at circuit definition time. No data-dependent loop bounds.

---

**End of Part 3**

---

# Part 4: Circuit Representation (R1CS)

> **Purpose**: Understanding circuits is critical for transpiling to Groth16. This section covers arithmetic circuits, R1CS constraint systems, and the bridge from imperative code to declarative constraints.

## 4.1 Arithmetic Circuits

### What is an Arithmetic Circuit?

**Definition**: An **arithmetic circuit** is a directed acyclic graph (DAG) where:
- **Nodes**: Operations (gates) performing field arithmetic
- **Edges**: Wires carrying field element values
- **Input nodes**: Variables provided by prover/verifier
- **Output nodes**: Final computed values

**Field**: All arithmetic performed in finite field $\mathbb{F}_p$ for large prime $p$.

### Example: Small Circuit

**Computation**: $f(x, y) = (x + y) \cdot (x - y)$

**Circuit**:
```
Inputs: x, y

    x       y
     \     /
      \   /
       Add        (gate 1: z₁ = x + y)
         \
          \       x       y
           \       \     /
            \       \   /
             \       Sub     (gate 2: z₂ = x - y)
              \       /
               \     /
                 Mul         (gate 3: z₃ = z₁ · z₂)
                  |
               Output: z₃
```

**Gates**:
1. **Addition gate**: $z_1 = x + y$
2. **Subtraction gate**: $z_2 = x - y$
3. **Multiplication gate**: $z_3 = z_1 \cdot z_2$

**Size**: 3 gates (1 addition, 1 subtraction, 1 multiplication)

**Depth**: 2 (longest path from input to output)

### Boolean vs Arithmetic Circuits

**Boolean circuits**:
- Operations: AND, OR, NOT
- Values: {0, 1}
- Used in: Hardware design, Boolean satisfiability

**Arithmetic circuits**:
- Operations: +, -, × (in field $\mathbb{F}_p$)
- Values: Elements of $\mathbb{F}_p$
- Used in: SNARKs, zero-knowledge proofs

**Why arithmetic circuits for SNARKs**: Field arithmetic maps naturally to cryptographic primitives (elliptic curves, polynomial commitments).

### Circuit Satisfiability

**Problem**: Given circuit $C$ and output value $y$, does there exist input $x$ such that $C(x) = y$?

**For SNARKs**: Prover claims to know witness $w$ such that $C(x, w) = y$ where:
- $x$: Public input
- $w$: Private witness (secret)
- $y$: Public output

**Verifier checks**: Proof that such $w$ exists, without learning $w$.

---

## 4.2 R1CS Constraint Systems

### Definition of R1CS

**R1CS (Rank-1 Constraint System)**: Standard representation for arithmetic circuits as systems of quadratic constraints.

**Formal definition**: An R1CS instance consists of:
- **Field**: $\mathbb{F}_p$ for prime $p$
- **Variables**: $z = (z_0, z_1, \ldots, z_{m-1})$ where $z_0 = 1$ (constant one)
- **Constraints**: Set of $n$ constraints, each of form:
  $$(\mathbf{A}_i \cdot z) \cdot (\mathbf{B}_i \cdot z) = \mathbf{C}_i \cdot z$$

Where $\mathbf{A}_i, \mathbf{B}_i, \mathbf{C}_i \in \mathbb{F}_p^m$ are coefficient vectors.

**Constraint structure**:
- **Left operand**: $a_i = \mathbf{A}_i \cdot z = \sum_{j=0}^{m-1} A_{i,j} \cdot z_j$
- **Right operand**: $b_i = \mathbf{B}_i \cdot z = \sum_{j=0}^{m-1} B_{i,j} \cdot z_j$
- **Output**: $c_i = \mathbf{C}_i \cdot z = \sum_{j=0}^{m-1} C_{i,j} \cdot z_j$
- **Constraint**: $a_i \cdot b_i = c_i$

**Why "Rank-1"**: Each constraint is a product of two linear combinations (rank-1 quadratic form).

### Small Example: Addition

**Constraint**: $z_3 = z_1 + z_2$

**R1CS form**: Must express as $(A \cdot z) \cdot (B \cdot z) = C \cdot z$

**Trick**: Use trivial multiplication by 1:
$$(z_1 + z_2) \cdot 1 = z_3$$

**Coefficient vectors** (assuming $z = [1, z_1, z_2, z_3]$):
- $\mathbf{A} = [0, 1, 1, 0]$ so $\mathbf{A} \cdot z = 0 \cdot 1 + 1 \cdot z_1 + 1 \cdot z_2 + 0 \cdot z_3 = z_1 + z_2$
- $\mathbf{B} = [1, 0, 0, 0]$ so $\mathbf{B} \cdot z = 1 \cdot 1 = 1$
- $\mathbf{C} = [0, 0, 0, 1]$ so $\mathbf{C} \cdot z = 1 \cdot z_3 = z_3$

**Constraint**: $(z_1 + z_2) \cdot 1 = z_3$ ✓

### Small Example: Multiplication

**Constraint**: $z_3 = z_1 \cdot z_2$

**R1CS form**:
$$z_1 \cdot z_2 = z_3$$

**Coefficient vectors** (assuming $z = [1, z_1, z_2, z_3]$):
- $\mathbf{A} = [0, 1, 0, 0]$ so $\mathbf{A} \cdot z = z_1$
- $\mathbf{B} = [0, 0, 1, 0]$ so $\mathbf{B} \cdot z = z_2$
- $\mathbf{C} = [0, 0, 0, 1]$ so $\mathbf{C} \cdot z = z_3$

**Constraint**: $z_1 \cdot z_2 = z_3$ ✓

### Worked Example: $(x + y) \cdot (x - y)$

**Computation**: $f(x, y) = (x + y) \cdot (x - y)$

**Variables**: $z = [1, x, y, z_1, z_2, z_3]$ where:
- $z_0 = 1$ (constant)
- $z_1 = x$ (input)
- $z_2 = y$ (input)
- $z_3 = x + y$ (intermediate)
- $z_4 = x - y$ (intermediate)
- $z_5 = z_3 \cdot z_4$ (output)

**Constraint 1**: $z_3 = x + y$
$$(x + y) \cdot 1 = z_3$$
- $\mathbf{A}_1 = [0, 1, 1, 0, 0, 0]$ → $A_1 \cdot z = x + y$
- $\mathbf{B}_1 = [1, 0, 0, 0, 0, 0]$ → $B_1 \cdot z = 1$
- $\mathbf{C}_1 = [0, 0, 0, 1, 0, 0]$ → $C_1 \cdot z = z_3$

**Constraint 2**: $z_4 = x - y$
$$(x - y) \cdot 1 = z_4$$
- $\mathbf{A}_2 = [0, 1, -1, 0, 0, 0]$ → $A_2 \cdot z = x - y$
- $\mathbf{B}_2 = [1, 0, 0, 0, 0, 0]$ → $B_2 \cdot z = 1$
- $\mathbf{C}_2 = [0, 0, 0, 0, 1, 0]$ → $C_2 \cdot z = z_4$

**Constraint 3**: $z_5 = z_3 \cdot z_4$
$$z_3 \cdot z_4 = z_5$$
- $\mathbf{A}_3 = [0, 0, 0, 1, 0, 0]$ → $A_3 \cdot z = z_3$
- $\mathbf{B}_3 = [0, 0, 0, 0, 1, 0]$ → $B_3 \cdot z = z_4$
- $\mathbf{C}_3 = [0, 0, 0, 0, 0, 1]$ → $C_3 \cdot z = z_5$

**Matrices** ($n = 3$ constraints, $m = 6$ variables):
$$\mathbf{A} = \begin{bmatrix}
0 & 1 & 1 & 0 & 0 & 0 \\
0 & 1 & -1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}$$

$$\mathbf{B} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0
\end{bmatrix}$$

$$\mathbf{C} = \begin{bmatrix}
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

**Verification** (example values $x = 3, y = 2$):
$$z = [1, 3, 2, 5, 1, 5]$$

Check constraint 1: $(3 + 2) \cdot 1 = 5$ ✓
Check constraint 2: $(3 - 2) \cdot 1 = 1$ ✓
Check constraint 3: $5 \cdot 1 = 5$ ✓

### Matrix Form of R1CS

**Compact notation**: R1CS can be written as:
$$\mathbf{A}z \circ \mathbf{B}z = \mathbf{C}z$$

Where $\circ$ denotes **element-wise (Hadamard) product**:
$$(\mathbf{A}z \circ \mathbf{B}z)_i = (\mathbf{A}z)_i \cdot (\mathbf{B}z)_i$$

**Dimension**:
- $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{F}_p^{n \times m}$ (constraint matrices)
- $z \in \mathbb{F}_p^m$ (witness vector)
- $\mathbf{A}z, \mathbf{B}z, \mathbf{C}z \in \mathbb{F}_p^n$ (constraint evaluations)

**Satisfiability**: Witness $z$ **satisfies** R1CS instance $(\mathbf{A}, \mathbf{B}, \mathbf{C})$ if:
$$\forall i \in [n]. (\mathbf{A}_i \cdot z) \cdot (\mathbf{B}_i \cdot z) = \mathbf{C}_i \cdot z$$

### Public vs Private Variables

**Partition of witness**: $z = (z_{\text{pub}}, z_{\text{priv}})$
- **Public variables** $z_{\text{pub}}$: Known to both prover and verifier (inputs, outputs)
- **Private variables** $z_{\text{priv}}$: Known only to prover (witness)

**Statement**: "I know $z_{\text{priv}}$ such that $(z_{\text{pub}}, z_{\text{priv}})$ satisfies the R1CS instance"

**Example**: Proving knowledge of square root

**Statement**: "I know $w$ such that $w^2 = y$" where $y$ is public.

**R1CS**:
- Public: $z_0 = 1$, $z_1 = y$
- Private: $z_2 = w$
- Constraint: $w \cdot w = y$
  - $\mathbf{A} = [0, 0, 1]$, $\mathbf{B} = [0, 0, 1]$, $\mathbf{C} = [0, 1, 0]$
  - $(w) \cdot (w) = y$ ✓

---

## 4.3 From Imperative Code to Circuits

### The Flattening Process

**Goal**: Convert imperative program (with control flow) to flat constraint system.

**Challenge**: Circuits are **static** (no branches, no loops with data-dependent bounds).

### Step 1: Control Flow Flattening

**Conditional statements** become **multiplexers** (selectors).

**Source code**:
```rust
let result = if condition {
    x + y
} else {
    x - y
};
```

**Flattened**:
```rust
let then_value = x + y;
let else_value = x - y;
let result = condition * then_value + (1 - condition) * else_value;
```

**Intuition**: Compute both branches, select result based on condition.

**Condition as field element**:
- $\text{condition} = 0 \implies \text{result} = \text{else\_value}$
- $\text{condition} = 1 \implies \text{result} = \text{then\_value}$

**R1CS constraints**:
1. $z_{\text{then}} = x + y$
2. $z_{\text{else}} = x - y$
3. $z_{\text{result}} = \text{condition} \cdot z_{\text{then}} + (1 - \text{condition}) \cdot z_{\text{else}}$

### Step 2: Loop Unrolling

**Fixed-iteration loops** unrolled to repeated statements.

**Source code**:
```rust
let mut sum = 0;
for i in 0..3 {
    sum += array[i];
}
```

**Unrolled**:
```rust
let sum_0 = 0;
let sum_1 = sum_0 + array[0];
let sum_2 = sum_1 + array[1];
let sum_3 = sum_2 + array[2];
let sum = sum_3;
```

**R1CS constraints** (one per iteration):
1. $\text{sum}_1 = \text{sum}_0 + \text{array}[0]$
2. $\text{sum}_2 = \text{sum}_1 + \text{array}[1]$
3. $\text{sum}_3 = \text{sum}_2 + \text{array}[2]$

**Data-dependent loops**: Must bound iteration count, use conditional execution inside.

### Step 3: SSA Conversion

**Why SSA helps**: Each variable assigned once $\implies$ direct mapping to circuit wire.

**Example**:
```rust
let x = 5;
let y = x + 3;
let x = x * 2;  // Reassignment
let z = x + y;
```

**SSA form**:
```rust
let x₁ = 5;
let y₁ = x₁ + 3;
let x₂ = x₁ * 2;
let z₁ = x₂ + y₁;
```

**Circuit variables**: $z = [1, 5, x_1, y_1, x_2, z_1] = [1, 5, 5, 8, 10, 18]$

### Step 4: Constraint Generation

**For each operation** in SSA form, generate corresponding R1CS constraint.

**Operation types**:

**Addition**: $z_c = z_a + z_b$
$$(z_a + z_b) \cdot 1 = z_c$$

**Multiplication**: $z_c = z_a \cdot z_b$
$$z_a \cdot z_b = z_c$$

**Constant multiplication**: $z_b = k \cdot z_a$ (where $k$ is constant)
$$(k \cdot z_a) \cdot 1 = z_b$$

**Boolean constraint**: Ensure $z_a \in \{0, 1\}$
$$z_a \cdot (1 - z_a) = 0$$

(Only satisfied if $z_a = 0$ or $z_a = 1$)

### Worked Example: Full Compilation

**Source code**:
```rust
fn square_sum(x: Field, y: Field) -> Field {
    let x_sq = x * x;
    let y_sq = y * y;
    let result = x_sq + y_sq;
    result
}
```

**Step 1: SSA form** (already in SSA):
```
x₁ = x          (input)
y₁ = y          (input)
z₁ = x₁ * x₁
z₂ = y₁ * y₁
z₃ = z₁ + z₂
```

**Step 2: Circuit variables**:
$z = [1, x, y, z_1, z_2, z_3]$

**Step 3: R1CS constraints**:

**Constraint 1**: $z_1 = x \cdot x$
- $\mathbf{A}_1 = [0, 1, 0, 0, 0, 0]$, $\mathbf{B}_1 = [0, 1, 0, 0, 0, 0]$, $\mathbf{C}_1 = [0, 0, 0, 1, 0, 0]$

**Constraint 2**: $z_2 = y \cdot y$
- $\mathbf{A}_2 = [0, 0, 1, 0, 0, 0]$, $\mathbf{B}_2 = [0, 0, 1, 0, 0, 0]$, $\mathbf{C}_2 = [0, 0, 0, 0, 1, 0]$

**Constraint 3**: $z_3 = z_1 + z_2$
- $\mathbf{A}_3 = [0, 0, 0, 1, 1, 0]$, $\mathbf{B}_3 = [1, 0, 0, 0, 0, 0]$, $\mathbf{C}_3 = [0, 0, 0, 0, 0, 1]$

**Verification** (example $x = 3, y = 4$):
$$z = [1, 3, 4, 9, 16, 25]$$

Check: $3 \cdot 3 = 9$ ✓, $4 \cdot 4 = 16$ ✓, $(9 + 16) \cdot 1 = 25$ ✓

### Optimization: Reducing Constraints

**Problem**: Naive compilation produces many constraints.

**Optimization strategies**:

1. **Constant folding**: Evaluate constants at compile time
   ```
   Before: z₁ = 2 * 3, z₂ = z₁ + 5
   After:  z₂ = 11  (no constraints needed)
   ```

2. **Common subexpression elimination**: Reuse computed values
   ```
   Before: z₁ = x * x, z₂ = x * x  (2 constraints)
   After:  z₁ = x * x, z₂ = z₁     (1 constraint, 1 copy)
   ```

3. **Linear combination merging**: Combine multiple additions
   ```
   Before: z₁ = a + b, z₂ = z₁ + c  (2 constraints)
   After:  z₂ = a + b + c           (1 constraint)
   ```

---

## 4.4 Witness Generation

### What is a Witness?

**Definition**: A **witness** is an assignment of values to all circuit variables that satisfies the constraints.

**Components**:
- **Public inputs**: Known to verifier (e.g., statement)
- **Private inputs**: Known only to prover (e.g., solution)
- **Intermediate values**: Computed during execution

**Example** (square root):
- Public: $y = 25$
- Private: $w = 5$ (witness: square root of 25)
- Constraint: $w \cdot w = y$
- Full witness: $z = [1, 25, 5]$

### Witness Generation Algorithm

**Input**: R1CS instance $(\mathbf{A}, \mathbf{B}, \mathbf{C})$, public inputs, private inputs

**Output**: Full witness $z$ satisfying constraints

**Algorithm** (forward execution):
```
1. Initialize witness vector z with known values (public + private inputs)
2. Topologically sort constraints by dependency
3. For each constraint (in order):
     a. Compute intermediate value from known values
     b. Add to witness vector
4. Verify all constraints satisfied
```

**Example**: For $f(x, y) = (x + y) \cdot (x - y)$ with $x = 3, y = 2$:

**Step 1**: Initialize
- $z_0 = 1$ (constant)
- $z_1 = 3$ (input $x$)
- $z_2 = 2$ (input $y$)

**Step 2**: Compute $z_3 = x + y$
- $z_3 = 3 + 2 = 5$

**Step 3**: Compute $z_4 = x - y$
- $z_4 = 3 - 2 = 1$

**Step 4**: Compute $z_5 = z_3 \cdot z_4$
- $z_5 = 5 \cdot 1 = 5$

**Final witness**: $z = [1, 3, 2, 5, 1, 5]$

### Witness Generation for Conditional Execution

**Source code**:
```rust
let result = if condition {
    x + y
} else {
    x - y
};
```

**Witness generation**:
```
1. Evaluate condition (assume true, i.e., condition = 1)
2. Compute both branches:
   - then_value = x + y
   - else_value = x - y
3. Select result:
   - result = 1 * then_value + (1 - 1) * else_value
   - result = then_value
```

**Key point**: Both branches computed, even though only one selected. Circuit must be deterministic.

### Witness Generation Challenges

**Challenge 1: Non-determinism**

Some operations require "hints" from prover.

**Example**: Division $z = x / y$

Cannot directly compute from constraint $z \cdot y = x$ without knowing $z$.

**Solution**: Prover computes $z = x / y$ externally, provides as hint.

**Challenge 2: Cryptographic operations**

Operations like hashing, signature verification are expensive in circuits.

**Strategy**: Use circuit-friendly primitives (e.g., Poseidon hash instead of SHA256).

---

**End of Part 4**

---

# Part 5: The Extraction Problem

> **Purpose**: Extraction is the core challenge of automated transpilation. This section covers techniques for programmatically capturing program semantics from existing implementations, focusing on the zkLean pattern and its application to Jolt-to-Groth16 conversion.

## 5.1 Static Analysis vs Tracing

### Two Approaches to Understanding Code

**Problem**: Given an implementation in language $L$, extract its semantic specification.

**Approach 1: Static Analysis**

**Definition**: Analyze source code structure without executing it.

**Techniques**:
- AST traversal
- Type inference
- Data flow analysis
- Abstract interpretation

**Advantages**:
- Complete: Captures all possible execution paths
- No need to run code
- Can analyze code with undefined inputs

**Disadvantages**:
- May over-approximate (report impossible behaviors)
- Difficult for complex control flow
- Hard to handle dynamism (virtual dispatch, function pointers)

**Example** (extracting function signature):
```rust
// Source
fn add(x: i32, y: i32) -> i32 {
    x + y
}

// Static analysis extracts:
FunctionSignature {
    name: "add",
    parameters: [(x, i32), (y, i32)],
    return_type: i32
}
```

**Approach 2: Tracing (Dynamic Analysis)**

**Definition**: Execute code with instrumented runtime to observe behavior.

**Techniques**:
- Execution tracing
- Profiling
- Concrete interpretation

**Advantages**:
- Precise: Only captures actual behavior
- Handles dynamism naturally
- Simple implementation (wrap operations)

**Disadvantages**:
- Incomplete: Only sees executed paths
- Requires concrete inputs
- May miss edge cases

**Example** (tracing addition):
```rust
// Instrumented execution
let x = 5;
let y = 3;
let result = add(x, y);  // Trace: add(5, 3) → 8
```

### Hybrid Approach: zkLean Pattern

**Key insight** from PR #1060: Combine static and dynamic analysis.

**Strategy**:
1. **Static extraction**: Capture structure (function signatures, types, control flow)
2. **Dynamic tracing**: Execute builders/trait methods to capture operations
3. **Intermediate representation**: Store as language-neutral data structure

**Example workflow**:
```rust
// Jolt code defines constraints via builder
let mut builder = R1CSBuilder::new();
CS::uniform_constraints(&mut builder);

// Instead of executing constraints,
// instrument builder to record operations:
struct RecordingBuilder {
    constraints: Vec<Constraint>
}

impl R1CSBuilder for RecordingBuilder {
    fn add_constraint(&mut self, a: LinearCombination, b: LinearCombination, c: LinearCombination) {
        self.constraints.push(Constraint { a, b, c });
    }
}

// After execution, constraints vector contains extracted spec
```

**Why this works**: Rust trait system allows substituting recording implementation for real implementation, capturing operations as data.

---

## 5.2 Operation Capture Techniques

### The Builder Pattern for Extraction

**Problem**: How to execute code that performs computations, but capture the computation structure instead of computing?

**Solution**: **Builder pattern** with instrumented API.

### Example: Field Arithmetic Extraction

**Jolt code** (performs actual field arithmetic):
```rust
use ark_bn254::Fr;

fn compute(x: Fr, y: Fr) -> Fr {
    let z = x + y;
    let w = z * x;
    w
}
```

**Goal**: Extract operations as IR without executing field arithmetic.

**Approach**: Define symbolic field type that records operations.

**Symbolic field type**:
```rust
#[derive(Clone)]
struct SymbolicField {
    id: VarId,  // Unique variable identifier
    op: Option<Box<Operation>>
}

enum Operation {
    Input(String),
    Add(SymbolicField, SymbolicField),
    Mul(SymbolicField, SymbolicField),
    Constant(BigInt)
}

impl Add for SymbolicField {
    type Output = SymbolicField;

    fn add(self, other: SymbolicField) -> SymbolicField {
        let id = next_var_id();
        SymbolicField {
            id,
            op: Some(Box::new(Operation::Add(self, other)))
        }
    }
}

impl Mul for SymbolicField {
    type Output = SymbolicField;

    fn mul(self, other: SymbolicField) -> SymbolicField {
        let id = next_var_id();
        SymbolicField {
            id,
            op: Some(Box::new(Operation::Mul(self, other)))
        }
    }
}
```

**Extraction**:
```rust
let x = SymbolicField::input("x");  // x₁
let y = SymbolicField::input("y");  // x₂

let result = compute(x, y);  // Executes symbolically

// result.op contains:
// Mul(
//   Add(Input("x"), Input("y")),
//   Input("x")
// )
```

**Generated IR**:
```
z = x + y
w = z * x
return w
```

### Trait-Based Extraction (zkLean Pattern)

**Key pattern from PR #1060**: Use Rust traits to abstract over real vs symbolic execution.

**Define trait for operations**:
```rust
trait FieldOps {
    type Field;

    fn add(&self, a: Self::Field, b: Self::Field) -> Self::Field;
    fn mul(&self, a: Self::Field, b: Self::Field) -> Self::Field;
    fn constant(&self, value: u64) -> Self::Field;
}
```

**Concrete implementation** (actual computation):
```rust
struct ConcreteField;

impl FieldOps for ConcreteField {
    type Field = Fr;  // Actual field type

    fn add(&self, a: Fr, b: Fr) -> Fr {
        a + b  // Real addition
    }

    fn mul(&self, a: Fr, b: Fr) -> Fr {
        a * b  // Real multiplication
    }

    fn constant(&self, value: u64) -> Fr {
        Fr::from(value)
    }
}
```

**Symbolic implementation** (extraction):
```rust
struct SymbolicField {
    ops: Vec<Operation>
}

impl FieldOps for SymbolicField {
    type Field = VarId;  // Just track variable IDs

    fn add(&self, a: VarId, b: VarId) -> VarId {
        let result_id = self.ops.len();
        self.ops.push(Operation::Add(a, b));
        result_id
    }

    fn mul(&self, a: VarId, b: VarId) -> VarId {
        let result_id = self.ops.len();
        self.ops.push(Operation::Mul(a, b));
        result_id
    }

    fn constant(&self, value: u64) -> VarId {
        let result_id = self.ops.len();
        self.ops.push(Operation::Constant(value));
        result_id
    }
}
```

**Generic algorithm** (works with both):
```rust
fn algorithm<F: FieldOps>(ops: &F, x: F::Field, y: F::Field) -> F::Field {
    let z = ops.add(x, y);
    let w = ops.mul(z, x);
    w
}

// Execute concretely:
let concrete = ConcreteField;
let result = algorithm(&concrete, Fr::from(3), Fr::from(5));  // Computes 24

// Extract symbolically:
let mut symbolic = SymbolicField { ops: vec![] };
let result_id = algorithm(&symbolic, 0, 1);  // Records operations
// symbolic.ops now contains [Add(0, 1), Mul(2, 0)]
```

**Why this works**: Rust's monomorphization generates separate compiled versions for each trait implementation, so same source code executes differently depending on trait.

### Handling Control Flow During Extraction

**Challenge**: Branches depend on runtime values.

**Example**:
```rust
fn conditional_add<F: FieldOps>(ops: &F, x: F::Field, y: F::Field, condition: bool) -> F::Field {
    if condition {
        ops.add(x, y)
    } else {
        x
    }
}
```

**Problem**: During symbolic extraction, `condition` value unknown.

**Solution 1: Extract both branches**
```rust
let then_result = ops.add(x, y);
let else_result = x;
let result = ops.select(condition_var, then_result, else_result);
```

**Solution 2: Require condition known at extraction time**

Only extract cases actually used:
```rust
// Extract version with condition=true
let extracted_true = extract_with_condition(true);

// Extract version with condition=false
let extracted_false = extract_with_condition(false);

// Generate circuit with conditional selection between versions
```

**zkLean approach**: Generates separate constraints for each case, uses circuit flags to select.

---

## 5.3 Memory Management for IR

### Why Memory Management Matters

**Problem**: Extraction generates large ASTs/IRs (potentially millions of nodes for complex circuits).

**Naive approach**:
```rust
enum Expr {
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Var(VarId)
}
```

**Issue**: Each `Box` is separate heap allocation. For deep expressions, causes:
- Memory fragmentation
- Cache misses
- Allocation overhead

### Arena Allocation

**Key optimization from PR #1060**: Use arena (bump allocator) for AST nodes.

**Concept**: Pre-allocate large contiguous memory block, allocate nodes sequentially.

**Benefits**:
- Fast allocation (just increment pointer)
- Good cache locality (nodes near each other in memory)
- Simple deallocation (free entire arena at once)

**Implementation**:
```rust
use typed_arena::Arena;

struct ExprArena<'a> {
    arena: Arena<Expr<'a>>
}

enum Expr<'a> {
    Add(&'a Expr<'a>, &'a Expr<'a>),
    Mul(&'a Expr<'a>, &'a Expr<'a>),
    Var(VarId)
}

impl<'a> ExprArena<'a> {
    fn new() -> Self {
        ExprArena { arena: Arena::new() }
    }

    fn add(&'a self, left: &'a Expr<'a>, right: &'a Expr<'a>) -> &'a Expr<'a> {
        self.arena.alloc(Expr::Add(left, right))
    }

    fn mul(&'a self, left: &'a Expr<'a>, right: &'a Expr<'a>) -> &'a Expr<'a> {
        self.arena.alloc(Expr::Mul(left, right))
    }
}
```

**Usage**:
```rust
let arena = ExprArena::new();

let x = arena.alloc(Expr::Var(0));
let y = arena.alloc(Expr::Var(1));
let z = arena.add(x, y);
let w = arena.mul(z, x);

// All expressions in contiguous memory
// Deallocated together when arena dropped
```

**Performance impact**: PR #1060 notes this was critical for handling 64-bit instruction extraction (large MLEs).

### Interning (Hash-Consing)

**Problem**: Same subexpression appears multiple times.

**Example**:
```rust
let a = x + y;
let b = x + y;  // Same as a
let c = a * b;
```

**Naive IR**: Stores `x + y` twice (duplicated nodes).

**Interning**: Store each unique expression once, reuse reference.

**Implementation**:
```rust
use std::collections::HashMap;

struct ExprInterner<'a> {
    arena: Arena<Expr<'a>>,
    cache: HashMap<ExprKey, &'a Expr<'a>>
}

#[derive(Hash, Eq, PartialEq)]
enum ExprKey {
    Add(usize, usize),  // (left_id, right_id)
    Mul(usize, usize),
    Var(VarId)
}

impl<'a> ExprInterner<'a> {
    fn add(&'a mut self, left: &'a Expr<'a>, right: &'a Expr<'a>) -> &'a Expr<'a> {
        let key = ExprKey::Add(expr_id(left), expr_id(right));

        if let Some(&cached) = self.cache.get(&key) {
            return cached;  // Reuse existing node
        }

        let new_expr = self.arena.alloc(Expr::Add(left, right));
        self.cache.insert(key, new_expr);
        new_expr
    }
}
```

**Benefits**:
- Reduced memory usage (no duplicates)
- Automatic common subexpression elimination
- Structural equality by pointer comparison

**Trade-off**: Hash table overhead vs memory savings. Worth it for large circuits.

---

## 5.4 Optimization: Common Subexpression Elimination

### CSE in Extracted IR

**Problem**: Extraction may generate redundant computations.

**Example**:
```rust
let a = x * x;
let b = y + 3;
let c = x * x;  // Same as a
let d = a + c;
```

**Without CSE**: 4 operations
**With CSE**: 3 operations (reuse `a` for `c`)

### CSE Algorithm for DAG IR

**Input**: IR as expression DAG
**Output**: Optimized DAG with shared subexpressions

**Algorithm**:
```
1. Traverse IR in post-order
2. For each node:
     a. Compute structural hash (hash of operation + children hashes)
     b. Look up in hash table
     c. If found, replace with existing node
     d. If not found, add to hash table
3. Update all references to use deduplicated nodes
```

**Implementation**:
```rust
fn eliminate_common_subexpressions(root: &Expr) -> &Expr {
    let mut seen: HashMap<ExprHash, &Expr> = HashMap::new();
    let mut interned_arena = ExprInterner::new();

    fn visit<'a>(
        expr: &Expr,
        seen: &mut HashMap<ExprHash, &'a Expr<'a>>,
        arena: &'a mut ExprInterner<'a>
    ) -> &'a Expr<'a> {
        let hash = compute_hash(expr);

        if let Some(&cached) = seen.get(&hash) {
            return cached;  // Reuse
        }

        let new_expr = match expr {
            Expr::Add(left, right) => {
                let left = visit(left, seen, arena);
                let right = visit(right, seen, arena);
                arena.add(left, right)
            }
            Expr::Mul(left, right) => {
                let left = visit(left, seen, arena);
                let right = visit(right, seen, arena);
                arena.mul(left, right)
            }
            Expr::Var(id) => arena.var(*id)
        };

        seen.insert(hash, new_expr);
        new_expr
    }

    visit(root, &mut seen, &mut interned_arena)
}
```

### Example: CSE in Action

**Before CSE**:
```
     Add
    /   \
  Mul   Mul
  / \   / \
 x   x x   x
```

4 nodes (2 multiplications computed twice)

**After CSE**:
```
       Add
      /   \
     /     \
    Mul    Mul  (same node, shared)
    / \
   x   x
```

3 nodes (1 multiplication shared)

**R1CS constraint count**: Reduced from 2 to 1 multiplication constraint.

### Constant Folding in Extracted IR

**Problem**: Extraction may create operations on known constants.

**Example**:
```rust
let a = 2 + 3;
let b = a * 4;
```

**Without folding**: 2 operations
**With folding**: 0 operations (compile-time constant `b = 20`)

**Algorithm**:
```
1. Traverse IR in post-order
2. For each node:
     a. If all children are constants, evaluate
     b. Replace node with constant result
     c. Otherwise, keep node
```

**Implementation**:
```rust
fn constant_fold(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(left, right) => {
            let left = constant_fold(left);
            let right = constant_fold(right);

            match (&left, &right) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
                _ => Expr::Add(Box::new(left), Box::new(right))
            }
        }
        Expr::Mul(left, right) => {
            let left = constant_fold(left);
            let right = constant_fold(right);

            match (&left, &right) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
                _ => Expr::Mul(Box::new(left), Box::new(right))
            }
        }
        Expr::Var(id) => Expr::Var(*id),
        Expr::Const(val) => Expr::Const(*val)
    }
}
```

### Dead Code Elimination in Extracted IR

**Problem**: Extracted IR may contain unused computations.

**Example**:
```rust
let a = x + y;
let b = a * 2;  // Unused
let c = x - y;
return c;
```

**Without DCE**: 3 operations
**With DCE**: 1 operation (only `c` needed)

**Algorithm** (mark-and-sweep):
```
1. Mark phase:
     a. Start from root (output expressions)
     b. Recursively mark all children as "live"

2. Sweep phase:
     a. Remove all unmarked nodes
```

**Implementation**:
```rust
fn eliminate_dead_code(roots: &[&Expr]) -> Vec<Expr> {
    let mut live: HashSet<ExprId> = HashSet::new();

    // Mark phase
    fn mark(expr: &Expr, live: &mut HashSet<ExprId>) {
        if live.contains(&expr_id(expr)) {
            return;  // Already marked
        }

        live.insert(expr_id(expr));

        match expr {
            Expr::Add(left, right) | Expr::Mul(left, right) => {
                mark(left, live);
                mark(right, live);
            }
            Expr::Var(_) | Expr::Const(_) => {}
        }
    }

    for root in roots {
        mark(root, &mut live);
    }

    // Sweep phase: only keep live expressions
    filter_by_liveness(live)
}
```

---

**End of Part 5**

---

# Appendices

## Appendix A: Notation Reference

### Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbb{F}_p$ | Finite field of prime order $p$ |
| $[\![P]\!]$ | Semantic function (meaning of program $P$) |
| $\Gamma$ | Type environment (context) |
| $\Gamma \vdash e : \tau$ | Expression $e$ has type $\tau$ in context $\Gamma$ |
| $L_s, L_t$ | Source and target languages |
| $\mathbf{A}, \mathbf{B}, \mathbf{C}$ | R1CS constraint matrices |
| $z$ | Witness vector |
| $\circ$ | Hadamard (element-wise) product |
| $G = (V, E)$ | Graph with vertices $V$ and edges $E$ |
| $A \text{ dom } B$ | Block $A$ dominates block $B$ |
| $\text{DF}(A)$ | Dominance frontier of block $A$ |
| $\phi(x_1, x_2)$ | Phi function (SSA merge) |

### Programming Notation

| Notation | Meaning |
|----------|---------|
| `x: i32` | Variable `x` of type `i32` |
| `&T` | Shared reference to type `T` |
| `&mut T` | Mutable reference to type `T` |
| `Box<T>` | Heap-allocated value of type `T` |
| `Vec<T>` | Dynamically-sized array of `T` |
| `Option<T>` | Either `Some(T)` or `None` |
| `→` | Function arrow (maps input to output) |

## Appendix B: Connection to Jolt Transpilation

### Application to Jolt Verifier

**Challenge**: Convert Jolt's Rust verifier to Gnark circuits for Groth16 proving.

**Approach** (applying concepts from this document):

1. **Part 0-1**: Understand Jolt's RISC-V execution model and compilation pipeline
   - Verifier operates over execution traces (instruction sequences)
   - Sumcheck verification is core operation

2. **Part 2**: Extract verifier as SSA-based IR
   - Use zkLean pattern: instrument field operations to capture computation graph
   - Result: IR capturing sumcheck operations, polynomial evaluations, etc.

3. **Part 3**: Transpile IR to Gnark
   - Map Rust field types (`ark_bn254::Fr`) to Gnark variables (`frontend.Variable`)
   - Translate control flow (mostly sequential, some conditionals)
   - Handle memory model (Jolt verifier is pure, simplifies translation)

4. **Part 4**: Generate R1CS constraints
   - Each field operation becomes R1CS constraint
   - Sumcheck rounds become sequences of polynomial evaluation constraints
   - Opening proofs become commitmentverification constraints

5. **Part 5**: Optimize extracted circuit
   - Apply CSE to reduce redundant field operations
   - Constant-fold known values
   - Eliminate dead code

**Key insights from this document for Jolt**:

- **Extraction via traits** (5.2): Jolt's use of traits (`JoltField`, `CommitmentScheme`) enables symbolic execution for extraction
- **Arena allocation** (5.3): Critical for handling large polynomial expressions
- **SSA form** (2.3): Natural representation for linear constraint systems
- **Circuit flattening** (4.3): Sumcheck loops must be unrolled to fixed number of rounds

### Specific Jolt Components

**Sumcheck extraction**:
- Extract polynomial evaluation logic (Horner's method)
- Capture round structure (16 rounds per sumcheck)
- Generate constraints for consistency checks

**Dory PCS extraction** (with hints from PR #975):
- Expensive operations (GT exponentiations) replaced by hints
- Generate constraints verifying hint correctness
- Use Hyrax auxiliary proof

**Future work** (lattice PCS):
- Similar extraction approach
- Generate constraints for lattice operations (error polynomials, ring arithmetic)
- May use quotient-based optimizations

## Appendix C: Further Reading

### Computer Architecture
- Hennessy & Patterson, "Computer Architecture: A Quantitative Approach"
- RISC-V ISA Specification (https://riscv.org/specifications/)

### Compiler Design
- Aho et al., "Compilers: Principles, Techniques, and Tools" (Dragon Book)
- Appel, "Modern Compiler Implementation in ML"
- Cooper & Torczon, "Engineering a Compiler"

### Program Analysis
- Nielson et al., "Principles of Program Analysis"
- Cytron et al., "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph" (original SSA paper)

### SNARKs and Circuits
- Groth, "On the Size of Pairing-based Non-interactive Arguments" (Groth16 paper)
- Setty et al., "Spartan: Efficient and General-Purpose zkSNARKs without Trusted Setup"
- Ben-Sasson et al., "SNARKs for C: Verifying Program Executions Succinctly and in Zero Knowledge"

### Transpilation
- Visser, "A Survey of Strategies in Rule-Based Program Transformation Systems"
- Lattner & Adve, "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation"

---

**End of Document**


---

## 0.7 Extended RISC-V Deep Dive

### Complete RV64IMAC Instruction Reference

**Purpose**: Jolt implements RV64IMAC (64-bit Base Integer + M + A + C extensions). Understanding every instruction is critical for verifier implementation.

#### Instruction Format Summary

All base instructions are 32 bits (4 bytes). Compressed instructions are 16 bits.

**Format overview**:
- **R-type**: Register-register operations (ADD, SUB, MUL, etc.)
- **I-type**: Immediate operations (ADDI, LD, JALR)
- **S-type**: Store operations (SD, SW, SH, SB)
- **B-type**: Branch operations (BEQ, BNE, BLT, BGE)
- **U-type**: Upper immediate (LUI, AUIPC)
- **J-type**: Jump operations (JAL)

### RV64I: Base Integer Instructions

#### Arithmetic Operations

**ADD** (Add)
```
Format: R-type
Encoding: funct7=0000000, funct3=000, opcode=0110011
Syntax: add rd, rs1, rs2
Semantics: rd = rs1 + rs2 (64-bit addition, overflow wraps)
Example: add x5, x3, x4  // x5 = x3 + x4
```

**ADDI** (Add Immediate)
```
Format: I-type
Encoding: funct3=000, opcode=0010011
Syntax: addi rd, rs1, imm
Semantics: rd = rs1 + sign_extend(imm[11:0])
Example: addi x5, x3, 10  // x5 = x3 + 10
Range: imm ∈ [-2048, 2047]
```

**ADDIW** (Add Immediate Word)
```
Format: I-type
Encoding: funct3=000, opcode=0011011
Syntax: addiw rd, rs1, imm
Semantics: rd = sign_extend((rs1[31:0] + sign_extend(imm[11:0]))[31:0])
Note: Operates on lower 32 bits, sign-extends result to 64 bits
Example: addiw x5, x3, -5  // x5 = sign_extend((x3[31:0] - 5)[31:0])
```

**SUB** (Subtract)
```
Format: R-type
Encoding: funct7=0100000, funct3=000, opcode=0110011
Syntax: sub rd, rs1, rs2
Semantics: rd = rs1 - rs2
Example: sub x5, x3, x4  // x5 = x3 - x4
```

**SUBW** (Subtract Word)
```
Format: R-type
Encoding: funct7=0100000, funct3=000, opcode=0111011
Syntax: subw rd, rs1, rs2
Semantics: rd = sign_extend((rs1[31:0] - rs2[31:0])[31:0])
Example: subw x5, x3, x4
```

#### Logical Operations

**AND** (Bitwise AND)
```
Format: R-type
Encoding: funct7=0000000, funct3=111, opcode=0110011
Syntax: and rd, rs1, rs2
Semantics: rd = rs1 & rs2
Example: and x5, x3, x4
```

**ANDI** (AND Immediate)
```
Format: I-type
Encoding: funct3=111, opcode=0010011
Syntax: andi rd, rs1, imm
Semantics: rd = rs1 & sign_extend(imm[11:0])
Example: andi x5, x3, 0xFF  // Mask lower byte
```

**OR** (Bitwise OR)
```
Format: R-type
Encoding: funct7=0000000, funct3=110, opcode=0110011
Syntax: or rd, rs1, rs2
Semantics: rd = rs1 | rs2
Example: or x5, x3, x4
```

**ORI** (OR Immediate)
```
Format: I-type
Encoding: funct3=110, opcode=0010011
Syntax: ori rd, rs1, imm
Semantics: rd = rs1 | sign_extend(imm[11:0])
Example: ori x5, x3, 0x10
```

**XOR** (Bitwise XOR)
```
Format: R-type
Encoding: funct7=0000000, funct3=100, opcode=0110011
Syntax: xor rd, rs1, rs2
Semantics: rd = rs1 ^ rs2
Example: xor x5, x3, x4
```

**XORI** (XOR Immediate)
```
Format: I-type
Encoding: funct3=100, opcode=0010011
Syntax: xori rd, rs1, imm
Semantics: rd = rs1 ^ sign_extend(imm[11:0])
Example: xori x5, x3, -1  // Bitwise NOT (x5 = ~x3)
```

#### Shift Operations

**SLL** (Shift Left Logical)
```
Format: R-type
Encoding: funct7=0000000, funct3=001, opcode=0110011
Syntax: sll rd, rs1, rs2
Semantics: rd = rs1 << (rs2[5:0])  // Shift amount from lower 6 bits of rs2
Example: sll x5, x3, x4  // x5 = x3 << (x4 & 0x3F)
```

**SLLI** (Shift Left Logical Immediate)
```
Format: I-type (special)
Encoding: funct7=0000000, funct3=001, opcode=0010011
Syntax: slli rd, rs1, shamt
Semantics: rd = rs1 << shamt
Range: shamt ∈ [0, 63]
Example: slli x5, x3, 2  // x5 = x3 << 2 (multiply by 4)
```

**SRL** (Shift Right Logical)
```
Format: R-type
Encoding: funct7=0000000, funct3=101, opcode=0110011
Syntax: srl rd, rs1, rs2
Semantics: rd = rs1 >> (rs2[5:0])  // Zero-extend (logical shift)
Example: srl x5, x3, x4
```

**SRLI** (Shift Right Logical Immediate)
```
Format: I-type (special)
Encoding: funct7=0000000, funct3=101, opcode=0010011
Syntax: srli rd, rs1, shamt
Semantics: rd = rs1 >> shamt  // Zero-extend
Example: srli x5, x3, 2  // x5 = x3 >> 2 (unsigned divide by 4)
```

**SRA** (Shift Right Arithmetic)
```
Format: R-type
Encoding: funct7=0100000, funct3=101, opcode=0110011
Syntax: sra rd, rs1, rs2
Semantics: rd = rs1 >> (rs2[5:0])  // Sign-extend (arithmetic shift)
Example: sra x5, x3, x4
Note: Preserves sign bit (bit 63)
```

**SRAI** (Shift Right Arithmetic Immediate)
```
Format: I-type (special)
Encoding: funct7=0100000, funct3=101, opcode=0010011
Syntax: srai rd, rs1, shamt
Semantics: rd = rs1 >> shamt  // Sign-extend
Example: srai x5, x3, 2  // Signed divide by 4 (rounding toward -∞)
```

#### Comparison Operations

**SLT** (Set Less Than)
```
Format: R-type
Encoding: funct7=0000000, funct3=010, opcode=0110011
Syntax: slt rd, rs1, rs2
Semantics: rd = (rs1 <s rs2) ? 1 : 0  // Signed comparison
Example: slt x5, x3, x4  // x5 = 1 if x3 < x4 (signed), else 0
```

**SLTI** (Set Less Than Immediate)
```
Format: I-type
Encoding: funct3=010, opcode=0010011
Syntax: slti rd, rs1, imm
Semantics: rd = (rs1 <s sign_extend(imm)) ? 1 : 0
Example: slti x5, x3, 100
```

**SLTU** (Set Less Than Unsigned)
```
Format: R-type
Encoding: funct7=0000000, funct3=011, opcode=0110011
Syntax: sltu rd, rs1, rs2
Semantics: rd = (rs1 <u rs2) ? 1 : 0  // Unsigned comparison
Example: sltu x5, x3, x4
```

**SLTIU** (Set Less Than Immediate Unsigned)
```
Format: I-type
Encoding: funct3=011, opcode=0010011
Syntax: sltiu rd, rs1, imm
Semantics: rd = (rs1 <u sign_extend(imm)) ? 1 : 0
Example: sltiu x5, x3, 100
Note: Despite name, immediate is sign-extended, then compared unsigned
```

#### Load Operations

**LD** (Load Doubleword)
```
Format: I-type
Encoding: funct3=011, opcode=0000011
Syntax: ld rd, offset(rs1)
Semantics: rd = M[rs1 + sign_extend(offset)][63:0]
Example: ld x5, 8(x3)  // Load 8 bytes from address (x3 + 8)
Alignment: Must be 8-byte aligned (address % 8 == 0)
```

**LW** (Load Word)
```
Format: I-type
Encoding: funct3=010, opcode=0000011
Syntax: lw rd, offset(rs1)
Semantics: rd = sign_extend(M[rs1 + sign_extend(offset)][31:0])
Example: lw x5, 4(x3)  // Load 4 bytes, sign-extend to 64 bits
Alignment: Must be 4-byte aligned
```

**LWU** (Load Word Unsigned)
```
Format: I-type
Encoding: funct3=110, opcode=0000011
Syntax: lwu rd, offset(rs1)
Semantics: rd = zero_extend(M[rs1 + sign_extend(offset)][31:0])
Example: lwu x5, 4(x3)  // Load 4 bytes, zero-extend to 64 bits
```

**LH** (Load Halfword)
```
Format: I-type
Encoding: funct3=001, opcode=0000011
Syntax: lh rd, offset(rs1)
Semantics: rd = sign_extend(M[rs1 + sign_extend(offset)][15:0])
Example: lh x5, 2(x3)  // Load 2 bytes, sign-extend
Alignment: Must be 2-byte aligned
```

**LHU** (Load Halfword Unsigned)
```
Format: I-type
Encoding: funct3=101, opcode=0000011
Syntax: lhu rd, offset(rs1)
Semantics: rd = zero_extend(M[rs1 + sign_extend(offset)][15:0])
Example: lhu x5, 2(x3)  // Load 2 bytes, zero-extend
```

**LB** (Load Byte)
```
Format: I-type
Encoding: funct3=000, opcode=0000011
Syntax: lb rd, offset(rs1)
Semantics: rd = sign_extend(M[rs1 + sign_extend(offset)][7:0])
Example: lb x5, 1(x3)  // Load 1 byte, sign-extend
Alignment: No alignment requirement
```

**LBU** (Load Byte Unsigned)
```
Format: I-type
Encoding: funct3=100, opcode=0000011
Syntax: lbu rd, offset(rs1)
Semantics: rd = zero_extend(M[rs1 + sign_extend(offset)][7:0])
Example: lbu x5, 1(x3)  // Load 1 byte, zero-extend
```

#### Store Operations

**SD** (Store Doubleword)
```
Format: S-type
Encoding: funct3=011, opcode=0100011
Syntax: sd rs2, offset(rs1)
Semantics: M[rs1 + sign_extend(offset)][63:0] = rs2[63:0]
Example: sd x5, 8(x3)  // Store 8 bytes from x5 to address (x3 + 8)
Alignment: Must be 8-byte aligned
```

**SW** (Store Word)
```
Format: S-type
Encoding: funct3=010, opcode=0100011
Syntax: sw rs2, offset(rs1)
Semantics: M[rs1 + sign_extend(offset)][31:0] = rs2[31:0]
Example: sw x5, 4(x3)  // Store lower 4 bytes of x5
Alignment: Must be 4-byte aligned
```

**SH** (Store Halfword)
```
Format: S-type
Encoding: funct3=001, opcode=0100011
Syntax: sh rs2, offset(rs1)
Semantics: M[rs1 + sign_extend(offset)][15:0] = rs2[15:0]
Example: sh x5, 2(x3)  // Store lower 2 bytes of x5
Alignment: Must be 2-byte aligned
```

**SB** (Store Byte)
```
Format: S-type
Encoding: funct3=000, opcode=0100011
Syntax: sb rs2, offset(rs1)
Semantics: M[rs1 + sign_extend(offset)][7:0] = rs2[7:0]
Example: sb x5, 1(x3)  // Store lowest byte of x5
Alignment: No alignment requirement
```

#### Branch Operations

**BEQ** (Branch if Equal)
```
Format: B-type
Encoding: funct3=000, opcode=1100011
Syntax: beq rs1, rs2, offset
Semantics: if (rs1 == rs2) PC = PC + sign_extend(offset)
Example: beq x3, x4, label  // Jump to label if x3 == x4
Range: offset ∈ [-4096, 4094], must be multiple of 2
```

**BNE** (Branch if Not Equal)
```
Format: B-type
Encoding: funct3=001, opcode=1100011
Syntax: bne rs1, rs2, offset
Semantics: if (rs1 != rs2) PC = PC + sign_extend(offset)
Example: bne x3, x4, label
```

**BLT** (Branch if Less Than)
```
Format: B-type
Encoding: funct3=100, opcode=1100011
Syntax: blt rs1, rs2, offset
Semantics: if (rs1 <s rs2) PC = PC + sign_extend(offset)
Example: blt x3, x4, label  // Jump if x3 < x4 (signed)
```

**BGE** (Branch if Greater or Equal)
```
Format: B-type
Encoding: funct3=101, opcode=1100011
Syntax: bge rs1, rs2, offset
Semantics: if (rs1 >=s rs2) PC = PC + sign_extend(offset)
Example: bge x3, x4, label
```

**BLTU** (Branch if Less Than Unsigned)
```
Format: B-type
Encoding: funct3=110, opcode=1100011
Syntax: bltu rs1, rs2, offset
Semantics: if (rs1 <u rs2) PC = PC + sign_extend(offset)
Example: bltu x3, x4, label  // Jump if x3 < x4 (unsigned)
```

**BGEU** (Branch if Greater or Equal Unsigned)
```
Format: B-type
Encoding: funct3=111, opcode=1100011
Syntax: bgeu rs1, rs2, offset
Semantics: if (rs1 >=u rs2) PC = PC + sign_extend(offset)
Example: bgeu x3, x4, label
```

#### Jump Operations

**JAL** (Jump and Link)
```
Format: J-type
Encoding: opcode=1101111
Syntax: jal rd, offset
Semantics: 
  rd = PC + 4          // Save return address
  PC = PC + sign_extend(offset)
Example: jal x1, function  // Call function, return address in x1 (ra)
Range: offset ∈ [-1048576, 1048574], must be multiple of 2
Common usage: jal ra, func  (function call)
              jal zero, label  (unconditional jump, no link)
```

**JALR** (Jump and Link Register)
```
Format: I-type
Encoding: funct3=000, opcode=1100111
Syntax: jalr rd, offset(rs1)
Semantics:
  t = PC + 4
  PC = (rs1 + sign_extend(offset)) & ~1  // Clear bit 0
  rd = t
Example: jalr x0, 0(x1)  // Return (jump to ra, no link)
         jalr x1, 0(x5)  // Indirect call
Common usage: jalr zero, 0(ra)  (return from function)
              jalr ra, 0(t0)    (indirect call via t0)
```

#### Upper Immediate Operations

**LUI** (Load Upper Immediate)
```
Format: U-type
Encoding: opcode=0110111
Syntax: lui rd, imm
Semantics: rd = sign_extend(imm[31:12] << 12)
Example: lui x5, 0x12345  // x5 = 0x0000000012345000
Use: Combined with ADDI to construct 32-bit constants
     lui x5, 0x12345
     addi x5, x5, 0x678  // x5 = 0x12345678
```

**AUIPC** (Add Upper Immediate to PC)
```
Format: U-type
Encoding: opcode=0010111
Syntax: auipc rd, imm
Semantics: rd = PC + sign_extend(imm[31:12] << 12)
Example: auipc x5, 0x1000  // x5 = PC + 0x1000000
Use: Position-independent code, accessing data relative to PC
```

#### System Operations

**ECALL** (Environment Call)
```
Format: I-type (special)
Encoding: imm=0, funct3=000, opcode=1110011
Syntax: ecall
Semantics: Trap to operating system (syscall)
Use: System calls in Jolt (I/O operations)
```

**EBREAK** (Environment Break)
```
Format: I-type (special)
Encoding: imm=1, funct3=000, opcode=1110011
Syntax: ebreak
Semantics: Trap to debugger
Use: Breakpoints, assertions
```

### RV64M: Multiply/Divide Extension

**MUL** (Multiply Lower)
```
Format: R-type
Encoding: funct7=0000001, funct3=000, opcode=0110011
Syntax: mul rd, rs1, rs2
Semantics: rd = (rs1 * rs2)[63:0]  // Lower 64 bits of product
Example: mul x5, x3, x4
Cycles: Typically 3-5 cycles (vs 1 for add)
```

**MULH** (Multiply High Signed)
```
Format: R-type
Encoding: funct7=0000001, funct3=001, opcode=0110011
Syntax: mulh rd, rs1, rs2
Semantics: rd = (rs1 * rs2)[127:64]  // Upper 64 bits, both signed
Example: mulh x5, x3, x4
Use: Detecting overflow in multiplication
```

**MULHU** (Multiply High Unsigned)
```
Format: R-type
Encoding: funct7=0000001, funct3=011, opcode=0110011
Syntax: mulhu rd, rs1, rs2
Semantics: rd = (rs1 * rs2)[127:64]  // Upper 64 bits, both unsigned
Example: mulhu x5, x3, x4
```

**MULHSU** (Multiply High Signed-Unsigned)
```
Format: R-type
Encoding: funct7=0000001, funct3=010, opcode=0110011
Syntax: mulhsu rd, rs1, rs2
Semantics: rd = (rs1 * rs2)[127:64]  // Upper 64 bits, rs1 signed, rs2 unsigned
Example: mulhsu x5, x3, x4
```

**DIV** (Divide Signed)
```
Format: R-type
Encoding: funct7=0000001, funct3=100, opcode=0110011
Syntax: div rd, rs1, rs2
Semantics: rd = rs1 / rs2  // Signed division, round toward zero
           if rs2 == 0: rd = -1
Example: div x5, x3, x4
Cycles: Typically 30-40 cycles (very expensive)
Special: Division by zero returns -1 (no exception)
         Overflow (MIN / -1) returns MIN
```

**DIVU** (Divide Unsigned)
```
Format: R-type
Encoding: funct7=0000001, funct3=101, opcode=0110011
Syntax: divu rd, rs1, rs2
Semantics: rd = rs1 / rs2  // Unsigned division
           if rs2 == 0: rd = 2^64 - 1
Example: divu x5, x3, x4
```

**REM** (Remainder Signed)
```
Format: R-type
Encoding: funct7=0000001, funct3=110, opcode=0110011
Syntax: rem rd, rs1, rs2
Semantics: rd = rs1 % rs2  // Signed remainder
           if rs2 == 0: rd = rs1
Example: rem x5, x3, x4
Note: Sign of result matches dividend (rs1)
```

**REMU** (Remainder Unsigned)
```
Format: R-type
Encoding: funct7=0000001, funct3=111, opcode=0110011
Syntax: remu rd, rs1, rs2
Semantics: rd = rs1 % rs2  // Unsigned remainder
           if rs2 == 0: rd = rs1
Example: remu x5, x3, x4
```

### Virtual Instruction Sequences in Jolt

**Problem**: Division is expensive to prove directly. Jolt expands DIV/DIVU/REM/REMU into virtual sequences using advice.

**Example: DIV expansion**:
```
// RISC-V: div x5, x10, x11  (x5 = x10 / x11)

// Expanded to:
1. VLOAD v0, ADVICE_QUOT    // Prover provides quotient guess
2. VLOAD v1, ADVICE_REM     // Prover provides remainder guess
3. MUL v2, v0, x11          // v2 = quotient * divisor
4. ADD v3, v2, v1           // v3 = quotient * divisor + remainder
5. ASSERT_EQ v3, x10        // Verify: quotient * divisor + remainder == dividend
6. SLTU v4, v1, x11         // v4 = 1 if remainder < divisor
7. ASSERT_EQ v4, 1          // Verify: remainder < divisor
8. MOV x5, v0               // Store quotient to destination
```

**Why this works**:
- Prover computes quotient externally, provides as "advice"
- Verifier checks correctness via constraints
- Constraints are cheap (MUL, ADD, comparison)
- Avoids expensive division circuit

### RV64A: Atomic Extension

**Key concept**: Atomic read-modify-write operations for concurrent programming.

**LR.D** (Load Reserved Doubleword)
```
Format: R-type (special)
Encoding: funct7=0001000, rs2=00000, funct3=011, opcode=0101111
Syntax: lr.d rd, (rs1)
Semantics: 
  rd = M[rs1]
  reserve(rs1)  // Set reservation on address
Example: lr.d x5, (x3)
Use: First half of atomic compare-and-swap
```

**SC.D** (Store Conditional Doubleword)
```
Format: R-type (special)
Encoding: funct7=0001100, funct3=011, opcode=0101111
Syntax: sc.d rd, rs2, (rs1)
Semantics:
  if reservation_valid(rs1):
    M[rs1] = rs2
    rd = 0  // Success
  else:
    rd = 1  // Failure
Example: sc.d x5, x6, (x3)
Use: Second half of atomic compare-and-swap
```

**Atomic compare-and-swap example**:
```asm
retry:
  lr.d t0, (a0)       # Load current value, set reservation
  bne t0, a1, fail    # If not expected value, fail
  sc.d t1, a2, (a0)   # Try to store new value
  bnez t1, retry      # If failed, retry
  li a0, 1            # Success
  ret
fail:
  li a0, 0            # Failure
  ret
```

**AMO operations** (Atomic Memory Operations):

**AMOADD.D** (Atomic Add Doubleword)
```
Format: R-type (special)
Encoding: funct7=0000000, funct3=011, opcode=0101111
Syntax: amoadd.d rd, rs2, (rs1)
Semantics:
  t = M[rs1]
  M[rs1] = t + rs2
  rd = t
Example: amoadd.d x5, x6, (x3)  // Atomic x5 = *x3; *x3 += x6
Use: Atomic increment, lock-free counters
```

Similar operations: **AMOSWAP**, **AMOAND**, **AMOOR**, **AMOXOR**, **AMOMAX**, **AMOMIN**

### RV64C: Compressed Extension

**Purpose**: 16-bit instructions for code density (reduce code size by ~25-30%).

**Key properties**:
- Instructions must be 2-byte aligned
- Can freely mix 32-bit and 16-bit instructions
- Compressed instructions expand to 32-bit equivalents

**Common compressed instructions**:

**C.ADDI** (Compressed Add Immediate)
```
Format: CI-type (16 bits)
Encoding: funct3=000, opcode=01
Syntax: c.addi rd, imm
Expands to: addi rd, rd, imm
Example: c.addi x5, 4  →  addi x5, x5, 4
Constraints: rd != x0, imm != 0
Size: 16 bits (vs 32 for ADDI)
```

**C.LD** (Compressed Load Doubleword)
```
Format: CL-type (16 bits)
Encoding: funct3=011, opcode=00
Syntax: c.ld rd', offset(rs1')
Expands to: ld rd, offset(rs1)
Example: c.ld x10, 8(x8)  →  ld x10, 8(x8)
Constraints: Uses only registers x8-x15 (common registers)
             offset ∈ [0, 248], multiple of 8
```

**C.SD** (Compressed Store Doubleword)
```
Format: CS-type (16 bits)
Encoding: funct3=111, opcode=00
Syntax: c.sd rs2', offset(rs1')
Expands to: sd rs2, offset(rs1)
Example: c.sd x10, 16(x8)  →  sd x10, 16(x8)
```

**C.J** (Compressed Jump)
```
Format: CJ-type (16 bits)
Encoding: funct3=101, opcode=01
Syntax: c.j offset
Expands to: jal x0, offset
Example: c.j label  →  jal x0, label
Use: Unconditional jump (no link)
```

**C.JALR** (Compressed Jump and Link Register)
```
Format: CR-type (16 bits)
Encoding: funct4=1001, opcode=10
Syntax: c.jalr rs1
Expands to: jalr x1, 0(rs1)
Example: c.jalr x5  →  jalr x1, 0(x5)
Use: Function call via register
```

**C.MV** (Compressed Move)
```
Format: CR-type (16 bits)
Encoding: funct4=1000, opcode=10
Syntax: c.mv rd, rs2
Expands to: add rd, x0, rs2
Example: c.mv x5, x6  →  add x5, x0, x6
```

### RISC-V Calling Convention (Deep Dive)

#### Function Call Sequence

**Caller responsibilities**:
1. Place arguments in registers a0-a7 (first 8 integer args)
2. Additional arguments on stack
3. Save caller-saved registers (t0-t6, a0-a7) if needed
4. Execute `call` pseudo-instruction (expands to `auipc`+`jalr` or just `jal`)

**Callee responsibilities**:
1. Save return address (ra) if function calls other functions
2. Save callee-saved registers (s0-s11) if modified
3. Allocate stack frame for local variables
4. Execute function body
5. Place return value in a0 (a1 for second return value)
6. Restore saved registers
7. Deallocate stack frame
8. Return via `ret` (pseudo-instruction, expands to `jalr x0, 0(ra)`)

#### Stack Frame Layout

**Example function**:
```rust
fn complex_function(a: u64, b: u64, c: u64) -> u64 {
    let x = a + b;
    let y = helper(x, c);  // Calls another function
    x + y
}
```

**Compiled assembly**:
```asm
complex_function:
    # Prologue
    addi sp, sp, -32      # Allocate 32-byte frame
    sd   ra, 24(sp)       # Save return address (will call helper)
    sd   s0, 16(sp)       # Save s0 (frame pointer, optional)
    sd   s1, 8(sp)        # Save s1 (will use for local var)
    addi s0, sp, 32       # Set frame pointer to previous sp

    # Body
    add  s1, a0, a1       # s1 = a + b (save in callee-saved register)
    mv   a0, s1           # First arg to helper: x
    mv   a1, a2           # Second arg to helper: c
    call helper           # Call helper (result in a0)
    add  a0, s1, a0       # a0 = x + y (return value)

    # Epilogue
    ld   s1, 8(sp)        # Restore s1
    ld   s0, 16(sp)       # Restore s0
    ld   ra, 24(sp)       # Restore return address
    addi sp, sp, 32       # Deallocate frame
    ret                   # Return (jalr x0, 0(ra))
```

**Stack frame**:
```
Higher addresses
    ↑
    │  (caller's frame)
    ├────────────────┤  ← sp before call
    │  Return addr   │  +24(sp)
    ├────────────────┤
    │  Saved s0      │  +16(sp)
    ├────────────────┤
    │  Saved s1      │  +8(sp)
    ├────────────────┤
    │  Local vars    │  +0(sp)
    └────────────────┘  ← sp after prologue (32 bytes lower)
    │
    ↓
Lower addresses
```

#### Register Usage Summary

| Registers | ABI Name | Usage | Saved by |
|-----------|----------|-------|----------|
| x0 | zero | Constant 0 | N/A |
| x1 | ra | Return address | Caller |
| x2 | sp | Stack pointer | Callee |
| x3 | gp | Global pointer | N/A |
| x4 | tp | Thread pointer | N/A |
| x5-x7 | t0-t2 | Temporaries | Caller |
| x8 | s0/fp | Saved/Frame pointer | Callee |
| x9 | s1 | Saved register | Callee |
| x10-x11 | a0-a1 | Arguments/Return | Caller |
| x12-x17 | a2-a7 | Arguments | Caller |
| x18-x27 | s2-s11 | Saved registers | Callee |
| x28-x31 | t3-t6 | Temporaries | Caller |

**Key principle**: Caller-saved registers may be clobbered by function calls. Callee-saved registers must be preserved.

### Pseudo-Instructions

RISC-V assemblers support pseudo-instructions (synthetic instructions expanded to real instructions):

| Pseudo | Expansion | Purpose |
|--------|-----------|---------|
| `nop` | `addi x0, x0, 0` | No operation |
| `li rd, imm` | `lui rd, imm[31:12]; addi rd, rd, imm[11:0]` | Load immediate |
| `mv rd, rs` | `addi rd, rs, 0` | Copy register |
| `not rd, rs` | `xori rd, rs, -1` | Bitwise NOT |
| `neg rd, rs` | `sub rd, x0, rs` | Negate |
| `j offset` | `jal x0, offset` | Unconditional jump |
| `jr rs` | `jalr x0, 0(rs)` | Jump register |
| `ret` | `jalr x0, 0(ra)` | Return from function |
| `call offset` | `auipc x1, offset[31:12]; jalr x1, offset[11:0](x1)` | Function call |

### Memory-Mapped I/O in Jolt

**Jolt's I/O mechanism**: Special memory regions trigger I/O operations.

**Memory layout**:
```
0x00000000 - 0x7FFFFFFF: Input region
0x80000000 - 0xFFFFFFFF: DRAM
0x????? - 0x?????:       Output region (implementation-specific)
```

**Reading input**:
```asm
# Read from input device at offset 0
li   t0, 0x00000000
ld   a0, 0(t0)        # Load input value
```

**Writing output**:
```asm
# Write to output device
li   t0, OUTPUT_ADDR
sd   a0, 0(t0)        # Store output value
```

**Panic mechanism**:
```asm
# Signal panic (for guest assertions)
li   t0, PANIC_ADDR
li   t1, 1
sb   t1, 0(t0)        # Write non-zero to panic location
```

**Cycle tracking** (Jolt-specific):
```rust
// Guest code
use jolt::{start_cycle_tracking, end_cycle_tracking};

start_cycle_tracking("section_name");
// Code to profile
end_cycle_tracking("section_name");
```

Compiles to RISC-V with memory-mapped tracking operations.

---

**End of Extended RISC-V Deep Dive**


---

## 0.8. Multilinear Polynomials and Multilinear Extensions (MLEs)

### 0.8.1. Why Multilinear Polynomials Matter

**Context**: In zkVM systems like Jolt, execution traces (sequences of register states, memory operations, instruction executions) are represented as polynomials over finite fields. The sumcheck protocol, which is the verification engine of Jolt, operates on multilinear polynomials.

**Key insight**: Any vector of data can be uniquely represented as a multilinear polynomial. This representation enables efficient cryptographic verification via the sumcheck protocol.

**Definition (Multilinearity)**: A polynomial $p(x_1, x_2, \ldots, x_n)$ over field $\mathbb{F}$ is **multilinear** if it has degree at most 1 in each variable individually.

**Examples**:
- **Multilinear**: $p(x_1, x_2) = 3x_1 + 2x_2 + 5x_1x_2 + 7$
  - Degree in $x_1$: 1
  - Degree in $x_2$: 1
  - Total degree: 2 (can be higher than number of variables)

- **NOT multilinear**: $q(x_1, x_2) = x_1^2 + x_2$
  - Degree in $x_1$: 2 (violates multilinearity)

- **Multilinear in 3 variables**: $r(x_1, x_2, x_3) = x_1 + x_2x_3 + x_1x_2x_3$
  - Each variable appears with degree $\leq 1$ in each term

**Number of terms**: A multilinear polynomial in $n$ variables has at most $2^n$ terms (one for each subset of variables).

**General form**:
$$
p(x_1, \ldots, x_n) = \sum_{S \subseteq [n]} c_S \prod_{i \in S} x_i
$$

where $[n] = \{1, 2, \ldots, n\}$ and $c_S \in \mathbb{F}$ are coefficients.

### 0.8.2. The Boolean Hypercube

**Definition**: The $n$-dimensional **Boolean hypercube** is the set:
$$
\{0, 1\}^n = \{(b_1, b_2, \ldots, b_n) : b_i \in \{0, 1\}\}
$$

**Size**: Contains exactly $2^n$ points.

**Examples**:
- $n=1$: $\{0, 1\}^1 = \{0, 1\}$ (2 points, a line segment)
- $n=2$: $\{0, 1\}^2 = \{(0,0), (0,1), (1,0), (1,1)\}$ (4 corners of a square)
- $n=3$: $\{0, 1\}^3 =$ 8 corners of a cube

**Visualization for $n=2$**:
```
     (0,1) -------- (1,1)
       |              |
       |              |
       |              |
     (0,0) -------- (1,0)
```

**Visualization for $n=3$**:
```
        (0,1,1) -------- (1,1,1)
         /|               /|
        / |              / |
   (0,0,1)----------(1,0,1)|
       |  |             |  |
       |(0,1,0)---------|-(1,1,0)
       | /              | /
       |/               |/
   (0,0,0)----------(1,0,0)
```

**Connection to data**: If you have a vector of $N = 2^n$ values, you can index them by points on the Boolean hypercube.

**Example** (vector of 4 values):
$$
\text{Data: } [v_0, v_1, v_2, v_3]
$$

Index by $\{0,1\}^2$:
- $(0,0) \to v_0$
- $(0,1) \to v_1$
- $(1,0) \to v_2$
- $(1,1) \to v_3$

This indexing is the key to multilinear extensions.

### 0.8.3. Multilinear Extensions (MLEs)

**Problem**: Given a function $f: \{0,1\}^n \to \mathbb{F}$, we want a polynomial $\tilde{f}(x_1, \ldots, x_n)$ that:
1. Agrees with $f$ on the Boolean hypercube: $\tilde{f}(b) = f(b)$ for all $b \in \{0,1\}^n$
2. Is multilinear
3. Extends $f$ to the entire field: $\tilde{f}: \mathbb{F}^n \to \mathbb{F}$

**Theorem (Uniqueness)**: There exists a unique multilinear polynomial $\tilde{f}$ satisfying properties 1 and 2.

**Definition**: The **multilinear extension (MLE)** of $f: \{0,1\}^n \to \mathbb{F}$ is the unique multilinear polynomial $\tilde{f}$ agreeing with $f$ on $\{0,1\}^n$.

**Lagrange basis formula**: The MLE can be written explicitly:
$$
\tilde{f}(x_1, \ldots, x_n) = \sum_{b \in \{0,1\}^n} f(b) \cdot \chi_b(x_1, \ldots, x_n)
$$

where $\chi_b$ is the **Lagrange basis polynomial** for point $b = (b_1, \ldots, b_n)$:
$$
\chi_b(x_1, \ldots, x_n) = \prod_{i=1}^n \left( b_i x_i + (1 - b_i)(1 - x_i) \right)
$$

**Key property**: $\chi_b$ is 1 at point $b$ and 0 at all other Boolean points:
$$
\chi_b(b') = \begin{cases}
1 & \text{if } b' = b \\
0 & \text{if } b' \neq b, b' \in \{0,1\}^n
\end{cases}
$$

**Example 1** ($n=1$, vector $[v_0, v_1]$):

Function $f$:
- $f(0) = v_0$
- $f(1) = v_1$

Lagrange basis:
- $\chi_0(x) = (1-x)$
  - Check: $\chi_0(0) = 1$, $\chi_0(1) = 0$ ✓
- $\chi_1(x) = x$
  - Check: $\chi_1(0) = 0$, $\chi_1(1) = 1$ ✓

MLE:
$$
\tilde{f}(x) = v_0 \cdot (1-x) + v_1 \cdot x = v_0 + (v_1 - v_0) x
$$

This is the linear interpolation between $v_0$ and $v_1$.

**Verification**:
- $\tilde{f}(0) = v_0 + (v_1 - v_0) \cdot 0 = v_0$ ✓
- $\tilde{f}(1) = v_0 + (v_1 - v_0) \cdot 1 = v_1$ ✓

**Example 2** ($n=2$, vector $[v_{00}, v_{01}, v_{10}, v_{11}]$):

Function $f$:
- $f(0,0) = v_{00}$
- $f(0,1) = v_{01}$
- $f(1,0) = v_{10}$
- $f(1,1) = v_{11}$

Lagrange basis polynomials:
- $\chi_{(0,0)}(x_1, x_2) = (1-x_1)(1-x_2)$
- $\chi_{(0,1)}(x_1, x_2) = (1-x_1)x_2$
- $\chi_{(1,0)}(x_1, x_2) = x_1(1-x_2)$
- $\chi_{(1,1)}(x_1, x_2) = x_1 x_2$

MLE:
$$
\begin{align}
\tilde{f}(x_1, x_2) &= v_{00} \cdot (1-x_1)(1-x_2) + v_{01} \cdot (1-x_1)x_2 \\
&\quad + v_{10} \cdot x_1(1-x_2) + v_{11} \cdot x_1 x_2
\end{align}
$$

This is **bilinear interpolation** over the unit square.

**Expanded form**:
$$
\begin{align}
\tilde{f}(x_1, x_2) &= v_{00} + (v_{10} - v_{00})x_1 + (v_{01} - v_{00})x_2 \\
&\quad + (v_{00} - v_{01} - v_{10} + v_{11})x_1 x_2
\end{align}
$$

**Concrete example** (field $\mathbb{F}_7$, vector $[3, 5, 2, 4]$):
$$
\tilde{f}(x_1, x_2) = 3 + (2-3)x_1 + (5-3)x_2 + (3-5-2+4)x_1x_2
$$
$$
= 3 - x_1 + 2x_2 + 0 \cdot x_1x_2 = 3 - x_1 + 2x_2
$$

**Verification**:
- $\tilde{f}(0,0) = 3 - 0 + 0 = 3$ ✓
- $\tilde{f}(0,1) = 3 - 0 + 2 = 5$ ✓
- $\tilde{f}(1,0) = 3 - 1 + 0 = 2$ ✓
- $\tilde{f}(1,1) = 3 - 1 + 2 = 4$ ✓

**Evaluate at non-Boolean point**: $\tilde{f}(2, 3) = 3 - 2 + 2 \cdot 3 = 3 - 2 + 6 = 7 \equiv 0 \pmod{7}$

### 0.8.4. The Equality Polynomial

**Definition**: The **equality polynomial** $\text{eq}(x, y)$ for $x, y \in \mathbb{F}^n$ is:
$$
\text{eq}(x, y) = \prod_{i=1}^n (x_i y_i + (1-x_i)(1-y_i))
$$

**Key property**: For Boolean inputs:
$$
\text{eq}(b, b') = \begin{cases}
1 & \text{if } b = b' \\
0 & \text{if } b \neq b'
\end{cases}
$$

**Observation**: The Lagrange basis polynomial is the equality polynomial with one argument fixed:
$$
\chi_b(x) = \text{eq}(b, x)
$$

**Example** ($n=2$):
$$
\text{eq}((x_1, x_2), (y_1, y_2)) = (x_1 y_1 + (1-x_1)(1-y_1)) \cdot (x_2 y_2 + (1-x_2)(1-y_2))
$$

**Simplified**:
$$
\text{eq}(x, y) = (1 + x_1 - y_1 - x_1 + x_1 y_1)(1 + x_2 - y_2 - x_2 + x_2 y_2)
$$
$$
= (1 - (x_1 - y_1)^2 / \text{something...})
$$

Actually, let's expand correctly:
$$
x_1 y_1 + (1-x_1)(1-y_1) = x_1 y_1 + 1 - x_1 - y_1 + x_1 y_1 = 1 - x_1 - y_1 + 2x_1 y_1
$$

Wait, that's not quite right either. Let me recalculate:
$$
x_i y_i + (1-x_i)(1-y_i) = x_i y_i + 1 - x_i - y_i + x_i y_i = 1 - x_i - y_i + 2x_i y_i
$$

Hmm, that doesn't look right. Let me verify with Boolean values:
- $(x_i, y_i) = (0, 0)$: $0 \cdot 0 + (1-0)(1-0) = 0 + 1 = 1$ ✓
- $(x_i, y_i) = (0, 1)$: $0 \cdot 1 + (1-0)(1-1) = 0 + 0 = 0$ ✓
- $(x_i, y_i) = (1, 0)$: $1 \cdot 0 + (1-1)(1-0) = 0 + 0 = 0$ ✓
- $(x_i, y_i) = (1, 1)$: $1 \cdot 1 + (1-1)(1-1) = 1 + 0 = 1$ ✓

So the formula is correct. Let me expand it properly:
$$
x_i y_i + (1-x_i)(1-y_i) = x_i y_i + 1 - y_i - x_i + x_i y_i
$$

Wait, I need to be more careful:
$$
(1-x_i)(1-y_i) = 1 - x_i - y_i + x_i y_i
$$

So:
$$
x_i y_i + (1-x_i)(1-y_i) = x_i y_i + 1 - x_i - y_i + x_i y_i = 1 - x_i - y_i + 2x_i y_i
$$

That's still not matching the standard form. Let me check the definition again...

Actually, the standard form in many papers is:
$$
\text{eq}(x, y) = \prod_{i=1}^n (x_i y_i + (1-x_i)(1-y_i))
$$

which is exactly what I wrote. The single factor simplifies to:
$$
x_i y_i + (1-x_i)(1-y_i)
$$

Let me just verify this works for $n=1$:
- $\text{eq}(0, 0) = 0 \cdot 0 + (1-0)(1-0) = 1$ ✓
- $\text{eq}(0, 1) = 0 \cdot 1 + (1-0)(1-1) = 0$ ✓
- $\text{eq}(1, 0) = 1 \cdot 0 + (1-1)(1-0) = 0$ ✓
- $\text{eq}(1, 1) = 1 \cdot 1 + (1-1)(1-1) = 1$ ✓

Good. For $n=2$:
- $\text{eq}((0,0), (0,0)) = 1 \cdot 1 = 1$ ✓
- $\text{eq}((0,0), (0,1)) = 1 \cdot 0 = 0$ ✓
- $\text{eq}((0,0), (1,0)) = 0 \cdot 1 = 0$ ✓
- $\text{eq}((0,1), (0,1)) = 1 \cdot 1 = 1$ ✓

Perfect.

**Alternative form** (often used in implementations):
$$
\text{eq}(x, y) = \prod_{i=1}^n (1 - (x_i - y_i)^2) \quad \text{(only works in characteristic } \neq 2\text{)}
$$

But this is less common because it doesn't work over $\mathbb{F}_2$.

**Connection to MLEs**: The MLE formula can be rewritten using equality:
$$
\tilde{f}(x) = \sum_{b \in \{0,1\}^n} f(b) \cdot \text{eq}(b, x)
$$

**Why this matters**: In the sumcheck protocol, the verifier needs to evaluate the equality polynomial many times. Optimized implementations cache these evaluations.

### 0.8.5. Efficient Evaluation of MLEs

**Problem**: Given $\tilde{f}$ defined by vector $[v_0, v_1, \ldots, v_{2^n-1}]$, evaluate $\tilde{f}(r)$ for $r \in \mathbb{F}^n$.

**Naive approach**: Expand the Lagrange basis formula:
$$
\tilde{f}(r) = \sum_{b \in \{0,1\}^n} v_b \cdot \text{eq}(b, r)
$$

**Cost**: $O(2^n)$ field operations (one multiplication and addition per Boolean point).

**This is optimal**: Cannot do better than $O(2^n)$ in general, because we must touch all $2^n$ coefficients at least once.

**But**: We can optimize the constant factors and cache intermediate results.

**Recursive evaluation** (for equality polynomial):

Base case ($n=1$):
$$
\text{eq}((b_1), (r_1)) = b_1 r_1 + (1-b_1)(1-r_1)
$$

Recursive case ($n > 1$):
$$
\text{eq}((b_1, \ldots, b_n), (r_1, \ldots, r_n)) = \text{eq}((b_1), (r_1)) \cdot \text{eq}((b_2, \ldots, b_n), (r_2, \ldots, r_n))
$$

**Precomputation strategy**:
1. Compute $\text{eq}((b_1), (r_1))$ for $b_1 \in \{0, 1\}$ (2 values)
2. Compute $\text{eq}((b_1, b_2), (r_1, r_2))$ for $(b_1, b_2) \in \{0,1\}^2$ (4 values) using cached values from step 1
3. Continue for all $n$ dimensions

**Total cost**: Still $O(2^n)$, but with better cache locality.

**Example** ($n=2$, $r = (r_1, r_2)$):

Precompute:
- $e_0 = \text{eq}(0, r_1) = 1 - r_1$
- $e_1 = \text{eq}(1, r_1) = r_1$

Then:
- $\text{eq}((0,0), r) = e_0 \cdot (1 - r_2)$
- $\text{eq}((0,1), r) = e_0 \cdot r_2$
- $\text{eq}((1,0), r) = e_1 \cdot (1 - r_2)$
- $\text{eq}((1,1), r) = e_1 \cdot r_2$

**MLE evaluation**:
$$
\tilde{f}(r) = v_{00} \cdot e_0 (1-r_2) + v_{01} \cdot e_0 r_2 + v_{10} \cdot e_1 (1-r_2) + v_{11} \cdot e_1 r_2
$$

**Factored form**:
$$
\tilde{f}(r) = e_0 (v_{00}(1-r_2) + v_{01} r_2) + e_1 (v_{10}(1-r_2) + v_{11} r_2)
$$

This reduces the number of multiplications from 8 to 6.

**In Jolt**: The file `jolt-core/src/poly/eq_poly.rs` implements highly optimized equality polynomial evaluation with caching and vectorization.

### 0.8.6. Sumcheck Protocol and MLEs

**Context**: The sumcheck protocol is an interactive proof where the prover convinces the verifier that:
$$
H = \sum_{b \in \{0,1\}^n} g(b)
$$

for some polynomial $g$.

**Key requirement**: $g$ must be efficiently evaluable at any point $r \in \mathbb{F}^n$ by both prover and verifier.

**Why MLEs are perfect**: If $g$ is a product of MLEs:
$$
g(x) = \tilde{f}_1(x) \cdot \tilde{f}_2(x) \cdots \tilde{f}_k(x)
$$

then evaluating $g(r)$ reduces to evaluating each $\tilde{f}_i(r)$, which takes $O(2^n)$ per MLE.

**Example** (simplified Jolt RAM verification):

Let:
- $\tilde{a}(x)$: MLE of memory addresses accessed
- $\tilde{v}(x)$: MLE of values read/written
- $\tilde{t}(x)$: MLE of timestamps

We want to verify:
$$
\sum_{x \in \{0,1\}^n} \tilde{a}(x) \cdot \tilde{v}(x) \cdot \tilde{t}(x) = H
$$

The sumcheck protocol proceeds in $n$ rounds. In round $i$, the prover sends a univariate polynomial $s_i(X)$ claiming:
$$
s_i(X) = \sum_{b_{i+1}, \ldots, b_n \in \{0,1\}} g(r_1, \ldots, r_{i-1}, X, b_{i+1}, \ldots, b_n)
$$

where $r_1, \ldots, r_{i-1}$ are challenges from previous rounds.

**Key point**: Computing $s_i(X)$ requires evaluating $g$ at $O(2^{n-i})$ points, which is feasible because $g$ is a product of MLEs.

**Final verification**: After $n$ rounds, verifier has random point $r = (r_1, \ldots, r_n)$ and needs to evaluate $g(r)$. If each $\tilde{f}_i$ is committed via a polynomial commitment scheme, verifier can request opening proofs for $\tilde{f}_i(r)$.

**Why this matters for Jolt**: Jolt's execution trace consists of many vectors (register states, memory states, instruction opcodes). Each vector is represented as an MLE. The sumcheck protocol verifies correctness of the execution by checking polynomial identities over these MLEs.

### 0.8.7. Connection to Jolt: Registers as MLEs

**Concrete example**: Jolt's register file over 4 cycles with 4 registers.

**Execution trace** (register values at each cycle):
```
Cycle 0: [r0=0, r1=5, r2=10, r3=0]
Cycle 1: [r0=0, r1=5, r2=15, r3=0]  // r2 += 5
Cycle 2: [r0=0, r1=5, r2=15, r3=5]  // r3 = r1
Cycle 3: [r0=0, r1=0, r2=15, r3=5]  // r1 = 0
```

**Flatten to vector** (lexicographic order: cycle, then register):
$$
[0, 5, 10, 0,  0, 5, 15, 0,  0, 5, 15, 5,  0, 0, 15, 5]
$$

**Size**: 16 values = $2^4$, so we need $n=4$ variables.

**Index by Boolean hypercube**: Let $x = (x_1, x_2, x_3, x_4)$ where:
- $(x_1, x_2)$ encodes cycle number (0-3)
- $(x_3, x_4)$ encodes register number (0-3)

**Mapping**:
```
(0,0,0,0) -> cycle 0, reg 0 -> 0
(0,0,0,1) -> cycle 0, reg 1 -> 5
(0,0,1,0) -> cycle 0, reg 2 -> 10
(0,0,1,1) -> cycle 0, reg 3 -> 0
(0,1,0,0) -> cycle 1, reg 0 -> 0
...
(1,1,1,1) -> cycle 3, reg 3 -> 5
```

**MLE**: $\tilde{R}(x_1, x_2, x_3, x_4)$ is the multilinear extension of this 16-element vector.

**Semantic interpretation**: $\tilde{R}(c_1, c_2, r_1, r_2)$ interpolates register values across cycles and register indices.

**Example evaluation**: What is $\tilde{R}(0.5, 0, 0, 1)$?

This is "halfway between cycle 0 and cycle 1, at register 1."
- At cycle 0, reg 1: value 5
- At cycle 1, reg 1: value 5

So we expect $\tilde{R}(0.5, 0, 0, 1) = 5$.

Let's verify (using the formula is tedious, but the intuition is that multilinear interpolation along the first variable gives 5).

**Use in sumcheck**: Jolt's register checking sumcheck verifies:
$$
\sum_{x \in \{0,1\}^4} \tilde{R}_{\text{read}}(x) \cdot \text{eq}(x, r_{\text{cycle}}, r_{\text{reg}}) = v_{\text{claimed}}
$$

This checks that the value read from register $r_{\text{reg}}$ at cycle $r_{\text{cycle}}$ is indeed $v_{\text{claimed}}$.

### 0.8.8. Dense vs Sparse Polynomials

**Dense representation**: Store all $2^n$ coefficients explicitly.
- **Memory**: $O(2^n)$
- **Evaluation**: $O(2^n)$
- **Use case**: When most coefficients are non-zero

**Sparse representation**: Store only non-zero terms.
- **Memory**: $O(k)$ where $k$ = number of non-zero terms
- **Evaluation**: $O(k)$
- **Use case**: When $k \ll 2^n$

**Example** (sparse polynomial in 10 variables):

Suppose $g(x_1, \ldots, x_{10})$ is only non-zero at 5 points:
- $g(0, 0, 0, 0, 0, 0, 0, 0, 0, 1) = 7$
- $g(0, 0, 0, 0, 1, 0, 0, 0, 0, 0) = 3$
- $g(1, 0, 0, 0, 0, 0, 0, 0, 0, 0) = 5$
- $g(1, 1, 0, 0, 0, 0, 0, 0, 0, 0) = 2$
- $g(1, 1, 1, 1, 1, 1, 1, 1, 1, 1) = 9$

**Dense storage**: $2^{10} = 1024$ coefficients.

**Sparse storage**: 5 entries: $\{(b, g(b)) : g(b) \neq 0\}$.

**Sparse MLE**:
$$
\tilde{g}(x) = \sum_{b : g(b) \neq 0} g(b) \cdot \text{eq}(b, x)
$$
$$
= 7 \cdot \text{eq}((0,\ldots,0,1), x) + 3 \cdot \text{eq}((0,\ldots,1,0,\ldots,0), x) + \cdots
$$

**Evaluation cost**: $O(5) = O(k)$ instead of $O(1024) = O(2^n)$.

**In Jolt**: Sparse polynomials are used extensively:
- **Instruction tables**: Most CPU operations (like ADD) are only defined for specific operand patterns
- **Bytecode**: Program bytecode is sparse (most addresses are unused)
- **SPARK commitment**: Optimized polynomial commitment for sparse polynomials (used in Spartan)

**File**: `jolt-core/src/poly/sparse_mlpoly.rs`

### 0.8.9. Binding Variables: Partial Evaluation

**Definition**: **Binding** a variable means fixing it to a specific value and producing a new polynomial in fewer variables.

**Example** ($n=3$):

Let $\tilde{f}(x_1, x_2, x_3) = x_1 + 2x_2 + 3x_1 x_2 + x_3 + x_1 x_3$.

**Bind $x_1 = 5$**:
$$
\tilde{f}(5, x_2, x_3) = 5 + 2x_2 + 3 \cdot 5 \cdot x_2 + x_3 + 5 x_3
$$
$$
= 5 + (2 + 15)x_2 + (1 + 5)x_3 = 5 + 17x_2 + 6x_3
$$

This is now a polynomial in 2 variables.

**Why this matters for sumcheck**: The sumcheck protocol binds one variable per round.

**Round-by-round example** (sum over $\{0,1\}^3$):

**Initial claim**:
$$
H = \sum_{b \in \{0,1\}^3} g(b)
$$

**Round 1**: Prover sends $s_1(X)$ where:
$$
s_1(X) = \sum_{b_2, b_3 \in \{0,1\}} g(X, b_2, b_3)
$$

This is a univariate polynomial (degree $\leq d$ where $d$ is max degree of $g$).

**Verifier checks**: $s_1(0) + s_1(1) = H$, then sends random challenge $r_1 \in \mathbb{F}$.

**Round 2**: Prover sends $s_2(X)$ where:
$$
s_2(X) = \sum_{b_3 \in \{0,1\}} g(r_1, X, b_3)
$$

Note: $g(r_1, X, b_3)$ is the **binding** of $g$ with $x_1 = r_1$.

**Verifier checks**: $s_2(0) + s_2(1) = s_1(r_1)$, then sends $r_2$.

**Round 3**: Prover sends $s_3(X) = g(r_1, r_2, X)$.

**Verifier checks**: $s_3(0) + s_3(1) = s_2(r_2)$, then sends $r_3$.

**Final check**: Verifier evaluates $g(r_1, r_2, r_3)$ (either directly or via polynomial commitment opening) and checks $s_3(r_3) = g(r_1, r_2, r_3)$.

**Key insight**: Each round binds one more variable, reducing the problem size by half.

### 0.8.10. Logarithmic Proof Size

**Why MLEs enable succinct proofs**: The sumcheck protocol for $g$ over $\{0,1\}^n$ has:
- **Rounds**: $n$
- **Prover messages**: $n$ univariate polynomials of degree $\leq d$
- **Proof size**: $O(n \cdot d)$ field elements

Since $n = \log_2(N)$ where $N = 2^n$ is the vector size, proof size is **logarithmic in the data size**.

**Example** (verifying sum of 1 million values):
- $N = 10^6 \approx 2^{20}$, so $n = 20$
- If $g$ has degree 3, proof size: $20 \times 3 = 60$ field elements
- Each field element: 32 bytes (for BN254 scalar field)
- **Total proof size**: $60 \times 32 = 1920$ bytes ≈ 2 KB

Compare to sending the entire vector: $10^6 \times 32 = 32$ MB.

**This is the power of succinct proofs**: Verifying a huge computation requires only logarithmic communication.

**In Jolt**: The execution trace has $T$ cycles (typically millions). Using MLEs and sumcheck, Jolt proves correctness with proof size $O(\log T)$ per component, not $O(T)$.

### 0.8.11. Split Equality Polynomials

**Problem**: In some cases, we want to evaluate $\text{eq}(x, y)$ where $x$ and $y$ have different "regions" of variables.

**Example** (Jolt RAM verification):

We have:
- Cycle index: $x_{\text{cycle}} \in \{0,1\}^{\log T}$
- Memory address: $x_{\text{addr}} \in \{0,1\}^{\log K}$

We need to evaluate:
$$
\text{eq}((x_{\text{cycle}}, x_{\text{addr}}), (r_{\text{cycle}}, r_{\text{addr}}))
$$

**Split form**:
$$
\text{eq}((x_{\text{cycle}}, x_{\text{addr}}), (r_{\text{cycle}}, r_{\text{addr}})) = \text{eq}(x_{\text{cycle}}, r_{\text{cycle}}) \cdot \text{eq}(x_{\text{addr}}, r_{\text{addr}})
$$

**Why this helps**: We can precompute and cache the two factors separately.

**In Jolt**: The file `jolt-core/src/poly/split_eq_poly.rs` implements this optimization.

**Use case**: When evaluating many MLEs indexed by $(cycle, register)$ or $(cycle, address)$, the cycle component $\text{eq}(x_{\text{cycle}}, r_{\text{cycle}})$ is shared across all evaluations.

### 0.8.12. Practical Considerations in Jolt

**1. Field choice**: Jolt uses the BN254 scalar field $\mathbb{F}_r$ where:
$$
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
$$

This is a 254-bit prime. Arithmetic is done modulo $r$.

**2. Memory layout**: MLEs are stored as contiguous vectors in memory (`Vec<Fr>` in Rust).

**3. Parallelization**: MLE evaluations are embarrassingly parallel. Jolt uses Rayon for parallel sumcheck rounds.

**4. Commitment**: MLEs are committed using Dory PCS (polynomial commitment scheme). The commitment is a point on an elliptic curve (32 bytes).

**5. Batched openings**: At the end of the sumcheck DAG, many polynomials need to be opened at random points. Jolt batches these openings into a single proof (amortizes cost).

**6. Virtual vs committed polynomials**:
- **Committed**: Explicitly committed via PCS (e.g., register states, memory states)
- **Virtual**: Computed on-the-fly from other polynomials (e.g., intermediate sumcheck values)

**7. Example sizes in Jolt**:
- Fibonacci (100 iterations): ~1000 cycles, register polynomial MLE has $2^{\log 1000 + \log 64} \approx 2^{16}$ coefficients
- SHA2-chain (10 hashes): ~100k cycles, register polynomial has $2^{23}$ coefficients

### 0.8.13. Summary: Why MLEs Are Central to Jolt

**1. Natural representation**: Execution traces are vectors; MLEs are the unique polynomial extensions.

**2. Sumcheck compatibility**: MLEs compose well under multiplication and addition, enabling efficient sumcheck proofs.

**3. Efficient evaluation**: $O(2^n)$ evaluation is unavoidable, but MLEs achieve this with good constant factors.

**4. Logarithmic communication**: Proof size is $O(n) = O(\log N)$, not $O(N)$.

**5. Cryptographic commitments**: Dory PCS commits to MLEs succinctly (single elliptic curve point).

**6. Modular design**: Each Jolt component (R1CS, RAM, Registers, Instructions, Bytecode) represents its data as MLEs and uses sumcheck to verify correctness.

**The big picture**: Jolt's architecture is "MLEs + sumcheck + polynomial commitments." Understanding MLEs is prerequisite to understanding how Jolt verifies RISC-V execution in zero knowledge.

---


## 0.9. Formal Semantics and Operational Semantics

### 0.9.1. What is Formal Semantics?

**Motivation**: To prove that a transpiler (e.g., Rust verifier $\to$ Gnark circuit) is correct, we need a mathematical definition of what each program "means."

**Definition**: **Formal semantics** is a mathematical framework for precisely defining the meaning of programs.

**Three main approaches**:
1. **Operational semantics**: Defines meaning via execution steps (state transitions)
2. **Denotational semantics**: Maps programs to mathematical objects (functions, domains)
3. **Axiomatic semantics**: Defines meaning via logical assertions (Hoare logic)

**For transpilation**, operational semantics is most relevant because:
- It models step-by-step execution
- Directly corresponds to trace-based verification (as in Jolt)
- Easier to formalize correctness of instruction-by-instruction translation

### 0.9.2. Operational Semantics: Small-Step vs Big-Step

**Small-step semantics** (also called structural operational semantics):
- Defines single execution steps
- Relation: $\langle s, \sigma \rangle \to \langle s', \sigma' \rangle$
  - $s$: statement/expression
  - $\sigma$: state (memory, registers)
  - $\to$: transition relation
- Read as: "In state $\sigma$, executing $s$ takes one step to $s'$ in state $\sigma'$"

**Big-step semantics** (also called natural semantics):
- Defines complete evaluation
- Relation: $\langle s, \sigma \rangle \Downarrow \langle v, \sigma' \rangle$
- Read as: "In state $\sigma$, executing $s$ terminates with result $v$ and final state $\sigma'$"

**For zkVMs**: Small-step semantics is more natural because we model cycle-by-cycle execution.

### 0.9.3. Defining a Toy Language: IMP

**IMP** is a minimal imperative language used in formal semantics courses.

**Syntax**:
```
Arithmetic expressions:  a ::= n | x | a1 + a2 | a1 - a2 | a1 * a2
Boolean expressions:     b ::= true | false | a1 = a2 | a1 <= a2 | !b | b1 && b2
Commands:                c ::= skip | x := a | c1; c2 | if b then c1 else c2 | while b do c
```

**State**: A function $\sigma: \text{Var} \to \mathbb{Z}$ mapping variables to integers.

**Notation**: $\sigma[x \mapsto v]$ is state $\sigma$ updated so that $x$ maps to $v$:
$$
\sigma[x \mapsto v](y) = \begin{cases}
v & \text{if } y = x \\
\sigma(y) & \text{if } y \neq x
\end{cases}
$$

### 0.9.4. Small-Step Semantics for Arithmetic Expressions

**Evaluation relation**: $\langle a, \sigma \rangle \to \langle a', \sigma \rangle$

**Rules**:

**(Variable lookup)**:
$$
\langle x, \sigma \rangle \to \langle n, \sigma \rangle \quad \text{where } n = \sigma(x)
$$

**(Addition - left operand)**:
$$
\frac{\langle a_1, \sigma \rangle \to \langle a_1', \sigma \rangle}{\langle a_1 + a_2, \sigma \rangle \to \langle a_1' + a_2, \sigma \rangle}
$$

**(Addition - right operand)**: (only when left is a value)
$$
\frac{\langle a_2, \sigma \rangle \to \langle a_2', \sigma \rangle}{\langle n + a_2, \sigma \rangle \to \langle n + a_2', \sigma \rangle}
$$

**(Addition - compute)**:
$$
\langle n_1 + n_2, \sigma \rangle \to \langle n_3, \sigma \rangle \quad \text{where } n_3 = n_1 + n_2
$$

**Example** (evaluate $x + (3 * y)$ where $\sigma(x) = 2, \sigma(y) = 4$):

$$
\begin{align}
&\langle x + (3 * y), \sigma \rangle \\
\to& \langle 2 + (3 * y), \sigma \rangle \quad \text{(lookup $x$)} \\
\to& \langle 2 + (3 * 4), \sigma \rangle \quad \text{(lookup $y$)} \\
\to& \langle 2 + 12, \sigma \rangle \quad \text{(compute $3 * 4$)} \\
\to& \langle 14, \sigma \rangle \quad \text{(compute $2 + 12$)}
\end{align}
$$

**Key observation**: The semantics defines a deterministic evaluation order (left-to-right in this example).

### 0.9.5. Small-Step Semantics for Commands

**Evaluation relation**: $\langle c, \sigma \rangle \to \langle c', \sigma' \rangle$

**Rules**:

**(Skip)**:
No rule! $\text{skip}$ is already a terminal state.

**(Assignment - evaluate RHS)**:
$$
\frac{\langle a, \sigma \rangle \to \langle a', \sigma \rangle}{\langle x := a, \sigma \rangle \to \langle x := a', \sigma \rangle}
$$

**(Assignment - compute)**:
$$
\langle x := n, \sigma \rangle \to \langle \text{skip}, \sigma[x \mapsto n] \rangle
$$

**(Sequence - left)**:
$$
\frac{\langle c_1, \sigma \rangle \to \langle c_1', \sigma' \rangle}{\langle c_1; c_2, \sigma \rangle \to \langle c_1'; c_2, \sigma' \rangle}
$$

**(Sequence - skip left)**:
$$
\langle \text{skip}; c_2, \sigma \rangle \to \langle c_2, \sigma \rangle
$$

**(If - evaluate condition)**:
$$
\frac{\langle b, \sigma \rangle \to \langle b', \sigma \rangle}{\langle \text{if } b \text{ then } c_1 \text{ else } c_2, \sigma \rangle \to \langle \text{if } b' \text{ then } c_1 \text{ else } c_2, \sigma \rangle}
$$

**(If - true branch)**:
$$
\langle \text{if true then } c_1 \text{ else } c_2, \sigma \rangle \to \langle c_1, \sigma \rangle
$$

**(If - false branch)**:
$$
\langle \text{if false then } c_1 \text{ else } c_2, \sigma \rangle \to \langle c_2, \sigma \rangle
$$

**(While - unfold)**:
$$
\langle \text{while } b \text{ do } c, \sigma \rangle \to \langle \text{if } b \text{ then } (c; \text{while } b \text{ do } c) \text{ else skip}, \sigma \rangle
$$

**Example** (execute $x := 3; y := x + 1$ where $\sigma_0 = \{x \mapsto 0, y \mapsto 0\}$):

$$
\begin{align}
&\langle x := 3; y := x + 1, \sigma_0 \rangle \\
\to& \langle \text{skip}; y := x + 1, \sigma_1 \rangle \quad \text{where } \sigma_1 = \sigma_0[x \mapsto 3] \\
\to& \langle y := x + 1, \sigma_1 \rangle \\
\to& \langle y := 3 + 1, \sigma_1 \rangle \quad \text{(lookup $x$ in $\sigma_1$)} \\
\to& \langle y := 4, \sigma_1 \rangle \\
\to& \langle \text{skip}, \sigma_2 \rangle \quad \text{where } \sigma_2 = \sigma_1[y \mapsto 4]
\end{align}
$$

Final state: $\sigma_2 = \{x \mapsto 3, y \mapsto 4\}$.

### 0.9.6. Transition Systems and Traces

**Transition system**: A tuple $(S, \to, S_0)$ where:
- $S$: Set of states
- $\to \subseteq S \times S$: Transition relation
- $S_0 \subseteq S$: Initial states

**For IMP**: 
- $S = \text{Cmd} \times \text{State} \cup \{\text{skip}\} \times \text{State}$
- $\to$: Defined by small-step rules
- $S_0 = \{(c, \sigma) : c \text{ is initial command}, \sigma \text{ is initial state}\}$

**Trace**: A sequence of states $s_0, s_1, s_2, \ldots, s_n$ where $s_i \to s_{i+1}$ for all $i$.

**Terminal state**: $s$ is terminal if there is no $s'$ such that $s \to s'$.

**Terminating execution**: A finite trace ending in a terminal state.

**Example** (trace for $x := 1; x := x + 1$ with $\sigma_0(x) = 0$):

$$
\begin{align}
s_0 &= \langle x := 1; x := x + 1, \{x \mapsto 0\} \rangle \\
s_1 &= \langle \text{skip}; x := x + 1, \{x \mapsto 1\} \rangle \\
s_2 &= \langle x := x + 1, \{x \mapsto 1\} \rangle \\
s_3 &= \langle x := 1 + 1, \{x \mapsto 1\} \rangle \\
s_4 &= \langle x := 2, \{x \mapsto 1\} \rangle \\
s_5 &= \langle \text{skip}, \{x \mapsto 2\} \rangle
\end{align}
$$

**Connection to Jolt**: A RISC-V execution is a trace in the RISC-V transition system. Jolt verifies that the claimed trace is valid according to the RISC-V semantics.

### 0.9.7. Semantic Equivalence and Correctness

**Definition**: Two programs $c_1$ and $c_2$ are **semantically equivalent** (written $c_1 \equiv c_2$) if for all initial states $\sigma$:
- If $\langle c_1, \sigma \rangle \to^* \langle \text{skip}, \sigma_1 \rangle$, then $\langle c_2, \sigma \rangle \to^* \langle \text{skip}, \sigma_2 \rangle$ and $\sigma_1 = \sigma_2$
- Vice versa

where $\to^*$ denotes zero or more transitions (reflexive transitive closure).

**Transpiler correctness**: A transpiler from language $L_1$ to $L_2$ is **correct** if for all programs $p_1 \in L_1$:
$$
p_1 \equiv \text{transpile}(p_1)
$$

where equivalence is defined with respect to some semantic mapping between $L_1$ and $L_2$ states.

**Example** (simple optimization):

Original: $x := 2; y := x * 5$

Optimized: $x := 2; y := 10$

These are **semantically equivalent** assuming $x$ is not used after the assignment to $y$.

**Proving equivalence**: Requires showing that for all initial states, both programs produce the same final state. This can be done via:
1. **Simulation**: Show that every step of $p_1$ corresponds to zero or more steps of $p_2$
2. **Bisimulation**: Stronger notion where steps correspond in both directions
3. **Denotational equivalence**: Show both programs denote the same mathematical function

### 0.9.8. RISC-V Formal Semantics

**RISC-V state**: Tuple $\sigma = (PC, R, M)$ where:
- $PC \in \mathbb{N}$: Program counter
- $R: \text{RegId} \to \mathbb{Z}_{2^{64}}$: Register file (32 registers x0-x31)
- $M: \mathbb{N} \to \mathbb{Z}_{2^8}$: Memory (byte-addressable)

**Instruction semantics**: For each instruction opcode, define transition rule.

**Example** (ADD instruction):

**Encoding**: `ADD rd, rs1, rs2` (R-type)

**Semantics**:
$$
\frac{\begin{array}{c}
I(PC) = \text{ADD}(rd, rs1, rs2) \\
v_1 = R(rs1), \quad v_2 = R(rs2) \\
v = (v_1 + v_2) \mod 2^{64}
\end{array}}{\langle PC, R, M \rangle \to \langle PC + 4, R[rd \mapsto v], M \rangle}
$$

where $I(PC)$ is the instruction at address $PC$.

**Example** (BEQ instruction):

**Encoding**: `BEQ rs1, rs2, imm` (B-type)

**Semantics** (taken branch):
$$
\frac{\begin{array}{c}
I(PC) = \text{BEQ}(rs1, rs2, imm) \\
R(rs1) = R(rs2)
\end{array}}{\langle PC, R, M \rangle \to \langle PC + imm, R, M \rangle}
$$

**Semantics** (not taken):
$$
\frac{\begin{array}{c}
I(PC) = \text{BEQ}(rs1, rs2, imm) \\
R(rs1) \neq R(rs2)
\end{array}}{\langle PC, R, M \rangle \to \langle PC + 4, R, M \rangle}
$$

**Key point**: The formal semantics precisely defines when each instruction updates PC, registers, and memory.

**In Jolt**: The file `tracer/src/instruction/*.rs` implements these semantic rules for emulation. The zkVM proof verifies that the execution trace satisfies these rules.

### 0.9.9. Relating Two Semantics: Simulation

**Problem**: We have two languages $L_1$ and $L_2$ with semantics $\to_1$ and $\to_2$. We want to prove that a transpiler preserves behavior.

**Definition** (Forward simulation): A relation $R \subseteq S_1 \times S_2$ is a **forward simulation** if:
1. **Initial states**: If $s_1 \in S_{01}$, then there exists $s_2 \in S_{02}$ such that $(s_1, s_2) \in R$
2. **Step preservation**: If $(s_1, s_2) \in R$ and $s_1 \to_1 s_1'$, then there exists $s_2'$ such that $s_2 \to_2^* s_2'$ and $(s_1', s_2') \in R$

**Intuition**: Every step in $L_1$ can be "simulated" by zero or more steps in $L_2$, while preserving relation $R$.

**Diagram**:
```
s_1 ----> s_1'
 |         |
 R         R
 |         |
 v         v
s_2 --->* s_2'
```

**Example** (IMP $\to$ Three-Address Code):

**IMP program**: $x := a + b$

**TAC program**:
```
t1 = a
t2 = b
t3 = t1 + t2
x = t3
```

**Relation $R$**: $(s_{\text{IMP}}, s_{\text{TAC}}) \in R$ if variables in IMP state match corresponding variables in TAC state (ignoring temporaries).

**Proof sketch**:
- Initial: Both start with same variable values, $R$ holds
- Step: One IMP step $x := a + b$ simulated by 4 TAC steps
- Final: After TAC steps, $x$ has same value as after IMP step, $R$ holds

**Correctness theorem**: If $R$ is a forward simulation, then for all terminating $L_1$ executions, $L_2$ produces a related final state.

### 0.9.10. Applying to Jolt Transpilation

**Source**: Jolt verifier in Rust (`jolt-core/src/zkvm/`)

**Target**: Gnark circuit in Go

**Challenge**: These are very different semantic domains:
- Rust: Imperative, stack-based, with heap allocation
- Gnark: Declarative circuit description, no "execution"

**Approach 1** (Full rewrite): Don't formally relate semantics, just test equivalence:
- Run Rust verifier on test proofs $\to$ accept/reject
- Run Gnark circuit verifier on same proofs $\to$ accept/reject
- Check outputs match

**Pros**: Pragmatic, used in production (SP1, Risc0)
**Cons**: No formal guarantee, must test exhaustively

**Approach 2** (Extraction-based): Extract algebraic intermediate representation (IR), prove:
1. **Extraction correctness**: IR faithfully represents Rust semantics
2. **Generation correctness**: Gnark circuit implements IR semantics

**Extraction correctness**: Define relation $R_{\text{extract}}$ where:
$$
(s_{\text{Rust}}, s_{\text{IR}}) \in R_{\text{extract}} \iff \text{IR represents Rust state}
$$

Prove $R_{\text{extract}}$ is a simulation.

**Generation correctness**: Define relation $R_{\text{gen}}$ where:
$$
(s_{\text{IR}}, s_{\text{Gnark}}) \in R_{\text{gen}} \iff \text{Gnark witness satisfies IR constraints}
$$

Prove $R_{\text{gen}}$ is a simulation.

**Composition**: By transitivity, Rust $\to$ IR $\to$ Gnark preserves semantics.

### 0.9.11. Invariants and Verification Conditions

**Invariant**: A property that holds at every state in a trace.

**Example** (loop invariant):
```rust
let mut x = 0;
let mut i = 0;
while i < 10 {
    x += 2;
    i += 1;
}
// Post-condition: x = 20
```

**Loop invariant**: At start of each iteration, $x = 2i$.

**Proof**:
- **Base case**: Before loop, $i = 0, x = 0$, so $x = 2i$ ✓
- **Inductive step**: Assume $x = 2i$ at start of iteration. After loop body:
  - $x' = x + 2 = 2i + 2$
  - $i' = i + 1$
  - So $x' = 2i' $ ✓
- **Termination**: Loop exits when $i = 10$, so $x = 2 \cdot 10 = 20$ ✓

**In transpilation**: If we preserve invariants, we preserve correctness.

**Example** (sumcheck invariant):

Jolt's sumcheck has invariant: "After round $i$, the claimed sum equals $s_i(r_i)$ where $r_i$ is the verifier's challenge."

If Gnark circuit preserves this invariant, then transpiled verifier is correct.

### 0.9.12. Hoare Logic (Brief Introduction)

**Hoare triple**: $\{P\} \, c \, \{Q\}$ where:
- $P$: Precondition (predicate on states before $c$)
- $c$: Command
- $Q$: Postcondition (predicate on states after $c$)

**Meaning**: If $P$ holds before executing $c$, then $Q$ holds after $c$ (if $c$ terminates).

**Example**:
$$
\{x = 5\} \quad y := x + 1 \quad \{y = 6\}
$$

**Rules** (selection):

**(Assignment)**:
$$
\{Q[a/x]\} \quad x := a \quad \{Q\}
$$

where $Q[a/x]$ means "substitute $a$ for $x$ in $Q$."

**Example**:
$$
\{x + 1 = 6\} \quad y := x + 1 \quad \{y = 6\}
$$

Precondition simplifies to $\{x = 5\}$.

**(Sequence)**:
$$
\frac{\{P\} \, c_1 \, \{Q\} \quad \{Q\} \, c_2 \, \{R\}}{\{P\} \, c_1; c_2 \, \{R\}}
$$

**(If)**:
$$
\frac{\{P \land b\} \, c_1 \, \{Q\} \quad \{P \land \neg b\} \, c_2 \, \{Q\}}{\{P\} \, \text{if } b \text{ then } c_1 \text{ else } c_2 \, \{Q\}}
$$

**(While)**:
$$
\frac{\{I \land b\} \, c \, \{I\}}{\{I\} \, \text{while } b \text{ do } c \, \{I \land \neg b\}}
$$

where $I$ is the loop invariant.

**Use in transpilation**: If we can prove Hoare triples for the source program, and the transpiled program satisfies the same triples, then transpilation is correct.

### 0.9.13. Encoding Circuits as Predicates

**Key insight**: An R1CS constraint system can be viewed as a predicate on witness vectors.

**R1CS**: $(A, B, C, \mathbf{z})$ satisfies:
$$
(A\mathbf{z}) \circ (B\mathbf{z}) = C\mathbf{z}
$$

This is a predicate: $\text{Valid}(\mathbf{z}) \iff (A\mathbf{z}) \circ (B\mathbf{z}) = C\mathbf{z}$.

**Circuit as postcondition**: If we view circuit generation as a transformation:
$$
\text{Input: Jolt proof} \to \text{Output: Witness } \mathbf{z}
$$

then correctness is:
$$
\{\text{ValidJoltProof}(p)\} \quad \mathbf{z} := \text{GenerateWitness}(p) \quad \{\text{Valid}(\mathbf{z})\}
$$

**This connects imperative semantics (Jolt verifier) to declarative semantics (R1CS)**.

### 0.9.14. Tracing Semantics and Dynamic Analysis

**Tracing semantics**: Instead of defining transitions symbolically, execute program and record trace.

**Trace**: Sequence of concrete states $(\sigma_0, \sigma_1, \ldots, \sigma_n)$.

**Example** (Fibonacci):
```
Cycle 0: {PC=0x1000, x1=0, x2=1}
Cycle 1: {PC=0x1004, x1=1, x2=1}
Cycle 2: {PC=0x1008, x1=1, x2=2}
Cycle 3: {PC=0x100C, x1=2, x2=3}
...
```

**Jolt's approach**: Execute RISC-V program in emulator (`tracer`), record trace, then prove trace is valid according to RISC-V semantics.

**Advantage**: Don't need to symbolically analyze program, just verify concrete execution.

**Disadvantage**: Trace-specific, not a proof for all inputs.

**Solution**: User must prove the specific trace they care about (e.g., "Fibonacci(10) = 55").

### 0.9.15. Compositional Semantics

**Problem**: How do we define semantics for large programs modularly?

**Compositional semantics**: Meaning of composite program is function of meanings of parts:
$$
\text{Semantics}(c_1; c_2) = \text{Semantics}(c_1) \circ \text{Semantics}(c_2)
$$

**For transpilation**: If we extract components modularly and generate circuits compositionally, correctness follows from component correctness.

**Example** (Jolt verifier structure):

Verifier is compositional:
```
Verify(proof) = 
  VerifyR1CS(proof.r1cs) &&
  VerifyRAM(proof.ram) &&
  VerifyRegisters(proof.regs) &&
  VerifyInstructions(proof.instrs) &&
  VerifyBytecode(proof.bytecode)
```

**Transpiled verifier can mirror this structure**:
```
Circuit = 
  R1CSCircuit ||
  RAMCircuit ||
  RegistersCircuit ||
  InstructionsCircuit ||
  BytecodeCircuit
```

where $||$ denotes circuit composition (constraints combined).

**Correctness**: If each component circuit is correct, then composed circuit is correct.

### 0.9.16. Connecting to zkLean Extraction

**zkLean pattern** (from Jolt PR #1060):
1. Parse Rust verifier AST
2. Extract algebraic expressions (polynomial evaluations, sumcheck steps)
3. Generate Lean4 formal specification
4. (Future) Generate circuit from formal spec

**Semantic connection**:
- **Rust semantics**: Operational (state transitions)
- **Lean4 semantics**: Denotational (mathematical functions)
- **Circuit semantics**: Constraint satisfaction (predicates)

**Extraction as semantic translation**:
$$
\text{RustAST} \xrightarrow{\text{extract}} \text{LeanSpec} \xrightarrow{\text{codegen}} \text{GnarkCircuit}
$$

Each arrow should preserve semantics (be a simulation).

**Why formal semantics matters**: To prove correctness of this pipeline, we need precise mathematical definitions at each stage.

### 0.9.17. Summary: Formal Semantics for Transpilation

**Key takeaways**:

1. **Operational semantics** defines programs as state transition systems
2. **Traces** are sequences of states; Jolt verifies trace validity
3. **Semantic equivalence** means programs produce same final states
4. **Simulation** is the formal tool for proving transpiler correctness
5. **Invariants** are properties preserved across execution; useful for modular verification
6. **Hoare logic** connects imperative programs to logical predicates
7. **R1CS as predicate** enables viewing circuits as postconditions
8. **Compositional semantics** allows modular extraction and verification

**For Jolt-to-Groth16 transpilation**: We need to prove that the Gnark circuit accepts exactly the set of proofs that the Rust verifier accepts. This requires:
- Formalizing Rust verifier semantics (operational)
- Formalizing circuit semantics (constraint satisfaction)
- Proving simulation relation between them

**Practical approach**: Start with testing (differential testing), then gradually formalize critical components (sumcheck, polynomial evaluation).

---

