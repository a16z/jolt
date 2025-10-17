# Jolt Virtual Instructions Security Model

## 1. Architecture

### 1.1 Core Concepts

**Virtual Instruction:**
A virtual instruction is an instruction that does not exist in the original RISC-V program bytecode. Instead, the tracer generates these instructions as replacements for certain instructions during execution.

**Instruction Expansion:**
Certain instructions does not satisfy the requirements Jolt needs for instructions, i.e. have multi-linear extension. 
These instructions are replaced with sequences of one or more virtual instructions, where each one has that property. This expansion process is deterministic and predefined, both the prover and verifier know exactly which instructions will be expanded and what virtual instruction sequences they will produce. The list of instructions that are expanded is provided in [Appendix 5](#5-appendix-expanded-instructions).

**Virtual Registers:**
The Jolt zkVM provides 96 additional registers beyond the standard 32 RISC-V registers for virtual instructions. These virtual registers are exclusively for use by virtual instructions and are not accessible to regular RISC-V instructions. Virtual registers serve as temporary storage during the execution of virtual instruction sequences.

### 1.2 Register Architecture

The Jolt zkVM implements a 128-register system with three distinct categories:

**Register Allocation:**

| Register Range | Count | Type | Purpose | Access Restrictions |
|---|---|---|---|---|
| 0-31 | 32 | RISC-V | Standard RISC-V registers (x0 through x31) | • Accessible by RISC-V instructions<br>• Virtual instructions can only access if the original instruction would have accessed them |
| 32-38 | 7 | System Level | Reserved for system-level instruction expansions (DIV, REM, MULH, etc.) | • Only accessible to system-level virtual instructions<br>• Not accessible to user-level instructions or non-virtual instructions |
| 39-127 | 89 | User Level | Available for user-level instruction expansions (SHA2, etc.) | • Only accessible to user-level virtual instructions<br>• Not accessible to system-level instructions or non-virtual instructions |el

**Register Type Definitions:**

**System-level Virtual Registers (32-38):** These registers are reserved for expanding standard RISC-V instructions that cannot have multi-linear extensions (MLE) in Jolt's constraint system. Instructions like DIV, REM, and MULH require complex operations that don't fit Jolt's MLE requirements, so they must be broken down into simpler virtual instructions that do satisfy these requirements.

**User-level Virtual Registers (39-127):** These registers are allocated for custom high-level instructions that users define for efficiency improvements for common operations like SHA2 compression. These expansions optimize performance by reducing the number of instructions in the trace while maintaining the same computational result.

### 1.3 Access Control Rules

**RISC-V Register Access:**
When a RISC-V instruction is expanded into virtual instructions, each virtual instruction in the sequence can only read from or write to the RISC-V registers that the original instruction would have accessed. For example, if the original instruction only reads x1 and x2 and writes to x3, then all virtual instructions in its expansion are restricted to (except virtual registers) these same registers with the same read or write access.

**Memory Access:**
Virtual instructions inherit the memory access permissions of their parent instruction. They can perform load and store operations, but only to memory locations that the original instruction would have been allowed to access. The memory access pattern must be equivalent to what the original instruction would have performed.

### 1.4 Instruction Expansion Constraints

**Non-Expandable Instructions:**
Certain RISC-V instructions cannot be a virtual instruction, primarily those that modify control flow or interact with the system environment. For a complete list of expandable and non-expandable instructions, see [Appendix 4](#4-appendix-non-expandable-and-allowed-instructions-for-virtual-instructions).

**Functional Equivalence Requirement:**
Every instruction expansion must maintain perfect functional equivalence with the original instruction. After executing either the original instruction or its virtual instruction expansion, the resulting state must be identical. This is equivalent to:
- All RISC-V register values must match
- Memory contents must be identical
- The program counter must advance correctly

### 1.5 System Configuration

**Virtual Register Capacity:**
The system allocates 7 virtual registers for system-level instruction expansions. This number has been determined through analysis of all current system-level expansions.

### 1.6 Additional Properties

**Recursive Expansion:**
Virtual instructions can expand recursively into other virtual instructions, creating nested expansions. For example:
- DIV expands to ~25 virtual instructions, including MULH
- MULH itself expands to additional virtual instructions  
- Each level may require temporary registers for intermediate values

**Address Consistency:**
All virtual instructions of an instruction share the same address (program counter). Each instruction has a field named `remaining_virtual_sequence` that is None for non-virtual instructions, N-1 for the first virtual instruction of an instruction with N virtual instructions, and 0 for the last virtual instruction.

### 1.7 References

The concept of Virtual Instructions was first formally introduced in Section 6.1 of the Jolt paper: https://eprint.iacr.org/2023/1217.pdf


## 2. Security Claims
### zkVM Soundness

### T1: Read/Write to Memory and Register
In terms of register and memory write, Jolt does not distinguish virtual and non-virtual instructions. So, Jolt considers a set of 128 registers, and do not differntiate memory reads/writes from virtual instructions and non-virtual instructions. The security of Twist protocol results the security of read-writes to memory and register by virtual Instructions.

### T2: Read/Write to unallowed memory/register
If we assume the bytecode is correct, and the prover/verifier checks that, this cannot happen, as it requires a change in the bytecode by a malicious prover, which the bytecode check prevents that.


### T1: Correct register read/write
Jolt treats all 128 registers uniformly and applies the Twist protocol for read-write checking. Twist's soundness guarantees register consistency.

### T2: Sequential execution of all virtual instructions
**Two enforcement mechanisms:**
1. **Dual PC tracking**: 
   - `unexpanded_pc`: Actual instruction address (R1CS constraints apply to non-virtual instructions) to ensure correct execution flow.
   - `pc`: For virtual instructions, unexpanded_pc is the same for all virtual instructions and unexpanded_pc constraints are not helpful. 
   Jolt defines a new concept named pc (which is not the standard definition of pc, that one is unexpanded_pc) that starts from 0 and increments for each instruction, including virtual instructions (check BytecodePreprocessing::pc_map)
   - Example: NOP, ADD, MULH, ADD
   - unexpanded_pc: 0x0, 0x4, 0x8, 0xC
   - pc: 0, 1, (2-8 for MULH virtual instructions), 9

2. **Circuit flags**:
   - `InlineSequenceInstruction`: True for all virtual instructions
   - `DoNotUpdateUnexpandedPC`: True for all except last virtual instruction 
   - `isFirstInSequence`: True for the first virtual instruction

The following R1CS constraint enforces the prover to proceed through the instructions one by one (preserve the order) as long as it is a virtual instruction. 
```
if InlineSequenceInstruction {
    assert!(NextPC == PC + 1)
}
```


The following constraint forces the prover not to update unexpanded_pc as long as they read a virtual instruction (except the last one, which should update the unexpanded_pc as needed for the next instruction). This constraint enforces that we do not exit early from instructions and complete them to the end. 
```
if DoNotUpdateUnexpandedPC {
    assert!(NextUnexpandedPC == UnexpandedPC)
}
```


Finally, this constraint forces the prover not to skip the first virtual instructions. This forces that if the prover next reads a virtual instruction (j+1), and if the prover is not currently reading a non-virtual instruction, or the prover is reading the last virtual instruction (j), then j+1 should be the first virtual instruction. This prevents the attack of skipping the first virtual instructions. 
```
if InlineSequenceInstruction(j+1) && !DoNotUpdateUnexpandedPC(j) {
    assert!(isFirstInSequence(j+1))
}
```

### T3: What if the trace starts with a virtual instruction and a mlicious prover skips the first few ones?
This is handled by the pc sumcheck and the fact that the first instruction should have pc=0, and the rest are constrained by R1CS.


### T4: What if recursive expansion of virtual instructions causes register collision or exhaustion?
7 registers allocated based on worst-case analysis. Expandable if needed. We keep the virtual registers at the minimum number to be able to allocate more registers for user-level instructions, in case needed. However, in case 7 is not enough, we can increase it. 
As instructions are less likely to change constantly, this number will not change frequently.

### T5: Virtual register initial value?
Virtual instructions always initialize registers before use. Uninitialized usage causes undefined behavior (bytecode bug, not soundness issue).

### T6: What if prover uses registers outside range 32-38?
This is the responsibility of both the honest prover and verifier to check the correctness of the bytecode. We can add a check in the tracer to enforce this, but violating this is a problem in the code, not a violation of soundness. 

### T7: What if virtual instructions include PC-changing operations?
Similar to T6.

### T8: What if a virtual instruction does a memory operation like load or store?
There is no problem in doing this. Memory access is checked using the Twist protocol. In terms of memory, you can consider virtual instructions as regular instructions, and memory security of them is exactly the same as other instructions.

## 3. Audit Scope

### 3.1 zkVM (Soundness-Critical)
**These issues could allow malicious provers to prove incorrect claims:**
- **1.1** Register read-write consistency verification
- **1.2** In-order execution enforcement of virtual instructions

### 3.2 Tracer (Bytecode Security)
**These issues don't break zkVM soundness but create security vulnerabilities:**
- **2.1** Functional equivalence of an instruction with its expanded virtual instructions
- **2.2** There are no PC-modifying instructions in virtual expansions
- **2.3** Register range validation (32-38 only)
- **2.4** Register allocation without collision (currently handled by `allocate_register`, and panic in case we are out of available registers)
- **2.5** `remaining_virtual_instructions` is set correctly for each virtual instructions.

### 3.3 Additional Considerations
**Advice Values**: Some instructions (e.g., DIV and REM) receive prover advice and validate the correctness of the advice. This requires careful consideration to ensure all edge cases are detected.

**Efficiency**: Some instructions expand to a large number of virtual instructions (DIV → 25 instructions). We can reduce this by introducing new instructions, but that increases the security risk and auditing effort. It would be beneficial to reduce the number of instructions with currently available instructions.

## 4. Appendix: Non-Expandable and Allowed Instructions for Virtual Instructions

To maintain the security model, certain RISC-V instructions cannot be expanded into virtual instructions, while virtual instructions themselves are restricted to specific instruction types.

### 4.1 Non-Expandable Instructions (Cannot be replaced with virtual instructions)

**Control Flow Instructions:**
- **Jumps**: `jal`, `jalr`
- **Branches**: `beq`, `bne`, `blt`, `bge`, `bltu`, `bgeu`
- **PC-relative addressing**: `auipc`

**System Instructions:**
- **Environment calls**: `ecall` (OS interaction)

### 4.2 Allowed Instructions for Virtual Instructions

Virtual instructions themselves can only use the following instruction types (straight-line sequences only):

**Computation:**
- **Arithmetic**: `addi`, `add`, `sub`, `slti`, `sltiu`, `slt`, `sltu`
- **Logical**: `xori`, `xor`, `ori`, `or`, `andi`, `and`
- **Shifts**: `slli`, `sll`, `srli`, `srl`, `srai`, `sra`
- **Immediate loading**: `lui`
- **Multiplication/Division** (M Extension): `mul`, `mulh`, `mulhsu`, `mulhu`, `div`, `divu`, `rem`, `remu`

**Memory Operations:**
- **Loads**: `lb`, `lh`, `lw`, `ld`, `lbu`, `lhu`, `lwu` (register-based addressing only)
- **Stores**: `sb`, `sh`, `sw`, `sd`

**64-bit Operations** (RV64I):
- `addiw`, `addw`, `subw`, `slliw`, `sllw`, `srliw`, `srlw`, `sraiw`, `sraw`


**Customized Instructions**:
- `XORROT`, `XORROTW`, `ROTR`, `ANDN`

## 5. Appendix: Expanded Instructions

*(To be completed - This section will contain the list of RISC-V instructions that are expanded into virtual instructions and their expansion details)*
