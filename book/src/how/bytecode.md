# Bytecode

Jolt proves the validity of registers and RAM using offline memory checking.

## Decoding an ELF file

The tracer iterates over the `.text` sections of the program ELF file and decodes RISC-V instructions. Each instruction gets mapped to a `BytecodeRow` struct: 

```rust
pub struct BytecodeRow {
    /// Memory address as read from the ELF.
    address: usize,
    /// Packed instruction/circuit flags, used for r1cs
    bitflags: u64,
    /// Index of the destination register for this instruction (0 if register is unused).
    rd: u64,
    /// Index of the first source register for this instruction (0 if register is unused).
    rs1: u64,
    /// Index of the second source register for this instruction (0 if register is unused).
    rs2: u64,
    /// "Immediate" value for this instruction (0 if unused).
    imm: u64,
}
```

The registers (`rd`, `rs1`, `rs2`) and `imm` are described in the [RISC-V spec](https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf). 
The `bitflags` field is described in greater detail [below](#bitflags).

The preprocessed bytecode serves as the (read-only) "memory" over which we perform offline memory checking. 

![bytecode_trace](../imgs/bytecode_trace.png)

We currently assume the program bytecode is sufficiently short and known by the verifier, so the prover doesn't need to provide a commitment to it –– the verifier can compute the MLE of the bytecode on its own. 
This guarantees that the bytecode trace is consistent with the agreed-upon program.

Instead of $(a, v, t)$ tuples, each step in the bytecode trace is mapped to a tuple $(a, v_\texttt{bitflags}, v_\texttt{rd}, v_\texttt{rs1}, v_\texttt{rs2}, v_\texttt{imm}, t)$.
Otherwise, the offline memory checking proceeds as described in [Offline Memory Checking](../background/memory-checking.md).


## Bitflags

The `bitflags` of a given instruction is the concatenation of its [circuit flags](./r1cs_constraints.md#circuit-and-instruction-flags) and [instruction flags](./instruction_lookups.md).
This concatenation is enforced by R1CS constraints.

![bitflags](../imgs/bitflags.png)