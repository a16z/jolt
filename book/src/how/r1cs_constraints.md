# R1CS constraints

Jolt uses R1CS constraints to enforce certain rules of the RISC-V fetch-decode-execute loop
and to ensure
consistency between the proofs for the different modules of Jolt ([instruction lookups](./instruction_lookups.md), [read-write memory](./read_write_memory.md), and [bytecode](./bytecode.md)).

## Uniformity

Jolt's R1CS is uniform, which means
that the constraint matrices for an entire program are just repeated copies of the constraint
matrices for a single CPU step.
Each step is conceptually simple and involves around 60 constraints and 80 variables.

## Input Variables and constraints

The inputs required for the constraint system for a single CPU step are:

#### Pertaining to bytecode
* Bytecode read address: the index in the program code read at this step.
* The preprocessed representation of the instruction: (`elf_address`, `bitflags`, `rs1`, `rs2`, `rd`, `imm`).

#### Pertaining to read-write memory
* The (starting) RAM address read by the instruction: if the instruction is not a load/store, this is 0.
* The bytes written to or read from memory.

####  Pertaining to instruction lookups
* The chunks of the instruction's operands `x` and `y`.
* The chunks of the lookup query. These are typically some combination of the operand chunks (e.g. the i-th chunk of the lookup query is often the concatenation of `x_i` and `y_i`).
* The lookup output.

### Circuit and instruction flags:
* There are twelve circuit flags (`opflags` in the Jolt paper) used in Jolt's R1CS constraints.
They are enumatered in `CircuitFlags` and computed in `to_circuit_flags` (see [`rv_trace.rs`](https://github.com/a16z/jolt/blob/main/common/src/rv_trace.rs))
Circuit flags depend only on the instruction as it appears in the bytecode, so they are computed as part of
the preprocessed bytecode in Jolt.
    1. `LeftOperandIsPC`: 1 if the first lookup operand is the program counter; 0 otherwise (first lookup operand is RS1 value).
    1. `RightOperandIsImm`: 1 if the second lookup operand is `imm`; 0 otherwise (second lookup operand is RS2 value).
    1. `Load`: 1 if the instruction is a load (i.e. `LB`, `LH`, etc.)
    1. `Store`: 1 if the instruction is a store (i.e. `SB`, `SH`, etc.)
    1. `Jump`: 1 if the instruction is a jump (i.e. `JAL`, `JALR`)
    1. `Branch`: 1 if the instruction is a branch (i.e. `BEQ`, `BNE`, etc.)
    1. `WriteLookupOutputToRD`: 1 if the lookup output is to be stored in `rd` at the end of the step.
    1. `ImmSignBit`: Used in load/store and branch instructions where the immediate value used as an offset
    1. `ConcatLookupQueryChunks`: Indicates whether the instruction performs a concat-type lookup.
    1. `Virtual`: 1 if the instruction is "virtual", as defined in Section 6.1 of the Jolt paper.
    1. `Assert`: 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    1. `DoNotUpdatePC`: Used in virtual sequences; the program counter should be the same for the full seqeuence.
* Instruction flags: these are the unary bits used to indicate instruction is executed at a given step.
There are as many per step as the number of unique instruction lookup tables in Jolt.

#### Constraint system

The constraints for a CPU step are detailed in the `uniform_constraints` and `non_uniform_constraints` functions in [`constraints.rs`](https://github.com/a16z/jolt/blob/main/jolt-core/src/r1cs/constraints.rs).

### Reusing commitments

As with most SNARK backends, Spartan requires computing a commitment to the inputs
to the constraint system.
A catch (and an optimization feature) in Jolt is that most of the inputs
are also used as inputs to proofs in the other modules. For example,
the address and values pertaining to the bytecode are used in the bytecode memory-checking proof,
and the lookup chunks, output and flags are used in the instruction lookup proof.
For Jolt to be sound, it must be ensured that the same inputs are fed to all relevant proofs.
We do this by re-using the commitments themselves.
Spartan is adapted to take pre-committed witness variables.

## Exploiting uniformity

The uniformity of the constraint system allows us to heavily optimize both the prover and verifier.
The main changes involved in making this happen are:
- Spartan is modified to only take in the constraint matrices a single step, and the total number of steps.
Using this, the prover and verifier can efficiently calculate the multilinear extensions of the full R1CS matrices.
- The commitment format of the witness values is changed to reflect uniformity.
All versions of a variable corresponding to each time step is committed together.
This affects nearly all variables committed to in Jolt.
- The inputs and witnesses are provided to the constraint system as segments.
- Additional constraints are used to enforce consistency of the state transferred between CPU steps.

These changes and their impact on the code are visible in [`spartan.rs`](https://github.com/a16z/jolt/blob/main/jolt-core/src/r1cs/spartan.rs).
