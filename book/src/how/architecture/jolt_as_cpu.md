# Jolt as a CPU

One way to understand Jolt is to map the components of the proof system to the components of the virtual machine (VM) whose functionality they prove.

## Jolt's four components

A VM does two things:

- Repeatedly execute the fetch-decode-execute logic of its instruction set architecture.
- Perform reads and writes to Random Access Memory (RAM).

The Jolt paper depicts these two tasks mapped to three components in the Jolt proof system:

![Jolt Alpha](../imgs/figure2.png)

The Jolt codebase is similarly organized, but instead separates read-write memory (comprising registers and RAM) from program code (aka bytecode, which is read-only), for a total of five components:

![fetch-decode-execute](../imgs/fetch_decode_execute.png)

### RAM

To handle reads/writes to RAM Jolt uses the Twist memory checking argument. TODO

*For more details: [RAM](./ram.md)*

### Registers

Similar to RAM, Jolt uses the Twist memory checking argument to handle reads/writes to registers.

*For more details: [Registers](./registers.md)*


### R1CS constraints

To handle the "fetch" part of the fetch-decode-execute loop, there is a minimal R1CS instance (about 30 constraints per cycle of the RISC-V VM). These constraints handle program counter (PC) updates and serves as the "glue" enforcing consistency between polynomials used in the components below. Jolt uses [Spartan](https://eprint.iacr.org/2019/550), optimized for the highly-structured nature of the constraint system (i.e. the same small set of constraints are applied to every cycle in the execution trace). This is implemented in [jolt-core/src/r1cs](./r1cs_constraints.md).

*For more details: [R1CS constraints](./r1cs_constraints.md)*

### Instruction execution

To handle the "execute" part of the fetch-decode-execute loop, Jolt invokes the Shout lookup argument. The lookup argument maps every instruction to its output. This is implemented in [instruction_lookups.rs](https://github.com/a16z/jolt/blob/main/jolt-core/src/jolt/vm/instruction_lookups.rs).

*For more details: [Instruction execution](./instruction_execution.md)*

### Bytecode

To handle the "decode" part of the fetch-decode-execute loop, Jolt uses another instance of Shout. The bytecode of the guest program is "decoded" in preprocessing, and the prover subsequently invokes offline memory-checking on the sequence of reads from this decoded bytecode corresponding to the execution trace being proven. This is implemented in [bytecode.rs](https://github.com/a16z/jolt/blob/main/jolt-core/src/jolt/vm/bytecode.rs).

*For more details: [Bytecode](./bytecode.md)*
