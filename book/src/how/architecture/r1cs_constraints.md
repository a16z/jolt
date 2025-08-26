# R1CS constraints

Jolt employs Rank-1 Constraint System (R1CS) constraints to enforce the state transition function of its RISC-V virtual machine and to serve as a bridge between its various [components](./architecture.md#jolts-five-components).

## Enforcing state transition

The R1CS constraints defined in `constraints.rs` include conditions ensuring that the program counter (PC) is updated correctly for each execution step.
These constraints capture:

- The "normal" case, where the PC is incremented to point to the next instruction in the bytecode

- Jump and branch instructions, where the PC may be updated according to some condition and offset

## Linking sumcheck instances

Jolt leverages Twist and Shout instances, each comprising sumchecks.
Jolt's R1CS constraints provide essential "links" between these sumchecks, enforcing relationships between different polynomials.
For example, R1CS ensures consistency between independent components such as the RAM and register Twist instances: for instructions like `LB` or `LW`, an R1CS constraint ensures the value read from RAM matches the value written into the destination register (`rd`).
This is critical because RAM and registers operate as independent Twist memory checking instances, and we need to enforce this relationship between otherwise disjoint witnesses.

## Circuit Flags (Op Flags)
Circuit flags, also referred to as operation (op) flags, are boolean indicators associated with each instruction.
These circuit flags are deterministically derived from the instruction's opcode and operands, as they appear in the bytecode.
For this reason, they are treated as "read values" (rv) in the context of the [bytecode](./bytecode.md) Shout instance.

Some examples of circuit flags:

- Jump Flag: Indicates jump instructions (JAL, JALR).

- Branch Flag: Indicates branch instructions (e.g. BEQ, BNE).

- Load Flag: Indicates load instructions (e.g. LB, LW).

- Noop Flag: Indicates virtual no-op instructions, used to pad execution traces to a power-of-two length.

These flags appear in the R1CS constraints.
For example, the "load" flag enables the aforementioned RAM-to-register constraint specifically for load instructions.
