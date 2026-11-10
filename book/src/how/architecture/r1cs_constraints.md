# R1CS constraints

Jolt employs Rank-1 Constraint System (R1CS) constraints to enforce the state transition function of its RISC-V virtual machine and to serve as a bridge between its various [components](./architecture.md#jolts-five-components).
Jolt uses [Spartan](./spartan.md) to prove its R1CS constraints.

## Enforcing state transition

The R1CS constraints defined in `jolt-claims/src/protocols/composed/r1cs/rv64.rs` include conditions ensuring that the program counter (PC) is updated correctly for each execution step.
These constraints capture:

- The "normal" case, where the PC is incremented to point to the next instruction in the bytecode

- Jump and branch instructions, where the PC may be updated according to some condition and offset

## Linking sumcheck instances

Jolt leverages Twist and Shout instances, each comprising sumchecks.
Jolt's R1CS constraints provide essential "links" between these sumchecks, enforcing relationships between different polynomials.
For example, R1CS ensures consistency between independent components such as the RAM and register Twist instances: for load instructions such as `LD`, an R1CS constraint ensures the value read from RAM matches the value written into the destination register (`rd`).
This is critical because RAM and registers operate as independent Twist memory checking instances, and we need to enforce this relationship between otherwise disjoint witnesses.

## Arithmetic instructions

While Jolt's design uniquely leverages lookups for most RISC-V instructions, arithmetic instructions are the exception to this rule.
Most arithmetic instructions (addition, subtraction, multiplication) are primarily constrained using R1CS constraints, with the lookup only serving to truncate potential overflow bits.

For these instructions, we can emulate the 64-bit arithmetic using native field arithmetic, since Jolt's proof field is big enough to perform these operations without overflow. The field is the BN254 scalar field with [Dory](../dory.md), or a 128-bit prime field with [Akita](../akita.md).
E.g. to add two 64-bit numbers `x` and `y`, we can add their $\mathbb{F}_r$ equivalents $x$ and $y$, knowing that $x + y \in \mathbb{F}_r$ will be equivalent to the desired 65-bit sum.
We then employ a range-check lookup to truncate the potential overflow bit, thus matching the behavior of the RISC-V spec.

## Circuit Flags
Circuit flags, also referred to as operation (op) flags, are boolean indicators associated with each instruction.
These circuit flags are deterministically derived from the instruction's opcode and operands as they appear in the bytecode.
For this reason, they are treated as "read values" (rv) in the context of the [bytecode](./bytecode.md) Shout instance.

Some examples of circuit flags:

- Jump Flag: Indicates jump instructions (JAL, JALR).

- Load Flag: Indicates load instructions (`LD`, and the `VirtualLoad*` doubleword loads that sub-word loads expand into).

- AddOperands Flag: Indicates that the instruction operands should be added "in the field", as described [above](#arithmetic-instructions) (e.g. ADD, ADDI, AUIPC).

These flags appear in the R1CS constraints.
For example, the "load" flag enables the aforementioned RAM-to-register constraint specifically for load instructions.

## Stopping at a repeatable self-jump

A real trace may end only after a jump that targets its own source PC and
repeats that target when executed again. The bytecode-authenticated `CanEnd`
flag records whether the source instruction can do this. Direct JAL rows set
it. Direct JALR rows use the source instruction length and the static
`self_jump_repeats()` predicate: a JALR that overwrites its own base register
must still target its PC after writing the link. Generated rows carry the
source classification, and only the terminal row of an eligible expansion
can set the flag. EBREAK's complete JAL+0 expansion is eligible.

With `H = Jump - ShouldJump`, the outer R1CS includes

```text
H * (LookupOutput - UnexpandedPC + 2^64 * (1 - CanEnd)) = 0.
```

At a transition to padding, `ShouldJump` is zero. The same holds at the
last domain index, where the shift relation treats the missing successor as
a no-op. The target and source PC are 64-bit words, `CanEnd` is authenticated
as a Boolean bytecode flag, and the field characteristic exceeds `2^65`.
These bounds make the row enforce both `CanEnd = 1` and target equality.
An unstable self-targeting JALR cannot end the trace.

The prover pads to `max(256, next_power_of_two(executed_rows))`. A real
terminal row can occupy the last domain index without an extra padding row
or a verifier-side boundary assertion.
