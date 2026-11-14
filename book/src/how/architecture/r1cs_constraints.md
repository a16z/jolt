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

- Jump Flag: Indicates jump instructions (JAL, JALR) and the two halts.

- Halt and HaltPanic Flags: `Halt` marks both halts (`jolt.success` and `EBREAK`), and `HaltPanic` marks the `EBREAK` halt only. See [Ending a run](#ending-a-run).

- Load Flag: Indicates load instructions (`LD`, and the `VirtualLoad*` doubleword loads that sub-word loads expand into).

- AddOperands Flag: Indicates that the instruction operands should be added "in the field", as described [above](#arithmetic-instructions) (e.g. ADD, ADDI, AUIPC).

These flags appear in the R1CS constraints.
For example, the "load" flag enables the aforementioned RAM-to-register constraint specifically for load instructions.

## Ending a run

A Jolt run ends at a halt: `jolt.success` (the custom-2 word `0x0000305B`) for success, or `EBREAK` (and `c.ebreak`) for a failure.
Each halt lowers to one bytecode row, of kind `JoltSuccess` or `JoltPanic`.
Both kinds carry the Jump flag and have no lookup table, so instruction read-RAF forces their lookup output to 0: a halt is a jump to address 0, where only padding rows live.
`JoltSuccess` also carries `Halt`, and `JoltPanic` carries `Halt` and `HaltPanic`; no other instruction carries either flag.
After the halt, the trace holds only padding no-ops up to its power-of-two length.

Two constraints implement this end rule:

- Row 14 (`NextUnexpPCEqLookupIfJump`) is $\textsf{Jump} \cdot (\textsf{NextUnexpandedPC} - \textsf{LookupOutput}) = 0$, so every jump's successor address is its lookup output, and a halt's is 0.
- Row 21 (`JumpHasSuccessorUnlessHalt`, lane 3 of Spartan product virtualization) has factors $\textsf{Jump}$ and $1 - \textsf{NextIsNoop}$, and its output depends on the statement's panic bit $\Pi$:

$$
\textsf{Jump} \cdot (1 - \textsf{NextIsNoop}) = \textsf{Jump} - (1 - 2\Pi) \cdot \textsf{Halt} - 2 \cdot \textsf{HaltPanic}
$$

$\Pi$ is the verifier's `panic` input, which enters the claim only as the derived public `SpartanProductVirtualizationPublic::ClaimedPanic`.
`PRODUCT_JUMP_LANE_OUTPUT` in `jolt-claims` owns this output; the verifier, the optimized prover kernel and BlindFold evaluate the same expression, and the reference kernel computes the lane from the trace.
$\textsf{NextIsNoop}$ is 1 when the next row is padding and at the last cycle $T-1$, so the lane reduces per row to:

| Row | (Jump, Halt, HaltPanic) | Lane 3 reduces to | Forces |
|---|---|---|---|
| Not a jump | (0, 0, 0) | $0 = 0$ | nothing; rows 15 and 16 apply |
| JAL or JALR | (1, 0, 0) | $1 - \textsf{NextIsNoop} = 1$ | a real successor, so not the last cycle |
| `jolt.success` | (1, 1, 0) | $1 - \textsf{NextIsNoop} = 2\Pi$ | $\Pi = 0$, and padding or the end follows |
| `EBREAK` | (1, 1, 1) | $1 - \textsf{NextIsNoop} = 2\Pi - 2$ | $\Pi = 1$, and padding or the end follows |

The excluded cases need a field characteristic other than 2 and 3: a success halt under $\Pi = 1$ would need $\textsf{NextIsNoop} = -1$, and an `EBREAK` halt under $\Pi = 0$ would need $\textsf{NextIsNoop} = 3$.
Both proof fields qualify.
Under either value of $\Pi$, the lane also rejects every flag combination that no instruction carries.
This form keeps lane 3's coefficients in the full-trace matrix in $\{\pm 1, -2\}$.

Only a halt can precede padding.
A padding row has address 0, and the shifted columns are 0 at the last cycle, so a real row followed by padding, or at $T-1$, has $\textsf{NextUnexpandedPC} = 0$.
If that row is not a jump, row 15 (a taken branch) or row 16 sets its $\textsf{NextUnexpandedPC}$ from its own address: its own address plus the branch offset, the next sequential address, or its own address inside an expanded sequence.
None of these is 0, because bytecode addresses are at least `RAM_START_ADDRESS` ($2^{31}$) and a branch offset is a 13-bit immediate, so the row is a jump, and lane 3 makes it a halt.
A padding row keeps the next address at 0 (row 16), so no real row follows padding.
The real rows therefore form a prefix that ends at exactly one halt: a proof with $\Pi = 0$ attests a run that ended at `jolt.success`, and one with $\Pi = 1$ a run that ended at `EBREAK`.

Two admission checks complete the rule:

- A halt is a whole source instruction. A halt as the last row of a longer expansion would end the proof in the middle of an instruction, so bytecode preprocessing rejects such a sequence (`PreprocessingError::HaltInSequence`) and expansion rejects a halt among the final rows of a multi-row sequence (`ExpansionError::HaltInSequence`).
- The entry row is a real instruction. Bytecode read-RAF pins cycle 0 to the key's entry row; if that row were padding, an all-padding trace would verify under either value of $\Pi$. Full-mode verification under such a key fails with `VerifierError::EntryNotInstruction`. A committed key carries no rows: the verifier rejects only entry index 0 and trusts the key for the rest, as it does for the committed rows themselves. The prover refuses a padding entry and a trace that is empty or does not end at a halt.
