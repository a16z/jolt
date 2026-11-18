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
- Row 21 (`NextIsNoopEqIsNoopUnlessHalt`, lane 3 of Spartan product virtualization) has factors $2$ and $\textsf{NextIsNoop} - \textsf{IsNoop}$, and its output depends on the statement's panic bit $\Pi$:

$$
2 \cdot (\textsf{NextIsNoop} - \textsf{IsNoop}) = (2 - \Pi) \cdot \textsf{Halt} + \textsf{HaltPanic}
$$

$\Pi$ is the verifier's `panic` input, which enters the claim only as the derived public `SpartanProductVirtualizationPublic::ClaimedPanic`.
`PRODUCT_HALT_LANE_OUTPUT` in `jolt-claims` owns this output; the verifier, the optimized prover kernel and BlindFold evaluate the same expression, and the reference kernel computes the lane from the trace.
$\textsf{IsNoop}$ is 1 on a padding row.
$\textsf{NextIsNoop}$ is the next row's $\textsf{IsNoop}$, and 1 at the last cycle $T-1$.
On a halt the output is $2 - \Pi + \textsf{HaltPanic}$, so the lane reduces per row to:

| Row | (IsNoop, Halt, HaltPanic) | Lane 3 reduces to | Forces |
|---|---|---|---|
| Real, not a halt | (0, 0, 0) | $2 \cdot \textsf{NextIsNoop} = 0$ | a real successor, so not the last cycle |
| Padding | (1, 0, 0) | $2 \cdot (\textsf{NextIsNoop} - 1) = 0$ | padding or the end follows |
| `jolt.success` | (0, 1, 0) | $2 \cdot \textsf{NextIsNoop} = 2 - \Pi$ | $\Pi = 0$, and padding or the end follows |
| `EBREAK` | (0, 1, 1) | $2 \cdot \textsf{NextIsNoop} = 3 - \Pi$ | $\Pi = 1$, and padding or the end follows |

No other assignment of the four cells satisfies the lane.
The left side is $-2$, $0$ or $2$, and the output is 0 only on a row with neither flag and 2 only on a halt whose `HaltPanic` equals $\Pi$; a halt of the other kind, and `HaltPanic` without `Halt`, give 1 or 3.
In particular a padding row that carries a halt flag fails, whatever follows it.
This needs a field characteristic other than 2, 3 and 5, which keeps $-2$, $0$, $1$, $2$ and $3$ distinct; both proof fields qualify.
The lane does not read `Jump`, so it treats `Halt` without `Jump` as a halt; no instruction kind carries that combination.
Bytecode read-RAF binds each cycle's flags to its bytecode row's: the verifier computes them from the row's kind for a full program, and trusts the committed flag lanes of a committed one, as for every flag.

Only a halt can precede padding, and only padding can follow it.
Bytecode read-RAF pins cycle 0 to the entry row, which is real.
A real row that is not a halt has a real successor, a halt of the claimed kind has a padding successor, a padding row has a padding successor, and the last cycle is a halt or padding.
The real rows therefore form a prefix that ends at exactly one halt: a proof with $\Pi = 0$ attests a run that ended at `jolt.success`, and one with $\Pi = 1$ a run that ended at `EBREAK`.
The lane reads no address, so this holds wherever the code is loaded and whatever a branch offset or a jump target is.

Two admission checks complete the rule:

- A halt is a whole source instruction. A halt as the last row of a longer expansion would end the proof in the middle of an instruction, so bytecode preprocessing rejects such a sequence (`PreprocessingError::HaltInSequence`) and expansion rejects a halt among the final rows of a multi-row sequence (`ExpansionError::HaltInSequence`).
- The entry row is a real instruction. Bytecode read-RAF pins cycle 0 to the key's entry row; if that row were padding, an all-padding trace would verify under either value of $\Pi$. Full-mode verification under such a key fails with `VerifierError::EntryNotInstruction`. A committed key carries no rows: its builder refuses an entry that is not an instruction before committing, and the verifier rejects only entry index 0 and trusts the key for the rest, as it does for the committed rows themselves. The prover refuses a padding entry and a trace that is empty or does not end at a halt.
