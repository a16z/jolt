# M extension

Jolt supports the RISC-V "M" extension for integer multiplication and division.
The instructions included in this extension are described [here](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html).
For RV32, the M extension includes 8 instructions: `MUL`, `MULH`, `MULHSU`, `MULU`, `DIV`, `DIVU`, `REM`, and `REMU`.

The [Jolt paper](https://eprint.iacr.org/2023/1217.pdf) describes how to handle the M extension instructions in Section 6,
but our implementation deviates from the paper in a couple ways (described below).

## Virtual sequences

Section 6.1 of the Jolt paper introduces virtual instructions and registers –– some of the M extension
instructions cannot be implemented as a single subtable decomposition, but rather must be split into
a sequence of instructions which together compute the output and places it in the destination register.
In our implementation, these sequences are captured by the `VirtualInstructionSequence` trait.

The instructions that comprise such a sequence can be a combination of "real" RISC-V instructions and "virtual"
instructions which only appear in the context of virtual sequences.
We also introduce 32 virtual registers as "scratch space" where instructions in a virtual sequence
can write intermediate values.

## Deviations from the Jolt paper

There are three inconsistencies between the virtual sequences provided in Section 6.3
of the Jolt paper, and the RISC-V specification. Namely:

1. The Jolt prover (as described in the paper) would fail to produce a valid proof
if it encountered a division by zero; since the divisor `y` is 0, the `ASSERT_LTU`/`ASSERT_LT_ABS` would
always fail (for `DIVU` and `DIV`, respectively).
1. The MLE provided for `ASSERT_LT_ABS` in Section 6.1.1 doesn't account for two's complement.
1. The `ASSERT_EQ_SIGNS` instruction should always return true if the remainder is 0.

To address these issues, our implementation of `DIVU`, `DIV`, `REMU`, and `REM` deviate from the
Jolt paper in the following ways.

### `DIVU` virtual sequence

1. `ADVICE` --, --, --, $v_q$   `// store non-deterministic advice` $q$ `into `$v_q$
1. `ADVICE` --, --, --, $v_r$   `// store non-deterministic advice` $r$ `into `$v_r$
1. `MUL` $v_q$, $r_y$, --, $v_{qy}$   `// compute q * y`
1. `ASSERT_VALID_UNSIGNED_REMAINDER` $v_r$, $r_y$, --, --   `// assert that y == 0 || r < y`
1. `ASSERT_LTE` $v_{qy}$, $r_x$, --, --   `// assert q * y <= x`
1. `ASSERT_VALID_DIV0` $r_y$, $v_q$, --, --   `// assert that y != 0 || q == 2 ** WORD_SIZE - 1`
1. `ADD` $v_{qy}$, $v_r$, --, $v_0$   `// compute q * y + r`
1. `ASSERT_EQ` $v_0$, $x$, --, --
1. `MOVE` $v_q$, --, --, `rd`

### `REMU` virtual sequence

1. `ADVICE` --, --, --, $v_q$   `// store non-deterministic advice` $q$ `into `$v_q$
1. `ADVICE` --, --, --, $v_r$   `// store non-deterministic advice` $r$ `into `$v_r$
1. `MUL` $v_q$, $r_y$, --, $v_{qy}$   `// compute q * y`
1. `ASSERT_VALID_UNSIGNED_REMAINDER` $v_r$, $r_y$, --, --   `// assert that y == 0 || r < y`
1. `ASSERT_LTE` $v_{qy}$, $r_x$, --, --   `// assert q * y <= x`
1. `ADD` $v_{qy}$, $v_r$, --, $v_0$   `// compute q * y + r`
1. `ASSERT_EQ` $v_0$, $x$, --, --
1. `MOVE` $v_r$, --, --, `rd`

### `DIV` virtual sequence

1. `ADVICE` --, --, --, $v_q$   `// store non-deterministic advice` $q$ `into `$v_q$
1. `ADVICE` --, --, --, $v_r$   `// store non-deterministic advice` $r$ `into `$v_r$
1. `ASSERT_VALID_SIGNED_REMAINDER` $v_r$, $r_y$, --, --   `// assert that r == 0 || y == 0 || (|r| < |y| && sign(r) == sign(y))`
1. `ASSERT_VALID_DIV0` $r_y$, $v_q$, --, --   `// assert that y != 0 || q == 2 ** WORD_SIZE - 1`
1. `MUL` $v_q$, $r_y$, --, $v_{qy}$   `// compute q * y`
1. `ADD` $v_{qy}$, $v_r$, --, $v_0$   `// compute q * y + r`
1. `ASSERT_EQ` $v_0$, $x$, --, --
1. `MOVE` $v_q$, --, --, `rd`

### `REM` virtual sequence

1. `ADVICE` --, --, --, $v_q$   `// store non-deterministic advice` $q$ `into `$v_q$
1. `ADVICE` --, --, --, $v_r$   `// store non-deterministic advice` $r$ `into `$v_r$
1. `ASSERT_VALID_SIGNED_REMAINDER` $v_r$, $r_y$, --, --   `// assert that r == 0 || y == 0 || (|r| < |y| && sign(r) == sign(y))`
1. `MUL` $v_q$, $r_y$, --, $v_{qy}$   `// compute q * y`
1. `ADD` $v_{qy}$, $v_r$, --, $v_0$   `// compute q * y + r`
1. `ASSERT_EQ` $v_0$, $x$, --, --
1. `MOVE` $v_r$, --, --, `rd`

## R1CS constraints

### Circuit flags

With the M extension we introduce the following circuit flags:

1. `is_virtual`: Is this instruction part of a virtual sequence?
1. `is_assert`: Is this instruction an `ASSERT_*` instruction?
1. `do_not_update_pc`: If this instruction is virtual and *not the last one in its sequence*,
then we should *not* update the PC.
This is because all instructions in virtual sequences are mapped to the same ELF address.

### Uniform constraints

The following constraints are enforced for every step of the execution trace:

1. If the instruction is a `MUL`, `MULU`, or `MULHU`, the lookup query is the product
of the two operands `x * y` (field multiplication of two 32-bit values).
1. If the instruction is a `MOV` or `MOVSIGN`, the lookup query is a single operand `x`
(read from the first source register `rs1`).
1. If the instruction is an assert, the lookup output must be true.

### Program counter constraints

Each instruction in the preprocessed [bytecode](./bytecode.md) contains its (compressed)
memory address as given by the ELF file.
This is used to compute the expected program counter for each step in the program trace.

If the `do_not_update_pc` flag is set, we constrain the next PC value to be equal to the current one.
This handles the fact that all instructions in virtual sequences are mapped to the same ELF address.

This also means we need some other mechanism to ensure that virtual sequences are executed in *order* and in *full*.
If the current instruction is virtual, we can constrain the next instruction in the trace to be the
next instruction in the bytecode.
We observe that the virtual sequences used in the M extension don't involve jumps or branches,
so this should always hold, *except* if we encounter a virtual instruction followed by a padding instruction.
But that should never happen because an execution trace should always end with some return handling,
which shouldn't involve a virtual sequence.
