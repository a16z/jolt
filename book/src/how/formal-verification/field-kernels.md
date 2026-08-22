# Formal verification of field kernels

This chapter explains what Jolt proves about the AArch64 addition and
subtraction kernels for `Prime128OffsetA7F7`. It also explains what the proof
does not cover.

The field modulus is

```text
p = 2^128 - 2^32 + 22537
  = 0xffffffffffffffffffffffff00005809
```

Every field value has a unique integer representative between `0` and
`p - 1`. The kernels accept two such representatives in two 64 bit limbs and
return the unique representative of their sum or difference modulo `p`.

## The connection from Rust to the theorem

There are four layers.

```text
public Rust operation
        |
        v
fixed AArch64 instruction body included by Rust
        |
        +------> optimized public witness is inspected byte for byte
        |
        v
standalone object built from the same instruction body
        |
        v
HOL Light theorem imports and proves those object bytes
```

The shared instruction files are the source of truth for the proved machine
code. The Rust implementation includes them with `asm!`. The standalone
objects include them with the assembler. The artifact checker compares the
object and public witness against an independent list of expected words.
Finally, HOL Light imports the standalone object and refuses to run the proof
if its words differ.

This closes a common gap in low level verification. The theorem is not about
a handwritten instruction listing that merely resembles production code. It
is about the exact words included by the production field operation.

## What the addition theorem says

For canonical inputs `m` and `n`, the addition theorem says that calling the
machine code returns `(m + n) mod p` as a canonical value.

The code first adds the two 128 bit inputs. It then conditionally adds the
small offset `2^32 - 22537`. This offset is equal to `2^128` modulo `p`.
The condition records whether the original addition crossed the 128 bit
boundary. A final conditional selection chooses the corrected or uncorrected
value.

The proof follows the carry flags produced by each instruction. It shows that
the selected value is both congruent to `m + n` modulo `p` and inside the
canonical range.

## What the subtraction theorem says

For canonical inputs `m` and `n`, the subtraction theorem says that calling
the machine code returns `(m - n) mod p` as a canonical value.

The code subtracts `n` from `m`. If this borrows, it subtracts the small offset
`2^32 - 22537` from the wrapped 128 bit result. This is the same as adding the
modulus after an ordinary negative subtraction. If there is no borrow, the
first result is already canonical.

The proof follows the borrow flag and proves both cases.

## Reading the HOL Light statements

The proof files contain two theorem levels for each operation.

The body theorem describes the arithmetic instructions before `ret`. Its
precondition fixes the program counter, input registers, and modulus offset.
Its postcondition states the result in the output registers. It also lists the
registers and condition flags that the code may change.

The subroutine theorem adds the return instruction and the normal AArch64
procedure call convention. This is the theorem that describes a callable
function.

The notation `ensures arm` means this: if the stated conditions are true before
execution, then after the modeled instructions finish, the stated result is
true. It also records which parts of machine state may have changed.

The proofs are in HOL Light rather than Lean. A proof script uses tactics to
symbolically execute instructions, derive facts about carries and borrows, and
finish the integer arithmetic. The theorem produced at the end is checked by
the small HOL Light kernel.

## Exact claim

Jolt proves the following claim on AArch64.

1. The standalone addition and subtraction objects contain the expected
   instruction words.
2. HOL Light proves those words implement canonical field addition and
   subtraction for canonical inputs.
3. The optimized public Rust witnesses contain the same words and call the
   public `Prime128OffsetA7F7` operations.

This claim covers scalar addition and subtraction only. It does not cover the
packed SIMD implementation, multiplication, the full proof system, or an
arbitrary downstream executable.

## Trust boundary

The result still relies on several components.

* The field type must maintain its canonical input invariant.
* Rust and the linker must honor the inline assembly contract.
* The HOL Light AArch64 model must match the processor.
* HOL Light, OCaml, the operating system, and the hardware are trusted to run
  the checker correctly.
* A downstream application must verify that its final optimized binary still
  contains and reaches the proved operation.

Jolt owns the theorem and the public operation witness because Jolt owns the
field kernel. Akita owns its final verifier binary, so Akita must scan that
binary at the exact Jolt revision it links. Neither repository should claim
the other half of this boundary without running its own check.

## Running the proof

Follow [`proofs/hol-light/README.md`](../../../../proofs/hol-light/README.md).
The single check command builds fresh objects and a fresh public witness,
compares their instruction words, builds both HOL Light proofs, and runs the
subroutine theorems.
