# Reading a machine theorem

This page reads one HOL Light machine theorem as an engineering contract. You
do not need prior HOL Light knowledge.

## The shape of the claim

The Fp128 x86-64 addition theorem has this simplified shape.

```ocaml
!a0 a1 b0 b1 pc.
  ensures x86
    (\s. bytes_loaded s (word pc) code /\
         read RIP s = word pc /\
         read RDI s = a0 /\ read RSI s = a1 /\
         read RDX s = b0 /\ read RCX s = b1)
    (\s. read RIP s = word end_pc /\
         (value a0 a1 < p /\ value b0 b1 < p
          ==> value (read RAX s) (read RDX s) =
              (value a0 a1 + value b0 b1) MOD p))
    (MAYCHANGE registers ,, MAYCHANGE flags ,, MAYCHANGE events)
```

The actual theorem uses library names for each concept. The structure above is
the same.

The Fp64 theorem has the same structure with one word for each input and one
word for the result. Its canonical precondition is `val a < 2^64 - 59` and
`val b < 2^64 - 59`.

## Universal inputs

The first line starts with `!`.

```ocaml
!a0 a1 b0 b1 pc.
```

This means the theorem applies to every choice of these values. `a0` and `a1`
are the low and high 64 bit limbs of the first input. `b0` and `b1` are the
limbs of the second input. `pc` is the address where the code is loaded.

The proof does not enumerate test cases. The theorem variables stand for all
possible 64 bit words.

## Initial machine state

The first function passed to `ensures x86` is the precondition.

```ocaml
bytes_loaded s (word pc) code
```

This fixes the exact instruction bytes in memory at `pc`.

```ocaml
read RIP s = word pc
```

This says execution starts at the first instruction.

The register equations place the four input limbs in the physical registers
used by the object. The values are variables, but the register names are fixed
because register numbers are encoded in x86 instructions.

## Final machine state

The second function passed to `ensures x86` is the postcondition. It states
where execution stops and what result registers contain.

For addition, the arithmetic part is an implication.

```text
if a < p and b < p, then result = (a + b) mod p
```

This is the canonical input rule. If either input is not canonical, the theorem
still proves that execution reaches the end while staying inside the frame
condition. It makes no arithmetic claim about that result.

The field type must therefore preserve canonical values. Checked decoding and
safe constructors enforce that rule in Rust. Internal field operations are
designed and tested to preserve it. That type invariant is not yet a HOL Light
theorem about the whole Rust program.

The right hand side uses `MOD p`, so it is in the range from zero through
`p - 1`. Equality to that expression proves both modular correctness and a
canonical result.

## The frame condition

The third argument lists all state that may change.

```ocaml
MAYCHANGE [RIP; RAX; ...] ,,
MAYCHANGE SOME_FLAGS ,,
MAYCHANGE [events]
```

This is the theorem version of a changed register declaration. State not named
by the frame must remain unchanged.

The `events` component records externally visible processor events in the
model. Allowing it to change does not prove a side channel claim. It lets the
functional theorem focus on the arithmetic result and ordinary machine state.

## Body theorem and subroutine theorem

Each kernel has two theorem levels.

The body theorem starts at the first instruction and stops before `ret`. It
states the arithmetic result and the exact state that the body may change.

The subroutine theorem adds the procedure call convention. On x86-64 it states
that `ret` reads the return address from the stack, increases `rsp` by eight
bytes, and transfers control to that address. It permits only the registers and
flags that the System V convention allows a callee to change.

The BMI2 and ADX multiplication body already forms the result in `rax:rdx`.
Its subroutine theorem therefore covers the complete object with no result copy
between the proved arithmetic and the procedure return.

## How the instruction proof works

The proof first symbolically executes the exact bytes. The x86 model produces
equations for register results and flags.

For an addition with an incoming carry, one equation has this form.

```text
2^64 * carry_out + output_word
    = left_word + right_word + carry_in
```

For multiplication, the model gives low and high output words whose combined
value equals the product of the input words.

`adcx` and `adox` use separate carry flags. `adcx` reads and writes the carry
flag. `adox` reads and writes the overflow flag. `mulx` leaves both flags
unchanged. The optimized multiplication proof follows both chains and proves
that the last carry from each chain is included in the top product limb.

The proof then combines the word equations into integer equations. It proves
the field result in stages.

```text
instruction equations
    -> exact 256 bit product
    -> first Solinas fold
    -> bound on the remaining high limb
    -> second Solinas fold
    -> one final canonical correction
```

The proof never samples an input. Each algebra step uses the variables from the
theorem statement.

## Tactics and the HOL Light kernel

A tactic is an OCaml program that helps construct a proof. Tactics can be large.
A tactic cannot create an accepted theorem on its own.

HOL Light represents a proved statement as a theorem value. A small logical
kernel checks the primitive inference steps that create that value. If a tactic
has a bug, it should fail, take too long, or build a theorem different from the
one requested. It cannot bypass the abstract theorem type through the normal
HOL Light interface.

We still trust the HOL Light kernel, its OCaml runtime, and the definitions of
the x86 and AArch64 instructions. We also trust the host hardware and operating
system that run the checker. The trust boundary page lists these assumptions in
one place.

## What to review in a theorem

A reviewer should read the statement before reading tactics.

Check these questions.

1. Does the byte array name the intended object?
2. Are the input registers and limb order correct?
3. Is the canonical input condition explicit?
4. Does the result use the intended operation and modulus?
5. Is the result in the platform return registers?
6. Does the frame list every changed register and flag?
7. Does the subroutine theorem cover `ret` and the right procedure call
   convention?

Only after the statement is right should a reviewer inspect how the proof
derives it.
