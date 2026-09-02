# Trust boundary and review guide

Formal verification gives a precise result. It does not make every nearby
component correct. This page separates what Jolt proves, checks, tests, and
trusts.

## Four evidence words

This documentation uses four words with fixed meanings.

| Word | Meaning |
| --- | --- |
| Proven | HOL Light derives the stated claim from the exact modeled instructions |
| Checked | A tool compares artifacts and rejects an unexpected result |
| Tested | A program checks selected examples or environments |
| Trusted | The claim assumes this component behaves correctly |

A byte comparison is checked evidence. It is not an arithmetic proof. A
million random products are strong test evidence. They do not cover every
possible input pair.

## Evidence at each layer

| Layer | Current evidence |
| --- | --- |
| Arithmetic specification | Reviewed equations for scalar addition, subtraction, and multiplication modulo `2^128 - C` under the Fp128 offset bounds and modulo the `2^64 - 59` prime |
| Exact proof object | Generic Fp128 HOL Light body theorems, A7F7 callable subroutine theorems, and Fp64 body and callable subroutine theorems |
| Linux inspection witness | Complete byte equality with the proved object |
| Darwin AArch64 inspection witness | Complete byte equality with the proved object |
| Darwin x86-64 inspection witness | Exact wrapper bytes checked, while frame semantics remain outside the theorem |
| Arbitrary inlined callers | Rust and LLVM compiler contract trusted |
| Fp64 proof build selection | Exact registered target and feature profile checked before witness compilation |
| CPU availability at deployment | Trusted deployment condition for optional x86 features |
| Final downstream executable | Required for a release claim, but not performed by Jolt |
| Complete field use | Extension fields, packed arithmetic, unreduced accumulation, squaring, and inversion have separate obligations |

## Why not use only idiomatic Rust?

Idiomatic Rust is easier for many engineers to read. It removes many memory
safety risks. Jolt keeps a portable Rust implementation for these reasons.

Rust does not prove modular arithmetic for every input. The generated
instructions can also change with the compiler version, optimization flags,
and target CPU. Tests can miss a carry failure that occurs only at a boundary
value.

Jolt uses handwritten Fp128 assembly because measurements show a useful gain
and because HOL Light can verify the exact instructions. The Fp64 production
path stays in Rust because a native assembly experiment was slower. For Fp64,
HOL Light proves a matching standalone sequence and the artifact checker ties
it to one compiled inspection function.

The proof gives a stronger functional claim than tests. It does not make
assembly or compiler output safe by itself. Jolt still needs correct compiler
contracts, correct feature selection, and a final integration check.

If assembly does not improve performance, its unsafe boundary and proof upkeep
are difficult to justify. The parameterized paths remain because native
measurements show a gain over the portable implementation. The A7F7 BMI2 and
ADX variant remains separate because not every x86-64 processor supports those
instructions and because its bytes embed the A7F7 offset. Fp64 keeps its Rust
production path because the assembly experiment was slower.

## What the arithmetic theorem rules out

Under its stated preconditions, the theorem rules out arithmetic errors in the
modeled instruction sequence.

Examples include:

* A missing carry between limbs.
* A borrow mask with the wrong sign.
* A reduction constant with one wrong bit.
* A multiplication term added to the wrong limb.
* A final correction that returns a noncanonical value.

Tests are still useful. They catch errors in paths outside the theorem and give
fast feedback. Once the exact machine theorem passes, adding more random cases
does not strengthen its universal arithmetic statement.

## What the theorem does not rule out

### A wrong Rust or assembly compiler boundary

The bytes can be correct while the surrounding program is wrong. For inline
assembly, a missing changed register can tell LLVM that a value survived when
the assembly actually destroyed it. For generic Rust, a compiler can produce
different instructions in different callers. The theorem proves the object in
isolation. The inspection witness checks one complete compilation. Other
inlined callers still rely on the compiler.

### A wrong dispatch path

The theorem does not prove which branch the Rust program selects. Jolt builds
and checks both registered x86-64 configurations. The compiler removes the
unused feature branch when it builds the program. A downstream application
must still use the intended field type and build settings.

### An unknown build identity

The Fp64 proof runner does not guess from the host operating system. It resolves
one checked in build entry from the complete Rust target triple. That entry
also fixes the target features, Cargo release profile, object format, wrapper
policy, Rust toolchain, proof sources, and theorem names.

Proof linkage fails when the target or feature set is absent from the matrix.
The runner also rejects ambient Rust flags and profile overrides. This prevents
an unreviewed Windows, musl, mobile, or native CPU build from inheriting a
nearby Linux or Darwin claim.

The clean runner emits a JSON run record after byte checking and theorem marker
checking pass. The record lists the selected identities and artifact hashes.
It has no signature or attestation. A reviewer must trust its CI provenance or
repeat the run. It is not proof for arbitrary inlined callers or a downstream
executable.

### Unsupported hardware

HOL Light defines the behavior of BMI2 and ADX instructions. It does not prove
that the deployment processor implements them. A binary that enables those
features requires compatible hardware.

### Side channels

The scalar fragments are branchless and do not use secret dependent data
addresses. This is useful design evidence.

The current theorem does not prove constant time, timing noninterference,
speculative behavior, cache behavior, power leakage, or resistance to hardware
faults. It also does not rule out processor errata.

### A bad processor model

The proof is relative to the instruction definitions in the pinned
`s2n-bignum` x86 and AArch64 models. We trust those definitions to match real
processors. Existing s2n-bignum proofs and instruction simulation provide
substantial review and test evidence, but they do not remove this assumption.

### A compromised proof environment

We trust the HOL Light logical kernel, OCaml runtime, operating system, and host
hardware. We pin the HOL Light and `s2n-bignum` revisions so reviewers can
reproduce the same definitions and tools.

### Noncanonical inputs

The arithmetic conclusion assumes both inputs are below `p`. An unsafe internal
constructor can violate that rule if its caller is wrong. Checked decoding
rejects an out of range value. Safe field operations are designed to return a
canonical value. The whole program preservation argument is not yet formalized.

### Other field operations

The scalar add, subtract, and multiply theorems do not automatically prove
extension field formulas, squaring, inversion, packed SIMD code, or unreduced
accumulators. The current Fp64 proof also does not cover
`Prime63Offset259`. Inversion relies on the separate theorem that the modulus
is prime.

## Inline code and callable objects

The proof object is a complete callable function. Fp128 production operations
stay inline because a function call was measured to cost more than the small
add and subtract bodies. Fp64 production operations also stay inline, but LLVM
generates them from Rust rather than from a shared assembly fragment.

This choice preserves performance and leaves a compiler boundary. The shared
fragment and inspection witness make divergence visible. They do not prove the
machine code around every inlined copy.

An alternative design would call the proved object in production. That gives a
shorter connection from theorem to executed symbol. It also adds a call and
return. Jolt should use that design only where measurements show that the call
does not harm the hot path.

## Linux and Darwin

The System V theorem matches Linux x86-64. The optimized Linux inspection
witness must be byte identical to the complete object.

The Darwin compiler adds a frame setup and teardown. Jolt checks those bytes
exactly. HOL Light currently proves the arithmetic and return sequence inside
that wrapper, not the frame instructions themselves. A reviewer should not
describe the complete Darwin function as proved.

Windows x86-64 uses a different procedure call convention. The current
subroutine theorem does not cover it.

## Downstream release checklist

Before claiming that a deployed binary uses a proved field operation, record
the following evidence.

1. The exact Jolt commit.
2. The registered Fp64 build matrix entry when the Fp64 claim is used.
3. The Rust compiler version, target triple, and target CPU features.
4. The HOL Light and `s2n-bignum` commits.
5. The object hash, witness hash, proof log hash, and theorem names from a clean proof run.
6. The selected field type and operation call path.
7. The final binary symbol or instruction location.
8. Evidence that the claimed execution path reaches that code.
9. Every operation and representation that remains outside the claim.

The current legacy `akita` feature still reaches the external Akita field
implementation. The Jolt Fp128 theorems do not cover that runtime path. The
field cutover and downstream binary inspection must happen before a claim about
the complete Akita path is valid.

## What a final executable scan can establish

Jolt includes `scripts/check_field_proof_final_binary.py` for this last
inspection step. An exact match means that a linked executable contains the
proved instruction bytes at decoded instruction boundaries. The report records
the executable hash, symbol, operation, and address. The checker can fail a
release check when an expected operation is absent.

The checker can also report the same decoded instruction pattern under a
consistent register renaming. This result is only a candidate for another
machine proof. The existing theorem names exact registers, and those register
numbers are part of the instruction bytes. A report about matching structure
does not change that theorem.

The compiler can also move the Solinas constant load outside the arithmetic
body. In that case, a proof for the body must assume a constant register value
or prove the earlier instructions that prepared it. The report marks that
register as an external input. Reviewers must not treat an unproved input value
as a checked constant.

The current external Akita path demonstrates both issues. Its AArch64 compiler
output contains many addition and multiplication bodies with the same decoded
pattern as the Jolt proof objects. LLVM chose different registers and moved the
A7F7 constant load. Its subtraction allocation also differs in how input and
output registers share storage. The exact Jolt object theorem therefore does
not apply to those copies.

The scan proves neither reachability nor completeness by itself. A matching
sequence can belong to prover code, unused generic code, or another operation
with the same short instruction shape. A release claim still needs a path from
the verifier entry point to the matched code. It also needs evidence that every
external value, including the reduction constant, has the required value.

## Claim language

Use this form.

> HOL Light proves functional correctness of these exact instruction bytes
> under the pinned processor model and canonical input precondition.

For the Linux inspection witness, add this statement.

> The artifact checker confirms that the configured optimized Linux inspection
> witness is byte identical to the proved callable object.

Then state the remaining boundary.

> The proof does not currently establish every inline call site or a downstream
> executable.

Do not shorten this to “Fp128 is fully verified,” “Fp64 is fully verified,”
“the Rust implementation is proved,” or “the final binary is verified.” Those
sentences claim more than the current evidence establishes.
