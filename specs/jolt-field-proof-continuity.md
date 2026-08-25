# Jolt field proof continuity

This record identifies where the restored field assets came from and states
the ownership boundary after the Akita field migration.

## Source revisions

| Asset | Source revision | Jolt destination |
| --- | --- | --- |
| Fp128 architecture add and subtract kernels | Akita main `6918a2018da6061d6399c74b542017027a7b74b8` | `crates/jolt-field/src/solinas/fp128/add_sub.rs` |
| A7F7 AArch64 instruction bodies and proof objects | Akita proof head `241cde109751ae28d02b55c92ac54e923a6a92af` | `crates/jolt-field/asm/aarch64/` |
| A7F7 x86-64 instruction bodies and proof objects | This Jolt continuation | `crates/jolt-field/asm/x86_64/` |
| AArch64 HOL Light addition and subtraction theorems | Akita proof head `241cde109751ae28d02b55c92ac54e923a6a92af` | `proofs/hol-light/` |
| x86-64 HOL Light addition and subtraction theorems | This Jolt continuation | `proofs/hol-light/` |
| A7F7 AArch64 multiplication body and HOL Light theorems | This Jolt continuation | `crates/jolt-field/asm/aarch64/` and `proofs/hol-light/` |
| A7F7 baseline x86-64 multiplication body and HOL Light theorems | This Jolt continuation | `crates/jolt-field/asm/x86_64/` and `proofs/hol-light/` |
| A7F7 BMI2 and ADX x86-64 multiplication body and HOL Light theorems | This Jolt continuation | `crates/jolt-field/asm/x86_64/` and `proofs/hol-light/` |
| A7F7 primality certificate | This Jolt continuation | `proofs/hol-light/fp128_prime.ml` |
| Inspection witness and artifact checker design | Akita proof head `241cde109751ae28d02b55c92ac54e923a6a92af` | `crates/jolt-field/examples/` and `scripts/` |
| Field specific legacy transcript dispatch | Jolt PR 1745 commit `6b5d3ff` | `crates/jolt-prover-legacy/` |
| Fp64 and extension corrections | Jolt PR 1794 head `4a0d4a33265c6fc7c1dc0e97046b67773a8320ea` | Stacked base of this work |

The Fp128 add and subtract assembly omission predates the recent Jolt refresh.
Alberto removed those specializations in commit
`e1cbc17a3b31bbb3593d0242edbd0bd74a033c08`. The later refresh retained that
choice. The proof connection was newer Akita work and therefore never existed
in the original Jolt migration branch.

## Ownership after migration

Jolt owns the field implementation, exact AArch64 and x86-64 bodies, proof
objects, HOL Light theorems, inspection witness, and proof workflow.

The architecture kernels are opt-in through the `jolt-field/asm` feature.
`solinas` without `asm` uses portable Rust, while `fp128-proof-linkage` implies
`asm` so proof artifact checks cannot accidentally inspect the portable path.
Differential fuzzing compares the selected assembly kernels with portable
arithmetic on AArch64, baseline x86-64, and x86-64 with BMI2 and ADX.

Akita owns the check that its final verifier executable contains the proved
operation from the exact Jolt revision selected by Cargo. This downstream
check cannot be performed by the Jolt repository alone.

## Transcript decision

BN254 and Dory retain Jolt's historical scalar challenge byte reversal. The
shared Fp128 field uses Akita's direct little endian convention. This changes
old Jolt packed Akita proof transcripts. Such proofs are intentionally not
supported because neither Akita nor the Jolt integration promises backward
compatibility.

## Proof scope

The current HOL Light claim covers scalar AArch64 and x86-64 addition,
subtraction, and multiplication for `Prime128OffsetA7F7`, under canonical
input assumptions. x86-64 has a baseline multiplication theorem and a separate
theorem for builds that enable both BMI2 and ADX. The proofs also show that the
A7F7 modulus is prime. The x86-64 callable theorems continue through any result
moves into `rax:rdx` and through `ret`. The complete optimized Linux inspection
witness symbols are byte identical to those proved objects. Darwin
x86-64 adds an exact frame wrapper that the artifact checker checks but HOL
Light does not yet prove. Normal field operations still inline the arithmetic
bodies, so the compiler code around arbitrary inlined copies remains outside
the theorem. The proofs do not cover the small offset immediate kernels,
generic fallback kernels, packed SIMD arithmetic, squaring, inversion, the
complete Rust verifier, or a downstream final executable.

The Book chapter [Formal verification of field kernels](../book/src/how/formal-verification/field-kernels.md)
and its linked guides define the evidence words, theorem shape, source to byte
connection, and remaining trust boundary.
