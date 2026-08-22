# Fp128 HOL Light proofs

These proofs cover scalar addition and subtraction for
`Prime128OffsetA7F7` on AArch64 and x86-64. They also cover scalar
multiplication on AArch64. They import the exact instruction words or bytes
from standalone objects. Production Rust includes the same instruction body
files.

Start with the Book chapter
[Formal verification of field kernels](../../book/src/how/formal-verification/field-kernels.md).
It explains the arithmetic, theorem shape, production connection, and trust
boundary.

The final theorems cover callable functions, including `ret` and the relevant
procedure call convention. The AArch64 and x86-64 addition and subtraction
objects include the production constant load, so those theorems do not assume
a prepared offset register. All arithmetic theorems assume canonical inputs
and prove the canonical result modulo
`0xffffffffffffffffffffffff00005809`. A separate certificate proves that this
modulus is prime.

The public witness calls the normal Rust field operation. The artifact checker
confirms that its optimized machine code contains the proved instruction body.
Jolt does not inspect an Akita executable. Akita must perform that final binary
check at its pinned Jolt revision.

## Requirements

You need `llvm-objdump`, an OCaml environment that can build HOL Light proofs,
and local checkouts of HOL Light and `s2n-bignum`. The x86-64 proof can run on
Apple Silicon when `clang` can emit an x86-64 ELF object for HOL Light to load.

CI pins these revisions.

* HOL Light commit `433477862bb90b328a593e012e09390e99b2439b`
* `s2n-bignum` commit `ac31a43db30953037abd1b64b540e65cf31f4c67`

## Fast byte check

Use this after changing an instruction body, build script, Rust assembly, or
artifact checker.

```sh
./proofs/hol-light/check.sh bytes x86_64
```

This builds the optimized public witness and answers only whether the proof
object and production witness still contain the expected bytes. It does not
start HOL Light. The default target directory is persistent, so unchanged
dependencies are reused.

Use `aarch64` instead of `x86_64` to check the AArch64 path.

## Interactive theorem development

Start one persistent session for one operation.

```sh
HOL_LIGHT_DIR=/path/to/hol-light \
S2N_BIGNUM_DIR=/path/to/s2n-bignum \
  ./proofs/hol-light/dev.sh x86_64 sub
```

The initial bytecode load imports the x86 model, exact object, execution rule,
and shared lemmas. It can take several minutes. The session prints a command
that reloads only the editable correctness file. Reloads then take seconds and
syntax or tactic failures do not require another Cargo build or model load.

Use `./proofs/hol-light/dev.sh aarch64 mul` for AArch64 multiplication. Its
editable theorem file is also reloaded inside one persistent processor-model
session.

## Clean final check

Run the reproducible release check with fresh build output.

```sh
HOL_LIGHT_DIR=/path/to/hol-light \
S2N_BIGNUM_DIR=/path/to/s2n-bignum \
  ./proofs/hol-light/check.sh all x86_64 --clean
```

Each architecture builds one proof program and loads its processor model once
for all covered operations. A local run without `--clean` reuses that program when
the HOL Light revision, `s2n-bignum` revision, generated entry, and proof
sources are unchanged. It streams output while also preserving logs. A failed
clean run preserves its temporary workspace and prints the path. A successful
clean run removes it. CI runs clean checks independently for AArch64 and
x86-64.

Packed SIMD operations, x86-64 multiplication, squaring, inversion, small
offset immediate kernels, and generic fallback kernels are outside the
present proof scope.
