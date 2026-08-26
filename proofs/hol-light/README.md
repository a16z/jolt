# Fp128 HOL Light proofs

These proofs cover the register-parameterized scalar addition, subtraction,
and multiplication bodies for every valid Fp128 offset on AArch64 and
baseline x86-64. They also prove complete callable objects for
`Prime128OffsetA7F7`, including the literal constant load and return sequence,
and the A7F7-specific x86-64 BMI2 and ADX multiplication object. Production
Rust includes the same instruction body files.

The assembly kernels are opt-in through the `jolt-field/asm` feature. Builds
that enable `solinas` without `asm` use the portable Rust implementation on
every architecture. The inspection-only `fp128-proof-linkage` feature implies
`asm`, so the byte checker always inspects the assembly path it is intended to
connect to the proof objects.

Start with the Book chapter
[Formal verification of field kernels](../../book/src/how/formal-verification/field-kernels.md).
It explains the arithmetic, theorem shape, production connection, and trust
boundary.

The generic body theorems quantify over `C`, assume the same offset bounds as
the Rust `Fp128` implementation, and start immediately after the fixture
object's constant-load instruction with `C` in `x4` or `r8`. They prove the
canonical result modulo `2^128 - C` for every canonical input. Both public
offsets, 275 and A7F7, have machine-checked proofs that they satisfy those
bounds.

The A7F7 corollaries cover complete callable AArch64 and Linux x86-64 witness
functions. The baseline x86-64 bodies copy their internal results to the
System V return registers `rax:rdx`. The BMI2 and ADX multiplication body
creates its result directly in those registers. These corollaries prove the
literal load, result moves where present, `ret` behavior, and an ABI-safe
frame. A separate certificate proves that the A7F7 modulus is prime. The
Darwin x86-64 compiler adds a fixed frame wrapper. The artifact checker checks
that wrapper exactly, but the current theorem does not cover its frame
instructions.

The inspection witness calls the normal Rust field operation. On AArch64 and
Linux x86-64, the artifact checker confirms that its complete optimized symbol is
byte identical to the proved object. On Darwin x86-64, the checker requires an
exact frame wrapper around the proved sequence. Normal field operations still
inline the arithmetic fragment, so the theorems do not cover the machine code
around every inlined copy. Jolt does not inspect downstream executables. A
downstream project must perform that final binary check at its pinned Jolt
revision.

The detailed [source-to-bytes walkthrough](../../book/src/how/formal-verification/field-kernels-source-to-bytes.md),
[theorem guide](../../book/src/how/formal-verification/field-kernels-reading-theorem.md),
and [trust boundary](../../book/src/how/formal-verification/field-kernels-trust-boundary.md)
explain what each check establishes and what remains trusted.

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

This builds the baseline inspection witness and another witness with BMI2 and
ADX enabled. It answers only whether each proof object and complete witness
symbol have the expected bytes. It does not start HOL Light. The default target
directory is persistent, so unchanged dependencies are reused.

Use `aarch64` instead of `x86_64` to check the AArch64 path.

## Differential fuzzing

The `fp128_asm_differential` target compares assembly with portable arithmetic
for both public offsets, 275 and A7F7, and a test-only generic offset 173, for
addition, subtraction, and multiplication. The extra offset ensures the
register-parameterized path stays independently reachable. On AArch64 it also
compares the assembly square and fused multiply-add kernels. CI fuzzes the
target natively on AArch64, on baseline x86-64 through the general fuzz
workflow, and on x86-64 with BMI2 and ADX enabled through this proof workflow.

Run it locally from `crates/jolt-field` on a supported architecture.

```sh
cargo +nightly fuzz run fp128_asm_differential -- -max_total_time=120
```

Fuzzing complements the theorem and byte checks. It does not replace either:
the theorem covers every canonical input for the proved instruction sequence,
while fuzzing continuously tests the Rust dispatch and assembly interface
against an independent portable implementation.

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

Use `mul` instead of `sub` for baseline x86-64 multiplication. Use
`mul_bmi2_adx` for the BMI2 and ADX theorem. Use
`./proofs/hol-light/dev.sh aarch64 mul` for AArch64 multiplication. Each
editable theorem file reloads inside one persistent processor model session.

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

Packed SIMD operations, squaring, inversion, and fused multiply-add remain
outside the present machine-proof scope. They retain differential or ordinary
test coverage as described in the Book chapter.
