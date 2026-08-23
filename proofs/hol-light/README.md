# HOL Light field kernel proofs

The Fp128 proofs cover scalar addition, subtraction, and multiplication for
`Prime128OffsetA7F7` on AArch64 and x86-64. Production Rust includes the exact
instruction fragments imported by those proofs.

The Fp64 proofs cover scalar addition, subtraction, and multiplication for
`Prime64Offset59` on AArch64 and x86-64. They include baseline x86-64 and BMI2
multiplication. Production keeps its faster generic Rust implementation. An
artifact checker confirms byte identity for Linux x86-64 and for both Darwin
and Linux AArch64. Darwin and Linux AArch64 use separate exact objects and
separate theorems because Rust can schedule independent instructions and choose
temporary registers differently on each target. On Darwin x86-64, the checker
checks one exact compiler frame around the proved sequence. No Fp64 proof uses
AVX, AVX2, or AVX-512.

The checked in Fp64 build matrix is
[`fp64-certified-builds.json`](fp64-certified-builds.json). It is the source of
truth for registered target triples and feature profiles. It also selects the
wrapper policy, proof files, and theorem names. The expected instruction bytes
remain independent constants in the artifact checker and the HOL Light
imports. This prevents a matrix edit from approving changed code by itself.

Start with the Book chapter
[Formal verification of field kernels](../../book/src/how/formal-verification/field-kernels.md).
It explains the arithmetic, theorem shape, production connection, and trust
boundary.

The [scalar Fp64 guide](../../book/src/how/formal-verification/field-kernels-fp64.md)
states the precise Fp64 claim, explains the `2^64 - 59` reduction, and lists
the remaining Fp64 work.

The final theorems cover complete callable Darwin AArch64, Linux AArch64, and
Linux x86-64 witness functions. The baseline x86-64 bodies copy their internal
results to the System V return registers `rax:rdx`. The BMI2 and ADX
multiplication body
creates its result directly in those registers. The theorems prove the result
moves where present, the `ret` stack behavior, and an ABI safe frame. The
Darwin x86-64 compiler adds a fixed frame wrapper. The artifact checker checks
that wrapper exactly, but the current theorem does not cover its frame
instructions. Every proved object loads its own reduction constant, so none of
the theorems assumes a prepared offset register. All arithmetic theorems assume
canonical inputs and prove the canonical result modulo
`0xffffffffffffffffffffffff00005809`. A separate certificate proves that this
modulus is prime.

The inspection witness calls the normal Rust field operation. On Darwin
AArch64, Linux AArch64, and Linux x86-64, the artifact checker confirms that
its complete optimized symbol is byte identical to the matching proved object.
On Darwin x86-64, the checker requires an exact frame wrapper around the proved
sequence. Normal field operations still inline the arithmetic fragment, so the
theorems do not cover the machine code around every inlined copy. Jolt does not
inspect downstream executables. A downstream project must perform that final
binary check at its pinned Jolt revision.

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

Packed SIMD operations, squaring, inversion, small offset immediate kernels,
and generic fallback kernels are outside the present proof scope.

## Scalar Fp64 checks

Run the fast Fp64 byte check with

```sh
./proofs/hol-light/check-fp64.sh bytes x86_64 \
  --matrix-entry x86_64-unknown-linux-gnu
```

The matrix entry must match the actual compilation target. Use
`aarch64-apple-darwin` on Apple Silicon and
`aarch64-unknown-linux-gnu` on Linux AArch64. Darwin x86-64 is registered as
`x86_64-apple-darwin-inspection-only`. Its exact compiler frame is checked, but
the callable frame is not covered by the current theorem.

Start a persistent Fp64 proof session with

```sh
HOL_LIGHT_DIR=/path/to/hol-light \
S2N_BIGNUM_DIR=/path/to/s2n-bignum \
  ./proofs/hol-light/dev-fp64.sh x86_64 mul_bmi2
```

Run the complete clean Fp64 check with

```sh
HOL_LIGHT_DIR=/path/to/hol-light \
S2N_BIGNUM_DIR=/path/to/s2n-bignum \
  ./proofs/hol-light/check-fp64.sh all x86_64 \
    --matrix-entry x86_64-unknown-linux-gnu \
    --evidence-out /path/to/fp64-x86-64-linux-gnu.json \
    --clean
```

Use `aarch64` for the AArch64 byte and clean checks. The Fp64 theorem sessions
accept `add`, `sub`, and `mul` on both architectures. The x86-64 session also
accepts `mul_bmi2`.

The runner rejects an unregistered target or feature set. It also rejects an
unregistered toolchain, release profile, or ambient Rust code generation flag.
It supplies every release profile value recorded by the matrix directly to
Cargo and marks the build with the current matrix contract. A direct Cargo build with
`fp64-proof-linkage` is not supported and fails without that contract. The
exact byte comparison remains the final authority for the compiler output.

A successful clean run writes an unauthenticated JSON run record when
`--evidence-out` is present. That file records the exact build identity and
artifact hashes. It also records the proof library commits, theorem names, and
proof log hash. The file has no signature or attestation. A reviewer must trust
its CI provenance or repeat the run. It does not claim that a downstream
executable reaches the checked inspection function.

## Inspecting a final executable

The proof checks above stop at a small inspection function. Use the final
executable checker to inspect a linked application.

```sh
python3 scripts/check_field_proof_final_binary.py \
  --architecture x86_64 \
  --binary /path/to/application \
  --require-family fp128 \
  --json-output /path/to/final-binary-report.json
```

The checker disassembles the executable and starts a match only at a decoded
instruction boundary. It records the executable hash, symbol, operation, and
instruction address. `--require-family fp128` fails unless it finds exact
instances of addition, subtraction, and multiplication. Use `fp64` for the
Fp64 objects.

Inline assembly can have the same instruction pattern with different physical
registers. You can pass an exact proof object to find those cases.

```sh
--proof-object fp128:add=/path/to/fp128_add.o
```

The report calls these results structural candidate proof instances. They do
not inherit the existing theorem. A completed proof still needs to cover the
new instruction bytes and the values prepared before the matched body. In
particular, the checker reports a Solinas constant register as an external
input when the compiler moved its load outside the body.

The checker also does not prove that program control reaches a reported
address. A release check must start from the verifier entry point and establish
that reachability separately.
