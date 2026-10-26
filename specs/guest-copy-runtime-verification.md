# Spec: Guest copy runtime verification

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @GUJustin                      |
| Created     | 2026-10-26                     |
| Status      | in progress                    |
| PR          | #2155                          |

## Summary

#2142 replaced the guest `memcpy` and `memmove` with hand-written RV64
assembly (`jolt-platform/src/memory.S`, linked with `--wrap=memcpy` and
`--wrap=memmove`): about 700 lines with computed jumps into `norvc` tables,
forward and backward block loops, and a shared leaf that rebuilds misaligned
source words with shifts. Every guest runs this code, and a wrong byte silently
corrupts guest memory in every program that copies. Its direct test,
`check_memory_copy_abi` (`crates/jolt-verifier/tests/guest_runtime.rs`),
compares the arena and callee-saved registers over a fixed grid of lengths,
alignments and four distances. It does not check which addresses the runtime
reads, the alignment of its accesses, the arguments the fallback receives, or
inputs off the grid. This spec adds two independent checks: a fuzzed invariant
that runs the runtime as linked into real guests on the tracer, and a Lean
proof that the assembled machine code is correct for all inputs.

## Intent

### Goal

Establish, by fuzzing on the production execution engine and by a
machine-checked proof about the assembled bytes, that `__wrap_memcpy`,
`__wrap_memmove` and `__jolt_copy_misaligned` implement memmove semantics, keep
the RISC-V calling convention, touch only the argument ranges with naturally
aligned accesses, and dispatch to the original implementation exactly when
documented.

### Invariants

For a call with `a0 = dst`, `a1 = src`, `a2 = n` and return address `ra`:

1. **Copy semantics.** On return, `[dst, dst + n)` holds the bytes
   `[src, src + n)` held at the call (memmove semantics, so overlapping ranges
   are handled in either direction), and every other byte is unchanged.
2. **ABI.** Control returns to `ra` with `a0 = dst`; `sp`, `gp`, `tp`, `ra` and
   `s0`–`s11` are preserved. Only `t0`–`t6` and `a1`–`a7` may change.
3. **Memory safety.** Every load the runtime's own instructions issue lies in
   `[src, src + n)`, every store in `[dst, dst + n)`, and every access is
   naturally aligned.
4. **Dispatch.** The runtime tail-calls the original implementation
   (`__real_memcpy`/`__real_memmove`) exactly when `dst` and `src` differ
   modulo 8 and `12 ≤ n < 544`, with the caller's arguments, return address
   and callee-saved registers intact; it enters `__jolt_copy_misaligned`
   exactly when they differ modulo 8 and `n ≥ 544`.
5. **Termination.** Every call returns (or tail-calls the fallback) after
   finitely many instructions.

Ranges whose one-past-the-end address is `2^64` are out of scope: C cannot
form them, and the proof shows two places where `memory.S` relies on that.

The fuzzed invariant is the `jolt-eval` invariant `guest_copy_runtime`
(`#[invariant(Test, Fuzz, RedTeam)]`), checking 1–4 with a tick bound for 5.

### Non-Goals

1. Changing `memory.S` or its performance.
2. Proving Jolt's expansion of subword loads and stores into aligned
   doubleword sequences, or the R1CS that constrains it. The fuzzed invariant
   covers the expansion on the tracer; the proof is about RV64 ISA semantics.
3. Proving that a particular guest ELF links these bytes with `--wrap`. The
   fuzzed invariant covers the normally built guests.
4. Verifying the original `memcpy`/`memmove` implementations (musl,
   compiler-builtins), which the runtime falls back to.

## Evaluation

### Acceptance Criteria

- [ ] `guest_copy_runtime` checks invariants 1–4 on every input in a `no_std`
      guest (compiler-builtins fallback) and a `std` guest (musl fallback),
      including calls the fallback makes back into the runtime.
- [ ] A committed seed corpus executes every instruction of all three routines
      in both guests, enforced by a regular CI test.
- [ ] The fuzz workflow runs `guest_copy_runtime` and triggers on changes to
      `jolt-platform/`.
- [ ] Deliberately injected bugs in `memory.S` (wrong data, wrong direction,
      clobbered callee-saved register, out-of-range read, wrong thresholds)
      are detected.
- [ ] A Lean 4 project states 1–5 for the bytes `memory.S` assembles to, and
      proves them with no `sorry` and no axioms beyond `propext`,
      `Classical.choice` and `Quot.sound` (no `native_decide`, no
      `bv_decide`), enforced by an axiom audit in the build.
- [ ] CI reassembles `memory.S`, fails if the extracted bytes differ from the
      ones the proof is about, and rebuilds the proof whenever `memory.S` or
      the proof changes.

### Testing Strategy

- `cargo nextest run -p jolt-eval -E 'test(/guest_copy_runtime/)'` replays the
  seeds and checks instruction coverage (runs in the `Test jolt-prover` CI lane,
  which already builds guests).
- `scripts/fuzz.py --workspace jolt-eval --target guest_copy_runtime` runs the
  fuzz target under the PR, daily and weekly budgets in the manifest.
- `proofs/lean/copy-runtime/check.sh` regenerates, builds and audits the proof
  (its own workflow).
- The existing `check_memory_copy_abi` tests keep passing. No prover or
  verifier behaviour changes, so no clear/ZK/fixture runs are affected.

### Performance

No runtime change. CI cost: one jolt-eval test run in an existing lane, the
fuzz budgets, and a Lean job (about 80 s to check the proof locally, plus
toolchain setup).

## Design

### Architecture

**Fuzzed invariant** (`jolt-eval/src/invariant/guest_copy_runtime.rs`). Setup
builds `btreemap-guest` (no_std) and `stdlib-guest` (std) as
`check_memory_copy_abi` does, locates the three routines by ELF symbol, and
decodes them with the tracer's decoder. Each input is one call: routine,
length (biased toward the thresholds), source page offset, placement (identical,
overlapping either way, adjacent or far), arena contents and garbage in the
caller-saved registers. The harness ticks the tracer from the call to the
return. Before each instruction inside the runtime it checks the decoded
access against the argument ranges; at the fallback entry and at return it
checks registers; at the end it compares both ranges plus 512-byte guard bands
with memmove of an address-keyed snapshot.

**Lean proof** (`proofs/lean/copy-runtime/`, Lean 4 core and Std only).
`regenerate.sh` assembles `memory.S` with the pinned rustc exactly as the guest
build does, and `extract.py` writes the section bytes, relocations and the
`llvm-objdump` listing as Lean data. A hand-written RV64I/RVC semantics for the
instructions that occur executes these bytes in a sandbox that faults on any
access outside the argument ranges or misaligned, so a non-faulting run proves
invariant 3. The decoder is cross-checked against LLVM's listing by kernel
evaluation. Relocations are applied as lld does, and the two relocated jumps are
the exits to the leaf and the fallback. The main theorems
(`CopyRuntime/Theorems.lean`) state 1–5 for each wrapper and the leaf and for
the wrapper linked with the leaf; `CopyRuntime/Audit.lean` fails the build on
any other axiom.

### Alternatives Considered

- **Prove the `.S` source.** Rejected: the computed jumps depend on the
  assembler's encodings (the `norvc` tables) and on relocation handling, so the
  object bytes are what must be verified.
- **HOL Light with s2n-bignum, as for the field arithmetic proofs.**
  s2n-bignum models x86-64 and AArch64, not RISC-V, so it would need a new ISA
  model anyway; a small Lean project with no dependencies is cheaper to build
  in CI.
- **A generated RISC-V model (Sail's Lean output).** It would shrink the
  trusted base, but brings a large generated dependency to build and execute
  symbolically. The hand-written semantics covers about 30 instructions and
  its decoder is checked against LLVM.
- **Differential testing against a host `memmove` in a separate emulator.**
  Rejected: the tracer is the engine whose execution is proved, including its
  expansion of subword accesses.

## Documentation

`FUZZING.md` and `jolt-eval/README.md` list the new target.
`proofs/lean/copy-runtime/README.md` documents what is proved, the trusted
base, what is not covered, and how to build and extend the proof. No book
changes: guest-visible behaviour is unchanged.

## References

- #2142 (the copy runtime), `jolt-platform/src/memory.S`
- `crates/jolt-verifier/tests/guest_runtime.rs` (`check_memory_copy_abi`)
- The RISC-V Unprivileged ISA manual (RV64I, the C extension and its expansions)
- `specs/production-fuzz-infra.md` (fuzz manifest and budgets)
