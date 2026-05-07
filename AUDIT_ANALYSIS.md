# Audit Report Analysis — 2026-04-15

**Source:** `jolt-audit-report.html` by Justin Thaler (with Claude assistance)
**Scope:** jolt-core, tracer — soundness focus
**Report verdict:** "No soundness vulnerabilities found."

## Summary

| # | Finding | Severity | Verdict | Action |
|---|---------|----------|---------|--------|
| 1 | LD/SD address arithmetic doesn't wrap at 64 bits | LOW | REAL (completeness) | Document as unsupported |
| 2 | `has_side_effects()` defaults to false | INFO | REAL | Remove trait default; explicit per impl |
| 3 | RAM ValCheck alignment `debug_assert` only | INFO | REAL | Promote to runtime `assert_eq!` |
| 4 | DoryGlobals uses process-wide mutable state | INFO | REAL | Out of scope (large refactor) |
| 5 | Dory transcript label flattening | INFO | REAL (non-trivial) | Deferred (API change required) |

Two findings fixed in this PR (2, 3). One documented (1). Two deferred with rationale (4, 5).

---

## Finding 1 — LD/SD address arithmetic doesn't wrap at 64 bits

**File:** `jolt-core/src/zkvm/r1cs/constraints.rs:237–241`

### What the audit claims

The R1CS constraint
```
RamAddress = Rs1Value + Imm
```
computes in the BN254 field (~254 bits). RISC-V RV64 computes the effective
address as a 64-bit wrapping sum (`rs1 + sign_ext(imm)` mod 2^64). If
`Rs1Value + Imm >= 2^64`, the field sum differs from the RISC-V wrapping sum.
LD and SD are the only load/store ops not decomposed into virtual sequences,
so they are directly exposed.

### What the code does

- `tracer/src/instruction/ld.rs:18-19`: emulator uses `.wrapping_add(...)`.
- `tracer/src/instruction/sd.rs:25`:    emulator uses `.wrapping_add(...)`.
- `jolt-core/src/zkvm/r1cs/constraints.rs:237-241`: R1CS enforces
  `RamAddress = Rs1Value + Imm` in the field — no wrap, no range check.

If a program computes an effective address that overflows u64:
- Emulator records the wrapped address in the trace.
- Prover's `RamAddress` witness is the wrapped u64.
- `Rs1Value + Imm` in the field is the non-wrapped sum.
- The constraint fails → proof fails.

### Verdict: REAL (completeness, not soundness)

The constraint is strictly stronger than RISC-V allows. Soundness is
unaffected: a malicious prover cannot exploit this to forge proofs; it only
prevents honest provers from proving programs that wrap.

In practice, load/store addresses are near `0x80000000` (heap / static
data), far from the 2^64 wrap boundary. Real programs don't trigger this.

### Action: document

Adding a 64-bit wrap to the constraint requires bit decomposition of
`Rs1Value + Imm` plus a conditional 2^64 subtraction on overflow. That is an
intrusive R1CS change touching soundness-critical code. The audit report
itself suggests "either fixing or documenting as unsupported."

Given:
- No soundness impact.
- No practical completeness impact for realistic programs.
- R1CS changes in this area carry soundness risk.

We document the limitation at the constraint definition.

---

## Finding 2 — `has_side_effects()` defaults to false

**File:** `tracer/src/instruction/mod.rs:380-382`

### What the audit claims

After the x0 fix (#1255), an instruction with `rd = x0` is replaced with a
NOP unless `has_side_effects()` returns true. The trait's default returns
`false`, so any future direct `impl RISCVInstruction` that forgets to
override would silently drop its side effects when `rd = x0`.

### What the code does

- `tracer/src/instruction/mod.rs:380-382`: default returns `false`.
- `tracer/src/utils/instruction_macros.rs:84-86`: the `declare_riscv_instr!`
  macro emits `has_side_effects` wired from the `side_effects = true/false`
  parameter (macro default is `false`). All in-macro instructions are
  explicit.
- Direct trait impls outside the macro:
  - `tracer/src/instruction/inline.rs:129`: explicitly `true`.
  - `tracer/src/instruction/virtual_advice.rs:23-55`: does **not** override;
    relies on the default. `VirtualAdvice` writes only to `rd`, so "no side
    effects beyond writing rd" is correct — but this is load-bearing and
    undocumented.

The audit's concern is forward-looking: the next direct impl could silently
be wrong.

### Verdict: REAL (hardening)

### Action: remove trait default

Remove the default body from `RISCVInstruction::has_side_effects`. Every
direct impl must now declare its side-effect status explicitly.
`VirtualAdvice` gains an explicit `fn has_side_effects(&self) -> bool { false }`
to preserve current behavior (rd-only write).

---

## Finding 3 — RAM ValCheck alignment checked only by `debug_assert`

**File:** `jolt-core/src/zkvm/ram/val_check.rs:113-120` (prover),
`174-183` (verifier)

### What the audit claims

The alignment between `RamVal` and `RamValFinal` opening points'
`r_address` components is checked only via `debug_assert_eq!`, which
evaluates to nothing in release builds. A future refactor could break the
alignment silently in production.

### What the code does

Both `new_from_prover` and `new_from_verifier` read two opening points:
- `r_address` from `RamVal` @ `RamReadWriteChecking`
- `r_out` from `RamValFinal` @ `RamOutputCheck`

After Stage 2 alignment, these are supposed to share identical address
components. The check is:
```rust
#[cfg(debug_assertions)]
{
    let r_out = ...;
    debug_assert_eq!(r_out.r, r_address.r);
}
```

### Verdict: REAL (hardening)

The check is off-hot-path (once per proof, not per sumcheck round) — cost
is negligible. Relying on `debug_assert` for an invariant that underpins
the soundness argument of the unified RAM sumcheck is fragile.

### Action: promote to runtime `assert_eq!`

Remove the `#[cfg(debug_assertions)]` gate and swap `debug_assert_eq!` for
`assert_eq!`. One check per proof on both sides.

---

## Finding 4 — DoryGlobals uses process-wide mutable state

**File:** `jolt-core/src/poly/commitment/dory/dory_globals.rs`

### What the audit claims

`CURRENT_CONTEXT` (an `AtomicU8`) and the three `RwLock<Option<usize>>`
tuples are process-wide. Two concurrent proofs in the same process would
race on the context pointer: proof A in `Main` while proof B switches to
`TrustedAdvice` would make `get_T()` return the wrong value for proof A.

Currently not triggered because Jolt runs one proof per process and
intra-proof parallelism is within a single context phase (rayon within one
stage).

### Verdict: REAL, out of scope

Accurate as described. Fixing it requires threading a `DoryContext`
argument (or handle) through a large call graph: commitment routines,
opening proofs, streaming commitment, and all Dory-adjacent sumcheck
hooks. This is the refactor the audit acknowledges ("not a one-liner").

### Action: deferred

Not fixed in this PR. No current bug triggers it. Would block concurrent
in-process proof generation if that workflow is added.

---

## Finding 5 — Dory transcript label flattening

**File:** `jolt-core/src/poly/commitment/dory/wrappers.rs:348-403`

### What the audit claims

`JoltToDoryTranscript` discards Dory's internal labels and substitutes four
fixed labels (`b"dory_bytes"`, `b"dory_field"`, `b"dory_group"`,
`b"dory_serde"`). Different Dory protocol messages with distinct semantic
labels end up sharing the same transcript label; they are disambiguated by
content and ordering only. No exploit identified.

### What the code does

```rust
fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
    ...
    transcript.append_bytes(b"dory_bytes", bytes);
}
```
and similar for `append_field`, `append_group`, `append_serde`.

### Verdict: REAL, but not "low effort"

The audit calls this a "low-effort cleanup." It is not:
- Jolt's `Transcript::append_bytes` (and friends) require
  `label: &'static [u8]`.
- Dory passes `&[u8]` (non-static) labels through its generic `DoryTranscript`
  trait.
- Bridging requires either:
  - Adding a dynamic-label variant to Jolt's `Transcript` trait (API change
    spanning Blake2b, Keccak, Poseidon impls), or
  - Embedding the label inside the message bytes (no real domain
    separation, just a content change).

No soundness concern. Changing this also invalidates every existing proof
serialization format.

### Action: deferred

Noted in PR body. Appropriate follow-up is a transcript-API design
decision, not a mechanical patch.

---

## Test coverage

- `cargo nextest run -p jolt-core muldiv --features host` — primary e2e
  correctness check in standard mode.
- `cargo nextest run -p jolt-core muldiv --features host,zk` — ZK mode e2e.

Finding 3's new runtime assert exercises the same code path as the
previous `debug_assert`, so existing ValCheck tests cover it.
Finding 2's refactor is compile-time: removing the trait default forces
explicit decl. Any impl that silently relied on the default would break
the build — compile success is the test.
