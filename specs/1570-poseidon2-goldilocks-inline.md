# Spec: Poseidon2-Goldilocks Inline

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @jay-clarke                    |
| Created     | 2026-05-24                     |
| Revised     | 2026-06-10                     |
| Status      | proposed                       |
| PR          | #1570                          |

## Summary

Add a Jolt inline for the canonical 8-wide Poseidon2 permutation over the Goldilocks field.
Poseidon2-Goldilocks is a common ZK-native permutation used for proof-friendly commitments and
Merkle-tree style constructions. Executing it as ordinary guest Rust expands into many traced RISC-V
instructions; an inline lets Jolt recognize this specific operation and replace it with a
deterministic, tested virtual-instruction expansion.

## Intent

### Goal

Provide a `jolt-inlines-poseidon2-goldilocks` crate that exposes a guest-callable
`poseidon2_permute(&mut [u64; 8])`, registers a `Poseidon2Goldilocks` inline extension with the Jolt
tracer, and expands the custom instruction into a sequence that is byte-equivalent to Plonky3's
canonical `Poseidon2Goldilocks<8>` permutation.

### Invariants

1. The host reference implementation produces the same output as Plonky3's canonical
   `Poseidon2Goldilocks<8>` for every tested state.
2. The sequence-builder output, when executed through Jolt's inline emulator harness, produces the
   same output as the host reference implementation.
3. Goldilocks arithmetic stays in the field `p = 2^64 - 2^32 + 1`; multiplication reduction must
   match `u128` modular arithmetic for edge cases and random stress inputs.
4. Round constants are materialized in the same order as the permutation executes them: external
   initial constants, internal constants, then external final constants.
5. The internal diagonal matches Plonky3's `MATRIX_DIAG_8_GOLDILOCKS`.
6. The inline mutates only the 8-limb state buffer supplied by `rs1`; round constants are embedded
   in the inline expansion as virtual immediates.
7. The inline is gated behind a distinct `InlineExtension::Poseidon2Goldilocks` entry so profiles can
   opt into it explicitly.
8. The `(opcode, funct3, funct7)` encoding occupies a funct7 namespace not used by any existing
   `jolt-inlines-*` crate (see "Opcode allocation").

No `jolt-eval` invariant is proposed in this initial patch because existing inline crates primarily
validate these properties through crate-local unit and emulator tests. A follow-up could add a shared
inline-permutation equivalence invariant if maintainers want a broader framework-level check.

### Non-Goals

1. Supporting Poseidon2 widths other than 8.
2. Supporting fields other than Goldilocks.
3. Providing a sponge/hash API beyond the raw 8-limb permutation.
4. Changing existing Poseidon transcript code over BN254.
5. Replacing or modifying any existing hash/curve inline.
6. Claiming a specific performance improvement before benchmark review.

## Opcode allocation

The existing `jolt-inlines-*` crates follow a consistent convention under the shared
`INLINE_OPCODE = 0x0B`: each crate owns one `funct7` namespace, and `funct3` enumerates operations
within that crate. Current allocations on main:

| funct7 | Crate       | funct3 values in use                 |
|--------|-------------|--------------------------------------|
| 0x00   | sha2        | 0x00 (SHA256), 0x01 (SHA256_INIT)    |
| 0x01   | keccak256   | 0x00                                 |
| 0x02   | blake2      | 0x00                                 |
| 0x03   | blake3      | 0x00, 0x01 (KEYED64)                 |
| 0x04   | bigint      | 0x00                                 |
| 0x05   | secp256k1   | 0x00-0x07                            |
| 0x06   | grumpkin    | 0x00, 0x01                           |
| 0x07   | p256        | 0x00-0x07                            |

This crate therefore claims:

```text
POSEIDON2_GOLDILOCKS_FUNCT7 = 0x08
POSEIDON2_GOLDILOCKS_FUNCT3 = 0x00
```

which also matches the crate's `inline_extension_code = 8`. Future operations in this crate, such as
a fused two-to-one compression, should take `funct3 = 0x01, 0x02, ...` under `funct7 = 0x08`.

Pre-merge check: reconfirm the allocation against the authoritative dispatch table in
`jolt-inlines/sdk` on current main after rebase.

## Evaluation

### Acceptance Criteria

- [ ] `cargo check -p jolt-inlines-poseidon2-goldilocks` passes.
- [ ] `cargo check -p jolt-inlines-poseidon2-goldilocks --features host` passes.
- [ ] `cargo test -p jolt-inlines-poseidon2-goldilocks --features host` passes.
- [ ] `cargo test -p jolt-riscv` passes after adding the new inline extension.
- [ ] `cargo fmt --check` and workspace clippy pass on the rebased branch.
- [ ] Host permutation tests match Plonky3's default `Poseidon2Goldilocks<8>` path.
- [ ] Host permutation tests match an explicitly constructed generic Plonky3 `Poseidon2` path.
- [ ] Host permutation tests match committed known-answer vectors.
- [ ] Inline emulator tests match the host reference for fixed and randomized states.
- [ ] Goldilocks multiplication tests match `u128` modular arithmetic for edge and random stress cases.
- [ ] The new inline is registered through the existing `register_inlines!` mechanism.
- [ ] The `(0x0B, 0x00, 0x08)` encoding is confirmed free on rebased main.

### Testing Strategy

Existing tests that must continue passing:

- `cargo test -p jolt-riscv`

New tests added under `jolt-inlines/poseidon2-goldilocks`:

- Field multiplication reduction tests against `u128`.
- Plonky3 parity tests for all-zero, known, near-modulus, and randomized states.
- Known-answer tests: fixed `(input state -> output state)` vectors committed as constants.
- Round-constant layout tests against Plonky3 constants.
- Internal diagonal tests against Plonky3 constants.
- Sequence-builder determinism tests.
- Inline emulator tests for full permutation and isolated sub-operations.

Feature coverage is `--features host`, matching the existing inline-crate testing pattern. No `zk`
feature coverage is required for this crate-local inline expansion patch.

### Performance

This PR does not claim an end-to-end speedup over ordinary guest Rust. The initial goal is a
reviewable, deterministic, Plonky3-compatible inline surface that gives Jolt a dedicated hook for
future Poseidon2-specific optimization.

The current inline expansion emits **22,315 virtual instructions per permutation**, including the
virtual-register reset instructions appended by `finalize_inline`. The crate-local deterministic
emission test pins this count so future refactors cannot silently regress trace size.

As a local sanity check, the same permutation compiled as ordinary no-std guest Rust and measured
with Jolt's `start_cycle_tracking` / `end_cycle_tracking` markers emitted:

```text
"poseidon2_plain": 20504 RV64IMAC cycles + 8 virtual instructions = 20512 total cycles
```

That comparison is intentionally not a merge-blocking benchmark: it is a single traced-row count for
one marked guest region, not wall-clock time, prover time, padded proof trace length, or a complete
`jolt-eval` objective. It does show that this first inline is not yet smaller than optimized guest
Rust on raw traced-row count. The main value of this patch is correctness, API shape, opcode
allocation, and a deterministic expansion that can be improved behind the same guest call.

This count is produced by:

- embedding all 86 round constants as virtual immediates instead of loading them from guest memory,
- using the corrected Goldilocks `mul_mod` reduction path,
- using a no-left-alias `add_mod` variant at call sites where the destination cannot alias the left
  operand, and
- leaving the more invasive lazy-reduction optimization out of this initial PR.

The follow-up performance path is explicit: lazy reduction with a written bounds argument,
diagonal-constant specialization where profitable, a two-to-one compression operation for
Merkle-style use cases, and eventually a prover-side precompile if maintainers want Poseidon2 to
move beyond ordinary RV instruction semantics. If maintainers want a maintained comparison against
plain guest Rust, the natural follow-up is a `jolt-eval` objective comparing the two paths
end-to-end.

## Design

### Architecture

The change follows the existing `jolt-inlines/*` crate pattern:

- `jolt-inlines/poseidon2-goldilocks/src/sdk.rs` exposes the guest API. In RISC-V guest builds, it
  emits the custom inline instruction. In host builds, it calls the host reference implementation.
  On non-RISC-V, non-host targets it panics with a clear message.
- `jolt-inlines/poseidon2-goldilocks/src/exec.rs` contains the standalone host reference
  implementation of the 8-wide Poseidon2-Goldilocks permutation.
- `jolt-inlines/poseidon2-goldilocks/src/sequence_builder.rs` expands the inline instruction into
  virtual RISC-V instructions.
- `jolt-inlines/poseidon2-goldilocks/src/host.rs` registers the inline using the existing
  `register_inlines!` macro.
- `crates/jolt-riscv/src/profile.rs` adds `InlineExtension::Poseidon2Goldilocks`.
- The root workspace includes the new crate and adds Plonky3 crates as dev/test dependencies for
  parity checks.

### Instruction contract

```text
.insn r INLINE_OPCODE, POSEIDON2_GOLDILOCKS_FUNCT3, POSEIDON2_GOLDILOCKS_FUNCT7, x0, rs1, x0
```

- `rs1` points to 8 writable, 8-byte-aligned `u64` limbs representing the state, permuted in place.
- `rs2` is unused and should be encoded as `x0` by SDK callers.
- The inline writes only the 8-limb state buffer and reads no guest memory other than that state.
- Round constants are embedded in the expansion as virtual immediates.

### Alternatives Considered

1. **Leave Poseidon2 as ordinary guest Rust.** Rejected because Poseidon2-Goldilocks is a
   proof-native primitive likely to appear in Jolt guest programs; inline support gives Jolt a
   dedicated, testable hook for this hot operation and future optimization work.
2. **Expose a hash/sponge API instead of the raw permutation.** Rejected for the initial version.
   The raw permutation is the narrowest reusable primitive and avoids committing to one
   absorption/domain-separation policy.
3. **Support multiple widths immediately.** Rejected to keep the review surface small. Width 8 is
   the concrete Plonky3-compatible instance covered by the current implementation and tests.
4. **Use fixed test vectors only.** Considered as an alternative to Plonky3 dev dependencies. This
   patch does both: Plonky3 parity tests for breadth, plus committed known-answer vectors that
   survive if maintainers later choose to drop the Plonky3 dev-dependencies.
5. **Load round constants from a guest-memory table via `rs2`.** Rejected. The Jolt inline
   assembler can materialize a full `u64` immediate in one virtual `ADDI`, so a memory table does not
   reduce instruction count. Embedding constants also removes 86 guest-memory reads, avoids adding a
   688-byte constants table to guest data, and eliminates the possibility of callers passing the
   wrong `rs2` pointer.

## Documentation

No Jolt book changes are required for the initial patch because this adds an internal inline crate
and does not change user-facing Jolt APIs or examples. If maintainers want to advertise the inline,
a follow-up can add a short entry to the inline documentation alongside the existing hash and curve
inlines.

## Execution

The implementation should:

1. Add the new inline crate under `jolt-inlines/poseidon2-goldilocks`.
2. Add a `Poseidon2Goldilocks` inline extension entry.
3. Register the inline with `register_inlines!`.
4. Implement field addition, multiplication, S-box, external MDS, internal diffusion, and round
   scheduling for the 8-wide Goldilocks instance.
5. Encode at `(INLINE_OPCODE = 0x0B, funct3 = 0x00, funct7 = 0x08)` per "Opcode allocation".
6. Align SPDX headers with the crate license.
7. Keep the module documentation in sync with the `rs1`-only instruction contract.
8. Add parity, KAT, and emulator tests described above.
9. Rebase onto current main; rerun `fmt`/`clippy`/workspace tests; reconfirm the opcode table.

## References

- [Poseidon2 paper](https://eprint.iacr.org/2023/323)
- [Plonky3 repository](https://github.com/Plonky3/Plonky3)
- Existing Jolt inline crates under `jolt-inlines/`
