# Spec: Poseidon2-Goldilocks Inline

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @jay-clarke                    |
| Created     | 2026-05-24                     |
| Status      | proposed                       |
| PR          | #1570                          |

## Summary

Add a Jolt inline for the canonical 8-wide Poseidon2 permutation over the Goldilocks field. Poseidon2-Goldilocks is a common ZK-native permutation used for proof-friendly commitments and Merkle-tree style constructions. Executing it as ordinary guest Rust expands into many traced RISC-V instructions; an inline lets Jolt recognize this specific operation and replace it with a deterministic, tested virtual-instruction expansion.

## Intent

### Goal

Provide a `jolt-inlines-poseidon2-goldilocks` crate that exposes a guest-callable `poseidon2_permute(&mut [u64; 8])`, registers a `Poseidon2Goldilocks` inline extension with the Jolt tracer, and expands the custom instruction into a sequence that is byte-equivalent to Plonky3's canonical `Poseidon2Goldilocks<8>` permutation.

### Invariants

1. The host reference implementation produces the same output as Plonky3's canonical `Poseidon2Goldilocks<8>` for every tested state.
2. The sequence-builder output, when executed through Jolt's inline emulator harness, produces the same output as the host reference implementation.
3. Goldilocks arithmetic stays in the field `p = 2^64 - 2^32 + 1`; multiplication reduction must match `u128` modular arithmetic for edge cases and random stress inputs.
4. Round constants are loaded in the same order as the permutation executes them: external initial constants, internal constants, then external final constants.
5. The internal diagonal matches Plonky3's `MATRIX_DIAG_8_GOLDILOCKS`.
6. The inline mutates only the 8-limb state buffer supplied by `rs1` and reads round constants from the table supplied by `rs2`.
7. The inline is gated behind a distinct `InlineExtension::Poseidon2Goldilocks` entry so profiles can opt into it explicitly.

No `jolt-eval` invariant is proposed in this initial patch because existing inline crates primarily validate these properties through crate-local unit and emulator tests. A follow-up could add a shared inline-permutation equivalence invariant if maintainers want a broader framework-level check.

### Non-Goals

1. Supporting Poseidon2 widths other than 8.
2. Supporting fields other than Goldilocks.
3. Providing a sponge/hash API beyond the raw 8-limb permutation.
4. Changing existing Poseidon transcript code over BN254.
5. Replacing or modifying any existing hash/curve inline.
6. Claiming a specific performance improvement before benchmark review.

## Evaluation

### Acceptance Criteria

- [ ] `cargo check -p jolt-inlines-poseidon2-goldilocks` passes.
- [ ] `cargo check -p jolt-inlines-poseidon2-goldilocks --features host` passes.
- [ ] `cargo test -p jolt-inlines-poseidon2-goldilocks --features host` passes.
- [ ] `cargo test -p jolt-riscv` passes after adding the new inline extension.
- [ ] Host permutation tests match Plonky3's default `Poseidon2Goldilocks<8>` path.
- [ ] Host permutation tests match an explicitly constructed generic Plonky3 `Poseidon2` path.
- [ ] Inline emulator tests match the host reference for fixed and randomized states.
- [ ] Goldilocks multiplication tests match `u128` modular arithmetic for edge and random stress cases.
- [ ] The new inline is registered through the existing `register_inlines!` mechanism.

### Testing Strategy

Existing tests that must continue passing:

- `cargo test -p jolt-riscv`

New tests added under `jolt-inlines/poseidon2-goldilocks`:

- Field multiplication reduction tests against `u128`.
- Plonky3 parity tests for all-zero, known, near-modulus, and randomized states.
- Round-constant layout tests against Plonky3 constants.
- Internal diagonal tests against Plonky3 constants.
- Sequence-builder determinism tests.
- Inline emulator tests for full permutation and isolated sub-operations.

Feature coverage is `--features host`, matching the existing inline-crate testing pattern. No `zk` feature coverage is required for this crate-local inline expansion patch.

### Performance

The expected direction is fewer traced instructions than executing the same Poseidon2 permutation as ordinary guest Rust. This PR does not set a hard speedup target. If maintainers want a merge-blocking benchmark, the natural follow-up is a small benchmark comparing:

1. plain guest Rust Poseidon2-Goldilocks-8 execution, and
2. the inline opcode expanded through `Poseidon2GoldilocksPermutation::build_sequence`.

No existing `jolt-eval` objective is modified in this initial patch.

## Design

### Architecture

The change follows the existing `jolt-inlines/*` crate pattern:

- `jolt-inlines/poseidon2-goldilocks/src/sdk.rs` exposes the guest API. In RISC-V guest builds, it emits a custom inline instruction. In host builds, it calls the host reference implementation.
- `jolt-inlines/poseidon2-goldilocks/src/exec.rs` contains the standalone host reference implementation of the 8-wide Poseidon2-Goldilocks permutation.
- `jolt-inlines/poseidon2-goldilocks/src/sequence_builder.rs` expands the inline instruction into virtual RISC-V instructions.
- `jolt-inlines/poseidon2-goldilocks/src/host.rs` registers the inline using the existing `register_inlines!` macro.
- `crates/jolt-riscv/src/profile.rs` adds `InlineExtension::Poseidon2Goldilocks`.
- The root workspace includes the new crate and adds Plonky3 crates as dev/test dependencies for parity checks.

The instruction contract is:

- `rs1` points to 8 writable `u64` limbs representing the state.
- `rs2` points to the static 86-element round-constant table.
- The operation permutes the state in place.

### Alternatives Considered

1. **Leave Poseidon2 as ordinary guest Rust.** Rejected because Poseidon2-Goldilocks is a proof-native primitive likely to appear in Jolt guest programs; inline support should reduce trace size for this hot operation.
2. **Expose a hash/sponge API instead of the raw permutation.** Rejected for the initial version. The raw permutation is the narrowest reusable primitive and avoids committing to one absorption/domain-separation policy.
3. **Support multiple widths immediately.** Rejected to keep the review surface small. Width 8 is the concrete Plonky3-compatible instance covered by the current implementation and tests.
4. **Use fixed test vectors only.** Considered as an alternative to Plonky3 dev dependencies. The current patch uses Plonky3 directly for stronger parity coverage, but fixed vectors would be a reasonable maintainer preference if dependency surface is a concern.

## Documentation

No Jolt book changes are required for the initial patch because this adds an internal inline crate and does not change user-facing Jolt APIs or examples. If maintainers want to advertise the inline, a follow-up can add a short entry to the inline documentation alongside the existing hash and curve inlines.

## Execution

The implementation should:

1. Add the new inline crate under `jolt-inlines/poseidon2-goldilocks`.
2. Add a `Poseidon2Goldilocks` inline extension entry.
3. Register the inline with `register_inlines!`.
4. Implement field addition, multiplication, S-box, external MDS, internal diffusion, and round scheduling for the 8-wide Goldilocks instance.
5. Add parity and emulator tests described above.

## References

- [Poseidon2 paper](https://eprint.iacr.org/2023/323)
- [Plonky3 repository](https://github.com/Plonky3/Plonky3)
- Existing Jolt inline crates under `jolt-inlines/`
