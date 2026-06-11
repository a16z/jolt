// SPDX-License-Identifier: MIT

//! Public Poseidon2-Goldilocks API for guests and hosts.
//!
//! In a RISC-V guest build (no_std), `poseidon2_permute` emits the
//! custom inline opcode and the Jolt prover dispatches it to the
//! `sequence_builder`.
//!
//! In a host build (feature = "host"), `poseidon2_permute` calls the
//! reference implementation in `exec.rs`.
//!
//! On non-RISC-V non-host targets (rare — basically tooling builds
//! that don't enable the `host` feature) the function panics with a
//! clear message.

use crate::Poseidon2GoldilocksState;

/// Permute an 8-element Goldilocks state in place.
///
/// # Safety
///
/// `state` must point to exactly `STATE_WIDTH` (= 8) contiguous u64
/// values that are writable for the duration of the call. The pointer
/// must be 8-byte aligned.
#[inline(always)]
pub fn poseidon2_permute(state: &mut Poseidon2GoldilocksState) {
    unsafe {
        poseidon2_permute_inner(state.as_mut_ptr());
    }
}

// ────────────────────────────────────────────────────────────────────────
// Custom inline opcode dispatch
// ────────────────────────────────────────────────────────────────────────

/// RISC-V guest path: emit the custom inline opcode.
///
/// Memory contract enforced by the sequence builder:
/// - `rs1` → pointer to the 8-element state (read+written in place).
/// - Round constants are embedded in the inline expansion as virtual
///   immediates; `rs2` is unused.
///
/// # Safety
///
/// `state` must be a valid, 8-byte-aligned pointer to 64 bytes of
/// readable+writable memory.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
unsafe fn poseidon2_permute_inner(state: *mut u64) {
    use crate::{INLINE_OPCODE, POSEIDON2_GOLDILOCKS_FUNCT3, POSEIDON2_GOLDILOCKS_FUNCT7};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, x0",
        opcode = const INLINE_OPCODE,
        funct3 = const POSEIDON2_GOLDILOCKS_FUNCT3,
        funct7 = const POSEIDON2_GOLDILOCKS_FUNCT7,
        rs1 = in(reg) state,
        options(nostack)
    );
}

/// Host path: dispatch to the reference implementation in `exec.rs`.
#[cfg(feature = "host")]
#[inline(always)]
unsafe fn poseidon2_permute_inner(state: *mut u64) {
    let slice = core::slice::from_raw_parts_mut(state, crate::STATE_WIDTH);
    let arr: &mut [u64; 8] = slice
        .try_into()
        .expect("Poseidon2 state must be exactly 8 u64 elements");
    crate::exec::execute_poseidon2_permutation(arr);
}

/// Non-RISC-V, non-host targets: fail loudly.
#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
#[inline(always)]
unsafe fn poseidon2_permute_inner(_state: *mut u64) {
    panic!(
        "poseidon2_permute requires either the `host` feature or a \
         RISC-V target. Add `features = [\"host\"]` for tooling builds."
    );
}
