/// Makes proof unsatisfiable.
/// On RISC-V, emits a VirtualAssertEQ(0, 1) that the prover cannot satisfy, then panics.
/// On all other targets, panics directly.
#[inline(always)]
pub fn spoil_proof() -> ! {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        let u = 0u64;
        let v = 1u64;
        core::arch::asm!(
            ".insn b {opcode}, {funct3}, {rs1}, {rs2}, . + 2",
            opcode = const 0x5B,
            funct3 = const 0b001,
            rs1 = in(reg) u,
            rs2 = in(reg) v,
            options(nostack)
        );
    }
    panic!("proof spoiled");
}

/// Unwrap `Result` or `Option`, spoiling the proof on error/None instead of producing a valid proof of panic.
///
/// Use when a malicious prover should not be able to produce any proof if the condition fails
/// (e.g. cryptographic assertions). Do NOT use for input validation or expected error cases.
pub trait UnwrapOrSpoilProof<T> {
    fn unwrap_or_spoil_proof(self) -> T;
}

impl<T, E> UnwrapOrSpoilProof<T> for Result<T, E> {
    #[inline(always)]
    fn unwrap_or_spoil_proof(self) -> T {
        match self {
            Ok(v) => v,
            Err(_) => spoil_proof(),
        }
    }
}

impl<T> UnwrapOrSpoilProof<T> for Option<T> {
    #[inline(always)]
    fn unwrap_or_spoil_proof(self) -> T {
        match self {
            Some(v) => v,
            None => spoil_proof(),
        }
    }
}
