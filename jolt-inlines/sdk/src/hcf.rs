/// Halt-and-catch-fire: makes proof unsatisfiable.
/// On RISC-V guest builds, emits an illegal branch that the prover cannot satisfy.
/// On all other targets, panics.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub fn hcf() {
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
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
pub fn hcf() {
    panic!("hcf called on non-RISC-V target without host feature");
}

#[cfg(feature = "host")]
pub fn hcf() {
    panic!("explicit host code panic function called");
}
