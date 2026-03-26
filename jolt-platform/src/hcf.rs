/// Halt-and-catch-fire: makes proof unsatisfiable.
/// On RISC-V, emits a VirtualAssertEQ(0, 1) that the prover cannot satisfy, then panics.
/// On all other targets, panics directly.
#[inline(always)]
pub fn hcf() -> ! {
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
    panic!("hcf: proof spoiled");
}
