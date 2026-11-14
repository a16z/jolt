/// Ends the run at `ebreak`, Jolt's failure halt, for every `code`.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[no_mangle]
pub extern "C" fn platform_exit(_code: i32) -> ! {
    unsafe {
        core::arch::asm!("ebreak", options(noreturn));
    }
}
