use core::arch::global_asm;

global_asm!(include_str!("memory.S"));
global_asm!(include_str!("compare.S"));
