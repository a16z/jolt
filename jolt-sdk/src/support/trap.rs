extern crate zeroos;

use zeroos::arch::riscv::TrapFrame;

use riscv::register::mcause::Exception;

#[inline(always)]
fn mcause_is_interrupt(mcause: usize) -> bool {
    mcause >> (usize::BITS as usize - 1) != 0
}

#[inline(always)]
fn mcause_code(mcause: usize) -> usize {
    // RISC-V encodes interrupts by setting the top bit of mcause; the rest is the code.
    mcause & ((1usize << (usize::BITS as usize - 1)) - 1)
}

#[inline(always)]
fn advance_mepc_for_breakpoint(regs: *mut TrapFrame) {
    unsafe {
        let pc = (*regs).mepc;
        (*regs).mepc = pc.wrapping_add(instr_len(pc));
    }
}

#[inline(always)]
fn instr_len(addr: usize) -> usize {
    let halfword = unsafe { core::ptr::read_unaligned(addr as *const u16) };
    if (halfword & 0b11) == 0b11 {
        4
    } else {
        2
    }
}

/// # Safety
/// `regs` must be a non-null pointer to a valid `TrapFrame` for the current CPU trap context.
#[no_mangle]
pub unsafe extern "C" fn trap_handler(regs: *mut u8) {
    let regs = regs as *mut TrapFrame;
    let mcause = (*regs).mcause;
    if mcause_is_interrupt(mcause) {
        // Interrupt handling is disabled
        return;
    }

    match mcause_code(mcause) {
        // Handle envcalls (syscalls) from any privilege mode.
        code if code == (Exception::UserEnvCall as usize)
            || code == (Exception::SupervisorEnvCall as usize)
            || code == (Exception::MachineEnvCall as usize) =>
        {
            let pc = (*regs).mepc;
            (*regs).mepc = pc + 4;

            #[cfg(feature = "debug")]
            {
                let nr = (*regs).a7;
                zeroos::debug::writeln!(
                    "[syscall] {}",
                    zeroos::os::linux::syscall_name(nr)
                );
            }

            let ret = zeroos::foundation::kfn::trap::ksyscall(
                (*regs).a0,
                (*regs).a1,
                (*regs).a2,
                (*regs).a3,
                (*regs).a4,
                (*regs).a5,
                (*regs).a7,
            );
            (*regs).a0 = ret as usize;
        }
        code if code == (Exception::Breakpoint as usize) => {
            advance_mepc_for_breakpoint(regs);
        }
        code => {
            zeroos::foundation::kfn::kexit(code as i32);
        }
    }
}
