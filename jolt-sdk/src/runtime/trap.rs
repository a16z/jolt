extern crate zeroos;

use zeroos::arch::riscv::TrapFrame;

use riscv::interrupt::machine::Exception;

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

/// RISC-V Linux syscall numbers the trap handler answers itself.
const SYS_WRITE: usize = 64;
const SYS_CLOCK_GETTIME: usize = 113;

/// # Safety
/// `regs` must be a non-null pointer to a valid `TrapFrame` for the current CPU trap context.
#[no_mangle]
pub unsafe extern "C" fn trap_handler(regs: *mut u8) {
    let regs = regs as *mut TrapFrame;
    let mcause = (*regs).mcause;
    if mcause_is_interrupt(mcause) {
        return;
    }

    match mcause_code(mcause) {
        code if code == (Exception::UserEnvCall as usize)
            || code == (Exception::SupervisorEnvCall as usize)
            || code == (Exception::MachineEnvCall as usize) =>
        {
            let pc = (*regs).mepc;
            (*regs).mepc = pc + 4;

            #[cfg(feature = "debug")]
            {
                let nr = (*regs).a7;
                zeroos::debug::writeln!("[syscall] {}", zeroos::os::linux::syscall_name(nr));
            }

            // write(2) to stdout or stderr: route the bytes to the host console,
            // so std guests' `println!` and panic messages reach the tracer
            // instead of failing with ENOSYS and aborting silently. Without the
            // `stdout` feature ZeroOS has no console; with it, this takes the
            // console's path (one host call per write, keeping multi-byte UTF-8
            // intact) and skips the VFS. Descriptors 1 and 2 always reach the
            // host, even if the guest closed or redirected them.
            if (*regs).a7 == SYS_WRITE && ((*regs).a0 == 1 || (*regs).a0 == 2) {
                super::__platform_stdout_write((*regs).a1 as *const u8, (*regs).a2);
                (*regs).a0 = (*regs).a2;
                return;
            }
            // clock_gettime(2): the zkVM has no clock, so every clock reads as
            // a zero timespec rather than failing with ENOSYS; std's
            // `Instant::now` aborts on failure, and verifier code may take
            // timestamps for diagnostics. A fixed value also keeps execution
            // deterministic. Documented for guest authors under "Time" in
            // book/src/usage/guests_hosts/guests.md.
            if (*regs).a7 == SYS_CLOCK_GETTIME {
                let ts = (*regs).a1 as *mut u64;
                if !ts.is_null() {
                    ts.write(0);
                    ts.add(1).write(0);
                }
                (*regs).a0 = 0;
                return;
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
