//! Linux syscall handling for jolt-emu
//! Based on spike's approach - handle syscalls in the emulator, not the guest

use super::cpu::Cpu;
use core::sync::atomic::{AtomicUsize, Ordering};

// RISC-V Linux syscall numbers
const SYS_WRITE: i64 = 64;
const SYS_EXIT: i64 = 93;
const SYS_EXIT_GROUP: i64 = 94;
const SYS_BRK: i64 = 214;
const SYS_MMAP: i64 = 222;
const SYS_MUNMAP: i64 = 215;
const SYS_MPROTECT: i64 = 226;
const SYS_SET_TID_ADDRESS: i64 = 96;
const SYS_RT_SIGACTION: i64 = 134;
const SYS_RT_SIGPROCMASK: i64 = 135;
const SYS_PRLIMIT64: i64 = 261;
const SYS_GETRANDOM: i64 = 278;
const SYS_CLOCK_GETTIME: i64 = 113;
const SYS_IOCTL: i64 = 29;
const SYS_FSTAT: i64 = 80;
const SYS_WRITEV: i64 = 66;
const SYS_READLINKAT: i64 = 78;
const SYS_FUTEX: i64 = 98;

// Errno values
const EBADF: i64 = 9;
const ENOMEM: i64 = 12;
const ENOTTY: i64 = 25;
const EINVAL: i64 = 22;
const ENOSYS: i64 = 38;

// Heap allocator state (shared across all Cpu instances)
static HEAP_POS: AtomicUsize = AtomicUsize::new(0);

impl Cpu {
    pub fn handle_syscall(
        &mut self,
        nr: i64,
        a0: usize,
        a1: usize,
        a2: usize,
        _a3: usize,
        _a4: usize,
        _a5: usize,
    ) -> i64 {
        match nr {
            SYS_WRITE => {
                // sys_write(fd, buf, count)
                if a0 == 1 || a0 == 2 {
                    // stdout/stderr
                    for i in 0..a2 {
                        if let Ok((byte, _)) = self.mmu.load((a1 + i) as u64) {
                            print!("{}", byte as char);
                        }
                    }
                    a2 as i64
                } else {
                    -EBADF
                }
            }
            SYS_EXIT | SYS_EXIT_GROUP => {
                // sys_exit / sys_exit_group
                tracing::info!("SYSCALL EXIT: exit_code={}", a0);
                self.exit_code = Some(a0 as u32);
                0
            }
            SYS_BRK => {
                // sys_brk - force musl to use mmap
                -ENOMEM
            }
            SYS_MMAP => {
                // sys_mmap(addr, length, prot, flags, fd, offset)
                // Simple bump allocator
                let length = a1;
                let page_size = 4096;
                let pages = (length + page_size - 1) / page_size;
                let size = pages * page_size;

                let offset = HEAP_POS.fetch_add(size, Ordering::Relaxed);

                // Use a heap region within the valid RAM range (0x80000000 to 0x87a12000)
                // Start after stack which ends around 0x85c8b000, use 0x86000000 for safety
                const HEAP_START: usize = 0x86000000;
                const HEAP_SIZE: usize = 26 * 1024 * 1024; // ~26MB (up to 0x87a12000)

                if offset + size > HEAP_SIZE {
                    return -ENOMEM;
                }

                let ptr = HEAP_START + offset;
                // Zero the memory
                for i in 0..size {
                    let _ = self.mmu.store((ptr + i) as u64, 0);
                }
                ptr as i64
            }
            SYS_MUNMAP | SYS_MPROTECT => {
                // sys_munmap / sys_mprotect
                0
            }
            SYS_SET_TID_ADDRESS => {
                // sys_set_tid_address - return a fake TID
                1
            }
            SYS_RT_SIGACTION | SYS_RT_SIGPROCMASK | SYS_PRLIMIT64 | SYS_FUTEX => {
                // Signal handling / resource limits / futex - return success
                0
            }
            SYS_GETRANDOM => {
                // sys_getrandom(buf, len, flags) - provide deterministic random
                for i in 0..a1 {
                    let _ = self.mmu.store((a0 + i) as u64, 0x42);
                }
                a1 as i64
            }
            SYS_CLOCK_GETTIME => {
                // sys_clock_gettime - return zero time
                // timespec is two i64s: tv_sec, tv_nsec
                let _ = self.mmu.store_doubleword(a1 as u64, 0); // tv_sec
                let _ = self.mmu.store_doubleword((a1 + 8) as u64, 0); // tv_nsec
                0
            }
            SYS_IOCTL => {
                // sys_ioctl - return ENOTTY (not a terminal)
                -ENOTTY
            }
            SYS_FSTAT => {
                // sys_fstat - minimal stat for stdout/stderr
                if a0 == 1 || a0 == 2 {
                    // Zero the struct and set st_mode to character device
                    for i in 0..144 {
                        // sizeof(struct stat) = 144
                        let _ = self.mmu.store((a1 + i) as u64, 0);
                    }
                    // st_mode is at offset 16 (u32)
                    let st_mode = 0x2000u32; // S_IFCHR
                    let _ = self.mmu.store_word((a1 + 16) as u64, st_mode);
                    // st_blksize is at offset 56 (i64)
                    let _ = self.mmu.store_doubleword((a1 + 56) as u64, 4096);
                    0
                } else {
                    -EBADF
                }
            }
            SYS_WRITEV => {
                // sys_writev - write multiple buffers
                if a0 == 1 || a0 == 2 {
                    let iov_cnt = a2;
                    let mut total_written = 0;

                    for i in 0..iov_cnt {
                        // struct iovec { void *iov_base; size_t iov_len; }
                        let iov_addr = a1 + i * 16; // sizeof(iovec) = 16
                        let (iov_base, _) = self.mmu.load_doubleword(iov_addr as u64).unwrap_or((0, Default::default()));
                        let (iov_len, _) = self.mmu.load_doubleword((iov_addr + 8) as u64).unwrap_or((0, Default::default()));

                        for j in 0..iov_len {
                            if let Ok((byte, _)) = self.mmu.load((iov_base + j) as u64) {
                                print!("{}", byte as char);
                            }
                        }
                        total_written += iov_len;
                    }
                    total_written as i64
                } else {
                    -EBADF
                }
            }
            SYS_READLINKAT => {
                // sys_readlinkat - return EINVAL
                -EINVAL
            }
            _ => {
                tracing::warn!("Unknown syscall: {}", nr);
                -ENOSYS
            }
        }
    }
}
