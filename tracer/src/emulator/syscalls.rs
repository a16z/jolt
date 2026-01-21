//! Linux syscall handling for jolt-emu
//! Based on spike's approach - handle syscalls in the emulator, not the guest

use super::cpu::Cpu;
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::sync::OnceLock;

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
static BRK_POS: AtomicUsize = AtomicUsize::new(0);

static SYSCALL_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "std")]
fn trace_syscalls_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JOLT_TRACE_SYSCALLS").is_some())
}

/// Compute the usable heap range for Linux-style syscalls.
///
/// When running with a `JoltDevice`, the guest linker script reserves:
///   heap_start = RAM_START_ADDRESS + program_size + stack_size
///   heap_end   = heap_start + memory_size
///
/// This matches the emulator's `MemoryLayout` model. Use it for SYS_BRK/SYS_MMAP
/// so musl/Rust allocations stay within the configured RAM.
fn heap_range(cpu: &Cpu) -> Option<(usize, usize)> {
    let dev = cpu.mmu.jolt_device.as_ref()?;
    let layout = &dev.memory_layout;

    // In MemoryLayout::new():
    //   stack_end   = RAM_START_ADDRESS + program_size
    //   stack_start = stack_end + stack_size
    //   memory_end  = stack_start + memory_size
    let heap_start = layout
        .stack_end
        .checked_add(layout.stack_size)?
        .try_into()
        .ok()?;
    let heap_end: usize = layout.memory_end.try_into().ok()?;
    Some((heap_start, heap_end))
}

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
        let count = SYSCALL_COUNT.fetch_add(1, Ordering::Relaxed).wrapping_add(1);

        #[cfg(feature = "std")]
        if trace_syscalls_enabled() && count <= 200 {
            tracing::info!("syscall[{count}] nr={nr} a0=0x{a0:x} a1=0x{a1:x} a2=0x{a2:x}");
        }

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
                #[cfg(feature = "std")]
                if trace_syscalls_enabled() {
                    tracing::info!("SYSCALL EXIT: code={}", a0);
                }
                self.exit_code = Some(a0 as u32);
                0
            }
            SYS_BRK => {
                // sys_brk(addr)
                // If addr == 0: return current brk.
                // Else: attempt to set brk to addr; on failure, return current brk.
                let Some((heap_start, heap_end)) = heap_range(self) else {
                    return -ENOMEM;
                };

                // Initialize brk to heap_start on first use.
                let mut cur = BRK_POS.load(Ordering::Relaxed);
                if cur == 0 {
                    cur = heap_start;
                    BRK_POS.store(cur, Ordering::Relaxed);
                }

                let requested = a0;
                if requested == 0 {
                    return cur as i64;
                }

                // Linux brk semantics: fail by returning old break.
                if requested < heap_start || requested > heap_end {
                    return cur as i64;
                }

                // Move brk. We don't reclaim space on shrink.
                BRK_POS.store(requested, Ordering::Relaxed);

                // Ensure mmap bump pointer never hands out overlapping space.
                let used = requested.saturating_sub(heap_start);
                let prev = HEAP_POS.load(Ordering::Relaxed);
                if used > prev {
                    HEAP_POS.store(used, Ordering::Relaxed);
                }

                requested as i64
            }
            SYS_MMAP => {
                // sys_mmap(addr, length, prot, flags, fd, offset)
                // Simple bump allocator within the configured heap range.
                let Some((heap_start, heap_end)) = heap_range(self) else {
                    return -ENOMEM;
                };

                let length = a1;
                let page_size = 4096usize;
                let size = length
                    .checked_add(page_size - 1)
                    .map(|n| n / page_size)
                    .and_then(|pages| pages.checked_mul(page_size))
                    .unwrap_or(0);
                if size == 0 {
                    return -EINVAL;
                }

                let offset = HEAP_POS.fetch_add(size, Ordering::Relaxed);
                let ptr = heap_start.saturating_add(offset);
                let end = ptr.saturating_add(size);
                if ptr < heap_start || end > heap_end {
                    // Roll back is non-trivial with Atomics; just signal OOM.
                    return -ENOMEM;
                }

                // Memory is initially zeroed by MMU initialization, so no need to clear.
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
