//! Data-independent timing for secret-dependent prover code.
//!
//! Arm's FEAT_DIT adds the PSTATE bit DIT. While it is set, the instructions
//! the architecture lists as DIT-affected (among them integer arithmetic,
//! logic, shifts, conditional selects and most Advanced SIMD data
//! processing) take time independent of their operand values. Without it a
//! core may, and Apple M-series cores do, time some of them by their
//! operands. DIT is per-thread state that the kernel saves and restores
//! across context switches, so every thread that touches a secret must set
//! it: [`DitGuard`] does so for the current thread, [`install`] for a rayon
//! pool's workers.
//!
//! On aarch64 cores without FEAT_DIT, and on every other target, both are
//! no-ops and [`is_enabled`] is `false`. x86's counterpart, DOITM, is a
//! model-specific register that only the kernel can write; enabling it is
//! out of scope here.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;
use std::marker::PhantomData;

use rayon::{ThreadPoolBuildError, ThreadPoolBuilder};

/// Sets DIT on the current thread while alive, then restores the state it
/// found. Guards nest: dropping an inner guard leaves DIT set for the outer
/// one. Not `Send`, since DIT belongs to the thread that set it.
///
/// WARNING: setting the bit is a compiler barrier for memory accesses only.
/// Code that must run under DIT belongs in a call made while the guard is
/// alive, not in arithmetic the optimizer could move across its creation.
#[must_use = "DIT is restored when the guard is dropped"]
pub struct DitGuard {
    previous: bool,
    _thread: PhantomData<*const ()>,
}

impl DitGuard {
    /// Sets DIT on the current thread, where FEAT_DIT exists.
    pub fn enable() -> Self {
        let previous = is_enabled();
        if is_supported() {
            // SAFETY: the core implements FEAT_DIT (checked just above).
            unsafe { arch::write(true) };
        }
        Self {
            previous,
            _thread: PhantomData,
        }
    }
}

impl Drop for DitGuard {
    fn drop(&mut self) {
        if is_supported() {
            // SAFETY: the core implements FEAT_DIT (checked just above).
            unsafe { arch::write(self.previous) };
        }
    }
}

/// Whether the core implements FEAT_DIT; always `false` off aarch64.
pub fn is_supported() -> bool {
    arch::supported()
}

/// Whether DIT is set on the current thread.
pub fn is_enabled() -> bool {
    // SAFETY: the core implements FEAT_DIT (the left operand).
    is_supported() && unsafe { arch::read() }
}

/// Runs `op` in a fresh rayon pool of `rayon::current_num_threads()`
/// workers, each holding a [`DitGuard`] for its whole life, and returns its
/// result. `op` itself runs on a worker, so it and every rayon job it
/// spawns run under DIT; work it sends elsewhere (another pool, a
/// `std::thread`) does not.
pub fn install<R: Send>(op: impl FnOnce() -> R + Send) -> Result<R, ThreadPoolBuildError> {
    thread_local! {
        /// The worker's guard, dropped with the thread.
        static WORKER_GUARD: Cell<Option<DitGuard>> = const { Cell::new(None) };
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .thread_name(|index| format!("jolt-dit-{index}"))
        .start_handler(|_| WORKER_GUARD.set(Some(DitGuard::enable())))
        .build()?;
    Ok(pool.install(op))
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use std::arch::asm;

    /// PSTATE.DIT as read through the `DIT` system register.
    const DIT_BIT: u64 = 1 << 24;

    pub(crate) fn supported() -> bool {
        std::arch::is_aarch64_feature_detected!("dit")
    }

    /// # Safety
    ///
    /// The core implements FEAT_DIT; otherwise the instruction is undefined.
    pub(crate) unsafe fn read() -> bool {
        let value: u64;
        // SAFETY: `mrs <Xt>, DIT` (spelled by its encoding, S3_3_C4_C2_5,
        // so no assembler feature flag is needed) only reads PSTATE.DIT into
        // a register; the caller guarantees FEAT_DIT.
        unsafe {
            asm!(
                "mrs {value}, s3_3_c4_c2_5",
                value = out(reg) value,
                options(nomem, nostack, preserves_flags),
            );
        }
        value & DIT_BIT != 0
    }

    /// # Safety
    ///
    /// The core implements FEAT_DIT; otherwise the instruction is undefined.
    pub(crate) unsafe fn write(enabled: bool) {
        // SAFETY: `msr DIT, #imm` (encodings 0xd503415f for #1, 0xd503405f
        // for #0) writes only PSTATE.DIT of this thread and touches no
        // memory, stack or condition flags; the caller guarantees FEAT_DIT.
        // `nomem` is left off so the compiler does not move memory accesses
        // across the write.
        unsafe {
            if enabled {
                asm!(".inst 0xd503415f", options(nostack, preserves_flags));
            } else {
                asm!(".inst 0xd503405f", options(nostack, preserves_flags));
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod arch {
    pub(crate) fn supported() -> bool {
        false
    }

    /// # Safety
    ///
    /// Never called: [`supported`] is `false`.
    pub(crate) unsafe fn read() -> bool {
        false
    }

    /// # Safety
    ///
    /// Never called: [`supported`] is `false`.
    pub(crate) unsafe fn write(_enabled: bool) {}
}

#[cfg(test)]
mod tests {
    use rayon::ThreadPoolBuildError;

    use super::{install, is_enabled, is_supported, DitGuard};

    #[test]
    fn guard_sets_and_restores() {
        let before = is_enabled();
        {
            let _outer = DitGuard::enable();
            assert_eq!(is_enabled(), is_supported());
            {
                let _inner = DitGuard::enable();
                assert_eq!(is_enabled(), is_supported());
            }
            assert_eq!(is_enabled(), is_supported(), "inner drop kept outer DIT");
        }
        assert_eq!(is_enabled(), before);
    }

    #[test]
    fn install_sets_dit_on_every_worker_only() -> Result<(), ThreadPoolBuildError> {
        let before = is_enabled();
        let threads = rayon::current_num_threads();
        let workers = install(|| rayon::broadcast(|_| is_enabled()))?;
        assert_eq!(workers.len(), threads);
        assert!(workers.iter().all(|&enabled| enabled == is_supported()));
        assert_eq!(is_enabled(), before);
        Ok(())
    }
}
