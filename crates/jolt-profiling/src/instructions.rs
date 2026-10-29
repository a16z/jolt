//! Process-wide count of user-space instructions retired, from a Linux
//! hardware performance counter (`perf_event_open(2)`).
//!
//! [`setup_tracing`](crate::setup_tracing) opens one counter on its calling
//! thread with `inherit` set, so it also counts every thread and child
//! process that thread transitively spawns afterwards — rayon's global pool
//! included, provided the pool is first used after setup. Threads that
//! already exist are not counted; setup opens the counter after spawning the
//! chrome writer and metrics-monitor threads to keep the profiler's own work
//! out of the counts.
//!
//! Readings are served only on that opening thread. A process-wide delta is
//! a span's own cost only while nothing else runs, which holds for the
//! sequential orchestration on the opening thread (its parallel work runs
//! on pool threads inside the span) but not for spans on pool threads,
//! which overlap their siblings. A read sums the live inherited children in
//! one syscall (~6 µs with 16 threads on a Zen 4 VM).
//!
//! Requires a PMU the kernel exposes and `perf_event_paranoid <= 2`
//! (kernel-mode instructions are excluded). Elsewhere opening fails and
//! every reading is absent.

use std::cell::Cell;
use std::fs::File;
use std::io::{Read, Result};
use std::sync::OnceLock;

static PROCESS_COUNTER: OnceLock<InstructionCounter> = OnceLock::new();

thread_local! {
    static ON_OPENING_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Opens the process counter on the calling thread.
pub(crate) fn open_process_counter() -> Result<()> {
    let counter = InstructionCounter::open()?;
    // `setup_tracing` installs the global subscriber, so it runs once per
    // process; a second counter would have nothing to add.
    let _ = PROCESS_COUNTER.set(counter);
    ON_OPENING_THREAD.set(true);
    Ok(())
}

/// The process counter's current value; `None` off the opening thread, if
/// the counter was never opened, or if the read failed.
pub(crate) fn read_process_counter() -> Option<InstructionReading> {
    // `try_with`: spans can close during thread-local teardown.
    if !ON_OPENING_THREAD.try_with(Cell::get).unwrap_or(false) {
        return None;
    }
    PROCESS_COUNTER.get()?.read().ok()
}

/// One counter sample. The kernel's enabled and running times diverge only
/// while it multiplexes the counter off the PMU, during which the count
/// misses instructions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InstructionReading {
    count: u64,
    time_enabled_ns: u64,
    time_running_ns: u64,
}

impl InstructionReading {
    /// Instructions retired between `earlier` and `self`; `None` when the
    /// counter was multiplexed during the interval, so the difference would
    /// undercount.
    pub(crate) fn instructions_since(self, earlier: Self) -> Option<u64> {
        let enabled = self.time_enabled_ns.checked_sub(earlier.time_enabled_ns)?;
        let running = self.time_running_ns.checked_sub(earlier.time_running_ns)?;
        if enabled != running {
            return None;
        }
        self.count.checked_sub(earlier.count)
    }
}

struct InstructionCounter(File);

impl InstructionCounter {
    fn open() -> Result<Self> {
        perf::open_instruction_counter().map(Self)
    }

    /// Decodes the `read_format` layout requested at open: value, then
    /// enabled and running time, as native-endian `u64`s.
    fn read(&self) -> Result<InstructionReading> {
        let mut buf = [0u8; 24];
        (&self.0).read_exact(&mut buf)?;
        let word = |i: usize| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&buf[8 * i..8 * (i + 1)]);
            u64::from_ne_bytes(bytes)
        };
        Ok(InstructionReading {
            count: word(0),
            time_enabled_ns: word(1),
            time_running_ns: word(2),
        })
    }
}

/// `PerfEventAttr::flags` packs the C bitfield, and bitfield bit order
/// follows endianness; only the little-endian allocation is encoded here.
#[cfg(all(target_os = "linux", target_endian = "little"))]
mod perf {
    use std::fs::File;
    use std::io::{Error, Result};
    use std::os::fd::{FromRawFd, RawFd};

    /// `struct perf_event_attr` truncated to `PERF_ATTR_SIZE_VER0` (64
    /// bytes); the kernel zero-extends the newer fields.
    #[repr(C)]
    #[derive(Default)]
    struct PerfEventAttr {
        kind: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events: u32,
        bp_type: u32,
        config1: u64,
    }

    const PERF_TYPE_HARDWARE: u32 = 0;
    const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
    const PERF_FORMAT_TOTAL_TIME_ENABLED: u64 = 1 << 0;
    const PERF_FORMAT_TOTAL_TIME_RUNNING: u64 = 1 << 1;
    const ATTR_FLAG_INHERIT: u64 = 1 << 1;
    const ATTR_FLAG_EXCLUDE_KERNEL: u64 = 1 << 5;
    const ATTR_FLAG_EXCLUDE_HV: u64 = 1 << 6;
    const PERF_FLAG_FD_CLOEXEC: libc::c_ulong = 1 << 3;

    pub(super) fn open_instruction_counter() -> Result<File> {
        let attr = PerfEventAttr {
            kind: PERF_TYPE_HARDWARE,
            size: size_of::<PerfEventAttr>() as u32,
            config: PERF_COUNT_HW_INSTRUCTIONS,
            read_format: PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
            flags: ATTR_FLAG_INHERIT | ATTR_FLAG_EXCLUDE_KERNEL | ATTR_FLAG_EXCLUDE_HV,
            ..PerfEventAttr::default()
        };
        // Calling thread, any CPU, no group leader.
        let (pid, cpu, group_fd): (libc::pid_t, libc::c_int, libc::c_int) = (0, -1, -1);
        // SAFETY: `attr` is a live, initialized `perf_event_attr` prefix
        // whose `size` field bounds the kernel's read of it; the remaining
        // arguments are plain integers.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &raw const attr,
                pid,
                cpu,
                group_fd,
                PERF_FLAG_FD_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(Error::last_os_error());
        }
        // SAFETY: the syscall returned a new descriptor that nothing else
        // owns.
        Ok(unsafe { File::from_raw_fd(fd as RawFd) })
    }
}

#[cfg(not(all(target_os = "linux", target_endian = "little")))]
mod perf {
    use std::fs::File;
    use std::io::{Error, ErrorKind, Result};

    pub(super) fn open_instruction_counter() -> Result<File> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "hardware instruction counters need Linux perf_event_open",
        ))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::hint::black_box;
    use std::io::ErrorKind;

    use libc::EOPNOTSUPP;

    use super::*;

    #[test]
    fn multiplexed_interval_has_no_count() {
        let at = |count, time_enabled_ns, time_running_ns| InstructionReading {
            count,
            time_enabled_ns,
            time_running_ns,
        };
        assert_eq!(
            at(150, 20, 20).instructions_since(at(100, 10, 10)),
            Some(50)
        );
        assert_eq!(at(150, 20, 15).instructions_since(at(100, 10, 10)), None);
    }

    /// Runs wherever the host grants PMU access; returns early on the
    /// errors that mean it does not (no PMU, `perf_event_paranoid` above 2,
    /// non-Linux), so any other open failure — a malformed attr — fails.
    /// Multiplexed intervals are also unavailable when competing counters
    /// occupy PMU slots.
    #[test]
    fn counts_threads_spawned_after_open() {
        let counter = match InstructionCounter::open() {
            Ok(counter) => counter,
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::NotFound | ErrorKind::PermissionDenied | ErrorKind::Unsupported
                ) || e.raw_os_error() == Some(EOPNOTSUPP) =>
            {
                return;
            }
            Err(e) => panic!("perf_event_open failed: {e}"),
        };
        const ITERATIONS: u64 = 10_000_000;
        let before = counter.read().unwrap();
        std::thread::spawn(|| {
            for i in 0..ITERATIONS {
                let _ = black_box(i);
            }
        })
        .join()
        .unwrap();
        let after = counter.read().unwrap();
        let Some(retired) = after.instructions_since(before) else {
            return;
        };
        assert!(
            retired >= ITERATIONS,
            "{retired} instructions for a {ITERATIONS}-iteration loop on a spawned thread"
        );
    }
}
