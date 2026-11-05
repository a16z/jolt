use std::fs::File;
use std::io::{Error, Read, Result};
use std::os::fd::{FromRawFd, RawFd};

use libc::{c_int, c_ulong, pid_t, SYS_perf_event_open};

use super::{CounterSource, CounterValues};

/// One counter sample. The kernel's enabled and running times diverge only
/// while it multiplexes the counter off the PMU, during which the count
/// misses instructions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reading {
    count: u64,
    time_enabled_ns: u64,
    time_running_ns: u64,
}

impl Reading {
    /// Instructions retired between `earlier` and `self`; `None` when the
    /// counter was multiplexed during the interval, so the difference would
    /// undercount.
    pub(crate) fn since(self, earlier: Self) -> Option<CounterValues> {
        let enabled = self.time_enabled_ns.checked_sub(earlier.time_enabled_ns)?;
        let running = self.time_running_ns.checked_sub(earlier.time_running_ns)?;
        if enabled != running {
            return None;
        }
        Some(CounterValues {
            instructions: self.count.checked_sub(earlier.count)?,
            cycles: None,
            perflevel0: None,
        })
    }
}

pub(super) struct Counter(File);

impl Counter {
    #[expect(
        clippy::unused_self,
        reason = "shared platform interface; macOS selects the source at runtime"
    )]
    pub(super) fn source(&self) -> CounterSource {
        CounterSource::LinuxPerfEvent
    }

    pub(super) fn open() -> Result<Self> {
        let attr = PerfEventAttr {
            kind: PERF_TYPE_HARDWARE,
            size: size_of::<PerfEventAttr>() as u32,
            config: PERF_COUNT_HW_INSTRUCTIONS,
            read_format: PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
            flags: ATTR_FLAG_INHERIT | ATTR_FLAG_EXCLUDE_KERNEL | ATTR_FLAG_EXCLUDE_HV,
            ..PerfEventAttr::default()
        };
        // Calling thread, any CPU, no group leader.
        let (pid, cpu, group_fd): (pid_t, c_int, c_int) = (0, -1, -1);
        // SAFETY: `attr` is a live, initialized `perf_event_attr` prefix
        // whose `size` field bounds the kernel's read of it; the remaining
        // arguments are plain integers.
        let fd = unsafe {
            libc::syscall(
                SYS_perf_event_open,
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
        Ok(Self(unsafe { File::from_raw_fd(fd as RawFd) }))
    }

    /// Decodes the `read_format` layout requested at open: value, then
    /// enabled and running time, as native-endian `u64`s.
    pub(super) fn read(&self) -> Result<Reading> {
        let mut buf = [0u8; 24];
        (&self.0).read_exact(&mut buf)?;
        let word = |i: usize| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&buf[8 * i..8 * (i + 1)]);
            u64::from_ne_bytes(bytes)
        };
        Ok(Reading {
            count: word(0),
            time_enabled_ns: word(1),
            time_running_ns: word(2),
        })
    }
}

/// `struct perf_event_attr` truncated to `PERF_ATTR_SIZE_VER0` (64
/// bytes); the kernel zero-extends the newer fields. The flags encode the
/// little-endian bitfield order, matching this module's target gate.
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
const PERF_FLAG_FD_CLOEXEC: c_ulong = 1 << 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplexed_interval_has_no_count() {
        let at = |count, time_enabled_ns, time_running_ns| Reading {
            count,
            time_enabled_ns,
            time_running_ns,
        };
        assert_eq!(
            at(150, 20, 20)
                .since(at(100, 10, 10))
                .map(|values| values.instructions),
            Some(50)
        );
        assert_eq!(at(150, 20, 15).since(at(100, 10, 10)), None);
    }
}
