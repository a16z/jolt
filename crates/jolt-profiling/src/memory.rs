//! Peak RSS and physical-footprint queries.

#[cfg(target_os = "macos")]
use libc::{rusage_info_v4, RUSAGE_INFO_V4};

/// Process-lifetime peak RSS in bytes, from `getrusage(RUSAGE_SELF)`.
///
/// The kernel-maintained high-water mark — unlike a sampling monitor it
/// cannot miss short allocation spikes, so it is the headline memory number
/// for benchmark runs. `ru_maxrss` is reported in bytes on macOS and
/// kibibytes on Linux; normalized to bytes here. Returns `None` on
/// non-unix targets or if the syscall fails.
#[cfg(unix)]
pub fn peak_rss_bytes() -> Option<u64> {
    // SAFETY: getrusage writes a complete rusage struct into the provided
    // storage on success (return value 0).
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) != 0 {
            return None;
        }
        usage
    };
    let raw = usage.ru_maxrss as u64;
    Some(if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    })
}

/// Process-lifetime peak RSS in bytes. Always `None` on non-unix targets.
#[cfg(not(unix))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "macos")]
fn rusage_info() -> Option<rusage_info_v4> {
    // SAFETY: on success (return value 0) proc_pid_rusage writes a complete
    // `rusage_info_v4` for the requested flavor into the provided storage.
    unsafe {
        let mut info: rusage_info_v4 = std::mem::zeroed();
        let status = libc::proc_pid_rusage(libc::getpid(), RUSAGE_INFO_V4, (&raw mut info).cast());
        (status == 0).then_some(info)
    }
}

/// Current physical footprint in bytes, sampled from the kernel's
/// `ri_phys_footprint` counter. Unlike RSS it includes pages macOS has
/// compressed under memory pressure. `None` off macOS; no separate footprint counter is provided.
pub fn current_footprint_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        rusage_info().map(|info| info.ri_phys_footprint)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Process-lifetime peak physical footprint in bytes.
///
/// On macOS, memory pressure makes the kernel compress cold pages, which
/// drop out of RSS: [`peak_rss_bytes`] can under-report a compressed run by
/// half. `ri_lifetime_max_phys_footprint` counts them, and is the figure
/// `/usr/bin/time -l` prints as "peak memory footprint". Elsewhere this falls back to [`peak_rss_bytes`]; it does not account
/// for swapped or compressed pages.
pub fn peak_footprint_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        rusage_info().map(|info| info.ri_lifetime_max_phys_footprint)
    }
    #[cfg(not(target_os = "macos"))]
    {
        peak_rss_bytes()
    }
}

/// The process-lifetime memory high-water marks a profiled run reports.
/// Sample once, right after the workload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeakMemory {
    /// [`peak_rss_bytes`].
    pub rss_bytes: Option<u64>,
    /// [`peak_footprint_bytes`].
    pub footprint_bytes: Option<u64>,
}

impl PeakMemory {
    pub fn sample() -> Self {
        Self {
            rss_bytes: peak_rss_bytes(),
            footprint_bytes: peak_footprint_bytes(),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn footprint_covers_a_live_allocation() {
        const BYTES: usize = 64 << 20;
        let block = std::hint::black_box(vec![1u8; BYTES]);
        assert!(peak_footprint_bytes().unwrap() >= BYTES as u64);
        #[cfg(target_os = "macos")]
        assert!(current_footprint_bytes().unwrap() >= BYTES as u64);
        drop(block);
    }
}
