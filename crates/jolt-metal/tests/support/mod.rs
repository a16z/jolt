//! Shared setup for tests that run on the GPU.

use std::fs::File;
use std::process;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread;
use std::time::Duration;

use jolt_metal::runtime::Device;

/// No single test may hold the GPU longer than this.
const WATCHDOG: Duration = Duration::from_mins(2);

/// Exclusive use of the GPU for one test, plus a hang watchdog.
///
/// nextest runs each test in its own process, so the lock is a file lock:
/// concurrent GPU tests would add timing noise and make a hang impossible to
/// attribute. If the test outlives [`WATCHDOG`], the process aborts with a
/// diagnostic instead of wedging the run (GPU hangs have panicked the macOS
/// kernel during earlier Metal work in this repository).
pub struct GpuGuard {
    _lock: File,
    _disarm: Sender<()>,
}

#[expect(clippy::print_stderr, reason = "the watchdog reports why it aborted")]
pub fn gpu(test: &'static str) -> (GpuGuard, Device) {
    let lock = File::create(std::env::temp_dir().join("jolt-metal-gpu.lock")).unwrap();
    lock.lock().unwrap();
    let (disarm, armed) = mpsc::channel::<()>();
    let _watchdog = thread::spawn(move || {
        if armed.recv_timeout(WATCHDOG) == Err(RecvTimeoutError::Timeout) {
            eprintln!("jolt-metal watchdog: `{test}` exceeded {WATCHDOG:?}; aborting");
            process::abort();
        }
    });
    let device = Device::system_default().unwrap();
    (
        GpuGuard {
            _lock: lock,
            _disarm: disarm,
        },
        device,
    )
}

/// SplitMix64: a fixed-seed input generator with no extra dependencies.
pub struct SplitMix64(pub u64);

impl Iterator for SplitMix64 {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Some(z ^ (z >> 31))
    }
}
