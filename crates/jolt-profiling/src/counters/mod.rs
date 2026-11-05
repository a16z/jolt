//! Hardware counters sampled on the orchestrating thread. Linux counts
//! user-space instructions on that thread and its future descendants; macOS
//! counts instructions and cycles across all process threads in user and kernel
//! mode. Concurrent worker spans must not claim the same process-wide delta.

use std::cell::Cell;
use std::io::Result as IoResult;
use std::sync::OnceLock;

#[cfg(feature = "summary")]
use schemars::JsonSchema;
#[cfg(feature = "summary")]
use serde::{Deserialize, Serialize};

#[cfg(all(target_os = "linux", target_endian = "little"))]
mod linux;
#[cfg(all(target_os = "linux", target_endian = "little"))]
use linux as platform;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(not(any(
    target_os = "macos",
    all(target_os = "linux", target_endian = "little")
)))]
mod unsupported;
#[cfg(not(any(
    target_os = "macos",
    all(target_os = "linux", target_endian = "little")
)))]
use unsupported as platform;

use platform::Counter;
pub(crate) use platform::Reading;

/// Instructions and cycles on the cores of `hw.perflevel0`, the highest
/// performance level, in hardware counter units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "summary",
    derive(Serialize, Deserialize, JsonSchema),
    serde(deny_unknown_fields)
)]
pub struct CoreCounters {
    pub instructions: u64,
    pub cycles: u64,
}

/// Counter values for one sample or interval. Cycles are available on macOS;
/// the `hw.perflevel0` subset additionally requires the V6 resource-usage
/// flavor, which the backend selects only on hosts with several performance
/// levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterValues {
    pub instructions: u64,
    pub cycles: Option<u64>,
    pub perflevel0: Option<CoreCounters>,
}

impl CounterValues {
    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            instructions: self.instructions.checked_add(other.instructions)?,
            cycles: combine_optional(self.cycles, other.cycles, u64::checked_add).ok()?,
            perflevel0: combine_optional(self.perflevel0, other.perflevel0, |a, b| {
                Some(CoreCounters {
                    instructions: a.instructions.checked_add(b.instructions)?,
                    cycles: a.cycles.checked_add(b.cycles)?,
                })
            })
            .ok()?,
        })
    }

    #[cfg(any(target_os = "macos", test))]
    fn checked_sub(self, earlier: Self) -> Option<Self> {
        Some(Self {
            instructions: self.instructions.checked_sub(earlier.instructions)?,
            cycles: combine_optional(self.cycles, earlier.cycles, u64::checked_sub).ok()?,
            perflevel0: combine_optional(self.perflevel0, earlier.perflevel0, |a, b| {
                Some(CoreCounters {
                    instructions: a.instructions.checked_sub(b.instructions)?,
                    cycles: a.cycles.checked_sub(b.cycles)?,
                })
            })
            .ok()?,
        })
    }
}

fn combine_optional<T>(
    a: Option<T>,
    b: Option<T>,
    op: impl FnOnce(T, T) -> Option<T>,
) -> Result<Option<T>, ()> {
    match (a, b) {
        (Some(a), Some(b)) => op(a, b).map(Some).ok_or(()),
        (None, None) => Ok(None),
        _ => Err(()),
    }
}

/// Hardware API used for the run; unsupported counters have no source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "summary",
    derive(Serialize, Deserialize, JsonSchema),
    serde(rename_all = "snake_case")
)]
pub enum CounterSource {
    LinuxPerfEvent,
    MacosRusageV4,
    MacosRusageV6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "summary",
    derive(Serialize, Deserialize, JsonSchema),
    serde(rename_all = "snake_case")
)]
pub enum PrivilegeScope {
    User,
    UserAndKernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "summary",
    derive(Serialize, Deserialize, JsonSchema),
    serde(rename_all = "snake_case")
)]
pub enum ThreadScope {
    /// Opening thread and subsequently spawned descendant threads/processes.
    OpeningThreadAndDescendants,
    /// All threads of this process, including pre-existing workers and profiler
    /// threads; child processes are excluded.
    AllProcessThreads,
}

/// Scope of the hardware counts in a summary. Counts from different scopes
/// are not directly comparable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "summary",
    derive(Serialize, Deserialize, JsonSchema),
    serde(deny_unknown_fields)
)]
pub struct HardwareCounterMetadata {
    pub source: CounterSource,
    pub privilege_scope: PrivilegeScope,
    pub thread_scope: ThreadScope,
}

impl CounterSource {
    pub fn metadata(self) -> HardwareCounterMetadata {
        let (privilege_scope, thread_scope) = match self {
            Self::LinuxPerfEvent => (
                PrivilegeScope::User,
                ThreadScope::OpeningThreadAndDescendants,
            ),
            Self::MacosRusageV4 | Self::MacosRusageV6 => (
                PrivilegeScope::UserAndKernel,
                ThreadScope::AllProcessThreads,
            ),
        };
        HardwareCounterMetadata {
            source: self,
            privilege_scope,
            thread_scope,
        }
    }
}

static PROCESS_COUNTER: OnceLock<Counter> = OnceLock::new();

thread_local! {
    static ON_OPENING_THREAD: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn open_process_counter() -> IoResult<()> {
    let counter = Counter::open()?;
    let _ = PROCESS_COUNTER.set(counter);
    ON_OPENING_THREAD.set(true);
    Ok(())
}

pub(crate) fn counter_source() -> Option<CounterSource> {
    PROCESS_COUNTER.get().map(Counter::source)
}

pub(crate) fn read_process_counter() -> Option<Reading> {
    // Spans can close during thread-local teardown.
    if !ON_OPENING_THREAD.try_with(Cell::get).unwrap_or(false) {
        return None;
    }
    PROCESS_COUNTER.get()?.read().ok()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::hint::black_box;
    use std::io::ErrorKind;

    use libc::EOPNOTSUPP;

    use super::*;

    #[test]
    fn counter_arithmetic_preserves_absence_and_rejects_invalid_intervals() {
        let before = CounterValues {
            instructions: 100,
            cycles: Some(200),
            perflevel0: Some(CoreCounters {
                instructions: 40,
                cycles: 80,
            }),
        };
        let after = CounterValues {
            instructions: 150,
            cycles: Some(280),
            perflevel0: before.perflevel0,
        };
        let delta = after.checked_sub(before).unwrap();
        assert_eq!(
            delta,
            CounterValues {
                instructions: 50,
                cycles: Some(80),
                perflevel0: Some(CoreCounters {
                    instructions: 0,
                    cycles: 0
                }),
            }
        );
        assert_eq!(before.checked_add(delta), Some(after));
        assert_eq!(before.checked_sub(after), None);
        assert_eq!(
            after.checked_sub(CounterValues {
                cycles: Some(281),
                ..before
            }),
            None
        );
        assert_eq!(
            after.checked_sub(CounterValues {
                perflevel0: None,
                ..before
            }),
            None
        );
        assert_eq!(
            after.checked_add(CounterValues {
                instructions: u64::MAX,
                ..before
            }),
            None
        );
        let v4 = CounterValues {
            perflevel0: None,
            ..before
        };
        assert_eq!(v4.checked_sub(v4).unwrap().perflevel0, None);
        let linux = CounterValues { cycles: None, ..v4 };
        assert_eq!(linux.checked_sub(linux).unwrap().cycles, None);
    }

    /// Runs on hosts granting PMU access; capability and permission failures
    /// skip, while malformed API requests fail. The macOS physical-hardware
    /// test additionally requires availability and checks pre-existing workers.
    #[test]
    fn counts_threads_spawned_after_open() {
        let counter = match Counter::open() {
            Ok(counter) => counter,
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::NotFound | ErrorKind::PermissionDenied | ErrorKind::Unsupported
                ) || e.raw_os_error() == Some(EOPNOTSUPP) =>
            {
                return
            }
            Err(e) => panic!("opening hardware counter failed: {e}"),
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
        let Some(retired) = after.since(before) else {
            return;
        };
        assert!(
            retired.instructions >= ITERATIONS,
            "{} instructions for a {ITERATIONS}-iteration loop on a spawned thread",
            retired.instructions
        );
    }
}
