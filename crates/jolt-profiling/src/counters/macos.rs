use std::ffi::CStr;
use std::io::{Error, ErrorKind, Result};
use std::mem::offset_of;

use libc::{c_int, rusage_info_v4, EINVAL, ENOENT, RUSAGE_INFO_V4};

use super::{CoreCounters, CounterSource, CounterValues};

// libc currently stops at V4. This extends its prefix with Apple's V5/V6 ABI:
// https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/resource.h
#[repr(C)]
struct RusageInfoV6 {
    v4: rusage_info_v4,
    ri_flags: u64,
    ri_user_ptime: u64,
    ri_system_ptime: u64,
    ri_pinstructions: u64,
    ri_pcycles: u64,
    ri_energy_nj: u64,
    ri_penergy_nj: u64,
    ri_secure_time_in_system: u64,
    ri_secure_ptime_in_system: u64,
    ri_neural_footprint: u64,
    ri_lifetime_max_neural_footprint: u64,
    ri_interval_max_neural_footprint: u64,
    ri_reserved: [u64; 9],
}

const _: () = {
    assert!(size_of::<rusage_info_v4>() == 296);
    assert!(offset_of!(rusage_info_v4, ri_instructions) == 248);
    assert!(offset_of!(rusage_info_v4, ri_cycles) == 256);
    assert!(size_of::<RusageInfoV6>() == 464);
    assert!(offset_of!(RusageInfoV6, ri_pinstructions) == 320);
    assert!(offset_of!(RusageInfoV6, ri_pcycles) == 328);
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
enum Flavor {
    V4 = RUSAGE_INFO_V4,
    V6 = 6,
}

impl Flavor {
    /// V6 is worth requesting only with at least two performance levels:
    /// XNU fills its subset from `CLUSTER_TYPE_P` CPUs, which is
    /// `hw.perflevel0` there and absent on single-level hosts such as Intel
    /// Macs, where the subset would read as a constant zero.
    fn select(perf_levels: c_int, mut probe: impl FnMut(Self) -> Result<()>) -> Result<Self> {
        if perf_levels >= 2 {
            match probe(Self::V6) {
                Ok(()) => return Ok(Self::V6),
                Err(error) if error.raw_os_error() == Some(EINVAL) => {}
                Err(error) => return Err(error),
            }
        }
        probe(Self::V4)?;
        Ok(Self::V4)
    }
}

pub(super) struct Counter {
    flavor: Flavor,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Reading(CounterValues);

impl Reading {
    pub(crate) fn since(self, earlier: Self) -> Option<CounterValues> {
        self.0.checked_sub(earlier.0)
    }
}

impl Counter {
    pub(super) fn open() -> Result<Self> {
        if sysctl_int(c"kern.monotonic.supported")? != Some(1) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "macOS hardware counters unavailable",
            ));
        }
        // hw.nperflevels (macOS 12) predates V6 (macOS 13), so a kernel
        // without it has no V6 subset to report.
        let perf_levels = sysctl_int(c"hw.nperflevels")?.unwrap_or(1);
        let flavor = Flavor::select(perf_levels, |flavor| Self { flavor }.read().map(|_| ()))?;
        Ok(Self { flavor })
    }

    pub(super) fn source(&self) -> CounterSource {
        match self.flavor {
            Flavor::V4 => CounterSource::MacosRusageV4,
            Flavor::V6 => CounterSource::MacosRusageV6,
        }
    }

    pub(super) fn read(&self) -> Result<Reading> {
        // SAFETY: this repr(C) struct consists entirely of integer fields.
        // The full V6-sized buffer also accommodates V4; its unused tail stays
        // initialized and is never interpreted as a counter on the V4 path.
        let mut usage: RusageInfoV6 = unsafe { std::mem::zeroed() };
        // SAFETY: the layout assertions above pin the buffer's ABI and size.
        // proc_pid_rusage writes the selected flavor into this live buffer.
        let status = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                self.flavor as c_int,
                (&raw mut usage).cast(),
            )
        };
        if status != 0 {
            return Err(Error::last_os_error());
        }
        let perflevel0 = match self.flavor {
            Flavor::V4 => None,
            Flavor::V6 => Some(CoreCounters {
                instructions: usage.ri_pinstructions,
                cycles: usage.ri_pcycles,
            }),
        };
        Ok(Reading(CounterValues {
            instructions: usage.v4.ri_instructions,
            cycles: Some(usage.v4.ri_cycles),
            perflevel0,
        }))
    }
}

/// Reads an `int` sysctl; `None` when this kernel lacks `name`.
fn sysctl_int(name: &CStr) -> Result<Option<c_int>> {
    let mut value: c_int = 0;
    let mut len = size_of::<c_int>();
    // SAFETY: the NUL-terminated name and writable int/length are live;
    // a null newp makes this a read-only sysctl query.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        let error = Error::last_os_error();
        return match error.raw_os_error() {
            Some(ENOENT) => Ok(None),
            _ => Err(error),
        };
    }
    if len != size_of::<c_int>() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("unexpected {name:?} size {len}"),
        ));
    }
    Ok(Some(value))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::hint::black_box;
    use std::io::Error;
    use std::sync::mpsc;

    use libc::{EACCES, EINVAL, EPERM};

    use super::{Counter, Flavor};

    #[test]
    fn v6_requires_multiple_perf_levels_and_falls_back_only_on_einval() {
        for perf_levels in [0, 1] {
            assert_eq!(
                Flavor::select(perf_levels, |flavor| match flavor {
                    Flavor::V6 => panic!("V6 probed with {perf_levels} perf levels"),
                    Flavor::V4 => Ok(()),
                })
                .unwrap(),
                Flavor::V4,
            );
        }
        assert_eq!(Flavor::select(2, |_| Ok(())).unwrap(), Flavor::V6);
        assert_eq!(
            Flavor::select(2, |flavor| match flavor {
                Flavor::V6 => Err(Error::from_raw_os_error(EINVAL)),
                Flavor::V4 => Ok(()),
            })
            .unwrap(),
            Flavor::V4,
        );
        assert_eq!(
            Flavor::select(2, |flavor| match flavor {
                Flavor::V6 => Err(Error::from_raw_os_error(EPERM)),
                Flavor::V4 => Ok(()),
            })
            .unwrap_err()
            .raw_os_error(),
            Some(EPERM),
        );
        assert_eq!(
            Flavor::select(2, |flavor| match flavor {
                Flavor::V6 => Err(Error::from_raw_os_error(EINVAL)),
                Flavor::V4 => Err(Error::from_raw_os_error(EACCES)),
            })
            .unwrap_err()
            .raw_os_error(),
            Some(EACCES),
        );
    }

    /// Run with `cargo nextest run -p jolt-profiling --run-ignored only
    /// -E 'test(counts_preexisting_workers)'` on physical Apple Silicon.
    #[test]
    #[ignore = "requires physical Apple Silicon with kern.monotonic.supported=1"]
    fn counts_preexisting_workers() {
        const ITERATIONS: u64 = 10_000_000;
        let (start_tx, start_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            start_rx.recv().unwrap();
            for i in 0..ITERATIONS {
                let _ = black_box(i);
            }
        });
        let counter = Counter::open().unwrap();
        let before = counter.read().unwrap();
        start_tx.send(()).unwrap();
        worker.join().unwrap();
        let retired = counter.read().unwrap().since(before).unwrap();
        assert!(retired.instructions >= ITERATIONS);
        assert!(retired.cycles.unwrap() > 0);
    }
}
