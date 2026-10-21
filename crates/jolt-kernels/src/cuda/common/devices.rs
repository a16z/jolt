//! Device partitions and host-side task dispatch.
//!
//! Witness windows are equal, power-of-two cycle ranges so low-to-high binding
//! stays local until only the device-index variables remain. The planner uses
//! one window for small or incompatible domains. `CycleWindow::residency` adds
//! the next row needed by flag extraction without counting it in the work partition.

use super::context::{device_count, enter_device};

pub(crate) type DeviceTask<'a, T, E> = Box<dyn FnOnce() -> Result<T, E> + Send + 'a>;

pub(crate) struct CycleWindow {
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl CycleWindow {
    pub(crate) const fn end(&self) -> usize {
        self.start + self.len
    }

    pub(crate) fn residency(&self, cycles: usize) -> Self {
        Self {
            start: self.start,
            len: self.len + usize::from(self.end() < cycles),
        }
    }
}

const MIN_WITNESS_WINDOW: usize = 1 << 12;

fn whole_domain(cycles: usize) -> Vec<CycleWindow> {
    vec![CycleWindow {
        start: 0,
        len: cycles,
    }]
}

pub(crate) fn plan_witness_windows(cycles: usize, devices: usize) -> Vec<CycleWindow> {
    if devices < 2 || !cycles.is_power_of_two() {
        return whole_domain(cycles);
    }
    let windows = 1usize << devices.ilog2();
    if cycles < windows * MIN_WITNESS_WINDOW {
        return whole_domain(cycles);
    }
    let len = cycles / windows;
    (0..windows)
        .map(|window| CycleWindow {
            start: window * len,
            len,
        })
        .collect()
}

pub(crate) fn witness_windows(cycles: usize) -> Vec<CycleWindow> {
    plan_witness_windows(cycles, device_count().max(1))
}

pub(crate) fn fan_out<T, E>(tasks: Vec<DeviceTask<'_, T, E>>) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
{
    fan_out_with(tasks, true)
}

fn fan_out_with<T, E>(tasks: Vec<DeviceTask<'_, T, E>>, bind: bool) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
{
    let mut tasks = tasks.into_iter();
    let Some(first) = tasks.next() else {
        return Ok(Vec::new());
    };
    let rest: Vec<_> = tasks.collect();
    if rest.is_empty() {
        return Ok(vec![first()?]);
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = rest
            .into_iter()
            .enumerate()
            .map(|(index, task)| {
                scope.spawn(move || {
                    let _device = bind.then(|| enter_device(index + 1));
                    task()
                })
            })
            .collect();
        let mut results = Vec::with_capacity(handles.len() + 1);
        let mut outcome = first().map(|value| results.push(value));
        for handle in handles {
            match handle.join() {
                Ok(Ok(value)) => {
                    if outcome.is_ok() {
                        results.push(value);
                    }
                }
                Ok(Err(error)) => {
                    if outcome.is_ok() {
                        outcome = Err(error);
                    }
                }
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        outcome.map(|()| results)
    })
}

#[cfg(test)]
mod tests {
    use super::{plan_witness_windows, CycleWindow, MIN_WITNESS_WINDOW};

    const COUNTS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 64];

    fn covers(windows: &[CycleWindow], cycles: usize) {
        let mut next = 0;
        for window in windows {
            assert_eq!(
                window.start, next,
                "the windows leave a gap or overlap at cycle {next}",
            );
            next = window.end();
        }
        assert_eq!(next, cycles, "the windows do not cover the cycle domain");
    }

    #[test]
    fn a_witness_plan_splits_every_device_count_into_equal_power_of_two_windows() {
        for log_t in [13usize, 16, 20, 22, 25] {
            let cycles = 1usize << log_t;
            for devices in COUNTS {
                let windows = plan_witness_windows(cycles, devices);
                covers(&windows, cycles);
                assert!(
                    windows.len() <= devices,
                    "the plan for {devices} device(s) at 2^{log_t} asked for {} windows",
                    windows.len(),
                );
                assert!(
                    windows.len().is_power_of_two(),
                    "the eq-table shard split needs a power-of-two window count, got {}",
                    windows.len(),
                );
                for window in &windows {
                    assert_eq!(
                        window.len,
                        cycles / windows.len(),
                        "the windows are not an even split across {devices} device(s)",
                    );
                    assert!(
                        window.len.is_power_of_two(),
                        "a window of {} cycles cannot carry a cycle-variable binding",
                        window.len,
                    );
                }
            }
        }
    }

    #[test]
    fn a_witness_plan_uses_the_largest_power_of_two_device_subset() {
        let cycles = 1usize << 22;
        for devices in COUNTS {
            assert_eq!(
                plan_witness_windows(cycles, devices).len(),
                1usize << devices.ilog2(),
                "{devices} device(s) did not fill the largest power-of-two subset available",
            );
        }
    }

    #[test]
    fn a_witness_plan_declines_a_domain_too_small_or_unaligned_to_split() {
        for devices in COUNTS {
            let windows = 1usize << devices.ilog2();
            let smallest = plan_witness_windows(windows * MIN_WITNESS_WINDOW / 2, devices);
            assert_eq!(
                smallest.len(),
                1,
                "{devices} device(s) split a domain below the {MIN_WITNESS_WINDOW}-cycle floor",
            );
            assert_eq!(
                plan_witness_windows(3 * MIN_WITNESS_WINDOW * windows, devices).len(),
                1,
                "{devices} device(s) split a domain that is not a power of two",
            );
        }
    }

    #[test]
    fn a_witness_window_carries_a_halo_on_every_boundary_but_the_last() {
        let cycles = 1usize << 22;
        for devices in COUNTS {
            let windows = plan_witness_windows(cycles, devices);
            let last = windows.len() - 1;
            for (index, window) in windows.iter().enumerate() {
                let resident = window.residency(cycles);
                assert_eq!(resident.start, window.start);
                assert_eq!(
                    resident.len,
                    window.len + usize::from(index != last),
                    "window {index} of {} on {devices} device(s) has the wrong halo; Spartan's \
                     R1CS columns read cycle t + 1",
                    windows.len(),
                );
            }
        }
    }
}
