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

pub(crate) fn plan_committed_windows(
    cycles: usize,
    row_width: usize,
    devices: usize,
) -> Vec<CycleWindow> {
    let canonical = plan_witness_windows(cycles, devices);
    if row_width > 0
        && canonical
            .iter()
            .all(|window| window.len.is_multiple_of(row_width))
    {
        return canonical;
    }
    whole_domain(cycles)
}

pub(crate) fn committed_windows(cycles: usize, row_width: usize) -> Vec<CycleWindow> {
    plan_committed_windows(cycles, row_width, device_count().max(1))
}

pub(crate) fn plan_device_windows(
    cycles: usize,
    alignment: usize,
    devices: usize,
) -> Vec<CycleWindow> {
    if devices < 2 || alignment == 0 || !cycles.is_multiple_of(alignment) {
        return whole_domain(cycles);
    }
    let blocks = cycles / alignment;
    if blocks < devices {
        return whole_domain(cycles);
    }
    let base = blocks / devices;
    let remainder = blocks % devices;
    let mut windows = Vec::with_capacity(devices);
    let mut start = 0;
    for device in 0..devices {
        let blocks = base + usize::from(device < remainder);
        let len = blocks * alignment;
        windows.push(CycleWindow { start, len });
        start += len;
    }
    windows
}

pub(crate) fn device_windows(cycles: usize, alignment: usize) -> Vec<CycleWindow> {
    plan_device_windows(cycles, alignment, device_count().max(1))
}

pub(crate) fn device_selections(row_counts: &[usize], devices: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..row_counts.len()).collect();
    order.sort_by_key(|&index| {
        (
            core::cmp::Reverse(row_counts.get(index).copied().unwrap_or(0)),
            index,
        )
    });
    let mut load = vec![0usize; devices];
    let mut selections: Vec<Vec<usize>> = (0..devices).map(|_| Vec::new()).collect();
    for index in order {
        let rows = row_counts.get(index).copied().unwrap_or(0);
        let lightest = load
            .iter()
            .enumerate()
            .min_by_key(|&(device, &pending)| (pending, device))
            .map_or(0, |(device, _)| device);
        if let Some(pending) = load.get_mut(lightest) {
            *pending += rows;
        }
        if let Some(selection) = selections.get_mut(lightest) {
            selection.push(index);
        }
    }
    for selection in &mut selections {
        selection.sort_unstable();
    }
    selections
}

pub(crate) fn fan_out<T, E>(tasks: Vec<DeviceTask<'_, T, E>>) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
{
    fan_out_with(tasks, true)
}

pub(crate) fn fan_out_bound<T, E>(tasks: Vec<DeviceTask<'_, T, E>>) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
{
    fan_out_with(tasks, false)
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
    use super::{
        device_selections, plan_committed_windows, plan_device_windows, plan_witness_windows,
        CycleWindow, MIN_WITNESS_WINDOW,
    };

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

    #[test]
    fn a_committed_plan_matches_the_witness_plan_whenever_the_rows_divide_it() {
        for log_t in [13usize, 16, 22] {
            let cycles = 1usize << log_t;
            for devices in COUNTS {
                let witness = plan_witness_windows(cycles, devices);
                for row_width in [1usize, 8, 64, 1 << 12] {
                    let committed = plan_committed_windows(cycles, row_width, devices);
                    covers(&committed, cycles);
                    let aligned = witness
                        .iter()
                        .all(|window| window.len.is_multiple_of(row_width));
                    assert_eq!(
                        committed.len(),
                        if aligned { witness.len() } else { 1 },
                        "a {row_width}-column row over {devices} device(s) at 2^{log_t} neither                          followed the witness split nor fell back to the whole domain; the commit                          and the stage-8 opening must derive the SAME list from (cycles,                          row_width) or the opening misses every parked column",
                    );
                }
            }
        }
    }

    #[test]
    fn device_selections_cover_every_column_once_and_balance_rows() {
        let shape = [1024usize, 1024, 1024, 512, 512, 16];
        for devices in [1usize, 2, 3, 5, 8] {
            let selections = device_selections(&shape, devices);
            let mut seen: Vec<usize> = selections.concat();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..shape.len()).collect::<Vec<_>>(),
                "every column must land on exactly one device",
            );
            let loads: Vec<usize> = selections
                .iter()
                .map(|selection| selection.iter().map(|&index| shape[index]).sum())
                .collect();
            let spread =
                loads.iter().max().copied().unwrap_or(0) - loads.iter().min().copied().unwrap_or(0);
            let widest = shape.iter().max().copied().unwrap_or(0);
            assert!(
                spread <= widest,
                "row load spread {spread} exceeds the widest column {widest} across {devices} \
                 devices: {loads:?}",
            );
        }
    }

    #[test]
    fn a_device_plan_splits_aligned_blocks_evenly_across_every_device_count() {
        for alignment in [1usize, 8, 1 << 10] {
            for blocks in [1usize, 5, 8, 12, 64] {
                let cycles = blocks * alignment;
                for devices in COUNTS {
                    let windows = plan_device_windows(cycles, alignment, devices);
                    covers(&windows, cycles);
                    assert!(windows.len() <= devices.max(1));
                    let mut counts: Vec<usize> = windows
                        .iter()
                        .map(|window| {
                            assert!(
                                window.len.is_multiple_of(alignment),
                                "a window of {} cycles splits an alignment block",
                                window.len,
                            );
                            window.len / alignment
                        })
                        .collect();
                    counts.sort_unstable();
                    let spread = counts.last().copied().unwrap_or_default()
                        - counts.first().copied().unwrap_or_default();
                    assert!(
                        spread <= 1,
                        "{blocks} block(s) over {devices} device(s) landed {spread} blocks apart, \
                         so the resident set is not evenly split",
                    );
                }
            }
        }
    }
}
