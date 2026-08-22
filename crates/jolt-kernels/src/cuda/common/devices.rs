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
}

pub(crate) fn device_windows(cycles: usize, alignment: usize) -> Vec<CycleWindow> {
    let devices = device_count().max(1);
    if devices == 1 || alignment == 0 || !cycles.is_multiple_of(alignment) {
        return vec![CycleWindow {
            start: 0,
            len: cycles,
        }];
    }
    let blocks = cycles / alignment;
    if blocks < devices {
        return vec![CycleWindow {
            start: 0,
            len: cycles,
        }];
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

pub(crate) fn fan_out<T, E>(tasks: Vec<DeviceTask<'_, T, E>>) -> Result<Vec<T>, E>
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
                    let _device = enter_device(index + 1);
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
