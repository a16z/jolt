use super::context::enter_device;

pub(crate) type DeviceTask<'a, T, E> = Box<dyn FnOnce() -> Result<T, E> + Send + 'a>;

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
