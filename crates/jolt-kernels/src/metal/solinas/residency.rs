//! GPU residency requested off the submit path.
//!
//! The driver otherwise wires a Shared allocation inside the first command
//! buffer that binds it, with CPU and GPU idle. Requesting residency while the
//! producer fills the rows, or under other work before that command buffer,
//! moves that work onto a helper thread.
//!
//! Residency is requested through a transient set: removing a committed
//! allocation makes it eligible for nonresidency again, so a later command
//! buffer may still pay the wiring. Measured runs kept the warm-up after
//! removal except across Stage 0's trace commit, which evicts the Stage-1
//! rows; no API guarantees it.

use std::{
    ptr,
    thread::{Builder, JoinHandle},
};

use metal::{
    foreign_types::ForeignType,
    objc::{
        rc::autoreleasepool,
        runtime::{Class, Object, Sel},
        Message,
    },
    Buffer, DeviceRef,
};

/// Pending residency requests for one producer's allocations. Dropping it waits
/// for the helper, so the helper's buffer references never outlive the
/// producer scope that owns the guard (including its error exits).
#[must_use = "dropping the guard immediately waits for the residency request"]
pub(crate) struct ResidencyPrefetch(Option<JoinHandle<()>>);

impl Drop for ResidencyPrefetch {
    fn drop(&mut self) {
        if let Some(helper) = self.0.take() {
            let _ = helper.join();
        }
    }
}

/// Starts a best-effort residency warm-up for `buffers` on a helper thread;
/// dropping the guard joins it.
pub(super) fn prefetch(buffers: Vec<Buffer>) -> ResidencyPrefetch {
    prefetch_after(buffers, || {})
}

fn prefetch_after(
    buffers: Vec<Buffer>,
    before: impl FnOnce() + Send + 'static,
) -> ResidencyPrefetch {
    let helper = Builder::new()
        .name("jolt-metal-residency".into())
        .spawn(move || {
            before();
            for buffer in &buffers {
                autoreleasepool(|| request_residency(buffer));
            }
        })
        .map_err(|err| tracing::warn!(%err, "metal residency helper did not start"))
        .ok();
    ResidencyPrefetch(helper)
}

/// A +1 MTLResidencySet of `device`, or `None` before macOS 15.
pub(super) fn new_residency_set(device: &DeviceRef) -> Option<*mut Object> {
    let descriptor_class = Class::get("MTLResidencySetDescriptor")?;
    // SAFETY: MTLResidencySetDescriptor `new` and MTLDevice
    // `newResidencySetWithDescriptor:error:` with their declared argument and
    // return types; the +1 descriptor is released here, the +1 set by the caller.
    unsafe {
        let descriptor = descriptor_class
            .send_message::<_, *mut Object>(Sel::register("new"), ())
            .ok()?;
        let mut error: *mut Object = ptr::null_mut();
        let set = device.send_message::<_, *mut Object>(
            Sel::register("newResidencySetWithDescriptor:error:"),
            (descriptor, &raw mut error),
        );
        let _ = (*descriptor).send_message::<_, ()>(Sel::register("release"), ());
        set.ok().filter(|set| !set.is_null())
    }
}

fn request_residency(buffer: &Buffer) {
    let Some(set) = new_residency_set(buffer.device()) else {
        return;
    };
    // SAFETY: MTLResidencySet selectors with the declared argument and return
    // types on the +1 set from new_residency_set, released below. The set never
    // leaves this thread.
    unsafe {
        let allocation = buffer.as_ptr().cast::<Object>();
        let _ = (*set).send_message::<_, ()>(Sel::register("addAllocation:"), (allocation,));
        let _ = (*set).send_message::<_, ()>(Sel::register("commit"), ());
        let _ = (*set).send_message::<_, ()>(Sel::register("requestResidency"), ());
        let _ = (*set).send_message::<_, ()>(Sel::register("removeAllocation:"), (allocation,));
        let _ = (*set).send_message::<_, ()>(Sel::register("commit"), ());
        let _ = (*set).send_message::<_, ()>(Sel::register("release"), ());
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use metal::{Device, MTLResourceOptions};

    use super::prefetch_after;

    #[test]
    fn dropped_guard_releases_the_retired_rows() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let before = device.current_allocated_size();
        let rows = vec![device.new_buffer(1 << 20, MTLResourceOptions::StorageModeShared)];
        let (open_gate, gate) = mpsc::channel::<()>();
        let guard = prefetch_after(rows.clone(), move || {
            let _ = gate.recv();
        });
        drop(rows);
        // The helper stays pending until the gate opens: after the check
        // below, or after the timeout that keeps a joining drop live.
        let (checked, checked_signal) = mpsc::channel::<()>();
        let releaser = thread::spawn(move || {
            let _ = checked_signal.recv_timeout(Duration::from_millis(500));
            let _ = open_gate.send(());
        });
        drop(guard);
        let released = device.current_allocated_size() == before;
        let _ = checked.send(());
        releaser.join().unwrap();
        assert!(
            released,
            "a dropped guard left the helper holding retired rows"
        );
    }
}
