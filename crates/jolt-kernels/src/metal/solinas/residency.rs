//! GPU residency requested off the submit path for freshly allocated rows.
//!
//! The driver otherwise wires a Shared allocation inside the first command
//! buffer that binds it, with CPU and GPU idle. Requesting residency while the
//! producer fills the rows moves that work onto a helper thread.
//!
//! Residency is requested through a transient set: removing a committed
//! allocation makes it eligible for nonresidency again, so a later command
//! buffer may still pay the wiring. Measured runs kept the warm-up after
//! removal; no API guarantees it.

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
    Buffer,
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

/// Requests residency for `buffers` on a helper thread until the guard drops.
pub(super) fn prefetch(buffers: Vec<Buffer>) -> ResidencyPrefetch {
    let helper = Builder::new()
        .name("jolt-metal-residency".into())
        .spawn(move || {
            for buffer in &buffers {
                autoreleasepool(|| request_residency(buffer));
            }
        })
        .map_err(|err| tracing::warn!(%err, "metal residency helper did not start"))
        .ok();
    ResidencyPrefetch(helper)
}

fn request_residency(buffer: &Buffer) {
    let Some(descriptor_class) = Class::get("MTLResidencySetDescriptor") else {
        return;
    };
    // SAFETY: MTLResidencySet(Descriptor)/MTLDevice selectors with the declared
    // argument and return types; `new` and `newResidencySetWithDescriptor:error:`
    // return +1 objects released below. The set never leaves this thread.
    unsafe {
        let Ok(descriptor) =
            descriptor_class.send_message::<_, *mut Object>(Sel::register("new"), ())
        else {
            return;
        };
        let mut error: *mut Object = ptr::null_mut();
        let set = buffer.device().send_message::<_, *mut Object>(
            Sel::register("newResidencySetWithDescriptor:error:"),
            (descriptor, &raw mut error),
        );
        let _ = (*descriptor).send_message::<_, ()>(Sel::register("release"), ());
        let Ok(set) = set else { return };
        if set.is_null() {
            return;
        }
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
mod tests {
    use metal::{Device, MTLResourceOptions};

    use super::prefetch;

    #[test]
    fn dropped_guard_releases_the_retired_rows() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let before = device.current_allocated_size();
        let rows: Vec<_> = (0..8)
            .map(|_| device.new_buffer(256 << 20, MTLResourceOptions::StorageModeShared))
            .collect();
        let guard = prefetch(rows.clone());
        drop(rows);
        drop(guard);
        assert_eq!(device.current_allocated_size(), before);
    }
}
