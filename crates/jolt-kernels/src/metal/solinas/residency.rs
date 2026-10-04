//! Off-submit-path GPU residency for large fresh Shared buffers.
//!
//! The Metal driver wires an allocation into the GPU address space inside the
//! first command buffer that binds it, synchronously, with CPU and GPU idle.
//! Under memory pressure (2^28 proofs run at ~100 GB footprint on 128 GiB) the
//! kernel must reclaim pages first, and one first-use command buffer stalled for
//! up to 2.1 s. Requesting residency at allocation moves that work to a helper
//! thread while the prover keeps running, and the CPU then fills wired pages
//! instead of faulting fresh ones in.
//!
//! Residency outlives the transient residency set: the driver keeps the
//! allocation wired after `removeAllocation` + `commit` until the buffer is
//! freed. Nothing here is required for correctness; if the driver later evicts
//! the allocation, its first command buffer pays the original wiring cost.

use std::{
    ptr,
    sync::{
        mpsc::{self, Sender},
        OnceLock,
    },
    thread::Builder,
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

static HELPER: OnceLock<Option<Sender<Buffer>>> = OnceLock::new();

/// Queues `buffer` for residency on the process-wide helper thread.
///
/// The helper holds a reference until it has run, so a buffer dropped by its
/// owner in the meantime is freed only after its turn.
pub(super) fn prefetch(buffer: &Buffer) {
    let helper = HELPER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Buffer>();
        Builder::new()
            .name("jolt-metal-residency".into())
            .spawn(move || {
                for buffer in receiver {
                    autoreleasepool(|| {
                        if let Err(err) = request_residency(&buffer) {
                            tracing::debug!(target: "jolt::metal", err, "residency request skipped");
                        }
                    });
                }
            })
            .map_err(|err| tracing::warn!(%err, "metal residency helper did not start"))
            .ok()
            .map(|_| sender)
    });
    if let Some(sender) = helper {
        let _ = sender.send(buffer.clone());
    }
}

fn request_residency(buffer: &Buffer) -> Result<(), String> {
    let descriptor_class =
        Class::get("MTLResidencySetDescriptor").ok_or("MTLResidencySet needs macOS 15")?;
    // SAFETY: the selectors are MTLResidencySet(Descriptor)/MTLDevice API with
    // the argument and return types declared here; `new` and
    // `newResidencySetWithDescriptor:error:` return +1 objects released below.
    unsafe {
        let descriptor: *mut Object = descriptor_class
            .send_message(Sel::register("new"), ())
            .map_err(|err| err.to_string())?;
        let mut error: *mut Object = ptr::null_mut();
        let set: Result<*mut Object, _> = buffer.device().send_message(
            Sel::register("newResidencySetWithDescriptor:error:"),
            (descriptor, &raw mut error),
        );
        let _: Result<(), _> = (*descriptor).send_message(Sel::register("release"), ());
        let set = set.map_err(|err| err.to_string())?;
        if set.is_null() {
            return Err("newResidencySetWithDescriptor failed".into());
        }
        let allocation = buffer.as_ptr().cast::<Object>();
        let result = [
            ("addAllocation:", Some(allocation)),
            ("commit", None),
            ("requestResidency", None),
            ("removeAllocation:", Some(allocation)),
            ("commit", None),
        ]
        .into_iter()
        .try_for_each(|(selector, argument)| {
            let selector = Sel::register(selector);
            match argument {
                Some(argument) => (*set).send_message::<_, ()>(selector, (argument,)),
                None => (*set).send_message::<_, ()>(selector, ()),
            }
            .map_err(|err| err.to_string())
        });
        let _: Result<(), _> = (*set).send_message(Sel::register("release"), ());
        result
    }
}
