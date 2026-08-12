use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{CudaContext, PinnedHostSlice};

use super::error::CudaError;

pub struct StagingBuffer {
    pinned: Option<PinnedHostSlice<u64>>,
}

impl StagingBuffer {
    const fn new() -> Self {
        Self { pinned: None }
    }

    pub fn ensure(
        &mut self,
        context: &Arc<CudaContext>,
        len: usize,
    ) -> Result<&mut PinnedHostSlice<u64>, CudaError> {
        let grow = self.pinned.as_ref().is_none_or(|pinned| pinned.len() < len);
        if grow {
            // SAFETY: `alloc_pinned` leaves the allocation uninitialized. Every
            // read of this buffer goes through `as_slice(..len)` after a fill
            // of the same prefix — `fill_from`/`memcpy_dtoh` below write the
            // prefix before any caller reads it — so no uninitialized element
            // is ever observed.
            let pinned = unsafe { context.alloc_pinned::<u64>(len.next_power_of_two()) }?;
            self.pinned = Some(pinned);
        }
        self.pinned.as_mut().ok_or_else(|| CudaError::NoDevice {
            reason: "pinned staging buffer vanished after allocation".to_owned(),
        })
    }
}

#[derive(Clone)]
pub struct StagingPool {
    buffer: Arc<Mutex<StagingBuffer>>,
}

impl StagingPool {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(StagingBuffer::new())),
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, StagingBuffer> {
        self.buffer.lock().unwrap_or_else(|poisoned| {
            self.buffer.clear_poison();
            poisoned.into_inner()
        })
    }
}

impl Default for StagingPool {
    fn default() -> Self {
        Self::new()
    }
}
