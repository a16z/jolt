//! Guest RAM plane: one anonymous mmap covering `[RAM_START_ADDRESS,
//! RAM_START_ADDRESS + size)`. `MemoryPlane::new` enforces doubleword-aligned
//! sizes, and `GuestState::high_water` stays doubleword aligned and at most
//! `size`, so generated code can admit an aligned start below the high-water
//! mark and access the whole doubleword at `mem_base + (guest_addr -
//! RAM_START_ADDRESS)`. Untouched pages cost no physical memory
//! (`MAP_NORESERVE`), and copies read only the prefix below the mark.
//!
//! Addresses below `RAM_START_ADDRESS` (the `JoltDevice` I/O segments and the
//! I/O gap) never reach this plane; generated code routes them to
//! `extern "C"` helpers.

use common::constants::RAM_START_ADDRESS;
use common::jolt_device::MemoryLayout;
use jolt_program::execution::TraceError;

pub struct MemoryPlane {
    base: *mut u8,
    size: usize,
    /// One past the highest offset `init_from_image` wrote.
    image_end: u64,
}

// SAFETY: MemoryPlane exclusively owns its mapping; sending it between
// threads transfers that ownership.
unsafe impl Send for MemoryPlane {}

impl MemoryPlane {
    /// Map a zeroed plane of a nonzero, doubleword-aligned size.
    pub fn new(size: usize) -> Result<Self, TraceError> {
        if size == 0 || !size.is_multiple_of(8) {
            return Err(TraceError::Backend(
                "guest memory plane size must be a nonzero multiple of eight",
            ));
        }
        // SAFETY: anonymous private mapping with no file backing; length is
        // nonzero and page-rounded by the kernel. The pointer is checked
        // before use.
        let base = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(TraceError::Backend("failed to mmap guest memory plane"));
        }
        Ok(Self {
            base: base.cast::<u8>(),
            size,
            image_end: 0,
        })
    }

    /// Map the plane for `layout`: its RAM up to the end of the RAM domain.
    pub fn for_layout(layout: &MemoryLayout) -> Result<Self, TraceError> {
        Self::new(layout.ram_size()? as usize)
    }

    /// Write the program image (`(guest_address, byte)` pairs) into the plane.
    pub fn init_from_image(&mut self, memory_init: &[(u64, u8)]) -> Result<(), TraceError> {
        for &(address, byte) in memory_init {
            let offset = address
                .checked_sub(RAM_START_ADDRESS)
                .filter(|&o| o < self.size as u64)
                .ok_or(TraceError::Backend(
                    "program image byte outside the guest memory plane",
                ))?;
            // SAFETY: offset < size, so the write is within the mapping.
            unsafe { self.base.add(offset as usize).write(byte) };
            self.image_end = self.image_end.max(offset + 1);
        }
        Ok(())
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn image_end(&self) -> u64 {
        self.image_end
    }

    /// The plane below `high_water`, which holds every nonzero byte.
    fn touched(&self, high_water: u64) -> &[u8] {
        let len = usize::try_from(high_water).map_or(self.size, |len| len.min(self.size));
        // SAFETY: new maps size bytes, all readable, and len <= size.
        unsafe { core::slice::from_raw_parts(self.base, len) }
    }

    /// The little-endian word at guest `address`, or `None` where the plane
    /// does not hold all four bytes.
    pub fn word_at(&self, address: u64) -> Option<u32> {
        let offset = usize::try_from(address.checked_sub(RAM_START_ADDRESS)?).ok()?;
        // SAFETY: new maps size bytes, all readable.
        let plane = unsafe { core::slice::from_raw_parts(self.base, self.size) };
        let bytes = plane.get(offset..offset.checked_add(4)?)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Copy out the plane below `high_water`, for a chunk checkpoint.
    pub fn snapshot(&self, high_water: u64) -> Vec<u8> {
        self.touched(high_water).to_vec()
    }

    /// Copy a snapshot into a fresh plane, returning the high-water mark that
    /// covers it.
    pub fn restore(&mut self, snapshot: &[u8]) -> u64 {
        let len = snapshot.len().min(self.size);
        // SAFETY: len <= self.size, and snapshot is a distinct allocation.
        unsafe { self.base.copy_from_nonoverlapping(snapshot.as_ptr(), len) };
        // The size is a multiple of eight, so the mark stays within it.
        (len as u64).next_multiple_of(8)
    }

    /// Copy out all nonzero bytes, all below `high_water`, as `(offset,
    /// byte)` pairs in ascending order, keyed by offset from
    /// `RAM_START_ADDRESS`, as `final_memory_image` takes them.
    pub fn materialized_nonzero_bytes(&self, high_water: u64) -> Vec<(u64, u8)> {
        let mut bytes = Vec::new();
        for (offset, &byte) in self.touched(high_water).iter().enumerate() {
            if byte != 0 {
                bytes.push((offset as u64, byte));
            }
        }
        bytes
    }
}

impl Drop for MemoryPlane {
    fn drop(&mut self) {
        // SAFETY: base/size describe the mapping created in `new`.
        let _ = unsafe { libc::munmap(self.base.cast(), self.size) };
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryPlane;

    #[test]
    fn rejects_unaligned_size() {
        assert!(MemoryPlane::new(9).is_err());
    }
}
