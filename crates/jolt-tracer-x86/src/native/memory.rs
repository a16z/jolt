//! Guest RAM plane: one anonymous mmap with logical range `[RAM_START_ADDRESS,
//! RAM_START_ADDRESS + size)` and backing rounded up to whole doublewords.
//! Generated code admits aligned starts within the logical range and addresses
//! them as `mem_base + (guest_addr - RAM_START_ADDRESS)`. Untouched pages cost no
//! physical memory (`MAP_NORESERVE`).
//!
//! Addresses below `RAM_START_ADDRESS` (the `JoltDevice` I/O region) never
//! reach this plane; generated code routes them to `extern "C"` helpers.

use common::constants::RAM_START_ADDRESS;
use jolt_program::execution::TraceError;

pub struct MemoryPlane {
    base: *mut u8,
    size: usize,
    backing_size: usize,
}

// SAFETY: MemoryPlane exclusively owns its mapping; sending it between
// threads transfers that ownership.
unsafe impl Send for MemoryPlane {}

impl MemoryPlane {
    /// Map a zeroed plane with logical size `size` and whole-doubleword backing,
    /// matching `tracer::emulator::memory::MemoryData`.
    pub fn new(size: usize) -> Result<Self, TraceError> {
        let backing_size = size.checked_next_multiple_of(8).ok_or(TraceError::Backend(
            "guest memory plane size overflows doubleword backing",
        ))?;
        // SAFETY: anonymous private mapping with no file backing; length is
        // nonzero and page-rounded by the kernel. The pointer is checked
        // before use.
        let base = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                backing_size,
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
            backing_size,
        })
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
        }
        Ok(())
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy the whole plane out, for a chunk checkpoint.
    pub fn to_vec(&self) -> Vec<u8> {
        // SAFETY: new maps backing_size bytes, including the final partial word.
        unsafe { core::slice::from_raw_parts(self.base, self.backing_size) }.to_vec()
    }

    pub fn restore(&mut self, image: &[u8]) {
        let len = image.len().min(self.backing_size);
        // SAFETY: len <= self.backing_size, and image is a distinct allocation.
        unsafe { self.base.copy_from_nonoverlapping(image.as_ptr(), len) };
    }

    /// Copy out all nonzero bytes as `(address, byte)` pairs, RAM-relative
    /// (address 0 = `RAM_START_ADDRESS`) — the convention the reference
    /// backend's `Memory::materialized_nonzero_bytes` uses for
    /// `TraceOutput::final_memory`.
    pub fn materialized_nonzero_bytes(&self) -> Vec<(u64, u8)> {
        let mut bytes = Vec::new();
        // SAFETY: new maps backing_size bytes, including the final partial word.
        let slice = unsafe { core::slice::from_raw_parts(self.base, self.backing_size) };
        for (offset, &byte) in slice.iter().enumerate() {
            if byte != 0 {
                bytes.push((offset as u64, byte));
            }
        }
        bytes
    }
}

impl Drop for MemoryPlane {
    fn drop(&mut self) {
        // SAFETY: base/backing_size describe the mapping created in `new`.
        let _ = unsafe { libc::munmap(self.base.cast(), self.backing_size) };
    }
}
