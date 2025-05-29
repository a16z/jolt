//! SHA-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha2` crate.

/// SHA-256 hasher state.
pub struct Sha256 {
    /// Current hash state (8 x 32-bit words)
    state: [u32; 8],
    /// Buffer for incomplete blocks
    buffer: [u8; 64],
    /// Number of bytes in the buffer
    buffer_len: usize,
    /// Total number of bytes processed
    total_len: u64,
    /// Whether this is the initial block
    initial: bool,
}

impl Sha256 {
    /// Creates a new SHA-256 hasher.
    #[inline]
    #[allow(clippy::uninit_assumed_init, invalid_value)]
    pub fn new() -> Self {
        Self {
            state: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            buffer: [0u8; 64],
            buffer_len: 0,
            total_len: 0,
            initial: true,
        }
    }

    /// Writes data to the hasher.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        if input.is_empty() {
            return;
        }

        self.total_len += input.len() as u64;
        let mut offset = 0;

        // If we have buffered data, try to fill the buffer and process it
        if self.buffer_len > 0 {
            let needed = 64 - self.buffer_len;
            let to_copy = needed.min(input.len());
            // Safety: buffer_len < 64, so destination is within bounds
            // input slice is valid for at least to_copy bytes
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    to_copy,
                );
            }
            self.buffer_len += to_copy;
            offset += to_copy;

            if self.buffer_len == 64 {
                self.process_block_from_buffer();
                self.buffer_len = 0;
            }
        }

        // Process complete 64-byte blocks directly from input
        while offset + 64 <= input.len() {
            self.process_block_from_slice(&input[offset..offset + 64]);
            offset += 64;
        }

        // Buffer any remaining bytes
        let remaining = input.len() - offset;
        if remaining > 0 {
            // Safety: remaining < 64, input has at least remaining bytes from offset
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    self.buffer.as_mut_ptr(),
                    remaining,
                );
            }
            self.buffer_len = remaining;
        }
    }

    /// Processes a block from the internal buffer.
    #[inline(always)]
    fn process_block_from_buffer(&mut self) {
        // Safety: buffer is properly aligned and sized for [u32; 16] access
        let mut w: [u32; 16] = unsafe { *(self.buffer.as_ptr() as *const [u32; 16]) };
        #[cfg(target_endian = "little")]
        for w in &mut w {
            *w = w.swap_bytes();
        }

        // Safety: w and state are properly aligned arrays with no overlap
        unsafe {
            if self.initial {
                sha256_compression_initial(w.as_ptr(), self.state.as_mut_ptr());
                self.initial = false;
            } else {
                sha256_compression(w.as_ptr(), self.state.as_mut_ptr());
            };
        }
    }

    /// Processes a block directly from a slice.
    #[inline(always)]
    fn process_block_from_slice(&mut self, block: &[u8]) {
        // Safety: block is guaranteed to be exactly 64 bytes by the caller
        let mut w: [u32; 16] = unsafe { *(block.as_ptr() as *const [u32; 16]) };
        #[cfg(target_endian = "little")]
        w.iter_mut().for_each(|w| *w = w.swap_bytes());

        // Safety: w and state are properly aligned arrays with no overlap
        unsafe {
            if self.initial {
                sha256_compression_initial(w.as_ptr(), self.state.as_mut_ptr());
                self.initial = false;
            } else {
                sha256_compression(w.as_ptr(), self.state.as_mut_ptr());
            };
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; 32] {
        // Pad the message according to SHA-256 specification
        let bit_len = self.total_len * 8;

        // Append padding bit
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        // If there's not enough room for the length, process this block
        if self.buffer_len > 56 {
            // Zero the rest of the buffer and process
            // Safety: buffer_len <= 64, so we're writing within bounds
            unsafe {
                core::ptr::write_bytes(
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    0,
                    64 - self.buffer_len,
                );
            }
            self.process_block_from_buffer();
            self.buffer_len = 0;
        }

        // Zero from current position to 56
        // Safety: buffer_len <= 56 here, so we're writing within bounds
        unsafe {
            core::ptr::write_bytes(
                self.buffer.as_mut_ptr().add(self.buffer_len),
                0,
                56 - self.buffer_len,
            );

            // Write length as 64-bit big-endian directly
            // Safety: buffer has 64 bytes, writing u64 at offset 56 is within bounds
            *(self.buffer.as_mut_ptr().add(56) as *mut u64) = bit_len.to_be();
        }

        self.process_block_from_buffer();

        #[cfg(target_endian = "little")]
        for w in &mut self.state {
            *w = w.swap_bytes();
        }
        unsafe { core::mem::transmute(self.state) }
    }

    /// Computes SHA-256 hash of the input data in one call.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; 32] {
        let mut hasher = Self::new();
        hasher.update(input);
        hasher.finalize()
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Calls the SHA256 compression custom instruction
///
/// # Arguments
/// * `input` - Pointer to 16 u32 words (64 bytes) of input data
/// * `state` - Pointer to 8 u32 words (32 bytes) of initial state - will be overwritten by result
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of readable and writable memory
/// - Both pointers must be properly aligned for u32 access (4-byte alignment)
/// - The memory regions must not overlap
#[cfg(not(feature = "host"))]
pub unsafe fn sha256_compression(input: *const u32, state: *mut u32) {
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x00, x0, {}, {}",
        in(reg) input,
        in(reg) state,
        options(nostack)
    );
}
/// Calls the SHA256 compression custom instruction
///
/// # Arguments
/// * `input` - Pointer to 16 u32 words (64 bytes) of input data
/// * `state` - Pointer to 8 u32 words (32 bytes) of initial state - will be overwritten by result
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of readable and writable memory
/// - Both pointers must be properly aligned for u32 access (4-byte alignment)
/// - The memory regions must not overlap
#[cfg(feature = "host")]
pub unsafe fn sha256_compression(input: *const u32, state: *mut u32) {
    let input_array = *(input as *const [u32; 16]);
    let state_array = *(state as *const [u32; 8]);
    let result = tracer::instruction::precompile_sha256::execute_sha256_compression(
        state_array,
        input_array,
    );
    std::ptr::copy_nonoverlapping(result.as_ptr(), state, 8)
}

/// Calls the SHA256 compression custom instruction with initial block
///
/// # Arguments
/// * `input` - Pointer to 16 u32 words (64 bytes) of input data
/// * `state` - Pointer to 8 u32 words (32 bytes) - result will be written here
///
/// Uses the SHA256 initial state constants internally
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of writable memory
/// - Both pointers must be properly aligned for u32 access (4-byte alignment)
/// - The memory regions must not overlap
#[cfg(not(feature = "host"))]
pub unsafe fn sha256_compression_initial(input: *const u32, state: *mut u32) {
    core::arch::asm!(
        ".insn r 0x0B, 0x1, 0x00, x0, {}, {}",
        in(reg) input,
        in(reg) state,
        options(nostack)
    );
}
/// Calls the SHA256 compression custom instruction with initial block
///
/// # Arguments
/// * `input` - Pointer to 16 u32 words (64 bytes) of input data
/// * `state` - Pointer to 8 u32 words (32 bytes) - result will be written here
///
/// Uses the SHA256 initial state constants internally
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of writable memory
/// - Both pointers must be properly aligned for u32 access (4-byte alignment)
/// - The memory regions must not overlap
#[cfg(feature = "host")]
pub unsafe fn sha256_compression_initial(input: *const u32, state: *mut u32) {
    let input = *(input as *const [u32; 16]);
    let result = tracer::instruction::precompile_sha256::execute_sha256_compression_initial(input);
    std::ptr::copy_nonoverlapping(result.as_ptr(), state, 8)
}
