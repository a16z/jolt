//! SHA-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha2` crate.

/// SHA-256 hasher state.
pub struct Sha256 {
    /// Current hash state (8 x 32-bit words)
    state: [u32; 8],
    /// Buffer for incomplete blocks - aligned for u32 access
    buffer: [u32; 16],
    /// Number of bytes in the buffer
    buffer_len: usize,
    /// Total number of bytes processed
    total_len: u64,
    /// Whether this is the initial block
    initial: bool,
}

impl Sha256 {
    /// Creates a new SHA-256 hasher.
    #[allow(clippy::uninit_assumed_init, invalid_value)]
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            state: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            buffer: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            buffer_len: 0,
            total_len: 0,
            initial: true,
        }
    }

    /// Writes data to the hasher.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        let input_len = input.len();
        if input_len == 0 {
            return;
        }

        self.total_len += input_len as u64;
        let mut offset = 0;

        // Cast buffer to u8 for byte-level operations
        let buffer_u8 =
            unsafe { core::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u8, 64) };

        // Handle partial buffer
        if self.buffer_len != 0 {
            let needed = 64 - self.buffer_len;
            let to_copy = needed.min(input_len);

            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    buffer_u8.as_mut_ptr().add(self.buffer_len),
                    to_copy,
                );
            }

            self.buffer_len += to_copy;
            offset = to_copy;

            if self.buffer_len == 64 {
                #[cfg(target_endian = "little")]
                {
                    // Swap bytes in-place - unrolled
                    self.buffer[0] = self.buffer[0].swap_bytes();
                    self.buffer[1] = self.buffer[1].swap_bytes();
                    self.buffer[2] = self.buffer[2].swap_bytes();
                    self.buffer[3] = self.buffer[3].swap_bytes();
                    self.buffer[4] = self.buffer[4].swap_bytes();
                    self.buffer[5] = self.buffer[5].swap_bytes();
                    self.buffer[6] = self.buffer[6].swap_bytes();
                    self.buffer[7] = self.buffer[7].swap_bytes();
                    self.buffer[8] = self.buffer[8].swap_bytes();
                    self.buffer[9] = self.buffer[9].swap_bytes();
                    self.buffer[10] = self.buffer[10].swap_bytes();
                    self.buffer[11] = self.buffer[11].swap_bytes();
                    self.buffer[12] = self.buffer[12].swap_bytes();
                    self.buffer[13] = self.buffer[13].swap_bytes();
                    self.buffer[14] = self.buffer[14].swap_bytes();
                    self.buffer[15] = self.buffer[15].swap_bytes();
                }

                unsafe {
                    if self.initial {
                        sha256_compression_initial(self.buffer.as_ptr(), self.state.as_mut_ptr());
                        self.initial = false;
                    } else {
                        sha256_compression(self.buffer.as_ptr(), self.state.as_mut_ptr());
                    }
                }

                self.buffer_len = 0;
            }
        }

        // Process complete blocks directly
        let remaining_blocks = (input_len - offset) >> 6; // div by 64
        let blocks_end = offset + (remaining_blocks << 6);

        // Process blocks in batches to improve cache locality
        while offset < blocks_end {
            // Load directly into aligned buffer
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    buffer_u8.as_mut_ptr(),
                    64,
                );
            }

            #[cfg(target_endian = "little")]
            {
                // Unroll swap loop for better performance
                self.buffer[0] = self.buffer[0].swap_bytes();
                self.buffer[1] = self.buffer[1].swap_bytes();
                self.buffer[2] = self.buffer[2].swap_bytes();
                self.buffer[3] = self.buffer[3].swap_bytes();
                self.buffer[4] = self.buffer[4].swap_bytes();
                self.buffer[5] = self.buffer[5].swap_bytes();
                self.buffer[6] = self.buffer[6].swap_bytes();
                self.buffer[7] = self.buffer[7].swap_bytes();
                self.buffer[8] = self.buffer[8].swap_bytes();
                self.buffer[9] = self.buffer[9].swap_bytes();
                self.buffer[10] = self.buffer[10].swap_bytes();
                self.buffer[11] = self.buffer[11].swap_bytes();
                self.buffer[12] = self.buffer[12].swap_bytes();
                self.buffer[13] = self.buffer[13].swap_bytes();
                self.buffer[14] = self.buffer[14].swap_bytes();
                self.buffer[15] = self.buffer[15].swap_bytes();
            }

            unsafe {
                if self.initial {
                    sha256_compression_initial(self.buffer.as_ptr(), self.state.as_mut_ptr());
                    self.initial = false;
                } else {
                    sha256_compression(self.buffer.as_ptr(), self.state.as_mut_ptr());
                }
            }

            offset += 64;
        }

        // Buffer remaining bytes
        let remaining = input_len - offset;
        if remaining > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    buffer_u8.as_mut_ptr(),
                    remaining,
                );
            }
            self.buffer_len = remaining;
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len << 3; // * 8

        // Cast buffer to u8 for padding
        let buffer_u8 =
            unsafe { core::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u8, 64) };

        // Add padding byte
        buffer_u8[self.buffer_len] = 0x80;
        let padding_start = self.buffer_len + 1;

        // Determine if we need an extra block
        if self.buffer_len < 56 {
            // Single block case - zero padding and add length
            unsafe {
                // Zero fill from padding_start to 56
                core::ptr::write_bytes(
                    buffer_u8.as_mut_ptr().add(padding_start),
                    0,
                    56 - padding_start,
                );

                // Write length as big-endian u64
                *(buffer_u8.as_mut_ptr().add(56) as *mut u64) = bit_len.to_be();
            }

            #[cfg(target_endian = "little")]
            {
                // Swap all 16 words
                self.buffer[0] = self.buffer[0].swap_bytes();
                self.buffer[1] = self.buffer[1].swap_bytes();
                self.buffer[2] = self.buffer[2].swap_bytes();
                self.buffer[3] = self.buffer[3].swap_bytes();
                self.buffer[4] = self.buffer[4].swap_bytes();
                self.buffer[5] = self.buffer[5].swap_bytes();
                self.buffer[6] = self.buffer[6].swap_bytes();
                self.buffer[7] = self.buffer[7].swap_bytes();
                self.buffer[8] = self.buffer[8].swap_bytes();
                self.buffer[9] = self.buffer[9].swap_bytes();
                self.buffer[10] = self.buffer[10].swap_bytes();
                self.buffer[11] = self.buffer[11].swap_bytes();
                self.buffer[12] = self.buffer[12].swap_bytes();
                self.buffer[13] = self.buffer[13].swap_bytes();
                self.buffer[14] = self.buffer[14].swap_bytes();
                self.buffer[15] = self.buffer[15].swap_bytes();
            }

            unsafe {
                if self.initial {
                    sha256_compression_initial(self.buffer.as_ptr(), self.state.as_mut_ptr());
                } else {
                    sha256_compression(self.buffer.as_ptr(), self.state.as_mut_ptr());
                }
            }
        } else {
            // Two block case
            unsafe {
                // Zero fill rest of first block
                core::ptr::write_bytes(
                    buffer_u8.as_mut_ptr().add(padding_start),
                    0,
                    64 - padding_start,
                );
            }

            #[cfg(target_endian = "little")]
            {
                self.buffer[0] = self.buffer[0].swap_bytes();
                self.buffer[1] = self.buffer[1].swap_bytes();
                self.buffer[2] = self.buffer[2].swap_bytes();
                self.buffer[3] = self.buffer[3].swap_bytes();
                self.buffer[4] = self.buffer[4].swap_bytes();
                self.buffer[5] = self.buffer[5].swap_bytes();
                self.buffer[6] = self.buffer[6].swap_bytes();
                self.buffer[7] = self.buffer[7].swap_bytes();
                self.buffer[8] = self.buffer[8].swap_bytes();
                self.buffer[9] = self.buffer[9].swap_bytes();
                self.buffer[10] = self.buffer[10].swap_bytes();
                self.buffer[11] = self.buffer[11].swap_bytes();
                self.buffer[12] = self.buffer[12].swap_bytes();
                self.buffer[13] = self.buffer[13].swap_bytes();
                self.buffer[14] = self.buffer[14].swap_bytes();
                self.buffer[15] = self.buffer[15].swap_bytes();
            }

            unsafe {
                if self.initial {
                    sha256_compression_initial(self.buffer.as_ptr(), self.state.as_mut_ptr());
                    self.initial = false;
                } else {
                    sha256_compression(self.buffer.as_ptr(), self.state.as_mut_ptr());
                }
            }

            // Second block: all zeros except length at the end
            self.buffer[0] = 0;
            self.buffer[1] = 0;
            self.buffer[2] = 0;
            self.buffer[3] = 0;
            self.buffer[4] = 0;
            self.buffer[5] = 0;
            self.buffer[6] = 0;
            self.buffer[7] = 0;
            self.buffer[8] = 0;
            self.buffer[9] = 0;
            self.buffer[10] = 0;
            self.buffer[11] = 0;
            self.buffer[12] = 0;
            self.buffer[13] = 0;

            // Write length in last 8 bytes
            #[cfg(target_endian = "little")]
            {
                self.buffer[14] = (bit_len >> 32) as u32;
                self.buffer[15] = bit_len as u32;
                self.buffer[14] = self.buffer[14].swap_bytes();
                self.buffer[15] = self.buffer[15].swap_bytes();
            }
            #[cfg(target_endian = "big")]
            {
                self.buffer[14] = (bit_len >> 32) as u32;
                self.buffer[15] = bit_len as u32;
            }

            unsafe {
                sha256_compression(self.buffer.as_ptr(), self.state.as_mut_ptr());
            }
        }

        // Convert state to bytes
        let mut result = [0u8; 32];
        unsafe {
            let result_u32 = core::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u32, 8);

            #[cfg(target_endian = "little")]
            {
                result_u32[0] = self.state[0].swap_bytes();
                result_u32[1] = self.state[1].swap_bytes();
                result_u32[2] = self.state[2].swap_bytes();
                result_u32[3] = self.state[3].swap_bytes();
                result_u32[4] = self.state[4].swap_bytes();
                result_u32[5] = self.state[5].swap_bytes();
                result_u32[6] = self.state[6].swap_bytes();
                result_u32[7] = self.state[7].swap_bytes();
            }
            #[cfg(target_endian = "big")]
            {
                core::ptr::copy_nonoverlapping(self.state.as_ptr(), result_u32.as_mut_ptr(), 8);
            }
        }

        result
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
    #[inline(always)]
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
