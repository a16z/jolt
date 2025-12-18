//! SHA-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha2` crate.

use core::mem::MaybeUninit;

/// SHA-256 hasher state.
pub struct Sha256 {
    /// Current hash state (8 x 32-bit words)
    ///
    /// # Safety invariants
    /// - Uninitialized until first compression function call
    /// - After first `sha256_compression_initial` call, all 8 words are initialized
    /// - Remains initialized for the lifetime of the hasher
    state: [MaybeUninit<u32>; 8],
    /// Buffer for incomplete blocks - aligned for u32 access
    ///
    /// # Safety invariants  
    /// - First `buffer_len` bytes contain valid data when viewed as `*mut u8`
    /// - Only elements 0..(buffer_len / 4) are fully initialized u32 values
    /// - During block processing, all 16 words (64 bytes) are initialized before compression
    /// - After compression, `buffer_len` is reset to 0 (buffer contents are don't-care)
    ///
    /// # Memory layout
    /// - Can be safely cast to `*mut u8` for byte-level operations
    /// - Must maintain u32 alignment for word-level access
    buffer: [MaybeUninit<u32>; 16],
    /// Number of bytes in the buffer
    buffer_len: usize,
    /// Total number of bytes processed
    total_len: u64,
    /// Whether this is the initial block
    initial: bool,
}

impl Sha256 {
    /// Creates a new SHA-256 hasher.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            // We these uninitialized as a cycle optimization
            state: [MaybeUninit::uninit(); 8],
            buffer: [MaybeUninit::uninit(); 16],
            buffer_len: 0,
            total_len: 0,
            initial: true,
        }
    }

    #[inline(always)]
    unsafe fn buffer_as_u32_mut(&mut self) -> &mut [u32] {
        core::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut u32, 16)
    }

    #[inline(always)]
    unsafe fn state_as_u32(&self) -> &[u32] {
        core::slice::from_raw_parts(self.state.as_ptr() as *const u32, 8)
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
                    let buf = unsafe { self.buffer_as_u32_mut() };
                    Sha256::swap_bytes(buf);
                }

                unsafe {
                    self.sha256_compress();
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
                let buf = unsafe { self.buffer_as_u32_mut() };
                Sha256::swap_bytes(buf);
            }

            unsafe {
                self.sha256_compress();
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
                let buf = unsafe { self.buffer_as_u32_mut() };
                Sha256::swap_bytes(buf);
            }

            unsafe {
                self.sha256_compress();
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
                let buf = unsafe { self.buffer_as_u32_mut() };
                Sha256::swap_bytes(buf);
            }

            unsafe {
                self.sha256_compress();
            }

            // Second block: all zeros except length at the end
            // Unroll the loop for cycle optimization
            self.buffer[0].write(0);
            self.buffer[1].write(0);
            self.buffer[2].write(0);
            self.buffer[3].write(0);
            self.buffer[4].write(0);
            self.buffer[5].write(0);
            self.buffer[6].write(0);
            self.buffer[7].write(0);
            self.buffer[8].write(0);
            self.buffer[9].write(0);
            self.buffer[10].write(0);
            self.buffer[11].write(0);
            self.buffer[12].write(0);
            self.buffer[13].write(0);

            // Write length in last 8 bytes (big-endian u32 values)
            // Note: No swap needed here because the second block buffer
            // is NOT passed through swap_bytes before compression
            self.buffer[14].write((bit_len >> 32) as u32);
            self.buffer[15].write(bit_len as u32);

            unsafe {
                sha256_compression(
                    self.buffer.as_ptr() as *const u32,
                    self.state.as_mut_ptr() as *mut u32,
                );
            }
        }

        // Convert state to bytes
        let mut result = [0u8; 32];
        unsafe {
            let result_u32 = core::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u32, 8);
            let state = self.state_as_u32();

            #[cfg(target_endian = "little")]
            {
                // Unroll the loop for cycle optimization
                result_u32[0] = swap_bytes(state[0]);
                result_u32[1] = swap_bytes(state[1]);
                result_u32[2] = swap_bytes(state[2]);
                result_u32[3] = swap_bytes(state[3]);
                result_u32[4] = swap_bytes(state[4]);
                result_u32[5] = swap_bytes(state[5]);
                result_u32[6] = swap_bytes(state[6]);
                result_u32[7] = swap_bytes(state[7]);
            }
            #[cfg(target_endian = "big")]
            {
                core::ptr::copy_nonoverlapping(state.as_ptr(), result_u32.as_mut_ptr(), 8);
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

    #[inline(always)]
    unsafe fn sha256_compress(&mut self) {
        if self.initial {
            sha256_compression_initial(
                self.buffer.as_ptr() as *const u32,
                self.state.as_mut_ptr() as *mut u32,
            );
            self.initial = false;
        } else {
            sha256_compression(
                self.buffer.as_ptr() as *const u32,
                self.state.as_mut_ptr() as *mut u32,
            );
        }
    }

    #[cfg(target_endian = "little")]
    #[inline(always)]
    fn swap_bytes(buf: &mut [u32]) {
        // Unroll the loop for cycle optimization
        buf[0] = swap_bytes(buf[0]);
        buf[1] = swap_bytes(buf[1]);
        buf[2] = swap_bytes(buf[2]);
        buf[3] = swap_bytes(buf[3]);
        buf[4] = swap_bytes(buf[4]);
        buf[5] = swap_bytes(buf[5]);
        buf[6] = swap_bytes(buf[6]);
        buf[7] = swap_bytes(buf[7]);
        buf[8] = swap_bytes(buf[8]);
        buf[9] = swap_bytes(buf[9]);
        buf[10] = swap_bytes(buf[10]);
        buf[11] = swap_bytes(buf[11]);
        buf[12] = swap_bytes(buf[12]);
        buf[13] = swap_bytes(buf[13]);
        buf[14] = swap_bytes(buf[14]);
        buf[15] = swap_bytes(buf[15]);
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
pub(crate) unsafe fn sha256_compression(input: *const u32, state: *mut u32) {
    use crate::{INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const SHA256_FUNCT3,
        funct7 = const SHA256_FUNCT7,
        rs1 = in(reg) state,
        rs2 = in(reg) input,
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
pub(crate) unsafe fn sha256_compression(input: *const u32, state: *mut u32) {
    use crate::exec;

    let input_array = *(input as *const [u32; 16]);
    let state_array = *(state as *const [u32; 8]);
    let result = exec::execute_sha256_compression(state_array, input_array);
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
pub(crate) unsafe fn sha256_compression_initial(input: *const u32, state: *mut u32) {
    use crate::{INLINE_OPCODE, SHA256_INIT_FUNCT3, SHA256_INIT_FUNCT7};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const SHA256_INIT_FUNCT3,
        funct7 = const SHA256_INIT_FUNCT7,
        rs1 = in(reg) state,
        rs2 = in(reg) input,
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
pub(crate) unsafe fn sha256_compression_initial(input: *const u32, state: *mut u32) {
    use crate::exec;

    let input = *(input as *const [u32; 16]);
    let result = exec::execute_sha256_compression_initial(input);
    std::ptr::copy_nonoverlapping(result.as_ptr(), state, 8)
}

/// Swap bytes of a u32 - uses virtual instruction on RISC-V, fallback on host
#[cfg(not(feature = "host"))]
fn swap_bytes(mut v: u32) -> u32 {
    unsafe {
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {r_inout}, {r_inout}, 0",
            opcode = const crate::VIRTUAL_INSTRUCTION_TYPE_I_OPCODE,
            funct3 = const crate::REV8W_FUNCT3,
            r_inout = inout(reg) v,
            options(nostack)
        );
    }
    v
}

#[cfg(feature = "host")]
fn swap_bytes(v: u32) -> u32 {
    v.swap_bytes()
}
