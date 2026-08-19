//! SHA-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha2` crate.

use core::mem::MaybeUninit;

/// SHA-256 hasher state.
#[repr(C, align(8))]
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

        // WARNING: byte-level buffer pointers are derived fresh inside each
        // unsafe block. Holding a `&mut [u8]` view of `self.buffer` across the
        // `&mut self` calls below (`buffer_as_u32_mut`, `sha256_compress`)
        // would be aliasing UB.

        // Handle partial buffer
        if self.buffer_len != 0 {
            let needed = 64 - self.buffer_len;
            let to_copy = needed.min(input_len);

            // SAFETY: `buffer_len < 64` on entry and `to_copy <= 64 - buffer_len`,
            // so the copy stays within the 64-byte buffer.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    (self.buffer.as_mut_ptr() as *mut u8).add(self.buffer_len),
                    to_copy,
                );
            }

            self.buffer_len += to_copy;
            offset = to_copy;

            if self.buffer_len == 64 {
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
            // SAFETY: copies exactly one 64-byte block into the 64-byte buffer.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    self.buffer.as_mut_ptr() as *mut u8,
                    64,
                );
            }

            unsafe {
                self.sha256_compress();
            }

            offset += 64;
        }

        // Buffer remaining bytes
        let remaining = input_len - offset;
        if remaining > 0 {
            // SAFETY: `remaining < 64` because all complete blocks were consumed above.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    self.buffer.as_mut_ptr() as *mut u8,
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

        // Add padding byte
        // SAFETY: `update` never leaves a full buffer, so `buffer_len < 64` and
        // the write is in bounds. Pointer derived fresh (see WARNING in `update`).
        unsafe {
            (self.buffer.as_mut_ptr() as *mut u8)
                .add(self.buffer_len)
                .write(0x80);
        }
        let padding_start = self.buffer_len + 1;

        // Determine if we need an extra block
        if self.buffer_len < 56 {
            // Single block case - zero padding and add length
            // SAFETY: `padding_start <= 56`, so the zero fill ends at byte 56 and
            // the length write covers bytes 56..64. The u64 store is 8-byte
            // aligned: the struct is `repr(C, align(8))` and `buffer` sits at
            // offset 32, so buffer byte 56 is 8-byte aligned.
            unsafe {
                let buffer_ptr = self.buffer.as_mut_ptr() as *mut u8;
                // Zero fill from padding_start to 56
                core::ptr::write_bytes(buffer_ptr.add(padding_start), 0, 56 - padding_start);

                // Write length as big-endian u64
                (buffer_ptr.add(56) as *mut u64).write(bit_len.to_be());
            }

            unsafe {
                self.sha256_compress();
            }
        } else {
            // Two block case
            // SAFETY: `padding_start <= 64`; zero fill stays within the buffer.
            unsafe {
                // Zero fill rest of first block
                core::ptr::write_bytes(
                    (self.buffer.as_mut_ptr() as *mut u8).add(padding_start),
                    0,
                    64 - padding_start,
                );
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

            // Store the length with the same big-endian byte layout as message blocks.
            self.buffer[14].write(((bit_len >> 32) as u32).to_be());
            self.buffer[15].write((bit_len as u32).to_be());

            unsafe {
                sha256_compression(
                    self.buffer.as_ptr() as *const u32,
                    self.state.as_mut_ptr() as *mut u32,
                );
            }
        }

        // Convert state to big-endian bytes
        // SAFETY: state is fully initialized (a compression ran above).
        let state = unsafe { self.state_as_u32() };

        // Unrolled for cycle optimization
        #[cfg(target_endian = "little")]
        let words: [u32; 8] = [
            swap_bytes(state[0]),
            swap_bytes(state[1]),
            swap_bytes(state[2]),
            swap_bytes(state[3]),
            swap_bytes(state[4]),
            swap_bytes(state[5]),
            swap_bytes(state[6]),
            swap_bytes(state[7]),
        ];
        #[cfg(target_endian = "big")]
        let words: [u32; 8] = [
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        ];

        // SAFETY: by-value transmute between arrays of equal size; `u8` has no
        // alignment requirement and the byte order was fixed above.
        unsafe { core::mem::transmute::<[u32; 8], [u8; 32]>(words) }
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
/// * `input` - Pointer to a 64-byte block in big-endian byte order
/// * `state` - Pointer to 8 u32 words (32 bytes) of initial state - will be overwritten by result
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of readable and writable memory
/// - Both pointers must be 8-byte aligned (required for doubleword loads on 64-bit targets)
/// - The memory regions must not overlap
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
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
/// * `input` - Pointer to a 64-byte block in big-endian byte order
/// * `state` - Pointer to 8 u32 words (32 bytes) of initial state - will be overwritten by result
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of readable and writable memory
/// - Both pointers must be 8-byte aligned (required for doubleword loads on 64-bit targets)
/// - The memory regions must not overlap
#[cfg(feature = "host")]
pub(crate) unsafe fn sha256_compression(input: *const u32, state: *mut u32) {
    use crate::exec;

    let input_array = (*(input as *const [u32; 16])).map(u32::from_be);
    let state_array = *(state as *const [u32; 8]);
    let result = exec::execute_sha256_compression(state_array, input_array);
    std::ptr::copy_nonoverlapping(result.as_ptr(), state, 8)
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
pub(crate) unsafe fn sha256_compression(_input: *const u32, _state: *mut u32) {
    panic!("sha256_compression requires RISC-V target or host feature");
}

/// Calls the SHA256 compression custom instruction with initial block
///
/// # Arguments
/// * `input` - Pointer to a 64-byte block in big-endian byte order
/// * `state` - Pointer to 8 u32 words (32 bytes) - result will be written here
///
/// Uses the SHA256 initial state constants internally
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of writable memory
/// - Both pointers must be 8-byte aligned (required for doubleword loads on 64-bit targets)
/// - The memory regions must not overlap
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
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
/// * `input` - Pointer to a 64-byte block in big-endian byte order
/// * `state` - Pointer to 8 u32 words (32 bytes) - result will be written here
///
/// Uses the SHA256 initial state constants internally
///
/// # Safety
/// - `input` must be a valid pointer to at least 64 bytes of readable memory
/// - `state` must be a valid pointer to at least 32 bytes of writable memory
/// - Both pointers must be 8-byte aligned (required for doubleword loads on 64-bit targets)
/// - The memory regions must not overlap
#[cfg(feature = "host")]
pub(crate) unsafe fn sha256_compression_initial(input: *const u32, state: *mut u32) {
    use crate::exec;

    let input = (*(input as *const [u32; 16])).map(u32::from_be);
    let result = exec::execute_sha256_compression_initial(input);
    std::ptr::copy_nonoverlapping(result.as_ptr(), state, 8)
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
pub(crate) unsafe fn sha256_compression_initial(_input: *const u32, _state: *mut u32) {
    panic!("sha256_compression_initial requires RISC-V target or host feature");
}

/// Swap bytes of a u32 - uses virtual instruction on RISC-V, fallback on host
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
fn swap_bytes(mut v: u32) -> u32 {
    unsafe {
        core::arch::asm!(
            ".insn r {opcode}, {funct3}, {funct7}, {r_inout}, {r_inout}, x0",
            opcode = const crate::CUSTOM_OPCODE,
            funct3 = const crate::FUNCT3_VIRTUAL_R,
            funct7 = const crate::FUNCT7_VIRTUAL_REV8W,
            r_inout = inout(reg) v,
            options(nostack)
        );
    }
    v
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
fn swap_bytes(v: u32) -> u32 {
    v.swap_bytes()
}
