//! Keccak-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha3` crate.
//! On the host

const RATE_IN_BYTES: usize = 136;
const RATE_IN_U64: usize = RATE_IN_BYTES / 8;
const HASH_LEN: usize = 32;

/// Keccak-256 hasher state.
pub struct Keccak256 {
    /// The 25-word (1600-bit) Keccak state.
    state: [u64; 25],
    /// Buffer for incomplete blocks.
    buffer: [u8; RATE_IN_BYTES],
    /// Number of bytes in the buffer.
    buffer_len: usize,
}

impl Keccak256 {
    /// Creates a new Keccak-256 hasher.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            state: [0; 25],
            buffer: [0; RATE_IN_BYTES],
            buffer_len: 0,
        }
    }

    /// Writes data to the hasher.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        let mut offset = 0;

        // Absorb input into the buffer
        if self.buffer_len > 0 {
            let needed = RATE_IN_BYTES - self.buffer_len;
            let to_copy = needed.min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + to_copy]
                .copy_from_slice(&input[..to_copy]);
            self.buffer_len += to_copy;
            offset += to_copy;

            if self.buffer_len == RATE_IN_BYTES {
                self.absorb_buffer();
            }
        }

        // Process full blocks directly from input
        while offset + RATE_IN_BYTES <= input.len() {
            self.absorb_slice(&input[offset..offset + RATE_IN_BYTES]);
            offset += RATE_IN_BYTES;
        }

        // Buffer any remaining input
        let remaining = input.len() - offset;
        if remaining > 0 {
            self.buffer[..remaining].copy_from_slice(&input[offset..]);
            self.buffer_len = remaining;
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; HASH_LEN] {
        // Pad the message. Keccak uses `0x01` padding.
        self.buffer[self.buffer_len] = 0x01;
        self.buffer[RATE_IN_BYTES - 1] |= 0x80;

        self.absorb_buffer();

        let mut hash = [0u8; HASH_LEN];
        let state_bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(self.state.as_ptr() as *const u8, 200) };
        hash.copy_from_slice(&state_bytes[..HASH_LEN]);
        hash
    }

    /// Computes Keccak-256 hash of the input data in one call.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; HASH_LEN] {
        let mut hasher = Self::new();
        hasher.update(input);
        hasher.finalize()
    }

    /// Absorbs a full block from the internal buffer into the state.
    fn absorb_buffer(&mut self) {
        for i in 0..RATE_IN_U64 {
            let word = u64::from_le_bytes(self.buffer[i * 8..(i + 1) * 8].try_into().unwrap());
            self.state[i] ^= word;
        }
        unsafe {
            keccak_f(self.state.as_mut_ptr());
        }
        self.buffer_len = 0;
    }

    /// Absorbs a full block from a slice into the state.
    fn absorb_slice(&mut self, block: &[u8]) {
        for i in 0..RATE_IN_U64 {
            let word = u64::from_le_bytes(block[i * 8..(i + 1) * 8].try_into().unwrap());
            self.state[i] ^= word;
        }
        unsafe {
            keccak_f(self.state.as_mut_ptr());
        }
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Calls the Keccak-f[1600] permutation custom instruction.
///
/// # Arguments
/// * `state` - Pointer to the 25-word (200-byte) Keccak state, which will be permuted in-place.
///
/// # Safety
/// - `state` must be a valid pointer to 200 bytes of readable and writable memory.
/// - The pointer must be properly aligned for u64 access (8-byte alignment).
#[cfg(not(feature = "host"))]
pub unsafe fn keccak_f(state: *mut u64) {
    // We use funct7=0x01 to distinguish Keccak from SHA-256 (funct7=0x00)
    // using the custom-0 opcode (0x0B).
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x01, x0, {}, x0",
        in(reg) state,
        options(nostack)
    );
}

#[cfg(feature = "host")]
pub unsafe fn keccak_f(state: *mut u64) {
    // On the host, we call our own reference implementation from the tracer crate.
    let state_slice = core::slice::from_raw_parts_mut(state, 25);
    tracer::instruction::inline_keccak256::execute_keccak_f(
        state_slice
            .try_into()
            .expect("State slice was not 25 words"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn test_keccak256_empty() {
        let hash = Keccak256::digest(b"");
        assert_eq!(
            hash,
            hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
        );
    }
}
