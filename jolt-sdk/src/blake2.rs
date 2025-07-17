//! Blake2b hash function implementation optimized for Jolt zkVM.
//!
//! This module provides Blake2b-256 and Blake2b-512 hash functions.
//! On the host, it calls the tracer implementation. On the guest (zkVM),
//! it uses a custom RISC-V instruction for efficient proving.

const BLOCK_SIZE: usize = 128; // Blake2b block size in bytes
const BLOCK_SIZE_U64: usize = BLOCK_SIZE / 8; // 16 words
const STATE_SIZE: usize = 64; // Blake2b state size in bytes
const STATE_SIZE_U64: usize = STATE_SIZE / 8; // 8 words
const OUTPUT_SIZE: usize = 64;

/// Blake2b hasher state for variable output lengths.
pub struct Blake2b {
    /// The 8-word (64-byte) Blake2b hash state.
    h: [u64; STATE_SIZE_U64],
    /// Buffer for incomplete blocks.
    buffer: [u8; BLOCK_SIZE],
    /// Number of bytes in the buffer.
    buffer_len: usize,
    /// Total number of bytes processed.
    counter: u64,
}

impl Blake2b {
    /// Creates a new Blake2b hasher with specified output length.
    #[inline(always)]
    pub fn new(output_len: usize) -> Self {
        assert!(output_len > 0 && output_len <= 64, "Invalid output length");
        
        // Blake2b initialization vector
        let mut h = [
            0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
            0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
        ];
        
        // XOR h[0] with parameter block: 0x01010000 ^ (kk << 8) ^ nn
        // where kk=0 (unkeyed) and nn=output_len
        h[0] ^= 0x01010000 ^ (output_len as u64);
        
        Self {
            h,
            buffer: [0; BLOCK_SIZE],
            buffer_len: 0,
            counter: 0
        }
    }

    /// Writes data to the hasher.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        if input.len() == 0 {
            return;
        }
        let mut offset = 0;

        // If there's existing data in buffer, try to fill it
        if self.buffer_len > 0 {
            let space_available = BLOCK_SIZE - self.buffer_len;
            let bytes_to_copy = space_available.min(input.len());
            
            self.buffer[self.buffer_len..self.buffer_len + bytes_to_copy]
                .copy_from_slice(&input[..bytes_to_copy]);
            self.buffer_len += bytes_to_copy;
            offset = bytes_to_copy;

            // If buffer is now full, process it
            if self.buffer_len == BLOCK_SIZE {
                self.counter += BLOCK_SIZE as u64;
                self.compress_buffer();
            }
        }

        // Process complete blocks directly from input
        // Using 2*BLOCK_SIZE, we do not need to compress at this step for dd = 1
        while offset + BLOCK_SIZE < input.len() {
            self.counter += BLOCK_SIZE as u64;
            self.compress_slice(&input[offset..offset + BLOCK_SIZE]);
            offset += BLOCK_SIZE;
        }

        // Store any remaining bytes in buffer
        let remaining = input.len() - offset;
        if remaining > 0 {
            self.buffer[..remaining].copy_from_slice(&input[offset..]);
            self.buffer_len = remaining;
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE] {
        // Add remaining bytes to counter
        self.counter += self.buffer_len as u64;
        
        // Pad buffer with zeros
        self.buffer[self.buffer_len..].fill(0);
        
        // Process final block
        self.compress_final_buffer();
        
        // Extract hash bytes
        let mut hash = [0u8; OUTPUT_SIZE];
        let state_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(self.h.as_ptr() as *const u8, STATE_SIZE)
        };
        hash.copy_from_slice(&state_bytes[..OUTPUT_SIZE]);
        hash
    }

    /// Computes Blake2b hash of the input data in one call.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; OUTPUT_SIZE] {
        let mut hasher = Self::new(OUTPUT_SIZE);
        hasher.update(input);
        hasher.finalize()
    }

    /// Compresses the internal buffer.
    fn compress_buffer(&mut self) {
        // Convert buffer to u64 words
        let mut message = [0u64; BLOCK_SIZE_U64];
        for i in 0..BLOCK_SIZE_U64 {
            message[i] = u64::from_le_bytes(
                self.buffer[i * 8..(i + 1) * 8].try_into().unwrap()
            );
        }
        
        unsafe {
            blake2b_compress(
                self.h.as_mut_ptr(),
                message.as_ptr(),
                self.counter,
                0, // not final
            );
        }
        self.buffer_len = 0;
    }

    /// Compresses a slice directly.
    fn compress_slice(&mut self, block: &[u8]) {
        // Convert block to u64 words
        let mut message = [0u64; BLOCK_SIZE_U64];
        for i in 0..BLOCK_SIZE_U64 {
            message[i] = u64::from_le_bytes(
                block[i * 8..(i + 1) * 8].try_into().unwrap()
            );
        }
        
        unsafe {
            blake2b_compress(
                self.h.as_mut_ptr(),
                message.as_ptr(),
                self.counter,
                0, // not final
            );
        }
    }

    /// Compresses the final buffer.
    fn compress_final_buffer(&mut self) {
        // Convert buffer to u64 words
        let mut message = [0u64; BLOCK_SIZE_U64];
        for i in 0..BLOCK_SIZE_U64 {
            message[i] = u64::from_le_bytes(
                self.buffer[i * 8..(i + 1) * 8].try_into().unwrap()
            );
        }
        
        unsafe {
            blake2b_compress(
                self.h.as_mut_ptr(),
                message.as_ptr(),
                self.counter,
                1, // final
            );
        }
    }
}



impl Default for Blake2b {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Calls the Blake2b compression custom instruction.
///
/// # Arguments
/// * `state` - Pointer to the 8-word (64-byte) Blake2b state
/// * `message` - Pointer to the 16-word (128-byte) message block
/// * `counter` - Counter value (number of bytes processed)
/// * `is_final` - Final block flag (1 if final, 0 otherwise)
///
/// # Safety
/// - `state` must be a valid pointer to 64 bytes of readable and writable memory.
/// - `message` must be a valid pointer to 128 bytes of readable memory.
/// - Both pointers must be properly aligned for u64 access (8-byte alignment).
#[cfg(not(feature = "host"))]
pub unsafe fn blake2b_compress(
    state: *mut u64,
    message: *const u64,
    counter: u64,
    is_final: u64,
) {
    // Memory layout for Blake2 instruction:
    // rs1: points to state (64 bytes)
    // rs2: points to message block (128 bytes) + counter (8 bytes) + final flag (8 bytes)
    
    // We need to set up memory with the message block followed by counter and final flag
    let mut block_data = [0u64; 18]; // 16 words message + 1 word counter + 1 word final
    
    // Copy message
    core::ptr::copy_nonoverlapping(message, block_data.as_mut_ptr(), 16);
    // Set counter and final flag
    block_data[16] = counter;
    block_data[17] = is_final;
    
    // Call Blake2 instruction using funct7=0x02 to distinguish from Keccak (0x01) and SHA-256 (0x00)
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x02, x0, {}, {}",
        in(reg) state,
        in(reg) block_data.as_ptr(),
        options(nostack)
    );
}

#[cfg(feature = "host")]
pub unsafe fn blake2b_compress(
    state: *mut u64,
    message: *const u64,
    counter: u64,
    is_final: u64,
) {
    // On the host, we call our reference implementation from the tracer crate.
    let state_slice = core::slice::from_raw_parts_mut(state, 8);
    let message_slice = core::slice::from_raw_parts(message, 16);
    let message_array: [u64; 16] = message_slice.try_into()
        .expect("Message slice was not 16 words");
    
    tracer::instruction::inline_blake2::execute_blake2b_compression(
        state_slice.try_into().expect("State slice was not 8 words"),
        &message_array,
        counter,
        is_final != 0,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn test_blake2b_empty() {
        let hash = Blake2b::digest(b"");
        assert_eq!(
            hash,
            hex!("786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce")
        );
    }

    #[test]
    fn test_blake2b_with_lt_128_byte_input() {
        let hash = Blake2b::digest(b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456");
        assert_eq!(
            hash,
            hex!("2d95bd8dfdf8c4077f9bf54fe1a622e8bff985727a1f937f05c19608b93afbde331cc949d67cf29f3cbe081f2a853c13131b7f8f5d162810eec2e0001df9199f")
        );
    }

    #[test]
    fn test_blake2b_with_128_byte_input() {
        let hash = Blake2b::digest(b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
        assert_eq!(
            hash,
            hex!("687222a8b7e18fe2351529741f9f377dbfe57ccc40ffacd7dad6457eb0f5434b308c25eeb85f2c434889877eae9cfcda86e2220bbedb5ddeeef1db1b76113997")
        );
    }

    #[test]
    fn test_blake2b_with_gt_128_byte_input() {
        let hash = Blake2b::digest(b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef456789abcdef0123456789abcdef0123456789abcdef");
        assert_eq!(
            hash,
            hex!("eec6581ca2d51e7f8bff0cb9e0742b454bad4d28bb5078737a6bce318bb29902ca6c2fd4c412d9ed6bb2940692b39012b69ab81ca33cca4d292f3a095cd84007")
        );
    }

    #[test]
    fn test_blake2b_with_256_byte_input() {
        let hash = Blake2b::digest(b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
        assert_eq!(
            hash,
            hex!("342949a83f4809037dcb71d5d527ef9c8060c20cda8a7e4414bcca487e9bc5726e0d4646b7f869b3f3decb362508ec4672c3314ad345d1c36377fc1f3020585c")
        );
    }
}