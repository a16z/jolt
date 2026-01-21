//! This file provides high-level API to use BLAKE3 compression, both in host and guest mode.
#[cfg(feature = "host")]
use crate::FLAG_KEYED_HASH;
use crate::{
    BLOCK_INPUT_SIZE_IN_BYTES, CHAINING_VALUE_LEN, COUNTER_LEN, FLAG_CHUNK_END, FLAG_CHUNK_START,
    FLAG_ROOT, IV, MSG_BLOCK_LEN, OUTPUT_SIZE_IN_BYTES,
};

/// 8-byte aligned 32-byte hash/key type for BLAKE3 operations.
///
/// This type ensures proper alignment for efficient 64-bit load/store operations
/// in the RISC-V instruction sequence.
#[repr(align(8))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AlignedHash32(pub [u8; 32]);

impl AlignedHash32 {
    /// Create a new AlignedHash32 from a byte array
    #[inline(always)]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Create a zeroed AlignedHash32
    #[inline(always)]
    pub const fn zeroed() -> Self {
        Self([0u8; 32])
    }

    /// Get a reference to the inner bytes
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Get a mutable reference to the inner bytes
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        &mut self.0
    }
}

impl From<[u8; 32]> for AlignedHash32 {
    #[inline(always)]
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl From<AlignedHash32> for [u8; 32] {
    #[inline(always)]
    fn from(hash: AlignedHash32) -> Self {
        hash.0
    }
}

impl AsRef<[u8; 32]> for AlignedHash32 {
    #[inline(always)]
    fn as_ref(&self) -> &[u8; 32] {
        &self.0
    }
}

impl AsMut<[u8; 32]> for AlignedHash32 {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [u8; 32] {
        &mut self.0
    }
}

pub struct Blake3 {
    /// Hash state (8 x 32-bit words)
    h: [u32; CHAINING_VALUE_LEN],
    /// Buffer for incomplete blocks
    buffer: [u8; BLOCK_INPUT_SIZE_IN_BYTES],
    /// Current buffer length
    buffer_len: usize,
    /// Total bytes processed
    counter: u64,
}

/// Note: Current implementation only supports hashing up to 64 bytes (single block).
impl Blake3 {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            h: IV,
            buffer: [0; BLOCK_INPUT_SIZE_IN_BYTES],
            buffer_len: 0,
            counter: 0,
        }
    }

    /// Processes input data incrementally.
    ///
    /// # Panics
    /// Panics if the total input exceeds 64 bytes.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        let input_len = input.len();
        if input_len == 0 {
            return;
        }

        let remaining_capacity = BLOCK_INPUT_SIZE_IN_BYTES - self.buffer_len;

        if input_len > remaining_capacity {
            panic!(
                "Buffer overflow: trying to add {input_len} bytes but only {remaining_capacity} bytes remaining"
            );
        }

        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.buffer.as_mut_ptr().add(self.buffer_len),
                input_len,
            );
        }

        self.buffer_len += input_len;
    }

    /// Finalizes the hash and returns the digest.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE_IN_BYTES] {
        // Zero padding
        if self.buffer_len < BLOCK_INPUT_SIZE_IN_BYTES {
            unsafe {
                core::ptr::write_bytes(
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    0,
                    BLOCK_INPUT_SIZE_IN_BYTES - self.buffer_len,
                );
            }
        }

        compression_caller(
            &mut self.h,
            &self.buffer,
            self.counter,
            self.buffer_len as u32,
            FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT,
        );

        to_bytes(self.h)
    }

    /// Computes BLAKE3 hash in one call (max 64 bytes input).
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; OUTPUT_SIZE_IN_BYTES] {
        let len = input.len();
        if len > BLOCK_INPUT_SIZE_IN_BYTES {
            panic!("Input too large: {len} bytes, max is {BLOCK_INPUT_SIZE_IN_BYTES}");
        }

        let mut h = IV;
        compress_direct(
            &mut h,
            input,
            0,
            len as u32,
            FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT,
        );
        to_bytes(h)
    }
}

#[inline(always)]
fn compression_caller(
    hash_state: &mut [u32; CHAINING_VALUE_LEN],
    message_block: &[u8],
    counter: u64,
    input_bytes_num: u32,
    flags: u32,
) {
    let mut message = [0u32; MSG_BLOCK_LEN + COUNTER_LEN + 2];
    debug_assert_eq!(message_block.len(), BLOCK_INPUT_SIZE_IN_BYTES);

    #[cfg(target_endian = "little")]
    unsafe {
        core::ptr::copy_nonoverlapping(
            message_block.as_ptr() as *const u32,
            message.as_mut_ptr(),
            MSG_BLOCK_LEN,
        );
    }

    #[cfg(target_endian = "big")]
    {
        // For big-endian, we need to convert each u32
        for i in 0..MSG_BLOCK_LEN {
            let offset = i * 4;
            message[i] = u32::from_le_bytes([
                message_block[offset],
                message_block[offset + 1],
                message_block[offset + 2],
                message_block[offset + 3],
            ]);
        }
    }

    message[MSG_BLOCK_LEN] = counter as u32;
    message[MSG_BLOCK_LEN + 1] = (counter >> 32) as u32;
    message[MSG_BLOCK_LEN + COUNTER_LEN] = input_bytes_num;
    message[MSG_BLOCK_LEN + COUNTER_LEN + 1] = flags;

    unsafe {
        blake3_compress(hash_state.as_mut_ptr(), message.as_ptr());
    }
}

impl Default for Blake3 {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert hash state to output bytes.
#[inline(always)]
pub(crate) fn to_bytes(h: [u32; CHAINING_VALUE_LEN]) -> [u8; OUTPUT_SIZE_IN_BYTES] {
    #[cfg(target_endian = "little")]
    {
        unsafe { core::mem::transmute(h) }
    }

    #[cfg(target_endian = "big")]
    {
        let mut hash = [0u8; OUTPUT_SIZE_IN_BYTES];
        for i in 0..CHAINING_VALUE_LEN {
            let bytes = h[i].to_le_bytes();
            hash[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }
        hash
    }
}

/// Compress with direct copy to message array (no intermediate buffer).
/// Low-level BLAKE3 compress function with configurable flags.
///
/// This function performs a single BLAKE3 compression operation, suitable for:
/// - Merkle tree internal nodes (with `PARENT` flag)
/// - Merkle tree root (with `PARENT | ROOT` flags)
/// - Leaf CV computation (with `CHUNK_START | CHUNK_END` flags)
///
/// # Arguments
/// * `hash_state` - The chaining value (typically IV for fresh compression)
/// * `input` - Input data (up to 64 bytes, will be zero-padded)
/// * `counter` - Block counter (usually 0 for single-block operations)
/// * `input_bytes_num` - Actual number of input bytes (for padding)
/// * `flags` - BLAKE3 flags (CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH, etc.)
///
/// # Example
/// ```ignore
/// use blake3_inline::{compress_direct, IV, CHAINING_VALUE_LEN};
///
/// const PARENT: u32 = 1 << 2;
///
/// let mut cv = IV;
/// let input: [u8; 64] = /* left_cv || right_cv */;
/// compress_direct(&mut cv, &input, 0, 64, PARENT);
/// ```
#[inline(always)]
pub(crate) fn compress_direct(
    hash_state: &mut [u32; CHAINING_VALUE_LEN],
    input: &[u8],
    counter: u64,
    input_bytes_num: u32,
    flags: u32,
) {
    let mut message = [0u32; MSG_BLOCK_LEN + COUNTER_LEN + 2];
    let len = input.len();

    #[cfg(target_endian = "little")]
    if len > 0 {
        unsafe {
            // Copy input directly to message (padded with zeros)
            core::ptr::copy_nonoverlapping(input.as_ptr(), message.as_mut_ptr() as *mut u8, len);
        }
    }

    #[cfg(target_endian = "big")]
    {
        let full_words = len / 4;
        let remaining = len % 4;

        for i in 0..full_words {
            let offset = i * 4;
            message[i] = u32::from_le_bytes([
                input[offset],
                input[offset + 1],
                input[offset + 2],
                input[offset + 3],
            ]);
        }

        if remaining > 0 {
            let mut bytes = [0u8; 4];
            let offset = full_words * 4;
            for j in 0..remaining {
                bytes[j] = input[offset + j];
            }
            message[full_words] = u32::from_le_bytes(bytes);
        }
    }

    message[MSG_BLOCK_LEN] = counter as u32;
    message[MSG_BLOCK_LEN + 1] = (counter >> 32) as u32;
    message[MSG_BLOCK_LEN + COUNTER_LEN] = input_bytes_num;
    message[MSG_BLOCK_LEN + COUNTER_LEN + 1] = flags;

    unsafe {
        blake3_compress(hash_state.as_mut_ptr(), message.as_ptr());
    }
}

/// BLAKE3 compression function - guest implementation.
///
/// # Safety
/// - `chaining_value` must be a valid pointer to 32 bytes of readable and writable memory.
/// - `message` must be a valid pointer to 64 bytes of readable memory.
/// - Both pointers must be properly aligned for u32 access (4-byte alignment).
#[cfg(not(feature = "host"))]
pub(crate) unsafe fn blake3_compress(chaining_value: *mut u32, message: *const u32) {
    use crate::{BLAKE3_FUNCT3, BLAKE3_FUNCT7, INLINE_OPCODE};
    // Memory layout for BLAKE3 instruction:
    // rs1: points to chaining value (32 bytes)
    // rs2: points to message block (64 bytes) + counter (8 bytes) + block_len (4 bytes) + flags (4 bytes)

    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const BLAKE3_FUNCT3,
        funct7 = const BLAKE3_FUNCT7,
        rs1 = in(reg) chaining_value,
        rs2 = in(reg) message,
        options(nostack)
    );
}

/// BLAKE3 compression function - host implementation.
///
/// # Safety
/// - `chaining_value` must be a valid pointer to 32 bytes.
/// - `message` must be a valid pointer to 64 bytes.
#[cfg(feature = "host")]
pub(crate) unsafe fn blake3_compress(chaining_value: *mut u32, message: *const u32) {
    // Memory layout for BLAKE3 instruction:
    // message points to: message block (64 bytes / 16 u32s) + counter (8 bytes / 2 u32s) + block_len (4 bytes / 1 u32) + flags (4 bytes / 1 u32)

    // Extract the message block (first 16 u32s)
    let message_block = &*(message as *const [u32; 16]);

    // Extract counter (next 2 u32s at offset 16)
    let counter_ptr = message.add(16);
    let counter_low = *counter_ptr;
    let counter_high = *counter_ptr.add(1);
    let counter_array = [counter_low, counter_high];

    // Extract block_len (next u32 at offset 18)
    let block_len = *message.add(18);

    // Extract flags (next u32 at offset 19)
    let flags = *message.add(19);

    // On the host, we call our reference implementation from the exec module.
    crate::exec::execute_blake3_compression(
        &mut *(chaining_value as *mut [u32; 8]),
        message_block,
        &counter_array,
        block_len,
        flags,
    );
}

/// BLAKE3 Keyed64 - guest implementation (internal).
/// Hash two child CVs for Merkle tree.
/// ABI: rs1 = left, rs2 = right, rd = iv (in/out)
#[cfg(not(feature = "host"))]
#[inline(always)]
unsafe fn blake3_keyed64_compress(left: *const u32, right: *const u32, iv: *mut u32) {
    use crate::{BLAKE3_FUNCT7, BLAKE3_KEYED64_FUNCT3, INLINE_OPCODE};

    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const BLAKE3_KEYED64_FUNCT3,
        funct7 = const BLAKE3_FUNCT7,
        rd = in(reg) iv,
        rs1 = in(reg) left,
        rs2 = in(reg) right,
        options(nostack)
    );
}

/// BLAKE3 Keyed64 - host implementation (internal).
/// Matches blake3::keyed_hash for 64-byte input.
#[cfg(feature = "host")]
#[inline(always)]
unsafe fn blake3_keyed64_compress(left: *const u32, right: *const u32, key: *mut u32) {
    // Concatenate left || right as message
    let mut message = [0u32; 16];
    core::ptr::copy_nonoverlapping(left, message.as_mut_ptr(), 8);
    core::ptr::copy_nonoverlapping(right, message.as_mut_ptr().add(8), 8);

    let key_arr = &mut *(key as *mut [u32; 8]);

    // flags = CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
    crate::exec::execute_blake3_compression(
        key_arr,
        &message,
        &[0, 0],
        64,
        FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | FLAG_KEYED_HASH,
    );
}

/// BLAKE3 keyed_hash for 64-byte input: `blake3::keyed_hash(key, left || right)`
///
/// This is equivalent to calling `blake3::keyed_hash(&key, &[left, right].concat())`
/// but optimized for when left and right are in separate memory locations.
/// The key is modified in-place to contain the hash output.
///
/// # Arguments
/// * `left` - 32-byte left half of input (8-byte aligned via AlignedHash32)
/// * `right` - 32-byte right half of input (8-byte aligned via AlignedHash32)
/// * `key` - 32-byte key (8-byte aligned via AlignedHash32), will be overwritten with hash output
///
/// # Example
/// ```ignore
/// let left = AlignedHash32::new([0xAA; 32]);
/// let right = AlignedHash32::new([0xBB; 32]);
/// let mut key = AlignedHash32::new([...]); // your 32-byte key
/// blake3_keyed64(&left, &right, &mut key);
/// // key now contains blake3::keyed_hash(&original_key, &[left, right].concat())
/// ```
#[inline(always)]
pub fn blake3_keyed64(left: &AlignedHash32, right: &AlignedHash32, key: &mut AlignedHash32) {
    unsafe {
        blake3_keyed64_compress(
            left.0.as_ptr() as *const u32,
            right.0.as_ptr() as *const u32,
            key.0.as_mut_ptr() as *mut u32,
        );
    }
}

/// Standard BLAKE3 IV as AlignedHash32 (for use with blake3_keyed64)
pub const BLAKE3_IV: AlignedHash32 = AlignedHash32([
    0x67, 0xe6, 0x09, 0x6a, // 0x6a09e667 (little-endian)
    0x85, 0xae, 0x67, 0xbb, // 0xbb67ae85
    0x72, 0xf3, 0x6e, 0x3c, // 0x3c6ef372
    0x3a, 0xf5, 0x4f, 0xa5, // 0xa54ff53a
    0x7f, 0x52, 0x0e, 0x51, // 0x510e527f
    0x8c, 0x68, 0x05, 0x9b, // 0x9b05688c
    0xab, 0xd9, 0x83, 0x1f, // 0x1f83d9ab
    0x19, 0xcd, 0xe0, 0x5b, // 0x5be0cd19
]);

#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use super::Blake3;
    use crate::{test_utils::helpers::*, BLOCK_INPUT_SIZE_IN_BYTES};

    fn random_partition(data: &[u8]) -> Vec<&[u8]> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        if data.is_empty() {
            return vec![&[]];
        }
        let len = data.len();
        let mut parts = Vec::new();
        let mut partitioned_num = 0usize;
        // Use a fixed seed for deterministic test results
        let mut rng = StdRng::seed_from_u64(42);
        while partitioned_num < len {
            let remaining = len - partitioned_num;
            let take = rng.gen_range(1..=remaining);
            parts.push(&data[partitioned_num..partitioned_num + take]);
            partitioned_num += take;
        }
        parts
    }

    #[test]
    fn test_digest_matches_standard() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(67890);

        for _ in 0..1000 {
            let len = rng.gen::<usize>() % (BLOCK_INPUT_SIZE_IN_BYTES);
            let input = generate_random_bytes(len);
            let result = Blake3::digest(&input);
            let expected = compute_expected_result(&input);
            assert_eq!(result, expected, "digest mismatch for input={input:02x?}");
        }
    }

    #[test]
    fn test_streaming_update_finalize_matches_standard() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(54321);

        for _ in 0..1000 {
            let len = rng.gen::<usize>() % (BLOCK_INPUT_SIZE_IN_BYTES + 1);
            let data = generate_random_bytes(len);
            let expected = compute_expected_result(&data);
            let parts = random_partition(&data);
            let mut hasher = Blake3::new();
            for p in &parts {
                hasher.update(p);
            }
            let got = hasher.finalize();
            assert_eq!(got, expected, "stream mismatch for input={data:02x?}");
        }
    }

    #[test]
    #[should_panic]
    fn test_panics_on_larger_than_block_input() {
        // 65 bytes should trigger panic
        let input = vec![0u8; BLOCK_INPUT_SIZE_IN_BYTES + 1];
        let _ = Blake3::digest(&input);
    }

    #[test]
    fn test_edge_cases() {
        // Empty
        let empty: &[u8] = &[];
        assert_eq!(Blake3::digest(empty), compute_expected_result(empty));

        // Length 64 (block-size)
        let mut l64 = [0u8; 64];
        for (i, b) in l64.iter_mut().enumerate() {
            *b = 255 - i as u8;
        }
        assert_eq!(Blake3::digest(&l64), compute_expected_result(&l64));

        // All zeros (64 bytes)
        let zeros = [0u8; 64];
        assert_eq!(Blake3::digest(&zeros), compute_expected_result(&zeros));

        // All 0xFF (64 bytes)
        let maxes = [0xFFu8; 64];
        assert_eq!(Blake3::digest(&maxes), compute_expected_result(&maxes));
    }

    #[test]
    fn test_blake3_aligned_vs_unaligned() {
        // Test various sizes up to 64 bytes (Blake3 block size limit)
        let test_sizes = [0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64];

        for &size in &test_sizes {
            // Create aligned buffer
            let aligned: Vec<u8> = (0..size).map(|i| (i * 37 + 11) as u8).collect();

            // Create unaligned buffer by adding 1-byte offset
            let mut unaligned_buf = vec![0u8; size + 1];
            unaligned_buf[1..].copy_from_slice(&aligned);
            let unaligned = &unaligned_buf[1..];

            // Verify alignment difference
            if size > 0 {
                assert_ne!(
                    aligned.as_ptr() as usize % 4,
                    unaligned.as_ptr() as usize % 4,
                    "Test setup error: pointers should have different alignment"
                );
            }

            // Both should produce identical results
            let aligned_result = Blake3::digest(&aligned);
            let unaligned_result = Blake3::digest(unaligned);

            assert_eq!(
                aligned_result, unaligned_result,
                "Blake3: aligned vs unaligned mismatch at size {size}"
            );

            // Also verify against reference implementation
            let expected = compute_expected_result(&aligned);
            assert_eq!(
                aligned_result, expected,
                "Blake3: result doesn't match reference at size {size}"
            );
        }
    }

    #[test]
    fn test_blake3_keyed64_matches_reference() {
        // Test that blake3_keyed64 matches the official blake3::keyed_hash for 64-byte input
        use super::AlignedHash32;
        use rand::rngs::StdRng;
        use rand::{RngCore, SeedableRng};

        let mut rng = StdRng::seed_from_u64(77777);

        for _ in 0..100 {
            let mut left = AlignedHash32::zeroed();
            let mut right = AlignedHash32::zeroed();
            let mut key = AlignedHash32::zeroed();
            rng.fill_bytes(&mut left.0);
            rng.fill_bytes(&mut right.0);
            rng.fill_bytes(&mut key.0);

            // Concatenate left || right as 64-byte input
            let mut input = [0u8; 64];
            input[..32].copy_from_slice(&left.0);
            input[32..].copy_from_slice(&right.0);

            // Call official blake3::keyed_hash
            let expected = blake3::keyed_hash(&key.0, &input);

            // Call our keyed64 function (key is passed as iv, modified in-place)
            let mut result_iv = key;
            super::blake3_keyed64(&left, &right, &mut result_iv);

            assert_eq!(
                result_iv.0,
                *expected.as_bytes(),
                "blake3_keyed64 does not match blake3::keyed_hash"
            );
        }
    }

    #[test]
    fn test_blake3_keyed64_deterministic() {
        use super::AlignedHash32;

        let left = AlignedHash32::new([0xAAu8; 32]);
        let right = AlignedHash32::new([0xBBu8; 32]);

        let mut iv1 = super::BLAKE3_IV;
        let mut iv2 = super::BLAKE3_IV;

        super::blake3_keyed64(&left, &right, &mut iv1);
        super::blake3_keyed64(&left, &right, &mut iv2);

        assert_eq!(iv1, iv2, "blake3_keyed64 should be deterministic");
    }

    #[test]
    fn test_blake3_keyed64_different_inputs() {
        use super::AlignedHash32;

        let left1 = AlignedHash32::new([0x11u8; 32]);
        let right1 = AlignedHash32::new([0x22u8; 32]);
        let left2 = AlignedHash32::new([0x33u8; 32]);
        let right2 = AlignedHash32::new([0x44u8; 32]);

        let mut iv1 = super::BLAKE3_IV;
        let mut iv2 = super::BLAKE3_IV;

        super::blake3_keyed64(&left1, &right1, &mut iv1);
        super::blake3_keyed64(&left2, &right2, &mut iv2);

        assert_ne!(
            iv1, iv2,
            "Different inputs should produce different results"
        );
    }
}
