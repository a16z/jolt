pub type ChainingValue = [u32; crate::CHAINING_VALUE_LEN];
pub type MessageBlock = [u32; crate::MSG_BLOCK_LEN];

#[cfg(test)]
pub mod helpers {
    pub fn generate_random_bytes(len: usize) -> Vec<u8> {
        use rand::rngs::StdRng;
        use rand::{RngCore, SeedableRng};

        let mut buf = vec![0u8; len];
        let mut rng = StdRng::seed_from_u64(12345);
        rng.fill_bytes(&mut buf);
        buf
    }

    pub fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    pub fn compute_expected_result(input: &[u8]) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        blake3::hash(input).as_bytes()[0..crate::OUTPUT_SIZE_IN_BYTES]
            .try_into()
            .unwrap()
    }

    pub fn compute_keyed_expected_result(
        input: &[u8],
        key: [u32; crate::CHAINING_VALUE_LEN],
    ) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        let mut key_bytes = [0u8; 32];
        for (i, word) in key.iter().enumerate() {
            key_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        blake3::keyed_hash(&key_bytes, input).as_bytes()[0..crate::OUTPUT_SIZE_IN_BYTES]
            .try_into()
            .unwrap()
    }
}
