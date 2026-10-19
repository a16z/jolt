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

    pub fn compute_expected_result(input: &[u8]) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        blake3::hash(input).as_bytes()[0..crate::OUTPUT_SIZE_IN_BYTES]
            .try_into()
            .unwrap()
    }
}
