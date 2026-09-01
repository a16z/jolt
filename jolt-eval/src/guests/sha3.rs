use super::GuestConfig;

/// SHA-3 (Keccak) guest: hashes an input buffer of `self.0` bytes (value 5u8
/// each, matching `e2e_profiling.rs`).
pub struct Sha3(pub usize);

impl Default for Sha3 {
    fn default() -> Self {
        Self(2048)
    }
}

impl GuestConfig for Sha3 {
    fn package(&self) -> &str {
        "sha3-guest"
    }
    fn label(&self) -> String {
        format!("sha3_{}", self.0)
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&vec![5u8; self.0]).unwrap()
    }
}
