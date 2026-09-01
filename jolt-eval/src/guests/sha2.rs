use super::GuestConfig;

/// SHA-2 guest: hashes an input buffer of `self.0` bytes (value 5u8 each,
/// matching `e2e_profiling.rs`).
pub struct Sha2(pub usize);

impl Default for Sha2 {
    fn default() -> Self {
        Self(2048)
    }
}

impl GuestConfig for Sha2 {
    fn package(&self) -> &str {
        "sha2-guest"
    }
    fn label(&self) -> String {
        format!("sha2_{}", self.0)
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&vec![5u8; self.0]).unwrap()
    }
}
