use super::GuestConfig;

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
    fn func(&self) -> Option<&str> {
        Some("sha3")
    }
    fn label(&self) -> String {
        format!("sha3_{}", self.0)
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&vec![5u8; self.0]).unwrap()
    }
}
