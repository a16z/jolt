use super::GuestConfig;

/// SHA-3 chain guest: iteratively hashes input `num_iters` times.
pub struct Sha3Chain {
    pub input: [u8; 32],
    pub num_iters: u32,
}

impl Default for Sha3Chain {
    fn default() -> Self {
        // e2e_profiling.rs default
        Self {
            input: [5u8; 32],
            num_iters: 20,
        }
    }
}

impl GuestConfig for Sha3Chain {
    fn package(&self) -> &str {
        "sha3-chain-guest"
    }
    fn label(&self) -> String {
        format!("sha3_chain_{}", self.num_iters)
    }
    fn input(&self) -> Vec<u8> {
        let mut inputs = postcard::to_stdvec(&self.input).unwrap();
        inputs.append(&mut postcard::to_stdvec(&self.num_iters).unwrap());
        inputs
    }
}
