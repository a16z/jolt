use super::GuestConfig;

/// SHA-2 chain guest: iteratively hashes input `num_iters` times.
pub struct Sha2Chain {
    pub input: [u8; 32],
    pub num_iters: u32,
}

impl Default for Sha2Chain {
    fn default() -> Self {
        Self {
            input: [5u8; 32],
            num_iters: 100,
        }
    }
}

impl GuestConfig for Sha2Chain {
    fn package(&self) -> &str {
        "sha2-chain-guest"
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&(self.input, self.num_iters)).unwrap()
    }
    fn bench_name(&self) -> String {
        format!("prover_time_sha2_chain_{}", self.num_iters)
    }
}
