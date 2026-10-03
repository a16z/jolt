use super::GuestConfig;

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

pub const CYCLES_PER_SHA256: f64 = 3396.0;

impl Sha2Chain {
    pub fn with_target_cycles(target_cycles: usize) -> Self {
        let num_iters = std::cmp::max(1, (target_cycles as f64 / CYCLES_PER_SHA256) as u32);
        Self {
            input: [5u8; 32],
            num_iters,
        }
    }

    pub fn profiling_default() -> Self {
        Self::with_target_cycles(((1usize << 24) as f64 * 0.9) as usize)
    }
}

impl GuestConfig for Sha2Chain {
    fn package(&self) -> &str {
        "sha2-chain-guest"
    }
    fn label(&self) -> String {
        format!("sha2_chain_{}", self.num_iters)
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&(self.input, self.num_iters)).unwrap()
    }
}
