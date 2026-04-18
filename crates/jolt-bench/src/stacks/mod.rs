pub mod core;
pub mod modular;

use crate::programs::Program;

#[derive(Clone, Debug)]
pub struct IterMetrics {
    pub prove_ms: f64,
    pub verify_ms: f64,
    pub peak_rss_mb: u64,
    pub proof_bytes: u64,
}

pub enum StackOutcome {
    Metrics(Vec<IterMetrics>),
}

pub trait StackRunner {
    fn run(
        &self,
        program: Program,
        iters: usize,
        warmup: usize,
        log_t: Option<usize>,
        num_iters_override: Option<u32>,
    ) -> StackOutcome;
}
