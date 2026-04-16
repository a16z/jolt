pub mod core;
pub mod modular;

use crate::output::Run;
use crate::programs::Program;

/// Raw per-iteration measurements emitted by a stack runner. The orchestrator
/// aggregates these into a median `Run`.
#[derive(Clone, Debug)]
pub struct IterMetrics {
    pub prove_ms: f64,
    pub verify_ms: f64,
    pub peak_rss_mb: u64,
    pub proof_bytes: u64,
}

/// A stack runner executes N iterations of prove+verify on a given program
/// and returns either per-iteration metrics or an unsupported `Run`.
pub enum StackOutcome {
    Metrics(Vec<IterMetrics>),
    Unsupported(Run),
}

pub trait StackRunner {
    fn run(
        &self,
        program: Program,
        iters: usize,
        warmup: usize,
        log_t: Option<usize>,
    ) -> StackOutcome;
}
