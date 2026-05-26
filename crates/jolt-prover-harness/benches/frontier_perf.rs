use jolt_prover_harness::{evaluate_perf, PerfGate, RunMetrics};

fn main() {
    let gate = PerfGate::canonical_frontier();
    let core = RunMetrics::new(Some(100.0), Some(1_000), None);
    let modular = RunMetrics::new(Some(104.0), Some(1_020), None);
    let _evaluation = evaluate_perf(gate, &core, &modular);
}
