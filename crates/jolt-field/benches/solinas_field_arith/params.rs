use std::env;

#[derive(Clone, Copy)]
pub(crate) struct ArithmeticBenchParams {
    pub(crate) latency_iters: usize,
    pub(crate) inverse_latency_iters: usize,
    pub(crate) throughput_iters: usize,
    pub(crate) inverse_throughput_iters: usize,
    pub(crate) streams: usize,
}

impl ArithmeticBenchParams {
    pub(crate) fn from_env(
        prefix: &str,
        latency_default: usize,
        throughput_default: usize,
    ) -> Self {
        let latency_iters = env_usize(&format!("{prefix}_LATENCY_ITERS"), latency_default);
        let inverse_latency_iters =
            env_usize(&format!("{prefix}_INVERSE_LATENCY_ITERS"), 128).min(latency_iters);
        let throughput_iters = env_usize(&format!("{prefix}_THROUGHPUT_ITERS"), throughput_default);
        let inverse_throughput_iters = env_usize(&format!("{prefix}_INVERSE_THROUGHPUT_ITERS"), 32);
        let streams = env_usize(&format!("{prefix}_STREAMS"), 8);

        assert!(latency_iters > 0, "{prefix}_LATENCY_ITERS must be > 0");
        assert!(
            inverse_latency_iters > 0,
            "{prefix}_INVERSE_LATENCY_ITERS must be > 0"
        );
        assert!(
            throughput_iters > 0,
            "{prefix}_THROUGHPUT_ITERS must be > 0"
        );
        assert!(
            inverse_throughput_iters > 0,
            "{prefix}_INVERSE_THROUGHPUT_ITERS must be > 0"
        );
        assert!(streams > 0, "{prefix}_STREAMS must be > 0");

        Self {
            latency_iters,
            inverse_latency_iters,
            throughput_iters,
            inverse_throughput_iters,
            streams,
        }
    }
}

pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}
