use std::time::Duration;

use rand::RngCore;

pub(crate) fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    lo | (hi << 64)
}

/// Per-logical-op time returned from `iter_custom`.
///
/// Criterion divides the returned duration by its batch `iters` again; only count logical ops
/// inside one batch (e.g. `latency_iters` or `latency_iters * WIDTH`).
pub(crate) fn duration_per_logical_op(elapsed: Duration, logical_ops_per_batch: u64) -> Duration {
    Duration::from_secs_f64(elapsed.as_secs_f64() / logical_ops_per_batch.max(1) as f64)
}
