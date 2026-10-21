use super::WhiteoutError;

/// Runs `op` with DIT set on its thread and on every rayon worker it uses
/// (A11): the prover's secret-dependent arithmetic is constant time only
/// under DIT on cores that time integer operations by their operands.
pub fn with_dit<T: Send>(op: impl FnOnce() -> T + Send) -> Result<T, WhiteoutError> {
    jolt_dit::install(op).map_err(|_| WhiteoutError::DitPool)
}
