//! The canonical `OneHotTrace` commitment-group shape, derived from the
//! proof config and the program shape alone — what a caller needs to build
//! the packed scheme's setup params without instantiating any prover (the
//! params constructor itself is scheme-specific, so this crate exposes only
//! the shape).

use jolt_claims::protocols::jolt::lattice::{
    OneHotTraceSetupShape, OneHotTraceShape, ONE_HOT_TRACE_LAYOUT,
};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_verifier::stages::formula_dimensions_from_parts;
use jolt_verifier::VerifierError;

use crate::ProverConfig;

/// The `OneHotTrace` group's setup dimensions, canonical layout digest, and
/// one-hot chunk size for one proof shape: `config` carries the trace length
/// and one-hot config, `bytecode_len` the (padded) program's bytecode size.
/// Stage 0 validates the supplied setup against these same values
/// fail-closed, so a setup built from this shape always passes.
pub fn one_hot_trace_setup_shape(
    config: &ProverConfig,
    bytecode_len: usize,
) -> Result<(OneHotTraceSetupShape, [u8; 32], usize), VerifierError> {
    if !config.trace_length.is_power_of_two() {
        return Err(VerifierError::InvalidTraceLength {
            got: config.trace_length,
            max: usize::MAX,
        });
    }
    if !config.ram_K.is_power_of_two() {
        return Err(VerifierError::InvalidRamK {
            got: config.ram_K,
            min: 1,
            max: usize::MAX,
        });
    }
    let log_t = config.trace_length.ilog2() as usize;
    let log_k_chunk = config.one_hot_config.committed_chunk_bits();
    let formula_dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        bytecode_len,
        config.ram_K,
        JoltRelationId::HammingWeightClaimReduction,
    )?;
    let shape = OneHotTraceShape {
        ra_layout: formula_dimensions.ra_layout,
        log_t,
        log_k_chunk,
    };
    let batch_failed =
        |error: jolt_openings::OpeningsError| VerifierError::FinalOpeningBatchFailed {
            reason: error.to_string(),
        };
    let setup_shape = ONE_HOT_TRACE_LAYOUT
        .setup_shape(&shape)
        .map_err(batch_failed)?;
    let digest = ONE_HOT_TRACE_LAYOUT
        .layout_digest(&shape)
        .map_err(batch_failed)?;
    Ok((setup_shape, digest, 1usize << log_k_chunk))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert metadata admission")]
mod tests {
    use super::*;
    use crate::config::{one_hot_config, read_write_config};
    use jolt_claims::protocols::jolt::TracePolynomialOrder;

    fn config(log_t: usize) -> ProverConfig {
        ProverConfig {
            trace_length: 1usize << log_t,
            ram_K: 1 << 24,
            rw_config: read_write_config(log_t, 24),
            one_hot_config: one_hot_config(log_t),
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
        }
    }

    #[test]
    fn maximum_trace_uses_the_44_variable_packed_shape() {
        let (shape, _, one_hot_k) = one_hot_trace_setup_shape(&config(31), 1 << 24).unwrap();
        assert_eq!(shape.num_vars, 44);
        assert_eq!(shape.num_polys, 1);
        assert_eq!(one_hot_k, 256);
    }

    #[test]
    fn malformed_domains_fail_before_logarithms() {
        let mut shape = config(12);
        shape.trace_length = 0;
        assert!(matches!(
            one_hot_trace_setup_shape(&shape, 16),
            Err(VerifierError::InvalidTraceLength { .. })
        ));
        shape.trace_length = 1 << 12;
        shape.ram_K = 0;
        assert!(matches!(
            one_hot_trace_setup_shape(&shape, 16),
            Err(VerifierError::InvalidRamK { .. })
        ));
    }
}
