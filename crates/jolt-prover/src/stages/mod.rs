//! Per-stage prover recipes, mirroring `jolt-verifier`'s `stages/` layout.

use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_verifier::{CheckedInputs, VerifierError};

use crate::ProverConfig;

mod drivers;
pub mod stage0;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6a;
pub mod stage6b;
pub mod stage7;
pub mod stage8;

/// The one-hot formula dimensions, built exactly as the verifier's
/// `build_formula_dimensions` builds them (which reads the one-hot config off
/// the proof; the prover reads it off its own derived config — stage 0 wrote
/// that same value to the wire). `stage` attributes a geometry failure to the
/// consuming relation.
pub(crate) fn formula_dimensions(
    checked: &CheckedInputs,
    config: &ProverConfig,
    bytecode_len: usize,
    stage: JoltRelationId,
) -> Result<JoltFormulaDimensions, VerifierError> {
    let log_t = checked.trace_length.ilog2() as usize;
    JoltFormulaDimensions::try_from(config.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        bytecode_len,
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: error.to_string(),
    })
}
