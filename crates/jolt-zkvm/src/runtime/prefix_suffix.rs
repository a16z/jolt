//! Prefix-suffix decomposition types for instruction lookup sumchecks.
//!
//! All evaluation logic is now data-driven via `InstanceConfig` fields
//! (prefix_mle_rules, checkpoint_rules, combine_entries, suffix_ops,
//! suffix_at_empty). No protocol-specific evaluator trait needed.

use jolt_field::Field;

pub use jolt_compute::LookupTraceData;

/// Per-phase buffer data produced by the suffix scatter handler.
pub struct PhaseBuffers<F: Field> {
    pub suffix_polys: Vec<Vec<Vec<F>>>,
    pub q_left: [Vec<F>; 2],
    pub q_right: [Vec<F>; 2],
    pub q_identity: [Vec<F>; 2],
    pub p_left: [Option<Vec<F>>; 2],
    pub p_right: [Option<Vec<F>>; 2],
    pub p_identity: [Option<Vec<F>>; 2],
}
