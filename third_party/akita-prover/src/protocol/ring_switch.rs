//! Prover-owned helpers for the Akita ring-switch handoff.
use crate::api::commitment::{
    validate_commit_inner_shape, validate_commit_level_params, validate_commit_outer_input_nonempty,
};
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::protocol::ring_relation::compute_relation_quotient;
use crate::protocol::RingRelationWitness;
use crate::{
    tensor_pack_recursive_witness, CommitmentComputeBackend, RecursiveCommitmentHintCache,
    RecursiveWitnessFlat, RingSwitchComputeBackend,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::ring::eval_ring_at_pows;
use akita_algebra::ring::scalar_powers;
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_config::CommitmentConfig;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField, LiftBase,
    MulBase, RandomSampling,
};
use akita_transcript::labels::{CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::dispatch_ring_dim_result;
use akita_types::RingRelationInstance;
use akita_types::{
    embed_ring_subfield_scalar, gadget_row_scalars, r_decomp_levels, AkitaCommitmentHint,
    AkitaExpandedSetup, FlatDigitBlocks, FlatRingVec, FpExtEncoding, LevelParams, MRowLayout,
    RingCommitment, RingMultiplierOpeningPoint, RingOpeningPoint,
};

mod coeffs;
mod commit;
mod evals;
mod finalize;
#[cfg(test)]
mod tests;

#[cfg(not(feature = "zk"))]
pub use coeffs::RingSwitchTerminalArtifacts;
pub use coeffs::{build_w_coeffs, ring_switch_build_w, RingSwitchBuildOutput};
pub use commit::{commit_next_w, commit_w, NextWitnessCommitment};
pub use evals::{build_w_evals_compact, compute_m_evals_x};
pub use finalize::ring_switch_finalize;

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<E: FieldCore> {
    /// Compact evaluation table of w, stored as x-outer/y-inner slices.
    pub w_evals_compact: Vec<i8>,
    /// Physical x width before zero-extension to the next power of two.
    pub live_x_cols: usize,
    /// Evaluation table of M_alpha(x) (tau1-weighted).
    pub m_evals_x: Vec<E>,
    /// Evaluation table of alpha powers (y dimension).
    pub alpha_evals_y: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<E>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}
