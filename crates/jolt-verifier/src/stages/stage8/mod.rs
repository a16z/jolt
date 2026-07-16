//! Stage 8: the final PCS opening. [`verify`] is the per-build entry point;
//! the feature-specific statement assembly lives beside it.

/// Homomorphic-build statement assembly: batch entries with unified-point
/// embedding scales.
#[cfg(not(feature = "akita"))]
mod homomorphic;
pub mod outputs;
/// Packed-build statement assembly: per-object packings, leaf-claim
/// resolution, and the joint opening call.
#[cfg(feature = "akita")]
mod packed;
#[cfg(not(feature = "akita"))]
mod precommitted;
/// The reconstruction phase that opens the stage-8 region on the packed path:
/// settles every virtualized word/chunk claim against its committed one-hot
/// decomposition, producing the packed leaf claims the opening consumes.
/// Public because its output-claims aggregate is part of the proof's clear
/// claims.
#[cfg(feature = "akita")]
pub mod reconstruction;
mod verify;

pub use outputs::{Stage8Output, Stage8ZkOutput};
pub use verify::verify;

/// Metadata Stage 8 must enforce before dispatching a native Wjolt opening.
#[cfg(feature = "akita")]
pub trait WJoltCommitmentMetadata {
    fn is_one_hot_backend(&self) -> bool;
    fn layout_digest(&self) -> [u8; 32];
    fn num_vars(&self) -> usize;
    fn poly_count(&self) -> usize;
    fn one_hot_k(&self) -> usize;
}

#[cfg(feature = "akita")]
impl WJoltCommitmentMetadata for jolt_akita::AkitaCommitment {
    fn is_one_hot_backend(&self) -> bool {
        self.backend_flavor() == jolt_akita::AkitaBackendFlavor::OneHot
    }

    fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest()
    }

    fn num_vars(&self) -> usize {
        self.num_vars()
    }

    fn poly_count(&self) -> usize {
        self.poly_count()
    }

    fn one_hot_k(&self) -> usize {
        self.one_hot_k()
    }
}

/// Shape metadata for the verifier-owned Wjolt setup.
#[cfg(feature = "akita")]
pub trait WJoltSetupMetadata {
    fn max_num_vars(&self) -> usize;
    fn max_num_polys_per_commitment_group(&self) -> usize;
    fn default_layout_digest(&self) -> [u8; 32];
    fn one_hot_k(&self) -> usize;
}

#[cfg(feature = "akita")]
impl WJoltSetupMetadata for jolt_akita::AkitaVerifierSetup {
    fn max_num_vars(&self) -> usize {
        self.max_num_vars()
    }

    fn max_num_polys_per_commitment_group(&self) -> usize {
        self.max_num_polys_per_commitment_group()
    }

    fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest()
    }

    fn one_hot_k(&self) -> usize {
        self.one_hot_k()
    }
}
