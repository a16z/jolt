//! Prover-only secret witness for the negacyclic-ring relation.

use crate::DecomposeFoldWitness;
use akita_field::FieldCore;
use akita_types::AkitaCommitmentHint;
use akita_types::FlatDigitBlocks;

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub struct RingRelationWitness<F: FieldCore, const D: usize> {
    pub z_folded_rings: DecomposeFoldWitness<F, D>,
    pub fold_grind_nonce: u32,
    pub e_hat: FlatDigitBlocks<D>,
    pub e_folded: Vec<akita_algebra::CyclotomicRing<F, D>>,
    pub hint: AkitaCommitmentHint<F, D>,
    #[cfg(feature = "zk")]
    pub d_blinding_digits: FlatDigitBlocks<D>,
}
