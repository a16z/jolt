/// Test-only helpers for this module that need access to private invariants
/// (`FlatBlocks`' monotonic `offsets` / contiguous `entries`, and the
/// non-wide reference path for `inner_ajtai_wide_onehot`).
///
/// Gated on `#[cfg(test)]` so the production binary never sees them.
#[cfg(test)]
use super::{CyclotomicRing, FlatBlocks, MultiChunkEntry, OneHotEntry, OneHotIndex, OneHotPoly};
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};

/// Reference ring-space evaluation for [`OneHotPoly`].
///
/// Computes the global weighted sum `y = Σᵢ scalars[i] · self[i]`.
/// `scalars` has length >= `num_ring_elems`; excess entries are ignored.
///
/// Only used by tests to cross-check fused prover paths
/// (e.g. `evaluate_and_fold`) against a straight-line implementation,
/// so it lives in `test_helpers` rather than on the production trait.
pub(crate) fn evaluate_ring_onehot<F, const D: usize, I>(
    poly: &OneHotPoly<F, D, I>,
    scalars: &[F],
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField,
    I: OneHotIndex,
{
    let onehot_k = poly.onehot_k;
    cfg_fold_reduce!(
        0..poly.indices.len(),
        || CyclotomicRing::<F, D>::zero(),
        |mut acc: CyclotomicRing<F, D>, chunk_idx: usize| {
            if let Some(raw) = poly.indices[chunk_idx] {
                let field_pos = chunk_idx * onehot_k + raw.as_usize();
                let ring_idx = field_pos / D;
                let coeff_idx = field_pos % D;
                if ring_idx < scalars.len() {
                    acc.coeffs[coeff_idx] += scalars[ring_idx];
                }
            }
            acc
        },
        |a, b| a + b
    )
}

/// Build a flat block layout from a pre-bucketed `Vec<Vec<E>>`.
///
/// The production paths (`FlatBlocks::<SingleChunkEntry>::from_indices`,
/// `FlatBlocks::<MultiChunkEntry>::from_indices`) stream entries directly
/// into the flat form without ever materialising per-block `Vec`s.
/// This constructor exists only so tests that hand-assemble
/// block-bucketed storage can still feed it into kernels that
/// consume `FlatBlocks`.
pub(crate) fn from_buckets<E>(buckets: Vec<Vec<E>>) -> FlatBlocks<E> {
    let num_blocks = buckets.len();
    let mut offsets = Vec::with_capacity(num_blocks + 1);
    let total: usize = buckets.iter().map(Vec::len).sum();
    let mut entries = Vec::with_capacity(total);
    offsets.push(0);
    for mut bucket in buckets {
        entries.append(&mut bucket);
        // `entries.len()` is bounded by `total = sum(Vec::len)` which
        // was accepted as `usize`; it is always safe to downcast to
        // `u32` on all supported layouts used by tests.
        offsets.push(u32::try_from(entries.len()).expect("flat block offset overflows u32"));
    }
    FlatBlocks { entries, offsets }
}

/// Reference (non-wide) multi-chunk inner Ajtai used to cross-check
/// [`super::inner_ajtai_wide_onehot`].
///
/// Production code always uses the wide accumulator; this simpler
/// variant only exists so tests can assert the two paths agree.
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_multi_chunk_t_only<F: FieldCore + CanonicalField, const D: usize>(
    A: &[Vec<CyclotomicRing<F, D>>],
    multi_chunk_entries: &[MultiChunkEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = A.len();
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];
    for entry in multi_chunk_entries {
        let col = entry.commit_col(num_digits);
        for a in 0..n_a {
            for &ci in entry.coeffs() {
                A[a][col].shift_accumulate_into(&mut t[a], ci as usize);
            }
        }
    }
    t
}
