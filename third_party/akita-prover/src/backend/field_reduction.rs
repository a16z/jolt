//! Tensor extension-opening packing helpers.

use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, CanonicalField, FromPrimitiveInt};
use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::pack_tensor_base_lift_i8_digits;
use std::sync::Arc;

use crate::compute::CommitmentComputeBackend;
use crate::{AkitaPolyOps, DensePoly, RecursiveWitnessFlat, SparseRingPoly};

/// Fold-facing polynomial wrapper for original roots and tensor-projected roots.
///
/// Non-EOR paths borrow the caller's original polynomial. EOR paths own the
/// materialized tensor projection, preserving dense and sparse projected storage.
#[derive(Debug, Clone)]
pub enum FoldInputPoly<'a, F: FieldCore, P, const D: usize> {
    /// Original, non-projected polynomial.
    Original(&'a P),
    /// Dense tensor-projected root polynomial.
    ProjectedDense(DensePoly<F, D>),
    /// Sparse signed-ring tensor-projected root polynomial.
    ProjectedSparse(Arc<SparseRingPoly<F, D>>),
}

impl<'a, F: FieldCore, P, const D: usize> FoldInputPoly<'a, F, P, D> {
    pub fn projected_dense(poly: DensePoly<F, D>) -> Self {
        Self::ProjectedDense(poly)
    }

    pub fn projected_sparse(poly: SparseRingPoly<F, D>) -> Self {
        Self::ProjectedSparse(Arc::new(poly))
    }
}

macro_rules! dispatch_fold_input {
    ($self:expr, $poly:ident => $body:expr) => {
        match $self {
            FoldInputPoly::Original($poly) => $body,
            FoldInputPoly::ProjectedDense($poly) => $body,
            FoldInputPoly::ProjectedSparse($poly) => $body,
        }
    };
}

impl<F, P, const D: usize> AkitaPolyOps<F, D> for FoldInputPoly<'_, F, P, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    P: AkitaPolyOps<F, D>,
{
    fn num_ring_elems(&self) -> usize {
        dispatch_fold_input!(self, poly => poly.num_ring_elems())
    }

    fn num_vars(&self) -> usize {
        dispatch_fold_input!(self, poly => poly.num_vars())
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Original(poly) => poly.onehot_chunk_size(),
            Self::ProjectedDense(_) | Self::ProjectedSparse(_) => None,
        }
    }

    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        dispatch_fold_input!(self, poly => poly.base_evals())
    }

    fn fold_blocks(
        &self,
        scalars: &[F],
        block_len: usize,
    ) -> Vec<akita_algebra::CyclotomicRing<F, D>> {
        dispatch_fold_input!(self, poly => poly.fold_blocks(scalars, block_len))
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[akita_algebra::CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<akita_algebra::CyclotomicRing<F, D>> {
        dispatch_fold_input!(self, poly => poly.fold_blocks_ring(scalars, block_len))
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (
        akita_algebra::CyclotomicRing<F, D>,
        Vec<akita_algebra::CyclotomicRing<F, D>>,
    ) {
        dispatch_fold_input!(self, poly => {
            poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
        })
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        fold_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (
        akita_algebra::CyclotomicRing<F, D>,
        Vec<akita_algebra::CyclotomicRing<F, D>>,
    ) {
        dispatch_fold_input!(self, poly => {
            poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
        })
    }

    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::MulBaseUnreduced<F>,
    {
        dispatch_fold_input!(self, poly => poly.tensor_extension_column_partials(logical_point))
    }

    fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        dispatch_fold_input!(self, poly => poly.tensor_packed_extension_evals::<E>())
    }

    fn tensor_packed_extension_poly<E>(&self) -> Result<DensePoly<F, D>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: akita_types::FpExtEncoding<F>,
    {
        dispatch_fold_input!(self, poly => poly.tensor_packed_extension_poly::<E>())
    }

    fn decompose_fold(
        &self,
        challenges: &[akita_challenges::SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> crate::DecomposeFoldWitness<F, D> {
        dispatch_fold_input!(self, poly => {
            poly.decompose_fold(challenges, block_len, num_digits, log_basis)
        })
    }

    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[akita_challenges::SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<crate::DecomposeFoldWitness<F, D>> {
        let first = *polys.first()?;
        match first {
            Self::Original(_) => {
                let mut originals = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::Original(inner) => originals.push(*inner),
                        Self::ProjectedDense(_) | Self::ProjectedSparse(_) => return None,
                    }
                }
                P::decompose_fold_batched(&originals, challenges, block_len, num_digits, log_basis)
            }
            Self::ProjectedDense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedDense(inner) => dense_polys.push(inner),
                        Self::Original(_) | Self::ProjectedSparse(_) => return None,
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_batched(
                    &dense_polys,
                    challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            Self::ProjectedSparse(_) => {
                let mut sparse_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedSparse(inner) => sparse_polys.push(inner.as_ref()),
                        Self::Original(_) | Self::ProjectedDense(_) => return None,
                    }
                }
                <SparseRingPoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_batched(
                    &sparse_polys,
                    challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
        }
    }

    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &akita_challenges::TensorChallenges,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<crate::DecomposeFoldWitness<F, D>>, AkitaError> {
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        match *first {
            Self::Original(_) => {
                let mut originals = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::Original(inner) => originals.push(*inner),
                        Self::ProjectedDense(_) | Self::ProjectedSparse(_) => return Ok(None),
                    }
                }
                P::decompose_fold_tensor_batched(
                    &originals, tensor, block_len, num_digits, log_basis,
                )
            }
            Self::ProjectedDense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedDense(inner) => dense_polys.push(inner),
                        Self::Original(_) | Self::ProjectedSparse(_) => return Ok(None),
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &dense_polys,
                    tensor,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            Self::ProjectedSparse(_) => {
                let mut sparse_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedSparse(inner) => sparse_polys.push(inner.as_ref()),
                        Self::Original(_) | Self::ProjectedDense(_) => return Ok(None),
                    }
                }
                <SparseRingPoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &sparse_polys,
                    tensor,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
        }
    }

    fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<crate::CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
        B: CommitmentComputeBackend<F>,
    {
        dispatch_fold_input!(self, poly => {
            poly.commit_inner(
                backend,
                prepared,
                n_a,
                block_len,
                num_blocks,
                num_digits_commit,
                num_digits_open,
                log_basis,
            )
        })
    }

    fn direct_root_witness(&self) -> Result<akita_types::CleartextWitnessProof<F>, AkitaError> {
        dispatch_fold_input!(self, poly => poly.direct_root_witness())
    }
}

fn tensor_extension_split<F, E>(context: &'static str) -> Result<(usize, usize), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let split_bits = E::EXT_DEGREE.trailing_zeros() as usize;
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("tensor extension width overflow".to_string()))?;
    if width != E::EXT_DEGREE || !E::EXT_DEGREE.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tensor extension {context} requires power-of-two extension degree"
        )));
    }
    Ok((split_bits, width))
}

/// Pack a logical recursive digit witness into the canonical tensor extension
/// ring-subfield layout.
///
/// For degree-one fields this is the identity. For small fields this stores
/// the extension-valued tensor table in the same ring-subfield layout used by
/// folded extension openings.
///
/// # Errors
///
/// Returns an error if the logical witness length is not compatible with the
/// full tensor split or if ring-subfield packing fails.
pub fn tensor_pack_recursive_witness<F, E, const D: usize>(
    logical_w: &RecursiveWitnessFlat,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (_split_bits, width) = tensor_extension_split::<F, E>("packing")?;
    let packed =
        pack_tensor_base_lift_i8_digits::<D>(logical_w.as_i8_digits(), E::EXT_DEGREE, width)?;
    Ok(RecursiveWitnessFlat::from_i8_digits(packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{AkitaError, FpExt4, Prime32Offset99};

    #[test]
    fn recursive_tensor_pack_rejects_non_divisible_digit_count() {
        type F = Prime32Offset99;
        type E = FpExt4<F>;
        const D: usize = 32;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1, 2, 3]);

        let err = tensor_pack_recursive_witness::<F, E, D>(&witness).unwrap_err();
        assert!(matches!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 3
            }
        ));
    }
}
