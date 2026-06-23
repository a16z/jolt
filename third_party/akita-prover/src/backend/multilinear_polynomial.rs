//! Canonical multilinear-polynomial wrapper for prover polynomial representations.
//!
//! This is the intended public wrapper for heterogeneous root batches. All
//! wrapped polynomials must still share the same commitment config and root
//! layout chosen by the caller, but one batch can contain dense, one-hot, and
//! recursive witness views.
//!
//! Homogeneous batches still reuse the existing backend-specific batched fast
//! paths; truly mixed batches fall back to the caller's per-polynomial
//! aggregation path.

use crate::compute::CommitmentComputeBackend;
use crate::{
    AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness, DensePoly, OneHotIndex, OneHotPoly,
    SuffixWitness,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges};
use akita_field::unreduced::HasWide;
use akita_field::{AkitaError, CanonicalField, FieldCore};

/// Borrowed multilinear-polynomial wrapper for prover polynomial batches.
///
/// This erases the polynomial representation (`DensePoly`, `OneHotPoly`, or
/// `SuffixWitness`) while preserving the operation-oriented `AkitaPolyOps`
/// interface that the commitment scheme consumes.
#[derive(Debug, Clone, Copy)]
pub enum MultilinearPolynomial<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    /// Dense multilinear polynomial.
    Dense(&'a DensePoly<F, D>),
    /// One-hot multilinear polynomial.
    OneHot(&'a OneHotPoly<F, D, I>),
    /// Recursive witness view.
    Witness(SuffixWitness<'a, F, D>),
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> MultilinearPolynomial<'a, F, D, I> {
    /// Wrap a dense polynomial.
    pub fn dense(poly: &'a DensePoly<F, D>) -> Self {
        Self::Dense(poly)
    }

    /// Wrap a one-hot polynomial.
    pub fn onehot(poly: &'a OneHotPoly<F, D, I>) -> Self {
        Self::OneHot(poly)
    }

    /// Wrap a recursive witness view.
    pub fn recursive(poly: SuffixWitness<'a, F, D>) -> Self {
        Self::Witness(poly)
    }
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> From<&'a DensePoly<F, D>>
    for MultilinearPolynomial<'a, F, D, I>
{
    fn from(poly: &'a DensePoly<F, D>) -> Self {
        Self::dense(poly)
    }
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> From<&'a OneHotPoly<F, D, I>>
    for MultilinearPolynomial<'a, F, D, I>
{
    fn from(poly: &'a OneHotPoly<F, D, I>) -> Self {
        Self::onehot(poly)
    }
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> From<SuffixWitness<'a, F, D>>
    for MultilinearPolynomial<'a, F, D, I>
{
    fn from(poly: SuffixWitness<'a, F, D>) -> Self {
        Self::recursive(poly)
    }
}

impl<F, const D: usize, I> AkitaPolyOps<F, D> for MultilinearPolynomial<'_, F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(poly) => poly.num_ring_elems(),
            Self::OneHot(poly) => poly.num_ring_elems(),
            Self::Witness(poly) => poly.num_ring_elems(),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => poly.num_vars(),
            Self::OneHot(poly) => poly.num_vars(),
            Self::Witness(poly) => poly.num_vars(),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(poly) => poly.onehot_chunk_size(),
            Self::OneHot(poly) => poly.onehot_chunk_size(),
            Self::Witness(poly) => poly.onehot_chunk_size(),
        }
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        match self {
            Self::Dense(poly) => poly.fold_blocks(scalars, block_len),
            Self::OneHot(poly) => poly.fold_blocks(scalars, block_len),
            Self::Witness(poly) => poly.fold_blocks(scalars, block_len),
        }
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        match self {
            Self::Dense(poly) => poly.fold_blocks_ring(scalars, block_len),
            Self::OneHot(poly) => poly.fold_blocks_ring(scalars, block_len),
            Self::Witness(poly) => poly.fold_blocks_ring(scalars, block_len),
        }
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        match self {
            Self::Dense(poly) => {
                poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
            }
            Self::OneHot(poly) => {
                poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
            }
            Self::Witness(poly) => {
                poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
            }
        }
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        match self {
            Self::Dense(poly) => {
                poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
            }
            Self::OneHot(poly) => {
                poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
            }
            Self::Witness(poly) => {
                poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
            }
        }
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        match self {
            Self::Dense(poly) => poly.decompose_fold(challenges, block_len, num_digits, log_basis),
            Self::OneHot(poly) => poly.decompose_fold(challenges, block_len, num_digits, log_basis),
            Self::Witness(poly) => {
                poly.decompose_fold(challenges, block_len, num_digits, log_basis)
            }
        }
    }

    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        let first = polys.first()?;
        match **first {
            Self::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::Dense(inner) => dense_polys.push(inner),
                        Self::OneHot(_) => return None,
                        Self::Witness(_) => return None,
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
            Self::OneHot(_) => {
                let mut onehot_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::OneHot(inner) => onehot_polys.push(inner),
                        Self::Dense(_) => return None,
                        Self::Witness(_) => return None,
                    }
                }
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::decompose_fold_batched(
                    &onehot_polys,
                    challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            Self::Witness(_) => None,
        }
    }

    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &TensorChallenges,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        match **first {
            Self::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::Dense(inner) => dense_polys.push(inner),
                        Self::OneHot(_) => return Ok(None),
                        Self::Witness(_) => return Ok(None),
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
            Self::OneHot(_) => {
                let mut onehot_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::OneHot(inner) => onehot_polys.push(inner),
                        Self::Dense(_) => return Ok(None),
                        Self::Witness(_) => return Ok(None),
                    }
                }
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &onehot_polys,
                    tensor,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            Self::Witness(_) => Ok(None),
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
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
        B: CommitmentComputeBackend<F>,
    {
        match self {
            Self::Dense(poly) => poly.commit_inner(
                backend,
                prepared,
                n_a,
                block_len,
                num_blocks,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
            Self::OneHot(poly) => poly.commit_inner(
                backend,
                prepared,
                n_a,
                block_len,
                num_blocks,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
            Self::Witness(poly) => poly.commit_inner(
                backend,
                prepared,
                n_a,
                block_len,
                num_blocks,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
        }
    }

    fn direct_root_witness(&self) -> Result<akita_types::CleartextWitnessProof<F>, AkitaError> {
        match self {
            Self::Dense(poly) => poly.direct_root_witness(),
            Self::OneHot(poly) => poly.direct_root_witness(),
            Self::Witness(poly) => poly.direct_root_witness(),
        }
    }
}
