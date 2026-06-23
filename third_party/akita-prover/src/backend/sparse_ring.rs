//! Sparse signed ring-coefficient polynomial backend.
//!
//! This is the natural backend for Frobenius-packed one-hot tables: after
//! canonical-basis packing, each original one-hot chunk becomes a small number
//! of signed monomial coefficients inside the committed ring table.

use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    MulBaseUnreduced,
};
use akita_types::{
    CleartextWitnessProof, FlatDigitBlocks, FlatRingVec, FpExtEncoding,
};
use std::sync::{Arc, OnceLock};

use crate::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use crate::compute::{CommitmentComputeBackend, FlatBlockTable, SparseRingCommitRowsPlan};
use crate::kernels::linear::decompose_rows_i8_into;
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness, FoldInputPoly};

mod tensor_fold;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseRingCoeff {
    ring_idx: u32,
    coeff_idx: u16,
    value: i8,
}

impl SparseRingCoeff {
    pub(crate) fn new(ring_idx: usize, coeff_idx: usize, value: i8) -> Result<Self, AkitaError> {
        if !matches!(value, -1 | 1) {
            return Err(AkitaError::InvalidInput(
                "sparse ring coefficients must be signed units".to_string(),
            ));
        }
        Ok(Self {
            ring_idx: u32::try_from(ring_idx).map_err(|_| {
                AkitaError::InvalidInput("sparse ring index exceeds u32".to_string())
            })?,
            coeff_idx: u16::try_from(coeff_idx).map_err(|_| {
                AkitaError::InvalidInput("sparse coefficient index exceeds u16".to_string())
            })?,
            value,
        })
    }

    #[inline]
    fn sort_key(self) -> (u32, u16, i8) {
        (self.ring_idx, self.coeff_idx, self.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseRingBlockEntry {
    pos_in_block: u32,
    coeff_idx: u16,
    value: i8,
}

impl SparseRingBlockEntry {
    #[inline]
    pub fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    #[inline]
    pub fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }

    #[inline]
    pub fn value(self) -> i8 {
        self.value
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseRingBlocks {
    entries: Vec<SparseRingBlockEntry>,
    offsets: Vec<u32>,
}

impl SparseRingBlocks {
    fn from_coeffs(
        coeffs: &[SparseRingCoeff],
        total_ring_elems: usize,
        block_len: usize,
    ) -> Result<Self, AkitaError> {
        if block_len == 0 || !block_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} must be a nonzero power of two"
            )));
        }
        if !total_ring_elems.is_multiple_of(block_len) {
            return Err(AkitaError::InvalidSize {
                expected: total_ring_elems,
                actual: block_len,
            });
        }
        if u32::try_from(block_len).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} exceeds u32::MAX"
            )));
        }
        let num_blocks = total_ring_elems / block_len;
        let mut offsets = Vec::with_capacity(num_blocks + 1);
        let mut entries = Vec::with_capacity(coeffs.len());
        offsets.push(0);
        let mut current_block = 0usize;
        for coeff in coeffs {
            let ring_idx = coeff.ring_idx as usize;
            if ring_idx >= total_ring_elems {
                return Err(AkitaError::InvalidInput(
                    "sparse ring coefficient index out of range".to_string(),
                ));
            }
            let block_idx = ring_idx / block_len;
            while current_block < block_idx {
                offsets.push(entries.len() as u32);
                current_block += 1;
            }
            entries.push(SparseRingBlockEntry {
                pos_in_block: (ring_idx % block_len) as u32,
                coeff_idx: coeff.coeff_idx,
                value: coeff.value,
            });
        }
        while current_block < num_blocks {
            offsets.push(entries.len() as u32);
            current_block += 1;
        }
        Ok(Self { entries, offsets })
    }

    #[inline]
    pub(crate) fn num_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub(crate) fn block(&self, idx: usize) -> &[SparseRingBlockEntry] {
        let lo = self.offsets[idx] as usize;
        let hi = self.offsets[idx + 1] as usize;
        &self.entries[lo..hi]
    }

    #[inline]
    fn table(&self) -> FlatBlockTable<'_, SparseRingBlockEntry> {
        FlatBlockTable::new(&self.entries, &self.offsets)
    }
}

/// Sparse polynomial whose ring coefficients are signed monomials.
#[derive(Debug, Clone)]
pub struct SparseRingPoly<F: FieldCore, const D: usize> {
    num_vars: usize,
    total_ring_elems: usize,
    coeffs: Vec<SparseRingCoeff>,
    block_cache: OnceLock<(usize, SparseRingBlocks)>,
    tensor_root_cache: OnceLock<(usize, Arc<SparseRingPoly<F, D>>)>,
    _marker: core::marker::PhantomData<F>,
}

impl<F: FieldCore, const D: usize> SparseRingPoly<F, D> {
    /// Build from `(ring_idx, coeff_idx, value)` triples.
    ///
    /// # Errors
    ///
    /// Returns an error when `D` cannot be represented by the sparse block
    /// format, the expected ring-element count does not match `num_vars`, or a
    /// supplied coefficient triple is out of range or has value other than
    /// `-1` or `1`.
    pub fn from_signed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
    ) -> Result<Self, AkitaError> {
        Self::from_signed_coeffs_with_order(num_vars, total_ring_elems, coeffs, false)
    }

    /// Build from `(ring_idx, coeff_idx, value)` triples already sorted by
    /// `(ring_idx, coeff_idx, value)`.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_signed_coeffs`], and also when the supplied triples are not
    /// sorted.
    pub fn from_sorted_signed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
    ) -> Result<Self, AkitaError> {
        Self::from_signed_coeffs_with_order(num_vars, total_ring_elems, coeffs, true)
    }

    /// Build from compact sparse coefficient triples.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_signed_coeffs`].
    pub(crate) fn from_packed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<SparseRingCoeff>,
    ) -> Result<Self, AkitaError> {
        Self::from_packed_coeffs_with_order(num_vars, total_ring_elems, coeffs, false)
    }

    /// Build from compact sparse coefficient triples already sorted by
    /// `(ring_idx, coeff_idx, value)`.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_sorted_signed_coeffs`].
    pub(crate) fn from_sorted_packed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<SparseRingCoeff>,
    ) -> Result<Self, AkitaError> {
        Self::from_packed_coeffs_with_order(num_vars, total_ring_elems, coeffs, true)
    }

    fn from_signed_coeffs_with_order(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
        already_sorted: bool,
    ) -> Result<Self, AkitaError> {
        let mut packed = Vec::with_capacity(coeffs.len());
        for (ring_idx, coeff_idx, value) in coeffs {
            packed.push(SparseRingCoeff::new(ring_idx, coeff_idx, value)?);
        }
        Self::from_packed_coeffs_with_order(num_vars, total_ring_elems, packed, already_sorted)
    }

    fn from_packed_coeffs_with_order(
        num_vars: usize,
        total_ring_elems: usize,
        mut packed: Vec<SparseRingCoeff>,
        already_sorted: bool,
    ) -> Result<Self, AkitaError> {
        if D > usize::from(u16::MAX) + 1 {
            return Err(AkitaError::InvalidInput(format!(
                "D={D} exceeds sparse coefficient index capacity"
            )));
        }
        let expected_ring_elems = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput("sparse arity overflow".to_string()))?
            .checked_div(D)
            .ok_or_else(|| AkitaError::InvalidInput("D must be nonzero".to_string()))?;
        if expected_ring_elems != total_ring_elems {
            return Err(AkitaError::InvalidSize {
                expected: expected_ring_elems,
                actual: total_ring_elems,
            });
        }
        let mut previous_key = None;
        for entry in &packed {
            if entry.ring_idx as usize >= total_ring_elems
                || entry.coeff_idx as usize >= D
                || !matches!(entry.value, -1 | 1)
            {
                return Err(AkitaError::InvalidInput(
                    "invalid sparse ring coefficient".to_string(),
                ));
            }
            let key = entry.sort_key();
            if already_sorted && previous_key.is_some_and(|previous| key < previous) {
                return Err(AkitaError::InvalidInput(
                    "sorted sparse ring constructor received unsorted coefficients".to_string(),
                ));
            }
            previous_key = Some(key);
        }
        if !already_sorted {
            packed.sort_unstable_by_key(|entry| entry.sort_key());
        }
        Ok(Self {
            num_vars,
            total_ring_elems,
            coeffs: packed,
            block_cache: OnceLock::new(),
            tensor_root_cache: OnceLock::new(),
            _marker: core::marker::PhantomData,
        })
    }

    fn blocks_for(&self, block_len: usize) -> Result<&SparseRingBlocks, AkitaError> {
        if let Some((cached_len, blocks)) = self.block_cache.get() {
            if *cached_len == block_len {
                return Ok(blocks);
            }
            return Err(AkitaError::InvalidInput(format!(
                "SparseRingPoly was first used with block_len={cached_len} but is now used with block_len={block_len}"
            )));
        }
        let (_, blocks) = self.block_cache.get_or_init(|| {
            let blocks =
                SparseRingBlocks::from_coeffs(&self.coeffs, self.total_ring_elems, block_len)
                    .expect("block_len validation is deterministic");
            (block_len, blocks)
        });
        Ok(blocks)
    }
}

impl<F, const D: usize> AkitaPolyOps<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("SparseRingPoly::fold_blocks: invalid block_len");
        cfg_into_iter!(0..blocks.num_blocks())
            .map(|block_idx| fold_sparse_block(blocks.block(block_idx), scalars, block_len))
            .collect()
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("SparseRingPoly::fold_blocks_ring: invalid block_len");
        cfg_into_iter!(0..blocks.num_blocks())
            .map(|block_idx| fold_sparse_block_ring(blocks.block(block_idx), scalars, block_len))
            .collect()
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks_ring(fold_scalars, block_len);
        let mut eval = CyclotomicRing::<F, D>::zero();
        for (f_i, s_i) in folded.iter().zip(eval_outer_scalars.iter()) {
            f_i.mul_accumulate_sparse_rhs_into(s_i, &mut eval);
        }
        (eval, folded)
    }

    fn evaluate_extension<E>(&self, point: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        if point.len() != self.num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.num_vars,
                actual: point.len(),
            });
        }
        let mut eval = E::zero();
        for entry in &self.coeffs {
            let field_pos = sparse_field_position::<D>(entry)?;
            let weight = eq_weight_at_index(point, field_pos);
            match entry.value {
                1 => eval += weight,
                -1 => eval -= weight,
                _ => unreachable!("sparse Frobenius coefficients are signed units"),
            }
        }
        Ok(eval)
    }

    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        sparse_tensor_extension_column_partials::<F, E, D>(
            self.num_vars,
            &self.coeffs,
            logical_point,
        )
    }

    fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        polys
            .iter()
            .map(|poly| poly.tensor_extension_column_partials(logical_point))
            .collect()
    }

    fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        sparse_tensor_packed_witness::<F, E, D>(self.num_vars, &self.coeffs).map(Some)
    }

    fn tensor_packed_extension_fold_input<E>(
        &self,
    ) -> Result<FoldInputPoly<'_, F, Self, D>, AkitaError>
    where
        Self: Sized,
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        Ok(FoldInputPoly::ProjectedSparse(
            self.tensor_packed_sparse_ring_poly::<E>()?,
        ))
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let blocks = self
            .blocks_for(block_len)
            .expect("SparseRingPoly::decompose_fold: invalid block_len");
        let num_blocks = challenges.len().min(blocks.num_blocks());
        let inner_width = block_len * num_digits;
        let coeff_accum =
            sparse_accumulate::<D>(blocks, challenges, num_blocks, inner_width, num_digits);
        let modulus = (-F::one()).to_canonical_u128() + 1;
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::decompose_fold_tensor_batched")]
    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        Ok(Some(tensor_fold::decompose_fold_batched_tensor_sparse(
            polys, tensor, block_len, num_digits,
        )?))
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::commit_inner")]
    fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        _num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let t = self.commit_inner_rows(backend, prepared, n_a, block_len, num_digits_commit)?;
        let decomposed_inner_rows =
            decompose_commit_rows::<F, D>(&t, n_a, num_digits_open, log_basis)?;
        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows,
        })
    }

    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        let total_coeffs = self.total_ring_elems.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidInput("sparse direct witness length overflow".to_string())
        })?;
        let mut coeffs = vec![F::zero(); total_coeffs];
        for entry in &self.coeffs {
            let idx = (entry.ring_idx as usize)
                .checked_mul(D)
                .and_then(|base| base.checked_add(entry.coeff_idx as usize))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("sparse direct witness index overflow".to_string())
                })?;
            coeffs[idx] += F::from_i8(entry.value);
        }
        Ok(CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(coeffs),
        ))
    }
}

impl<F, const D: usize> SparseRingPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn commit_inner_rows<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let blocks = self.blocks_for(block_len)?;
        backend.sparse_ring_commit_rows(
            prepared,
            SparseRingCommitRowsPlan {
                n_a,
                block_len,
                num_digits_commit,
                blocks: blocks.table(),
            },
        )
    }

    fn tensor_packed_sparse_ring_poly<E>(&self) -> Result<Arc<SparseRingPoly<F, D>>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        let (width, total_evals) = tensor_packing_shape::<F, E>(self.num_vars)?;
        if !D.is_multiple_of(width) {
            return Err(AkitaError::InvalidInput(
                "tensor width must divide root ring dimension".to_string(),
            ));
        }
        let double_width = width.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidInput(
                "tensor width is too large for root ring projection".to_string(),
            )
        })?;
        if D < double_width {
            return Err(AkitaError::InvalidInput(
                "root ring dimension must be at least twice the tensor width".to_string(),
            ));
        }
        if let Some((cached_width, poly)) = self.tensor_root_cache.get() {
            if *cached_width == width {
                return Ok(Arc::clone(poly));
            }
        }

        let packed_len = D / width;
        let half = D / double_width;
        let step = D / double_width;
        let total_ring_elems = total_evals / D;
        let mut coeffs = Vec::with_capacity(self.coeffs.len() * width.min(2));

        for entry in &self.coeffs {
            let field_pos = sparse_field_position::<D>(entry)?;
            let tail = field_pos / width;
            let coord = field_pos % width;
            let ring_idx = tail / packed_len;
            let slot_idx = tail % packed_len;
            if slot_idx < half {
                let shift = slot_idx;
                if coord == 0 {
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift, entry.value)?);
                } else {
                    let pos_offset = coord * step;
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift + pos_offset, entry.value)?);
                    coeffs.push(SparseRingCoeff::new(
                        ring_idx,
                        shift + D - pos_offset,
                        -entry.value,
                    )?);
                }
            } else {
                let shift = slot_idx - half + D / 2;
                if coord == 0 {
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift, entry.value)?);
                } else {
                    let pos_offset = coord * step;
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift - pos_offset, entry.value)?);
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift + pos_offset, entry.value)?);
                }
            }
        }

        let poly = SparseRingPoly::<F, D>::from_packed_coeffs(
            self.num_vars,
            total_ring_elems,
            coeffs,
        )?;
        let poly = Arc::new(poly);
        let _ = self.tensor_root_cache.set((width, Arc::clone(&poly)));
        if let Some((cached_width, cached_poly)) = self.tensor_root_cache.get() {
            if *cached_width == width {
                return Ok(Arc::clone(cached_poly));
            }
        }
        Ok(poly)
    }
}

fn sparse_tensor_extension_column_partials<F, E, const D: usize>(
    num_vars: usize,
    coeffs: &[SparseRingCoeff],
    logical_point: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: MulBaseUnreduced<F>,
{
    let (split_bits, width) = tensor_packing_shape::<F, E>(num_vars)?;
    if logical_point.len() != num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: num_vars,
            actual: logical_point.len(),
        });
    }
    let tail_point = &logical_point[split_bits..];
    let chunk_len = {
        #[cfg(feature = "parallel")]
        {
            let target_chunks = rayon::current_num_threads().max(1) * 4;
            (coeffs.len() / target_chunks).max(1 << 12)
        }
        #[cfg(not(feature = "parallel"))]
        {
            coeffs.len().max(1)
        }
    };
    let ranges = (0..coeffs.len())
        .step_by(chunk_len)
        .map(|start| (start, (start + chunk_len).min(coeffs.len())))
        .collect::<Vec<_>>();
    let partial_chunks = cfg_into_iter!(ranges)
        .map(|(start, end)| {
            let mut partials = vec![E::zero(); width];
            for entry in &coeffs[start..end] {
                let field_pos = sparse_field_position::<D>(entry)?;
                let head = field_pos & (width - 1);
                let tail = field_pos >> split_bits;
                let weight = eq_weight_at_index(tail_point, tail);
                match entry.value {
                    1 => partials[head] += weight,
                    -1 => partials[head] -= weight,
                    _ => unreachable!("sparse Frobenius coefficients are signed units"),
                }
            }
            Ok(partials)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let mut out = vec![E::zero(); width];
    for partials in partial_chunks {
        for (dst, value) in out.iter_mut().zip(partials) {
            *dst += value;
        }
    }
    Ok(out)
}

fn sparse_tensor_packed_witness<F, E, const D: usize>(
    num_vars: usize,
    coeffs: &[SparseRingCoeff],
) -> Result<SparseExtensionOpeningWitness<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (width, total_evals) = tensor_packing_shape::<F, E>(num_vars)?;
    let table_len = total_evals / width;
    let basis = signed_extension_basis::<F, E>(width);
    let entries = coeffs
        .iter()
        .map(|entry| {
            let field_pos = sparse_field_position::<D>(entry)?;
            let tail = field_pos / width;
            let head = field_pos % width;
            let value = match entry.value {
                1 => basis[head],
                -1 => E::zero() - basis[head],
                _ => unreachable!("sparse Frobenius coefficients are signed units"),
            };
            Ok((tail, value))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    SparseExtensionOpeningWitness::from_sorted_entries(table_len, entries)
}

fn signed_extension_basis<F, E>(width: usize) -> Vec<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    (0..width)
        .map(|head| {
            let mut coords = vec![F::zero(); width];
            coords[head] = F::one();
            E::from_base_slice(&coords)
        })
        .collect()
}

fn tensor_packing_shape<F, E>(num_vars: usize) -> Result<(usize, usize), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (split_bits, width) = akita_types::tensor_opening_split::<F, E>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidInput(
            "extension-opening tensor split exceeds polynomial arity".to_string(),
        ));
    }
    let total_evals = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize"))
    })?;
    Ok((width, total_evals))
}

#[inline]
fn sparse_field_position<const D: usize>(entry: &SparseRingCoeff) -> Result<usize, AkitaError> {
    (entry.ring_idx as usize)
        .checked_mul(D)
        .and_then(|base| base.checked_add(entry.coeff_idx as usize))
        .ok_or_else(|| AkitaError::InvalidInput("sparse field position overflow".to_string()))
}

fn eq_weight_at_index<E>(point: &[E], index: usize) -> E
where
    E: FieldCore,
{
    let mut weight = E::one();
    for (bit, r) in point.iter().copied().enumerate() {
        if ((index >> bit) & 1) == 0 {
            weight *= E::one() - r;
        } else {
            weight *= r;
        }
    }
    weight
}

fn fold_sparse_block<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    let mut coeffs = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            coeffs[entry.coeff_idx()] += scalars[pos] * F::from_i8(entry.value);
        }
    }
    CyclotomicRing::from_coefficients(coeffs)
}

fn fold_sparse_block_ring<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            match entry.value {
                1 => scalars[pos].shift_accumulate_into(&mut acc, entry.coeff_idx()),
                -1 => scalars[pos].shift_sub_into(&mut acc, entry.coeff_idx()),
                value => scalars[pos].shift_scale_accumulate_into(
                    &mut acc,
                    entry.coeff_idx(),
                    F::from_i8(value),
                ),
            }
        }
    }
    acc
}

fn sparse_accumulate<const D: usize>(
    blocks: &SparseRingBlocks,
    challenges: &[SparseChallenge],
    num_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);
    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let mut acc = vec![[0i32; D]; pos_end - pos_start];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = blocks.block(block_idx);
                let lo = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_start);
                let hi = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_end);
                if lo >= hi {
                    continue;
                }
                fill_rotated_challenge::<D>(&mut rotated, challenge);
                for entry in &entries[lo..hi] {
                    let local_pos = entry.pos_in_block() * num_digits - pos_start;
                    let rot = &rotated[entry.coeff_idx()];
                    let dst = &mut acc[local_pos];
                    let weight = entry.value as i32;
                    for k in 0..D {
                        dst[k] += weight * i32::from(rot[k]);
                    }
                }
            }
            acc
        })
        .collect();
    chunks.into_iter().flatten().collect()
}

type WeightedColEntry = (usize, u32, u16, i8);
type WeightedPosEntry = (u32, u16, i8);
const L2_TILE_BUDGET: usize = 1 << 21;
const MAX_WIDE_SHIFT_ACCUMULATIONS: usize = 1 << 15;

#[inline]
fn shift_signed_unit_into<W, const D: usize>(
    src: &WideCyclotomicRing<W, D>,
    dst: &mut WideCyclotomicRing<W, D>,
    coeff_idx: u16,
    value: i8,
) where
    W: AdditiveGroup,
{
    match value {
        1 => src.shift_accumulate_into(dst, coeff_idx as usize),
        -1 => src.shift_sub_into(dst, coeff_idx as usize),
        _ => unreachable!("sparse Frobenius coefficients are signed units"),
    }
}

#[inline]
fn shift_signed_unit_ring_into<F, const D: usize>(
    src: &CyclotomicRing<F, D>,
    dst: &mut CyclotomicRing<F, D>,
    coeff_idx: u16,
    value: i8,
) where
    F: FieldCore,
{
    match value {
        1 => src.shift_accumulate_into(dst, coeff_idx as usize),
        -1 => src.shift_sub_into(dst, coeff_idx as usize),
        _ => unreachable!("sparse Frobenius coefficients are signed units"),
    }
}

fn sparse_commit_rows_direct<F, const D: usize>(
    a_rows: &[&[CyclotomicRing<F, D>]],
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore,
{
    cfg_into_iter!(blocks)
        .map(|entries| {
            let mut accums = vec![CyclotomicRing::<F, D>::zero(); n_a];
            for entry in *entries {
                let col = entry.pos_in_block() * num_digits_commit;
                for (a_idx, a_row) in a_rows.iter().take(n_a).enumerate() {
                    shift_signed_unit_ring_into(
                        &a_row[col],
                        &mut accums[a_idx],
                        entry.coeff_idx,
                        entry.value,
                    );
                }
            }
            accums
        })
        .collect()
}

pub(crate) fn column_sweep_sparse<F, const D: usize>(
    a_rows: &[&[CyclotomicRing<F, D>]],
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    block_len: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_blocks = blocks.len();
    if blocks
        .iter()
        .any(|entries| entries.len() > MAX_WIDE_SHIFT_ACCUMULATIONS)
    {
        return sparse_commit_rows_direct(a_rows, blocks, n_a, num_digits_commit);
    }

    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = L2_TILE_BUDGET
        .checked_div(accum_bytes)
        .map_or(num_blocks, |tile| tile.max(1));

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_blocks);
            if block_start >= block_end {
                return Vec::new();
            }
            let my_count = block_end - block_start;
            let mut result = Vec::with_capacity(my_count);
            result.resize_with(my_count, Vec::new);
            let mut col_entries: Vec<WeightedColEntry> = Vec::new();
            let mut pos_offsets: Vec<usize> = Vec::new();
            let mut pos_cursor: Vec<usize> = Vec::new();
            let mut pos_entries: Vec<WeightedPosEntry> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_len = tile_end - tile_start;
                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                let tile_blocks = &blocks[(block_start + tile_start)..(block_start + tile_end)];
                let entry_count = tile_blocks
                    .iter()
                    .map(|entries| entries.len())
                    .sum::<usize>();
                // Dense tiles are cheaper to bucket by block position than to
                // comparison-sort by A-column.
                if entry_count >= block_len {
                    pos_offsets.clear();
                    pos_offsets.resize(block_len + 1, 0);
                    for block_entries in tile_blocks {
                        for entry in *block_entries {
                            pos_offsets[entry.pos_in_block() + 1] += 1;
                        }
                    }
                    for pos in 1..=block_len {
                        pos_offsets[pos] += pos_offsets[pos - 1];
                    }

                    pos_entries.clear();
                    pos_entries.resize(entry_count, (0, 0, 0));
                    pos_cursor.clear();
                    pos_cursor.extend_from_slice(&pos_offsets[..block_len]);
                    for (local_b, block_entries) in tile_blocks.iter().enumerate() {
                        for entry in *block_entries {
                            let pos = entry.pos_in_block();
                            let dst = pos_cursor[pos];
                            pos_cursor[pos] += 1;
                            pos_entries[dst] = (local_b as u32, entry.coeff_idx, entry.value);
                        }
                    }

                    for (a_idx, a_row) in a_rows.iter().take(n_a).enumerate() {
                        for pos in 0..block_len {
                            let start = pos_offsets[pos];
                            let end = pos_offsets[pos + 1];
                            if start == end {
                                continue;
                            }
                            let a_wide =
                                WideCyclotomicRing::from_ring(&a_row[pos * num_digits_commit]);
                            for &(local_b, coeff_idx, value) in &pos_entries[start..end] {
                                shift_signed_unit_into(
                                    &a_wide,
                                    &mut accums[local_b as usize][a_idx],
                                    coeff_idx,
                                    value,
                                );
                            }
                        }
                    }
                } else {
                    col_entries.clear();
                    for local_b in 0..tile_len {
                        for entry in blocks[block_start + tile_start + local_b] {
                            col_entries.push((
                                entry.pos_in_block() * num_digits_commit,
                                local_b as u32,
                                entry.coeff_idx,
                                entry.value,
                            ));
                        }
                    }
                    col_entries.sort_unstable_by_key(|&(col, _, _, _)| col);

                    for (a_idx, a_row) in a_rows.iter().take(n_a).enumerate() {
                        let mut idx = 0usize;
                        while idx < col_entries.len() {
                            let col = col_entries[idx].0;
                            let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                            while idx < col_entries.len() && col_entries[idx].0 == col {
                                let (_, local_b, coeff_idx, value) = col_entries[idx];
                                shift_signed_unit_into(
                                    &a_wide,
                                    &mut accums[local_b as usize][a_idx],
                                    coeff_idx,
                                    value,
                                );
                                idx += 1;
                            }
                        }
                    }
                }
                for (local_b, row_accums) in accums.into_iter().enumerate() {
                    result[tile_start + local_b] =
                        row_accums.into_iter().map(|w| w.reduce()).collect();
                }
            }
            result
        })
        .collect();

    let mut out = Vec::with_capacity(num_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

fn decompose_commit_rows<F, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    n_a: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let zero_block_len = n_a.checked_mul(num_digits_open).ok_or_else(|| {
        AkitaError::InvalidSetup("commit witness digit block length overflow".to_string())
    })?;
    let mut out = FlatDigitBlocks::zeroed(vec![zero_block_len; rows.len()])?;
    let dst_blocks = out.split_blocks_mut();
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(rows))
        .for_each(|(dst, row)| {
            if !row.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(row, dst, num_digits_open, log_basis);
            }
        });
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(rows.iter())
        .for_each(|(dst, row)| {
            if !row.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(row, dst, num_digits_open, log_basis);
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DensePoly;
    use akita_field::{FpExt4, Prime128OffsetA7F7 as F};
    use akita_types::{tensor_column_partials_from_base_evals, tensor_packed_witness_evals};

    #[test]
    fn sparse_ring_fold_matches_dense_reference() {
        const D: usize = 8;
        let sparse = SparseRingPoly::<F, D>::from_signed_coeffs(
            5,
            4,
            vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)],
        )
        .unwrap();
        let mut dense_coeffs = vec![CyclotomicRing::<F, D>::zero(); 4];
        dense_coeffs[0].coeffs[1] += F::one();
        dense_coeffs[1].coeffs[3] -= F::one();
        dense_coeffs[3].coeffs[2] += F::one();
        let dense = DensePoly::<F, D>::from_ring_coeffs(dense_coeffs);
        let scalars = (0..2)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_u64(10 + idx * 10 + k as u64)
                }))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sparse.fold_blocks_ring(&scalars, 2),
            dense.fold_blocks_ring(&scalars, 2)
        );
    }

    #[test]
    fn sorted_sparse_ring_constructor_rejects_unsorted_coeffs() {
        const D: usize = 8;
        let sorted =
            SparseRingPoly::<F, D>::from_sorted_signed_coeffs(5, 4, vec![(0, 1, 1), (2, 3, -1)])
                .unwrap();
        assert_eq!(sorted.num_ring_elems(), 4);

        assert!(SparseRingPoly::<F, D>::from_sorted_signed_coeffs(
            5,
            4,
            vec![(2, 3, -1), (0, 1, 1)],
        )
        .is_err());
    }

    #[test]
    fn sparse_ring_constructor_rejects_non_signed_unit_coefficients() {
        const D: usize = 8;
        for value in [-2, 0, 2] {
            assert!(matches!(
                SparseRingPoly::<F, D>::from_signed_coeffs(5, 4, vec![(0, 1, value)]),
                Err(AkitaError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn sparse_ring_tensor_hooks_match_dense_reference() {
        const D: usize = 8;
        type E = FpExt4<F>;
        let num_vars = 5;
        let total_ring_elems = 4;
        let coeffs = vec![(0, 1, 1), (1, 3, -1), (3, 6, 1)];
        let sparse =
            SparseRingPoly::<F, D>::from_signed_coeffs(num_vars, total_ring_elems, coeffs.clone())
                .unwrap();
        let mut base_evals = vec![F::zero(); 1 << num_vars];
        for (ring_idx, coeff_idx, value) in coeffs {
            base_evals[ring_idx * D + coeff_idx] += F::from_i8(value);
        }
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(idx as u64 + 2),
                    F::from_u64(3 * idx as u64 + 4),
                    F::from_u64(5 * idx as u64 + 6),
                    F::from_u64(7 * idx as u64 + 8),
                ])
            })
            .collect::<Vec<_>>();

        let expected_partials =
            tensor_column_partials_from_base_evals::<F, E>(num_vars, &base_evals, &point).unwrap();
        let got_partials = sparse.tensor_extension_column_partials::<E>(&point).unwrap();
        assert_eq!(got_partials, expected_partials);

        let expected_packed = tensor_packed_witness_evals::<F, E>(num_vars, &base_evals).unwrap();
        let expected_entries = expected_packed
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| *value != E::zero())
            .collect::<Vec<_>>();
        let got = sparse
            .tensor_packed_extension_sparse_evals::<E>()
            .unwrap()
            .unwrap();
        assert_eq!(got.table_len(), expected_packed.len());
        assert_eq!(got.entries(), expected_entries.as_slice());
    }

    #[test]
    fn packed_sparse_ring_constructor_matches_tuple_constructor() {
        const D: usize = 8;
        let tuples = vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)];
        let packed = tuples
            .iter()
            .copied()
            .map(|(ring_idx, coeff_idx, value)| {
                SparseRingCoeff::new(ring_idx, coeff_idx, value).unwrap()
            })
            .collect::<Vec<_>>();
        let from_tuples = SparseRingPoly::<F, D>::from_signed_coeffs(5, 4, tuples).unwrap();
        let from_packed = SparseRingPoly::<F, D>::from_packed_coeffs(5, 4, packed).unwrap();

        let scalars = (0..2)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_u64(20 + idx * 10 + k as u64)
                }))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            from_packed.fold_blocks_ring(&scalars, 2),
            from_tuples.fold_blocks_ring(&scalars, 2)
        );
    }
}
