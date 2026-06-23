//! Recursive witness helpers for later Akita prove levels.
//!
//! Recursive levels do not operate on a caller-provided polynomial anymore.
//! Instead they carry a flat digit witness `w` that is re-chunked under the
//! current ring dimension `D` on demand. [`RecursiveWitnessFlat`] owns the
//! D-agnostic digit buffer, while [`SuffixWitness`] provides the
//! zero-copy D-specific operations used by recursive folding and handoff paths.

#![allow(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc)]

use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};

use crate::backend::poly_helpers::{
    balanced_digit_decompose_fold_partitioned, build_decompose_fold_witness,
};
use crate::compute::{CommitmentComputeBackend, RecursiveWitnessCommitRowsPlan};
use crate::kernels::linear::decompose_rows_i8_into;
use akita_field::ExtField;
use akita_types::{tensor_packed_witness_evals, FlatDigitBlocks, FpExtEncoding};
use std::marker::PhantomData;

use crate::{AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness, FoldInputPoly};

/// D-agnostic owner for the recursive witness vector `w`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecursiveWitnessFlat {
    digits: Vec<i8>,
}

impl RecursiveWitnessFlat {
    pub fn from_i8_digits(digits: Vec<i8>) -> Self {
        Self { digits }
    }

    pub fn as_i8_digits(&self) -> &[i8] {
        &self.digits
    }

    pub fn len(&self) -> usize {
        self.digits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.digits.is_empty()
    }

    pub fn view<F: FieldCore, const D: usize>(
        &self,
    ) -> Result<SuffixWitness<'_, F, D>, AkitaError> {
        SuffixWitness::from_i8_digits(&self.digits)
    }
}

impl AsRef<[i8]> for RecursiveWitnessFlat {
    fn as_ref(&self) -> &[i8] {
        self.as_i8_digits()
    }
}

/// D-specific zero-copy view over a flat recursive witness.
#[derive(Debug, Clone, Copy)]
pub struct SuffixWitness<'a, F: FieldCore, const D: usize> {
    coeffs: &'a [[i8; D]],
    padded_ring_elems: usize,
    _marker: PhantomData<F>,
}

impl<'a, F: FieldCore, const D: usize> SuffixWitness<'a, F, D> {
    pub fn from_i8_digits(digits: &'a [i8]) -> Result<Self, AkitaError> {
        let (coeffs, remainder) = digits.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: digits.len(),
            });
        }

        Ok(Self {
            coeffs,
            padded_ring_elems: coeffs.len().next_power_of_two().max(1),
            _marker: PhantomData,
        })
    }

    #[inline]
    fn block_elem(
        &self,
        block_idx: usize,
        col_idx: usize,
        num_blocks: usize,
    ) -> Option<&'a [i8; D]> {
        self.coeffs.get(block_idx + col_idx * num_blocks)
    }

    pub fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }

    #[inline]
    fn num_blocks_for_block_len(&self, block_len: usize) -> usize {
        self.coeffs.len().div_ceil(block_len).max(1)
    }
}

impl<F, const D: usize> AkitaPolyOps<F, D> for SuffixWitness<'_, F, D>
where
    F: FieldCore + CanonicalField,
{
    fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }

    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        let expected_len = self.padded_ring_elems.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidInput("recursive base evals length overflow".to_string())
        })?;
        let mut base_evals = Vec::with_capacity(expected_len);
        for coeffs in self.coeffs {
            base_evals.extend(coeffs.iter().copied().map(F::from_i8));
        }
        base_evals.resize(expected_len, F::zero());
        Ok(base_evals)
    }

    fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let num_vars = self.num_vars();
        let base_evals = self.base_evals()?;
        tensor_packed_witness_evals::<F, E>(num_vars, &base_evals)
    }

    fn tensor_packed_extension_fold_input<E>(
        &self,
    ) -> Result<FoldInputPoly<'_, F, Self, D>, AkitaError>
    where
        Self: Sized,
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        Ok(FoldInputPoly::Original(self))
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let num_blocks = self.num_blocks_for_block_len(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = [F::zero(); D];
                for (col_idx, &scalar) in scalars.iter().take(block_len).enumerate() {
                    let Some(ring) = self.block_elem(block_idx, col_idx, num_blocks) else {
                        break;
                    };
                    for (coeff, &d) in acc.iter_mut().zip(ring.iter()) {
                        if d != 0 {
                            *coeff += scalar * F::from_i8(d);
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect()
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let num_blocks = self.num_blocks_for_block_len(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (col_idx, scalar) in scalars.iter().take(block_len).enumerate() {
                    let Some(digits) = self.block_elem(block_idx, col_idx, num_blocks) else {
                        break;
                    };
                    let ring = CyclotomicRing::<F, D>::from_coefficients(
                        digits.map(|digit| F::from_i8(digit)),
                    );
                    ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
                }
                acc
            })
            .collect()
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let num_blocks = self.num_blocks_for_block_len(block_len);
        let folded = cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = [F::zero(); D];
                for (col_idx, &scalar) in fold_scalars.iter().take(block_len).enumerate() {
                    let Some(ring) = self.block_elem(block_idx, col_idx, num_blocks) else {
                        break;
                    };
                    for (coeff, &d) in acc.iter_mut().zip(ring.iter()) {
                        if d != 0 {
                            *coeff += scalar * F::from_i8(d);
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect::<Vec<_>>();
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let num_blocks = self.num_blocks_for_block_len(block_len);
        let folded = cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (col_idx, scalar) in fold_scalars.iter().take(block_len).enumerate() {
                    let Some(digits) = self.block_elem(block_idx, col_idx, num_blocks) else {
                        break;
                    };
                    let ring = CyclotomicRing::<F, D>::from_coefficients(
                        digits.map(|digit| F::from_i8(digit)),
                    );
                    ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
                }
                acc
            })
            .collect::<Vec<_>>();
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + (*f_i * *s_i)
            });
        (eval, folded)
    }

    #[tracing::instrument(skip_all, name = "SuffixWitness::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let inner_width = block_len * num_digits;
        let num_blocks = challenges.len();

        let q = (-F::one()).to_canonical_u128() + 1;
        let coeffs = self.coeffs;
        let coeff_accum = balanced_digit_decompose_fold_partitioned::<D>(
            coeffs,
            challenges,
            num_blocks,
            block_len,
            num_blocks,
            num_digits,
            inner_width,
        );
        build_decompose_fold_witness::<F, D>(coeff_accum, q)
    }

    fn decompose_fold_batched(
        _polys: &[&Self],
        _challenges: &[SparseChallenge],
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        None
    }

    fn decompose_fold_tensor_batched(
        _polys: &[&Self],
        _tensor: &TensorChallenges,
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        Ok(None)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_rows: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let t = backend.recursive_witness_commit_rows(
            prepared,
            RecursiveWitnessCommitRowsPlan {
                coeffs: self.coeffs,
                n_rows,
                block_len,
                num_blocks,
                num_digits_commit,
                log_basis,
            },
        )?;

        let block_sizes: Vec<usize> = t.iter().map(|t_i| t_i.len() * num_digits_open).collect();
        let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t))
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(t.iter())
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows: t_hat,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn logical_rows_use_strided_column_major_indices() {
        let digits: Vec<i8> = (0..20).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w
            .view::<akita_field::Prime128OffsetA7F7, 2>()
            .expect("view");
        let num_blocks = 4;
        let block_len = (w.len() / 2).div_ceil(num_blocks);

        let row = |block_idx: usize| -> Vec<[i8; 2]> {
            (0..block_len)
                .filter_map(|col_idx| view.block_elem(block_idx, col_idx, num_blocks).copied())
                .collect()
        };

        assert_eq!(row(0), vec![[0, 1], [8, 9], [16, 17]]);
        assert_eq!(row(1), vec![[2, 3], [10, 11], [18, 19]]);
        assert_eq!(row(2), vec![[4, 5], [12, 13]]);
        assert_eq!(row(3), vec![[6, 7], [14, 15]]);
    }

    fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            F::from_u64(offset + idx as u64 + 1)
        }))
    }

    #[test]
    fn ring_fold_matches_dense_multiplication_reference() {
        const D: usize = 4;
        let digits = vec![1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12];
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let scalars = vec![ring::<D>(10), ring::<D>(20)];
        let got = view.fold_blocks_ring(&scalars, 2);

        let expected = (0..2)
            .map(|block_idx| {
                (0..2).fold(CyclotomicRing::<F, D>::zero(), |acc, col_idx| {
                    let Some(digits) = view.block_elem(block_idx, col_idx, 2) else {
                        return acc;
                    };
                    let coeff = CyclotomicRing::from_coefficients(digits.map(F::from_i8));
                    acc + coeff * scalars[col_idx]
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(got, expected);
    }

    #[test]
    fn fused_evaluation_uses_layout_block_stride() {
        const D: usize = 4;
        let digits = (0..24).map(|idx| idx as i8 - 12).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let block_len = 3;
        let eval_outer_scalars = vec![F::from_u64(2), F::from_u64(5)];
        let fold_scalars = vec![F::from_u64(7), F::from_u64(11), F::from_u64(13)];

        let expected_folded = view.fold_blocks(&fold_scalars, block_len);
        let expected_eval = expected_folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        let (eval, folded) = view.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, block_len);

        assert_eq!(folded, expected_folded);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn fused_ring_evaluation_uses_layout_block_stride() {
        const D: usize = 4;
        let digits = (0..24).map(|idx| idx as i8 - 12).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let block_len = 3;
        let eval_outer_scalars = vec![ring::<D>(2), ring::<D>(5)];
        let fold_scalars = vec![ring::<D>(7), ring::<D>(11), ring::<D>(13)];

        let expected_folded = view.fold_blocks_ring(&fold_scalars, block_len);
        let expected_eval = expected_folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + (*f_i * *s_i)
            });
        let (eval, folded) =
            view.evaluate_and_fold_ring(&eval_outer_scalars, &fold_scalars, block_len);

        assert_eq!(folded, expected_folded);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn akita_poly_ops_delegates_to_recursive_witness_layout() {
        const D: usize = 16;
        let digits = (0..48).map(|idx| (idx % 7) as i8 - 3).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let eval_outer_scalars = vec![F::from_u64(3), F::from_u64(5)];
        let fold_scalars = vec![F::from_u64(7), F::from_u64(11)];
        let challenges = vec![
            SparseChallenge {
                positions: vec![0, 2],
                coeffs: vec![1, -1],
            },
            SparseChallenge {
                positions: vec![1, 3],
                coeffs: vec![2, 1],
            },
        ];

        assert_eq!(
            <SuffixWitness<'_, F, D> as AkitaPolyOps<F, D>>::evaluate_and_fold(
                &view,
                &eval_outer_scalars,
                &fold_scalars,
                2,
            ),
            view.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, 2)
        );
        assert_eq!(
            <SuffixWitness<'_, F, D> as AkitaPolyOps<F, D>>::decompose_fold(
                &view,
                &challenges,
                2,
                1,
                0,
            ),
            view.decompose_fold(&challenges, 2, 1, 0)
        );

        let wrapped = crate::MultilinearPolynomial::<F, D>::recursive(view);
        assert_eq!(
            wrapped.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, 2),
            view.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, 2)
        );
        assert_eq!(
            wrapped.decompose_fold(&challenges, 2, 1, 0),
            view.decompose_fold(&challenges, 2, 1, 0)
        );
    }
}
