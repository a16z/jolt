//! Prover-facing API surface for the Akita PCS.
//!
//! This crate owns prover-side polynomial backends, setup artifacts, recursive
//! witness construction, ring-switch handoff, and Akita-specific sumcheck
//! provers. Config and schedule policy live in `akita-config`.

pub mod api;
pub mod backend;
pub mod compute;
pub mod kernels;
pub mod protocol;
mod validation;

use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::{
    embed_ring_subfield_vector, CleartextWitnessProof, FlatDigitBlocks, FpExtEncoding,
    OpeningPoints,
};

pub use api::{
    batched_commit, batched_commit_with_params, commit, commit_setup_prefix, commit_with_params,
    prepare_batched_commit_inputs, prepare_commit_inputs, AkitaProverSetup, CommitmentProver,
};
pub use backend::FoldInputPoly;
pub use backend::{
    tensor_pack_recursive_witness, DensePoly, MultiChunkEntry, MultilinearPolynomial, OneHotIndex,
    OneHotPoly, RecursiveCommitmentHintCache, RecursiveWitnessFlat, SingleChunkEntry,
    SparseRingBlockEntry, SparseRingPoly, SuffixWitness,
};
pub use compute::{
    CommitmentComputeBackend, ComputeBackendSetup, CpuBackend, CpuPreparedSetup,
    CyclicRowsComputeBackend, DenseCommitInput, DenseCommitRowsPlan, DigitRowsComputeBackend,
    FlatBlockTable, OneHotCommitBlocks, OneHotCommitRowsPlan, PreparedCrtNttProfile,
    ProverComputeBackend, RecursiveWitnessCommitRowsPlan, RingSwitchComputeBackend,
    RingSwitchQuotientRowsPlan, RingSwitchRelationRows, RingSwitchRelationRowsPlan,
    SparseRingCommitRowsPlan,
};
pub use protocol::fold_grind::ProverTranscriptGrind;
pub use protocol::fold_grind_observer::{FoldGrindObservation, FoldGrindObserverGuard};
pub use protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
pub use protocol::{
    batched_prove, commit_next_w, prepare_batched_prove_inputs, prove, prove_root,
    prove_root_direct, prove_suffix, prove_terminal_root_fold_with_params,
    PreparedBatchedProveInputs, ProveLevelOutput, RecursiveSuffixOutcome, RingSwitchOutput,
    SuffixProverState,
};
pub use protocol::{RingRelationInstance, RingRelationProver, RingRelationWitness};
/// One PCS commitment and the polynomials it bundles, all opened at the batch's
/// shared opening point.
///
/// `polynomials` is the exact bundle committed by the prover commitment API;
/// `commitment` and `hint` are the corresponding outputs for that bundle.
#[derive(Debug, Clone)]
pub struct CommittedPolynomials<'a, P, C, H> {
    /// Polynomials addressable by claim `poly_idx` values at this point.
    pub polynomials: &'a [P],
    /// Commitment for `polynomials`.
    pub commitment: &'a C,
    /// Prover-side hint for `commitment`.
    pub hint: H,
}

impl<'a, P, C, H> CommittedPolynomials<'a, P, C, H> {
    /// Number of polynomials addressable by opening-batch claims at this point.
    pub fn poly_count(&self) -> usize {
        self.polynomials.len()
    }
}

/// Batched prover input: one shared opening point plus commitment bundles.
///
/// Mirror of [`akita_types::VerifierClaims`]: `(shared_point, Vec<CommittedPolynomials>)`.
/// See `akita_types::proof::scheme` for the single-point batching contract.
pub type ProverClaims<'a, F, P, C, H> =
    (OpeningPoints<'a, F>, Vec<CommittedPolynomials<'a, P, C, H>>);

/// Prover-side output of the decompose + challenge-fold step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeFoldWitness<F: FieldCore, const D: usize> {
    /// Folded witness rows in ring form.
    pub z_folded_rings: Vec<CyclotomicRing<F, D>>,
    /// Centered integer coefficients for each `z_folded_rings` row.
    pub centered_coeffs: Vec<[i32; D]>,
    /// Infinity norm of `centered_coeffs`.
    pub centered_inf_norm: u32,
}

/// Prover-side output of the inner Ajtai commit step.
pub struct CommitInnerWitness<F: FieldCore, const D: usize> {
    /// Recombined inner `A * s_i` rows, grouped by block.
    pub recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Digit decompositions of `A * s_i` in flat column-major order plus
    /// explicit block boundaries.
    pub decomposed_inner_rows: FlatDigitBlocks<D>,
}

/// Operations the Akita commitment scheme needs from a root polynomial.
///
/// Each method corresponds to a place in commit/prove that consumes polynomial
/// data. Implementations decide how to carry out each operation: dense
/// decomposition, sparse one-hot tricks, digit-plane bypasses, or other
/// backend-specific strategies.
#[allow(clippy::too_many_arguments)]
pub trait AkitaPolyOps<F: FieldCore, const D: usize>: Clone + Send + Sync {
    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// Total number of variables (field-element dimension).
    ///
    /// Derived from `num_ring_elems() * D`, which equals `2^num_vars`.
    ///
    /// # Panics
    ///
    /// Panics if `num_ring_elems() * D` overflows `usize`.
    fn num_vars(&self) -> usize {
        let total = self
            .num_ring_elems()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        debug_assert!(
            total.is_power_of_two(),
            "total field elements must be a power of 2"
        );
        total.trailing_zeros() as usize
    }

    /// One-hot chunk size for sparse one-hot backends.
    ///
    /// `None` means this backend is not a one-hot root representation. Configs
    /// that use a non-default one-hot security policy validate this value at
    /// commit/prove boundaries before relying on the tighter SIS schedule.
    fn onehot_chunk_size(&self) -> Option<usize> {
        None
    }

    /// Materialize the base-field evaluation table for this polynomial view.
    ///
    /// Root backends usually expose this through their direct witness payload.
    /// Recursive suffix views can override it to convert their carried digit
    /// representation into the padded field-element cube they represent.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot expose base-field evaluations.
    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        let witness = self.direct_root_witness()?;
        let field_elems = witness.as_field_elements().ok_or_else(|| {
            AkitaError::InvalidInput("base evals require field-element witness payload".to_string())
        })?;
        Ok(field_elems.coeffs().to_vec())
    }

    /// Prover per-block fold.
    ///
    /// For each contiguous block of `block_len` ring elements, computes
    /// `sum_j scalars[j] * self[i * block_len + j]`.
    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>>;

    /// Prover per-block fold with ring multipliers.
    ///
    /// This is the extension-field baseline path: extension opening weights are
    /// embedded into the ring-subfield of `R_F`, then act on witness rings by
    /// ordinary ring multiplication. Degree-one openings use constant ring
    /// multipliers and specialize to [`Self::fold_blocks`].
    fn fold_blocks_ring(
        &self,
        _scalars: &[CyclotomicRing<F, D>],
        _block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        panic!("backend must override fold_blocks_ring")
    }

    /// Fused fold + evaluation in one pass over the polynomial.
    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks(fold_scalars, block_len);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    /// Fused ring-multiplier fold + evaluation in one pass over the polynomial.
    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks_ring(fold_scalars, block_len);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + (*f_i * *s_i)
            });
        (eval, folded)
    }

    /// Evaluate the root polynomial at an extension-field point.
    ///
    /// Backends with sparse structure should override this method. The default
    /// materializes the direct root witness and folds it as a dense
    /// multilinear table.
    ///
    /// # Errors
    ///
    /// Returns an error if the point has the wrong arity or if the backend
    /// cannot expose a field-element root witness.
    fn evaluate_extension<E>(&self, point: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        let num_vars = self.num_vars();
        if point.len() != num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: num_vars,
                actual: point.len(),
            });
        }
        let base_evals = self.base_evals()?;
        let expected_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput("root extension evaluation table length overflow".to_string())
        })?;
        if base_evals.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: base_evals.len(),
            });
        }
        let mut layer = base_evals
            .iter()
            .copied()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        for &r in point {
            let one_minus_r = E::one() - r;
            let next_len = layer.len() / 2;
            for i in 0..next_len {
                layer[i] = layer[2 * i] * one_minus_r + layer[2 * i + 1] * r;
            }
            layer.truncate(next_len);
        }
        Ok(layer[0])
    }

    /// Compute the tensor-column partials used by root extension-opening
    /// reduction.
    ///
    /// Backends with sparse structure should override this method. The default
    /// materializes the direct root witness and evaluates all tensor heads in
    /// one pass over the tail equality table.
    ///
    /// # Errors
    ///
    /// Returns an error if the point has the wrong arity, if the backend
    /// cannot expose a field-element root witness, or if the tensor shape is
    /// invalid.
    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let num_vars = self.num_vars();
        if logical_point.len() != num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: num_vars,
                actual: logical_point.len(),
            });
        }
        let base_evals = self.base_evals()?;
        akita_types::tensor_column_partials_from_base_evals::<F, E>(
            num_vars,
            &base_evals,
            logical_point,
        )
    }

    /// Compute tensor-column partials for several polynomials at one point.
    ///
    /// Backends may override this to share point-dependent work across a
    /// same-point batch. The default preserves the scalar method behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if any polynomial rejects the point or tensor shape.
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

    /// Materialize the tensor-packed root witness table used by extension
    /// opening reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot expose a field-element root
    /// witness or if the tensor-packing shape is invalid.
    fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let num_vars = self.num_vars();
        let base_evals = self.base_evals()?;
        akita_types::tensor_packed_witness_evals::<F, E>(num_vars, &base_evals)
    }

    /// Materialize a sparse tensor-packed root witness when the backend can
    /// preserve sparsity through extension-opening reduction.
    ///
    /// Dense backends return `Ok(None)` and use
    /// [`Self::tensor_packed_extension_evals`].
    ///
    /// # Errors
    ///
    /// Returns an error if the backend's sparse tensor-packed shape is
    /// malformed.
    fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        Ok(None)
    }

    /// Build a sparse linear combination of tensor-packed root witnesses.
    ///
    /// Backends that can combine sparse transformed roots directly should
    /// override this method. The default preserves the same sparse/dense
    /// fallback contract as [`Self::tensor_packed_extension_sparse_evals`].
    ///
    /// # Errors
    ///
    /// Returns an error if the coefficient list does not match `polys`, or if
    /// any sparse tensor-packed witness is malformed.
    fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        if polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: polys.len(),
                actual: coeffs.len(),
            });
        }
        let mut witnesses = Vec::with_capacity(polys.len());
        for poly in polys {
            let Some(witness) = poly.tensor_packed_extension_sparse_evals::<E>()? else {
                return Ok(None);
            };
            witnesses.push(witness);
        }
        Ok(Some(SparseExtensionOpeningWitness::linear_combination(
            coeffs.iter().copied().zip(witnesses.iter()),
        )?))
    }

    /// Materialize the tensor-packed root polynomial committed for extension
    /// opening reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if the source witness cannot be tensor-packed or if
    /// the ring-subfield embedding rejects the transformed root shape.
    fn tensor_packed_extension_poly<E>(&self) -> Result<DensePoly<F, D>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        let evals = self.tensor_packed_extension_evals::<E>()?;
        let packed_len = D / E::EXT_DEGREE;
        if packed_len == 0 {
            return Err(AkitaError::InvalidInput(
                "extension degree exceeds root ring dimension".to_string(),
            ));
        }
        let mut rings = Vec::with_capacity(evals.len().div_ceil(packed_len));
        for chunk in evals.chunks(packed_len) {
            let mut values = chunk.to_vec();
            values.resize(packed_len, E::zero());
            rings.push(embed_ring_subfield_vector::<F, E, D>(
                &values,
                AkitaError::InvalidInput(
                    "root transformed witness does not encode in the ring-subfield basis"
                        .to_string(),
                ),
            )?);
        }
        Ok(DensePoly::<F, D>::from_ring_coeffs(rings))
    }

    /// Materialize the committed tensor-projected fold input while preserving
    /// sparse transformed roots when available.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor packing or ring-subfield embedding rejects
    /// the transformed root shape.
    fn tensor_packed_extension_fold_input<E>(
        &self,
    ) -> Result<FoldInputPoly<'_, F, Self, D>, AkitaError>
    where
        Self: Sized,
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        Ok(FoldInputPoly::projected_dense(
            self.tensor_packed_extension_poly::<E>()?,
        ))
    }

    /// Prover decompose + challenge-fold step.
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D>;

    /// Optional fused batched variant of [`Self::decompose_fold`].
    fn decompose_fold_batched(
        _polys: &[&Self],
        _challenges: &[SparseChallenge],
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        None
    }

    /// Optional tensor-shaped batched variant of [`Self::decompose_fold`].
    ///
    /// Returns `Ok(Some(witness))` when the backend implements a tensor-shaped
    /// batched kernel, `Ok(None)` when it does not, and `Err(_)` when the
    /// backend attempted the tensor fold but rejected its input.
    fn decompose_fold_tensor_batched(
        _polys: &[&Self],
        _tensor: &TensorChallenges,
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        Ok(None)
    }

    /// Inner Ajtai commit step that also preserves recomposed inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the cached matrix-vector multiply or digit
    /// decomposition fails.
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
        B: CommitmentComputeBackend<F>;

    /// Materialize a direct root witness for zero-fold openings.
    ///
    /// # Errors
    ///
    /// Returns an error when this root representation cannot produce a direct
    /// witness payload.
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "root-direct witness is not supported for this polynomial type".to_string(),
        ))
    }
}

impl<F, const D: usize, P> AkitaPolyOps<F, D> for &P
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    fn num_ring_elems(&self) -> usize {
        <P as AkitaPolyOps<F, D>>::num_ring_elems(*self)
    }

    fn num_vars(&self) -> usize {
        <P as AkitaPolyOps<F, D>>::num_vars(*self)
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        <P as AkitaPolyOps<F, D>>::onehot_chunk_size(*self)
    }

    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        <P as AkitaPolyOps<F, D>>::base_evals(*self)
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        <P as AkitaPolyOps<F, D>>::fold_blocks(*self, scalars, block_len)
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        <P as AkitaPolyOps<F, D>>::fold_blocks_ring(*self, scalars, block_len)
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        <P as AkitaPolyOps<F, D>>::evaluate_and_fold(
            *self,
            eval_outer_scalars,
            fold_scalars,
            block_len,
        )
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        <P as AkitaPolyOps<F, D>>::evaluate_and_fold_ring(
            *self,
            eval_outer_scalars,
            fold_scalars,
            block_len,
        )
    }

    fn evaluate_extension<E>(&self, point: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        <P as AkitaPolyOps<F, D>>::evaluate_extension::<E>(*self, point)
    }

    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        <P as AkitaPolyOps<F, D>>::tensor_extension_column_partials::<E>(*self, logical_point)
    }

    fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let inner_refs: Vec<&P> = polys.iter().map(|poly| **poly).collect();
        P::tensor_extension_column_partials_batch(&inner_refs, logical_point)
    }

    fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        <P as AkitaPolyOps<F, D>>::tensor_packed_extension_evals::<E>(*self)
    }

    fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        <P as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_evals::<E>(*self)
    }

    fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        let inner_refs: Vec<&P> = polys.iter().map(|poly| **poly).collect();
        P::tensor_packed_extension_sparse_linear_combination(&inner_refs, coeffs)
    }

    fn tensor_packed_extension_poly<E>(&self) -> Result<DensePoly<F, D>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        <P as AkitaPolyOps<F, D>>::tensor_packed_extension_poly::<E>(*self)
    }

    fn tensor_packed_extension_fold_input<E>(
        &self,
    ) -> Result<FoldInputPoly<'_, F, Self, D>, AkitaError>
    where
        Self: Sized,
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        match <P as AkitaPolyOps<F, D>>::tensor_packed_extension_fold_input::<E>(*self)? {
            FoldInputPoly::Original(_) => Ok(FoldInputPoly::Original(self)),
            FoldInputPoly::ProjectedDense(poly) => Ok(FoldInputPoly::ProjectedDense(poly)),
            FoldInputPoly::ProjectedSparse(poly) => Ok(FoldInputPoly::ProjectedSparse(poly)),
        }
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        <P as AkitaPolyOps<F, D>>::decompose_fold(
            *self, challenges, block_len, num_digits, log_basis,
        )
    }

    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        let inner_refs: Vec<&P> = polys.iter().map(|poly| **poly).collect();
        P::decompose_fold_batched(&inner_refs, challenges, block_len, num_digits, log_basis)
    }

    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &TensorChallenges,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        let inner_refs: Vec<&P> = polys.iter().map(|poly| **poly).collect();
        P::decompose_fold_tensor_batched(&inner_refs, tensor, block_len, num_digits, log_basis)
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
        <P as AkitaPolyOps<F, D>>::commit_inner(
            *self,
            backend,
            prepared,
            n_a,
            block_len,
            num_blocks,
            num_digits_commit,
            num_digits_open,
            log_basis,
        )
    }

    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        <P as AkitaPolyOps<F, D>>::direct_root_witness(*self)
    }
}
