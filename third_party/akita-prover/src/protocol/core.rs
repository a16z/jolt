//! Prover core state shared by root orchestration during crate extraction.

use crate::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm,
    SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
};
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, NextWitnessCommitment, RingSwitchOutput,
};
use crate::protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover, SetupSumcheckProver};
#[cfg(feature = "zk")]
use crate::protocol::zk_hiding_commit::commit_zk_hiding_witness;
use crate::protocol::RingRelationProver;
use crate::{
    AkitaPolyOps, CommittedPolynomials, FoldInputPoly, ProverClaims, ProverComputeBackend,
    ProverTranscriptGrind, RecursiveCommitmentHintCache, RecursiveWitnessFlat,
    RingRelationInstance, RingRelationWitness,
};
use akita_algebra::CyclotomicRing;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, Invertible, MulBaseUnreduced, PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
#[cfg(feature = "zk")]
use akita_sumcheck::{
    CompressedUniPoly, EqFactoredUniPoly, SumcheckProofMasked, ZkSumcheckInstanceProverExt,
};
#[cfg(not(feature = "zk"))]
use akita_sumcheck::{SumcheckInstanceProverExt, SumcheckProof};
#[cfg(feature = "zk")]
use akita_transcript::labels::ABSORB_ZK_HIDING_COMMITMENT;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_NEXT_LEVEL_WITNESS_BINDING,
    ABSORB_STAGE2_NEXT_W_EVAL, ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_W_REMAINDER,
    CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::dispatch_ring_dim_result;
use akita_types::FpExtEncoding;
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_values_to_transcript,
    append_opening_batch_shape_to_transcript, basis_weights,
    batched_eval_target_from_opening_batch, build_trace_table_scaled,
    check_extension_opening_reduction_output, derive_tensor_extension_opening_claim,
    derive_tensor_extension_opening_claim_from_partials, embed_ring_subfield_scalar,
    embed_ring_subfield_vector, ensure_trace_stage2_supported, flatten_batched_commitment_rows,
    folded_root_supports_opening_shape, prepare_opening_point, recover_ring_subfield_inner_product,
    relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_current_w_len, root_direct_schedule,
    root_tensor_projection_enabled, sample_public_row_coefficients, schedule_is_root_direct,
    schedule_num_fold_levels, schedule_root_fold_step, stage2_trace_coeff,
    tensor_equality_factor_eval_at_point, tensor_equality_factor_evals, tensor_opening_split,
    tensor_packed_witness_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, trace_public_weights_recursive,
    trace_public_weights_root_terms, trace_weight_layout_from_segment, validate_batched_inputs,
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaCommitmentHint, AkitaExpandedSetup,
    AkitaIntermediateStage2Proof, AkitaLevelProof, AkitaStage1Proof, AkitaStage2Proof, BasisMode,
    BlockOrder, CleartextWitnessProof, ExecutionSchedule, ExtensionOpeningReductionProof,
    FlatRingVec, LevelParams, MRowLayout, OpeningBatch, OpeningBatchInput, OpeningBatchLimits,
    OpeningClaimKind, OpeningClaimSlot, PreparedOpeningPoint, RingCommitment,
    RingMultiplierOpeningPoint, RingRelationSegmentLayout, Schedule, SetupContributionMode,
    SetupPrefixProverRegistry, SetupSumcheckProof, Step, TerminalLevelProof, TraceTable,
};
#[cfg(feature = "zk")]
use akita_types::{stage1_tree_stage_shapes, sumcheck_rounds, PackedDigits, ZkHidingProof};
#[cfg(feature = "zk")]
use rand_core::OsRng;
use std::sync::Arc;

pub(in crate::protocol::core) struct ExtensionOpeningReduction<L: FieldCore> {
    pub(in crate::protocol::core) proof: ExtensionOpeningReductionProof<L>,
    /// EOR final sumcheck claim and transparent-factor evaluation. Retained so
    /// the prepare step can fail-fast cross-check the folded opening against
    /// the reduction output; the verifier enforces the same relation.
    pub(in crate::protocol::core) final_claim: L,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::core) final_claim_public: L,
    pub(in crate::protocol::core) final_factor: L,
}

mod extension_opening_reduction;
mod fold;
mod prove;
mod root_fold;
mod suffix;
#[cfg(test)]
mod tests;

pub(in crate::protocol::core) use extension_opening_reduction::*;
pub(in crate::protocol::core) use fold::{prepare_fold_inner, prove_fold, PreparedFold};
pub use prove::{batched_prove, prepare_batched_prove_inputs, prove, prove_root_direct};
pub use root_fold::{prove_root, prove_terminal_root_fold_with_params};
pub use suffix::{prove_suffix, SuffixProverState};

/// Cursor into the proof-level hiding witness allocated at batched-prove start.
#[cfg(feature = "zk")]
#[derive(Debug, PartialEq, Eq)]
pub struct ZkHidingProverState<F: FieldCore> {
    hiding_witness: Vec<F>,
    cursor: usize,
}

/// Top-level hiding commitment pieces fixed before transcript replay starts.
#[cfg(feature = "zk")]
#[derive(Debug, PartialEq, Eq)]
pub struct ZkHidingCommitment<F: FieldCore> {
    /// Wire-visible commitment to the proof-level hiding witness.
    pub u_blind: Vec<F>,
    /// Dedicated short Ajtai blinding digits used for `u_blind`.
    pub b_blinding_digits: Vec<i8>,
}

#[cfg(feature = "zk")]
impl<F: FieldCore> ZkHidingProverState<F> {
    fn new(hiding_witness: Vec<F>) -> Self {
        Self {
            hiding_witness,
            cursor: 0,
        }
    }

    fn take_values(&mut self, len: usize) -> Result<&[F], AkitaError> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(AkitaError::InvalidProof)?;
        let values = self
            .hiding_witness
            .get(self.cursor..end)
            .ok_or(AkitaError::InvalidProof)?;
        self.cursor = end;
        Ok(values)
    }

    fn take_ext_scalar<L>(&mut self) -> Result<L, AkitaError>
    where
        L: ExtField<F>,
    {
        Ok(L::from_base_slice(self.take_values(L::EXT_DEGREE)?))
    }

    fn into_proof(self, commitment: ZkHidingCommitment<F>) -> Result<ZkHidingProof<F>, AkitaError> {
        if self.cursor != self.hiding_witness.len() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(ZkHidingProof {
            u_blind: commitment.u_blind,
            hiding_witness: self.hiding_witness,
            b_blinding_digits: commitment.b_blinding_digits,
        })
    }

    fn take_next_w_eval_mask<L>(&mut self) -> Result<L, AkitaError>
    where
        L: ExtField<F>,
    {
        self.take_ext_scalar()
    }

    fn take_eq_factored_rounds<L>(
        &mut self,
        rounds: usize,
        degree: usize,
    ) -> Result<Vec<EqFactoredUniPoly<L>>, AkitaError>
    where
        L: ExtField<F>,
    {
        let stored_coeffs = EqFactoredUniPoly::<L>::stored_coeff_count_for_degree(degree);
        (0..rounds)
            .map(|_| {
                let coeffs = (0..stored_coeffs)
                    .map(|_| self.take_ext_scalar())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(EqFactoredUniPoly {
                    coeffs_except_linear_term: coeffs,
                })
            })
            .collect()
    }

    fn take_compressed_rounds<L>(
        &mut self,
        rounds: usize,
        degree: usize,
    ) -> Result<Vec<CompressedUniPoly<L>>, AkitaError>
    where
        L: ExtField<F>,
    {
        (0..rounds)
            .map(|_| {
                let coeffs = (0..degree)
                    .map(|_| self.take_ext_scalar())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(CompressedUniPoly {
                    coeffs_except_linear_term: coeffs,
                })
            })
            .collect()
    }

    fn take_current_level_pads<L>(
        &mut self,
        rounds: usize,
        b: usize,
    ) -> Result<ZkLevelRoundPads<L>, AkitaError>
    where
        L: ExtField<F>,
    {
        let mut stage1_round_pads = Vec::new();
        let mut stage1_child_claim_masks = Vec::new();
        for shape in stage1_tree_stage_shapes(rounds, b) {
            stage1_round_pads.push(
                self.take_eq_factored_rounds(shape.sumcheck_proof.0, shape.sumcheck_proof.1)?,
            );
            if shape.child_claims != 0 {
                stage1_child_claim_masks.push(
                    (0..shape.child_claims)
                        .map(|_| self.take_ext_scalar())
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        let stage2_round_pads = self.take_compressed_rounds(rounds, 3)?;
        Ok((
            stage1_round_pads,
            stage1_child_claim_masks,
            stage2_round_pads,
        ))
    }

    fn take_extension_opening_reduction_pads<L>(
        &mut self,
        partials: usize,
        rounds: usize,
    ) -> Result<(Vec<L>, Vec<CompressedUniPoly<L>>), AkitaError>
    where
        L: ExtField<F>,
    {
        let partial_masks = (0..partials)
            .map(|_| self.take_ext_scalar())
            .collect::<Result<Vec<_>, _>>()?;
        let round_pads =
            self.take_compressed_rounds(rounds, akita_types::EXTENSION_OPENING_REDUCTION_DEGREE)?;
        Ok((partial_masks, round_pads))
    }
}

#[cfg(feature = "zk")]
fn masked_sumcheck_final_claim<E: FieldCore>(
    input_claim: E,
    proof: &SumcheckProofMasked<E>,
    challenges: &[E],
) -> Result<E, AkitaError> {
    if proof.masked_round_polys.len() != challenges.len() {
        return Err(AkitaError::InvalidSize {
            expected: challenges.len(),
            actual: proof.masked_round_polys.len(),
        });
    }
    Ok(proof
        .masked_round_polys
        .iter()
        .zip(challenges)
        .fold(input_claim, |claim, (poly, r)| {
            poly.eval_from_hint(&claim, r)
        }))
}

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore, L: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: AkitaLevelProof<F, L>,
    /// Suffix prover state for the next level.
    pub next_state: SuffixProverState<F, L>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: FieldCore, L: FieldCore> {
    /// Recursive suffix proof steps: intermediate folds followed by terminal.
    pub steps: Vec<AkitaLevelProof<F, L>>,
    /// Proof-level ZK hiding witness state after all suffix masks are consumed.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProverState<F>,
    /// Total fold-level count reached, including the root level and the
    /// terminal level.
    pub num_levels: usize,
}

#[cfg(not(feature = "zk"))]
pub(in crate::protocol::core) type Stage2ProveResult<L> =
    (SumcheckProof<L>, Vec<L>, AkitaStage2Prover<L>);
#[cfg(feature = "zk")]
pub(in crate::protocol::core) type Stage2ProveResult<L> =
    (SumcheckProofMasked<L>, Vec<L>, AkitaStage2Prover<L>);

fn scalar_opening_from_folded_ring<F, E, L, const D: usize>(
    folded_ring: &CyclotomicRing<F, D>,
    prepared_point: &PreparedOpeningPoint<F, L, D>,
    inner_opening_point: &[E],
    basis: BasisMode,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
    L: FieldCore,
{
    if <E as ExtField<F>>::EXT_DEGREE == 1 {
        return (*folded_ring * prepared_point.packed_inner_point.sigma_m1())
            .coefficients()
            .first()
            .copied()
            .map(E::lift_base)
            .ok_or_else(|| AkitaError::InvalidInput("empty folded opening ring".to_string()));
    }
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "claim-field degree must divide the ring dimension into power-of-two slots".to_string(),
        ));
    }
    let packed_slots = D / <E as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if inner_opening_point.len() > packed_inner_bits
        && inner_opening_point[packed_inner_bits..]
            .iter()
            .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: inner_opening_point.len(),
        });
    }
    let mut point =
        inner_opening_point[..inner_opening_point.len().min(packed_inner_bits)].to_vec();
    point.resize(packed_inner_bits, E::zero());
    let weights = basis_weights(&point, basis)?;
    let packed_inner_point = embed_ring_subfield_vector::<F, E, D>(
        &weights,
        AkitaError::InvalidInput(
            "root opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    recover_ring_subfield_inner_product::<F, E, D>(folded_ring, &packed_inner_point)
}

fn row_coefficient_rings<F, L, const D: usize>(
    coefficients: &[L],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    L: FpExtEncoding<F>,
{
    coefficients
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, L, D>(
                coefficient,
                AkitaError::InvalidInput(
                    "public-row coefficient does not encode in the ring-subfield basis".to_string(),
                ),
            )
        })
        .collect()
}

/// Config-free flattened view of batched prover claims.
pub struct PreparedBatchedProveInputs<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    /// Shared opening point.
    pub opening_point: &'a [E],
    /// Commitments in commitment-group order.
    pub commitments: Vec<RingCommitment<F, D>>,
    /// Normalized opening-batch summary that owns canonical root claim routing.
    pub opening_batch: OpeningBatch,
    /// Polynomials flattened in claim order.
    pub flat_polys: Vec<&'a P>,
    /// Commitment hints in commitment-group order.
    pub commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
}

#[cfg(feature = "zk")]
type ZkLevelRoundPads<L> = (
    Vec<Vec<akita_sumcheck::EqFactoredUniPoly<L>>>,
    Vec<Vec<L>>,
    Vec<akita_sumcheck::CompressedUniPoly<L>>,
);

#[cfg(feature = "zk")]
fn push_random_ext_scalar_slots<F, L>(out: &mut Vec<F>, rng: &mut OsRng)
where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    out.extend((0..L::EXT_DEGREE).map(|_| F::random(&mut *rng)));
}

#[cfg(feature = "zk")]
fn append_zk_stage2_pad_slots<F, L>(rounds: usize, out: &mut Vec<F>, rng: &mut OsRng)
where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    for _ in 0..rounds * 3 {
        push_random_ext_scalar_slots::<F, L>(out, rng);
    }
}

#[cfg(feature = "zk")]
fn append_zk_level_pad_slots<F, L>(
    params: &LevelParams,
    next_w_len: usize,
    include_stage1: bool,
    out: &mut Vec<F>,
    rng: &mut OsRng,
) -> Result<(), AkitaError>
where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    let rounds = sumcheck_rounds(params.ring_dimension, next_w_len);
    if !include_stage1 {
        append_zk_stage2_pad_slots::<F, L>(rounds, out, rng);
        return Ok(());
    }
    let b = 1usize << params.log_basis;
    for shape in stage1_tree_stage_shapes(rounds, b) {
        let stored_coeffs =
            EqFactoredUniPoly::<L>::stored_coeff_count_for_degree(shape.sumcheck_proof.1);
        for _ in 0..shape.sumcheck_proof.0 * stored_coeffs {
            push_random_ext_scalar_slots::<F, L>(out, rng);
        }
        for _ in 0..shape.child_claims {
            push_random_ext_scalar_slots::<F, L>(out, rng);
        }
    }
    append_zk_stage2_pad_slots::<F, L>(rounds, out, rng);
    Ok(())
}

#[cfg(feature = "zk")]
fn append_zk_extension_reduction_slots<F, L>(
    partials: usize,
    rounds: usize,
    out: &mut Vec<F>,
    rng: &mut OsRng,
) where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    let round_coeffs = akita_types::EXTENSION_OPENING_REDUCTION_DEGREE;
    for _ in 0..(partials + rounds * round_coeffs) {
        push_random_ext_scalar_slots::<F, L>(out, rng);
    }
}

#[cfg(feature = "zk")]
fn build_zk_hiding_context<F, E, L, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    schedule: &Schedule,
    root_commit_params: &LevelParams,
    num_vars: usize,
    num_claims: usize,
    _num_root_points: usize,
) -> Result<(ZkHidingCommitment<F>, ZkHidingProverState<F>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F>,
    L: FpExtEncoding<F> + ExtField<F>,
    B: ProverComputeBackend<F>,
{
    let mut rng = OsRng;
    let fold_steps = schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(fold) => Some(fold),
            Step::Direct(_) => None,
        })
        .collect::<Vec<_>>();
    let mut hiding_witness = Vec::new();

    if root_tensor_projection_enabled::<F, E, L, D>(num_vars) {
        let split_bits = <L as ExtField<F>>::EXT_DEGREE.trailing_zeros() as usize;
        append_zk_extension_reduction_slots::<F, L>(
            num_claims * <L as ExtField<F>>::EXT_DEGREE,
            num_vars - split_bits,
            &mut hiding_witness,
            &mut rng,
        );
    }
    if let Some(root_step) = fold_steps.first() {
        // Terminal folds skip Stage 1 and consume only Stage 2 pads.
        let root_has_stage1 = fold_steps.len() > 1;
        append_zk_level_pad_slots::<F, L>(
            &root_step.params,
            root_step.next_w_len,
            root_has_stage1,
            &mut hiding_witness,
            &mut rng,
        )?;
        if fold_steps.len() > 1 {
            // Root fold scalar: added to the root level's final next-witness
            // evaluation claim (`w_eval`) after Stage 2. Terminal roots have
            // no next witness and therefore consume no next-w eval mask.
            push_random_ext_scalar_slots::<F, L>(&mut hiding_witness, &mut rng);
        }
        let mut current_opening_vars =
            sumcheck_rounds(root_step.params.ring_dimension, root_step.next_w_len);
        for (step_idx, step) in fold_steps.iter().enumerate().skip(1) {
            if <L as ExtField<F>>::EXT_DEGREE > 1 {
                let split_bits = <L as ExtField<F>>::EXT_DEGREE.trailing_zeros() as usize;
                append_zk_extension_reduction_slots::<F, L>(
                    <L as ExtField<F>>::EXT_DEGREE,
                    current_opening_vars - split_bits,
                    &mut hiding_witness,
                    &mut rng,
                );
            }
            // Terminal recursive folds skip Stage 1 and consume only Stage 2 pads.
            let include_stage1 = step_idx + 1 < fold_steps.len();
            append_zk_level_pad_slots::<F, L>(
                &step.params,
                step.next_w_len,
                include_stage1,
                &mut hiding_witness,
                &mut rng,
            )?;
            if include_stage1 {
                // Recursive fold scalar: added to non-terminal levels' final
                // next-witness evaluation claim (`w_eval`) after Stage 2.
                push_random_ext_scalar_slots::<F, L>(&mut hiding_witness, &mut rng);
            }
            current_opening_vars = sumcheck_rounds(step.params.ring_dimension, step.next_w_len);
        }
    }
    let (u_blind, b_blinding_digits) = commit_zk_hiding_witness::<F, B, D>(
        backend,
        prepared,
        root_commit_params,
        &hiding_witness,
    )?;
    Ok((
        ZkHidingCommitment {
            u_blind,
            b_blinding_digits,
        },
        ZkHidingProverState::new(hiding_witness),
    ))
}
