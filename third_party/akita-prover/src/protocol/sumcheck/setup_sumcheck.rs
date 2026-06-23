//! Setup-product sumcheck for a dense table against two disjoint factors.
//!
//! The table is laid out as `left * right_len + right`. The right factor is
//! bound first, then the left factor. This matches setup products of the form
//! `S(lambda, y) * omega(lambda) * alpha(y)` without materializing the full
//! `omega(lambda) * alpha(y)` table.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_algebra::uni_poly::UniPoly;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::{labels::ABSORB_SETUP_PREFIX_SLOT, Transcript};
use akita_types::{
    gadget_row_scalars, select_setup_prefix_slot, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    RingRelationInstance, SetupContributionPlan, SetupContributionPlanInputs,
    SetupPrefixProverRegistry, SETUP_OFFLOAD_D_SETUP, SETUP_SUMCHECK_DEGREE,
};

/// Proves `sum_{l,r} table[l,r] * left_factor[l] * right_factor[r]`.
pub struct SetupSumcheckProver<E: FieldCore> {
    table: Vec<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

/// Output of the setup product sumcheck prover.
pub struct SetupSumcheckProverOutput<E: FieldCore> {
    /// Claimed setup contribution fed into the stage-2 final row evaluation.
    pub claim: E,
    /// Degree-two product sumcheck over `S(lambda, y) * omega(lambda) * alpha(y)`.
    pub sumcheck: SumcheckProof<E>,
}

impl<E: FieldCore> SetupSumcheckProver<E> {
    /// Construct a factored product-sumcheck prover.
    ///
    /// # Errors
    ///
    /// Returns an error if factor lengths are not powers of two, are empty, or
    /// if `table.len() != left_factor.len() * right_factor.len()`.
    fn new(table: Vec<E>, left_factor: Vec<E>, right_factor: Vec<E>) -> Result<Self, AkitaError> {
        if left_factor.is_empty()
            || right_factor.is_empty()
            || !left_factor.len().is_power_of_two()
            || !right_factor.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "factored product dimensions must be non-empty powers of two".to_string(),
            ));
        }
        let expected_len = left_factor
            .len()
            .checked_mul(right_factor.len())
            .ok_or_else(|| AkitaError::InvalidInput("factored product size overflow".into()))?;
        if table.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: table.len(),
            });
        }

        let input_claim = product_claim(&table, &left_factor, &right_factor);
        let right_rounds = right_factor.len().trailing_zeros() as usize;
        let total_rounds = right_rounds + left_factor.len().trailing_zeros() as usize;
        Ok(Self {
            table,
            left_factor,
            right_factor,
            input_claim,
            right_rounds,
            total_rounds,
        })
    }

    /// Prove the setup-product sumcheck from packed setup rings and the
    /// ring-switch row evaluation that determines the factored weights.
    ///
    /// Internally derives the `(required, bar_omega, alpha_pows)` factored
    /// terms from the level parameters and ring relation, then builds the
    /// `lambda * D + y` setup table (zero-padded up to the next power-of-two
    /// lambda dimension) and runs the product sumcheck. The factored weights
    /// come entirely from the ring-switch row evaluation (`bar_omega`) and the
    /// `alpha` ring challenge; the public per-claim row coefficients are
    /// already folded into the relation upstream, so this product does not take
    /// them as a separate input.
    ///
    /// # Errors
    ///
    /// Returns an error if the ring-switch row evaluation fails, the setup
    /// slice is too small, the padded table size overflows, factor dimensions
    /// are invalid, or any sumcheck round polynomial exceeds its degree bound.
    #[allow(clippy::too_many_arguments)]
    pub fn prove<F, T, SampleRound, const D: usize>(
        expanded: &AkitaExpandedSetup<F>,
        prefix_slots: &SetupPrefixProverRegistry<F, D>,
        lp: &LevelParams,
        next_fold_level_params: &LevelParams,
        relation: &RingRelationInstance<F, D>,
        tau1: &[E],
        alpha: E,
        x_challenges: &[E],
        transcript: &mut T,
        sample_round: SampleRound,
    ) -> Result<SetupSumcheckProverOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + AkitaSerialize,
        T: Transcript<F>,
        SampleRound: FnMut(&mut T) -> E,
    {
        let (required, mut bar_omega, alpha_pows) =
            prepare_setup_sumcheck_terms::<F, E, D>(lp, relation, tau1, alpha, x_challenges)?;

        let natural_field_len = required.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidSetup("setup product natural field length overflow".to_string())
        })?;
        let setup_len = expanded.shared_matrix().total_ring_elements_at::<D>()?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected setup product".to_string(),
            ));
        }
        let setup_eval_len = if D == SETUP_OFFLOAD_D_SETUP {
            let setup_prefix_selection = select_setup_prefix_slot(
                expanded.seed(),
                setup_len,
                |slot_id| {
                    prefix_slots
                        .get(slot_id)
                        .map(|slot| (slot, slot.natural_len, slot.padded_len))
                },
                next_fold_level_params,
                natural_field_len,
                D,
                "selected setup-prefix slot does not cover setup product",
            )?;
            if let Some((slot, setup_eval_len)) = setup_prefix_selection {
                transcript.append_serde(ABSORB_SETUP_PREFIX_SLOT, &slot.id);
                setup_eval_len
            } else {
                setup_len
            }
        } else {
            setup_len
        };
        let setup_view = expanded.shared_matrix().ring_view::<D>(1, setup_eval_len)?;
        let setup_entries = setup_view.as_slice();

        let lambda_len = required.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup("setup product lambda length overflow".into())
        })?;
        bar_omega.resize(lambda_len, E::zero());

        let table_len = lambda_len.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidSetup("setup product table length overflow".into())
        })?;
        let mut setup_table = vec![E::zero(); table_len];
        cfg_chunks_mut!(&mut setup_table, D)
            .enumerate()
            .for_each(|(lambda, row)| {
                if lambda < required {
                    for (slot, &coeff) in row.iter_mut().zip(setup_entries[lambda].coefficients()) {
                        *slot = E::lift_base(coeff);
                    }
                }
            });

        let mut prover = Self::new(setup_table, bar_omega, alpha_pows)?;
        let claim = prover.input_claim();
        let (sumcheck, _challenges, _final_claim) =
            <Self as SumcheckInstanceProverExt<E>>::prove::<F, T, _>(
                &mut prover,
                transcript,
                sample_round,
            )?;
        Ok(SetupSumcheckProverOutput { claim, sumcheck })
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for SetupSumcheckProver<E> {
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    fn degree_bound(&self) -> usize {
        SETUP_SUMCHECK_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.right_rounds {
            accumulate_right_round(&self.table, &self.left_factor, &self.right_factor)
        } else {
            accumulate_left_round(&self.table, &self.left_factor, self.right_factor[0])
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        if round < self.right_rounds {
            fold_right_round(&mut self.table, &mut self.right_factor, r_round);
        } else {
            fold_left_round(&mut self.table, &mut self.left_factor, r_round);
        }
    }
}

/// Derive the factored product-sumcheck terms `(required, bar_omega, alpha_pows)`
/// from the level parameters and ring relation via the ring-switch row
/// evaluation.
fn prepare_setup_sumcheck_terms<F, E, const D: usize>(
    lp: &LevelParams,
    relation: &RingRelationInstance<F, D>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
) -> Result<(usize, Vec<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    let inputs = create_setup_contribution_inputs::<F, E, D>(relation, lp, tau1)?;
    let num_t_vectors = relation
        .opening_batch()
        .num_polys_per_commitment_group()
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count)
                .ok_or_else(|| AkitaError::InvalidSetup("t-vector count overflow".to_string()))
        })?;
    let fold_gadget = gadget_row_scalars::<F>(
        lp.num_digits_fold(num_t_vectors, F::modulus_bits())?,
        lp.log_basis,
    );
    let layout = relation.segment_layout(lp)?;
    let plan = SetupContributionPlan::prepare(
        &inputs,
        x_challenges,
        None,
        None,
        &fold_gadget,
        layout.offset_e,
        layout.offset_t,
        layout.offset_z,
        layout.offset_u,
    )?;
    let required = plan.required();
    let bar_omega = plan.materialize_bar_omega();
    Ok((required, bar_omega, alpha_pows.to_vec()))
}

/// Build the setup-contribution artifact from prover-owned relation data.
fn create_setup_contribution_inputs<F, E, const D: usize>(
    relation: &RingRelationInstance<F, D>,
    lp: &LevelParams,
    tau1: &[E],
) -> Result<SetupContributionPlanInputs<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    let opening_batch = relation.opening_batch();
    let num_polys_per_commitment_group = opening_batch.num_polys_per_commitment_group();
    let num_claims = relation.opening_batch().num_claims();
    let num_commitment_groups = num_polys_per_commitment_group.len();
    if num_commitment_groups != 1 {
        return Err(AkitaError::InvalidProof);
    }

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    if lp.num_blocks == 0 || !lp.num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".to_string(),
        ));
    }
    if lp.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "block_len must be non-zero".to_string(),
        ));
    }
    if depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "digit depths must be non-zero".to_string(),
        ));
    }

    let num_t_vectors = num_polys_per_commitment_group
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| AkitaError::InvalidSetup("batched t-vector count overflow".to_string()))?;
    let inner_width = lp
        .block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
    if lp.a_key.col_len() < inner_width {
        return Err(AkitaError::InvalidSetup(
            "A-key column width is too small for setup contribution layout".to_string(),
        ));
    }
    let max_point_poly_count = num_polys_per_commitment_group
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let expected_b_width = max_point_poly_count
        .checked_mul(lp.a_key.row_len())
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(lp.num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".to_string()))?;
    // Tiered: the stored first-tier `B'` is the full B width divided by the
    // reuse factor `tier_split` (mirrors the verifier-side check in
    // `akita-verifier`'s `prepare_ring_switch_row_eval_inner`).
    let expected_stored_b_width = if lp.f_key.is_some() {
        expected_b_width.div_ceil(lp.tier_split.max(1))
    } else {
        expected_b_width
    };
    if lp.b_key.col_len() < expected_stored_b_width {
        return Err(AkitaError::InvalidSetup(
            "B-key column width is too small for setup contribution layout".to_string(),
        ));
    }

    let m_row_layout = relation.m_row_layout();
    // Public-output M rows are enforced by the fused trace term, not M itself.
    let num_public_m_rows = 0usize;
    let rows = lp.m_row_count_for(num_commitment_groups, num_public_m_rows, m_row_layout)?;
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    Ok(SetupContributionPlanInputs {
        eq_tau1,
        num_t_vectors,
        num_blocks: lp.num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        block_len: lp.block_len,
        inner_width,
        n_a: lp.a_key.row_len(),
        n_d: lp.d_key.row_len(),
        m_row_layout,
        n_b: lp.b_key.row_len(),
        num_commitment_groups,
        rows,
        num_polys_per_commitment_group: num_polys_per_commitment_group.to_vec(),
        num_public_rows: num_public_m_rows,
        // Stage-3 (recursive setup-contribution mode) tiered support is a
        // follow-up; the default Direct verifier path uses `eval_at_point`.
        tier_split: lp.tier_split,
        n_f: lp.f_key.as_ref().map_or(0, |fk| fk.row_len()),
    })
}

fn product_claim<E: FieldCore>(table: &[E], left_factor: &[E], right_factor: &[E]) -> E {
    let right_len = right_factor.len();
    cfg_fold_reduce!(
        0..left_factor.len(),
        E::zero,
        |mut acc, left_idx| {
            let left_weight = left_factor[left_idx];
            let row = &table[left_idx * right_len..(left_idx + 1) * right_len];
            for (&value, &right_weight) in row.iter().zip(right_factor.iter()) {
                acc += value * left_weight * right_weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

fn accumulate_right_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_factor: &[E],
) -> (E, E, E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    cfg_fold_reduce!(
        0..left_factor.len(),
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), left_idx| {
            let left_weight = left_factor[left_idx];
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                let s0 = table[row_base + 2 * pair_idx];
                let s1 = table[row_base + 2 * pair_idx + 1];
                let f0 = left_weight * right_factor[2 * pair_idx];
                let f1 = left_weight * right_factor[2 * pair_idx + 1];
                let ds = s1 - s0;
                let df = f1 - f0;
                constant += s0 * f0;
                linear += s0 * df + ds * f0;
                quadratic += ds * df;
            }
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

fn accumulate_left_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_weight: E,
) -> (E, E, E) {
    let half = left_factor.len() / 2;
    cfg_fold_reduce!(
        0..half,
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), pair_idx| {
            let s0 = table[2 * pair_idx];
            let s1 = table[2 * pair_idx + 1];
            let f0 = left_factor[2 * pair_idx] * right_weight;
            let f1 = left_factor[2 * pair_idx + 1] * right_weight;
            let ds = s1 - s0;
            let df = f1 - f0;
            constant += s0 * f0;
            linear += s0 * df + ds * f0;
            quadratic += ds * df;
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

fn fold_pair<E: FieldCore>(left: E, right: E, r: E) -> E {
    left + r * (right - left)
}

fn fold_right_round<E: FieldCore>(table: &mut Vec<E>, right_factor: &mut Vec<E>, r: E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    let left_len = table.len() / right_len;
    let mut folded = vec![E::zero(); left_len * half];
    cfg_chunks_mut!(&mut folded, half)
        .enumerate()
        .for_each(|(left_idx, row)| {
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                row[pair_idx] = fold_pair(
                    table[row_base + 2 * pair_idx],
                    table[row_base + 2 * pair_idx + 1],
                    r,
                );
            }
        });
    let folded_right = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(right_factor[2 * idx], right_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *right_factor = folded_right;
    *table = folded;
}

fn fold_left_round<E: FieldCore>(table: &mut Vec<E>, left_factor: &mut Vec<E>, r: E) {
    let half = left_factor.len() / 2;
    let folded_table = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(table[2 * idx], table[2 * idx + 1], r))
        .collect::<Vec<_>>();
    let folded_left = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(left_factor[2 * idx], left_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *table = folded_table;
    *left_factor = folded_left;
}
