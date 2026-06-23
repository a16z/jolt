use super::*;

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw [`build_w_coeffs`] order:
/// `w[x * y_len + y]`, with x outer and y inner.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub fn build_w_evals_compact(
    w: &[i8],
    d: usize,
    extension_degree: usize,
) -> Result<(Vec<i8>, usize, usize), AkitaError> {
    if !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_x_cols = w.len() / d;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    if extension_degree == 1 {
        let ring_bits = d.trailing_zeros() as usize;
        return Ok((w.to_vec(), col_bits, ring_bits));
    }
    let packed_len = d / extension_degree;
    if packed_len == 0 || !packed_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "packed recursive witness has invalid slot count".to_string(),
        ));
    }
    let half = d / (2 * extension_degree);
    let mut compact = Vec::with_capacity(live_x_cols * packed_len);
    for ring in w.chunks_exact(d) {
        compact.extend_from_slice(&ring[..half]);
        compact.extend((half..packed_len).map(|low| ring[d / 2 + low - half]));
    }
    Ok((compact, col_bits, packed_len.trailing_zeros() as usize))
}

/// Unified M-table evaluation for the batched CWSS protocol.
///
/// All claims share one ring-level opening point. Commitment-group routing may
/// address multiple committed bundles.
///
/// # Errors
///
/// Returns an error if the batch shape, opening-point layout, challenge count,
/// or expanded matrix dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_m_evals_x_batched")]
pub fn compute_m_evals_x<F, E, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F, D>,
    challenges: &Challenges,
    alpha: E,
    alpha_pows: &[E],
    lp: &LevelParams,
    tau1: &[E],
    num_polys_per_commitment_group: &[usize],
    claim_to_commitment_group: &[usize],
    claim_poly_in_commitment_group: &[usize],
    gamma: &[E],
    num_public_rows: usize,
    m_row_layout: MRowLayout,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let num_claims = claim_to_commitment_group.len();
    if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
        return Err(AkitaError::InvalidInput(
            "batched prover opening-point layout mismatch".to_string(),
        ));
    }
    if ring_multiplier_point.a_len() < lp.block_len
        || ring_multiplier_point.b_len() != lp.num_blocks
    {
        return Err(AkitaError::InvalidInput(
            "batched prover ring-multiplier opening-point layout mismatch".to_string(),
        ));
    }
    if claim_to_commitment_group.len() != num_claims
        || claim_poly_in_commitment_group.len() != num_claims
    {
        return Err(AkitaError::InvalidInput(
            "batched prover commitment routing lengths do not match".to_string(),
        ));
    }
    let num_commitment_groups = num_polys_per_commitment_group.len();
    for claim_idx in 0..num_claims {
        let group_idx = claim_to_commitment_group[claim_idx];
        if group_idx >= num_commitment_groups
            || claim_poly_in_commitment_group[claim_idx]
                >= num_polys_per_commitment_group[group_idx]
        {
            return Err(AkitaError::InvalidInput(
                "batched prover commitment routing index out of range".to_string(),
            ));
        }
    }

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    let num_t_vectors = num_polys_per_commitment_group
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| AkitaError::InvalidSetup("batched t-vector count overflow".to_string()))?;
    let t_vector_to_group: Vec<(usize, usize)> = num_polys_per_commitment_group
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_poly_count)| {
            (0..group_poly_count).map(move |poly_idx| (group_idx, poly_idx))
        })
        .collect();
    // Per-commitment-group t-vector starting indices; precomputed so the
    // per-claim mapping below stays O(groups + claims).
    let t_vector_offsets: Vec<usize> = num_polys_per_commitment_group
        .iter()
        .scan(0usize, |acc, &count| {
            let offset = *acc;
            *acc += count;
            Some(offset)
        })
        .collect();
    let claim_to_t_vector: Vec<usize> = claim_to_commitment_group
        .iter()
        .zip(claim_poly_in_commitment_group.iter())
        .map(|(&group_idx, &poly_idx)| t_vector_offsets[group_idx] + poly_idx)
        .collect();

    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    let t_total_blocks = num_blocks
        .checked_mul(num_t_vectors)
        .ok_or_else(|| AkitaError::InvalidSetup("batched t block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let block_len = lp.block_len;
    let w_len = depth_open * total_blocks;
    let n_a = lp.a_key.row_len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    // Terminal layout drops the D-block from the M-matrix entirely; offsets
    // and per-row gates must use 0 for the n_d position.
    let n_d_active = match m_row_layout {
        MRowLayout::WithDBlock => n_d,
        MRowLayout::WithoutDBlock => 0,
    };
    let t_len = depth_open * n_a * t_total_blocks;
    #[cfg(feature = "zk")]
    let d_blinding_segment_len = match m_row_layout {
        MRowLayout::WithDBlock => {
            akita_types::zk::blinding_digit_plane_count::<F>(n_d, D, log_basis)
        }
        // Terminal omits the D-block, so its blinding columns vanish too.
        MRowLayout::WithoutDBlock => 0,
    };
    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_point =
        akita_types::zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_segment_len = num_commitment_groups
        .checked_mul(b_blinding_digit_planes_per_point)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK blinding width overflow".to_string()))?;
    let inner_width = block_len * depth_commit;
    let z_base_len = inner_width;
    let z_len = depth_fold
        .checked_mul(z_base_len)
        .ok_or_else(|| AkitaError::InvalidSetup("batched z width overflow".to_string()))?;
    let rows = lp.m_row_count_for(num_commitment_groups, num_public_rows, m_row_layout)?;
    let levels = r_decomp_levels::<F>(log_basis);
    // Tiered `û_concat` segment column count (flat, after `t̂`); `0` single-tier.
    let u_seg_len = if lp.f_key.is_some() {
        num_commitment_groups
            .checked_mul(lp.tier_split)
            .and_then(|w| w.checked_mul(n_b))
            .and_then(|w| w.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("tiered û segment overflow".to_string()))?
    } else {
        0
    };
    #[cfg(feature = "zk")]
    let total_cols = w_len
        .checked_add(d_blinding_segment_len)
        .and_then(|cols| cols.checked_add(t_len))
        .and_then(|cols| cols.checked_add(u_seg_len))
        .and_then(|cols| cols.checked_add(b_blinding_segment_len))
        .and_then(|cols| cols.checked_add(z_len))
        .and_then(|cols| cols.checked_add(rows.checked_mul(levels)?))
        .ok_or_else(|| AkitaError::InvalidSetup("expanded M width overflow".to_string()))?;
    #[cfg(not(feature = "zk"))]
    let total_cols = w_len
        .checked_add(t_len)
        .and_then(|cols| cols.checked_add(u_seg_len))
        .and_then(|cols| cols.checked_add(z_len))
        .and_then(|cols| cols.checked_add(rows.checked_mul(levels)?))
        .ok_or_else(|| AkitaError::InvalidSetup("expanded M width overflow".to_string()))?;

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let g1_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let g1_commit: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let x_len = total_cols.next_power_of_two();
    let mut out = Vec::with_capacity(x_len);

    let c_alphas: Vec<E> = match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => sparse
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
            .collect::<Result<_, _>>()?,
        Challenges::Tensor { factored: _ } => challenges.evals_at_pows::<F, E, D>(alpha_pows)?,
    };

    let max_group_poly_count = num_polys_per_commitment_group
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let d_message_width = total_blocks
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
    let d_width = d_message_width;
    let t_cols_per_vector = n_a
        .checked_mul(depth_open)
        .and_then(|len| len.checked_mul(num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B setup vector width overflow".to_string()))?;
    let b_message_width = max_group_poly_count
        .checked_mul(t_cols_per_vector)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup width overflow".to_string()))?;
    // Tiered: the stored first-tier matrix is `B'` of width `b_message_width /
    // tier_split`, reused across `tier_split` column-slices; the second-tier
    // `F` (`f_view`) commits the decomposed concatenated images `û_concat`.
    let tiered = lp.f_key.is_some();
    let tier_split = lp.tier_split;
    let n_f = lp.f_key.as_ref().map_or(0, |fk| fk.row_len());
    let b_width = if tiered {
        if tier_split == 0 || !b_message_width.is_multiple_of(tier_split) {
            return Err(AkitaError::InvalidSetup(
                "tiered B' width does not divide the per-group B width".to_string(),
            ));
        }
        b_message_width / tier_split
    } else {
        b_message_width
    };
    let width_f = if tiered {
        tier_split
            .checked_mul(n_b)
            .and_then(|w| w.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("tiered F width overflow".to_string()))?
    } else {
        0
    };
    let a_width = inner_width;
    let d_view = setup.shared_matrix.ring_view::<D>(n_d, d_width)?;
    let b_view = setup.shared_matrix.ring_view::<D>(n_b, b_width)?;
    let a_view = setup.shared_matrix.ring_view::<D>(n_a, a_width)?;
    let d_rows: Vec<_> = d_view.rows().collect();
    let b_rows: Vec<_> = b_view.rows().collect();
    let a_rows: Vec<_> = a_view.rows().collect();
    let f_rows: Vec<_> = if tiered {
        setup
            .shared_matrix
            .ring_view::<D>(n_f, width_f)?
            .rows()
            .collect()
    } else {
        Vec::new()
    };

    // Canonical row layout: consistency (1) | public (num_public_rows) |
    //   D (n_d_active) | COMMIT (F when tiered, else B) | B_inner (tiered) | A.
    let commit_rows_pg = if tiered { n_f } else { n_b };
    let b_inner_rows_pg = if tiered { tier_split * n_b } else { 0 };
    let consistency_weight = eq_tau1[0];
    let public_weights = &eq_tau1[1..(1 + num_public_rows)];
    let d_start = 1 + num_public_rows;
    let f_start = d_start + n_d_active;
    let b_inner_start = f_start + commit_rows_pg * num_commitment_groups;
    let a_start = b_inner_start + b_inner_rows_pg * num_commitment_groups;
    // Non-tiered alias used by the single-tier B-block scan below.
    let b_start = f_start;
    let a_weights = &eq_tau1[a_start..rows];
    let t_compound_per_block = n_a * depth_open;

    let uses_ring_multipliers = ring_multiplier_point.as_base().is_none();
    let row_coefficient_rings = if uses_ring_multipliers {
        Some(
            gamma
                .iter()
                .copied()
                .map(|coefficient| {
                    embed_ring_subfield_scalar::<F, E, D>(
                        coefficient,
                        AkitaError::InvalidInput(
                            "public-row coefficient does not encode in the ring-subfield basis"
                                .to_string(),
                        ),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let public_b_evals = (0..num_claims)
        .map(|claim_idx| {
            let coefficient_ring = row_coefficient_rings
                .as_ref()
                .map(|rings| &rings[claim_idx]);
            (0..num_blocks)
                .map(|block_idx| {
                    ring_multiplier_point.eval_b_with_coefficient(
                        block_idx,
                        gamma[claim_idx],
                        coefficient_ring,
                        alpha_pows,
                    )
                })
                .collect::<Result<Vec<_>, AkitaError>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let w_segment: Vec<E> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let d_phys_col = blk * depth_open + dig;
            let b_eval = public_b_evals[claim_idx][block_idx];
            let public_contrib = if num_public_rows == 0 {
                E::zero()
            } else {
                public_weights[0] * b_eval
            };
            let mut acc = (public_contrib + consistency_weight * c_alphas[blk]) * g1_open[dig];
            // Terminal layout: `n_d_active == 0`, so this loop is empty and
            // the D-block contribution is omitted.
            for (di, eq_i) in eq_tau1[d_start..(d_start + n_d_active)].iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&d_rows[di][d_phys_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    #[cfg(feature = "zk")]
    let d_blinding_segment: Vec<E> = if d_blinding_segment_len == 0 {
        Vec::new()
    } else {
        let d_weights = &eq_tau1[d_start..(d_start + n_d_active)];
        let d_zk_view = setup
            .zk_d_matrix()
            .ring_view::<D>(n_d, d_blinding_segment_len)?;
        let d_zk = d_zk_view.as_slice();
        let d_zk_stride = d_zk_view.num_cols();
        cfg_into_iter!(0..d_blinding_segment_len)
            .map(|local| {
                let mut acc = E::zero();
                for (row_idx, eq_i) in d_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i
                            * eval_ring_at_pows(&d_zk[row_idx * d_zk_stride + local], alpha_pows);
                    }
                }
                acc
            })
            .collect()
    };

    let mut challenge_sums_by_t_block = vec![E::zero(); t_total_blocks];
    for (claim_idx, &t_vector_idx) in claim_to_t_vector.iter().enumerate() {
        let dst_offset = t_vector_idx * num_blocks;
        let src_offset = claim_idx * num_blocks;
        for block_idx in 0..num_blocks {
            challenge_sums_by_t_block[dst_offset + block_idx] += c_alphas[src_offset + block_idx];
        }
    }
    let t_segment: Vec<E> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let compound_dig = x / t_total_blocks;
            let blk = x % t_total_blocks;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            let t_vector_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let (point_idx, poly_idx_within_group) = t_vector_to_group[t_vector_idx];
            let phys_claim_offset =
                block_idx * t_compound_per_block + a_idx * depth_open + digit_idx;
            let local_col = poly_idx_within_group * t_cols_per_vector + phys_claim_offset;
            let mut acc = a_weights[a_idx] * challenge_sums_by_t_block[blk] * g1_open[digit_idx];
            if tiered {
                // B_inner block: the stored B' is reused across `tier_split`
                // slices; `local_col` selects the slice and the within-slice
                // stored-B' column.
                let slice = local_col / b_width;
                let within = local_col % b_width;
                let base = b_inner_start + point_idx * (tier_split * n_b) + slice * n_b;
                for row_idx in 0..n_b {
                    let eq_i = eq_tau1[base + row_idx];
                    if !eq_i.is_zero() {
                        acc += eq_i * eval_ring_at_pows(&b_rows[row_idx][within], alpha_pows);
                    }
                }
            } else {
                let commitment_weights =
                    &eq_tau1[(b_start + point_idx * n_b)..(b_start + (point_idx + 1) * n_b)];
                for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i * eval_ring_at_pows(&b_rows[row_idx][local_col], alpha_pows);
                    }
                }
            }
            acc
        })
        .collect();

    // Tiered `û_concat` segment (flat, contiguous, right after `t̂`): the
    // second-tier `F` commit-block image plus the `B_inner` `-recompose(û)`
    // term, per `û` column. Empty for single-tier.
    let u_segment: Vec<E> = if tiered {
        let u_seg_len = num_commitment_groups * width_f;
        cfg_into_iter!(0..u_seg_len)
            .map(|c| {
                let group = c / width_f;
                let c_in_group = c % width_f;
                let slice_row = c_in_group / depth_open;
                let digit = c_in_group % depth_open;
                let mut acc = E::zero();
                // F (commit) block: F·û_concat.
                for f_row in 0..n_f {
                    let eq_i = eq_tau1[f_start + group * n_f + f_row];
                    if !eq_i.is_zero() {
                        acc += eq_i * eval_ring_at_pows(&f_rows[f_row][c_in_group], alpha_pows);
                    }
                }
                // B_inner RHS: -recompose(û) = -Σ_digit base^digit · û[digit].
                let b_inner_w = eq_tau1[b_inner_start + group * (tier_split * n_b) + slice_row];
                acc -= b_inner_w * g1_open[digit];
                acc
            })
            .collect()
    } else {
        Vec::new()
    };

    #[cfg(feature = "zk")]
    let b_blinding_segment: Vec<E> = if b_blinding_digit_planes_per_point == 0 {
        Vec::new()
    } else {
        // Each commitment group is committed independently with a group-local B
        // input `[group t_hat || group blinding]`; witness segments are
        // point-local but reuse the same stored per-commitment zkB row view.
        let b_zk_view = setup
            .zk_b_matrix()
            .ring_view::<D>(n_b, b_blinding_digit_planes_per_point)?;
        let b_zk = b_zk_view.as_slice();
        let b_zk_stride = b_zk_view.num_cols();
        cfg_into_iter!(0..b_blinding_segment_len)
            .map(|idx| {
                let group_stride = b_blinding_digit_planes_per_point;
                let point_idx = idx / group_stride;
                let local = idx % group_stride;
                let commitment_weights =
                    &eq_tau1[(b_start + point_idx * n_b)..(b_start + (point_idx + 1) * n_b)];
                let mut acc = E::zero();
                for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i
                            * eval_ring_at_pows(&b_zk[row_idx * b_zk_stride + local], alpha_pows);
                    }
                }
                acc
            })
            .collect()
    };

    let z_base: Vec<E> = cfg_into_iter!(0..z_base_len)
        .map(|k| {
            let local_k = k;
            let block_idx = local_k / depth_commit;
            let digit_idx = local_k % depth_commit;
            let a_eval = ring_multiplier_point.eval_a_at::<E>(block_idx, alpha_pows)?;
            let mut acc = consistency_weight * a_eval * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&a_rows[a_idx][local_k], alpha_pows);
                }
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    let z_total_blocks = block_len;
    let z_segment: Vec<E> = cfg_into_iter!(0..z_len)
        .map(|x| {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / depth_fold;
            let df = compound_dig % depth_fold;
            let blk = global_blk % block_len;
            let phys_k = blk * depth_commit + dc;
            -(z_base[phys_k] * fold_gadget[df])
        })
        .collect();

    let alpha_pow_d = alpha_pows[D - 1] * alpha;
    let denom = alpha_pow_d + E::one();
    let r_tail_len = rows * levels;
    let r_tail: Vec<E> = cfg_into_iter!(0..r_tail_len)
        .map(|idx| {
            let row_idx = idx / levels;
            let level_idx = idx % levels;
            -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
        })
        .collect();

    let z_first = akita_types::ring_column_z_first(lp);
    if z_first {
        out.extend(z_segment);
        out.extend(w_segment);
        out.extend(t_segment);
        out.extend(u_segment);
        #[cfg(feature = "zk")]
        out.extend(b_blinding_segment);
        #[cfg(feature = "zk")]
        out.extend(d_blinding_segment);
    } else {
        out.extend(w_segment);
        out.extend(t_segment);
        out.extend(u_segment);
        #[cfg(feature = "zk")]
        out.extend(b_blinding_segment);
        #[cfg(feature = "zk")]
        out.extend(d_blinding_segment);
        out.extend(z_segment);
    }
    out.extend(r_tail);
    out.resize(x_len, E::zero());
    Ok(out)
}
