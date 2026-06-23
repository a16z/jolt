use super::*;

/// Complete the ring switch after the caller has bound the next witness.
///
/// Samples challenges and builds the evaluation tables for the fused sumcheck.
/// The caller must first absorb either the next-witness commitment or the
/// terminal cleartext witness bytes into `transcript`.
///
/// Only the current level's `D` is needed for M-alpha expansion and
/// `alpha_evals_y`.
///
/// # Errors
///
/// Returns an error if the supplied gamma vector does not match the claim
/// count or if matrix expansion or evaluation-table construction fails.
#[tracing::instrument(skip_all, name = "ring_switch_finalize")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize<F, E, T, const D: usize>(
    instance: &RingRelationInstance<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    lp: &LevelParams,
    gamma: Option<&[E]>,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let default_gamma;
    let gamma = if let Some(gamma) = gamma {
        gamma
    } else {
        default_gamma = instance
            .gamma()
            .iter()
            .copied()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        &default_gamma
    };
    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let opening_batch = instance.opening_batch();
    let num_polys_per_commitment_group = opening_batch.num_polys_per_commitment_group();
    let num_commitment_groups = num_polys_per_commitment_group.len();
    let num_public_m_rows = 0usize;

    let num_ring_elems = w.len() / D;
    let live_x_cols = num_ring_elems;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch column count overflow".to_string()))?
        .trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = lp.m_row_count_for(num_commitment_groups, num_public_m_rows, m_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0: Vec<E> = match m_row_layout {
        MRowLayout::WithDBlock => (0..num_sc_vars)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
            .collect(),
        MRowLayout::WithoutDBlock => Vec::new(),
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let ring_alpha_evals_y = scalar_powers(alpha, D);
    let alpha_evals_y = scalar_powers(alpha, D);

    let claim_to_commitment_group = opening_batch.claim_to_commitment_group();
    let claim_poly_in_commitment_group = opening_batch.claim_poly_indices();
    let challenges = &instance.challenges;
    if gamma.len() != instance.opening_batch().num_claims() {
        return Err(AkitaError::InvalidInput(
            "ring-switch gamma length does not match claim count".to_string(),
        ));
    }

    #[cfg(feature = "parallel")]
    let (m_evals_x_result, w_result) = rayon::join(
        || {
            compute_m_evals_x::<F, E, D>(
                setup,
                instance.opening_point(),
                instance.ring_multiplier_point(),
                challenges,
                alpha,
                &ring_alpha_evals_y,
                lp,
                &tau1,
                num_polys_per_commitment_group,
                claim_to_commitment_group,
                claim_poly_in_commitment_group,
                gamma,
                num_public_m_rows,
                m_row_layout,
            )
        },
        || build_w_evals_compact(w.as_i8_digits(), D, 1),
    );
    #[cfg(not(feature = "parallel"))]
    let (m_evals_x_result, w_result) = {
        let m_evals_x = compute_m_evals_x::<F, E, D>(
            setup,
            instance.opening_point(),
            instance.ring_multiplier_point(),
            challenges,
            alpha,
            &ring_alpha_evals_y,
            lp,
            &tau1,
            num_polys_per_commitment_group,
            claim_to_commitment_group,
            claim_poly_in_commitment_group,
            gamma,
            num_public_m_rows,
            m_row_layout,
        )?;
        let w_compact = build_w_evals_compact(w.as_i8_digits(), D, 1);
        (Ok(m_evals_x), w_compact)
    };

    let m_evals_x = m_evals_x_result?;
    let (w_evals_compact, _, _) = w_result?;

    Ok(RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << lp.log_basis,
        alpha,
    })
}
