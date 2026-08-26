use super::*;

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field arithmetic in the body.
pub(super) fn add_stage1<F, PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    mut builder: Builder<F, VC::Output>,
    values: &mut SourceValues<F>,
) -> Result<Builder<F, VC::Output>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone,
{
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let uniskip_rounds = 1;
    let uniskip_degree = SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE;
    let uniskip_domain = SumcheckDomain::centered_integer(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE);
    builder = add_stage(
        builder,
        "stage1.outer_uniskip",
        SumcheckStatement::new(uniskip_rounds, uniskip_degree),
        domain_spec(uniskip_domain),
        input.stage1.uniskip_consistency.clone(),
        &input.stage1.uniskip_output_claims,
        values,
        vec![outer_uniskip_opening().into()],
        Vec::new(),
        Vec::new(),
        VerifierExpr::zero(),
        opening(outer_uniskip_opening()),
    )?;

    // The remainder sumcheck point is opening-derived: for the singleton remainder
    // batch the committed round challenges are the raw (un-reversed) point that the
    // clear path obtains from the bound remainder reduction.
    let remainder_challenges = input.stage1.remainder_consistency.challenges();
    let remainder_formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
        tau: &input.stage1.challenges.tau,
        uniskip: input.stage1.challenges.uniskip_challenge,
        remainder: &remainder_challenges,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanOuter,
        reason: error.to_string(),
    })?;
    // Under `field-inline` this coefficient table is the COMPOSED one (48
    // columns): the same source the clear path's factored check consumes, so
    // the appended FR weight publics bake automatically.
    for (id, value) in remainder_formula.public_coefficients() {
        values.public(VerifierPublicId::SpartanOuter(id), value)?;
    }

    // The composed opening row order: the 35 ordinary openings in canonical
    // order, then (under `field-inline`) the 13 FR-local columns in
    // appended-column order — the clear path's absorb order exactly.
    let opening_ids = stage1_spartan_outer_opening_ids(&dimensions);

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "log_t is an ilog2 result (< 64); 1 + log_t cannot overflow usize"
    )]
    let remainder_rounds = 1 + log_t;
    let remainder_domain = SumcheckDomain::BooleanHypercube;
    let [remainder_batching_coefficient] = input
        .stage1
        .remainder_consistency
        .batching_coefficients
        .as_slice()
    else {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: "stage1.outer_remainder: expected one batching coefficient".to_string(),
        });
    };
    let remainder_extra_vars = input
        .stage1
        .remainder_consistency
        .max_num_vars
        .checked_sub(remainder_rounds)
        .ok_or_else(|| VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage1.outer_remainder: {remainder_rounds} rounds exceed the batch's {} variables",
                input.stage1.remainder_consistency.max_num_vars
            ),
        })?;
    let input_claim = scale_expr(
        opening(outer_uniskip_opening()),
        *remainder_batching_coefficient * F::pow2(remainder_extra_vars),
    );
    let output_claim = scale_expr(
        stage1_spartan_outer_output_expr(&opening_ids),
        *remainder_batching_coefficient,
    );
    add_stage(
        builder,
        "stage1.outer_remainder",
        SumcheckStatement::new(
            input.stage1.remainder_consistency.max_num_vars,
            input.stage1.remainder_consistency.max_degree,
        ),
        domain_spec(remainder_domain),
        input.stage1.remainder_consistency.consistency.clone(),
        &input.stage1.remainder_output_claims,
        values,
        opening_ids,
        Vec::new(),
        Vec::new(),
        input_claim,
        output_claim,
    )
}

/// The composed stage-1 committed opening row order: the 35 ordinary
/// Spartan-outer openings in canonical (`dimensions.variables()`) order,
/// then — under `field-inline` — the 13 FR-local openings in appended-column
/// order. This is exactly `stage1::verify`'s absorb/commit order.
pub(super) fn stage1_spartan_outer_opening_ids(
    dimensions: &SpartanOuterDimensions,
) -> Vec<VerifierOpeningId> {
    #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
    let mut opening_ids: Vec<VerifierOpeningId> = dimensions
        .variables()
        .iter()
        .copied()
        .map(|variable| outer_opening(variable).into())
        .collect();
    #[cfg(feature = "field-inline")]
    opening_ids.extend(super::field_inline::stage1_appended_opening_ids());
    opening_ids
}

/// The composed factored quadratic form, mirroring the clear path's
/// `factored_output_claim`: `TauKernel · (AzConstant + Σ AzWeight(i)·o_i) ·
/// (BzConstant + Σ BzWeight(i)·o_i)` over the FULL composed opening vector —
/// the 35 ordinary openings followed (under `field-inline`) by the 13 FR-local
/// openings, weighted by the SAME `JoltSpartanOuterRemainder` coefficient
/// table at the appended indices.
pub(super) fn stage1_spartan_outer_output_expr<F: JoltField>(
    opening_ids: &[VerifierOpeningId],
) -> VerifierExpr<F> {
    let mut az = VerifierExpr::zero();
    let mut bz = VerifierExpr::zero();
    for (index, id) in opening_ids.iter().copied().enumerate() {
        az = az
            + derived(VerifierPublicId::SpartanOuter(
                JoltSpartanOuterPublic::AzWeight(index),
            )) * opening(id);
        bz = bz
            + derived(VerifierPublicId::SpartanOuter(
                JoltSpartanOuterPublic::BzWeight(index),
            )) * opening(id);
    }
    az = az
        + derived(VerifierPublicId::SpartanOuter(
            JoltSpartanOuterPublic::AzConstant,
        ));
    bz = bz
        + derived(VerifierPublicId::SpartanOuter(
            JoltSpartanOuterPublic::BzConstant,
        ));
    derived(VerifierPublicId::SpartanOuter(
        JoltSpartanOuterPublic::TauKernel,
    )) * az
        * bz
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use std::collections::BTreeMap;

    /// Resolve one of the coefficient-table publics from its `(id, value)`
    /// list; ids outside the SpartanOuter family resolve to zero (the stage-1
    /// output expression never names them).
    fn resolve_public(publics: &[(JoltSpartanOuterPublic, Fr)], id: &VerifierPublicId) -> Fr {
        match id {
            VerifierPublicId::SpartanOuter(public) => publics
                .iter()
                .find(|(candidate, _)| candidate == public)
                .map_or_else(Fr::zero, |(_, value)| *value),
            VerifierPublicId::Jolt(_) | VerifierPublicId::Challenge(_) => Fr::zero(),
            #[cfg(feature = "field-inline")]
            VerifierPublicId::FieldInline(_) | VerifierPublicId::FieldInlineChallenge(_) => {
                Fr::zero()
            }
        }
    }

    /// The composed opening row order is the clear absorb order: the 35
    /// ordinary openings in canonical order, then (FR-on) the 13 FR-local
    /// openings in appended-column order, matching the composed jolt-r1cs
    /// column count.
    #[test]
    fn stage1_opening_ids_follow_the_composed_column_order() {
        let dimensions = SpartanOuterDimensions::rv64(3);
        let ids = stage1_spartan_outer_opening_ids(&dimensions);

        let expected_len = jolt_r1cs::constraints::jolt::spartan_outer_opening_columns().len();
        assert_eq!(ids.len(), expected_len);
        #[cfg(not(feature = "field-inline"))]
        assert_eq!(ids.len(), 35);
        #[cfg(feature = "field-inline")]
        assert_eq!(ids.len(), 48);

        let ordinary: Vec<VerifierOpeningId> = dimensions
            .variables()
            .iter()
            .copied()
            .map(|variable| outer_opening(variable).into())
            .collect();
        assert_eq!(ids.get(..ordinary.len()), Some(ordinary.as_slice()));
        #[cfg(feature = "field-inline")]
        {
            let appended: Vec<VerifierOpeningId> =
                jolt_claims::protocols::field_inline::geometry::spartan::outer_output_openings()
                    .into_iter()
                    .map(VerifierOpeningId::from)
                    .collect();
            assert_eq!(ids.get(ordinary.len()..), Some(appended.as_slice()));
        }
    }

    /// The lowered composed output expression evaluates bit-identically to the
    /// composed factored form `JoltSpartanOuterRemainder::expected_output_claim`
    /// over the full selected opening vector — the same equation the clear
    /// stage-1 path checks (FR-on: 48 openings; FR-off: the rv64 35).
    #[test]
    fn lowered_output_expr_matches_the_composed_factored_form() {
        let log_t = 3usize;
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let tau: Vec<Fr> = (0..log_t as u64 + 2).map(|i| Fr::from_u64(2 + i)).collect();
        let remainder: Vec<Fr> = (0..=(log_t as u64))
            .map(|i| Fr::from_u64(100 + i))
            .collect();
        let uniskip = Fr::from_u64(17);

        let formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
            tau: &tau,
            uniskip,
            remainder: &remainder,
        })
        .unwrap();
        let opening_ids = stage1_spartan_outer_opening_ids(&dimensions);
        let openings: Vec<Fr> = (0..opening_ids.len() as u64)
            .map(|i| Fr::from_u64(1_000 + i))
            .collect();
        let factored = formula.expected_output_claim(&openings).unwrap();

        let publics = formula.public_coefficients();
        let opening_values: BTreeMap<VerifierOpeningId, Fr> = opening_ids
            .iter()
            .copied()
            .zip(openings.iter().copied())
            .collect();
        let lowered = stage1_spartan_outer_output_expr::<Fr>(&opening_ids).evaluate(
            |id| opening_values.get(id).copied().unwrap(),
            |_| Fr::zero(),
            |id| resolve_public(&publics, id),
        );

        assert_eq!(lowered, factored);
    }
}
