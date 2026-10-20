use jolt_claims::protocols::jolt::relations::spartan::OuterUniskip;

use super::*;

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field arithmetic in the body.
pub(super) fn add_stage1<F, PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    mut builder: ResidualBuilder<F>,
    values: &mut SourceValues<F>,
) -> Result<ResidualBuilder<F>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
{
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    builder = add_uniskip_stage(
        builder,
        "stage1.outer_uniskip",
        UniskipRound {
            degree: SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
            domain: SumcheckDomain::centered_integer(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE),
            rounds: &input.stage1.uniskip_rounds,
            output_claims: &input.stage1.uniskip_output_claims,
        },
        values,
        relation_claim(&OuterUniskip::new(dimensions.clone())),
    )?;

    // The remainder sumcheck point is opening-derived: for the singleton remainder
    // batch the masked round challenges are the raw (un-reversed) point that the
    // clear path obtains from the bound remainder reduction.
    let remainder_challenges = input.stage1.remainder_rounds.challenges();
    let remainder_formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
        tau: &input.stage1.challenges.tau,
        uniskip: input.stage1.challenges.uniskip_challenge,
        remainder: &remainder_challenges,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanOuter,
        reason: error.to_string(),
    })?;
    // Under `field-inline` this coefficient table is the COMPOSED one (50 columns): the same
    // source the clear path's factored check consumes, so the appended field-inline weight
    // publics resolve automatically.
    for (id, value) in remainder_formula.public_coefficients() {
        values.public(VerifierPublicId::SpartanOuter(id), value)?;
    }

    add_batched_stage(
        builder,
        "stage1.outer_remainder",
        SumcheckDomain::BooleanHypercube,
        &[stage1_spartan_outer_claim(&dimensions)],
        &input.stage1.remainder_rounds,
        &input.stage1.remainder_output_claims,
        values,
    )
}

fn stage1_spartan_outer_claim<F: JoltField>(dimensions: &SpartanOuterDimensions) -> MemberClaim<F> {
    use crate::stages::relations::SymbolicOf;
    use crate::stages::stage1::outer_remainder::OuterRemainder;
    relation_claim(&SymbolicOf::<F, OuterRemainder<F>>::new(dimensions.clone()))
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;
    use crate::stages::relations::MapCells;
    use crate::stages::stage1::outer_remainder::OuterRemainder;
    use crate::stages::stage1::outputs::Stage1BatchSumchecks;
    use core::convert::Infallible;
    use jolt_algebra::{Fr, Ring};
    use jolt_claims::protocols::jolt::geometry::spartan::outer_opening;
    use num_traits::Zero;
    use std::collections::BTreeMap;

    fn composed<I: Into<ComposedOpeningId>>(id: I) -> ComposedOpeningId {
        id.into()
    }

    /// The stage-1 output cell ids, in canonical (wire) order.
    fn stage1_committed_ids(log_t: usize) -> Vec<ComposedOpeningId> {
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let tau = vec![Fr::from_u64(2); log_t.saturating_add(2)];
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(dimensions, tau, Fr::from_u64(17)),
        };
        let challenges: Vec<Fr> = (0..=(log_t as u64)).map(Fr::from_u64).collect();
        let points = sumchecks
            .derive_opening_points(&challenges, &sumchecks.empty_input_points())
            .unwrap();
        let mut ids = Vec::new();
        let _: Result<_, Infallible> = points.outer_remainder.try_map_cells(&mut |id, _| {
            ids.push(composed(*id));
            Ok(Fr::zero())
        });
        ids
    }

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

    /// The output cell order is the composed R1CS column order the factored
    /// output formula consumes: the common value/flag openings in canonical
    /// order, then (with field-inline enabled) the five field value/product
    /// openings in appended-column order.
    #[test]
    fn stage1_opening_ids_follow_the_composed_column_order() {
        let dimensions = SpartanOuterDimensions::rv64(3);
        let ids = stage1_committed_ids(3);

        let expected_len =
            jolt_claims::protocols::composed::r1cs::spartan_outer_opening_columns().len();
        assert_eq!(ids.len(), expected_len);
        #[cfg(not(feature = "field-inline"))]
        assert_eq!(ids.len(), 35);
        #[cfg(feature = "field-inline")]
        assert_eq!(ids.len(), 50);

        let ordinary: Vec<ComposedOpeningId> = dimensions
            .variables()
            .iter()
            .copied()
            .map(|variable| outer_opening(variable).into())
            .collect();
        assert_eq!(ids.get(..ordinary.len()), Some(ordinary.as_slice()));
        #[cfg(feature = "field-inline")]
        {
            let appended: Vec<ComposedOpeningId> =
                jolt_claims::protocols::field_inline::geometry::spartan::outer_output_openings()
                    .into_iter()
                    .map(ComposedOpeningId::from)
                    .collect();
            assert_eq!(ids.get(ordinary.len()..), Some(appended.as_slice()));
        }
    }

    /// The lowered composed output expression evaluates bit-identically to the composed
    /// factored form `JoltSpartanOuterRemainder::expected_output_claim` over the full selected
    /// opening vector — the same equation the clear stage-1 path checks (with field-inline
    /// enabled: 50 openings; with field-inline disabled: the rv64 35).
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
        let opening_ids = stage1_committed_ids(log_t);
        let openings: Vec<Fr> = (0..opening_ids.len() as u64)
            .map(|i| Fr::from_u64(1_000 + i))
            .collect();
        let factored = formula.expected_output_claim(&openings).unwrap();

        let publics = formula.public_coefficients();
        let opening_values: BTreeMap<ComposedOpeningId, Fr> = opening_ids
            .iter()
            .copied()
            .zip(openings.iter().copied())
            .collect();
        let lowered = stage1_spartan_outer_claim::<Fr>(&dimensions)
            .output
            .expand()
            .evaluate(
                |id| opening_values.get(id).copied().unwrap(),
                |_| Fr::zero(),
                |id| resolve_public(&publics, id),
            );

        assert_eq!(lowered, factored);
    }
}
