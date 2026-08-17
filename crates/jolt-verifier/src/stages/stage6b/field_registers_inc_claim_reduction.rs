//! The stage 6b `FieldRegistersIncClaimReduction` sumcheck instance — the FR
//! increment reduction member (spec: `field-inline-protocol.md`, "Stage 6
//! Composition").
//!
//! Reduces the two semantic `FieldRdInc` openings — from the stage-4 FR
//! read/write checking and the stage-5 FR val evaluation, batched by the
//! member-drawn gamma (the spec's `eta`) — to the single reduced `FieldRdInc`
//! opening the stage-8 joint opening consumes. Its publics mirror the ordinary
//! [`IncClaimReduction`](super::inc_claim_reduction::IncClaimReduction)
//! derivations exactly: `EqReadWrite = Eq(this instance's bound cycle point,
//! the stage-4 FR read/write cycle point)` and `EqValEvaluation = Eq(bound
//! cycle point, the stage-5 FR val-evaluation cycle point)`.

use jolt_claims::protocols::field_inline::relations::claim_reductions::increments;
pub use jolt_claims::protocols::field_inline::relations::claim_reductions::increments::{
    FieldRegistersIncClaimReductionChallenges, FieldRegistersIncClaimReductionInputClaims,
    FieldRegistersIncClaimReductionOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldInlineRelationId, FieldRegistersIncClaimReductionPublic,
    FieldRegistersTraceDimensions,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage4::{Stage4OutputClaims, Stage4OutputPoints};
use crate::stages::stage5::{Stage5OutputClaims, Stage5OutputPoints};
use crate::VerifierError;

/// Wire the two consumed `FieldRdInc` opening *values* from the stage-4 FR
/// read/write checking and the stage-5 FR val evaluation. The upstream cells
/// are plain (non-optional) fields of the FR-on stage-4/5 claims, so presence
/// is a compile-time fact.
pub fn field_registers_inc_claim_reduction_input_values_from_upstream<F: Field>(
    stage4: &Stage4OutputClaims<F>,
    stage5: &Stage5OutputClaims<F>,
) -> FieldRegistersIncClaimReductionInputClaims<F> {
    FieldRegistersIncClaimReductionInputClaims {
        rd_inc_read_write: stage4.field_registers_read_write.rd_inc,
        rd_inc_val_evaluation: stage5.field_registers_val_evaluation.rd_inc,
    }
}

/// Wire the two consumed `FieldRdInc` opening *points* from the stage-4/5 FR
/// members' output points. ZK-agnostic.
pub fn field_registers_inc_claim_reduction_input_points_from_upstream<F: Field>(
    stage4: &Stage4OutputPoints<F>,
    stage5: &Stage5OutputPoints<F>,
) -> FieldRegistersIncClaimReductionInputClaims<Vec<F>> {
    FieldRegistersIncClaimReductionInputClaims {
        rd_inc_read_write: stage4.field_registers_read_write.rd_inc().to_vec(),
        rd_inc_val_evaluation: stage5.field_registers_val_evaluation.rd_inc().to_vec(),
    }
}

#[derive(Clone)]
pub struct FieldRegistersIncClaimReduction<F: Field> {
    symbolic: increments::ClaimReduction,
    read_write_cycle: Vec<F>,
    val_evaluation_cycle: Vec<F>,
}

impl<F: Field> FieldRegistersIncClaimReduction<F> {
    pub fn new(
        trace_dimensions: FieldRegistersTraceDimensions,
        read_write_cycle: Vec<F>,
        val_evaluation_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: increments::ClaimReduction::new(trace_dimensions),
            read_write_cycle,
            val_evaluation_cycle,
        }
    }

    /// The two upstream FR cycle points in relation order: stage-4 FR
    /// read/write, stage-5 FR val evaluation.
    pub fn cycle_points(&self) -> [&[F]; 2] {
        [&self.read_write_cycle, &self.val_evaluation_cycle]
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!(
            "{:?}",
            FieldInlineRelationId::FieldRegistersIncClaimReduction
        ),
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for FieldRegistersIncClaimReduction<F> {
    type Symbolic = increments::ClaimReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &FieldRegistersIncClaimReductionInputClaims<Vec<F>>,
    ) -> Result<FieldRegistersIncClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        // The reduced opening point is the reversed sumcheck point — the same
        // derivation as the ordinary increment claim reduction.
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(FieldRegistersIncClaimReductionOutputClaims {
            rd_inc: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        _input_points: &FieldRegistersIncClaimReductionInputClaims<Vec<F>>,
        output_points: &FieldRegistersIncClaimReductionOutputClaims<Vec<F>>,
        _challenges: &FieldRegistersIncClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let FieldInlineDerivedId::FieldRegistersIncClaimReduction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        let opening_point = output_points.rd_inc();
        let cycle = match public {
            FieldRegistersIncClaimReductionPublic::EqReadWrite => &self.read_write_cycle,
            FieldRegistersIncClaimReductionPublic::EqValEvaluation => &self.val_evaluation_cycle,
        };
        try_eq_mle(opening_point, cycle).map_err(public_input_failed)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    clippy::arithmetic_side_effects,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;

    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{IncClaimReductionPublic, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    use crate::stages::stage6b::inc_claim_reduction::{
        IncClaimReduction, IncClaimReductionChallenges, IncClaimReductionInputClaims,
    };

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn cycles(log_t: usize) -> (Vec<Fr>, Vec<Fr>) {
        (
            (0..log_t as u64).map(|i| fr(40 + i)).collect(),
            (0..log_t as u64).map(|i| fr(60 + i)).collect(),
        )
    }

    /// The FR reduction is trace-domain: `log_t` rounds, the default
    /// (suffix-bound) instance offset — the same batch window as the ordinary
    /// increment claim reduction — and its reduced opening point is the same
    /// reversed point, so both members bound in the same stage-6b batch derive
    /// identical opening points.
    #[test]
    fn reduced_opening_point_matches_the_ordinary_inc_reduction() {
        let log_t = 4usize;
        let (read_write_cycle, val_evaluation_cycle) = cycles(log_t);
        let field_relation = FieldRegistersIncClaimReduction::<Fr>::new(
            FieldRegistersTraceDimensions::new(log_t),
            read_write_cycle.clone(),
            val_evaluation_cycle.clone(),
        );
        let jolt_relation = IncClaimReduction::<Fr>::new(
            TraceDimensions::new(log_t),
            vec![fr(1); log_t],
            vec![fr(2); log_t],
            read_write_cycle,
            val_evaluation_cycle,
        );
        assert_eq!(field_relation.rounds(), log_t);
        assert_eq!(field_relation.rounds(), jolt_relation.rounds());

        let batch_num_vars = log_t + 3;
        assert_eq!(
            field_relation
                .instance_point_offset(batch_num_vars)
                .unwrap(),
            jolt_relation.instance_point_offset(batch_num_vars).unwrap(),
        );

        let point: Vec<Fr> = (0..log_t as u64).map(|i| fr(80 + i)).collect();
        let field_points = field_relation
            .derive_opening_points(
                &point,
                &FieldRegistersIncClaimReductionInputClaims::default(),
            )
            .unwrap();
        let jolt_points = jolt_relation
            .derive_opening_points(&point, &IncClaimReductionInputClaims::default())
            .unwrap();
        assert_eq!(field_points.rd_inc(), jolt_points.rd_inc());
        assert_eq!(
            field_points.rd_inc(),
            point.iter().rev().copied().collect::<Vec<_>>()
        );
    }

    /// The FR `EqReadWrite`/`EqValEvaluation` publics mirror the ordinary
    /// increment reduction's `EqRegistersReadWrite`/`EqRegistersValEvaluation`
    /// exactly: fed the same bound point and the same upstream cycle points,
    /// the derived values are equal — `Eq(reduced point, upstream cycle)` in
    /// the same argument orientation.
    #[test]
    fn eq_publics_match_the_ordinary_inc_reduction_derivations() {
        let log_t = 4usize;
        let (read_write_cycle, val_evaluation_cycle) = cycles(log_t);
        let field_relation = FieldRegistersIncClaimReduction::<Fr>::new(
            FieldRegistersTraceDimensions::new(log_t),
            read_write_cycle.clone(),
            val_evaluation_cycle.clone(),
        );
        // The jolt member's REGISTER cycle legs carry the same two points; its
        // RAM legs carry unrelated sentinels so a wrong-leg mirror would show.
        let jolt_relation = IncClaimReduction::<Fr>::new(
            TraceDimensions::new(log_t),
            vec![fr(90); log_t],
            vec![fr(91); log_t],
            read_write_cycle,
            val_evaluation_cycle,
        );

        let point: Vec<Fr> = (0..log_t as u64).map(|i| fr(70 + i)).collect();
        let field_input_points = FieldRegistersIncClaimReductionInputClaims::default();
        let jolt_input_points = IncClaimReductionInputClaims::default();
        let field_points = field_relation
            .derive_opening_points(&point, &field_input_points)
            .unwrap();
        let jolt_points = jolt_relation
            .derive_opening_points(&point, &jolt_input_points)
            .unwrap();

        let field_challenges = FieldRegistersIncClaimReductionChallenges { gamma: fr(1) };
        let jolt_challenges = IncClaimReductionChallenges { gamma: fr(1) };
        for (field_public, jolt_public) in [
            (
                FieldRegistersIncClaimReductionPublic::EqReadWrite,
                IncClaimReductionPublic::EqRegistersReadWrite,
            ),
            (
                FieldRegistersIncClaimReductionPublic::EqValEvaluation,
                IncClaimReductionPublic::EqRegistersValEvaluation,
            ),
        ] {
            let field_eq = field_relation
                .derive_output_term(
                    &FieldInlineDerivedId::FieldRegistersIncClaimReduction(field_public),
                    &field_input_points,
                    &field_points,
                    &field_challenges,
                )
                .unwrap();
            let jolt_eq = jolt_relation
                .derive_output_term(
                    &JoltDerivedId::IncClaimReduction(jolt_public),
                    &jolt_input_points,
                    &jolt_points,
                    &jolt_challenges,
                )
                .unwrap();
            assert_eq!(
                field_eq, jolt_eq,
                "{field_public:?} must mirror {jolt_public:?}"
            );
        }
    }
}
