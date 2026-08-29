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

use jolt_claims::protocols::field_inline::relations::claim_reductions::increments::ClaimReduction;
pub use jolt_claims::protocols::field_inline::relations::claim_reductions::increments::{
    FieldRegistersIncClaimReductionChallenges, FieldRegistersIncClaimReductionInputClaims,
    FieldRegistersIncClaimReductionOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldInlineRelationId, FieldRegistersIncClaimReductionPublic,
    FieldRegistersTraceDimensions,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;

use crate::stages::derivations;
use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[derive(Clone)]
pub struct FieldRegistersIncClaimReduction<F: JoltField> {
    symbolic: ClaimReduction,
    read_write_cycle: Vec<F>,
    val_evaluation_cycle: Vec<F>,
}

impl<F: JoltField> FieldRegistersIncClaimReduction<F> {
    pub fn new(
        trace_dimensions: FieldRegistersTraceDimensions,
        read_write_cycle: Vec<F>,
        val_evaluation_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: ClaimReduction::new(trace_dimensions),
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

impl<F: JoltField> ConcreteSumcheck<F> for FieldRegistersIncClaimReduction<F> {
    type Symbolic = ClaimReduction;

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
        let opening_point = derivations::reversed(sumcheck_point);
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
        derivations::eq_at_point(opening_point, cycle).map_err(public_input_failed)
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
    use jolt_field::{Fr, Ring};

    use crate::stages::stage6b::inc_claim_reduction::{
        IncClaimReduction, IncClaimReductionInputClaims,
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
}
