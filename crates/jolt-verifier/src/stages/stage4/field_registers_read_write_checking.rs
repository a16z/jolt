//! The stage 4 `FieldRegistersReadWriteChecking` sumcheck instance — the FR
//! Twist read/write member (spec: `field-inline-protocol.md`, "Stage 4
//! Composition").
//!
//! Relates the three FR value openings reduced at `r_prod` by the stage-2 FR
//! claim reduction (`FieldRdValue`, `FieldRs1Value`, `FieldRs2Value`, batched
//! by gamma) to the FR register-memory openings (`FieldRegistersVal`,
//! `FieldRs1Ra`, `FieldRs2Ra`, `FieldRdWa`, `FieldRdInc`) at the FR read/write
//! point, weighted by the `EqCycle` public.
//!
//! Owns the FR read/write opening-point derivation (the
//! `FieldRegistersReadWriteDimensions` phase split into `[address ‖ cycle]`)
//! and the `EqCycle` public-value computation, mirroring the ordinary
//! `RegistersReadWriteChecking`: `EqCycle = Eq(upstream reduced cycle point,
//! this instance's cycle sub-point)`.

use core::marker::PhantomData;

use jolt_claims::protocols::field_inline::relations::registers::ReadWriteChecking;
pub use jolt_claims::protocols::field_inline::relations::registers::{
    FieldRegistersReadWriteChallenges, FieldRegistersReadWriteInputClaims,
    FieldRegistersReadWriteOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldInlineRelationId, FieldRegistersReadWriteDimensions,
    FieldRegistersReadWritePublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;

use crate::stages::derivations;
use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[derive(Clone)]
pub struct FieldRegistersReadWriteChecking<F: JoltField> {
    symbolic: ReadWriteChecking,
    dimensions: FieldRegistersReadWriteDimensions,
    _field: PhantomData<F>,
}

impl<F: JoltField> FieldRegistersReadWriteChecking<F> {
    pub fn new(dimensions: FieldRegistersReadWriteDimensions) -> Self {
        Self {
            symbolic: ReadWriteChecking::new(dimensions),
            dimensions,
            _field: PhantomData,
        }
    }

    pub fn dimensions(&self) -> FieldRegistersReadWriteDimensions {
        self.dimensions
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!(
            "{:?}",
            FieldInlineRelationId::FieldRegistersReadWriteChecking
        ),
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for FieldRegistersReadWriteChecking<F> {
    type Symbolic = ReadWriteChecking;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &FieldRegistersReadWriteInputClaims<Vec<F>>,
    ) -> Result<FieldRegistersReadWriteOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .dimensions
            .read_write_opening_point(sumcheck_point)
            .map_err(public_input_failed)?
            .opening_point;
        Ok(FieldRegistersReadWriteOutputClaims {
            registers_val: opening_point.clone(),
            rs1_ra: opening_point.clone(),
            rs2_ra: opening_point.clone(),
            rd_wa: opening_point.clone(),
            rd_inc: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        input_points: &FieldRegistersReadWriteInputClaims<Vec<F>>,
        output_points: &FieldRegistersReadWriteOutputClaims<Vec<F>>,
        _challenges: &FieldRegistersReadWriteChallenges<F>,
    ) -> Result<F, VerifierError> {
        let FieldInlineDerivedId::FieldRegistersReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        match public_id {
            // The upstream reduced point (`r_prod`) is the fixed cycle; this
            // instance's cycle sub-point is the opening point past the FR
            // address prefix — literally the ordinary
            // `RegistersReadWriteChecking` derivation at the FR geometry.
            FieldRegistersReadWritePublic::EqCycle => derivations::eq_at_cycle(
                input_points.rd_value(),
                output_points.registers_val(),
                self.dimensions.log_k(),
                "field-register",
            )
            .map_err(public_input_failed),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;

    use jolt_claims::protocols::field_inline::FieldInlineConfig;
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// The config-pinned FR read/write point derivation: with `phase1 = log_t`
    /// and `phase2 = log_k` (no phase-3 rounds), the opening point is
    /// `[address ‖ cycle]` where the cycle is the reversed phase-1 slice and
    /// the address the reversed phase-2 slice — the
    /// `FieldRegistersReadWriteDimensions::read_write_opening_point` split.
    #[test]
    fn opening_point_splits_into_address_and_cycle_phases() {
        let log_t = 5usize;
        let dimensions = FieldInlineConfig::enabled().read_write_dimensions(log_t);
        let relation = FieldRegistersReadWriteChecking::<Fr>::new(dimensions);
        assert_eq!(relation.rounds(), log_t + dimensions.log_k());

        let point: Vec<Fr> = (0..relation.rounds() as u64).map(|i| fr(10 + i)).collect();
        let input_points = FieldRegistersReadWriteInputClaims::<Vec<Fr>>::default();
        let output_points = relation
            .derive_opening_points(&point, &input_points)
            .unwrap();

        let split = dimensions.read_write_opening_point(&point).unwrap();
        assert_eq!(output_points.registers_val(), split.opening_point);
        let (cycle_phase, address_phase) = point.split_at(log_t);
        let expected_cycle: Vec<Fr> = cycle_phase.iter().rev().copied().collect();
        let expected_address: Vec<Fr> = address_phase.iter().rev().copied().collect();
        assert_eq!(split.r_cycle, expected_cycle);
        assert_eq!(split.r_address, expected_address);
        assert_eq!(
            output_points.registers_val(),
            [expected_address, expected_cycle].concat()
        );

        // All five openings share the point.
        assert_eq!(output_points.registers_val(), output_points.rs1_ra());
        assert_eq!(output_points.registers_val(), output_points.rs2_ra());
        assert_eq!(output_points.registers_val(), output_points.rd_wa());
        assert_eq!(output_points.registers_val(), output_points.rd_inc());
    }
}
