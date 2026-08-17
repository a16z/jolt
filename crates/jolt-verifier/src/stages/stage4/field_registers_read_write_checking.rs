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

use jolt_claims::protocols::field_inline::relations::registers;
pub use jolt_claims::protocols::field_inline::relations::registers::{
    FieldRegistersReadWriteChallenges, FieldRegistersReadWriteInputClaims,
    FieldRegistersReadWriteOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldInlineRelationId, FieldRegistersReadWriteDimensions,
    FieldRegistersReadWritePublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints};
use crate::VerifierError;

/// Wire the consumed FR value opening *values* from stage 2's FR claim
/// reduction. The upstream cells are plain (non-optional) fields of the FR-on
/// stage-2 batch claims, so presence is a compile-time fact — an FR-on proof
/// without them fails proof deserialization / shape validation upstream.
pub fn field_registers_read_write_input_values_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputClaims<F>,
) -> FieldRegistersReadWriteInputClaims<F> {
    let reduction = &stage2.field_registers_claim_reduction;
    FieldRegistersReadWriteInputClaims {
        rd_value: reduction.rd_value,
        rs1_value: reduction.rs1_value,
        rs2_value: reduction.rs2_value,
    }
}

/// Wire the consumed FR opening *points* from stage 2's FR claim reduction,
/// all sharing that relation's reduced opening point (`r_prod`).
pub fn field_registers_read_write_input_points_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputPoints<F>,
) -> FieldRegistersReadWriteInputClaims<Vec<F>> {
    let reduction = &stage2.field_registers_claim_reduction;
    FieldRegistersReadWriteInputClaims {
        rd_value: reduction.rd_value().to_vec(),
        rs1_value: reduction.rs1_value().to_vec(),
        rs2_value: reduction.rs2_value().to_vec(),
    }
}

#[derive(Clone)]
pub struct FieldRegistersReadWriteChecking<F: Field> {
    symbolic: registers::ReadWriteChecking,
    dimensions: FieldRegistersReadWriteDimensions,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> FieldRegistersReadWriteChecking<F> {
    pub fn new(dimensions: FieldRegistersReadWriteDimensions) -> Self {
        Self {
            symbolic: registers::ReadWriteChecking::new(dimensions),
            dimensions,
            _field: core::marker::PhantomData,
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

impl<F: Field> ConcreteSumcheck<F> for FieldRegistersReadWriteChecking<F> {
    type Symbolic = registers::ReadWriteChecking;

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
            // address prefix — the same derivation as the ordinary
            // `RegistersReadWriteChecking`'s `EqCycle`.
            FieldRegistersReadWritePublic::EqCycle => {
                let fixed_cycle = input_points.rd_value();
                let registers_cycle = output_points
                    .registers_val()
                    .get(self.dimensions.log_k()..)
                    .ok_or_else(|| {
                        public_input_failed(
                            "field-register read-write opening point is shorter than the \
                             field-register address width",
                        )
                    })?;
                try_eq_mle(fixed_cycle, registers_cycle).map_err(public_input_failed)
            }
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
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        ReadWriteDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersReadWritePublic};
    use jolt_field::{Fr, FromPrimitiveInt};

    use crate::stages::stage4::registers_read_write_checking::{
        RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
    };

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

    /// The FR `EqCycle` mirrors the ordinary registers read-write derivation
    /// exactly: `Eq(upstream reduced cycle point, own cycle sub-point)`. Built
    /// at the jolt register shape (the jolt member's address slice is the
    /// `REGISTER_ADDRESS_BITS` constant), the two publics are equal at the same
    /// batch point and input point.
    #[test]
    fn field_registers_eq_cycle_matches_registers_read_write_derivation() {
        let log_t = 4usize;
        let log_k = REGISTER_ADDRESS_BITS;
        let jolt_relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
            log_t, log_k, log_t, log_k,
        ));
        let field_relation = FieldRegistersReadWriteChecking::<Fr>::new(
            FieldRegistersReadWriteDimensions::new(log_t, log_k, log_t, log_k),
        );
        assert_eq!(jolt_relation.rounds(), field_relation.rounds());

        let point: Vec<Fr> = (0..jolt_relation.rounds() as u64)
            .map(|i| fr(30 + i))
            .collect();
        let fixed_cycle: Vec<Fr> = (0..log_t as u64).map(|i| fr(60 + i)).collect();

        let jolt_input_points = RegistersReadWriteInputClaims::<Vec<Fr>> {
            rd_write_value: fixed_cycle.clone(),
            rs1_value: fixed_cycle.clone(),
            rs2_value: fixed_cycle.clone(),
        };
        let field_input_points = FieldRegistersReadWriteInputClaims::<Vec<Fr>> {
            rd_value: fixed_cycle.clone(),
            rs1_value: fixed_cycle.clone(),
            rs2_value: fixed_cycle,
        };

        let jolt_points = jolt_relation
            .derive_opening_points(&point, &jolt_input_points)
            .unwrap();
        let field_points = field_relation
            .derive_opening_points(&point, &field_input_points)
            .unwrap();
        assert_eq!(jolt_points.registers_val(), field_points.registers_val());

        let jolt_eq = jolt_relation
            .derive_output_term(
                &JoltDerivedId::RegistersReadWrite(RegistersReadWritePublic::EqCycle),
                &jolt_input_points,
                &jolt_points,
                &RegistersReadWriteChallenges { gamma: fr(1) },
            )
            .unwrap();
        let field_eq = field_relation
            .derive_output_term(
                &FieldInlineDerivedId::FieldRegistersReadWrite(
                    FieldRegistersReadWritePublic::EqCycle,
                ),
                &field_input_points,
                &field_points,
                &FieldRegistersReadWriteChallenges { gamma: fr(1) },
            )
            .unwrap();

        assert_eq!(field_eq, jolt_eq);
    }
}
