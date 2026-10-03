use core::marker::PhantomData;

use jolt_claims::protocols::field_inline::relations::registers::ReadWriteChecking;
pub use jolt_claims::protocols::field_inline::relations::registers::{
    FieldRegistersReadWriteChallenges, FieldRegistersReadWriteInputClaims,
    FieldRegistersReadWriteOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldRegistersReadWriteDimensions, FieldRegistersReadWritePublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;

use crate::stages::derivations;
use crate::stages::relations::{project_public, stage_claim_failed, ConcreteSumcheck};
use crate::stages::stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints};
use crate::VerifierError;

pub fn field_registers_read_write_input_values_from_upstream<F: JoltField>(
    stage2: &Stage2BatchOutputClaims<F>,
) -> FieldRegistersReadWriteInputClaims<F> {
    let reduction = &stage2.field_registers_claim_reduction;
    FieldRegistersReadWriteInputClaims {
        rd_value: reduction.rd_value,
        rs1_value: reduction.rs1_value,
        rs2_value: reduction.rs2_value,
    }
}

pub fn field_registers_read_write_input_points_from_upstream<F: JoltField>(
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
            .map_err(|reason| stage_claim_failed(self.id(), reason))?
            .opening_point;
        Ok(FieldRegistersReadWriteOutputClaims::from_shared_point(
            opening_point,
        ))
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        input_points: &FieldRegistersReadWriteInputClaims<Vec<F>>,
        output_points: &FieldRegistersReadWriteOutputClaims<Vec<F>>,
        _challenges: &FieldRegistersReadWriteChallenges<F>,
    ) -> Result<F, VerifierError> {
        match project_public(id)? {
            FieldRegistersReadWritePublic::EqCycle => derivations::eq_at_cycle(
                input_points.rd_value(),
                output_points.registers_val(),
                self.dimensions.log_k(),
                "field-register",
            )
            .map_err(|reason| stage_claim_failed(self.id(), reason)),
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

        assert_eq!(output_points.registers_val(), output_points.rs1_ra());
        assert_eq!(output_points.registers_val(), output_points.rs2_ra());
        assert_eq!(output_points.registers_val(), output_points.rd_wa());
        assert_eq!(output_points.registers_val(), output_points.rd_inc());
    }
}
