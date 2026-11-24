//! Committed bytecode row geometry and read-RAF wires for the selected profile.

#[cfg(all(feature = "akita", feature = "field-inline"))]
use crate::protocols::field_inline::{geometry::bytecode::FieldBytecodeLeg, FIELD_REGISTERS_LOG_K};
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    CommittedBytecodeProfile, REGISTER_COUNT,
};
#[cfg(all(feature = "akita", feature = "field-inline"))]
use crate::protocols::jolt::{
    geometry::claim_reductions::bytecode::{
        lane_weights, BytecodeLaneLayout, BytecodeLaneWeightInputs, BYTECODE_LANE_LAYOUT,
    },
    PointGeometryError,
};
#[cfg(all(feature = "akita", feature = "field-inline"))]
use jolt_field::JoltField;
#[cfg(all(feature = "akita", feature = "field-inline"))]
use jolt_poly::EqPolynomial;
#[cfg(all(feature = "akita", feature = "field-inline"))]
use jolt_riscv::FIELD_REGISTER_COUNT;
use jolt_riscv::{JoltInstructionRow, NormalizedOperands};

/// Whether every encoded selector fits its one-hot register block.
pub fn is_valid_committed_program_registers(row: &JoltInstructionRow) -> bool {
    let fits = |operands: NormalizedOperands, width: usize| {
        [operands.rd, operands.rs1, operands.rs2]
            .into_iter()
            .flatten()
            .all(|register| usize::from(register) < width)
    };
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    if !fits(row.field_operands(), usize::from(FIELD_REGISTER_COUNT)) {
        return false;
    }
    fits(row.integer_operands(), REGISTER_COUNT)
}

impl CommittedBytecodeProfile {
    #[cfg(not(all(feature = "akita", feature = "field-inline")))]
    pub(crate) const ACTIVE: Self = Self {
        extra_lanes: 0,
        extra_cycle_stages: &[],
        packing_domain: b"program-bytecode-whole-v1",
    };

    #[cfg(all(feature = "akita", feature = "field-inline"))]
    pub(crate) const ACTIVE: Self = Self {
        extra_lanes: 3 * FIELD_REGISTER_COUNT as usize,
        extra_cycle_stages: &[
            FieldBytecodeLeg::ORDER[0].stage(),
            FieldBytecodeLeg::ORDER[1].stage(),
        ],
        packing_domain: b"program-bytecode-field-selectors-v1",
    };
}

#[cfg(all(feature = "akita", feature = "field-inline"))]
impl BytecodeLaneLayout {
    pub const fn field_starts(self) -> [usize; 3] {
        let rd = self.raf_flag_idx + 1;
        let rs1 = rd + FIELD_REGISTER_COUNT as usize;
        [rd, rs1, rs1 + FIELD_REGISTER_COUNT as usize]
    }
}

#[cfg(all(feature = "akita", feature = "field-inline"))]
pub struct FieldRegisterLaneInputs<'a, F> {
    pub read_write_address: &'a [F],
    pub value_address: &'a [F],
    pub selector_gammas: ([F; 3], F),
}

#[cfg(all(feature = "akita", feature = "field-inline"))]
impl<F: JoltField> FieldRegisterLaneInputs<'_, F> {
    pub fn lane_weights(
        self,
        base: BytecodeLaneWeightInputs<'_, F>,
    ) -> Result<Vec<F>, PointGeometryError> {
        for point in [self.read_write_address, self.value_address] {
            let expected = FIELD_REGISTERS_LOG_K;
            if point.len() != expected {
                return Err(PointGeometryError::OpeningPointLengthMismatch {
                    expected,
                    got: point.len(),
                });
            }
        }
        let eta = base.eta;
        let mut weights = lane_weights(base)?;
        let ([rd_gamma, rs1_gamma, rs2_gamma], val_gamma) = self.selector_gammas;
        let [rd, rs1, rs2] = BYTECODE_LANE_LAYOUT.field_starts();
        for (index, leg) in FieldBytecodeLeg::ORDER.into_iter().enumerate() {
            let power =
                (0..CommittedBytecodeProfile::wire(index)).fold(F::one(), |power, _| power * eta);
            let point = leg.select(self.read_write_address, self.value_address);
            for (k, eq) in EqPolynomial::<F>::evals(point, None)
                .into_iter()
                .enumerate()
            {
                let weight = power * eq;
                match leg {
                    FieldBytecodeLeg::ReadWrite => {
                        weights[rd + k] += weight * rd_gamma;
                        weights[rs1 + k] += weight * rs1_gamma;
                        weights[rs2 + k] += weight * rs2_gamma;
                    }
                    FieldBytecodeLeg::Value => weights[rd + k] += weight * val_gamma,
                }
            }
        }
        Ok(weights)
    }
}
