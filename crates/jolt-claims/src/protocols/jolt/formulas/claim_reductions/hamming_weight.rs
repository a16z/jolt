use jolt_field::{Field, RingCore};

use crate::opening;

use super::super::super::{JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial};
use super::super::dimensions::{JoltFormulaPointError, JoltSumcheckSpec};
use super::super::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HammingWeightClaimReductionDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_k_chunk: usize,
}

impl HammingWeightClaimReductionDimensions {
    pub const fn new(layout: JoltRaPolynomialLayout, log_k_chunk: usize) -> Self {
        Self {
            layout,
            log_k_chunk,
        }
    }

    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_k_chunk, 2)
    }

    pub fn opening_point<F: Field>(
        self,
        challenges: &[F],
        r_cycle: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if challenges.len() != self.log_k_chunk {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.log_k_chunk,
                got: challenges.len(),
            });
        }

        let mut r_address = challenges.iter().rev().copied().collect::<Vec<_>>();
        r_address.extend_from_slice(r_cycle);
        Ok(r_address)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HammingWeightClaimReductionOutputOpenings {
    pub instruction_ra: Vec<JoltOpeningId>,
    pub bytecode_ra: Vec<JoltOpeningId>,
    pub ram_ra: Vec<JoltOpeningId>,
}

impl HammingWeightClaimReductionOutputOpenings {
    pub fn all(&self) -> Vec<JoltOpeningId> {
        self.instruction_ra
            .iter()
            .chain(&self.bytecode_ra)
            .chain(&self.ram_ra)
            .copied()
            .collect()
    }
}

pub fn claim_reduction_output_openings(
    dimensions: HammingWeightClaimReductionDimensions,
) -> HammingWeightClaimReductionOutputOpenings {
    let mut instruction_ra = Vec::with_capacity(dimensions.layout.instruction());
    let mut bytecode_ra = Vec::with_capacity(dimensions.layout.bytecode());
    let mut ram_ra = Vec::with_capacity(dimensions.layout.ram());

    for polynomial in dimensions.layout.polynomials() {
        match polynomial {
            JoltRaPolynomial::Instruction(_) => instruction_ra.push(reduced_claim(polynomial)),
            JoltRaPolynomial::Bytecode(_) => bytecode_ra.push(reduced_claim(polynomial)),
            JoltRaPolynomial::Ram(_) => ram_ra.push(reduced_claim(polynomial)),
        }
    }

    HammingWeightClaimReductionOutputOpenings {
        instruction_ra,
        bytecode_ra,
        ram_ra,
    }
}

pub(crate) fn hamming_weight_claim<F>(polynomial: JoltRaPolynomial) -> JoltExpr<F>
where
    F: RingCore,
{
    match polynomial {
        JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => JoltExpr::one(),
        JoltRaPolynomial::Ram(_) => opening(ram_hamming_weight()),
    }
}

pub(crate) fn booleanity_claim(polynomial: JoltRaPolynomial) -> JoltOpeningId {
    polynomial.opening(JoltRelationId::Booleanity)
}

pub(crate) fn virtualization_claim(polynomial: JoltRaPolynomial) -> JoltOpeningId {
    match polynomial {
        JoltRaPolynomial::Instruction(_) => JoltOpeningId::committed(
            polynomial.committed(),
            JoltRelationId::InstructionRaVirtualization,
        ),
        JoltRaPolynomial::Bytecode(_) => {
            JoltOpeningId::committed(polynomial.committed(), JoltRelationId::BytecodeReadRaf)
        }
        JoltRaPolynomial::Ram(_) => {
            JoltOpeningId::committed(polynomial.committed(), JoltRelationId::RamRaVirtualization)
        }
    }
}

pub(crate) fn reduced_claim(polynomial: JoltRaPolynomial) -> JoltOpeningId {
    polynomial.opening(JoltRelationId::HammingWeightClaimReduction)
}

pub(crate) fn ram_hamming_weight() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamHammingWeight,
        JoltRelationId::RamHammingBooleanity,
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::dimensions::JoltFormulaDimensionsError;
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn layout(
        instruction: usize,
        bytecode: usize,
        ram: usize,
    ) -> Result<JoltRaPolynomialLayout, JoltFormulaDimensionsError> {
        JoltRaPolynomialLayout::new(instruction, bytecode, ram)
    }

    fn dimensions_with_log_k_chunk(
        layout: JoltRaPolynomialLayout,
        log_k_chunk: usize,
    ) -> HammingWeightClaimReductionDimensions {
        HammingWeightClaimReductionDimensions::new(layout, log_k_chunk)
    }

    #[test]
    fn opening_point_reverses_address_and_appends_cycle() -> Result<(), Box<dyn std::error::Error>>
    {
        let dimensions = dimensions_with_log_k_chunk(layout(1, 1, 1)?, 3);
        let point = dimensions.opening_point(
            &[Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)],
            &[Fr::from_u64(5), Fr::from_u64(6)],
        )?;

        assert_eq!(
            point,
            vec![
                Fr::from_u64(4),
                Fr::from_u64(3),
                Fr::from_u64(2),
                Fr::from_u64(5),
                Fr::from_u64(6),
            ]
        );
        Ok(())
    }
}
