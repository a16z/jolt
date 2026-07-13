use std::num::NonZeroUsize;

use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_riscv::InstructionFlags;

use crate::opening;

use super::super::{
    InstructionReadRafPublic, JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId,
    JoltVirtualPolynomial,
};
use super::claim_reductions::instruction::{
    left_instruction_input_reduced, lookup_output_reduced, right_instruction_input_reduced,
};
use super::dimensions::{JoltFormulaDimensionsError, JoltFormulaPointError};
use super::spartan::{
    left_instruction_input_product, lookup_output_product, right_instruction_input_product,
};

pub(crate) const INPUT_VIRTUALIZATION_DEGREE: usize = 3;
pub(crate) const READ_RAF_BASE_DEGREE: usize = 2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InstructionReadRafDimensions {
    log_t: usize,
    instruction_address_bits: usize,
    num_virtual_ra_polys: NonZeroUsize,
}

impl InstructionReadRafDimensions {
    pub const fn new(
        log_t: usize,
        instruction_address_bits: usize,
        num_virtual_ra_polys: NonZeroUsize,
    ) -> Self {
        Self {
            log_t,
            instruction_address_bits,
            num_virtual_ra_polys,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn instruction_address_bits(self) -> usize {
        self.instruction_address_bits
    }

    pub fn num_virtual_ra_polys(self) -> usize {
        self.num_virtual_ra_polys.get()
    }

    pub const fn sumcheck_rounds(self) -> usize {
        self.instruction_address_bits + self.log_t
    }

    pub fn opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<InstructionReadRafOpeningPoint<F>, JoltFormulaPointError> {
        let expected = self.instruction_address_bits + self.log_t;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let (r_address, r_cycle) = challenges.split_at(self.instruction_address_bits);
        let r_cycle = r_cycle.iter().rev().copied().collect::<Vec<_>>();
        let r_address = r_address.to_vec();
        let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();

        Ok(InstructionReadRafOpeningPoint {
            r_address,
            r_cycle,
            opening_point,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionReadRafOpeningPoint<F: Field> {
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
}

impl TryFrom<(usize, usize, usize)> for InstructionReadRafDimensions {
    type Error = JoltFormulaDimensionsError;

    fn try_from(
        (log_t, instruction_address_bits, num_virtual_ra_polys): (usize, usize, usize),
    ) -> Result<Self, Self::Error> {
        if instruction_address_bits == 0 {
            return Err(JoltFormulaDimensionsError::Zero {
                name: "instruction_address_bits",
            });
        }
        Ok(Self::new(
            log_t,
            instruction_address_bits,
            NonZeroUsize::new(num_virtual_ra_polys).ok_or(JoltFormulaDimensionsError::Zero {
                name: "instruction virtual RA polynomial count",
            })?,
        ))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InstructionRaVirtualizationDimensions {
    log_t: usize,
    virtual_ra_polys: NonZeroUsize,
    committed_per_virtual: NonZeroUsize,
    committed_ra_polys: NonZeroUsize,
}

impl InstructionRaVirtualizationDimensions {
    pub fn new(
        log_t: usize,
        num_virtual_ra_polys: NonZeroUsize,
        num_committed_per_virtual: NonZeroUsize,
    ) -> Result<Self, JoltFormulaDimensionsError> {
        let num_committed_ra_polys = num_virtual_ra_polys
            .get()
            .checked_mul(num_committed_per_virtual.get())
            .ok_or(JoltFormulaDimensionsError::Overflow {
                name: "instruction committed RA polynomial count",
            })?;
        Ok(Self {
            log_t,
            virtual_ra_polys: num_virtual_ra_polys,
            committed_per_virtual: num_committed_per_virtual,
            committed_ra_polys: NonZeroUsize::new(num_committed_ra_polys).ok_or(
                JoltFormulaDimensionsError::Zero {
                    name: "instruction committed RA polynomial count",
                },
            )?,
        })
    }

    pub fn num_virtual_ra_polys(self) -> usize {
        self.virtual_ra_polys.get()
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub fn num_committed_per_virtual(self) -> usize {
        self.committed_per_virtual.get()
    }

    pub fn num_committed_ra_polys(self) -> usize {
        self.committed_ra_polys.get()
    }
}

impl TryFrom<(usize, usize, usize)> for InstructionRaVirtualizationDimensions {
    type Error = JoltFormulaDimensionsError;

    fn try_from(
        (log_t, num_virtual_ra_polys, num_committed_per_virtual): (usize, usize, usize),
    ) -> Result<Self, Self::Error> {
        Self::new(
            log_t,
            NonZeroUsize::new(num_virtual_ra_polys).ok_or(JoltFormulaDimensionsError::Zero {
                name: "instruction virtual RA polynomial count",
            })?,
            NonZeroUsize::new(num_committed_per_virtual).ok_or(
                JoltFormulaDimensionsError::Zero {
                    name: "committed RA polynomials per virtual RA",
                },
            )?,
        )
    }
}

pub fn input_virtualization_consistency_openings() -> [(JoltOpeningId, JoltOpeningId); 2] {
    [
        (
            left_instruction_input_reduced(),
            left_instruction_input_product(),
        ),
        (
            right_instruction_input_reduced(),
            right_instruction_input_product(),
        ),
    ]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionReadRafOutputOpenings {
    pub lookup_table_flags: Vec<JoltOpeningId>,
    pub instruction_ra: Vec<JoltOpeningId>,
    pub instruction_raf_flag: JoltOpeningId,
}

impl InstructionReadRafOutputOpenings {
    /// Total produced openings: the lookup-table flags, the virtual instruction-RA
    /// openings, and the single RAF flag. Single-sources the read-RAF output count
    /// so callers don't re-add the `+ 1` flag literal.
    pub fn opening_count(&self) -> usize {
        self.lookup_table_flags.len() + self.instruction_ra.len() + 1
    }
}

pub fn read_raf_output_openings(
    dimensions: InstructionReadRafDimensions,
) -> InstructionReadRafOutputOpenings {
    InstructionReadRafOutputOpenings {
        lookup_table_flags: LookupTableKind::<XLEN>::iter()
            .map(lookup_table_flag)
            .collect(),
        instruction_ra: (0..dimensions.num_virtual_ra_polys())
            .map(instruction_ra)
            .collect(),
        instruction_raf_flag: instruction_raf_flag(),
    }
}

pub fn read_raf_consistency_openings() -> [(JoltOpeningId, JoltOpeningId); 1] {
    [(lookup_output_reduced(), lookup_output_product())]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionRaVirtualizationOutputOpenings {
    pub committed_instruction_ra_by_virtual: Vec<Vec<JoltOpeningId>>,
}

impl InstructionRaVirtualizationOutputOpenings {
    pub fn all(&self) -> Vec<JoltOpeningId> {
        self.committed_instruction_ra_by_virtual
            .iter()
            .flatten()
            .copied()
            .collect()
    }
}

pub fn ra_virtualization_output_openings(
    dimensions: InstructionRaVirtualizationDimensions,
) -> InstructionRaVirtualizationOutputOpenings {
    let committed_instruction_ra_by_virtual = (0..dimensions.num_virtual_ra_polys())
        .map(|virtual_index| {
            let start = virtual_index * dimensions.num_committed_per_virtual();
            (start..start + dimensions.num_committed_per_virtual())
                .map(committed_instruction_ra)
                .collect()
        })
        .collect();

    InstructionRaVirtualizationOutputOpenings {
        committed_instruction_ra_by_virtual,
    }
}

pub(crate) fn eq_table_value(table: LookupTableKind<XLEN>) -> InstructionReadRafPublic {
    InstructionReadRafPublic::EqTableValue(table.index())
}

pub(crate) fn weighted_instruction_ra_sum<F>(
    dimensions: InstructionRaVirtualizationDimensions,
    gamma: JoltExpr<F>,
) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut sum = JoltExpr::zero();
    for i in 0..dimensions.num_virtual_ra_polys() {
        sum = sum + gamma.clone().pow(i) * opening(instruction_ra(i));
    }
    sum
}

pub(crate) fn instruction_ra_product<F>(dimensions: InstructionReadRafDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut product = JoltExpr::one();
    for i in 0..dimensions.num_virtual_ra_polys() {
        product = product * opening(instruction_ra(i));
    }
    product
}

pub(crate) fn committed_instruction_ra_product<F>(
    dimensions: InstructionRaVirtualizationDimensions,
    virtual_index: usize,
) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut product = JoltExpr::one();
    let start = virtual_index * dimensions.num_committed_per_virtual();
    for i in start..start + dimensions.num_committed_per_virtual() {
        product = product * opening(committed_instruction_ra(i));
    }
    product
}

pub fn instruction_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionRa(index),
        JoltRelationId::InstructionReadRaf,
    )
}

pub fn committed_instruction_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::InstructionRa(index),
        JoltRelationId::InstructionRaVirtualization,
    )
}

pub fn lookup_table_flag(table: LookupTableKind<XLEN>) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupTableFlag(table.index()),
        JoltRelationId::InstructionReadRaf,
    )
}

pub fn instruction_raf_flag() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionRafFlag,
        JoltRelationId::InstructionReadRaf,
    )
}

pub fn left_operand_is_rs1() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub fn rs1_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub fn left_operand_is_pc() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub fn unexpanded_pc() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub fn right_operand_is_rs2() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub fn rs2_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub fn right_operand_is_imm() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub fn imm() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Imm,
        JoltRelationId::InstructionInputVirtualization,
    )
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn read_raf_rejects_empty_dimensions() {
        assert!(InstructionReadRafDimensions::try_from((5, 128, 0)).is_err());
        assert!(InstructionReadRafDimensions::try_from((5, 0, 1)).is_err());
    }

    #[test]
    fn read_raf_opening_point_matches_core_order() {
        let dimensions = InstructionReadRafDimensions::try_from((3, 4, 1))
            .unwrap_or_else(|err| panic!("test read-RAF dimensions should be valid: {err}"));
        let challenges = (1..=7).map(Fr::from_u64).collect::<Vec<_>>();

        let point = dimensions
            .opening_point(&challenges)
            .unwrap_or_else(|err| panic!("opening point should normalize: {err}"));

        assert_eq!(
            point.r_address,
            vec![
                Fr::from_u64(1),
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(4),
            ]
        );
        assert_eq!(
            point.r_cycle,
            vec![Fr::from_u64(7), Fr::from_u64(6), Fr::from_u64(5)]
        );
        assert_eq!(
            point.opening_point,
            vec![
                Fr::from_u64(1),
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(4),
                Fr::from_u64(7),
                Fr::from_u64(6),
                Fr::from_u64(5),
            ]
        );
    }

    #[test]
    fn ra_virtualization_rejects_invalid_dimensions() {
        assert!(InstructionRaVirtualizationDimensions::try_from((5, 0, 1)).is_err());
        assert!(InstructionRaVirtualizationDimensions::try_from((5, 1, 0)).is_err());
        assert!(InstructionRaVirtualizationDimensions::try_from((5, usize::MAX, 2)).is_err());
    }
}
