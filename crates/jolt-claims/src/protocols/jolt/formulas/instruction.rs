use std::num::NonZeroUsize;

use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_riscv::InstructionFlags;

use crate::{challenge, opening, public, SameEvaluationAs};

use super::super::InstructionRaVirtualizationChallenge;
use super::super::{
    InstructionInputChallenge, InstructionInputPublic, InstructionRaVirtualizationPublic,
    InstructionReadRafChallenge, InstructionReadRafPublic, JoltChallengeId,
    JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationClaims,
    JoltRelationId, JoltVirtualPolynomial,
};
use super::dimensions::{
    JoltFormulaDimensionsError, JoltFormulaPointError, JoltSumcheckSpec, TraceDimensions,
};

pub(crate) const INPUT_VIRTUALIZATION_DEGREE: usize = 3;
const READ_RAF_BASE_DEGREE: usize = 2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InstructionReadRafDimensions {
    log_t: usize,
    instruction_address_bits: usize,
    num_virtual_ra_polys: NonZeroUsize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InstructionReadRafAddressLayout {
    address_bits: usize,
    virtual_ra_chunk_bits: usize,
}

impl InstructionReadRafAddressLayout {
    pub const fn address_bits(self) -> usize {
        self.address_bits
    }

    pub const fn virtual_ra_chunk_bits(self) -> usize {
        self.virtual_ra_chunk_bits
    }
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

    pub fn address_layout(
        self,
    ) -> Result<InstructionReadRafAddressLayout, JoltFormulaDimensionsError> {
        let virtual_ra_count = self.num_virtual_ra_polys();
        if !self
            .instruction_address_bits
            .is_multiple_of(virtual_ra_count)
        {
            return Err(JoltFormulaDimensionsError::NotDivisible {
                value_name: "instruction_address_bits",
                value: self.instruction_address_bits,
                divisor_name: "instruction virtual RA polynomial count",
                divisor: virtual_ra_count,
            });
        }
        Ok(InstructionReadRafAddressLayout {
            address_bits: self.instruction_address_bits,
            virtual_ra_chunk_bits: self.instruction_address_bits / virtual_ra_count,
        })
    }

    pub fn u128_address_layout(
        self,
    ) -> Result<InstructionReadRafAddressLayout, JoltFormulaDimensionsError> {
        let layout = self.address_layout()?;
        if layout.address_bits > u128::BITS as usize {
            return Err(JoltFormulaDimensionsError::Exceeds {
                value_name: "instruction_address_bits",
                value: layout.address_bits,
                max_name: "u128::BITS",
                max: u128::BITS as usize,
            });
        }
        Ok(layout)
    }

    pub fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(
            self.instruction_address_bits + self.log_t,
            self.num_virtual_ra_polys() + READ_RAF_BASE_DEGREE,
        )
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

    pub fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t, self.num_committed_per_virtual() + 1)
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

pub fn input_virtualization<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    use crate::protocols::jolt::relations::instruction::InputVirtualization;
    use crate::SymbolicSumcheck;
    let r = InputVirtualization::new(dimensions);
    JoltRelationClaims::new(
        InputVirtualization::id(),
        r.sumcheck(),
        r.input_expression::<F>(),
        r.output_expression::<F>(),
    )
    .with_consistency([
        left_instruction_input_reduced().same_evaluation_as(left_instruction_input_product()),
        right_instruction_input_reduced().same_evaluation_as(right_instruction_input_product()),
    ])
}

pub fn input_virtualization_input_openings() -> [JoltOpeningId; 2] {
    [
        right_instruction_input_product(),
        left_instruction_input_product(),
    ]
}

pub fn input_virtualization_output_openings() -> [JoltOpeningId; 8] {
    [
        right_operand_is_rs2(),
        rs2_value(),
        right_operand_is_imm(),
        imm(),
        left_operand_is_rs1(),
        rs1_value(),
        left_operand_is_pc(),
        unexpanded_pc(),
    ]
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

pub fn read_raf<F>(dimensions: InstructionReadRafDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    use crate::protocols::jolt::relations::instruction::ReadRaf;
    use crate::SymbolicSumcheck;
    let r = ReadRaf::new(dimensions);
    JoltRelationClaims::new(
        ReadRaf::id(),
        r.sumcheck(),
        r.input_expression::<F>(),
        r.output_expression::<F>(),
    )
    .with_consistency([lookup_output_reduced().same_evaluation_as(lookup_output_product())])
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

pub fn read_raf_input_openings() -> [JoltOpeningId; 3] {
    [
        lookup_output_reduced(),
        left_lookup_operand_reduced(),
        right_lookup_operand_reduced(),
    ]
}

pub fn read_raf_output_openings(
    dimensions: InstructionReadRafDimensions,
) -> InstructionReadRafOutputOpenings {
    InstructionReadRafOutputOpenings {
        lookup_table_flags: LookupTableKind::<XLEN>::iter()
            .map(read_raf_lookup_table_flag_opening)
            .collect(),
        instruction_ra: (0..dimensions.num_virtual_ra_polys())
            .map(read_raf_instruction_ra_opening)
            .collect(),
        instruction_raf_flag: read_raf_instruction_raf_flag_opening(),
    }
}

pub fn read_raf_consistency_openings() -> [(JoltOpeningId, JoltOpeningId); 1] {
    [(lookup_output_reduced(), lookup_output_product())]
}

pub fn read_raf_lookup_table_flag_opening(table: LookupTableKind<XLEN>) -> JoltOpeningId {
    lookup_table_flag(table)
}

pub fn read_raf_instruction_ra_opening(index: usize) -> JoltOpeningId {
    instruction_ra(index)
}

pub fn read_raf_instruction_raf_flag_opening() -> JoltOpeningId {
    instruction_raf_flag()
}

pub fn ra_virtualization<F>(
    dimensions: InstructionRaVirtualizationDimensions,
) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    use crate::protocols::jolt::relations::instruction::RaVirtualization;
    use crate::SymbolicSumcheck;
    let r = RaVirtualization::new(dimensions);
    JoltRelationClaims::new(
        RaVirtualization::id(),
        r.sumcheck(),
        r.input_expression::<F>(),
        r.output_expression::<F>(),
    )
    .with_input_challenges([JoltChallengeId::from(
        InstructionRaVirtualizationChallenge::Gamma,
    )])
}

pub fn ra_virtualization_eq_cycle_polynomial<F>(instruction_read_raf_cycle: &[F]) -> Polynomial<F>
where
    F: Field,
{
    let eq_point = instruction_read_raf_cycle
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    Polynomial::new(EqPolynomial::<F>::evals(&eq_point, None))
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

pub fn ra_virtualization_input_openings(
    dimensions: InstructionRaVirtualizationDimensions,
) -> Vec<JoltOpeningId> {
    (0..dimensions.num_virtual_ra_polys())
        .map(ra_virtualization_instruction_ra_opening)
        .collect()
}

pub fn ra_virtualization_output_openings(
    dimensions: InstructionRaVirtualizationDimensions,
) -> InstructionRaVirtualizationOutputOpenings {
    let committed_instruction_ra_by_virtual = (0..dimensions.num_virtual_ra_polys())
        .map(|virtual_index| {
            let start = virtual_index * dimensions.num_committed_per_virtual();
            (start..start + dimensions.num_committed_per_virtual())
                .map(ra_virtualization_committed_instruction_ra_opening)
                .collect()
        })
        .collect();

    InstructionRaVirtualizationOutputOpenings {
        committed_instruction_ra_by_virtual,
    }
}

pub fn ra_virtualization_instruction_ra_opening(index: usize) -> JoltOpeningId {
    instruction_ra(index)
}

pub fn ra_virtualization_committed_instruction_ra_opening(index: usize) -> JoltOpeningId {
    committed_instruction_ra(index)
}

pub(crate) fn input_challenge<F>(id: InstructionInputChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn input_public<F>(id: InstructionInputPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn read_raf_challenge<F>(id: InstructionReadRafChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn read_raf_public<F>(id: InstructionReadRafPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn ra_virtualization_challenge<F>(
    id: InstructionRaVirtualizationChallenge,
) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn ra_virtualization_public<F>(id: InstructionRaVirtualizationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
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

pub(crate) fn left_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn right_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

fn left_instruction_input_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::InstructionClaimReduction,
    )
}

fn right_instruction_input_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::InstructionClaimReduction,
    )
}

fn lookup_output_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn lookup_output_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::InstructionClaimReduction,
    )
}

pub(crate) fn left_lookup_operand_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftLookupOperand,
        JoltRelationId::InstructionClaimReduction,
    )
}

pub(crate) fn right_lookup_operand_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightLookupOperand,
        JoltRelationId::InstructionClaimReduction,
    )
}

fn instruction_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionRa(index),
        JoltRelationId::InstructionReadRaf,
    )
}

fn committed_instruction_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::InstructionRa(index),
        JoltRelationId::InstructionRaVirtualization,
    )
}

pub(crate) fn lookup_table_flag(table: LookupTableKind<XLEN>) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupTableFlag(table.index()),
        JoltRelationId::InstructionReadRaf,
    )
}

pub(crate) fn instruction_raf_flag() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionRafFlag,
        JoltRelationId::InstructionReadRaf,
    )
}

pub(crate) fn left_operand_is_rs1() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn rs1_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn left_operand_is_pc() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn unexpanded_pc() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn right_operand_is_rs2() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn rs2_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn right_operand_is_imm() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn imm() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Imm,
        JoltRelationId::InstructionInputVirtualization,
    )
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltConsistencyClaim, JoltPolynomialId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;

    fn read_raf_dimensions(num_virtual_ra_polys: usize) -> InstructionReadRafDimensions {
        InstructionReadRafDimensions::try_from((5, 128, num_virtual_ra_polys))
            .unwrap_or_else(|err| panic!("test read-RAF dimensions should be nonzero: {err}"))
    }

    fn ra_virtualization_dimensions(
        num_virtual_ra_polys: usize,
        num_committed_per_virtual: usize,
    ) -> InstructionRaVirtualizationDimensions {
        InstructionRaVirtualizationDimensions::try_from((
            5,
            num_virtual_ra_polys,
            num_committed_per_virtual,
        ))
        .unwrap_or_else(|err| panic!("test RA virtualization dimensions should be valid: {err}"))
    }

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    fn lookup_table_flags() -> Vec<JoltOpeningId> {
        LookupTableKind::<XLEN>::iter()
            .map(lookup_table_flag)
            .collect()
    }

    fn eq_table_value_publics() -> Vec<JoltPublicId> {
        LookupTableKind::<XLEN>::iter()
            .map(|table| JoltPublicId::from(eq_table_value(table)))
            .collect()
    }

    #[test]
    fn input_virtualization_exposes_expected_dependencies() {
        let claims = input_virtualization::<Fr>(trace_dimensions());

        assert_eq!(claims.id, JoltRelationId::InstructionInputVirtualization);
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(3));
        assert_eq!(
            claims.input.required_openings,
            input_virtualization_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            input_virtualization_output_openings().to_vec()
        );
        assert_eq!(
            claims.consistency,
            input_virtualization_consistency_openings()
                .into_iter()
                .map(|(left, right)| JoltConsistencyClaim::same_evaluation(left, right))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            claims.required_openings(),
            vec![
                right_instruction_input_product(),
                left_instruction_input_product(),
                right_operand_is_rs2(),
                rs2_value(),
                right_operand_is_imm(),
                imm(),
                left_operand_is_rs1(),
                rs1_value(),
                left_operand_is_pc(),
                unexpanded_pc(),
                left_instruction_input_reduced(),
                right_instruction_input_reduced(),
            ]
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(InstructionInputChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(InstructionInputChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(InstructionInputChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_publics,
            vec![JoltPublicId::from(InstructionInputPublic::EqProduct)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(InstructionInputPublic::EqProduct)]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn input_virtualization_evaluates_like_core_formula() {
        let claims = input_virtualization::<Fr>(trace_dimensions());

        let right_input = Fr::from_u64(3);
        let left_input = Fr::from_u64(5);
        let right_is_rs2 = Fr::from_u64(7);
        let rs2 = Fr::from_u64(11);
        let right_is_imm = Fr::from_u64(13);
        let imm_value = Fr::from_u64(17);
        let left_is_rs1 = Fr::from_u64(19);
        let rs1 = Fr::from_u64(23);
        let left_is_pc = Fr::from_u64(29);
        let pc = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let eq_product = Fr::from_u64(41);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == right_instruction_input_product() => right_input,
                id if id == left_instruction_input_product() => left_input,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == right_operand_is_rs2() => right_is_rs2,
                id if id == rs2_value() => rs2,
                id if id == right_operand_is_imm() => right_is_imm,
                id if id == imm() => imm_value,
                id if id == left_operand_is_rs1() => left_is_rs1,
                id if id == rs1_value() => rs1,
                id if id == left_operand_is_pc() => left_is_pc,
                id if id == unexpanded_pc() => pc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::InstructionInput(InstructionInputPublic::EqProduct) => eq_product,
                _ => zero,
            },
        );

        assert_eq!(input, right_input + gamma * left_input);
        assert_eq!(
            output,
            eq_product
                * (right_is_rs2 * rs2
                    + right_is_imm * imm_value
                    + gamma * left_is_rs1 * rs1
                    + gamma * left_is_pc * pc)
        );
    }

    #[test]
    fn read_raf_rejects_empty_dimensions() {
        assert!(InstructionReadRafDimensions::try_from((5, 128, 0)).is_err());
        assert!(InstructionReadRafDimensions::try_from((5, 0, 1)).is_err());
    }

    #[test]
    fn read_raf_address_layout_derives_virtual_ra_chunk_bits() {
        let layout = InstructionReadRafDimensions::try_from((5, 128, 4))
            .unwrap_or_else(|err| panic!("test read-RAF dimensions should be valid: {err}"))
            .address_layout()
            .unwrap_or_else(|err| panic!("address layout should derive: {err}"));

        assert_eq!(layout.address_bits(), 128);
        assert_eq!(layout.virtual_ra_chunk_bits(), 32);
    }

    #[test]
    fn read_raf_address_layout_rejects_invalid_widths() {
        assert_eq!(
            InstructionReadRafDimensions::try_from((5, 130, 4))
                .unwrap_or_else(|err| panic!("test read-RAF dimensions should be valid: {err}"))
                .address_layout(),
            Err(JoltFormulaDimensionsError::NotDivisible {
                value_name: "instruction_address_bits",
                value: 130,
                divisor_name: "instruction virtual RA polynomial count",
                divisor: 4,
            })
        );
        assert_eq!(
            InstructionReadRafDimensions::try_from((5, 192, 4))
                .unwrap_or_else(|err| panic!("test read-RAF dimensions should be valid: {err}"))
                .u128_address_layout(),
            Err(JoltFormulaDimensionsError::Exceeds {
                value_name: "instruction_address_bits",
                value: 192,
                max_name: "u128::BITS",
                max: 128,
            })
        );
    }

    #[test]
    fn read_raf_exposes_expected_dependencies() {
        let dimensions = read_raf_dimensions(2);
        let claims = read_raf::<Fr>(dimensions);
        let table_flags = lookup_table_flags();
        let table_value_publics = eq_table_value_publics();

        assert_eq!(claims.id, JoltRelationId::InstructionReadRaf);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(133, 4));
        assert_eq!(
            claims.input.required_openings,
            read_raf_input_openings().to_vec()
        );
        let mut expected_output_openings = vec![instruction_ra(0), instruction_ra(1)];
        expected_output_openings.extend(table_flags.iter().copied());
        expected_output_openings.push(instruction_raf_flag());
        assert_eq!(claims.output.required_openings, expected_output_openings);
        let output_openings = read_raf_output_openings(dimensions);
        assert_eq!(
            output_openings.instruction_ra,
            vec![instruction_ra(0), instruction_ra(1)]
        );
        assert_eq!(output_openings.lookup_table_flags, lookup_table_flags());
        assert_eq!(output_openings.instruction_raf_flag, instruction_raf_flag());
        assert_eq!(
            claims.consistency,
            read_raf_consistency_openings()
                .into_iter()
                .map(|(left, right)| JoltConsistencyClaim::same_evaluation(left, right))
                .collect::<Vec<_>>()
        );
        let mut expected_required_openings = vec![
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            instruction_ra(0),
            instruction_ra(1),
        ];
        expected_required_openings.extend(table_flags);
        expected_required_openings.extend([instruction_raf_flag(), lookup_output_product()]);
        assert_eq!(claims.required_openings(), expected_required_openings);
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(InstructionReadRafChallenge::Gamma)]
        );
        assert!(claims.output.required_challenges.is_empty());
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(InstructionReadRafChallenge::Gamma)]
        );
        let mut expected_output_publics = table_value_publics.clone();
        expected_output_publics.extend([
            JoltPublicId::from(InstructionReadRafPublic::EqRafConstant),
            JoltPublicId::from(InstructionReadRafPublic::EqRafFlag),
        ]);
        assert_eq!(claims.output.required_publics, expected_output_publics);
        let mut expected_required_publics = table_value_publics;
        expected_required_publics.extend([
            JoltPublicId::from(InstructionReadRafPublic::EqRafConstant),
            JoltPublicId::from(InstructionReadRafPublic::EqRafFlag),
        ]);
        assert_eq!(claims.required_publics(), expected_required_publics);
        assert_eq!(claims.num_challenges(), 1);
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
    fn read_raf_evaluates_like_core_formula() {
        let dimensions = read_raf_dimensions(2);
        let claims = read_raf::<Fr>(dimensions);

        let lookup_output = Fr::from_u64(3);
        let left_lookup_operand = Fr::from_u64(5);
        let right_lookup_operand = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let ra_0 = Fr::from_u64(2);
        let ra_1 = Fr::from_u64(3);
        let table_flags: Vec<_> = (0..LookupTableKind::<XLEN>::COUNT)
            .map(|i| Fr::from_u64(i as u64 + 5))
            .collect();
        let table_values: Vec<_> = (0..LookupTableKind::<XLEN>::COUNT)
            .map(|i| Fr::from_u64(2 * i as u64 + 13))
            .collect();
        let raf_constant = Fr::from_u64(23);
        let raf_flag_coeff = Fr::from_u64(29);
        let raf_flag = Fr::from_u64(31);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == lookup_output_reduced() => lookup_output,
                id if id == left_lookup_operand_reduced() => left_lookup_operand,
                id if id == right_lookup_operand_reduced() => right_lookup_operand,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == instruction_ra(0) => ra_0,
                id if id == instruction_ra(1) => ra_1,
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)),
                    relation: JoltRelationId::InstructionReadRaf,
                } => table_flags[index],
                id if id == instruction_raf_flag() => raf_flag,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::Gamma)
                | JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::InstructionReadRaf(InstructionReadRafPublic::EqTableValue(index)) => {
                    table_values[index]
                }
                JoltPublicId::InstructionReadRaf(InstructionReadRafPublic::EqRafConstant) => {
                    raf_constant
                }
                JoltPublicId::InstructionReadRaf(InstructionReadRafPublic::EqRafFlag) => {
                    raf_flag_coeff
                }
                _ => zero,
            },
        );

        assert_eq!(
            input,
            lookup_output + gamma * left_lookup_operand + gamma * gamma * right_lookup_operand
        );
        let table_sum = table_values
            .iter()
            .zip(table_flags.iter())
            .fold(zero, |sum, (value, flag)| sum + *value * *flag);
        assert_eq!(
            output,
            ra_0 * ra_1 * (table_sum + raf_constant + raf_flag_coeff * raf_flag)
        );
    }

    #[test]
    fn ra_virtualization_rejects_invalid_dimensions() {
        assert!(InstructionRaVirtualizationDimensions::try_from((5, 0, 1)).is_err());
        assert!(InstructionRaVirtualizationDimensions::try_from((5, 1, 0)).is_err());
        assert!(InstructionRaVirtualizationDimensions::try_from((5, usize::MAX, 2)).is_err());
    }

    #[test]
    fn ra_virtualization_exposes_expected_dependencies() {
        let dimensions = ra_virtualization_dimensions(3, 2);
        let claims = ra_virtualization::<Fr>(dimensions);

        assert_eq!(claims.id, JoltRelationId::InstructionRaVirtualization);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(5, 3));
        assert_eq!(
            claims.input.required_openings,
            ra_virtualization_input_openings(dimensions)
        );
        assert_eq!(
            claims.output.required_openings,
            ra_virtualization_output_openings(dimensions).all()
        );
        assert_eq!(
            ra_virtualization_output_openings(dimensions).committed_instruction_ra_by_virtual,
            vec![
                vec![committed_instruction_ra(0), committed_instruction_ra(1)],
                vec![committed_instruction_ra(2), committed_instruction_ra(3)],
                vec![committed_instruction_ra(4), committed_instruction_ra(5)],
            ]
        );
        assert!(claims.consistency.is_empty());
        assert_eq!(
            claims.required_openings(),
            vec![
                instruction_ra(0),
                instruction_ra(1),
                instruction_ra(2),
                committed_instruction_ra(0),
                committed_instruction_ra(1),
                committed_instruction_ra(2),
                committed_instruction_ra(3),
                committed_instruction_ra(4),
                committed_instruction_ra(5),
            ]
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(
                InstructionRaVirtualizationChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(
                InstructionRaVirtualizationChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                InstructionRaVirtualizationChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.challenge_index(JoltChallengeId::from(
                InstructionRaVirtualizationChallenge::Gamma
            )),
            Some(0)
        );
        assert_eq!(
            claims.output.required_publics,
            vec![JoltPublicId::from(
                InstructionRaVirtualizationPublic::EqCycle
            )]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(
                InstructionRaVirtualizationPublic::EqCycle
            )]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn ra_virtualization_preserves_gamma_dependency_for_single_virtual_ra() {
        let dimensions = ra_virtualization_dimensions(1, 1);
        let claims = ra_virtualization::<Fr>(dimensions);

        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(
                InstructionRaVirtualizationChallenge::Gamma
            )]
        );
        assert_eq!(claims.output.required_challenges, vec![]);
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                InstructionRaVirtualizationChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.output.required_publics,
            vec![JoltPublicId::from(
                InstructionRaVirtualizationPublic::EqCycle
            )]
        );
    }

    #[test]
    fn ra_virtualization_evaluates_like_core_formula() {
        let dimensions = ra_virtualization_dimensions(3, 2);
        let claims = ra_virtualization::<Fr>(dimensions);

        let virtual_ra = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        let committed_ra = [
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
            Fr::from_u64(23),
            Fr::from_u64(29),
        ];
        let gamma = Fr::from_u64(31);
        let eq_cycle = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Virtual(JoltVirtualPolynomial::InstructionRa(i)),
                    relation: JoltRelationId::InstructionReadRaf,
                } => virtual_ra[i],
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Committed(JoltCommittedPolynomial::InstructionRa(i)),
                    relation: JoltRelationId::InstructionRaVirtualization,
                } => committed_ra[i],
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::InstructionRaVirtualization(
                    InstructionRaVirtualizationPublic::EqCycle,
                ) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            virtual_ra[0] + gamma * virtual_ra[1] + gamma * gamma * virtual_ra[2]
        );
        assert_eq!(
            output,
            eq_cycle
                * (committed_ra[0] * committed_ra[1]
                    + gamma * committed_ra[2] * committed_ra[3]
                    + gamma * gamma * committed_ra[4] * committed_ra[5])
        );
    }

    #[test]
    fn ra_virtualization_eq_cycle_polynomial_reverses_read_raf_cycle() {
        let read_raf_cycle = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let eq_point = vec![Fr::from_u64(5), Fr::from_u64(3), Fr::from_u64(2)];

        assert_eq!(
            ra_virtualization_eq_cycle_polynomial(&read_raf_cycle).evals(),
            EqPolynomial::<Fr>::evals(&eq_point, None)
        );
    }
}
