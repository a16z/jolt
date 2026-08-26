use std::num::NonZeroUsize;

use jolt_field::{JoltField, Ring};
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

/// Whether instruction read-RAF must pin the lookup address to its canonical
/// representative in `[0, p)`.
///
/// The identity-RAF leg ties the committed `2^(2·XLEN)`-cell address to the
/// right lookup operand only *modulo* the field characteristic, so it is
/// injective exactly when `p >= 2^(2·XLEN)`. BN254's scalar field clears
/// `2^128` by 126 bits, and an aliased address `k + p ~ 2^254` is not even
/// representable in 128 one-hot address variables. The Akita fp128 field is
/// `p = 2^128 - 2^32 + 22537`, which falls short of `2^128` by 4_294_944_759,
/// giving every honest index below that window a second committable preimage
/// that the lookup table reads differently. See [`upper_half_all_ones`] for the
/// predicate that closes it.
///
/// This is a property of the *field*, not of the commitment scheme; `akita`
/// is the feature that selects fp128 and is the only configuration where the
/// precondition fails today. Read it from here rather than re-deriving
/// `cfg!(feature = "akita")` downstream — `jolt-kernels` has no `akita`
/// feature of its own, so a local `cfg!` there would silently be `false` and
/// desynchronize the prover from the verifier.
pub const CANONICAL_INSTRUCTION_ADDRESS: bool = cfg!(feature = "akita");

/// The multilinear extension of `1[high_XLEN(k) == 2^XLEN - 1]`, evaluated at
/// an instruction read-RAF address point.
///
/// `r_address` is most-significant-coordinate first (`IdentityPolynomial` folds
/// `r[i]` with weight `2^(len-1-i)`), so the upper half of the index is the
/// *leading* half of the point and the AND of those bits extends to their
/// product. Taking the trailing half instead would both miss the aliases and
/// reject honest cycles — `SUB(2^64-1, 0)` has index `0x1_FFFF_FFFF_FFFF_FFFF`,
/// whose low limb is all ones.
pub fn upper_half_all_ones<F: JoltField>(r_address: &[F]) -> F {
    r_address[..r_address.len() / 2]
        .iter()
        .copied()
        .fold(F::one(), |acc, coordinate| acc * coordinate)
}

#[cfg(test)]
mod canonical_address_tests {
    /// `p = 2^128 - 2^32 + 22537`, the Akita fp128 modulus.
    const P: u128 = u128::MAX - (1 << 32) + 22537 + 1;
    /// `c = 2^128 - p`, the width of the alias window.
    const C: u128 = (1 << 32) - 22537;
    /// The smallest address the predicate rejects.
    const FIRST_ALL_ONES: u128 = u128::MAX - (u64::MAX as u128);

    /// The load-bearing inequality, as a compile-time fact: the rejected band
    /// `[2^128 - 2^64, 2^128)` strictly contains the non-canonical band
    /// `[p, 2^128)`. Hence `U(k) = 0 ⟹ k < p`.
    const _: () = assert!(FIRST_ALL_ONES < P);

    fn upper_all_ones(k: u128) -> bool {
        (k >> 64) == u64::MAX as u128
    }

    /// The two arithmetic facts the whole construction rests on.
    ///
    /// 1. *Every alias is caught.* `k = r + p < 2^128` forces `r < c`, and since
    ///    `p`'s low limb is `2^64 - c`, adding `r < c` cannot carry into the high
    ///    limb — so every alias has an all-ones upper half.
    /// 2. *Nothing else is needed.* `U(k) = 0` means `k < 2^128 - 2^64`, and that
    ///    bound is strictly below `p`. So a surviving address is the canonical
    ///    representative of its class outright — the identity leg then pins it
    ///    exactly, with no range bound on `RightLookupOperand` required. That
    ///    matters because advice rows have `RafFlag = 1` but no R1CS constraint
    ///    on their right operand at all.
    #[test]
    fn canonical_predicate_separates_aliases_from_honest_indices() {
        assert_eq!(P, 2u128.pow(127) + (2u128.pow(127) - (1 << 32) + 22537));
        assert_eq!(C, u128::MAX - P + 1);
        assert_eq!(C, 4_294_944_759);

        // (1) every representable alias has an all-ones upper half
        for r in [0u128, 1, 2, C / 2, C - 2, C - 1] {
            let alias = P + r;
            assert_eq!(alias % P, r, "alias for r={r} is not congruent to r");
            assert!(
                upper_all_ones(alias),
                "alias for r={r} escapes the predicate"
            );
        }
        // r = c is the first value whose alias overflows 2^128, i.e. is not
        // representable as a committed address at all.
        assert!(P.checked_add(C).is_none());

        // (2) surviving addresses are canonical: !U(k) => k < p (see the
        // compile-time assertion above for the inequality itself).
        assert!(!upper_all_ones(FIRST_ALL_ONES - 1));
        assert!(upper_all_ones(FIRST_ALL_ONES));

        // Honest bounds for every non-interleaved family stay below that line.
        let add_max = 2u128 * (u64::MAX as u128); // x + y
        let sub_max = (u64::MAX as u128) + (1u128 << 64); // x + 2^64 - y
        let mul_max = (u64::MAX as u128) * (u64::MAX as u128);
        let advice_max = u64::MAX as u128;
        for (name, k) in [
            ("add", add_max),
            ("sub", sub_max),
            ("mul", mul_max),
            ("advice", advice_max),
        ] {
            assert!(!upper_all_ones(k), "{name} boundary is falsely flagged");
            assert!(k < FIRST_ALL_ONES, "{name} boundary escapes the bound");
        }
    }
}

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

    pub fn opening_point<F: JoltField>(
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
pub struct InstructionReadRafOpeningPoint<F: JoltField> {
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
        let _sumcheck_degree = num_committed_per_virtual.get().checked_add(1).ok_or(
            JoltFormulaDimensionsError::Overflow {
                name: "instruction RA virtualization sumcheck degree",
            },
        )?;
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
    F: Ring,
{
    let mut sum = JoltExpr::zero();
    for i in 0..dimensions.num_virtual_ra_polys() {
        sum = sum + gamma.clone().pow(i) * opening(instruction_ra(i));
    }
    sum
}

pub(crate) fn instruction_ra_product<F>(dimensions: InstructionReadRafDimensions) -> JoltExpr<F>
where
    F: Ring,
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
    F: Ring,
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
    use jolt_field::{Fr, Ring};

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
        assert!(InstructionRaVirtualizationDimensions::try_from((5, 1, usize::MAX)).is_err());
        assert!(InstructionRaVirtualizationDimensions::try_from((5, 1, usize::MAX - 1)).is_ok());
    }
}
