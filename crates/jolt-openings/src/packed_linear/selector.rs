use jolt_field::Field;
use jolt_poly::EqPolynomial;

use crate::{BatchOpeningStatement, OpeningsError, PackedLinearTerm, PhysicalView};

use super::types::{PackedLinearAddress, PackedLinearFamily, PackedLinearLayout};
use super::util::{checked_domain_size, invalid_batch, log2_power_of_two, offset_bit};

fn family_for_term<F, L>(
    layout: &L,
    term: &PackedLinearTerm<F>,
) -> Result<PackedLinearFamily, OpeningsError>
where
    L: PackedLinearLayout,
{
    layout
        .family(term.family)?
        .ok_or_else(|| invalid_batch("packed linear term references an unknown family"))
}

pub(super) fn validate_term<F, L>(
    layout: &L,
    term: &PackedLinearTerm<F>,
) -> Result<(), OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let family = family_for_term(layout, term)?;
    let row_vars = log2_power_of_two(family.rows, "packed family rows")?;
    if term.row_point.len() != row_vars {
        return Err(invalid_batch(format!(
            "packed linear term row point has {} variables but family requires {row_vars}",
            term.row_point.len()
        )));
    }
    if !family.alphabet_size.is_power_of_two() {
        return Err(invalid_batch(
            "packed linear verifier currently requires power-of-two alphabets",
        ));
    }
    if !family.limbs.is_power_of_two() {
        return Err(invalid_batch(
            "packed linear verifier currently requires power-of-two limb counts",
        ));
    }
    layout
        .rank(PackedLinearAddress {
            family: term.family,
            row: 0,
            limb: term.limb,
            symbol: term.symbol,
        })
        .map(|_| ())
}

pub(super) fn logical_coefficients<F, C, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
) -> Vec<F>
where
    F: Field,
{
    statement
        .claims
        .iter()
        .zip(gamma_powers)
        .map(|(claim, gamma)| *gamma * claim.scale)
        .collect()
}

pub(super) fn reduced_claim<F, C, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
) -> F
where
    F: Field,
{
    statement
        .claims
        .iter()
        .zip(gamma_powers)
        .fold(F::zero(), |acc, (claim, gamma)| {
            acc + *gamma * claim.scale * claim.claim
        })
}

pub(super) fn packed_selector_evals<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
) -> Result<Vec<F>, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let domain_size = checked_domain_size(layout.dimension())?;
    let mut selector = vec![F::zero(); domain_size];
    for (claim, gamma) in statement.claims.iter().zip(gamma_powers) {
        let PhysicalView::PackedLinear { terms, .. } = &claim.view else {
            return Err(invalid_batch(
                "packed linear selector requires PackedLinear views",
            ));
        };
        let claim_weight = *gamma * claim.scale;
        for term in terms {
            let family = family_for_term(layout, term)?;
            let row_weights = EqPolynomial::new(term.row_point.clone()).evaluations();
            if row_weights.len() != family.rows {
                return Err(invalid_batch(
                    "packed linear term row point does not match family row count",
                ));
            }
            let weight = claim_weight * term.coefficient;
            for (row, row_weight) in row_weights.iter().copied().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let rank = layout.rank(PackedLinearAddress {
                    family: term.family,
                    row,
                    limb: term.limb,
                    symbol: term.symbol,
                })?;
                selector[rank] += weight * row_weight;
            }
        }
    }
    Ok(selector)
}

pub(super) fn packed_selector_eval<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    point: &[F],
) -> Result<F, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    if point.len() != layout.dimension() {
        return Err(invalid_batch(format!(
            "packed linear selector point has {} variables but layout has {}",
            point.len(),
            layout.dimension()
        )));
    }
    let mut result = F::zero();
    for (claim, gamma) in statement.claims.iter().zip(gamma_powers) {
        let PhysicalView::PackedLinear { terms, .. } = &claim.view else {
            return Err(invalid_batch(
                "packed linear selector requires PackedLinear views",
            ));
        };
        let claim_weight = *gamma * claim.scale;
        for term in terms {
            result += packed_term_selector_eval(layout, term, point)? * claim_weight;
        }
    }
    Ok(result)
}

fn packed_term_selector_eval<F, L>(
    layout: &L,
    term: &PackedLinearTerm<F>,
    point: &[F],
) -> Result<F, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let family = family_for_term(layout, term)?;
    let row_vars = log2_power_of_two(family.rows, "packed family rows")?;
    if term.row_point.len() != row_vars {
        return Err(invalid_batch(format!(
            "packed linear term row point has {} variables but family requires {row_vars}",
            term.row_point.len()
        )));
    }
    let alphabet_vars = log2_power_of_two(family.alphabet_size, "packed alphabet")?;
    let limb_vars = log2_power_of_two(family.limbs, "packed limb count")?;
    let factors = [
        SelectorFactor::Fixed {
            value: term.symbol,
            bits: alphabet_vars,
        },
        SelectorFactor::Fixed {
            value: term.limb,
            bits: limb_vars,
        },
        SelectorFactor::RowEq {
            point: &term.row_point,
        },
    ];
    selector_eval_with_offset(point, family.offset, term.coefficient, &factors)
}

#[derive(Clone, Copy)]
enum SelectorFactor<'a, F> {
    Fixed { value: usize, bits: usize },
    RowEq { point: &'a [F] },
}

impl<F> SelectorFactor<'_, F>
where
    F: Field,
{
    fn bits(self) -> usize {
        match self {
            Self::Fixed { bits, .. } => bits,
            Self::RowEq { point } => point.len(),
        }
    }

    fn bit_weight(self, bit_index: usize, bit: usize) -> F {
        match self {
            Self::Fixed { value, .. } => {
                if ((value >> bit_index) & 1) == bit {
                    F::one()
                } else {
                    F::zero()
                }
            }
            Self::RowEq { point } => {
                let challenge = point[point.len() - 1 - bit_index];
                if bit == 1 {
                    challenge
                } else {
                    F::one() - challenge
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct CarryMatrix<F>([[F; 2]; 2]);

impl<F> CarryMatrix<F>
where
    F: Field,
{
    fn identity() -> Self {
        Self([[F::one(), F::zero()], [F::zero(), F::one()]])
    }

    fn zero() -> Self {
        Self([[F::zero(); 2]; 2])
    }

    fn add_assign(&mut self, carry_in: usize, carry_out: usize, value: F) {
        self.0[carry_in][carry_out] += value;
    }

    fn mul(&self, rhs: &Self) -> Self {
        let a = &self.0;
        let b = &rhs.0;
        Self([
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ])
    }
}

fn selector_eval_with_offset<F>(
    point: &[F],
    offset: usize,
    scale: F,
    factors: &[SelectorFactor<'_, F>],
) -> Result<F, OpeningsError>
where
    F: Field,
{
    let total_bits = factors.iter().map(|factor| factor.bits()).sum::<usize>();
    if total_bits > point.len() {
        return Err(invalid_batch(format!(
            "packed linear selector needs {total_bits} bits but point has {}",
            point.len()
        )));
    }

    let mut matrix = CarryMatrix::identity();
    let mut bit_cursor = 0usize;
    for &factor in factors {
        for bit_index in 0..factor.bits() {
            let bit_matrix = selector_bit_matrix(
                point[bit_cursor],
                offset_bit(offset, bit_cursor),
                factor,
                bit_index,
            );
            matrix = matrix.mul(&bit_matrix);
            bit_cursor += 1;
        }
    }
    for (bit_index, &challenge) in point.iter().enumerate().skip(bit_cursor) {
        let bit_matrix = fixed_zero_bit_matrix(challenge, offset_bit(offset, bit_index));
        matrix = matrix.mul(&bit_matrix);
    }
    Ok(scale * matrix.0[0][0])
}

pub(super) fn native_opening_point<F: Copy>(sumcheck_point_lsb: &[F]) -> Vec<F> {
    sumcheck_point_lsb.iter().rev().copied().collect()
}

fn selector_bit_matrix<F>(
    challenge: F,
    offset_bit: bool,
    factor: SelectorFactor<'_, F>,
    factor_bit_index: usize,
) -> CarryMatrix<F>
where
    F: Field,
{
    let mut matrix = CarryMatrix::zero();
    for local_bit in 0..=1 {
        let factor_weight = factor.bit_weight(factor_bit_index, local_bit);
        if factor_weight.is_zero() {
            continue;
        }
        add_transition(&mut matrix, challenge, offset_bit, local_bit, factor_weight);
    }
    matrix
}

fn fixed_zero_bit_matrix<F>(challenge: F, offset_bit: bool) -> CarryMatrix<F>
where
    F: Field,
{
    let mut matrix = CarryMatrix::zero();
    add_transition(&mut matrix, challenge, offset_bit, 0, F::one());
    matrix
}

fn add_transition<F>(
    matrix: &mut CarryMatrix<F>,
    challenge: F,
    offset_bit: bool,
    local_bit: usize,
    scale: F,
) where
    F: Field,
{
    for carry_in in 0..=1 {
        let sum = usize::from(offset_bit) + local_bit + carry_in;
        let output_bit = sum & 1;
        let carry_out = sum >> 1;
        let eq_weight = if output_bit == 1 {
            challenge
        } else {
            F::one() - challenge
        };
        matrix.add_assign(carry_in, carry_out, scale * eq_weight);
    }
}
