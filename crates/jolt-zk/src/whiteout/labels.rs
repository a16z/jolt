//! Row labels for the Σ-protocol's aggregation (Whiteout Protocol
//! `wo:prot:sigma`, step 1; §4.2, "Verifier cost of the zero rows").
//!
//! Every row of the lifted relation gets a distinct nonzero label
//! `a ∈ {0,1}^ℓ`, and the verifier's ℓ field challenges `ρ` weight it by
//! `μ_a = Π_{j: a_j = 1} ρ_j`. Label 0 is the constant monomial, reserved for
//! `ct(g)`. Operand `o` owns the block `64·o + r` for `r ∈ [1, 63]` (its
//! scalarity rows); label `64·o` is unused. Product rows follow from
//! `max(64·n_ops, 1)` in relation order, then affine rows.

use jolt_algebra::Field;

use super::WhiteoutError;

/// The label assignment of one relation shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowLabels {
    operands: usize,
    products: usize,
    affine: usize,
    base: usize,
    bits: usize,
}

impl RowLabels {
    pub fn new(operands: usize, products: usize, affine: usize) -> Result<Self, WhiteoutError> {
        let overflow = WhiteoutError::InvalidRelation("row labels overflow usize");
        let base = operands.checked_mul(64).ok_or(overflow.clone())?.max(1);
        let end = base
            .checked_add(products)
            .and_then(|end| end.checked_add(affine))
            .ok_or(overflow)?;
        let bits = (usize::BITS - end.saturating_sub(1).leading_zeros()) as usize;
        Ok(Self {
            operands,
            products,
            affine,
            base,
            bits: bits.max(1),
        })
    }

    /// `ℓ`, the number of aggregation challenges.
    pub fn num_challenges(&self) -> usize {
        self.bits
    }

    pub fn product(&self, index: usize) -> usize {
        self.base + index
    }

    pub fn affine(&self, index: usize) -> usize {
        self.base + self.products + index
    }

    /// Scalarity row `ct(X^{−r} P_x) = 0` of operand `operand`, `r ∈ [1, 63]`.
    pub fn scalarity(operand: usize, r: usize) -> usize {
        64 * operand + r
    }

    /// `μ_label = Π_{j: bit j of label} ρ_j`.
    pub fn weight<F: Field>(&self, label: usize, rho: &[F]) -> Result<F, WhiteoutError> {
        if rho.len() != self.bits {
            return Err(WhiteoutError::InvalidRelation(
                "aggregation challenge count differs from the labels",
            ));
        }
        Ok(rho
            .iter()
            .enumerate()
            .filter(|(bit, _)| (label >> bit) & 1 == 1)
            .fold(F::one(), |weight, (_, rho)| weight * *rho))
    }
}
