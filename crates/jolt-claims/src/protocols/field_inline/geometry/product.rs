use jolt_field::Ring;

use super::super::{FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial};

pub fn field_product_input_openings() -> [FieldInlineOpeningId; 1] {
    [field_product_opening()]
}

pub fn field_product_output_openings() -> [FieldInlineOpeningId; 2] {
    [field_rs1_value_product(), field_rs2_value_product()]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldRegistersProductLane {
    Product,
    InverseProduct,
}

impl FieldRegistersProductLane {
    pub fn input_opening(self) -> FieldInlineOpeningId {
        match self {
            Self::Product => field_product_opening(),
            Self::InverseProduct => field_inv_product_opening(),
        }
    }

    pub fn factor_openings(self) -> [FieldInlineOpeningId; 2] {
        match self {
            Self::Product => [field_rs1_value_product(), field_rs2_value_product()],
            Self::InverseProduct => [field_rs1_value_product(), field_rd_value_product()],
        }
    }
}

pub const fn selected_product_lanes() -> [FieldRegistersProductLane; 2] {
    [
        FieldRegistersProductLane::Product,
        FieldRegistersProductLane::InverseProduct,
    ]
}

pub fn selected_product_uniskip_input_openings() -> [FieldInlineOpeningId; 2] {
    selected_product_lanes().map(FieldRegistersProductLane::input_opening)
}

pub fn selected_product_remainder_output_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rs1_value_product(),
        field_rs2_value_product(),
        field_rd_value_product(),
    ]
}

/// The FR lanes' input values entering the composed product-uniskip input
/// claim: the `FieldProduct`/`FieldInvProduct` openings from the FR
/// Spartan-outer segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldProductLaneInputs<F> {
    pub product: F,
    pub inv_product: F,
}

impl<F: Ring> FieldProductLaneInputs<F> {
    fn input_value(&self, lane: FieldRegistersProductLane) -> F {
        match lane {
            FieldRegistersProductLane::Product => self.product,
            FieldRegistersProductLane::InverseProduct => self.inv_product,
        }
    }
}

/// The FR lanes' factor values entering the composed product-remainder output
/// claim: the three FR product-remainder row openings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldProductLaneFactors<F> {
    pub rs1_value: F,
    pub rs2_value: F,
    pub rd_value: F,
}

impl<F: Ring> FieldProductLaneFactors<F> {
    /// The lane's `(left, right)` factor values, in
    /// [`FieldRegistersProductLane::factor_openings`] order.
    fn factor_values(&self, lane: FieldRegistersProductLane) -> [F; 2] {
        match lane {
            FieldRegistersProductLane::Product => [self.rs1_value, self.rs2_value],
            FieldRegistersProductLane::InverseProduct => [self.rs1_value, self.rd_value],
        }
    }
}

/// The FR lanes' contribution to the composed product-uniskip input claim:
/// `Σ_i weights[base_lanes + i] · input_i` over the selected lane order, where
/// `weights` are the composed centered-domain Lagrange weights and the ordinary
/// product lanes occupy indices `[0, base_lanes)`. `None` if `weights` does not
/// cover the composed lane domain.
pub fn composed_uniskip_input_contribution<F: Ring>(
    weights: &[F],
    base_lanes: usize,
    inputs: &FieldProductLaneInputs<F>,
) -> Option<F> {
    let mut contribution = F::zero();
    for (index, lane) in selected_product_lanes().into_iter().enumerate() {
        let weight = *weights.get(base_lanes.checked_add(index)?)?;
        contribution += weight * inputs.input_value(lane);
    }
    Some(contribution)
}

/// The FR lanes' contributions to the composed product-remainder factors,
/// returned as `(left, right)`: each selected lane adds its composed weight
/// times its left/right factor value (the [`FieldRegistersProductLane::factor_openings`]
/// order). The composed remainder output claim is then
/// `tau_kernel · (ordinary_left + left) · (ordinary_right + right)`. `None` if
/// `weights` does not cover the composed lane domain.
pub fn composed_remainder_factor_contributions<F: Ring>(
    weights: &[F],
    base_lanes: usize,
    factors: &FieldProductLaneFactors<F>,
) -> Option<(F, F)> {
    let mut left = F::zero();
    let mut right = F::zero();
    for (index, lane) in selected_product_lanes().into_iter().enumerate() {
        let weight = *weights.get(base_lanes.checked_add(index)?)?;
        let [lane_left, lane_right] = factors.factor_values(lane);
        left += weight * lane_left;
        right += weight * lane_right;
    }
    Some((left, right))
}

pub(crate) fn field_product_opening() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldProduct,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_inv_product_opening() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldInvProduct,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_rs1_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_rs2_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_rd_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

#[cfg(test)]
#[expect(clippy::panic, reason = "tests may unwind via panic")]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    fn inputs() -> FieldProductLaneInputs<Fr> {
        FieldProductLaneInputs {
            product: Fr::from_u64(3),
            inv_product: Fr::from_u64(5),
        }
    }

    fn factors() -> FieldProductLaneFactors<Fr> {
        FieldProductLaneFactors {
            rs1_value: Fr::from_u64(7),
            rs2_value: Fr::from_u64(11),
            rd_value: Fr::from_u64(13),
        }
    }

    /// The composed helpers weight the selected lanes at the indices following
    /// the ordinary lanes, and each lane's factor mapping follows
    /// `FieldRegistersProductLane::factor_openings` (Product: rs1·rs2;
    /// InverseProduct: rs1·rd — the FINV guarded-inverse witness).
    #[test]
    fn composed_contributions_follow_selected_lane_order() {
        let base_lanes = 3usize;
        let weights = (1..=5).map(Fr::from_u64).collect::<Vec<_>>();
        let inputs = inputs();
        let factors = factors();
        let w3 = Fr::from_u64(4);
        let w4 = Fr::from_u64(5);

        assert_eq!(
            composed_uniskip_input_contribution(&weights, base_lanes, &inputs),
            Some(w3 * inputs.product + w4 * inputs.inv_product)
        );
        assert_eq!(
            composed_remainder_factor_contributions(&weights, base_lanes, &factors),
            Some((
                w3 * factors.rs1_value + w4 * factors.rs1_value,
                w3 * factors.rs2_value + w4 * factors.rd_value,
            ))
        );
    }

    /// The value carriers' per-lane mapping is the id-level lane table read
    /// back as values: `factor_values(lane)` resolves exactly
    /// `lane.factor_openings()`, and `input_value(lane)` resolves
    /// `lane.input_opening()`'s polynomial.
    #[test]
    fn lane_value_mapping_matches_lane_opening_table() {
        let inputs = inputs();
        let factors = factors();
        let resolve = |id: FieldInlineOpeningId| -> Fr {
            if id == field_rs1_value_product() {
                factors.rs1_value
            } else if id == field_rs2_value_product() {
                factors.rs2_value
            } else if id == field_rd_value_product() {
                factors.rd_value
            } else if id == field_product_opening() {
                inputs.product
            } else if id == field_inv_product_opening() {
                inputs.inv_product
            } else {
                panic!("unexpected opening {id:?}")
            }
        };

        for lane in selected_product_lanes() {
            assert_eq!(
                factors.factor_values(lane),
                lane.factor_openings().map(resolve),
            );
            assert_eq!(inputs.input_value(lane), resolve(lane.input_opening()));
        }
    }

    /// A weight vector that does not cover the composed lane domain is a miss,
    /// never a silent truncation.
    #[test]
    fn composed_contributions_reject_short_weight_vectors() {
        let base_lanes = 3usize;
        let short = (1..=4).map(Fr::from_u64).collect::<Vec<_>>();

        assert_eq!(
            composed_uniskip_input_contribution(&short, base_lanes, &inputs()),
            None
        );
        assert_eq!(
            composed_remainder_factor_contributions(&short, base_lanes, &factors()),
            None
        );
    }
}
