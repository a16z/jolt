use jolt_field::Field;
use serde::{Deserialize, Serialize};

pub use super::error::{DoryAssistFormulaDimensionsError, DoryAssistFormulaPointError};

pub const GT_ELEMENT_VARS: usize = 4;
pub const GT_EXP_BASE: usize = 4;
pub const GT_EXP_STEP_VARS: usize = 7;
pub const EC_SCALAR_MUL_STEP_VARS: usize = 8;
pub const BN254_MILLER_LOOP_LINE_EVENTS: usize = 87;
pub const BN254_MILLER_LOOP_SQUARE_OPS: usize = 63;
pub const BN254_MILLER_LOOP_ACCUMULATOR_OPS: usize =
    BN254_MILLER_LOOP_LINE_EVENTS + BN254_MILLER_LOOP_SQUARE_OPS;
pub const BN254_MILLER_LOOP_LINE_EVENT_VARS: usize = 7;
pub const BN254_MILLER_LOOP_ACCUMULATOR_OP_VARS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoryAssistSumcheckDomain {
    BooleanHypercube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistSumcheckSpec {
    pub domain: DoryAssistSumcheckDomain,
    pub rounds: usize,
    pub degree: usize,
}

impl DoryAssistSumcheckSpec {
    pub const fn boolean(rounds: usize, degree: usize) -> Self {
        Self {
            domain: DoryAssistSumcheckDomain::BooleanHypercube,
            rounds,
            degree,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistDimensions {
    pub gt: GtDimensions,
    pub g1: G1Dimensions,
    pub g2: G2Dimensions,
    pub miller_loop: MillerLoopDimensions,
    pub dory_reduce: DoryReduceDimensions,
    pub wiring: WiringDimensions,
    pub packing: PrefixPackingDimensions,
}

impl DoryAssistDimensions {
    pub const fn new(
        gt: GtDimensions,
        g1: G1Dimensions,
        g2: G2Dimensions,
        miller_loop: MillerLoopDimensions,
        dory_reduce: DoryReduceDimensions,
        wiring: WiringDimensions,
        packing: PrefixPackingDimensions,
    ) -> Self {
        Self {
            gt,
            g1,
            g2,
            miller_loop,
            dory_reduce,
            wiring,
            packing,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GtDimensions {
    exp_steps: usize,
    exp_instances: usize,
    mul_instances: usize,
}

impl GtDimensions {
    pub const fn new(
        exp_step_vars: usize,
        exp_instance_vars: usize,
        mul_instance_vars: usize,
    ) -> Self {
        Self {
            exp_steps: exp_step_vars,
            exp_instances: exp_instance_vars,
            mul_instances: mul_instance_vars,
        }
    }

    pub const fn exp_step_vars(self) -> usize {
        self.exp_steps
    }

    pub const fn exp_instance_vars(self) -> usize {
        self.exp_instances
    }

    pub const fn mul_instance_vars(self) -> usize {
        self.mul_instances
    }

    pub const fn gt_element_vars(self) -> usize {
        GT_ELEMENT_VARS
    }

    pub const fn exp_native_rounds(self) -> usize {
        self.exp_steps + GT_ELEMENT_VARS
    }

    pub const fn exp_batched_rounds(self) -> usize {
        self.exp_native_rounds() + self.exp_instances
    }

    pub const fn exp_base_power_rounds(self) -> usize {
        GT_ELEMENT_VARS + self.exp_instances
    }

    pub const fn mul_rounds(self) -> usize {
        GT_ELEMENT_VARS + self.mul_instances
    }

    pub const fn exp_sumcheck(self, degree: usize) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.exp_batched_rounds(), degree)
    }

    pub const fn exp_digit_selector_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.exp_batched_rounds(), 3)
    }

    pub const fn exp_digit_bitness_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.exp_batched_rounds(), 2)
    }

    pub const fn exp_base_power_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.exp_base_power_rounds(), 2)
    }

    pub const fn exp_shift_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.exp_batched_rounds(), 2)
    }

    pub const fn exp_boundary_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.exp_batched_rounds(), 2)
    }

    pub const fn mul_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.mul_rounds(), 2)
    }
}

impl Default for GtDimensions {
    fn default() -> Self {
        Self::new(GT_EXP_STEP_VARS, 0, 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct G1Dimensions {
    scalar_mul_steps: usize,
    scalar_mul_instances: usize,
    add_instances: usize,
}

impl G1Dimensions {
    pub const fn new(
        scalar_mul_step_vars: usize,
        scalar_mul_instance_vars: usize,
        add_instance_vars: usize,
    ) -> Self {
        Self {
            scalar_mul_steps: scalar_mul_step_vars,
            scalar_mul_instances: scalar_mul_instance_vars,
            add_instances: add_instance_vars,
        }
    }

    pub const fn scalar_mul_step_vars(self) -> usize {
        self.scalar_mul_steps
    }

    pub const fn scalar_mul_instance_vars(self) -> usize {
        self.scalar_mul_instances
    }

    pub const fn add_instance_vars(self) -> usize {
        self.add_instances
    }

    pub const fn scalar_mul_rounds(self) -> usize {
        self.scalar_mul_steps + self.scalar_mul_instances
    }

    pub const fn add_rounds(self) -> usize {
        self.add_instances
    }

    pub const fn scalar_mul_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.scalar_mul_rounds(), 5)
    }

    pub const fn scalar_mul_shift_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.scalar_mul_steps, 2)
    }

    pub const fn scalar_mul_boundary_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.scalar_mul_rounds(), 2)
    }

    pub const fn add_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.add_rounds(), 5)
    }
}

impl Default for G1Dimensions {
    fn default() -> Self {
        Self::new(EC_SCALAR_MUL_STEP_VARS, 0, 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2Dimensions {
    scalar_mul_steps: usize,
    scalar_mul_instances: usize,
    add_instances: usize,
}

impl G2Dimensions {
    pub const fn new(
        scalar_mul_step_vars: usize,
        scalar_mul_instance_vars: usize,
        add_instance_vars: usize,
    ) -> Self {
        Self {
            scalar_mul_steps: scalar_mul_step_vars,
            scalar_mul_instances: scalar_mul_instance_vars,
            add_instances: add_instance_vars,
        }
    }

    pub const fn scalar_mul_step_vars(self) -> usize {
        self.scalar_mul_steps
    }

    pub const fn scalar_mul_instance_vars(self) -> usize {
        self.scalar_mul_instances
    }

    pub const fn add_instance_vars(self) -> usize {
        self.add_instances
    }

    pub const fn scalar_mul_rounds(self) -> usize {
        self.scalar_mul_steps + self.scalar_mul_instances
    }

    pub const fn add_rounds(self) -> usize {
        self.add_instances
    }

    pub const fn scalar_mul_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.scalar_mul_rounds(), 5)
    }

    pub const fn scalar_mul_shift_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.scalar_mul_steps, 2)
    }

    pub const fn scalar_mul_boundary_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.scalar_mul_rounds(), 2)
    }

    pub const fn add_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.add_rounds(), 5)
    }
}

impl Default for G2Dimensions {
    fn default() -> Self {
        Self::new(EC_SCALAR_MUL_STEP_VARS, 0, 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MillerLoopDimensions {
    line_events: usize,
    pairs: usize,
    accumulator_ops: usize,
}

impl MillerLoopDimensions {
    pub const fn new(line_event_vars: usize, pair_vars: usize, accumulator_op_vars: usize) -> Self {
        Self {
            line_events: line_event_vars,
            pairs: pair_vars,
            accumulator_ops: accumulator_op_vars,
        }
    }

    pub const fn line_event_vars(self) -> usize {
        self.line_events
    }

    pub const fn pair_vars(self) -> usize {
        self.pairs
    }

    pub const fn accumulator_op_vars(self) -> usize {
        self.accumulator_ops
    }

    pub const fn line_step_rounds(self) -> usize {
        self.line_events + self.pairs
    }

    pub const fn line_evaluation_rounds(self) -> usize {
        self.line_events + self.pairs
    }

    pub const fn pair_product_rounds(self) -> usize {
        self.line_events + self.pairs
    }

    pub const fn accumulator_rounds(self) -> usize {
        self.accumulator_ops
    }

    pub const fn boundary_rounds(self) -> usize {
        self.accumulator_ops
    }

    pub const fn line_step_sumcheck(self, degree: usize) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.line_step_rounds(), degree)
    }

    pub const fn line_evaluation_sumcheck(self, degree: usize) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.line_evaluation_rounds(), degree)
    }

    pub const fn pair_product_sumcheck(self, degree: usize) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.pair_product_rounds(), degree)
    }

    pub const fn accumulator_sumcheck(self, degree: usize) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.accumulator_rounds(), degree)
    }

    pub const fn boundary_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.boundary_rounds(), 2)
    }
}

impl Default for MillerLoopDimensions {
    fn default() -> Self {
        Self::new(
            BN254_MILLER_LOOP_LINE_EVENT_VARS,
            0,
            BN254_MILLER_LOOP_ACCUMULATOR_OP_VARS,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryReduceDimensions {
    point_len: usize,
    reduce_rounds: usize,
}

impl DoryReduceDimensions {
    pub const fn new(point_len: usize, reduce_rounds: usize) -> Self {
        Self {
            point_len,
            reduce_rounds,
        }
    }

    pub const fn point_len(self) -> usize {
        self.point_len
    }

    pub const fn reduce_rounds(self) -> usize {
        self.reduce_rounds
    }

    pub const fn reduce_round_vars(self) -> usize {
        ceil_log2_usize(self.reduce_rounds)
    }

    pub const fn scalar_fold_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.reduce_round_vars(), 2)
    }

    pub const fn state_chain_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.reduce_round_vars(), 2)
    }

    pub const fn boundary_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.reduce_round_vars(), 2)
    }
}

impl Default for DoryReduceDimensions {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WiringDimensions {
    log_edges: usize,
}

impl WiringDimensions {
    pub const fn new(log_edges: usize) -> Self {
        Self { log_edges }
    }

    pub const fn log_edges(self) -> usize {
        self.log_edges
    }

    pub const fn sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.log_edges, 2)
    }

    pub fn copy_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<Vec<F>, DoryAssistFormulaPointError> {
        if challenges.len() != self.log_edges {
            return Err(DoryAssistFormulaPointError::ChallengeLengthMismatch {
                expected: self.log_edges,
                got: challenges.len(),
            });
        }

        Ok(challenges.iter().rev().copied().collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixPackingDimensions {
    packed_width: usize,
    max_poly_width: usize,
    num_claims: usize,
}

impl PrefixPackingDimensions {
    pub const fn new(
        packed_vars: usize,
        max_poly_vars: usize,
        num_claims: usize,
    ) -> Result<Self, DoryAssistFormulaDimensionsError> {
        if packed_vars < max_poly_vars {
            return Err(DoryAssistFormulaDimensionsError::InvalidPackingPrefix {
                packed_vars,
                poly_vars: max_poly_vars,
            });
        }
        Ok(Self {
            packed_width: packed_vars,
            max_poly_width: max_poly_vars,
            num_claims,
        })
    }

    pub const fn packed_vars(self) -> usize {
        self.packed_width
    }

    pub const fn max_poly_vars(self) -> usize {
        self.max_poly_width
    }

    pub const fn prefix_vars(self) -> usize {
        self.packed_width - self.max_poly_width
    }

    pub const fn num_claims(self) -> usize {
        self.num_claims
    }

    pub fn split_opening_point<F: Field>(
        self,
        opening_point: &[F],
    ) -> Result<PrefixPackingOpeningPoint<F>, DoryAssistFormulaPointError> {
        if opening_point.len() != self.packed_width {
            return Err(DoryAssistFormulaPointError::OpeningPointLengthMismatch {
                expected: self.packed_width,
                got: opening_point.len(),
            });
        }

        let (prefix, suffix) = opening_point.split_at(self.prefix_vars());
        Ok(PrefixPackingOpeningPoint {
            prefix: prefix.to_vec(),
            suffix: suffix.to_vec(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackingOpeningPoint<F: Field> {
    pub prefix: Vec<F>,
    pub suffix: Vec<F>,
}

const fn ceil_log2_usize(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "tests fail loudly on invalid fixture dimensions"
    )]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn gt_dimensions_track_native_and_batched_domains() {
        let dimensions = GtDimensions::new(7, 3, 4);

        assert_eq!(dimensions.exp_native_rounds(), 11);
        assert_eq!(dimensions.exp_batched_rounds(), 14);
        assert_eq!(dimensions.exp_base_power_rounds(), 7);
        assert_eq!(dimensions.mul_rounds(), 8);
        assert_eq!(
            dimensions.exp_sumcheck(8),
            DoryAssistSumcheckSpec::boolean(14, 8)
        );
        assert_eq!(
            dimensions.exp_digit_selector_sumcheck(),
            DoryAssistSumcheckSpec::boolean(14, 3)
        );
        assert_eq!(
            dimensions.exp_digit_bitness_sumcheck(),
            DoryAssistSumcheckSpec::boolean(14, 2)
        );
        assert_eq!(
            dimensions.exp_base_power_sumcheck(),
            DoryAssistSumcheckSpec::boolean(7, 2)
        );
        assert_eq!(
            dimensions.exp_shift_sumcheck(),
            DoryAssistSumcheckSpec::boolean(14, 2)
        );
        assert_eq!(
            dimensions.exp_boundary_sumcheck(),
            DoryAssistSumcheckSpec::boolean(14, 2)
        );
    }

    #[test]
    fn miller_loop_dimensions_separate_line_and_accumulator_domains() {
        let dimensions = MillerLoopDimensions::new(7, 2, 8);

        assert_eq!(BN254_MILLER_LOOP_LINE_EVENTS, 87);
        assert_eq!(BN254_MILLER_LOOP_SQUARE_OPS, 63);
        assert_eq!(BN254_MILLER_LOOP_ACCUMULATOR_OPS, 150);
        assert_eq!(dimensions.line_event_vars(), 7);
        assert_eq!(dimensions.pair_vars(), 2);
        assert_eq!(dimensions.accumulator_op_vars(), 8);
        assert_eq!(dimensions.line_step_rounds(), 9);
        assert_eq!(dimensions.line_evaluation_rounds(), 9);
        assert_eq!(dimensions.pair_product_rounds(), 9);
        assert_eq!(dimensions.accumulator_rounds(), 8);
        assert_eq!(dimensions.boundary_rounds(), 8);
        assert_eq!(
            dimensions.line_step_sumcheck(6),
            DoryAssistSumcheckSpec::boolean(9, 6)
        );
        assert_eq!(
            dimensions.line_evaluation_sumcheck(3),
            DoryAssistSumcheckSpec::boolean(9, 3)
        );
        assert_eq!(
            dimensions.pair_product_sumcheck(2),
            DoryAssistSumcheckSpec::boolean(9, 2)
        );
        assert_eq!(
            dimensions.accumulator_sumcheck(2),
            DoryAssistSumcheckSpec::boolean(8, 2)
        );
        assert_eq!(
            dimensions.boundary_sumcheck(),
            DoryAssistSumcheckSpec::boolean(8, 2)
        );
    }

    #[test]
    fn wiring_opening_point_reverses_sumcheck_order() {
        let dimensions = WiringDimensions::new(3);
        let challenges = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];

        assert_eq!(
            dimensions.copy_opening_point(&challenges).unwrap(),
            vec![Fr::from_u64(5), Fr::from_u64(3), Fr::from_u64(2)]
        );
    }

    #[test]
    fn prefix_packing_splits_prefix_and_suffix() {
        let dimensions = PrefixPackingDimensions::new(6, 4, 5).unwrap();
        let point = [
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
            Fr::from_u64(5),
            Fr::from_u64(6),
        ];
        let split = dimensions.split_opening_point(&point).unwrap();

        assert_eq!(dimensions.prefix_vars(), 2);
        assert_eq!(split.prefix, vec![Fr::from_u64(1), Fr::from_u64(2)]);
        assert_eq!(
            split.suffix,
            vec![
                Fr::from_u64(3),
                Fr::from_u64(4),
                Fr::from_u64(5),
                Fr::from_u64(6)
            ]
        );
    }

    #[test]
    fn prefix_packing_rejects_short_packed_domain() {
        assert_eq!(
            PrefixPackingDimensions::new(3, 4, 1),
            Err(DoryAssistFormulaDimensionsError::InvalidPackingPrefix {
                packed_vars: 3,
                poly_vars: 4,
            })
        );
    }
}
