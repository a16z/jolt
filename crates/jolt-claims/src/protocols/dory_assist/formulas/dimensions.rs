use jolt_field::Field;
use serde::{Deserialize, Serialize};

pub use super::error::{DoryAssistFormulaDimensionsError, DoryAssistFormulaPointError};

pub const GT_ELEMENT_VARS: usize = 4;
pub const GT_EXP_BASE: usize = 4;
pub const GT_EXP_STEP_VARS: usize = 7;
pub const EC_SCALAR_MUL_STEP_VARS: usize = 8;

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
    pub pairing: MultiMillerLoopDimensions,
    pub wiring: WiringDimensions,
    pub packing: PrefixPackingDimensions,
}

impl DoryAssistDimensions {
    pub const fn new(
        gt: GtDimensions,
        g1: G1Dimensions,
        g2: G2Dimensions,
        pairing: MultiMillerLoopDimensions,
        wiring: WiringDimensions,
        packing: PrefixPackingDimensions,
    ) -> Self {
        Self {
            gt,
            g1,
            g2,
            pairing,
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

    pub const fn exp_base_power_sumcheck(self) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.exp_base_power_rounds(), 2)
    }

    pub const fn exp_shift_sumcheck(self) -> DoryAssistSumcheckSpec {
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
pub struct MultiMillerLoopDimensions {
    loop_steps: usize,
    pairs: usize,
}

impl MultiMillerLoopDimensions {
    pub const fn new(loop_step_vars: usize, pair_vars: usize) -> Self {
        Self {
            loop_steps: loop_step_vars,
            pairs: pair_vars,
        }
    }

    pub const fn loop_step_vars(self) -> usize {
        self.loop_steps
    }

    pub const fn pair_vars(self) -> usize {
        self.pairs
    }

    pub const fn rounds(self) -> usize {
        self.loop_steps + self.pairs
    }

    pub const fn sumcheck(self, degree: usize) -> DoryAssistSumcheckSpec {
        DoryAssistSumcheckSpec::boolean(self.rounds(), degree)
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
            dimensions.exp_base_power_sumcheck(),
            DoryAssistSumcheckSpec::boolean(7, 2)
        );
        assert_eq!(
            dimensions.exp_shift_sumcheck(),
            DoryAssistSumcheckSpec::boolean(14, 2)
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
