//! Dory-assist proof payload types.

use jolt_claims::protocols::dory_assist::{
    formulas::{composition, dory_reduce, g1, g2, gt, miller_loop},
    DoryAssistBoundaryEndpoint, DoryAssistDimensions, DoryAssistOpeningId, DoryAssistPolynomialId,
    DoryAssistPublicId, DoryAssistRelationId, DoryAssistVirtualPolynomial, DoryReduceDimensions,
    DoryReducePolynomial, G1Dimensions, G1Polynomial, G2Dimensions, G2Polynomial, GtDimensions,
    GtPolynomial, MillerLoopConstant, MillerLoopDimensions, MillerLoopPolynomial,
    MillerLoopSelector, PrefixPackingDimensions, WiringDimensions,
};
use jolt_crypto::{Bn254Fq12, GrumpkinPoint};
use jolt_field::{Fq, FromPrimitiveInt, Invertible};
use jolt_hyrax::{HyraxCommitment, HyraxOpeningProof};
use serde::{Deserialize, Serialize};

use crate::stages::DoryAssistStageProofs;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistProof {
    pub dimensions: DoryAssistDimensions,
    pub stages: DoryAssistStageProofs,
    pub opening_proof: HyraxOpeningProof<Fq>,
    pub claims: DoryAssistProofClaims,
    pub dense_commitment: HyraxCommitment<GrumpkinPoint>,
    pub public_outputs: DoryAssistPublicOutputs,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistPublicOutputs {
    pub pre_final_exponentiation: Bn254Fq12,
}

impl DoryAssistPublicOutputs {
    pub fn pre_final_exponentiation_coefficients(
        &self,
    ) -> [Fq; miller_loop::MILLER_LOOP_GT_COEFFS] {
        let mut coefficients = [Fq::default(); miller_loop::MILLER_LOOP_GT_COEFFS];
        coefficients[..Bn254Fq12::COEFFICIENTS]
            .copy_from_slice(&self.pre_final_exponentiation.coefficients());
        coefficients
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistProofClaims {
    pub stage1: DoryAssistStage1Claims,
    pub opening: DoryAssistOpeningClaims,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistStage1Claims {
    pub public: DoryAssistStage1PublicClaims,
    pub gt_exponentiation: DoryAssistGtExponentiationClaims,
    pub gt_exponentiation_digit_selector: DoryAssistGtExponentiationDigitSelectorClaims,
    pub gt_exponentiation_base_power: DoryAssistGtExponentiationBasePowerClaims,
    pub gt_exponentiation_digit_bitness: DoryAssistGtExponentiationDigitBitnessClaims,
    pub gt_exponentiation_shift: DoryAssistGtExponentiationShiftClaims,
    pub gt_exponentiation_boundary: DoryAssistGtExponentiationBoundaryClaims,
    pub gt_multiplication: DoryAssistGtMultiplicationClaims,
    pub g1: DoryAssistG1Claims,
    pub g2: DoryAssistG2Claims,
    pub miller_loop: DoryAssistMillerLoopClaims,
    pub dory_reduce: DoryAssistDoryReduceClaims,
}

impl DoryAssistStage1Claims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        self.gt_exponentiation
            .opening_claim(id)
            .or_else(|| self.gt_exponentiation_digit_selector.opening_claim(id))
            .or_else(|| self.gt_exponentiation_base_power.opening_claim(id))
            .or_else(|| self.gt_exponentiation_digit_bitness.opening_claim(id))
            .or_else(|| self.gt_exponentiation_shift.opening_claim(id))
            .or_else(|| self.gt_exponentiation_boundary.opening_claim(id))
            .or_else(|| self.gt_multiplication.opening_claim(id))
            .or_else(|| self.g1.opening_claim(id))
            .or_else(|| self.g2.opening_claim(id))
            .or_else(|| self.miller_loop.opening_claim(id))
            .or_else(|| self.dory_reduce.opening_claim(id))
    }

    pub fn public_claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        self.public.claim(id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistStage1PublicClaims {
    pub input: DoryAssistInputPublicClaims,
    pub gt_shift_eq_kernel: Fq,
    pub dory_reduce_shift_eq_kernel: Fq,
    pub dory_reduce: DoryAssistDoryReducePublicClaims,
    pub native_final: DoryAssistNativeFinalPublicClaims,
    pub gt_exponentiation_boundary: DoryAssistGtExponentiationBoundaryPublicClaims,
    pub g1: DoryAssistG1PublicClaims,
    pub g2: DoryAssistG2PublicClaims,
    pub miller_loop: DoryAssistMillerLoopPublicClaims,
}

impl Default for DoryAssistStage1PublicClaims {
    fn default() -> Self {
        Self {
            input: DoryAssistInputPublicClaims::default(),
            gt_shift_eq_kernel: Fq::from_u64(1),
            dory_reduce_shift_eq_kernel: Fq::from_u64(1),
            dory_reduce: DoryAssistDoryReducePublicClaims::default(),
            native_final: DoryAssistNativeFinalPublicClaims::default(),
            gt_exponentiation_boundary: DoryAssistGtExponentiationBoundaryPublicClaims::default(),
            g1: DoryAssistG1PublicClaims::default(),
            g2: DoryAssistG2PublicClaims::default(),
            miller_loop: DoryAssistMillerLoopPublicClaims::default(),
        }
    }
}

impl DoryAssistStage1PublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        match *id {
            DoryAssistPublicId::GtShiftEqKernel => Some(self.gt_shift_eq_kernel),
            DoryAssistPublicId::DoryReduceShiftEqKernel => Some(self.dory_reduce_shift_eq_kernel),
            DoryAssistPublicId::NativeFinalCheckInput(index) => self.native_final.claim(index),
            _ => self.input.claim(id).or_else(|| {
                self.gt_exponentiation_boundary
                    .claim(id)
                    .or_else(|| self.dory_reduce.claim(id))
                    .or_else(|| self.g1.claim(id))
                    .or_else(|| self.g2.claim(id))
                    .or_else(|| self.miller_loop.claim(id))
            }),
        }
    }
}

pub const NATIVE_FINAL_GT_C_START: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_GT_C_START;
pub const NATIVE_FINAL_D1_START: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_D1_START;
pub const NATIVE_FINAL_D2_START: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_D2_START;
pub const NATIVE_FINAL_E1_START: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_E1_START;
pub const NATIVE_FINAL_E2_START: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_E2_START;
pub const NATIVE_FINAL_E1_INIT_START: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_E1_INIT_START;
pub const NATIVE_FINAL_D2_INIT_START: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_D2_INIT_START;
pub const NATIVE_FINAL_S1_ACC_INDEX: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_S1_ACC_INDEX;
pub const NATIVE_FINAL_S2_ACC_INDEX: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_S2_ACC_INDEX;
pub const NATIVE_FINAL_INPUT_LEN: usize = dory_reduce::DORY_REDUCE_NATIVE_FINAL_INPUT_LEN;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistNativeFinalPublicClaims {
    pub inputs: Vec<Fq>,
}

impl Default for DoryAssistNativeFinalPublicClaims {
    fn default() -> Self {
        Self {
            inputs: vec![Fq::default(); NATIVE_FINAL_INPUT_LEN],
        }
    }
}

impl DoryAssistNativeFinalPublicClaims {
    pub fn bind(&mut self, inputs: Vec<Fq>) {
        self.inputs = inputs;
    }

    pub fn claim(&self, index: usize) -> Option<Fq> {
        self.inputs.get(index).copied()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistInputPublicClaims {
    pub checked_input_digest: Fq,
    pub verifier_setup_digest: Fq,
    pub verifier_setup_artifacts: Vec<Fq>,
    pub dory_proof_artifacts: Vec<Fq>,
    pub jolt_commitments: Vec<Fq>,
    pub jolt_evaluation_claims: Vec<Fq>,
    pub dory_reduce_initial_e2: Vec<Fq>,
    pub transcript_scalars: Vec<Fq>,
}

impl DoryAssistInputPublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        match *id {
            DoryAssistPublicId::VerifierSetupDigest => Some(self.verifier_setup_digest),
            DoryAssistPublicId::VerifierSetupArtifact(index) => {
                self.verifier_setup_artifacts.get(index).copied()
            }
            DoryAssistPublicId::DoryProofArtifact(index) => {
                self.dory_proof_artifacts.get(index).copied()
            }
            DoryAssistPublicId::JoltCommitment(index) => self.jolt_commitments.get(index).copied(),
            DoryAssistPublicId::JoltEvaluationClaim(index) => {
                self.jolt_evaluation_claims.get(index).copied()
            }
            DoryAssistPublicId::DoryReduceInitialE2(index) => {
                self.dory_reduce_initial_e2.get(index).copied()
            }
            DoryAssistPublicId::TranscriptScalar(index) => {
                self.transcript_scalars.get(index).copied()
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistDoryReducePublicClaims {
    pub boundary_initial_selector: Fq,
    pub boundary_final_selector: Fq,
}

impl Default for DoryAssistDoryReducePublicClaims {
    fn default() -> Self {
        Self {
            boundary_initial_selector: Fq::from_u64(1),
            boundary_final_selector: Fq::from_u64(1),
        }
    }
}

impl DoryAssistDoryReducePublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        match *id {
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::DoryReduceBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
            } => Some(self.boundary_initial_selector),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::DoryReduceBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
            } => Some(self.boundary_final_selector),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistMillerLoopPublicClaims {
    pub line_double_selector: Fq,
    pub line_add_selector: Fq,
    pub two_inverse: Fq,
    pub twist_b: [Fq; 2],
    pub pair_product_shift_eq_kernel: Fq,
    pub pair_product_initial_selector: Fq,
    pub pair_product_final_selector: Fq,
    pub pair_product_initial_value: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub accumulator_shift_eq_kernel: Fq,
    pub boundary_initial_selector: Fq,
    pub boundary_final_selector: Fq,
    pub boundary_initial_value: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub output_gt: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
}

impl Default for DoryAssistMillerLoopPublicClaims {
    fn default() -> Self {
        Self {
            line_double_selector: Fq::from_u64(1),
            line_add_selector: Fq::default(),
            two_inverse: Fq::from_u64(2).inverse().unwrap_or_default(),
            twist_b: [Fq::default(); 2],
            pair_product_shift_eq_kernel: Fq::from_u64(1),
            pair_product_initial_selector: Fq::from_u64(1),
            pair_product_final_selector: Fq::from_u64(1),
            pair_product_initial_value: [Fq::default(); miller_loop::MILLER_LOOP_GT_COEFFS],
            accumulator_shift_eq_kernel: Fq::from_u64(1),
            boundary_initial_selector: Fq::from_u64(1),
            boundary_final_selector: Fq::from_u64(1),
            boundary_initial_value: [Fq::default(); miller_loop::MILLER_LOOP_GT_COEFFS],
            output_gt: [Fq::default(); miller_loop::MILLER_LOOP_GT_COEFFS],
        }
    }
}

impl DoryAssistMillerLoopPublicClaims {
    pub fn bind_pre_final_exponentiation(&mut self, outputs: &DoryAssistPublicOutputs) {
        self.output_gt = outputs.pre_final_exponentiation_coefficients();
    }

    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        match *id {
            DoryAssistPublicId::MillerLoopSelector {
                relation: DoryAssistRelationId::MillerLoopLineStep,
                selector: MillerLoopSelector::LineDouble,
            } => Some(self.line_double_selector),
            DoryAssistPublicId::MillerLoopSelector {
                relation: DoryAssistRelationId::MillerLoopLineStep,
                selector: MillerLoopSelector::LineAdd,
            } => Some(self.line_add_selector),
            DoryAssistPublicId::MillerLoopConstant(MillerLoopConstant::TwoInverse) => {
                Some(self.two_inverse)
            }
            DoryAssistPublicId::MillerLoopConstant(MillerLoopConstant::TwistB0) => {
                Some(self.twist_b[0])
            }
            DoryAssistPublicId::MillerLoopConstant(MillerLoopConstant::TwistB1) => {
                Some(self.twist_b[1])
            }
            DoryAssistPublicId::MillerLoopShiftEqKernel(
                DoryAssistRelationId::MillerLoopPairProduct,
            ) => Some(self.pair_product_shift_eq_kernel),
            DoryAssistPublicId::MillerLoopShiftEqKernel(
                DoryAssistRelationId::MillerLoopAccumulator,
            ) => Some(self.accumulator_shift_eq_kernel),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::MillerLoopPairProduct,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
            } => Some(self.pair_product_initial_selector),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::MillerLoopPairProduct,
                endpoint: DoryAssistBoundaryEndpoint::Final,
            } => Some(self.pair_product_final_selector),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::MillerLoopBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
            } => Some(self.boundary_initial_selector),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::MillerLoopBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
            } => Some(self.boundary_final_selector),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::MillerLoopPairProduct,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
                component,
            } => self.pair_product_initial_value.get(component).copied(),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::MillerLoopBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
                component,
            } => self.boundary_initial_value.get(component).copied(),
            DoryAssistPublicId::MillerLoopOutputGt(component) => {
                self.output_gt.get(component).copied()
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtExponentiationBoundaryPublicClaims {
    pub initial_selector: Fq,
    pub final_selector: Fq,
    pub initial_value: Fq,
    pub final_value: Fq,
}

impl Default for DoryAssistGtExponentiationBoundaryPublicClaims {
    fn default() -> Self {
        Self {
            initial_selector: Fq::from_u64(1),
            final_selector: Fq::from_u64(1),
            initial_value: Fq::default(),
            final_value: Fq::default(),
        }
    }
}

impl DoryAssistGtExponentiationBoundaryPublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        match *id {
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::GtExponentiationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
            } => Some(self.initial_selector),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::GtExponentiationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
            } => Some(self.final_selector),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::GtExponentiationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
                component: 0,
            } => Some(self.initial_value),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::GtExponentiationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
                component: 0,
            } => Some(self.final_value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtExponentiationClaims {
    pub shifted_accumulator: Fq,
    pub accumulator: Fq,
    pub digit_selector: Fq,
    pub quotient: Fq,
    pub modulus: Fq,
}

impl DoryAssistGtExponentiationClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == gt::exp_shifted_accumulator_opening() => Some(self.shifted_accumulator),
            id if id == gt::exp_accumulator_opening() => Some(self.accumulator),
            id if id == gt::exp_digit_selector_opening() => Some(self.digit_selector),
            id if id == gt::exp_quotient_opening() => Some(self.quotient),
            id if id == gt::exp_modulus_opening() => Some(self.modulus),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtExponentiationDigitSelectorClaims {
    pub digit_selector: Fq,
    pub digit_lo: Fq,
    pub digit_hi: Fq,
    pub base: Fq,
    pub base_squared: Fq,
    pub base_cubed: Fq,
}

impl Default for DoryAssistGtExponentiationDigitSelectorClaims {
    fn default() -> Self {
        Self {
            digit_selector: Fq::default(),
            digit_lo: Fq::from_u64(1),
            digit_hi: Fq::default(),
            base: Fq::default(),
            base_squared: Fq::default(),
            base_cubed: Fq::default(),
        }
    }
}

impl DoryAssistGtExponentiationDigitSelectorClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == gt::exp_digit_selector_base_4_opening() => Some(self.digit_selector),
            id if id == gt::exp_digit_bit_opening(0) => Some(self.digit_lo),
            id if id == gt::exp_digit_bit_opening(1) => Some(self.digit_hi),
            id if id == gt::exp_base_power_selector_opening(1) => Some(self.base),
            id if id == gt::exp_base_power_selector_opening(2) => Some(self.base_squared),
            id if id == gt::exp_base_power_selector_opening(3) => Some(self.base_cubed),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtExponentiationBasePowerClaims {
    pub base: Fq,
    pub base_squared: Fq,
    pub quotient_squared: Fq,
    pub modulus: Fq,
    pub base_cubed: Fq,
    pub quotient_cubed: Fq,
}

impl DoryAssistGtExponentiationBasePowerClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == gt::exp_base_power_checked_opening(1) => Some(self.base),
            id if id == gt::exp_base_power_checked_opening(2) => Some(self.base_squared),
            id if id == gt::exp_base_power_quotient_opening(2) => Some(self.quotient_squared),
            id if id == gt::exp_base_power_modulus_opening() => Some(self.modulus),
            id if id == gt::exp_base_power_checked_opening(3) => Some(self.base_cubed),
            id if id == gt::exp_base_power_quotient_opening(3) => Some(self.quotient_cubed),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtExponentiationDigitBitnessClaims {
    pub digit_lo: Fq,
    pub digit_hi: Fq,
}

impl Default for DoryAssistGtExponentiationDigitBitnessClaims {
    fn default() -> Self {
        Self {
            digit_lo: Fq::from_u64(1),
            digit_hi: Fq::default(),
        }
    }
}

impl DoryAssistGtExponentiationDigitBitnessClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == gt::exp_digit_bit_bitness_opening(0) => Some(self.digit_lo),
            id if id == gt::exp_digit_bit_bitness_opening(1) => Some(self.digit_hi),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtExponentiationShiftClaims {
    pub accumulator: Fq,
}

impl DoryAssistGtExponentiationShiftClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == gt::exp_accumulator_shift_opening() => Some(self.accumulator),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtExponentiationBoundaryClaims {
    pub accumulator: Fq,
    pub shifted_accumulator: Fq,
}

impl DoryAssistGtExponentiationBoundaryClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == gt::exp_boundary_accumulator_opening() => Some(self.accumulator),
            id if id == gt::exp_boundary_shifted_accumulator_opening() => {
                Some(self.shifted_accumulator)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtMultiplicationClaims {
    pub opening: DoryAssistGtMultiplicationOpeningClaims,
    pub rows: [DoryAssistGtMultiplicationRowClaims; composition::GT_MULTIPLICATION_ROWS],
}

impl Default for DoryAssistGtMultiplicationClaims {
    fn default() -> Self {
        Self {
            opening: DoryAssistGtMultiplicationOpeningClaims::default(),
            rows: [DoryAssistGtMultiplicationRowClaims::default();
                composition::GT_MULTIPLICATION_ROWS],
        }
    }
}

impl DoryAssistGtMultiplicationClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        self.opening
            .claim(gt_polynomial(id, DoryAssistRelationId::GtMultiplication)?)
    }

    pub fn row_claim(&self, row: usize, component: usize, polynomial: GtPolynomial) -> Option<Fq> {
        self.rows
            .get(row)
            .and_then(|row| row.claim(component, polynomial))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtMultiplicationOpeningClaims {
    pub left: Fq,
    pub right: Fq,
    pub output: Fq,
    pub quotient: Fq,
    pub modulus: Fq,
}

impl DoryAssistGtMultiplicationOpeningClaims {
    pub fn claim(&self, polynomial: GtPolynomial) -> Option<Fq> {
        match polynomial {
            GtPolynomial::MulLeft => Some(self.left),
            GtPolynomial::MulRight => Some(self.right),
            GtPolynomial::MulOutput => Some(self.output),
            GtPolynomial::MulQuotient => Some(self.quotient),
            GtPolynomial::Modulus => Some(self.modulus),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistGtMultiplicationRowClaims {
    pub left: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub right: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub output: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub quotient: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
}

impl DoryAssistGtMultiplicationRowClaims {
    pub fn claim(&self, component: usize, polynomial: GtPolynomial) -> Option<Fq> {
        match polynomial {
            GtPolynomial::MulLeft => self.left.get(component).copied(),
            GtPolynomial::MulRight => self.right.get(component).copied(),
            GtPolynomial::MulOutput => self.output.get(component).copied(),
            GtPolynomial::MulQuotient => self.quotient.get(component).copied(),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1PointClaims {
    pub x: Fq,
    pub y: Fq,
    pub infinity: Fq,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1CoordinateClaims {
    pub x: Fq,
    pub y: Fq,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2PointClaims {
    pub x: [Fq; 2],
    pub y: [Fq; 2],
    pub infinity: Fq,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2CoordinateClaims {
    pub x: [Fq; 2],
    pub y: [Fq; 2],
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1PublicClaims {
    pub scalar_multiplication_boundary: DoryAssistG1ScalarMultiplicationBoundaryPublicClaims,
}

impl DoryAssistG1PublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        self.scalar_multiplication_boundary.claim(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1ScalarMultiplicationBoundaryPublicClaims {
    pub initial_selector: Fq,
    pub final_selector: Fq,
    pub initial_value: DoryAssistG1PointClaims,
    pub final_value: DoryAssistG1CoordinateClaims,
}

impl Default for DoryAssistG1ScalarMultiplicationBoundaryPublicClaims {
    fn default() -> Self {
        Self {
            initial_selector: Fq::from_u64(1),
            final_selector: Fq::from_u64(1),
            initial_value: DoryAssistG1PointClaims::default(),
            final_value: DoryAssistG1CoordinateClaims::default(),
        }
    }
}

impl DoryAssistG1ScalarMultiplicationBoundaryPublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        match *id {
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::G1ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
            } => Some(self.initial_selector),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::G1ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
            } => Some(self.final_selector),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::G1ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
                component,
            } => g1_point_component(&self.initial_value, component),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::G1ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
                component,
            } => g1_coordinate_component(&self.final_value, component),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2PublicClaims {
    pub scalar_multiplication_boundary: DoryAssistG2ScalarMultiplicationBoundaryPublicClaims,
}

impl DoryAssistG2PublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        self.scalar_multiplication_boundary.claim(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2ScalarMultiplicationBoundaryPublicClaims {
    pub initial_selector: Fq,
    pub final_selector: Fq,
    pub initial_value: DoryAssistG2PointClaims,
    pub final_value: DoryAssistG2CoordinateClaims,
}

impl Default for DoryAssistG2ScalarMultiplicationBoundaryPublicClaims {
    fn default() -> Self {
        Self {
            initial_selector: Fq::from_u64(1),
            final_selector: Fq::from_u64(1),
            initial_value: DoryAssistG2PointClaims::default(),
            final_value: DoryAssistG2CoordinateClaims::default(),
        }
    }
}

impl DoryAssistG2ScalarMultiplicationBoundaryPublicClaims {
    pub fn claim(&self, id: &DoryAssistPublicId) -> Option<Fq> {
        match *id {
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::G2ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
            } => Some(self.initial_selector),
            DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::G2ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
            } => Some(self.final_selector),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::G2ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
                component,
            } => g2_point_component(&self.initial_value, component),
            DoryAssistPublicId::BoundaryValue {
                relation: DoryAssistRelationId::G2ScalarMultiplicationBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
                component,
            } => g2_coordinate_component(&self.final_value, component),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1Claims {
    pub scalar_multiplication: DoryAssistG1ScalarMultiplicationClaims,
    pub scalar_multiplication_shift: DoryAssistG1ScalarMultiplicationShiftClaims,
    pub scalar_multiplication_boundary: DoryAssistG1ScalarMultiplicationBoundaryClaims,
    pub addition: DoryAssistG1AdditionClaims,
}

impl DoryAssistG1Claims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        self.scalar_multiplication
            .opening_claim(id)
            .or_else(|| self.scalar_multiplication_shift.opening_claim(id))
            .or_else(|| self.scalar_multiplication_boundary.opening_claim(id))
            .or_else(|| self.addition.opening_claim(id))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1ScalarMultiplicationClaims {
    pub accumulator: DoryAssistG1PointClaims,
    pub doubled: DoryAssistG1PointClaims,
    pub shifted_accumulator: DoryAssistG1CoordinateClaims,
    pub bit: Fq,
    pub base: DoryAssistG1CoordinateClaims,
}

impl DoryAssistG1ScalarMultiplicationClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match g1_polynomial(id, DoryAssistRelationId::G1ScalarMultiplication)? {
            G1Polynomial::ScalarMulAccumulatorX => Some(self.accumulator.x),
            G1Polynomial::ScalarMulAccumulatorY => Some(self.accumulator.y),
            G1Polynomial::ScalarMulAccumulatorInfinity => Some(self.accumulator.infinity),
            G1Polynomial::ScalarMulDoubledX => Some(self.doubled.x),
            G1Polynomial::ScalarMulDoubledY => Some(self.doubled.y),
            G1Polynomial::ScalarMulDoubledInfinity => Some(self.doubled.infinity),
            G1Polynomial::ScalarMulShiftedAccumulatorX => Some(self.shifted_accumulator.x),
            G1Polynomial::ScalarMulShiftedAccumulatorY => Some(self.shifted_accumulator.y),
            G1Polynomial::ScalarMulBit => Some(self.bit),
            G1Polynomial::ScalarMulBaseX => Some(self.base.x),
            G1Polynomial::ScalarMulBaseY => Some(self.base.y),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1ScalarMultiplicationShiftClaims {
    pub shifted_accumulator: DoryAssistG1CoordinateClaims,
    pub accumulator: DoryAssistG1CoordinateClaims,
}

impl DoryAssistG1ScalarMultiplicationShiftClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == g1::scalar_mul_shifted_accumulator_x_opening() => {
                Some(self.shifted_accumulator.x)
            }
            id if id == g1::scalar_mul_shifted_accumulator_y_opening() => {
                Some(self.shifted_accumulator.y)
            }
            id if id == g1::scalar_mul_accumulator_x_opening() => Some(self.accumulator.x),
            id if id == g1::scalar_mul_accumulator_y_opening() => Some(self.accumulator.y),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1ScalarMultiplicationBoundaryClaims {
    pub accumulator: DoryAssistG1PointClaims,
    pub shifted_accumulator: DoryAssistG1CoordinateClaims,
}

impl DoryAssistG1ScalarMultiplicationBoundaryClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == g1::scalar_mul_boundary_accumulator_x_opening() => Some(self.accumulator.x),
            id if id == g1::scalar_mul_boundary_accumulator_y_opening() => Some(self.accumulator.y),
            id if id == g1::scalar_mul_boundary_accumulator_infinity_opening() => {
                Some(self.accumulator.infinity)
            }
            id if id == g1::scalar_mul_boundary_shifted_accumulator_x_opening() => {
                Some(self.shifted_accumulator.x)
            }
            id if id == g1::scalar_mul_boundary_shifted_accumulator_y_opening() => {
                Some(self.shifted_accumulator.y)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG1AdditionClaims {
    pub left: DoryAssistG1PointClaims,
    pub right: DoryAssistG1PointClaims,
    pub output: DoryAssistG1PointClaims,
    pub slope: Fq,
    pub inverse: Fq,
    pub branch_selectors: [Fq; 2],
}

impl Default for DoryAssistG1AdditionClaims {
    fn default() -> Self {
        Self {
            left: DoryAssistG1PointClaims::default(),
            right: DoryAssistG1PointClaims::default(),
            output: DoryAssistG1PointClaims::default(),
            slope: Fq::default(),
            inverse: Fq::default(),
            branch_selectors: [Fq::from_u64(1), Fq::default()],
        }
    }
}

impl DoryAssistG1AdditionClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match g1_polynomial(id, DoryAssistRelationId::G1Addition)? {
            G1Polynomial::AddInputLeftX => Some(self.left.x),
            G1Polynomial::AddInputLeftY => Some(self.left.y),
            G1Polynomial::AddInputLeftInfinity => Some(self.left.infinity),
            G1Polynomial::AddInputRightX => Some(self.right.x),
            G1Polynomial::AddInputRightY => Some(self.right.y),
            G1Polynomial::AddInputRightInfinity => Some(self.right.infinity),
            G1Polynomial::AddOutputX => Some(self.output.x),
            G1Polynomial::AddOutputY => Some(self.output.y),
            G1Polynomial::AddOutputInfinity => Some(self.output.infinity),
            G1Polynomial::AddSlope => Some(self.slope),
            G1Polynomial::AddInverse => Some(self.inverse),
            G1Polynomial::AddBranchSelector(index) => self.branch_selectors.get(index).copied(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2Claims {
    pub scalar_multiplication: DoryAssistG2ScalarMultiplicationClaims,
    pub scalar_multiplication_shift: DoryAssistG2ScalarMultiplicationShiftClaims,
    pub scalar_multiplication_boundary: DoryAssistG2ScalarMultiplicationBoundaryClaims,
    pub addition: DoryAssistG2AdditionClaims,
}

impl DoryAssistG2Claims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        self.scalar_multiplication
            .opening_claim(id)
            .or_else(|| self.scalar_multiplication_shift.opening_claim(id))
            .or_else(|| self.scalar_multiplication_boundary.opening_claim(id))
            .or_else(|| self.addition.opening_claim(id))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2ScalarMultiplicationClaims {
    pub accumulator: DoryAssistG2PointClaims,
    pub doubled: DoryAssistG2PointClaims,
    pub shifted_accumulator: DoryAssistG2CoordinateClaims,
    pub bit: Fq,
    pub base: DoryAssistG2CoordinateClaims,
}

impl DoryAssistG2ScalarMultiplicationClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match g2_polynomial(id, DoryAssistRelationId::G2ScalarMultiplication)? {
            G2Polynomial::ScalarMulAccumulatorX0 => Some(self.accumulator.x[0]),
            G2Polynomial::ScalarMulAccumulatorX1 => Some(self.accumulator.x[1]),
            G2Polynomial::ScalarMulAccumulatorY0 => Some(self.accumulator.y[0]),
            G2Polynomial::ScalarMulAccumulatorY1 => Some(self.accumulator.y[1]),
            G2Polynomial::ScalarMulAccumulatorInfinity => Some(self.accumulator.infinity),
            G2Polynomial::ScalarMulDoubledX0 => Some(self.doubled.x[0]),
            G2Polynomial::ScalarMulDoubledX1 => Some(self.doubled.x[1]),
            G2Polynomial::ScalarMulDoubledY0 => Some(self.doubled.y[0]),
            G2Polynomial::ScalarMulDoubledY1 => Some(self.doubled.y[1]),
            G2Polynomial::ScalarMulDoubledInfinity => Some(self.doubled.infinity),
            G2Polynomial::ScalarMulShiftedAccumulatorX0 => Some(self.shifted_accumulator.x[0]),
            G2Polynomial::ScalarMulShiftedAccumulatorX1 => Some(self.shifted_accumulator.x[1]),
            G2Polynomial::ScalarMulShiftedAccumulatorY0 => Some(self.shifted_accumulator.y[0]),
            G2Polynomial::ScalarMulShiftedAccumulatorY1 => Some(self.shifted_accumulator.y[1]),
            G2Polynomial::ScalarMulBit => Some(self.bit),
            G2Polynomial::ScalarMulBaseX0 => Some(self.base.x[0]),
            G2Polynomial::ScalarMulBaseX1 => Some(self.base.x[1]),
            G2Polynomial::ScalarMulBaseY0 => Some(self.base.y[0]),
            G2Polynomial::ScalarMulBaseY1 => Some(self.base.y[1]),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2ScalarMultiplicationShiftClaims {
    pub shifted_accumulator: DoryAssistG2CoordinateClaims,
    pub accumulator: DoryAssistG2CoordinateClaims,
}

impl DoryAssistG2ScalarMultiplicationShiftClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == g2::scalar_mul_shifted_accumulator_x0_opening() => {
                Some(self.shifted_accumulator.x[0])
            }
            id if id == g2::scalar_mul_shifted_accumulator_x1_opening() => {
                Some(self.shifted_accumulator.x[1])
            }
            id if id == g2::scalar_mul_shifted_accumulator_y0_opening() => {
                Some(self.shifted_accumulator.y[0])
            }
            id if id == g2::scalar_mul_shifted_accumulator_y1_opening() => {
                Some(self.shifted_accumulator.y[1])
            }
            id if id == g2::scalar_mul_accumulator_x0_opening() => Some(self.accumulator.x[0]),
            id if id == g2::scalar_mul_accumulator_x1_opening() => Some(self.accumulator.x[1]),
            id if id == g2::scalar_mul_accumulator_y0_opening() => Some(self.accumulator.y[0]),
            id if id == g2::scalar_mul_accumulator_y1_opening() => Some(self.accumulator.y[1]),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2ScalarMultiplicationBoundaryClaims {
    pub accumulator: DoryAssistG2PointClaims,
    pub shifted_accumulator: DoryAssistG2CoordinateClaims,
}

impl DoryAssistG2ScalarMultiplicationBoundaryClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match *id {
            id if id == g2::scalar_mul_boundary_accumulator_x0_opening() => {
                Some(self.accumulator.x[0])
            }
            id if id == g2::scalar_mul_boundary_accumulator_x1_opening() => {
                Some(self.accumulator.x[1])
            }
            id if id == g2::scalar_mul_boundary_accumulator_y0_opening() => {
                Some(self.accumulator.y[0])
            }
            id if id == g2::scalar_mul_boundary_accumulator_y1_opening() => {
                Some(self.accumulator.y[1])
            }
            id if id == g2::scalar_mul_boundary_accumulator_infinity_opening() => {
                Some(self.accumulator.infinity)
            }
            id if id == g2::scalar_mul_boundary_shifted_accumulator_x0_opening() => {
                Some(self.shifted_accumulator.x[0])
            }
            id if id == g2::scalar_mul_boundary_shifted_accumulator_x1_opening() => {
                Some(self.shifted_accumulator.x[1])
            }
            id if id == g2::scalar_mul_boundary_shifted_accumulator_y0_opening() => {
                Some(self.shifted_accumulator.y[0])
            }
            id if id == g2::scalar_mul_boundary_shifted_accumulator_y1_opening() => {
                Some(self.shifted_accumulator.y[1])
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistG2AdditionClaims {
    pub left: DoryAssistG2PointClaims,
    pub right: DoryAssistG2PointClaims,
    pub output: DoryAssistG2PointClaims,
    pub slope: [Fq; 2],
    pub inverse: [Fq; 2],
    pub branch_selectors: [Fq; 2],
}

impl Default for DoryAssistG2AdditionClaims {
    fn default() -> Self {
        Self {
            left: DoryAssistG2PointClaims::default(),
            right: DoryAssistG2PointClaims::default(),
            output: DoryAssistG2PointClaims::default(),
            slope: [Fq::default(); 2],
            inverse: [Fq::default(); 2],
            branch_selectors: [Fq::from_u64(1), Fq::default()],
        }
    }
}

impl DoryAssistG2AdditionClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match g2_polynomial(id, DoryAssistRelationId::G2Addition)? {
            G2Polynomial::AddInputLeftX0 => Some(self.left.x[0]),
            G2Polynomial::AddInputLeftX1 => Some(self.left.x[1]),
            G2Polynomial::AddInputLeftY0 => Some(self.left.y[0]),
            G2Polynomial::AddInputLeftY1 => Some(self.left.y[1]),
            G2Polynomial::AddInputLeftInfinity => Some(self.left.infinity),
            G2Polynomial::AddInputRightX0 => Some(self.right.x[0]),
            G2Polynomial::AddInputRightX1 => Some(self.right.x[1]),
            G2Polynomial::AddInputRightY0 => Some(self.right.y[0]),
            G2Polynomial::AddInputRightY1 => Some(self.right.y[1]),
            G2Polynomial::AddInputRightInfinity => Some(self.right.infinity),
            G2Polynomial::AddOutputX0 => Some(self.output.x[0]),
            G2Polynomial::AddOutputX1 => Some(self.output.x[1]),
            G2Polynomial::AddOutputY0 => Some(self.output.y[0]),
            G2Polynomial::AddOutputY1 => Some(self.output.y[1]),
            G2Polynomial::AddOutputInfinity => Some(self.output.infinity),
            G2Polynomial::AddSlope0 => Some(self.slope[0]),
            G2Polynomial::AddSlope1 => Some(self.slope[1]),
            G2Polynomial::AddInverse0 => Some(self.inverse[0]),
            G2Polynomial::AddInverse1 => Some(self.inverse[1]),
            G2Polynomial::AddBranchSelector(index) => self.branch_selectors.get(index).copied(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistMillerLoopClaims {
    pub line_step: DoryAssistMillerLoopLineStepClaims,
    pub line_evaluation: DoryAssistMillerLoopLineEvaluationClaims,
    pub pair_product: DoryAssistMillerLoopPairProductClaims,
    pub accumulator: DoryAssistMillerLoopAccumulatorClaims,
    pub boundary: DoryAssistMillerLoopBoundaryClaims,
}

impl DoryAssistMillerLoopClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        self.line_step
            .opening_claim(id)
            .or_else(|| self.line_evaluation.opening_claim(id))
            .or_else(|| self.pair_product.opening_claim(id))
            .or_else(|| self.accumulator.opening_claim(id))
            .or_else(|| self.boundary.opening_claim(id))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistMillerLoopLineStepClaims {
    pub state_x: [Fq; 2],
    pub state_y: [Fq; 2],
    pub state_z: [Fq; 2],
    pub addend_x: [Fq; 2],
    pub addend_y: [Fq; 2],
    pub shifted_state_x: [Fq; 2],
    pub shifted_state_y: [Fq; 2],
    pub shifted_state_z: [Fq; 2],
    pub line_coefficients: [[Fq; 2]; miller_loop::MILLER_LOOP_LINE_COEFFICIENTS],
}

impl DoryAssistMillerLoopLineStepClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match miller_loop_polynomial(id, DoryAssistRelationId::MillerLoopLineStep)? {
            MillerLoopPolynomial::G2LineStateX0 => Some(self.state_x[0]),
            MillerLoopPolynomial::G2LineStateX1 => Some(self.state_x[1]),
            MillerLoopPolynomial::G2LineStateY0 => Some(self.state_y[0]),
            MillerLoopPolynomial::G2LineStateY1 => Some(self.state_y[1]),
            MillerLoopPolynomial::G2LineStateZ0 => Some(self.state_z[0]),
            MillerLoopPolynomial::G2LineStateZ1 => Some(self.state_z[1]),
            MillerLoopPolynomial::G2LineAddendX0 => Some(self.addend_x[0]),
            MillerLoopPolynomial::G2LineAddendX1 => Some(self.addend_x[1]),
            MillerLoopPolynomial::G2LineAddendY0 => Some(self.addend_y[0]),
            MillerLoopPolynomial::G2LineAddendY1 => Some(self.addend_y[1]),
            MillerLoopPolynomial::G2LineShiftedStateX0 => Some(self.shifted_state_x[0]),
            MillerLoopPolynomial::G2LineShiftedStateX1 => Some(self.shifted_state_x[1]),
            MillerLoopPolynomial::G2LineShiftedStateY0 => Some(self.shifted_state_y[0]),
            MillerLoopPolynomial::G2LineShiftedStateY1 => Some(self.shifted_state_y[1]),
            MillerLoopPolynomial::G2LineShiftedStateZ0 => Some(self.shifted_state_z[0]),
            MillerLoopPolynomial::G2LineShiftedStateZ1 => Some(self.shifted_state_z[1]),
            MillerLoopPolynomial::LineCoefficient {
                coefficient,
                component,
            } => self
                .line_coefficients
                .get(coefficient)
                .and_then(|coefficient| coefficient.get(component))
                .copied(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistMillerLoopLineEvaluationClaims {
    pub g1_point_x: Fq,
    pub g1_point_y: Fq,
    pub line_coefficients: [[Fq; 2]; miller_loop::MILLER_LOOP_LINE_COEFFICIENTS],
    pub line_evaluation_coeffs: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
}

impl DoryAssistMillerLoopLineEvaluationClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match miller_loop_polynomial(id, DoryAssistRelationId::MillerLoopLineEvaluation)? {
            MillerLoopPolynomial::G1PointX => Some(self.g1_point_x),
            MillerLoopPolynomial::G1PointY => Some(self.g1_point_y),
            MillerLoopPolynomial::LineCoefficient {
                coefficient,
                component,
            } => self
                .line_coefficients
                .get(coefficient)
                .and_then(|coefficient| coefficient.get(component))
                .copied(),
            MillerLoopPolynomial::LineEvaluationCoeff(component) => {
                self.line_evaluation_coeffs.get(component).copied()
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistMillerLoopPairProductClaims {
    pub accumulator: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub shifted_accumulator: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub line_product: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub quotient: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
}

impl DoryAssistMillerLoopPairProductClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match miller_loop_polynomial(id, DoryAssistRelationId::MillerLoopPairProduct)? {
            MillerLoopPolynomial::PairProductAccumulatorCoeff(component) => {
                self.accumulator.get(component).copied()
            }
            MillerLoopPolynomial::PairProductShiftedAccumulatorCoeff(component) => {
                self.shifted_accumulator.get(component).copied()
            }
            MillerLoopPolynomial::PairLineProductCoeff(component) => {
                self.line_product.get(component).copied()
            }
            MillerLoopPolynomial::PairProductQuotientCoeff(component) => {
                self.quotient.get(component).copied()
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistMillerLoopAccumulatorClaims {
    pub accumulator: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub shifted_accumulator: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub quotient: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
}

impl DoryAssistMillerLoopAccumulatorClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match miller_loop_polynomial(id, DoryAssistRelationId::MillerLoopAccumulator)? {
            MillerLoopPolynomial::AccumulatorCoeff(component) => {
                self.accumulator.get(component).copied()
            }
            MillerLoopPolynomial::ShiftedAccumulatorCoeff(component) => {
                self.shifted_accumulator.get(component).copied()
            }
            MillerLoopPolynomial::AccumulatorQuotientCoeff(component) => {
                self.quotient.get(component).copied()
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistMillerLoopBoundaryClaims {
    pub accumulator: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
    pub shifted_accumulator: [Fq; miller_loop::MILLER_LOOP_GT_COEFFS],
}

impl DoryAssistMillerLoopBoundaryClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match miller_loop_polynomial(id, DoryAssistRelationId::MillerLoopBoundary)? {
            MillerLoopPolynomial::AccumulatorCoeff(component) => {
                self.accumulator.get(component).copied()
            }
            MillerLoopPolynomial::ShiftedAccumulatorCoeff(component) => {
                self.shifted_accumulator.get(component).copied()
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistDoryReduceClaims {
    pub transitions: Vec<DoryAssistOpeningClaim>,
    pub scalar_fold: DoryAssistDoryReduceScalarFoldClaims,
    pub state_chain: Vec<DoryAssistOpeningClaim>,
    pub boundary: Vec<DoryAssistOpeningClaim>,
}

impl DoryAssistDoryReduceClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        self.scalar_fold
            .opening_claim(id)
            .or_else(|| {
                self.transitions
                    .iter()
                    .find(|claim| claim.id == *id)
                    .map(|claim| claim.value)
            })
            .or_else(|| {
                self.state_chain
                    .iter()
                    .find(|claim| claim.id == *id)
                    .map(|claim| claim.value)
            })
            .or_else(|| {
                self.boundary
                    .iter()
                    .find(|claim| claim.id == *id)
                    .map(|claim| claim.value)
            })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistDoryReduceScalarFoldClaims {
    pub s1_accumulator: Fq,
    pub s1_next_accumulator: Fq,
    pub s1_fold_factor: Fq,
    pub s2_accumulator: Fq,
    pub s2_next_accumulator: Fq,
    pub s2_fold_factor: Fq,
}

impl DoryAssistDoryReduceScalarFoldClaims {
    pub fn opening_claim(&self, id: &DoryAssistOpeningId) -> Option<Fq> {
        match dory_reduce_polynomial(id, DoryAssistRelationId::DoryReduceScalarFold)? {
            DoryReducePolynomial::S1Accumulator => Some(self.s1_accumulator),
            DoryReducePolynomial::S1NextAccumulator => Some(self.s1_next_accumulator),
            DoryReducePolynomial::S1FoldFactor => Some(self.s1_fold_factor),
            DoryReducePolynomial::S2Accumulator => Some(self.s2_accumulator),
            DoryReducePolynomial::S2NextAccumulator => Some(self.s2_next_accumulator),
            DoryReducePolynomial::S2FoldFactor => Some(self.s2_fold_factor),
            _ => None,
        }
    }
}

fn g1_point_component(claims: &DoryAssistG1PointClaims, component: usize) -> Option<Fq> {
    match component {
        0 => Some(claims.x),
        1 => Some(claims.y),
        2 => Some(claims.infinity),
        _ => None,
    }
}

fn g1_coordinate_component(claims: &DoryAssistG1CoordinateClaims, component: usize) -> Option<Fq> {
    match component {
        0 => Some(claims.x),
        1 => Some(claims.y),
        _ => None,
    }
}

fn g2_point_component(claims: &DoryAssistG2PointClaims, component: usize) -> Option<Fq> {
    match component {
        0 => Some(claims.x[0]),
        1 => Some(claims.x[1]),
        2 => Some(claims.y[0]),
        3 => Some(claims.y[1]),
        4 => Some(claims.infinity),
        _ => None,
    }
}

fn g2_coordinate_component(claims: &DoryAssistG2CoordinateClaims, component: usize) -> Option<Fq> {
    match component {
        0 => Some(claims.x[0]),
        1 => Some(claims.x[1]),
        2 => Some(claims.y[0]),
        3 => Some(claims.y[1]),
        _ => None,
    }
}

fn g1_polynomial(
    id: &DoryAssistOpeningId,
    expected_relation: DoryAssistRelationId,
) -> Option<G1Polynomial> {
    match *id {
        DoryAssistOpeningId::Polynomial {
            polynomial: DoryAssistPolynomialId::Virtual(DoryAssistVirtualPolynomial::G1(polynomial)),
            relation,
        } if relation == expected_relation => Some(polynomial),
        DoryAssistOpeningId::Polynomial { .. } => None,
    }
}

fn g2_polynomial(
    id: &DoryAssistOpeningId,
    expected_relation: DoryAssistRelationId,
) -> Option<G2Polynomial> {
    match *id {
        DoryAssistOpeningId::Polynomial {
            polynomial: DoryAssistPolynomialId::Virtual(DoryAssistVirtualPolynomial::G2(polynomial)),
            relation,
        } if relation == expected_relation => Some(polynomial),
        DoryAssistOpeningId::Polynomial { .. } => None,
    }
}

fn miller_loop_polynomial(
    id: &DoryAssistOpeningId,
    expected_relation: DoryAssistRelationId,
) -> Option<MillerLoopPolynomial> {
    match *id {
        DoryAssistOpeningId::Polynomial {
            polynomial:
                DoryAssistPolynomialId::Virtual(DoryAssistVirtualPolynomial::MillerLoop(polynomial)),
            relation,
        } if relation == expected_relation => Some(polynomial),
        DoryAssistOpeningId::Polynomial { .. } => None,
    }
}

fn dory_reduce_polynomial(
    id: &DoryAssistOpeningId,
    expected_relation: DoryAssistRelationId,
) -> Option<DoryReducePolynomial> {
    match *id {
        DoryAssistOpeningId::Polynomial {
            polynomial:
                DoryAssistPolynomialId::Virtual(DoryAssistVirtualPolynomial::DoryReduce(polynomial)),
            relation,
        } if relation == expected_relation => Some(polynomial),
        DoryAssistOpeningId::Polynomial { .. } => None,
    }
}

fn gt_polynomial(
    id: &DoryAssistOpeningId,
    expected_relation: DoryAssistRelationId,
) -> Option<GtPolynomial> {
    match *id {
        DoryAssistOpeningId::Polynomial {
            polynomial: DoryAssistPolynomialId::Virtual(DoryAssistVirtualPolynomial::Gt(polynomial)),
            relation,
        } if relation == expected_relation => Some(polynomial),
        DoryAssistOpeningId::Polynomial { .. } => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistOpeningClaim {
    pub id: DoryAssistOpeningId,
    pub value: Fq,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistOpeningClaims {
    pub packed_point: Vec<Fq>,
    pub packed_eval: Fq,
}

impl Default for DoryAssistProof {
    fn default() -> Self {
        Self {
            dimensions: default_dory_assist_dimensions(),
            stages: DoryAssistStageProofs::default(),
            opening_proof: HyraxOpeningProof {
                combined_row: Vec::new(),
                combined_row_opening_scalar: Fq::default(),
            },
            claims: DoryAssistProofClaims::default(),
            dense_commitment: HyraxCommitment::default(),
            public_outputs: DoryAssistPublicOutputs::default(),
        }
    }
}

#[expect(
    clippy::expect_used,
    reason = "canonical default Dory-assist dimensions are statically valid"
)]
pub fn default_dory_assist_dimensions() -> DoryAssistDimensions {
    let unpacked = DoryAssistDimensions::new(
        GtDimensions::new(7, 2, 3),
        G1Dimensions::new(8, 2, 3),
        G2Dimensions::new(8, 2, 3),
        MillerLoopDimensions::new(7, 2, 8),
        DoryReduceDimensions::new(2, 1),
        WiringDimensions::new(6),
        PrefixPackingDimensions::new(0, 0, 0).expect("valid empty packing dimensions"),
    );
    let packing = composition::prefix_packing_catalog(unpacked)
        .minimal_dimensions()
        .expect("valid canonical packing dimensions");

    DoryAssistDimensions::new(
        unpacked.gt,
        unpacked.g1,
        unpacked.g2,
        unpacked.miller_loop,
        unpacked.dory_reduce,
        unpacked.wiring,
        packing,
    )
}
