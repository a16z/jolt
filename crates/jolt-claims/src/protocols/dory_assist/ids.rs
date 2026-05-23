use derive_more::From;
use serde::{Deserialize, Serialize};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DoryAssistRelationId {
    GtExponentiation,
    GtExponentiationDigitSelector,
    GtExponentiationBasePower,
    GtExponentiationShift,
    GtMultiplication,
    G1ScalarMultiplication,
    G1ScalarMultiplicationShift,
    G1Addition,
    G2ScalarMultiplication,
    G2ScalarMultiplicationShift,
    G2Addition,
    MultiMillerLoop,
    WiringGt,
    WiringG1,
    WiringG2,
    PrefixPacking,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GtChallenge {
    InstanceBatch,
    ConstraintBatch,
    ShiftGamma,
    BoundaryPoint,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum G1Challenge {
    ConstraintBatch,
    ShiftGamma,
    BoundaryPoint,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum G2Challenge {
    ConstraintBatch,
    ShiftGamma,
    BoundaryPoint,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PairingChallenge {
    MillerLoopBatch,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WiringChallenge {
    CopyPoint,
    EdgeBatch,
    TupleCompression,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PackingChallenge {
    PrefixPoint,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum DoryAssistChallengeId {
    Gt(GtChallenge),
    G1(G1Challenge),
    G2(G2Challenge),
    Pairing(PairingChallenge),
    Wiring(WiringChallenge),
    Packing(PackingChallenge),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DoryAssistCommittedPolynomial {
    DenseWitness,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GtPolynomial {
    ExpAccumulator,
    ExpShiftedAccumulator,
    ExpQuotient,
    ExpDigitSelector,
    ExpDigitBit(usize),
    ExpBasePower(usize),
    ExpBasePowerQuotient(usize),
    Modulus,
    MulLeft,
    MulRight,
    MulOutput,
    MulQuotient,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum G1Polynomial {
    ScalarMulAccumulatorX,
    ScalarMulAccumulatorY,
    ScalarMulAccumulatorInfinity,
    ScalarMulDoubledX,
    ScalarMulDoubledY,
    ScalarMulDoubledInfinity,
    ScalarMulShiftedAccumulatorX,
    ScalarMulShiftedAccumulatorY,
    ScalarMulBit,
    ScalarMulBaseX,
    ScalarMulBaseY,
    AddInputLeftX,
    AddInputLeftY,
    AddInputLeftInfinity,
    AddInputRightX,
    AddInputRightY,
    AddInputRightInfinity,
    AddOutputX,
    AddOutputY,
    AddOutputInfinity,
    AddSlope,
    AddInverse,
    AddBranchSelector(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum G2Polynomial {
    ScalarMulAccumulatorX0,
    ScalarMulAccumulatorX1,
    ScalarMulAccumulatorY0,
    ScalarMulAccumulatorY1,
    ScalarMulAccumulatorInfinity,
    ScalarMulDoubledX0,
    ScalarMulDoubledX1,
    ScalarMulDoubledY0,
    ScalarMulDoubledY1,
    ScalarMulDoubledInfinity,
    ScalarMulShiftedAccumulatorX0,
    ScalarMulShiftedAccumulatorX1,
    ScalarMulShiftedAccumulatorY0,
    ScalarMulShiftedAccumulatorY1,
    ScalarMulBit,
    ScalarMulBaseX0,
    ScalarMulBaseX1,
    ScalarMulBaseY0,
    ScalarMulBaseY1,
    AddInputLeftX0,
    AddInputLeftX1,
    AddInputLeftY0,
    AddInputLeftY1,
    AddInputLeftInfinity,
    AddInputRightX0,
    AddInputRightX1,
    AddInputRightY0,
    AddInputRightY1,
    AddInputRightInfinity,
    AddOutputX0,
    AddOutputX1,
    AddOutputY0,
    AddOutputY1,
    AddOutputInfinity,
    AddSlope0,
    AddSlope1,
    AddInverse0,
    AddInverse1,
    AddAuxiliary(usize),
    AddBranchSelector(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PairingPolynomial {
    MillerAccumulator,
    MillerLineEvaluation,
    MillerPointX,
    MillerPointY,
    MillerQuotient,
    PreFinalExponentiationOutput,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WiringPolynomial {
    Source,
    Destination,
    Enabled,
    Difference,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PackingPolynomial {
    DenseWitness,
    PrefixSelector,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum DoryAssistVirtualPolynomial {
    Gt(GtPolynomial),
    G1(G1Polynomial),
    G2(G2Polynomial),
    Pairing(PairingPolynomial),
    Wiring(WiringPolynomial),
    Packing(PackingPolynomial),
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum DoryAssistPolynomialId {
    Committed(DoryAssistCommittedPolynomial),
    Virtual(DoryAssistVirtualPolynomial),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DoryAssistOpeningId {
    Polynomial {
        polynomial: DoryAssistPolynomialId,
        relation: DoryAssistRelationId,
    },
}

impl DoryAssistOpeningId {
    pub fn polynomial(
        polynomial: impl Into<DoryAssistPolynomialId>,
        relation: DoryAssistRelationId,
    ) -> Self {
        Self::Polynomial {
            polynomial: polynomial.into(),
            relation,
        }
    }

    pub fn committed(
        polynomial: DoryAssistCommittedPolynomial,
        relation: DoryAssistRelationId,
    ) -> Self {
        Self::polynomial(polynomial, relation)
    }

    pub fn virtual_polynomial(
        polynomial: impl Into<DoryAssistVirtualPolynomial>,
        relation: DoryAssistRelationId,
    ) -> Self {
        Self::polynomial(polynomial.into(), relation)
    }

    pub fn dense_witness(relation: DoryAssistRelationId) -> Self {
        Self::committed(DoryAssistCommittedPolynomial::DenseWitness, relation)
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DoryAssistPublicId {
    DoryProofArtifact(usize),
    VerifierSetupDigest,
    JoltEvaluationClaim(usize),
    JoltCommitment(usize),
    TranscriptScalar(usize),
    GtShiftEqKernel,
    PrefixPackingWeight(usize),
    PreFinalExponentiationGt(usize),
    PairingFinalCheckInput(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DoryAssistValueType {
    Scalar,
    Fq2,
    G1,
    G2,
    Gt,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DoryAssistValueRef {
    Witness {
        polynomial: DoryAssistVirtualPolynomial,
        row: usize,
        component: usize,
    },
    Public {
        id: DoryAssistPublicId,
        component: usize,
    },
    Challenge(DoryAssistChallengeId),
    Constant(usize),
}
