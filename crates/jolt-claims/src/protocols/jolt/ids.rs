use derive_more::From;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltRelationId {
    SpartanOuter,
    SpartanProductVirtualization,
    SpartanShift,
    InstructionClaimReduction,
    InstructionInputVirtualization,
    InstructionReadRaf,
    InstructionRaVirtualization,
    RamReadWriteChecking,
    RamRafEvaluation,
    RamOutputCheck,
    RamValCheck,
    RamRaClaimReduction,
    RamHammingBooleanity,
    RamRaVirtualization,
    RegistersClaimReduction,
    RegistersReadWriteChecking,
    RegistersValEvaluation,
    BytecodeReadRaf,
    Booleanity,
    AdviceClaimReductionCyclePhase,
    AdviceClaimReduction,
    BytecodeClaimReductionCyclePhase,
    BytecodeClaimReduction,
    ProgramImageClaimReductionCyclePhase,
    ProgramImageClaimReduction,
    IncClaimReduction,
    HammingWeightClaimReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamReadWriteChallenge {
    Gamma,
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamValCheckChallenge {
    Gamma,
    LtCyclePlusGamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamRaClaimReductionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamRaVirtualizationChallenge {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamHammingBooleanityChallenge {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamRafEvaluationPublic {
    UnmapAddress,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamOutputCheckPublic {
    EqIoMask,
    NegEqIoMaskValIo,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamRaClaimReductionPublic {
    EqCycleRaf,
    EqCycleReadWrite,
    EqCycleValCheck,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BooleanityChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BooleanityPublic {
    EqAddressCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncClaimReductionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncClaimReductionPublic {
    EqRamReadWrite,
    EqRamValCheck,
    EqRegistersReadWrite,
    EqRegistersValEvaluation,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HammingWeightClaimReductionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HammingWeightClaimReductionPublic {
    EqBooleanity,
    EqVirtualization(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeReadRafChallenge {
    Gamma,
    Stage1Gamma,
    Stage2Gamma,
    Stage3Gamma,
    Stage4Gamma,
    Stage5Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeReadRafPublic {
    StageValue(usize),
    SpartanOuterRaf,
    SpartanShiftRaf,
    Entry,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltAdviceKind {
    Trusted,
    Untrusted,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AdviceClaimReductionPublic {
    FinalScale(JoltAdviceKind),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpartanShiftChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpartanShiftPublic {
    EqPlusOneOuter,
    EqPlusOneProduct,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpartanProductVirtualizationPublic {
    UniskipLagrangeWeight(usize),
    LagrangeWeight(usize),
    TauKernel,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpartanOuterPublic {
    QuadraticCoefficient { left: usize, right: usize },
    LinearCoefficient(usize),
    ConstantCoefficient,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersReadWriteChallenge {
    Gamma,
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersValEvaluationChallenge {
    LtCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersClaimReductionChallenge {
    Gamma,
    EqSpartan,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionClaimReductionChallenge {
    Gamma,
    EqSpartan,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionInputChallenge {
    Gamma,
    EqProduct,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionReadRafChallenge {
    Gamma,
    EqTableValue(usize),
    EqRafConstant,
    EqRafFlag,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionRaVirtualizationChallenge {
    Gamma,
    EqCycle,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum JoltChallengeId {
    RamReadWrite(RamReadWriteChallenge),
    RamValCheck(RamValCheckChallenge),
    RamRaClaimReduction(RamRaClaimReductionChallenge),
    RamRaVirtualization(RamRaVirtualizationChallenge),
    RamHammingBooleanity(RamHammingBooleanityChallenge),
    Booleanity(BooleanityChallenge),
    IncClaimReduction(IncClaimReductionChallenge),
    HammingWeightClaimReduction(HammingWeightClaimReductionChallenge),
    BytecodeReadRaf(BytecodeReadRafChallenge),
    SpartanShift(SpartanShiftChallenge),
    RegistersReadWrite(RegistersReadWriteChallenge),
    RegistersValEvaluation(RegistersValEvaluationChallenge),
    RegistersClaimReduction(RegistersClaimReductionChallenge),
    InstructionClaimReduction(InstructionClaimReductionChallenge),
    InstructionInput(InstructionInputChallenge),
    InstructionReadRaf(InstructionReadRafChallenge),
    InstructionRaVirtualization(InstructionRaVirtualizationChallenge),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltCommittedPolynomial {
    RdInc,
    RamInc,
    InstructionRa(usize),
    BytecodeRa(usize),
    BytecodeChunk(usize),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,
    ProgramImageInit,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltVirtualPolynomial {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa(usize),
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
    OpFlags(CircuitFlags),
    InstructionFlags(InstructionFlags),
    LookupTableFlag(usize),
    BytecodeValStage(usize),
    BytecodeReadRafAddrClaim,
    BooleanityAddrClaim,
    BytecodeClaimReductionIntermediate,
    ProgramImageInitContributionRw,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum JoltPolynomialId {
    Committed(JoltCommittedPolynomial),
    Virtual(JoltVirtualPolynomial),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltOpeningId {
    Polynomial {
        polynomial: JoltPolynomialId,
        relation: JoltRelationId,
    },
    UntrustedAdvice {
        relation: JoltRelationId,
    },
    TrustedAdvice {
        relation: JoltRelationId,
    },
}

impl JoltOpeningId {
    pub fn polynomial(polynomial: impl Into<JoltPolynomialId>, relation: JoltRelationId) -> Self {
        Self::Polynomial {
            polynomial: polynomial.into(),
            relation,
        }
    }

    pub fn committed(polynomial: JoltCommittedPolynomial, relation: JoltRelationId) -> Self {
        Self::polynomial(polynomial, relation)
    }

    pub fn virtual_polynomial(polynomial: JoltVirtualPolynomial, relation: JoltRelationId) -> Self {
        Self::polynomial(polynomial, relation)
    }

    pub fn untrusted_advice(relation: JoltRelationId) -> Self {
        Self::UntrustedAdvice { relation }
    }

    pub fn trusted_advice(relation: JoltRelationId) -> Self {
        Self::TrustedAdvice { relation }
    }
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum JoltPublicId {
    TraceLength,
    PaddedTraceLength,
    BytecodeLength,
    MemorySize,
    RamRafEvaluation(RamRafEvaluationPublic),
    RamOutputCheck(RamOutputCheckPublic),
    RamRaClaimReduction(RamRaClaimReductionPublic),
    Booleanity(BooleanityPublic),
    IncClaimReduction(IncClaimReductionPublic),
    HammingWeightClaimReduction(HammingWeightClaimReductionPublic),
    BytecodeReadRaf(BytecodeReadRafPublic),
    AdviceClaimReduction(AdviceClaimReductionPublic),
    SpartanShift(SpartanShiftPublic),
    SpartanProductVirtualization(SpartanProductVirtualizationPublic),
    SpartanOuter(SpartanOuterPublic),
    #[from(ignore)]
    PublicInput(usize),
    #[from(ignore)]
    PublicOutput(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_constructors_preserve_stage_context() {
        let relation = JoltRelationId::RamReadWriteChecking;

        assert_eq!(
            JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, relation),
            JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::RamInc),
                relation,
            }
        );
        assert_eq!(
            JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamVal, relation),
            JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamVal),
                relation,
            }
        );
    }
}
