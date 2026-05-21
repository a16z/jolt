use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltStageId {
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
pub enum RamRaVirtualizationPublic {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamHammingBooleanityPublic {
    EqCycle,
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

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltChallengeId {
    RamReadWrite(RamReadWriteChallenge),
    RamValCheck(RamValCheckChallenge),
    RamRaClaimReduction(RamRaClaimReductionChallenge),
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

impl From<RamReadWriteChallenge> for JoltChallengeId {
    fn from(value: RamReadWriteChallenge) -> Self {
        Self::RamReadWrite(value)
    }
}

impl From<RamValCheckChallenge> for JoltChallengeId {
    fn from(value: RamValCheckChallenge) -> Self {
        Self::RamValCheck(value)
    }
}

impl From<RamRaClaimReductionChallenge> for JoltChallengeId {
    fn from(value: RamRaClaimReductionChallenge) -> Self {
        Self::RamRaClaimReduction(value)
    }
}

impl From<BooleanityChallenge> for JoltChallengeId {
    fn from(value: BooleanityChallenge) -> Self {
        Self::Booleanity(value)
    }
}

impl From<IncClaimReductionChallenge> for JoltChallengeId {
    fn from(value: IncClaimReductionChallenge) -> Self {
        Self::IncClaimReduction(value)
    }
}

impl From<HammingWeightClaimReductionChallenge> for JoltChallengeId {
    fn from(value: HammingWeightClaimReductionChallenge) -> Self {
        Self::HammingWeightClaimReduction(value)
    }
}

impl From<BytecodeReadRafChallenge> for JoltChallengeId {
    fn from(value: BytecodeReadRafChallenge) -> Self {
        Self::BytecodeReadRaf(value)
    }
}

impl From<SpartanShiftChallenge> for JoltChallengeId {
    fn from(value: SpartanShiftChallenge) -> Self {
        Self::SpartanShift(value)
    }
}

impl From<RegistersReadWriteChallenge> for JoltChallengeId {
    fn from(value: RegistersReadWriteChallenge) -> Self {
        Self::RegistersReadWrite(value)
    }
}

impl From<RegistersValEvaluationChallenge> for JoltChallengeId {
    fn from(value: RegistersValEvaluationChallenge) -> Self {
        Self::RegistersValEvaluation(value)
    }
}

impl From<RegistersClaimReductionChallenge> for JoltChallengeId {
    fn from(value: RegistersClaimReductionChallenge) -> Self {
        Self::RegistersClaimReduction(value)
    }
}

impl From<InstructionClaimReductionChallenge> for JoltChallengeId {
    fn from(value: InstructionClaimReductionChallenge) -> Self {
        Self::InstructionClaimReduction(value)
    }
}

impl From<InstructionInputChallenge> for JoltChallengeId {
    fn from(value: InstructionInputChallenge) -> Self {
        Self::InstructionInput(value)
    }
}

impl From<InstructionReadRafChallenge> for JoltChallengeId {
    fn from(value: InstructionReadRafChallenge) -> Self {
        Self::InstructionReadRaf(value)
    }
}

impl From<InstructionRaVirtualizationChallenge> for JoltChallengeId {
    fn from(value: InstructionRaVirtualizationChallenge) -> Self {
        Self::InstructionRaVirtualization(value)
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltCommittedPolynomial {
    RdInc,
    RamInc,
    InstructionRa(usize),
    BytecodeRa(usize),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,
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
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltPolynomialId {
    Committed(JoltCommittedPolynomial),
    Virtual(JoltVirtualPolynomial),
}

impl From<JoltCommittedPolynomial> for JoltPolynomialId {
    fn from(value: JoltCommittedPolynomial) -> Self {
        Self::Committed(value)
    }
}

impl From<JoltVirtualPolynomial> for JoltPolynomialId {
    fn from(value: JoltVirtualPolynomial) -> Self {
        Self::Virtual(value)
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltOpeningId {
    Polynomial {
        polynomial: JoltPolynomialId,
        stage: JoltStageId,
    },
    UntrustedAdvice {
        stage: JoltStageId,
    },
    TrustedAdvice {
        stage: JoltStageId,
    },
}

impl JoltOpeningId {
    pub fn polynomial(polynomial: impl Into<JoltPolynomialId>, stage: JoltStageId) -> Self {
        Self::Polynomial {
            polynomial: polynomial.into(),
            stage,
        }
    }

    pub fn committed(polynomial: JoltCommittedPolynomial, stage: JoltStageId) -> Self {
        Self::polynomial(polynomial, stage)
    }

    pub fn virtual_polynomial(polynomial: JoltVirtualPolynomial, stage: JoltStageId) -> Self {
        Self::polynomial(polynomial, stage)
    }

    pub fn untrusted_advice(stage: JoltStageId) -> Self {
        Self::UntrustedAdvice { stage }
    }

    pub fn trusted_advice(stage: JoltStageId) -> Self {
        Self::TrustedAdvice { stage }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltPublicId {
    TraceLength,
    PaddedTraceLength,
    BytecodeLength,
    MemorySize,
    RamRafEvaluation(RamRafEvaluationPublic),
    RamOutputCheck(RamOutputCheckPublic),
    RamRaClaimReduction(RamRaClaimReductionPublic),
    RamRaVirtualization(RamRaVirtualizationPublic),
    RamHammingBooleanity(RamHammingBooleanityPublic),
    Booleanity(BooleanityPublic),
    IncClaimReduction(IncClaimReductionPublic),
    HammingWeightClaimReduction(HammingWeightClaimReductionPublic),
    BytecodeReadRaf(BytecodeReadRafPublic),
    AdviceClaimReduction(AdviceClaimReductionPublic),
    SpartanShift(SpartanShiftPublic),
    SpartanProductVirtualization(SpartanProductVirtualizationPublic),
    SpartanOuter(SpartanOuterPublic),
    PublicInput(usize),
    PublicOutput(usize),
}

impl From<RamRafEvaluationPublic> for JoltPublicId {
    fn from(value: RamRafEvaluationPublic) -> Self {
        Self::RamRafEvaluation(value)
    }
}

impl From<RamOutputCheckPublic> for JoltPublicId {
    fn from(value: RamOutputCheckPublic) -> Self {
        Self::RamOutputCheck(value)
    }
}

impl From<RamRaClaimReductionPublic> for JoltPublicId {
    fn from(value: RamRaClaimReductionPublic) -> Self {
        Self::RamRaClaimReduction(value)
    }
}

impl From<RamRaVirtualizationPublic> for JoltPublicId {
    fn from(value: RamRaVirtualizationPublic) -> Self {
        Self::RamRaVirtualization(value)
    }
}

impl From<RamHammingBooleanityPublic> for JoltPublicId {
    fn from(value: RamHammingBooleanityPublic) -> Self {
        Self::RamHammingBooleanity(value)
    }
}

impl From<BooleanityPublic> for JoltPublicId {
    fn from(value: BooleanityPublic) -> Self {
        Self::Booleanity(value)
    }
}

impl From<IncClaimReductionPublic> for JoltPublicId {
    fn from(value: IncClaimReductionPublic) -> Self {
        Self::IncClaimReduction(value)
    }
}

impl From<HammingWeightClaimReductionPublic> for JoltPublicId {
    fn from(value: HammingWeightClaimReductionPublic) -> Self {
        Self::HammingWeightClaimReduction(value)
    }
}

impl From<BytecodeReadRafPublic> for JoltPublicId {
    fn from(value: BytecodeReadRafPublic) -> Self {
        Self::BytecodeReadRaf(value)
    }
}

impl From<AdviceClaimReductionPublic> for JoltPublicId {
    fn from(value: AdviceClaimReductionPublic) -> Self {
        Self::AdviceClaimReduction(value)
    }
}

impl From<SpartanShiftPublic> for JoltPublicId {
    fn from(value: SpartanShiftPublic) -> Self {
        Self::SpartanShift(value)
    }
}

impl From<SpartanProductVirtualizationPublic> for JoltPublicId {
    fn from(value: SpartanProductVirtualizationPublic) -> Self {
        Self::SpartanProductVirtualization(value)
    }
}

impl From<SpartanOuterPublic> for JoltPublicId {
    fn from(value: SpartanOuterPublic) -> Self {
        Self::SpartanOuter(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_constructors_preserve_stage_context() {
        let stage = JoltStageId::RamReadWriteChecking;

        assert_eq!(
            JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, stage),
            JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::RamInc),
                stage,
            }
        );
        assert_eq!(
            JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamVal, stage),
            JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamVal),
                stage,
            }
        );
    }
}
