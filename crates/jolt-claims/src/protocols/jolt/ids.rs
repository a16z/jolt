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
    EqOnePlusGamma,
    EqGamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltChallengeId {
    RamReadWrite(RamReadWriteChallenge),
}

impl From<RamReadWriteChallenge> for JoltChallengeId {
    fn from(value: RamReadWriteChallenge) -> Self {
        Self::RamReadWrite(value)
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
    PublicInput(usize),
    PublicOutput(usize),
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
