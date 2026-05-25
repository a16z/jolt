//! ID types that preserve the existing `jolt-core` proof key space.

use jolt_riscv::{CircuitFlags, InstructionFlags};

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
pub enum SumcheckId {
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

impl SumcheckId {
    pub const COUNT: usize = 23;

    #[cfg(test)]
    pub const ALL: [Self; Self::COUNT] = [
        Self::SpartanOuter,
        Self::SpartanProductVirtualization,
        Self::SpartanShift,
        Self::InstructionClaimReduction,
        Self::InstructionInputVirtualization,
        Self::InstructionReadRaf,
        Self::InstructionRaVirtualization,
        Self::RamReadWriteChecking,
        Self::RamRafEvaluation,
        Self::RamOutputCheck,
        Self::RamValCheck,
        Self::RamRaClaimReduction,
        Self::RamHammingBooleanity,
        Self::RamRaVirtualization,
        Self::RegistersClaimReduction,
        Self::RegistersReadWriteChecking,
        Self::RegistersValEvaluation,
        Self::BytecodeReadRaf,
        Self::Booleanity,
        Self::AdviceClaimReductionCyclePhase,
        Self::AdviceClaimReduction,
        Self::IncClaimReduction,
        Self::HammingWeightClaimReduction,
    ];
}

impl From<SumcheckId> for u8 {
    fn from(value: SumcheckId) -> Self {
        value as u8
    }
}

impl TryFrom<u8> for SumcheckId {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SpartanOuter),
            1 => Ok(Self::SpartanProductVirtualization),
            2 => Ok(Self::SpartanShift),
            3 => Ok(Self::InstructionClaimReduction),
            4 => Ok(Self::InstructionInputVirtualization),
            5 => Ok(Self::InstructionReadRaf),
            6 => Ok(Self::InstructionRaVirtualization),
            7 => Ok(Self::RamReadWriteChecking),
            8 => Ok(Self::RamRafEvaluation),
            9 => Ok(Self::RamOutputCheck),
            10 => Ok(Self::RamValCheck),
            11 => Ok(Self::RamRaClaimReduction),
            12 => Ok(Self::RamHammingBooleanity),
            13 => Ok(Self::RamRaVirtualization),
            14 => Ok(Self::RegistersClaimReduction),
            15 => Ok(Self::RegistersReadWriteChecking),
            16 => Ok(Self::RegistersValEvaluation),
            17 => Ok(Self::BytecodeReadRaf),
            18 => Ok(Self::Booleanity),
            19 => Ok(Self::AdviceClaimReductionCyclePhase),
            20 => Ok(Self::AdviceClaimReduction),
            21 => Ok(Self::IncClaimReduction),
            22 => Ok(Self::HammingWeightClaimReduction),
            _ => Err(()),
        }
    }
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum CommittedPolynomial {
    RdInc,
    RamInc,
    InstructionRa(usize),
    BytecodeRa(usize),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum VirtualPolynomial {
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

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum PolynomialId {
    Committed(CommittedPolynomial),
    Virtual(VirtualPolynomial),
}

impl From<CommittedPolynomial> for PolynomialId {
    fn from(value: CommittedPolynomial) -> Self {
        Self::Committed(value)
    }
}

impl From<VirtualPolynomial> for PolynomialId {
    fn from(value: VirtualPolynomial) -> Self {
        Self::Virtual(value)
    }
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum OpeningId {
    Polynomial(PolynomialId, SumcheckId),
    UntrustedAdvice(SumcheckId),
    TrustedAdvice(SumcheckId),
}

impl OpeningId {
    pub fn polynomial(poly: impl Into<PolynomialId>, sc: SumcheckId) -> Self {
        Self::Polynomial(poly.into(), sc)
    }

    pub fn committed(poly: CommittedPolynomial, sc: SumcheckId) -> Self {
        Self::polynomial(poly, sc)
    }

    pub fn virt(poly: VirtualPolynomial, sc: SumcheckId) -> Self {
        Self::polynomial(poly, sc)
    }
}
