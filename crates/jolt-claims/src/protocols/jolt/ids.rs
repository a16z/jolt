use derive_more::From;
use jolt_openings::CommitmentGroupRole;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

use crate::Expr;

pub type JoltExpr<F> = Expr<F, JoltOpeningId, JoltDerivedId, JoltChallengeId>;

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
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamReadWritePublic {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamValCheckChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamValCheckPublic {
    LtCyclePlusGamma,
    /// `Val_init(r_address)`'s public portion — the part of the initial RAM
    /// evaluation not carried by committed advice / program-image openings.
    InitEval,
    /// The negated block selector (`-selector`) weighting one committed advice
    /// contribution (`untrusted`/`trusted`) in the `Val_init` decomposition.
    InitSelector(JoltAdviceKind),
    /// The negated selector (`-1`) weighting the committed program-image
    /// contribution in the `Val_init` decomposition (committed program mode).
    InitSelectorProgramImage,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamRaClaimReductionChallenge {
    Gamma,
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
pub enum RamRafEvaluationPublic {
    UnmapAddress,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamOutputCheckPublic {
    EqAddress,
    IoMask,
    ValIo,
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
    /// The batching weight γ^exponent, shared by the two terms of one opening.
    GammaPow {
        exponent: usize,
    },
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
    /// Akita path: `eq(r_booleanity_address, 0)` — the digit-zero baseline
    /// weight of a Booleanity leg (`specs/digit-zero-virtualization.md`).
    EqBooleanityAtDigitZero,
    EqVirtualization(usize),
    /// Akita path: `eq(r_virtualization_address_i, 0)` — the digit-zero
    /// baseline weight of a virtualization leg.
    EqVirtualizationAtDigitZero(usize),
    BalancedIncValueAtAddress,
    /// `gamma^k` for `k >= 2` as one derived leaf: the batching coefficient of
    /// the k-th leg.
    GammaPow(usize),
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
    /// Committed program mode: `eq(stage_cycle_point_s, r_cycle)` factor
    /// multiplying the staged `BytecodeValClaim(s)` opening. In full mode this
    /// factor is folded into `StageValue(s)` instead.
    StageCycleEq(usize),
    SpartanOuterRaf,
    SpartanShiftRaf,
    Entry,
    /// `challenge^exponent` for an exponent of at least two, as one derived
    /// leaf (see `geometry::bytecode::challenge_pow_expr`).
    ChallengePow {
        challenge: BytecodeReadRafChallenge,
        exponent: usize,
    },
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub enum JoltAdviceKind {
    Trusted,
    Untrusted,
}

impl JoltAdviceKind {
    /// Role descriptor for the final heterogeneous Akita opening.
    pub const fn group_role(self) -> CommitmentGroupRole {
        match self {
            Self::Untrusted => CommitmentGroupRole::new(0, b"untrusted_advice", "untrusted-advice"),
            Self::Trusted => CommitmentGroupRole::new(1, b"trusted_advice", "trusted-advice"),
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AdviceClaimReductionPublic {
    FinalScale(JoltAdviceKind),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeClaimReductionChallenge {
    Eta,
}

/// Uninhabited payload preserving retired identifier slots in the wire format.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RetiredIdentifier {}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeClaimReductionPublic {
    // Reserved codec slot; uninhabited and rejected during deserialization.
    Reserved0(RetiredIdentifier),
    OutputWeight,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProgramImageClaimReductionPublic {
    FinalScale,
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
    /// `LK(τ_high, r₀) · eq(τ_low, ·)` — the uni-skip kernel times the cycle eq.
    TauKernel,
    /// The `Az` linear-form weight of R1CS input column `i` (linear in the
    /// stream variable).
    AzWeight(usize),
    /// The `Bz` linear-form weight of R1CS input column `i`.
    BzWeight(usize),
    /// The `Az` linear form's public-column contribution.
    AzConstant,
    /// The `Bz` linear form's public-column contribution.
    BzConstant,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersReadWriteChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersReadWritePublic {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersValEvaluationPublic {
    LtCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersClaimReductionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersClaimReductionPublic {
    EqSpartan,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionClaimReductionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionClaimReductionPublic {
    EqSpartan,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionInputChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionInputPublic {
    EqProduct,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionReadRafChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionReadRafPublic {
    EqTableValue(usize),
    EqRafConstant,
    EqRafFlag,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionRaVirtualizationChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InstructionRaVirtualizationPublic {
    EqCycle,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum JoltChallengeId {
    RamReadWrite(RamReadWriteChallenge),
    RamValCheck(RamValCheckChallenge),
    RamRaClaimReduction(RamRaClaimReductionChallenge),
    Booleanity(BooleanityChallenge),
    IncClaimReduction(IncClaimReductionChallenge),
    HammingWeightClaimReduction(HammingWeightClaimReductionChallenge),
    BytecodeReadRaf(BytecodeReadRafChallenge),
    BytecodeClaimReduction(BytecodeClaimReductionChallenge),
    SpartanShift(SpartanShiftChallenge),
    RegistersReadWrite(RegistersReadWriteChallenge),
    RegistersClaimReduction(RegistersClaimReductionChallenge),
    InstructionClaimReduction(InstructionClaimReductionChallenge),
    InstructionInput(InstructionInputChallenge),
    InstructionReadRaf(InstructionReadRafChallenge),
    InstructionRaVirtualization(InstructionRaVirtualizationChallenge),
}

/// Lattice layouts supply an explicit canonical identifier order; `Ord` is
/// used only for keyed lookup and does not assign physical slots.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JoltCommittedPolynomial {
    RdInc,
    RamInc,
    InstructionRa(usize),
    BytecodeRa(usize),
    // Reserved codec slot; uninhabited and rejected during deserialization.
    Reserved4(RetiredIdentifier),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,
    ProgramImageInit,
    // Lattice-mode committed polynomials (columns of the native trace group); base
    // mode never constructs these. Appended for codec stability.
    BalancedIncDigit(usize),
    BalancedIncCarry,
    ProgramBytecode,
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
    BytecodeValClaim(usize),
    BytecodeReadRafAddrClaim,
    BooleanityAddrClaim,
    BytecodeClaimReductionIntermediate,
    ProgramImageInitContributionRw,
    // Lattice-mode: the gamma-batched RamInc/RdInc stream (its destination
    // selector is the existing `OpFlags(Store)`; its unsigned shift is a
    // constant folded into the chunk reconstruction). Appended for codec
    // stability.
    FusedInc,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum JoltPolynomialId {
    Committed(JoltCommittedPolynomial),
    Virtual(JoltVirtualPolynomial),
}

// Word-aligned so that the verifier's many id moves and comparisons are word
// operations rather than byte-granular ones, which a zkVM guest pays several
// trace rows per byte for.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(align(8))]
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
    /// The polynomial this opening refers to; advice openings resolve to
    /// their committed advice polynomials.
    pub const fn polynomial_id(self) -> JoltPolynomialId {
        match self {
            Self::Polynomial { polynomial, .. } => polynomial,
            Self::TrustedAdvice { .. } => {
                JoltPolynomialId::Committed(JoltCommittedPolynomial::TrustedAdvice)
            }
            Self::UntrustedAdvice { .. } => {
                JoltPolynomialId::Committed(JoltCommittedPolynomial::UntrustedAdvice)
            }
        }
    }

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
pub enum JoltDerivedId {
    RamReadWrite(RamReadWritePublic),
    RamValCheck(RamValCheckPublic),
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
    BytecodeClaimReduction(BytecodeClaimReductionPublic),
    ProgramImageClaimReduction(ProgramImageClaimReductionPublic),
    SpartanShift(SpartanShiftPublic),
    SpartanProductVirtualization(SpartanProductVirtualizationPublic),
    SpartanOuter(SpartanOuterPublic),
    RegistersReadWrite(RegistersReadWritePublic),
    RegistersValEvaluation(RegistersValEvaluationPublic),
    RegistersClaimReduction(RegistersClaimReductionPublic),
    InstructionClaimReduction(InstructionClaimReductionPublic),
    InstructionInput(InstructionInputPublic),
    InstructionReadRaf(InstructionReadRafPublic),
    InstructionRaVirtualization(InstructionRaVirtualizationPublic),
    /// Test-only derived id for toy relations that define their own
    /// `derive_output_term`. Gated on `any(test, feature = "test-utils")`
    /// rather than `test` alone because `cfg(test)` is per-crate: a downstream
    /// crate's tests (the naive prover's toy relation foremost) see this crate
    /// compiled without `cfg(test)` and reach the variant via the feature.
    /// Kept LAST so toggling the feature cannot shift the discriminants of
    /// real variants across feature-unified builds.
    #[cfg(any(test, feature = "test-utils"))]
    Test,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "wire regression tests fail loudly")]
mod tests {
    use super::*;

    #[test]
    fn advice_group_roles_fix_order_and_transcript_tags() {
        let untrusted = JoltAdviceKind::Untrusted.group_role();
        let trusted = JoltAdviceKind::Trusted.group_role();

        assert!(untrusted.order() < trusted.order());
        assert_eq!(untrusted.transcript_label(), b"untrusted_advice");
        assert_eq!(trusted.transcript_label(), b"trusted_advice");
    }

    #[test]
    fn retired_bytecode_identifiers_are_rejected_without_shifting_live_tags() {
        let config = bincode::config::standard();
        for (id, tag) in [
            (JoltCommittedPolynomial::RamRa(3), vec![5, 3]),
            (JoltCommittedPolynomial::ProgramImageInit, vec![8]),
            (JoltCommittedPolynomial::ProgramBytecode, vec![11]),
        ] {
            assert_eq!(bincode::serde::encode_to_vec(id, config).unwrap(), tag);
            let (decoded, consumed) =
                bincode::serde::decode_from_slice::<JoltCommittedPolynomial, _>(&tag, config)
                    .unwrap();
            assert_eq!(decoded, id);
            assert_eq!(consumed, tag.len());
        }
        assert!(
            bincode::serde::decode_from_slice::<JoltCommittedPolynomial, _>(&[4, 0], config)
                .is_err()
        );
        assert_eq!(
            bincode::serde::encode_to_vec(BytecodeClaimReductionPublic::OutputWeight, config)
                .unwrap(),
            vec![1]
        );
        assert!(
            bincode::serde::decode_from_slice::<BytecodeClaimReductionPublic, _>(&[0, 0], config)
                .is_err()
        );
    }
}

#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltRelationId {
    SpartanOuter = 0,
    SpartanProductVirtualization = 1,
    SpartanShift = 2,
    InstructionClaimReduction = 3,
    InstructionInputVirtualization = 4,
    InstructionReadRaf = 5,
    InstructionRaVirtualization = 6,
    RamReadWriteChecking = 7,
    RamRafEvaluation = 8,
    RamOutputCheck = 9,
    RamValCheck = 10,
    RamRaClaimReduction = 11,
    RamHammingBooleanity = 12,
    RamRaVirtualization = 13,
    RegistersClaimReduction = 14,
    RegistersReadWriteChecking = 15,
    RegistersValEvaluation = 16,
    BytecodeReadRaf = 17,
    Booleanity = 18,
    AdviceClaimReductionCyclePhase = 19,
    AdviceClaimReduction = 20,
    BytecodeClaimReductionCyclePhase = 21,
    BytecodeClaimReduction = 22,
    ProgramImageClaimReductionCyclePhase = 23,
    ProgramImageClaimReduction = 24,
    IncClaimReduction = 25,
    HammingWeightClaimReduction = 26,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamReadWriteChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamReadWritePublic {
    EqCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamValCheckChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamValCheckPublic {
    LtCyclePlusGamma = 0,
    InitEval = 1,
    InitSelector(value0: JoltAdviceKind) = 2,
    InitSelectorProgramImage = 3,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamRaClaimReductionChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamRaVirtualizationPublic {
    EqCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamHammingBooleanityPublic {
    EqCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamRafEvaluationPublic {
    UnmapAddress = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamOutputCheckPublic {
    EqAddress = 0,
    IoMask = 1,
    ValIo = 2,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RamRaClaimReductionPublic {
    EqCycleRaf = 0,
    EqCycleReadWrite = 1,
    EqCycleValCheck = 2,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; BooleanityChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; BooleanityPublic {
    EqAddressCycle = 0,
    GammaPow { exponent: usize } = 1,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; IncClaimReductionChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; IncClaimReductionPublic {
    EqRamReadWrite = 0,
    EqRamValCheck = 1,
    EqRegistersReadWrite = 2,
    EqRegistersValEvaluation = 3,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; HammingWeightClaimReductionChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; HammingWeightClaimReductionPublic {
    EqBooleanity = 0,
    EqBooleanityAtDigitZero = 1,
    EqVirtualization(value0: usize) = 2,
    EqVirtualizationAtDigitZero(value0: usize) = 3,
    BalancedIncValueAtAddress = 4,
    GammaPow(value0: usize) = 5,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; BytecodeReadRafChallenge {
    Gamma = 0,
    Stage1Gamma = 1,
    Stage2Gamma = 2,
    Stage3Gamma = 3,
    Stage4Gamma = 4,
    Stage5Gamma = 5,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; BytecodeReadRafPublic {
    StageValue(value0: usize) = 0,
    StageCycleEq(value0: usize) = 1,
    SpartanOuterRaf = 2,
    SpartanShiftRaf = 3,
    Entry = 4,
    ChallengePow { challenge: BytecodeReadRafChallenge, exponent: usize } = 5,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltAdviceKind {
    Trusted = 0,
    Untrusted = 1,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; AdviceClaimReductionPublic {
    FinalScale(value0: JoltAdviceKind) = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; BytecodeClaimReductionChallenge {
    Eta = 0,
} }
#[cfg(feature = "immutable-recipes")]
use crate::immutable_recipes::{KeySink, KeyVisit};
#[cfg(feature = "immutable-recipes")]
impl KeyVisit for RetiredIdentifier {
    fn visit(&self, _sink: &mut dyn KeySink) -> bool {
        match *self {}
    }
}
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; BytecodeClaimReductionPublic {
    Reserved0(value0: RetiredIdentifier) = 0,
    OutputWeight = 1,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; ProgramImageClaimReductionPublic {
    FinalScale = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; SpartanShiftChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; SpartanShiftPublic {
    EqPlusOneOuter = 0,
    EqPlusOneProduct = 1,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; SpartanProductVirtualizationPublic {
    UniskipLagrangeWeight(value0: usize) = 0,
    LagrangeWeight(value0: usize) = 1,
    TauKernel = 2,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; SpartanOuterPublic {
    TauKernel = 0,
    AzWeight(value0: usize) = 1,
    BzWeight(value0: usize) = 2,
    AzConstant = 3,
    BzConstant = 4,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RegistersReadWriteChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RegistersReadWritePublic {
    EqCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RegistersValEvaluationPublic {
    LtCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RegistersClaimReductionChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; RegistersClaimReductionPublic {
    EqSpartan = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionClaimReductionChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionClaimReductionPublic {
    EqSpartan = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionInputChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionInputPublic {
    EqProduct = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionReadRafChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionReadRafPublic {
    EqTableValue(value0: usize) = 0,
    EqRafConstant = 1,
    EqRafFlag = 2,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionRaVirtualizationChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; InstructionRaVirtualizationPublic {
    EqCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltChallengeId {
    RamReadWrite(value0: RamReadWriteChallenge) = 0,
    RamValCheck(value0: RamValCheckChallenge) = 1,
    RamRaClaimReduction(value0: RamRaClaimReductionChallenge) = 2,
    Booleanity(value0: BooleanityChallenge) = 3,
    IncClaimReduction(value0: IncClaimReductionChallenge) = 4,
    HammingWeightClaimReduction(value0: HammingWeightClaimReductionChallenge) = 5,
    BytecodeReadRaf(value0: BytecodeReadRafChallenge) = 6,
    BytecodeClaimReduction(value0: BytecodeClaimReductionChallenge) = 7,
    SpartanShift(value0: SpartanShiftChallenge) = 8,
    RegistersReadWrite(value0: RegistersReadWriteChallenge) = 9,
    RegistersClaimReduction(value0: RegistersClaimReductionChallenge) = 10,
    InstructionClaimReduction(value0: InstructionClaimReductionChallenge) = 11,
    InstructionInput(value0: InstructionInputChallenge) = 12,
    InstructionReadRaf(value0: InstructionReadRafChallenge) = 13,
    InstructionRaVirtualization(value0: InstructionRaVirtualizationChallenge) = 14,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltCommittedPolynomial {
    RdInc = 0,
    RamInc = 1,
    InstructionRa(value0: usize) = 2,
    BytecodeRa(value0: usize) = 3,
    Reserved4(value0: RetiredIdentifier) = 4,
    RamRa(value0: usize) = 5,
    TrustedAdvice = 6,
    UntrustedAdvice = 7,
    ProgramImageInit = 8,
    BalancedIncDigit(value0: usize) = 9,
    BalancedIncCarry = 10,
    ProgramBytecode = 11,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltVirtualPolynomial {
    PC = 0,
    UnexpandedPC = 1,
    NextPC = 2,
    NextUnexpandedPC = 3,
    NextIsNoop = 4,
    NextIsVirtual = 5,
    NextIsFirstInSequence = 6,
    LeftLookupOperand = 7,
    RightLookupOperand = 8,
    LeftInstructionInput = 9,
    RightInstructionInput = 10,
    Product = 11,
    ShouldJump = 12,
    ShouldBranch = 13,
    Rd = 14,
    Imm = 15,
    Rs1Value = 16,
    Rs2Value = 17,
    RdWriteValue = 18,
    Rs1Ra = 19,
    Rs2Ra = 20,
    RdWa = 21,
    LookupOutput = 22,
    InstructionRaf = 23,
    InstructionRafFlag = 24,
    InstructionRa(value0: usize) = 25,
    RegistersVal = 26,
    RamAddress = 27,
    RamRa = 28,
    RamReadValue = 29,
    RamWriteValue = 30,
    RamVal = 31,
    RamValInit = 32,
    RamValFinal = 33,
    RamHammingWeight = 34,
    UnivariateSkip = 35,
    OpFlags(value0: CircuitFlags) = 36,
    InstructionFlags(value0: InstructionFlags) = 37,
    LookupTableFlag(value0: usize) = 38,
    BytecodeValClaim(value0: usize) = 39,
    BytecodeReadRafAddrClaim = 40,
    BooleanityAddrClaim = 41,
    BytecodeClaimReductionIntermediate = 42,
    ProgramImageInitContributionRw = 43,
    FusedInc = 44,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltPolynomialId {
    Committed(value0: JoltCommittedPolynomial) = 0,
    Virtual(value0: JoltVirtualPolynomial) = 1,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltOpeningId {
    Polynomial { polynomial: JoltPolynomialId, relation: JoltRelationId } = 0,
    UntrustedAdvice { relation: JoltRelationId } = 1,
    TrustedAdvice { relation: JoltRelationId } = 2,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::jolt; JoltDerivedId {
    RamReadWrite(value0: RamReadWritePublic) = 0,
    RamValCheck(value0: RamValCheckPublic) = 1,
    RamRafEvaluation(value0: RamRafEvaluationPublic) = 2,
    RamOutputCheck(value0: RamOutputCheckPublic) = 3,
    RamRaClaimReduction(value0: RamRaClaimReductionPublic) = 4,
    RamRaVirtualization(value0: RamRaVirtualizationPublic) = 5,
    RamHammingBooleanity(value0: RamHammingBooleanityPublic) = 6,
    Booleanity(value0: BooleanityPublic) = 7,
    IncClaimReduction(value0: IncClaimReductionPublic) = 8,
    HammingWeightClaimReduction(value0: HammingWeightClaimReductionPublic) = 9,
    BytecodeReadRaf(value0: BytecodeReadRafPublic) = 10,
    AdviceClaimReduction(value0: AdviceClaimReductionPublic) = 11,
    BytecodeClaimReduction(value0: BytecodeClaimReductionPublic) = 12,
    ProgramImageClaimReduction(value0: ProgramImageClaimReductionPublic) = 13,
    SpartanShift(value0: SpartanShiftPublic) = 14,
    SpartanProductVirtualization(value0: SpartanProductVirtualizationPublic) = 15,
    SpartanOuter(value0: SpartanOuterPublic) = 16,
    RegistersReadWrite(value0: RegistersReadWritePublic) = 17,
    RegistersValEvaluation(value0: RegistersValEvaluationPublic) = 18,
    RegistersClaimReduction(value0: RegistersClaimReductionPublic) = 19,
    InstructionClaimReduction(value0: InstructionClaimReductionPublic) = 20,
    InstructionInput(value0: InstructionInputPublic) = 21,
    InstructionReadRaf(value0: InstructionReadRafPublic) = 22,
    InstructionRaVirtualization(value0: InstructionRaVirtualizationPublic) = 23,
    #[cfg(any(test, feature = "test-utils"))]
    Test = 24,
} }

#[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
use crate::immutable_recipes::record::RenderId;
#[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
use std::fmt::{Result as FmtResult, Write};
#[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
impl RenderId for RetiredIdentifier {
    fn render(&self, _writer: &mut dyn Write) -> FmtResult {
        match *self {}
    }
}
