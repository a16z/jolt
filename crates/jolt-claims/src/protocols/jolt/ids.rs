use derive_more::From;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

use crate::Expr;

/// The Jolt protocol's expression type: an [`Expr`](crate::Expr) over the Jolt id
/// families (openings, deriveds, challenges). Each relation's `input`/`output`
/// expression is a `JoltExpr<F>`.
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
    IncVirtualization,
    UnsignedIncChunkReconstruction,
    UntrustedAdviceReconstruction,
    TrustedAdviceReconstruction,
    ProgramImageReconstruction,
    BytecodeChunkReconstruction,
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
    /// Committed program mode: `eq(stage_cycle_point_s, r_cycle)` factor
    /// multiplying the staged `BytecodeValStage(s)` opening. In full mode this
    /// factor is folded into `StageValue(s)` instead.
    StageCycleEq(usize),
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
pub enum BytecodeClaimReductionChallenge {
    Eta,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeClaimReductionPublic {
    /// Final output coefficient of one committed bytecode chunk opening:
    /// `eq(r_bc_high)[chunk] * eq_combined * skip_scale`.
    ChunkOutputWeight(usize),
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
    QuadraticCoefficient { left: usize, right: usize },
    LinearCoefficient(usize),
    ConstantCoefficient,
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

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncVirtualizationChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncVirtualizationPublic {
    EqRamReadWrite,
    EqRamValCheck,
    EqRegistersReadWrite,
    EqRegistersValEvaluation,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UnsignedIncChunkReconstructionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UnsignedIncChunkReconstructionPublic {
    /// `eq(r_booleanity_address, r_address)` — the address half of reducing
    /// the chunk/msb openings produced at the booleanity point to this
    /// relation's bound point (also the normalizing address kernel of the msb
    /// legs, whose polynomial has no address variables).
    EqBooleanityAddress,
    /// `eq(r_booleanity_cycle, r_cycle)` — the cycle half of the booleanity
    /// reduction legs.
    EqBooleanityCycle,
    /// `eq(r_inc_virtualization, r_cycle)` — anchors the hamming and
    /// shifted-decode legs at the `IncVirtualization` cycle point, where the
    /// consumed `FusedInc` claim lives.
    EqIncCycle,
    /// The identity MLE `Σ_bit 2^bit · r_address[bit]` at the bound address
    /// point — decodes a one-hot chunk opening into its address value.
    IdentityAtAddress,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UntrustedAdviceReconstructionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UntrustedAdviceReconstructionPublic {
    /// `eq` over the full `(byte ‖ place ‖ word)` domain at the bound point —
    /// weights the booleanity leg.
    EqBytePlaceWord,
    /// `eq` over the `(place ‖ word)` sub-domain at the bound point — weights
    /// the per-byte-place hamming leg (byte variables are summed, not
    /// eq-bound).
    EqPlaceWord,
    /// The [`byte_decode_weight`](crate::protocols::jolt::lattice::geometry::byte_decode_weight)
    /// evaluation at the bound `(byte ‖ place)` coordinates — decodes the
    /// one-hot entries into the little-endian word value.
    ByteDecode,
    /// `eq` of the bound word coordinates against the advice claim
    /// reduction's word point — reduces the incoming word claim to this
    /// relation's bound point.
    EqWord,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustedAdviceReconstructionPublic {
    /// The [`byte_decode_weight`](crate::protocols::jolt::lattice::geometry::byte_decode_weight)
    /// evaluation at the bound `(byte ‖ place)` coordinates (the word point
    /// is fixed by the incoming claim, so no `eq` derived is needed).
    ByteDecode,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProgramImageReconstructionPublic {
    /// The [`byte_decode_weight`](crate::protocols::jolt::lattice::geometry::byte_decode_weight)
    /// evaluation at the bound `(byte ‖ place)` coordinates (the word point
    /// is fixed by the incoming claim).
    ByteDecode,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeChunkReconstructionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeChunkReconstructionPublic {
    /// The register-selector block of `eq(r_lane)` weights as a 5-variable
    /// multilinear, evaluated at the bound register coordinates.
    RegisterSelectorWeight(BytecodeRegisterLane),
    /// A single `eq(r_lane)` weight — the direct (0/1 flag) lanes, indexed by
    /// lane position in the committed lane layout.
    LaneWeight(usize),
    /// The lookup-selector block of `eq(r_lane)` weights, evaluated at the
    /// bound table-index coordinates.
    LookupSelectorWeight,
    /// The unexpanded-pc lane weight times the [`byte_decode_weight`](crate::protocols::jolt::lattice::geometry::byte_decode_weight)
    /// evaluation at the bound `(byte ‖ place)` coordinates.
    PcByteDecode,
    /// The imm lane weight times the [`byte_decode_weight`](crate::protocols::jolt::lattice::geometry::byte_decode_weight)
    /// evaluation at the bound `(byte ‖ place)` coordinates.
    ImmByteDecode,
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
    IncVirtualization(IncVirtualizationChallenge),
    UnsignedIncChunkReconstruction(UnsignedIncChunkReconstructionChallenge),
    UntrustedAdviceReconstruction(UntrustedAdviceReconstructionChallenge),
    BytecodeChunkReconstruction(BytecodeChunkReconstructionChallenge),
}

/// The register-selector lanes of a bytecode row, in committed lane order.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeRegisterLane {
    Rs1,
    Rs2,
    Rd,
}

impl BytecodeRegisterLane {
    pub const ALL: [Self; 3] = [Self::Rs1, Self::Rs2, Self::Rd];
}

/// WARNING: `Ord` is protocol data — the lattice `PrefixPacking` assigns
/// slots by `(num_vars, Ord)` order, so reordering variants silently changes
/// the packed witness layout.
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
    // Lattice-mode committed polynomials (slots of the packed witness); base
    // mode never constructs these. Appended for codec stability.
    UnsignedIncChunk(usize),
    UnsignedIncMsb,
    TrustedAdviceBytes,
    UntrustedAdviceBytes,
    // Lattice-mode precommitted bytecode decompositions: the per-lane one-hot /
    // flag / byte decompositions of `BytecodeChunk(chunk)`, plus the program
    // image byte encoding. Their claims are produced by the reconstruction
    // relations.
    BytecodeRegisterSelector {
        chunk: usize,
        lane: BytecodeRegisterLane,
    },
    BytecodeCircuitFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeInstructionFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeLookupSelector {
        chunk: usize,
    },
    BytecodeRafFlag {
        chunk: usize,
    },
    BytecodeUnexpandedPcBytes {
        chunk: usize,
    },
    BytecodeImmBytes {
        chunk: usize,
    },
    ProgramImageBytes,
}

impl JoltCommittedPolynomial {
    pub fn advice_bytes(kind: JoltAdviceKind) -> Self {
        match kind {
            JoltAdviceKind::Trusted => Self::TrustedAdviceBytes,
            JoltAdviceKind::Untrusted => Self::UntrustedAdviceBytes,
        }
    }
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
pub enum JoltDerivedId {
    TraceLength,
    PaddedTraceLength,
    BytecodeLength,
    MemorySize,
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
    #[from(ignore)]
    PublicInput(usize),
    #[from(ignore)]
    PublicOutput(usize),
    IncVirtualization(IncVirtualizationPublic),
    UnsignedIncChunkReconstruction(UnsignedIncChunkReconstructionPublic),
    UntrustedAdviceReconstruction(UntrustedAdviceReconstructionPublic),
    TrustedAdviceReconstruction(TrustedAdviceReconstructionPublic),
    ProgramImageReconstruction(ProgramImageReconstructionPublic),
    BytecodeChunkReconstruction(BytecodeChunkReconstructionPublic),
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
