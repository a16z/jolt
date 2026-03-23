//! Canonical polynomial identifiers for the Jolt protocol.
//!
//! Every polynomial in the protocol has a unique [`PolynomialId`]. The protocol
//! graph references polynomials by ID; the prover looks up evaluation tables
//! by ID; the verifier tracks opening claims by ID.
//!
//! This replaces the opaque `u64` tags in `zkvm::tags::poly`.

/// Identifies a polynomial in the Jolt protocol.
///
/// Committed polynomials have PCS commitments and are opened at the unified
/// point. Virtual polynomials are derived from R1CS or trace data and are
/// verified through sumcheck output formulas (no PCS opening needed).
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum PolynomialId {
    // Committed (PCS-opened)
    SpartanWitness,
    RamInc,
    RdInc,
    InstructionRa(usize),
    BytecodeRa(usize),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,

    // Virtual: memory subsystem
    RamReadValue,
    RamWriteValue,
    RamAddress,
    RamVal,
    RamValFinal,
    HammingWeight,

    // Virtual: register subsystem
    RdWriteValue,
    Rs1Value,
    Rs2Value,
    RegistersVal,
    Rs1Ra,
    Rs2Ra,
    RdWa,

    // Virtual: instruction lookups
    LookupOutput,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,

    // Virtual: instruction flags
    IsRdNotZero,
    WriteLookupToRdFlag,
    JumpFlag,
    BranchFlag,

    // Virtual: instruction input decomposition
    LeftIsRs1,
    LeftIsPc,
    RightIsRs2,
    RightIsImm,
    UnexpandedPc,
    Imm,

    // Virtual: shift (next-cycle)
    NextUnexpandedPc,
    NextPc,
    NextIsVirtual,
    NextIsFirstInSequence,
    NextIsNoop,
}

impl PolynomialId {
    /// Whether this polynomial has a PCS commitment.
    pub fn is_committed(&self) -> bool {
        matches!(
            self,
            Self::SpartanWitness
                | Self::RamInc
                | Self::RdInc
                | Self::InstructionRa(_)
                | Self::BytecodeRa(_)
                | Self::RamRa(_)
                | Self::TrustedAdvice
                | Self::UntrustedAdvice
        )
    }
}
