//! Canonical polynomial identifiers for the Jolt protocol.
//!
//! Every polynomial in the protocol — committed or virtual — has a unique
//! [`PolynomialId`]. The protocol module, witness buffers, and prover/verifier
//! all reference polynomials by this ID directly — no index remapping.

use serde::{Deserialize, Serialize};

use crate::module::PolyId;

/// Identifies a polynomial in the Jolt protocol.
///
/// Committed polynomials have PCS commitments and are opened at the
/// unified point. Virtual polynomials are derived during proving and
/// verified through sumcheck output formulas.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum PolynomialId {
    // ── Committed (PCS-opened) ──────────────────────────────────────
    SpartanWitness,
    RamInc,
    RdInc,
    InstructionRa(usize),
    BytecodeRa(usize),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,

    // ── Virtual: memory subsystem ───────────────────────────────────
    RamReadValue,
    RamWriteValue,
    RamAddress,
    RamVal,
    RamValFinal,
    HammingWeight,
    RamWa,

    // ── Virtual: register subsystem ─────────────────────────────────
    RdWriteValue,
    Rs1Value,
    Rs2Value,
    RegistersVal,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    Rd,

    // ── Virtual: instruction lookups ────────────────────────────────
    LookupOutput,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,

    // ── Virtual: R1CS products ──────────────────────────────────────
    Product,
    ShouldBranch,
    ShouldJump,

    // ── Virtual: instruction flags ──────────────────────────────────
    IsRdNotZero,
    WriteLookupToRdFlag,
    JumpFlag,
    BranchFlag,
    NoopFlag,

    // ── Virtual: instruction input decomposition ────────────────────
    LeftIsRs1,
    LeftIsPc,
    RightIsRs2,
    RightIsImm,
    UnexpandedPc,
    Imm,

    // ── Virtual: shift (next-cycle) ─────────────────────────────────
    NextUnexpandedPc,
    NextPc,
    NextIsVirtual,
    NextIsFirstInSequence,
    NextIsNoop,

    // ── Virtual: circuit flags ──────────────────────────────────────
    /// Index matches jolt-instructions CircuitFlags enum order.
    OpFlag(usize),

    // ── Virtual: bytecode ───────────────────────────────────────────
    ExpandedPc,
    InstructionRafFlag,
    LookupTableFlag(usize),
    BytecodeReadRafVal(usize),
    InstructionReadRafVal(usize),

    // ── Virtual: RAM subsystem ──────────────────────────────────────
    RamCombinedRa,
    RamRafRa,
    InstructionRafRa,
    BytecodeRafRa,

    // ── Virtual: Hamming weight reduction ────────────────────────────
    HammingG(usize),

    // ── Virtual: advice address phase ───────────────────────────────
    TrustedAdviceAddr,
    UntrustedAdviceAddr,

    // ── Virtual: Spartan internals ──────────────────────────────────
    SpartanEq,
    ProductLeft,
    ProductRight,
    OuterUniskipEval,
    ProductUniskipEval,

    // ── Virtual: Spartan R1CS ───────────────────────────────────────
    Az,
    Bz,
    Cz,
    CombinedRow,

    // ── Public: preprocessed ────────────────────────────────────────
    IoMask,
    ValIo,
    RamUnmap,
    RamInit,
    LookupTable,
    BytecodeTable(usize),
}

impl PolyId for PolynomialId {}

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
