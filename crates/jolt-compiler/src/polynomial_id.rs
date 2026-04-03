//! Canonical polynomial identifiers for the Jolt protocol.
//!
//! Every polynomial in the protocol — committed or virtual — has a unique
//! [`PolynomialId`]. The protocol module, witness buffers, and prover/verifier
//! all reference polynomials by this ID directly — no index remapping.

use serde::{Deserialize, Serialize};

use crate::descriptor::{PolySource, PolynomialDescriptor, R1csColumn, StorageHint, WitnessSlot};
use crate::module::PolynomialSpec;

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

impl PolynomialSpec for PolynomialId {
    fn descriptor(&self) -> PolynomialDescriptor {
        match self {
            // ── Committed: trace-derived dense ─────────────────────────
            Self::RdInc => PolynomialDescriptor {
                source: PolySource::Witness,
                committed: true,
                storage: StorageHint::Dense,
                witness_slot: Some(WitnessSlot::Dense(WitnessSlot::RD_INC)),
            },
            Self::RamInc => PolynomialDescriptor {
                source: PolySource::Witness,
                committed: true,
                storage: StorageHint::Dense,
                witness_slot: Some(WitnessSlot::Dense(WitnessSlot::RAM_INC)),
            },

            // ── Committed: inserted separately (not from trace) ───────
            Self::SpartanWitness | Self::TrustedAdvice | Self::UntrustedAdvice => {
                PolynomialDescriptor {
                    source: PolySource::Witness,
                    committed: true,
                    storage: StorageHint::Dense,
                    witness_slot: None,
                }
            }

            // ── Committed: trace-derived one-hot ──────────────────────
            Self::InstructionRa(i) => PolynomialDescriptor {
                source: PolySource::Witness,
                committed: true,
                storage: StorageHint::OneHot,
                witness_slot: Some(WitnessSlot::OneHotChunk {
                    source: WitnessSlot::INSTRUCTION,
                    dim: *i,
                }),
            },
            Self::BytecodeRa(i) => PolynomialDescriptor {
                source: PolySource::Witness,
                committed: true,
                storage: StorageHint::OneHot,
                witness_slot: Some(WitnessSlot::OneHotChunk {
                    source: WitnessSlot::BYTECODE,
                    dim: *i,
                }),
            },
            Self::RamRa(i) => PolynomialDescriptor {
                source: PolySource::Witness,
                committed: true,
                storage: StorageHint::OneHot,
                witness_slot: Some(WitnessSlot::OneHotChunk {
                    source: WitnessSlot::RAM,
                    dim: *i,
                }),
            },

            // ── R1CS: computed on demand ───────────────────────────────
            Self::Az => PolynomialDescriptor {
                source: PolySource::R1cs(R1csColumn::Az),
                committed: false,
                storage: StorageHint::OnDemand,
                witness_slot: None,
            },
            Self::Bz => PolynomialDescriptor {
                source: PolySource::R1cs(R1csColumn::Bz),
                committed: false,
                storage: StorageHint::OnDemand,
                witness_slot: None,
            },
            Self::Cz => PolynomialDescriptor {
                source: PolySource::R1cs(R1csColumn::Cz),
                committed: false,
                storage: StorageHint::OnDemand,
                witness_slot: None,
            },
            Self::CombinedRow => PolynomialDescriptor {
                source: PolySource::R1cs(R1csColumn::CombinedRow),
                committed: false,
                storage: StorageHint::OnDemand,
                witness_slot: None,
            },

            // ── Preprocessed: loaded from verifying key ────────────────
            Self::IoMask
            | Self::ValIo
            | Self::RamUnmap
            | Self::RamInit
            | Self::LookupTable
            | Self::BytecodeTable(_) => PolynomialDescriptor {
                source: PolySource::Preprocessed,
                committed: false,
                storage: StorageHint::Dense,
                witness_slot: None,
            },

            // ── Virtual / derived: everything else ─────────────────────
            _ => PolynomialDescriptor {
                source: PolySource::Derived,
                committed: false,
                storage: StorageHint::Dense,
                witness_slot: None,
            },
        }
    }
}

impl PolynomialId {
    /// Whether this polynomial has a PCS commitment.
    #[inline]
    pub fn is_committed(&self) -> bool {
        self.descriptor().committed
    }
}
