//! Canonical polynomial identifiers for the Jolt protocol.
//!
//! Every polynomial in the protocol — committed or virtual — has a unique
//! [`PolynomialId`]. The protocol module, witness buffers, and prover/verifier
//! all reference polynomials by this ID directly — no index remapping.

use serde::{Deserialize, Serialize};

use crate::descriptor::{PolySource, PolynomialDescriptor, R1csColumn, StorageHint, WitnessSlot};

/// Identifies a polynomial in the Jolt protocol.
///
/// Committed polynomials have PCS commitments and are opened at the
/// unified point. Virtual polynomials are derived during proving and
/// verified through sumcheck output formulas.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
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

    // FieldReg coprocessor: Committed polys.
    // ReadValue / WriteValue published pre-Twist and opened at rw_cycle via
    // CollectOpeningClaimAt (their values bind the Twist input_claim).
    FieldRegReadValue,
    FieldRegWriteValue,
    // Inc / Ra / Val — committed in the ideal case but runtime's segmented
    // reduce currently leaves their per-poly final-bind in a non-scalar state.
    // Declared here for namespace hygiene; treated as Virtual in Phase 1/2
    // standalone tests.
    FieldRegInc,
    FieldRegRa,
    FieldRegVal,

    // FieldReg coprocessor: Virtual polys (scratch/derived).
    FieldRegEqCycle,

    // BN254 Fr FieldOp R1CS columns (direct R1CS witness variables at indices
    // 40, 41, 42). Used by FADD/FSUB gates (rows 19-20) + FMUL/FINV gates
    // (rows 21-26) in `jolt-r1cs/src/constraints/rv64.rs`. Populated by the
    // event overlay `apply_field_op_events_to_r1cs`; evaluations flow through
    // the outer Spartan remaining sumcheck so the verifier can reconstruct
    // Az/Bz over rows 19-26.
    FieldOpOperandA,
    FieldOpOperandB,
    FieldOpResultValue,

    // Virtual: memory subsystem
    RamReadValue,
    RamWriteValue,
    RamAddress,
    RamVal,
    RamValFinal,
    HammingWeight,
    RamWa,

    // Virtual: register subsystem
    RdWriteValue,
    Rs1Value,
    Rs2Value,
    RegistersVal,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    Rd,

    // Virtual: instruction lookups
    LookupOutput,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,

    // Virtual: R1CS products
    Product,
    ShouldBranch,
    ShouldJump,

    // Virtual: instruction flags
    IsRdNotZero,
    WriteLookupToRdFlag,
    JumpFlag,
    BranchFlag,
    NoopFlag,

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

    // Virtual: circuit flags
    /// Index matches jolt-instructions CircuitFlags enum order.
    OpFlag(usize),

    // Virtual: bytecode
    ExpandedPc,
    InstructionRafFlag,
    LookupTableFlag(usize),
    BytecodeReadRafVal(usize),
    InstructionReadRafVal(usize),
    /// Per-cycle bytecode table index: pc_indices[t] ∈ {0..K-1}.
    /// Used by EqPushforward to build F[s] tables.
    BytecodePcIndex,
    /// Pushforward eq table for BytecodeReadRaf address phase.
    /// F[stage][k] = Σ_j eq(r_cycle_s, j) × 1{PC(j) == k}.
    BytecodeReadRafF(usize),
    /// One-hot polynomial at bytecode index of cycle 0 (from trace).
    BytecodeEntryTrace,
    /// One-hot polynomial at entry bytecode index (preprocessed).
    BytecodeEntryExpected,
    /// GammaVal[stage] = gamma^stage × (Val[stage] + RAF identity).
    /// Materialized by BytecodeVal InputBinding.
    BytecodeReadRafGammaVal(usize),
    /// gamma^7 × f_entry_expected. Materialized by ScaleByChallenge.
    BytecodeEntryWeighted,

    // Virtual: RAM subsystem
    RamCombinedRa,
    RamRafRa,
    /// Full T×K RAM access indicator (cycle-major layout).
    /// `indicator[t * K + k] = 1` if cycle `t` accesses address `k`, else `0`.
    RamRaIndicator,
    InstructionRafRa,
    BytecodeRafRa,

    // Virtual: Hamming weight reduction
    HammingG(usize),

    // Virtual: Booleanity transposed RA arrays
    /// Cycle-major transpose of ra_d: ra_cm[cycle * K + addr].
    BooleanityG(usize),

    // Virtual: advice address phase
    TrustedAdviceAddr,
    UntrustedAdviceAddr,

    // Virtual: per-cycle gather indices (compact integer vectors)
    /// Per-cycle register destination index rd[j] ∈ {0..K_REG-1}.
    /// Used by EqGather to build wa(j) = eq(r_address, rd[j]).
    RdGatherIndex,
    /// Per-cycle RAM address index addr[j] ∈ {0..K_RAM-1}.
    /// Used by EqGather to build RA(r_address, j) for RAM claim reduction.
    RamGatherIndex,

    // Virtual: instruction lookup materialized outputs
    /// Combined Val+RAF polynomial materialized during the InstructionReadRaf
    /// address→cycle transition. T elements, one per cycle.
    InstructionCombinedVal,

    // Virtual: RamRW eq tables (segmented)
    RamEqCycle,
    RamEqAddr,

    // Virtual: per-instance batched sumcheck eq tables
    BatchEq(usize),

    // Virtual: Spartan internals
    SpartanEq,
    ProductLeft,
    ProductRight,
    OuterUniskipEval,
    ProductUniskipEval,

    // Virtual: Spartan R1CS
    Az,
    Bz,
    Cz,
    CombinedRow,

    // Virtual: address-decomposition instance internals
    /// Suffix polynomial buffer: (table_idx, suffix_idx_within_table).
    InstanceSuffix(usize, usize),
    /// Q (read-RAF quotient) buffer: (component, half).
    /// Component: 0=left, 1=right, 2=identity. Half: 0 or 1.
    InstanceQ(usize, usize),
    /// P (read-RAF product) buffer: (component, half).
    /// Component: 0=left, 1=right, 2=identity. Half: 0 or 1.
    InstanceP(usize, usize),
    /// Expanding eq table for phase p.
    ExpandingTable(usize),
    /// Preprocessed mask buffer for prefix-MLE evaluation.
    /// The index is a compile-time-allocated `(mask_role, b_len)` id.
    /// Each buffer holds `2^b_len` field-encoded small integers indexed by `b`.
    PrefixMask(usize),

    /// Per-bytecode-entry field polynomial (K entries, bytecode-table-indexed).
    /// Flat index into the canonical field extraction order. See
    /// `BytecodeData::populate_preprocessed` for the index→field mapping.
    BytecodeField(usize),
    /// Eq-gathered register polynomial for bytecode stages 3/4.
    /// Index: 0=rd, 1=rs1, 2=rs2.
    BytecodeRegEq(usize),

    // Public: preprocessed
    IoMask,
    ValIo,
    RamUnmap,
    RamInit,
    LookupTable,
    BytecodeTable(usize),
}

/// Index constants for `BytecodeField(idx)` — canonical extraction order.
/// Used by both the compiler (to emit WeightedSum terms) and jolt-witness
/// (to extract per-field polynomials from BytecodeData).
pub mod bytecode_field {
    pub const ADDRESS: usize = 0;
    pub const IMM: usize = 1;
    /// `circuit_flags[f]` → index `CIRCUIT_FLAG_BASE + f` for f in 0..14.
    pub const CIRCUIT_FLAG_BASE: usize = 2;
    pub const IS_BRANCH: usize = 16;
    pub const LEFT_IS_RS1: usize = 17;
    pub const LEFT_IS_PC: usize = 18;
    pub const RIGHT_IS_RS2: usize = 19;
    pub const RIGHT_IS_IMM: usize = 20;
    pub const IS_NOOP: usize = 21;
    /// `!is_interleaved` (i.e., IS a RAF instruction).
    pub const RAF_FLAG: usize = 22;
    /// Register indices for EqGather. Sentinel 255 → eq_gather returns 0.
    pub const RD_INDEX: usize = 23;
    pub const RS1_INDEX: usize = 24;
    pub const RS2_INDEX: usize = 25;
    /// `table_flag[t]` → index `TABLE_FLAG_BASE + t`.
    pub const TABLE_FLAG_BASE: usize = 26;
}

impl PolynomialId {
    /// Returns the operational semantics for this polynomial.
    pub fn descriptor(&self) -> PolynomialDescriptor {
        match self {
            // Committed: trace-derived dense
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

            // Committed: inserted separately (not from trace)
            Self::SpartanWitness
            | Self::TrustedAdvice
            | Self::UntrustedAdvice
            | Self::FieldRegReadValue
            | Self::FieldRegWriteValue => PolynomialDescriptor {
                source: PolySource::Witness,
                committed: true,
                storage: StorageHint::Dense,
                witness_slot: None,
            },

            // FieldReg Inc/Ra/Val/EqCycle: computed by DerivedSource from
            // `FieldRegConfig.events` — no R1CS witness dependence.
            Self::FieldRegInc | Self::FieldRegRa | Self::FieldRegVal | Self::FieldRegEqCycle => {
                PolynomialDescriptor {
                    source: PolySource::Derived,
                    committed: true,
                    storage: StorageHint::Dense,
                    witness_slot: None,
                }
            }

            // Committed: trace-derived one-hot
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

            // R1CS: computed on demand
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

            // Preprocessed: loaded from verifying key
            Self::IoMask
            | Self::ValIo
            | Self::RamUnmap
            | Self::RamInit
            | Self::LookupTable
            | Self::BytecodeTable(_)
            | Self::BytecodePcIndex
            | Self::BytecodeEntryTrace
            | Self::BytecodeEntryExpected
            | Self::BytecodeField(_)
            | Self::PrefixMask(_) => PolynomialDescriptor {
                source: PolySource::Preprocessed,
                committed: false,
                storage: StorageHint::Dense,
                witness_slot: None,
            },

            // R1CS input variables: column slices of witness
            _ if self.r1cs_variable_index().is_some() => PolynomialDescriptor {
                source: PolySource::R1cs(R1csColumn::Variable(self.r1cs_variable_index().unwrap())),
                committed: false,
                storage: StorageHint::Dense,
                witness_slot: None,
            },

            // Virtual / derived: everything else
            _ => PolynomialDescriptor {
                source: PolySource::Derived,
                committed: false,
                storage: StorageHint::Dense,
                witness_slot: None,
            },
        }
    }

    /// Returns the R1CS variable index for polynomials that are columns of
    /// the per-cycle witness z-vector, or `None` for other polynomials.
    pub fn r1cs_variable_index(&self) -> Option<usize> {
        match self {
            Self::LeftInstructionInput => Some(1),
            Self::RightInstructionInput => Some(2),
            Self::Product => Some(3),
            Self::ShouldBranch => Some(4),
            Self::ExpandedPc => Some(5),
            Self::UnexpandedPc => Some(6),
            Self::Imm => Some(7),
            Self::RamAddress => Some(8),
            Self::Rs1Value => Some(9),
            Self::Rs2Value => Some(10),
            Self::RdWriteValue => Some(11),
            Self::RamReadValue => Some(12),
            Self::RamWriteValue => Some(13),
            Self::LeftLookupOperand => Some(14),
            Self::RightLookupOperand => Some(15),
            Self::NextUnexpandedPc => Some(16),
            Self::NextPc => Some(17),
            Self::NextIsVirtual => Some(18),
            Self::NextIsFirstInSequence => Some(19),
            Self::LookupOutput => Some(20),
            Self::ShouldJump => Some(21),
            Self::OpFlag(i) => Some(22 + i),
            Self::FieldOpOperandA => Some(40),
            Self::FieldOpOperandB => Some(41),
            Self::FieldOpResultValue => Some(42),
            _ => None,
        }
    }

    /// Whether this polynomial has a PCS commitment.
    #[inline]
    pub fn is_committed(&self) -> bool {
        self.descriptor().committed
    }
}
