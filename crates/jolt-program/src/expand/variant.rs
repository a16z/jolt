use jolt_riscv::{JoltInstruction, JoltInstructionKind, JoltInstructionRow};

/// Final bytecode of one source instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceExpansion {
    /// One sequence, run by every execution.
    Single(Vec<JoltInstruction>),
    /// Two sequences at the same address; each execution runs the one its
    /// effective address selects.
    Dual(DualExpansion),
}

impl SourceExpansion {
    /// Rows in bytecode order. A dual expansion lays out its aligned sequence
    /// first, so the address's first PC is the aligned head.
    pub fn into_bytecode(self) -> Vec<JoltInstruction> {
        match self {
            Self::Single(rows) => rows,
            Self::Dual(DualExpansion {
                mut aligned,
                misaligned,
                ..
            }) => {
                aligned.extend(misaligned);
                aligned
            }
        }
    }
}

/// A top-level `LH`/`LHU`/`LW`/`LWU`/`LD`/`SH`/`SW`/`SD`, expanded once for
/// naturally aligned and once for misaligned effective addresses.
///
/// The aligned sequence is the native `LD`/`SD` row or the natural-width
/// floor-addressed rows: a misaligned address leaves low bits in its
/// `RamAddress`, which RAM RAF evaluation rejects. The misaligned sequence is
/// assembled from byte-granular floor rows and is correct at every offset, so
/// no constraint pins which sequence runs; the honest tracer runs the
/// misaligned one exactly when [`DualAccess::is_misaligned`] holds. Every row
/// of `misaligned` carries `misaligned_variant`, which keeps the two
/// sequences' PCs apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualExpansion {
    pub access: DualAccess,
    pub aligned: Vec<JoltInstruction>,
    pub misaligned: Vec<JoltInstruction>,
}

/// The effective address `x[base] + offset` of a dual-expanded load or store
/// and the access width that selects its sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DualAccess {
    pub base: u8,
    pub offset: i128,
    pub width: DualWidth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualWidth {
    Halfword,
    Word,
    Doubleword,
}

impl DualWidth {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Halfword => 2,
            Self::Word => 4,
            Self::Doubleword => 8,
        }
    }
}

impl DualAccess {
    /// Reads the access off the head row of a dual expansion's aligned
    /// sequence: the native `LD`/`SD` row or the natural-width floor access at
    /// `x[rs1] + imm`.
    ///
    /// Expansion builds [`DualExpansion::access`] with this, so a backend that
    /// runs the flat bytecode dispatches exactly as the tracer does.
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "no other kind heads the aligned sequence of a dual expansion"
    )]
    pub fn from_aligned_head(row: &JoltInstructionRow) -> Option<Self> {
        let width = match row.instruction_kind {
            JoltInstructionKind::VirtualLoadH => DualWidth::Halfword,
            JoltInstructionKind::VirtualLoadW => DualWidth::Word,
            JoltInstructionKind::LD | JoltInstructionKind::SD => DualWidth::Doubleword,
            _ => return None,
        };
        Some(Self {
            base: row.operands.rs1?,
            offset: row.operands.imm,
            width,
        })
    }

    /// Whether an execution with `x[base] = base_value` runs the misaligned
    /// sequence: its effective address is not a multiple of the width.
    pub fn is_misaligned(self, base_value: u64) -> bool {
        !base_value
            .wrapping_add(self.offset as u64)
            .is_multiple_of(self.width.bytes())
    }
}
