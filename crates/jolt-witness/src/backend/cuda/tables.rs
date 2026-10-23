//! Instruction-profile metadata for packed device dispatch.
//!
//! Host row accessors supply flags and lookup-table indices for each supported
//! kind/row variant. `PACK_CIRCUIT_ORDER` and `PACK_INSTRUCTION_ORDER` define CUDA
//! bit positions independently of enum ordinals; device extraction combines
//! this metadata with per-cycle operand values.

use super::CUDA_PROFILE;
use jolt_lookup_tables::InstructionLookupTable;
use jolt_riscv::{
    CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction,
    JoltInstructionKind, JoltInstructionRow, NormalizedOperands,
};

use super::descriptors::descriptor;
use super::{FLAG_BIT_CIRCUIT_BASE, FLAG_BIT_INSTRUCTION_BASE, FLAG_BIT_RAF};
use crate::RV64_XLEN;
use crate::{WitnessError, JOLT_VM_LABEL};

pub(crate) const VARIANTS: usize = 12;

pub const PACK_CIRCUIT_ORDER: [CircuitFlags; 14] = [
    CircuitFlags::AddOperands,
    CircuitFlags::SubtractOperands,
    CircuitFlags::MultiplyOperands,
    CircuitFlags::Load,
    CircuitFlags::Store,
    CircuitFlags::Jump,
    CircuitFlags::WriteLookupOutputToRD,
    CircuitFlags::VirtualInstruction,
    CircuitFlags::Assert,
    CircuitFlags::DoNotUpdateUnexpandedPC,
    CircuitFlags::Advice,
    CircuitFlags::IsCompressed,
    CircuitFlags::IsFirstInSequence,
    CircuitFlags::IsLastInSequence,
];

pub const PACK_INSTRUCTION_ORDER: [InstructionFlags; 6] = [
    InstructionFlags::LeftOperandIsPC,
    InstructionFlags::RightOperandIsImm,
    InstructionFlags::LeftOperandIsRs1Value,
    InstructionFlags::RightOperandIsRs2Value,
    InstructionFlags::Branch,
    InstructionFlags::IsNoop,
];

pub const TABLE_INDEX_ABSENT: u32 = u32::MAX;

pub(crate) struct KindTables {
    pub(crate) input: Vec<u8>,
    pub(crate) operand: Vec<u8>,
    pub(crate) output: Vec<u8>,
    pub(crate) index: Vec<u8>,
    pub(crate) flags: Vec<u32>,
    pub(crate) table_index: Vec<u32>,
    pub(crate) count: u32,
    /// Dense-index-keyed: false where [`descriptor`] has no device rule for
    /// the kind. Building the tables stays infallible so a profile containing
    /// an undescribed kind does not disable the backend wholesale; a trace
    /// that actually uses one is rejected by [`Self::require_supported`].
    pub(crate) supported: Vec<bool>,
}

impl KindTables {
    /// Fails closed on the first row whose instruction kind has no device
    /// rule, naming it. Rows whose kind is outside the profile's dense index
    /// cannot be dispatched at all and are rejected the same way.
    pub(crate) fn require_supported(
        &self,
        kinds: impl Iterator<Item = u16>,
    ) -> Result<(), WitnessError> {
        for index in kinds {
            if !self.supported.get(index as usize).copied().unwrap_or(false) {
                let kind = JoltInstructionKind::ALL
                    .iter()
                    .find(|&&kind| {
                        CUDA_PROFILE
                            .jolt_dense_index(kind)
                            .is_some_and(|dense| dense.0 == index)
                    })
                    .map_or_else(
                        || format!("dense index {index}"),
                        |kind| format!("{kind:?}"),
                    );
                return Err(WitnessError::NotServed {
                    oracle: kind,
                    reason: "no CUDA device rule for this instruction kind",
                });
            }
        }
        Ok(())
    }
}

fn variant_row(kind: JoltInstructionKind, variant: usize) -> JoltInstructionRow {
    let sequence = match variant / 4 {
        0 => None,
        1 => Some(0),
        _ => Some(3),
    };
    JoltInstructionRow {
        instruction_kind: kind,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rs1: Some(2),
            rs2: Some(3),
            rd: Some(1),
            imm: 0,
        },
        virtual_sequence_remaining: sequence,
        is_compressed: (variant / 2) % 2 == 1,
        is_first_in_sequence: variant % 2 == 1,
    }
}

pub(crate) fn kind_tables() -> Result<KindTables, WitnessError> {
    let mut by_index: Vec<Option<JoltInstructionKind>> = Vec::new();
    for &kind in JoltInstructionKind::ALL {
        let Some(index) = CUDA_PROFILE.jolt_dense_index(kind) else {
            continue;
        };
        let index = index.0 as usize;
        if by_index.len() <= index {
            by_index.resize(index + 1, None);
        }
        by_index[index] = Some(kind);
    }

    let count = by_index.len();
    let mut tables = KindTables {
        input: vec![0u8; count],
        operand: vec![0u8; count],
        output: vec![0u8; count],
        index: vec![0u8; count],
        flags: vec![0u32; count * VARIANTS],
        table_index: vec![TABLE_INDEX_ABSENT; count],
        count: u32::try_from(count).map_err(|_| WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: "the instruction kind table exceeds a 32-bit index".to_owned(),
        })?,
        supported: vec![false; count],
    };

    for (index, entry) in by_index.iter().enumerate() {
        let Some(kind) = *entry else {
            continue;
        };
        let Some(described) = descriptor(kind) else {
            continue;
        };
        tables.supported[index] = true;
        tables.input[index] = described.input;
        tables.operand[index] = described.operand;
        tables.output[index] = described.output;
        tables.index[index] = described.index;

        let decode = |variant| JoltInstruction::from(variant_row(kind, variant));
        if let Some(table) =
            <JoltInstruction as InstructionLookupTable<RV64_XLEN>>::lookup_table(&decode(0))
        {
            tables.table_index[index] =
                u32::try_from(table.index()).map_err(|_| WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: "a lookup table index exceeds 32 bits".to_owned(),
                })?;
        }

        for variant in 0..VARIANTS {
            let instruction = decode(variant);
            let circuit = instruction.circuit_flags();
            let instruction_flags = instruction.instruction_flags();
            let mut mask = 0u32;
            for (slot, flag) in PACK_CIRCUIT_ORDER.into_iter().enumerate() {
                if circuit[flag] {
                    mask |= 1u32 << (FLAG_BIT_CIRCUIT_BASE + slot as u32);
                }
            }
            for (slot, flag) in PACK_INSTRUCTION_ORDER.into_iter().enumerate() {
                if instruction_flags[flag] {
                    mask |= 1u32 << (FLAG_BIT_INSTRUCTION_BASE + slot as u32);
                }
            }
            if !circuit.is_interleaved_operands() {
                mask |= 1u32 << FLAG_BIT_RAF;
            }
            tables.flags[index * VARIANTS + variant] = mask;
        }
    }

    Ok(tables)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test module")]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn the_kernel_source_agrees_on_the_variant_count() {
        let source = include_str!("kernels/atoms.cu");
        let expected = format!("#define VARIANTS {VARIANTS}");
        assert!(
            source.contains(&expected),
            "the CUDA source must declare `{expected}`",
        );
    }

    /// `require_supported` must gate exactly on whether [`descriptor`] has a
    /// device rule: accepting an undescribed kind would dispatch it through
    /// `atoms.cu`'s zeroed rule slots and silently produce wrong columns,
    /// while rejecting a described one would disable a working instruction.
    #[test]
    fn dispatch_is_gated_exactly_by_descriptor_coverage() {
        let tables = kind_tables().expect("kind tables");
        for &kind in JoltInstructionKind::ALL {
            let Some(dense) = CUDA_PROFILE.jolt_dense_index(kind) else {
                continue;
            };
            let accepted = tables.require_supported([dense.0].into_iter()).is_ok();
            assert_eq!(
                accepted,
                descriptor(kind).is_some(),
                "{kind:?}: dispatch acceptance disagrees with descriptor coverage",
            );
        }
    }

    #[test]
    fn the_kind_table_produces_distinct_flag_masks() {
        let tables = kind_tables().expect("kind tables");
        let distinct: BTreeSet<u32> = tables.flags.iter().copied().collect();
        assert!(
            distinct.len() > 8,
            "the kind table produced only {} distinct flag masks, so the flag bits are barely \
             exercised and a misplaced bit would not show",
            distinct.len(),
        );
        assert!(
            tables
                .table_index
                .iter()
                .any(|&index| index != TABLE_INDEX_ABSENT),
            "no kind targets a lookup table, so the table-index column is untested",
        );
        assert!(
            tables.table_index.contains(&TABLE_INDEX_ABSENT),
            "every kind targets a lookup table, so the absent sentinel is untested",
        );
    }

    #[test]
    fn the_variant_cross_product_changes_the_mask() {
        let tables = kind_tables().expect("kind tables");
        let varies = (0..tables.count as usize).any(|kind| {
            let base = tables.flags[kind * VARIANTS];
            (1..VARIANTS).any(|variant| tables.flags[kind * VARIANTS + variant] != base)
        });
        assert!(
            varies,
            "no kind's mask depends on the variant, so the sequence/compressed bits are ignored",
        );
    }
}
