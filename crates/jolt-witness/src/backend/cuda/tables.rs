use jolt_lookup_tables::InstructionLookupTable;
use jolt_riscv::{
    CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction,
    JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT,
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
        let Some(index) = RV64IMAC_JOLT.jolt_dense_index(kind) else {
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
    };

    for (index, entry) in by_index.iter().enumerate() {
        let Some(kind) = *entry else {
            continue;
        };
        let Some(described) = descriptor(kind) else {
            return Err(WitnessError::NotServed {
                oracle: format!("{kind:?}"),
                reason: "no device descriptor for this instruction kind",
            });
        };
        tables.input[index] = described.input;
        tables.operand[index] = described.operand;
        tables.output[index] = described.output;
        tables.index[index] = described.index;

        let decode = |variant: usize| {
            JoltInstruction::try_from(variant_row(kind, variant)).map_err(|_| {
                WitnessError::NotServed {
                    oracle: format!("{kind:?}"),
                    reason: "the instruction kind does not decode",
                }
            })
        };
        if let Some(table) =
            <JoltInstruction as InstructionLookupTable<RV64_XLEN>>::lookup_table(&decode(0)?)
        {
            tables.table_index[index] =
                u32::try_from(table.index()).map_err(|_| WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: "a lookup table index exceeds 32 bits".to_owned(),
                })?;
        }

        for variant in 0..VARIANTS {
            let instruction = decode(variant)?;
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

    #[test]
    fn the_kernel_source_agrees_on_the_variant_count() {
        let source = include_str!("kernels/atoms.cu");
        let expected = format!("#define VARIANTS {VARIANTS}");
        assert!(
            source.contains(&expected),
            "the CUDA source must declare `{expected}`",
        );
    }

    #[test]
    fn every_supported_kind_has_a_descriptor_and_distinct_flags() {
        let tables = kind_tables().expect("every supported kind must be describable");
        let distinct: std::collections::BTreeSet<u32> = tables.flags.iter().copied().collect();
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
