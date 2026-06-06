use jolt_lookup_tables::{JoltLookupQuery, LookupQuery};
use thiserror::Error;

use crate::execution::TraceRow;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum LookupIndexError {
    #[error("unsupported XLEN for Jolt lookup index derivation: {xlen}")]
    UnsupportedXlen { xlen: usize },
}

pub fn instruction_lookup_index<const XLEN: usize>(
    row: &TraceRow,
) -> Result<u128, LookupIndexError> {
    validate_xlen::<XLEN>()?;
    let query = JoltLookupQuery::new(row.instruction.instruction_kind, row);
    Ok(LookupQuery::<XLEN>::to_lookup_index(&query))
}

const fn validate_xlen<const XLEN: usize>() -> Result<(), LookupIndexError> {
    if XLEN == 0 || XLEN > 64 {
        Err(LookupIndexError::UnsupportedXlen { xlen: XLEN })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use jolt_lookup_tables::interleave_bits;
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};

    use crate::execution::{RegisterRead, RegisterState, RegisterWrite, TraceRow};

    use super::*;

    fn row(instruction_kind: JoltInstructionKind) -> TraceRow {
        TraceRow {
            instruction: JoltInstructionRow {
                instruction_kind,
                address: 0x8000_0000,
                operands: NormalizedOperands {
                    rd: Some(1),
                    rs1: Some(2),
                    rs2: Some(3),
                    imm: -1,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 10,
                }),
                rs2: Some(RegisterRead {
                    register: 3,
                    value: 4,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 0,
                    post_value: 99,
                }),
            },
            ..Default::default()
        }
    }

    fn kind(name: &str) -> JoltInstructionKind {
        let result = JoltInstructionKind::from_name(name);
        assert!(result.is_some(), "unknown Jolt instruction kind: {name}");
        match result {
            Some(kind) => kind,
            None => JoltInstructionKind::NoOp,
        }
    }

    #[test]
    fn addi_lookup_index_uses_lookup_table_semantics() {
        assert_eq!(
            instruction_lookup_index::<64>(&row(JoltInstructionKind::ADDI)),
            Ok((1_u128 << 64) + 9)
        );
    }

    #[test]
    fn jalr_lookup_index_uses_lookup_table_semantics() {
        assert_eq!(
            instruction_lookup_index::<64>(&row(JoltInstructionKind::JALR)),
            Ok((1_u128 << 64) + 9)
        );
    }

    #[test]
    fn default_lookup_index_interleaves_operands() {
        assert_eq!(
            instruction_lookup_index::<64>(&row(JoltInstructionKind::AND)),
            Ok(interleave_bits(10, 4))
        );
    }

    #[test]
    fn advice_lookup_index_uses_written_value() {
        assert_eq!(
            instruction_lookup_index::<64>(&row(kind("VirtualAdviceLoad"))),
            Ok(99)
        );
    }

    #[test]
    fn invalid_xlen_is_rejected() {
        assert_eq!(
            instruction_lookup_index::<0>(&row(JoltInstructionKind::ADDI)),
            Err(LookupIndexError::UnsupportedXlen { xlen: 0 })
        );
    }
}
