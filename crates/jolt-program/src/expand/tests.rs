use super::*;

use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{JoltInstruction, SourceInline, SourceRow, RV64IMAC_JOLT};
#[cfg(feature = "serialization")]
use serde::Deserialize;
#[cfg(feature = "serialization")]
use sha2::{Digest, Sha256};

#[cfg(feature = "serialization")]
#[derive(Debug, Deserialize)]
struct ExpansionParityCase {
    name: String,
    input: SourceInstruction,
    output_sha256: String,
}

fn source_row(
    instruction_kind: SourceInstructionKind,
    rd: Option<u8>,
    is_compressed: bool,
) -> SourceRow {
    let inline = (instruction_kind == SourceInstructionKind::Inline).then_some(SourceInline {
        opcode: 0x2b,
        funct3: 0,
        funct7: 0,
    });
    SourceRow {
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd,
            rs1: Some(1),
            rs2: Some(2),
            imm: 7,
        },
        inline,
        is_compressed,
    }
}

fn instruction(
    instruction_kind: SourceInstructionKind,
    rd: Option<u8>,
    is_compressed: bool,
) -> SourceInstruction {
    SourceInstruction::new(
        instruction_kind,
        source_row(instruction_kind, rd, is_compressed),
    )
}

fn final_instruction(row: JoltRow) -> Result<JoltInstruction, ExpansionError> {
    JoltInstruction::try_from(row).map_err(ExpansionError::IllegalTargetInstruction)
}

fn rows(instructions: Vec<JoltInstruction>) -> Vec<JoltRow> {
    instructions.into_iter().map(JoltRow::from).collect()
}

#[test]
fn side_effect_free_rd_zero_becomes_noop_addi() -> Result<(), ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let expanded = rows(expand_instruction(
        &instruction(SourceInstructionKind::ADD, Some(0), true),
        &mut allocator,
        RV64IMAC_JOLT,
    )?);

    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].instruction_kind, JoltInstructionKind::ADDI);
    assert_eq!(expanded[0].operands.rd, Some(0));
    assert_eq!(expanded[0].operands.rs1, Some(0));
    assert_eq!(expanded[0].operands.rs2, None);
    assert_eq!(expanded[0].operands.imm, 0);
    assert!(expanded[0].is_compressed);
    Ok(())
}

#[test]
fn side_effecting_rd_zero_rewrites_to_temporary_register() -> Result<(), ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let expanded = rows(expand_instruction(
        &instruction(SourceInstructionKind::JAL, Some(0), false),
        &mut allocator,
        RV64IMAC_JOLT,
    )?);

    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].instruction_kind, JoltInstructionKind::JAL);
    assert_eq!(expanded[0].operands.rd, Some(40));
    Ok(())
}

#[test]
fn trap_related_rd_zero_uses_instruction_expansion() -> Result<(), ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let input = instruction(SourceInstructionKind::ECALL, Some(0), false);
    let expanded = rows(expand_instruction(&input, &mut allocator, RV64IMAC_JOLT)?);

    assert_eq!(expanded.len(), 7);
    assert_eq!(expanded[0].instruction_kind, JoltInstructionKind::AUIPC);
    assert_eq!(expanded[6].instruction_kind, JoltInstructionKind::JALR);
    Ok(())
}

#[test]
fn inline_requires_provider() {
    let mut allocator = ExpansionAllocator::new();
    let input = instruction(SourceInstructionKind::Inline, Some(3), false);

    assert!(matches!(
        expand_instruction(&input, &mut allocator, RV64IMAC_JOLT),
        Err(ExpansionError::InlineProviderRequired)
    ));
}

#[test]
fn csr_zero_is_rejected() {
    for instruction_kind in [SourceInstructionKind::CSRRW, SourceInstructionKind::CSRRS] {
        let mut allocator = ExpansionAllocator::new();
        let mut input = source_row(instruction_kind, Some(3), false);
        input.operands.imm = 0;
        let input = SourceInstruction::new(instruction_kind, input);

        assert!(matches!(
            expand_instruction(&input, &mut allocator, RV64IMAC_JOLT),
            Err(ExpansionError::UnsupportedCsr(0))
        ));
    }
}

#[test]
fn lr_sc_expansions_restrict_address_to_ram() -> Result<(), ExpansionError> {
    for instruction_kind in [
        SourceInstructionKind::LRW,
        SourceInstructionKind::LRD,
        SourceInstructionKind::SCW,
        SourceInstructionKind::SCD,
    ] {
        let mut allocator = ExpansionAllocator::new();
        let expanded = rows(expand_instruction(
            &instruction(instruction_kind, Some(3), false),
            &mut allocator,
            RV64IMAC_JOLT,
        )?);

        assert_eq!(expanded[0].instruction_kind, JoltInstructionKind::LUI);
        assert_eq!(expanded[0].operands.rd, Some(40));
        assert_eq!(expanded[0].operands.imm, RAM_START_ADDRESS as i128);
        assert_eq!(
            expanded[1].instruction_kind,
            JoltInstructionKind::VirtualAssertLTE
        );
        assert_eq!(expanded[1].operands.rs1, Some(40));
        assert_eq!(expanded[1].operands.rs2, Some(1));
    }
    Ok(())
}

#[test]
fn sc_success_advice_is_not_position_dependent() -> Result<(), ExpansionError> {
    for instruction_kind in [SourceInstructionKind::SCW, SourceInstructionKind::SCD] {
        let mut allocator = ExpansionAllocator::new();
        let expanded = rows(expand_instruction(
            &instruction(instruction_kind, Some(3), false),
            &mut allocator,
            RV64IMAC_JOLT,
        )?);
        let advice_position = expanded.iter().position(|instruction| {
            instruction.instruction_kind == JoltInstructionKind::VirtualAdvice
        });

        assert!(
            matches!(advice_position, Some(position) if position > 1),
            "RAM-region prelude should precede success advice, got {advice_position:?}"
        );
    }
    Ok(())
}

#[test]
fn inline_rd_zero_is_remapped_before_provider() -> Result<(), ExpansionError> {
    #[derive(Default)]
    struct CapturingProvider {
        captured: Option<SourceInstruction>,
    }

    impl InlineExpansionProvider for CapturingProvider {
        fn expand_inline(
            &mut self,
            instruction: &SourceInstruction,
            _allocator: &mut ExpansionAllocator,
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<Vec<JoltInstruction>, ExpansionError> {
            self.captured = Some(*instruction);
            let row = instruction.row();
            Ok(vec![final_instruction(JoltRow {
                instruction_kind: JoltInstructionKind::ADDI,
                address: row.address,
                operands: NormalizedOperands {
                    rd: row.operands.rd,
                    rs1: Some(0),
                    rs2: None,
                    imm: 0,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            })?])
        }
    }

    let input = SourceRow {
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(0),
            rs1: Some(10),
            rs2: Some(20),
            imm: 0x0b,
        },
        inline: Some(SourceInline {
            opcode: 0x2b,
            funct3: 0,
            funct7: 0,
        }),
        is_compressed: false,
    };
    let mut allocator = ExpansionAllocator::new();
    let mut provider = CapturingProvider::default();
    let input = SourceInstruction::new(SourceInstructionKind::Inline, input);

    let expanded = rows(expand_instruction_with_provider(
        &input,
        &mut allocator,
        &mut provider,
        RV64IMAC_JOLT,
    )?);

    let expected = input.map_row(|mut row| {
        row.operands.rd = Some(40);
        row
    });

    assert_eq!(provider.captured, Some(expected));
    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].instruction_kind, JoltInstructionKind::ADDI);
    assert_eq!(expanded[0].operands.rd, Some(40));
    assert_eq!(expanded[0].virtual_sequence_remaining, Some(0));
    assert!(expanded[0].is_first_in_sequence);
    Ok(())
}

#[test]
fn inline_provider_output_is_validated_and_stamped() {
    struct BadProvider;

    impl InlineExpansionProvider for BadProvider {
        fn expand_inline(
            &mut self,
            instruction: &SourceInstruction,
            _allocator: &mut ExpansionAllocator,
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<Vec<JoltInstruction>, ExpansionError> {
            final_instruction(instruction.jolt_row()).map(|instruction| vec![instruction])
        }
    }

    let input = instruction(SourceInstructionKind::Inline, Some(3), true);
    let mut allocator = ExpansionAllocator::new();

    assert!(matches!(
        expand_instruction_with_provider(&input, &mut allocator, &mut BadProvider, RV64IMAC_JOLT),
        Err(ExpansionError::IllegalTargetInstruction(
            JoltInstructionKind::Inline
        ))
    ));
}

#[test]
fn inline_provider_allocator_resets_are_appended() -> Result<(), ExpansionError> {
    struct AllocatingProvider;

    impl InlineExpansionProvider for AllocatingProvider {
        fn expand_inline(
            &mut self,
            instruction: &SourceInstruction,
            allocator: &mut ExpansionAllocator,
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<Vec<JoltInstruction>, ExpansionError> {
            let row = instruction.row();
            let register = allocator.allocate_for_inline()?;
            allocator.release(register)?;
            Ok(vec![final_instruction(JoltRow {
                instruction_kind: JoltInstructionKind::ADDI,
                address: row.address,
                operands: NormalizedOperands {
                    rd: Some(register),
                    rs1: Some(0),
                    rs2: None,
                    imm: 1,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            })?])
        }
    }

    let input = instruction(SourceInstructionKind::Inline, Some(3), true);
    let mut allocator = ExpansionAllocator::new();
    let expanded = rows(expand_instruction_with_provider(
        &input,
        &mut allocator,
        &mut AllocatingProvider,
        RV64IMAC_JOLT,
    )?);

    assert_eq!(expanded.len(), 2);
    assert_eq!(expanded[0].virtual_sequence_remaining, Some(1));
    assert!(expanded[0].is_first_in_sequence);
    assert!(!expanded[0].is_compressed);
    assert_eq!(expanded[1].instruction_kind, JoltInstructionKind::ADDI);
    assert_eq!(expanded[1].operands.rs1, Some(0));
    assert_eq!(expanded[1].operands.imm, 0);
    assert_eq!(expanded[1].virtual_sequence_remaining, Some(0));
    assert!(expanded[1].is_compressed);
    Ok(())
}

#[test]
fn inline_provider_allows_sequences_larger_than_instruction_recipes() -> Result<(), ExpansionError>
{
    struct LargeProvider;

    impl InlineExpansionProvider for LargeProvider {
        fn expand_inline(
            &mut self,
            instruction: &SourceInstruction,
            _allocator: &mut ExpansionAllocator,
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<Vec<JoltInstruction>, ExpansionError> {
            let row = instruction.row();
            (0..=materialize::MAX_FINAL_ROWS_PER_SOURCE)
                .map(|_| {
                    final_instruction(JoltRow {
                        instruction_kind: JoltInstructionKind::ADDI,
                        address: row.address,
                        operands: NormalizedOperands {
                            rd: Some(0),
                            rs1: Some(0),
                            rs2: None,
                            imm: 0,
                        },
                        virtual_sequence_remaining: None,
                        is_first_in_sequence: false,
                        is_compressed: false,
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        }
    }

    let input = instruction(SourceInstructionKind::Inline, Some(3), true);
    let mut allocator = ExpansionAllocator::new();
    let expanded = rows(expand_instruction_with_provider(
        &input,
        &mut allocator,
        &mut LargeProvider,
        RV64IMAC_JOLT,
    )?);

    assert_eq!(expanded.len(), materialize::MAX_FINAL_ROWS_PER_SOURCE + 1);
    assert_eq!(
        expanded[0].virtual_sequence_remaining,
        Some(materialize::MAX_FINAL_ROWS_PER_SOURCE as u16)
    );
    assert!(expanded[0].is_first_in_sequence);
    assert!(expanded[materialize::MAX_FINAL_ROWS_PER_SOURCE].is_compressed);
    Ok(())
}

#[test]
fn source_only_expanders_are_not_target_legal() {
    macro_rules! assert_source_only {
        ($($kind:ident),* $(,)?) => {
            $(
                assert!(
                    !RV64IMAC_JOLT.supports_jolt(JoltInstructionKind::$kind),
                    concat!(stringify!($kind), " has an expander but is target-legal")
                );
            )*
        };
    }

    assert_source_only! {
        ADDIW, ADDW, SUBW, MULH, MULHSU, MULW,
        LB, LBU, LH, LHU, LW, LWU,
        AdviceLB, AdviceLH, AdviceLW, AdviceLD,
        AMOADDD, AMOANDD, AMOORD, AMOXORD, AMOSWAPD,
        AMOMAXD, AMOMAXUD, AMOMIND, AMOMINUD,
        AMOADDW, AMOANDW, AMOORW, AMOXORW, AMOSWAPW,
        AMOMAXW, AMOMAXUW, AMOMINW, AMOMINUW,
        LRD, LRW,
        DIV, DIVU, DIVW, DIVUW, REM, REMU, REMW, REMUW,
        SB, SCD, SCW, SH, SW,
        CSRRW, CSRRS, EBREAK, ECALL, MRET,
        SLL, SLLI, SLLW, SLLIW, SRL, SRLI, SRA, SRAI,
        SRLIW, SRAIW, SRLW, SRAW,
    }
    assert!(!RV64IMAC_JOLT.supports_jolt(JoltInstructionKind::Inline));
}

#[test]
fn recursive_helper_expansion_is_stamped_as_one_sequence() -> Result<(), ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let input = instruction(SourceInstructionKind::SLL, Some(3), true);
    let expanded = rows(expand_instruction(&input, &mut allocator, RV64IMAC_JOLT)?);

    assert!(expanded.len() > 1);
    for (i, row) in expanded.iter().enumerate() {
        assert_eq!(row.address, input.row().address);
        assert_eq!(
            row.virtual_sequence_remaining,
            Some((expanded.len() - i - 1) as u16)
        );
        assert_eq!(row.is_first_in_sequence, i == 0);
        assert_eq!(row.is_compressed, i + 1 == expanded.len());
    }
    assert!(expanded
        .iter()
        .all(|row| RV64IMAC_JOLT.supports_jolt(row.instruction_kind)));

    Ok(())
}

#[test]
#[cfg(feature = "serialization")]
fn expansion_matches_main_golden_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Expected hashes generated from baseline main commit 51d81a36e. This catches
    // recursive expansion order and virtual-register reuse regressions without
    // checking a giant expanded-row fixture into the repository.
    let cases: Vec<ExpansionParityCase> =
        serde_json::from_str(include_str!("fixtures/main_expand_parity_hashes.json"))?;

    for case in cases {
        let mut allocator = ExpansionAllocator::new();
        let expanded = rows(expand_instruction(
            &case.input,
            &mut allocator,
            RV64IMAC_JOLT,
        )?);
        let encoded = serde_json::to_vec(&expanded)?;
        let output_sha256 = hex::encode(Sha256::digest(encoded));

        assert_eq!(output_sha256, case.output_sha256, "{}", case.name);
    }

    Ok(())
}
