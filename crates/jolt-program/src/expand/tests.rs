#![expect(
    clippy::panic_in_result_fn,
    reason = "test assertions inside Result-returning tests"
)]
#![expect(clippy::indexing_slicing, reason = "tests index fixture data")]

use super::*;

use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{
    JoltInstruction, JoltInstructionKind as Kind, JoltInstructionProfile, SourceExtension,
    SourceInlineKey, SourceInstructionRow, RV64IMAC_JOLT,
};
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
) -> SourceInstructionRow {
    let inline = (instruction_kind == SourceInstructionKind::Inline).then_some(SourceInlineKey {
        opcode: 0x2b,
        funct3: 0,
        funct7: 0,
    });
    SourceInstructionRow {
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

fn rows(instructions: Vec<JoltInstruction>) -> Vec<JoltInstructionRow> {
    instructions
        .into_iter()
        .map(JoltInstructionRow::from)
        .collect()
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
            instruction.instruction_kind
                == JoltInstructionKind::VirtualAdvice(jolt_riscv::instructions::VirtualAdvice(()))
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
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<ExpandedInstructionSequence, ExpansionError> {
            self.captured = Some(*instruction);
            let row = instruction.row();
            let mut builder = InlineExpansionBuilder::new(*row);
            let rd = row.operands.rd.ok_or(ExpansionError::MalformedInstruction(
                "inline row missing rd",
            ))?;
            builder.emit_i(Kind::ADDI, rd, 0, 0);
            builder.finalize()
        }
    }

    let input = SourceInstructionRow {
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(0),
            rs1: Some(10),
            rs2: Some(20),
            imm: 0x0b,
        },
        inline: Some(SourceInlineKey {
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

    let mut expected_row = *input.row();
    expected_row.operands.rd = Some(40);
    let expected = SourceInstruction::new(input.kind(), expected_row);

    assert_eq!(provider.captured, Some(expected));
    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].instruction_kind, JoltInstructionKind::ADDI);
    assert_eq!(expanded[0].operands.rd, Some(40));
    assert_eq!(expanded[0].virtual_sequence_remaining, Some(0));
    assert!(expanded[0].is_first_in_sequence);
    Ok(())
}

#[test]
fn inline_provider_error_releases_rd_zero_temporary() -> Result<(), ExpansionError> {
    struct FailingProvider;

    impl InlineExpansionProvider for FailingProvider {
        fn expand_inline(
            &mut self,
            _instruction: &SourceInstruction,
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<ExpandedInstructionSequence, ExpansionError> {
            Err(ExpansionError::UnsupportedInstruction)
        }
    }

    let input = instruction(SourceInstructionKind::Inline, Some(0), false);
    let mut allocator = ExpansionAllocator::new();
    assert!(matches!(
        expand_instruction_with_provider(
            &input,
            &mut allocator,
            &mut FailingProvider,
            RV64IMAC_JOLT
        ),
        Err(ExpansionError::UnsupportedInstruction)
    ));

    let register = allocator.allocate()?;
    assert_eq!(register, 40);
    allocator.release(register)?;
    Ok(())
}

#[test]
fn inline_provider_output_is_validated_and_stamped() {
    const RV64I_ONLY: JoltInstructionProfile = JoltInstructionProfile {
        source_extensions: &[SourceExtension::Rv64I],
        inline_extensions: &[],
    };

    struct BadProvider;

    impl InlineExpansionProvider for BadProvider {
        fn expand_inline(
            &mut self,
            instruction: &SourceInstruction,
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<ExpandedInstructionSequence, ExpansionError> {
            let mut builder = InlineExpansionBuilder::new(*instruction.row());
            builder.emit_r(Kind::MUL, 1, 2, 3);
            builder.finalize()
        }
    }

    let input = instruction(SourceInstructionKind::Inline, Some(3), true);
    let mut allocator = ExpansionAllocator::new();

    assert!(matches!(
        expand_instruction_with_provider(&input, &mut allocator, &mut BadProvider, RV64I_ONLY),
        Err(ExpansionError::IllegalTargetInstruction(
            JoltInstructionKind::MUL
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
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<ExpandedInstructionSequence, ExpansionError> {
            let row = instruction.row();
            let mut builder = InlineExpansionBuilder::new(*row);
            let register = builder.allocate_for_inline()?;
            builder.emit_i(Kind::ADDI, *register, 0, 1);
            builder.release(register);
            builder.finalize()
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
            _profile: jolt_riscv::JoltInstructionProfile,
        ) -> Result<ExpandedInstructionSequence, ExpansionError> {
            let row = instruction.row();
            let mut builder = InlineExpansionBuilder::new(*row);
            for _ in 0..=materialize::MAX_FINAL_ROWS_PER_SOURCE {
                builder.emit_i(Kind::ADDI, 0, 0, 0);
            }
            builder.finalize()
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
                    SourceInstructionKind::$kind.jolt_kind().is_none(),
                    concat!(stringify!($kind), " has an expander but maps directly to a final row")
                );
            )*
        };
    }

    assert_source_only! {
        MULH, MULHSU,
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
        SLL, SLLI, SLLIW, SLLW, SRL, SRLI, SRA, SRAI,
        SRLIW, SRAIW, SRLW, SRAW,
    }
    assert_eq!(SourceInstructionKind::Inline.jolt_kind(), None);
}

#[test]
fn rv64i_word_shift_expansions_are_profile_closed() -> Result<(), ExpansionError> {
    const RV64I_ONLY: JoltInstructionProfile = JoltInstructionProfile {
        source_extensions: &[SourceExtension::Rv64I],
        inline_extensions: &[],
    };

    for instruction_kind in [
        SourceInstructionKind::SRLW,
        SourceInstructionKind::SRAW,
        SourceInstructionKind::SRLIW,
        SourceInstructionKind::SRAIW,
    ] {
        assert!(RV64I_ONLY.supports_source(instruction_kind));

        let mut allocator = ExpansionAllocator::new();
        let expanded = rows(expand_instruction(
            &instruction(instruction_kind, Some(3), false),
            &mut allocator,
            RV64I_ONLY,
        )?);

        assert!(
            expanded
                .iter()
                .all(|row| RV64I_ONLY.supports_jolt(row.instruction_kind)),
            "{instruction_kind:?} emitted a helper outside its source profile"
        );
    }

    Ok(())
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
    //
    // 16 of the 360 hashes were re-baselined when `emit_address` began wrapping
    // its offset through `format_i_imm`: exactly the `imm = -8` cases for LH/LHU/
    // LW/LWU/SH/SW, the accesses that emit an alignment assert. Byte accesses
    // (no assert) and non-negative offsets (wrap is the identity) are unchanged.
    //
    // 18 hashes were re-baselined when LW moved to the fused
    // VirtualWindowMaskW + VirtualPextSigned extraction: all LW cases, plus
    // LRW/SCW, which recursively embed the word-load expansion. A further 60
    // (LB/LBU/LH/LHU/LWU, 12 each) were re-baselined when the remaining loads
    // moved to their window-mask + parallel-extract sequences.
    // A further 78 (all six loads, 12 each, plus LRW/SCW) were re-baselined
    // when VirtualAlignAddr fused the ADDI + ANDI pair and the window masks
    // began taking the immediate directly, and again when the fused-load
    // virtual opcodes moved to 0x009a-0x009d after the W-shift tags took
    // 0x0091-0x0099 (kinds serialize as tags, so renumbering shifts hashes).
    let cases: Vec<ExpansionParityCase> =
        serde_json::from_str(include_str!("fixtures/main_expand_parity_hashes.json"))?;
    // WARNING: guards against accidental truncation when re-baselining (a
    // filtering rewrite of the fixture must not drop unchanged entries).
    assert_eq!(cases.len(), 360);

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
