#![expect(
    clippy::panic_in_result_fn,
    reason = "test assertions inside Result-returning tests"
)]
#![expect(clippy::indexing_slicing, reason = "tests index fixture data")]

use super::*;

use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{
    JoltInstructionKind as Kind, JoltInstructionProfile, SourceExtension, SourceInlineKey,
    SourceInstructionRow, RV64IMAC_JOLT,
};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serialization")]
use sha2::{Digest, Sha256};

#[cfg(feature = "serialization")]
#[derive(Debug, Deserialize, Serialize)]
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

fn rows(expansion: SourceExpansion) -> Vec<JoltInstructionRow> {
    expansion
        .into_bytecode()
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
fn csrrs_rd_zero_rs1_zero_becomes_noop_addi() -> Result<(), ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let mut row = source_row(SourceInstructionKind::CSRRS, Some(0), false);
    row.operands.rs1 = Some(0);
    row.operands.imm = 0x300;
    let expanded = rows(expand_instruction(
        &SourceInstruction::new(SourceInstructionKind::CSRRS, row),
        &mut allocator,
        RV64IMAC_JOLT,
    )?);

    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].instruction_kind, JoltInstructionKind::ADDI);
    assert_eq!(expanded[0].operands.rd, Some(0));
    assert_eq!(expanded[0].operands.rs1, Some(0));
    assert_eq!(expanded[0].operands.imm, 0);
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
fn sc_expansions_do_not_use_advice() -> Result<(), ExpansionError> {
    for instruction_kind in [SourceInstructionKind::SCW, SourceInstructionKind::SCD] {
        let mut allocator = ExpansionAllocator::new();
        let expanded = rows(expand_instruction(
            &instruction(instruction_kind, Some(3), false),
            &mut allocator,
            RV64IMAC_JOLT,
        )?);
        assert!(
            expanded
                .iter()
                .all(|row| !matches!(row.instruction_kind, Kind::VirtualAdvice(_))),
            "SC success must be determined by constrained reservation state"
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

fn dual_expansion(input: &SourceInstruction) -> Result<DualExpansion, ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let expansion = expand_instruction(input, &mut allocator, RV64IMAC_JOLT)?;
    assert_eq!(
        allocator.allocate()?,
        40,
        "{:?} leaked a temporary",
        input.kind()
    );
    match expansion {
        SourceExpansion::Dual(dual) => Ok(dual),
        SourceExpansion::Single(_) => Err(ExpansionError::MalformedInstruction(
            "dual load or store expanded once",
        )),
    }
}

#[test]
fn dual_accesses_stamp_each_sequence_on_its_own() -> Result<(), ExpansionError> {
    for (kind, width) in [
        (SourceInstructionKind::LH, DualWidth::Halfword),
        (SourceInstructionKind::LHU, DualWidth::Halfword),
        (SourceInstructionKind::SH, DualWidth::Halfword),
        (SourceInstructionKind::LW, DualWidth::Word),
        (SourceInstructionKind::LWU, DualWidth::Word),
        (SourceInstructionKind::SW, DualWidth::Word),
        (SourceInstructionKind::LD, DualWidth::Doubleword),
        (SourceInstructionKind::SD, DualWidth::Doubleword),
    ] {
        let rd = (!kind.name().starts_with('S')).then_some(3);
        for is_compressed in [false, true] {
            let input = instruction(kind, rd, is_compressed);
            let dual = dual_expansion(&input)?;
            assert_eq!(
                dual.access,
                DualAccess {
                    base: 1,
                    offset: 7,
                    width
                }
            );
            for (sequence, misaligned) in [(dual.aligned, false), (dual.misaligned, true)] {
                let rows: Vec<JoltInstructionRow> =
                    sequence.into_iter().map(JoltInstructionRow::from).collect();
                assert!(rows.iter().all(|row| row.address == input.row().address
                    && row.misaligned_variant == misaligned));
                if let [row] = rows.as_slice() {
                    if row.virtual_sequence_remaining.is_none() {
                        // A native LD/SD row is its own aligned sequence.
                        assert!(!misaligned && !row.is_first_in_sequence);
                        assert_eq!(row.is_compressed, is_compressed);
                        continue;
                    }
                }
                let last = rows.len() - 1;
                for (index, row) in rows.iter().enumerate() {
                    assert_eq!(
                        row.virtual_sequence_remaining,
                        Some((last - index) as u16),
                        "{kind:?}"
                    );
                    assert_eq!(row.is_first_in_sequence, index == 0, "{kind:?}");
                    assert_eq!(
                        row.is_compressed,
                        is_compressed && index == last,
                        "{kind:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn dual_rd_zero_load_writes_one_temporary_in_both_sequences() -> Result<(), ExpansionError> {
    for kind in [
        SourceInstructionKind::LH,
        SourceInstructionKind::LHU,
        SourceInstructionKind::LW,
        SourceInstructionKind::LWU,
        SourceInstructionKind::LD,
    ] {
        let dual = dual_expansion(&instruction(kind, Some(0), false))?;
        for sequence in [dual.aligned, dual.misaligned] {
            let last = sequence.last().copied().map(JoltInstructionRow::from);
            assert_eq!(last.and_then(|row| row.operands.rd), Some(40), "{kind:?}");
        }
    }
    Ok(())
}

/// The word accesses nested in `LR.W`/`SC.W` and the doubleword ones in the
/// `.D` atomics keep their aligned rows, so a misaligned atomic stays
/// unprovable; byte accesses never need a second sequence.
#[test]
fn atomics_and_byte_accesses_expand_once() -> Result<(), ExpansionError> {
    for kind in [
        SourceInstructionKind::LB,
        SourceInstructionKind::LBU,
        SourceInstructionKind::SB,
        SourceInstructionKind::LRW,
        SourceInstructionKind::SCW,
        SourceInstructionKind::LRD,
        SourceInstructionKind::SCD,
        SourceInstructionKind::AMOADDW,
        SourceInstructionKind::AMOSWAPD,
    ] {
        let rd = (kind != SourceInstructionKind::SB).then_some(3);
        let mut allocator = ExpansionAllocator::new();
        let expansion =
            expand_instruction(&instruction(kind, rd, false), &mut allocator, RV64IMAC_JOLT)?;
        assert!(matches!(expansion, SourceExpansion::Single(_)), "{kind:?}");
        assert!(rows(expansion).iter().all(|row| !row.misaligned_variant));
    }
    Ok(())
}

#[test]
#[cfg(feature = "serialization")]
fn expansion_matches_main_golden_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Pins expansion order, virtual-register reuse and emitted tags (kinds
    // serialize as tags) without checking the expanded rows in. Re-baseline with
    // `JOLT_UPDATE_EXPANSION_FIXTURES=1 cargo nextest run -p jolt-program -E
    // 'test(expansion_matches_main_golden_fixture)'` and review which cases moved.
    let cases: Vec<ExpansionParityCase> =
        serde_json::from_str(include_str!("fixtures/main_expand_parity_hashes.json"))?;
    // WARNING: guards against accidental truncation when re-baselining (a
    // filtering rewrite of the fixture must not drop unchanged entries).
    assert_eq!(cases.len(), 380);

    let bless = std::env::var_os("JOLT_UPDATE_EXPANSION_FIXTURES").is_some();
    let mut blessed = Vec::with_capacity(cases.len());
    for case in cases {
        let mut allocator = ExpansionAllocator::new();
        let expanded = rows(expand_instruction(
            &case.input,
            &mut allocator,
            RV64IMAC_JOLT,
        )?);
        let encoded = serde_json::to_vec(&expanded)?;
        let output_sha256 = hex::encode(Sha256::digest(encoded));

        if bless {
            blessed.push(ExpansionParityCase {
                output_sha256,
                ..case
            });
        } else {
            assert_eq!(output_sha256, case.output_sha256, "{}", case.name);
        }
    }
    if bless {
        std::fs::write(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/expand/fixtures/main_expand_parity_hashes.json"
            ),
            serde_json::to_string(&blessed)?,
        )?;
    }

    Ok(())
}
