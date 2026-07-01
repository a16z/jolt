use super::*;

use common::constants::{RAM_START_ADDRESS, RISCV_REGISTER_COUNT};
use jolt_riscv::{
    JoltInstruction, JoltInstructionProfile, SourceExtension, SourceInlineKey,
    SourceInstructionRow, RV64IMAC_JOLT,
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

// First step of the source-to-source transpiler: prove we can pull the symbolic
// expansion recipe (before register allocation / recursion) straight out of the
// expander for a single instruction. We expand SRL, print its ops, and check the
// two native instructions it emits (VirtualShiftRightBitmask then VirtualSRL).
#[test]
fn dump_srl_recipe() -> Result<(), ExpansionError> {
    // Build one SRL source instruction. rs1=1, rs2=2, imm=7 come from the helper;
    // rd=3 is passed here. None of these values matter for SRL's shape.
    // The recipe op enum is private to the expand module; pull it in here.
    use super::grammar::ExpansionOp;

    let input = instruction(SourceInstructionKind::SRL, Some(3), false);
    let recipe = expand_source_only_instruction(&input)?;

    // Print the raw recipe so we can eyeball it with `--nocapture`.
    for op in &recipe.ops {
        println!("{op:?}");
    }

    // Pull out just the native instructions this recipe emits, in order.
    let emitted: Vec<JoltInstructionKind> = recipe
        .ops
        .iter()
        .filter_map(|op| match op {
            ExpansionOp::Emit(row) => Some(row.instruction_kind),
            _ => None,
        })
        .collect();

    assert_eq!(
        emitted,
        vec![
            JoltInstructionKind::VirtualShiftRightBitmask(Default::default()),
            JoltInstructionKind::VirtualSRL,
        ]
    );
    Ok(())
}

// Print every op of one instruction's raw recipe so we can read the structure by
// hand. In an op: Emit = a native instruction that stays as-is, Expand = a
// non-native helper that gets expanded again later, Allocate/Release = scratch
// temps. rd=5 here; rs1=1, rs2=2, imm=7 come from the helper.
fn dump_recipe(kind: SourceInstructionKind) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let input = instruction(kind, Some(5), false);
    let recipe = expand_source_only_instruction(&input)?;
    println!("--- recipe for {kind:?} ---");
    for op in &recipe.ops {
        println!("{op:?}");
    }
    Ok(recipe)
}

// Load word. Unlike SRL, this recipe both Emits natives and Expands other
// non-native helpers (e.g. SRL), so the printout shows nested expansion.
#[test]
fn dump_lw_recipe() -> Result<(), ExpansionError> {
    let recipe = dump_recipe(SourceInstructionKind::LW)?;
    assert!(!recipe.ops.is_empty());
    Ok(())
}

// Signed division: a Class A recipe (the DIV/DIVW variants share one expander).
#[test]
fn dump_div_recipe() -> Result<(), ExpansionError> {
    let recipe = dump_recipe(SourceInstructionKind::DIV)?;
    assert!(!recipe.ops.is_empty());
    Ok(())
}

// Atomic add-word-doubleword: an atomic read-modify-write, one of the larger
// recipes, useful for seeing allocate/release and nested expansion together.
#[test]
fn dump_amoaddd_recipe() -> Result<(), ExpansionError> {
    let recipe = dump_recipe(SourceInstructionKind::AMOADDD)?;
    assert!(!recipe.ops.is_empty());
    Ok(())
}

// Relabel a concrete register number back to a symbol so the dump is generic
// rather than a single sample: 0 is x0, our sentinels 1/2/3 are the source
// operands rd/rs1/rs2, anything >= 40 is a virtual register (Lean's inlineTmp0 is
// v40), and anything else is a plain architectural register.
fn label(reg: u8) -> String {
    match reg {
        0 => "x0".to_string(),
        1 => "rd".to_string(),
        2 => "rs1".to_string(),
        3 => "rs2".to_string(),
        v if v >= RISCV_REGISTER_COUNT => format!("v{v}"),
        x => format!("x{x}"),
    }
}

// Format one fully-expanded row with symbolic operands.
fn fmt_row(row: &JoltInstructionRow) -> String {
    let ops = &row.operands;
    let mut parts = Vec::new();
    if let Some(rd) = ops.rd {
        parts.push(format!("rd={}", label(rd)));
    }
    if let Some(rs1) = ops.rs1 {
        parts.push(format!("rs1={}", label(rs1)));
    }
    if let Some(rs2) = ops.rs2 {
        parts.push(format!("rs2={}", label(rs2)));
    }
    // Always show the immediate, including 0, so it matches the Lean rows
    // (e.g. `ADDI x0, x0, 0`). Print it as signed 64-bit so masks like -8 show as
    // -8 rather than their unsigned 64-bit pattern.
    parts.push(format!("imm={}", ops.imm as i64));
    // Use the clean instruction name (e.g. "Addi") instead of the nested Debug
    // form `Addi(Addi(()))`.
    format!("{}  {}", row.instruction_kind.name(), parts.join(", "))
}

// Symbolic dump of a full expansion. We feed distinct sentinel registers
// (rd=1, rs1=2, rs2=3) through the real expansion path (which includes the
// rd==x0 handling in materialize), then relabel the numbers back to symbols. We
// run both rd==x0 and rd!=x0 so the branch shows up explicitly, the way the Lean
// `pureWritebackTraceProgram rd` wrapper does.
fn dump_symbolic(kind: SourceInstructionKind) -> Result<(), ExpansionError> {
    // Clean instruction name in the header, not the nested Debug form.
    println!("--- {} (symbolic) ---", kind.name());
    for (arm, rd) in [("rd == x0", 0u8), ("rd != x0", 1u8)] {
        let mut allocator = ExpansionAllocator::new();
        let mut row = source_row(kind, Some(rd), false);
        row.operands.rs1 = Some(2);
        row.operands.rs2 = Some(3);
        let input = SourceInstruction::new(kind, row);
        let expanded = rows(expand_instruction(&input, &mut allocator, RV64IMAC_JOLT)?);
        println!("{arm}:");
        for r in &expanded {
            println!("  {}", fmt_row(r));
        }
    }
    Ok(())
}

#[test]
fn dump_srl_symbolic() -> Result<(), ExpansionError> {
    dump_symbolic(SourceInstructionKind::SRL)
}

// Same generic dumper pointed at other instructions — no per-instruction code.
#[test]
fn dump_lw_symbolic() -> Result<(), ExpansionError> {
    dump_symbolic(SourceInstructionKind::LW)
}

#[test]
fn dump_div_symbolic() -> Result<(), ExpansionError> {
    dump_symbolic(SourceInstructionKind::DIV)
}

#[test]
fn dump_amoaddd_symbolic() -> Result<(), ExpansionError> {
    dump_symbolic(SourceInstructionKind::AMOADDD)
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
            builder.emit_i::<jolt_riscv::instructions::Addi>(rd, 0, 0);
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
            builder.emit_r::<jolt_riscv::instructions::Mul>(1, 2, 3);
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
            builder.emit_i::<jolt_riscv::instructions::Addi>(*register, 0, 1);
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
                builder.emit_i::<jolt_riscv::instructions::Addi>(0, 0, 0);
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
    assert_eq!(SourceInstructionKind::Inline.jolt_kind(), None);
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
