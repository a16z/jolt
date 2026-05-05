use super::*;

fn instruction(
    instruction_kind: InstructionKind,
    rd: Option<u8>,
    is_compressed: bool,
) -> NormalizedInstruction {
    NormalizedInstruction {
        instruction_kind,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd,
            rs1: Some(1),
            rs2: Some(2),
            imm: 7,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed,
    }
}

#[test]
fn side_effect_free_rd_zero_becomes_noop_addi() -> Result<(), ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let expanded = expand_instruction(
        &instruction(InstructionKind::ADD, Some(0), true),
        &mut allocator,
    )?;

    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].instruction_kind, InstructionKind::ADDI);
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
    let expanded = expand_instruction(
        &instruction(InstructionKind::JAL, Some(0), false),
        &mut allocator,
    )?;

    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0].instruction_kind, InstructionKind::JAL);
    assert_eq!(expanded[0].operands.rd, Some(40));
    Ok(())
}

#[test]
fn trap_related_rd_zero_uses_instruction_expansion() -> Result<(), ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let input = instruction(InstructionKind::ECALL, Some(0), false);
    let expanded = expand_instruction(&input, &mut allocator)?;

    assert_eq!(expanded.len(), 7);
    assert_eq!(expanded[0].instruction_kind, InstructionKind::AUIPC);
    assert_eq!(expanded[6].instruction_kind, InstructionKind::JALR);
    Ok(())
}

#[test]
fn inline_requires_provider() {
    let mut allocator = ExpansionAllocator::new();
    let input = instruction(InstructionKind::Inline, Some(3), false);

    assert!(matches!(
        expand_instruction(&input, &mut allocator),
        Err(ExpansionError::InlineProviderRequired)
    ));
}

#[test]
fn csr_zero_is_rejected() {
    for instruction_kind in [InstructionKind::CSRRW, InstructionKind::CSRRS] {
        let mut allocator = ExpansionAllocator::new();
        let mut input = instruction(instruction_kind, Some(3), false);
        input.operands.imm = 0;

        assert!(matches!(
            expand_instruction(&input, &mut allocator),
            Err(ExpansionError::UnsupportedCsr(0))
        ));
    }
}

#[test]
fn inline_rd_zero_is_remapped_before_provider() -> Result<(), ExpansionError> {
    #[derive(Default)]
    struct CapturingProvider {
        captured: Option<NormalizedInstruction>,
    }

    impl InlineExpansionProvider for CapturingProvider {
        fn expand_inline(
            &mut self,
            instruction: &NormalizedInstruction,
            _allocator: &mut ExpansionAllocator,
        ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
            self.captured = Some(*instruction);
            Ok(vec![*instruction])
        }
    }

    let input = NormalizedInstruction {
        instruction_kind: InstructionKind::Inline,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(0),
            rs1: Some(10),
            rs2: Some(20),
            imm: 0x0b,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    };
    let mut allocator = ExpansionAllocator::new();
    let mut provider = CapturingProvider::default();

    let expanded = expand_instruction_with_provider(&input, &mut allocator, &mut provider)?;

    let mut expected = input;
    expected.operands.rd = Some(40);

    assert_eq!(provider.captured, Some(expected));
    assert_eq!(expanded, vec![expected]);
    Ok(())
}
