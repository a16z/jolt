use arbitrary::{Arbitrary, Unstructured};
use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_program::expand::expand_program;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::{InstructionKind, NormalizedInstruction, NormalizedOperands};

use super::{CheckError, Invariant, InvariantViolation};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct BytecodeExpansionInput {
    pub instruction_selectors: Vec<u8>,
    pub compressed_mask: u8,
}

impl<'a> Arbitrary<'a> for BytecodeExpansionInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let len = u.int_in_range(1u8..=8)? as usize;
        let instruction_selectors = (0..len)
            .map(|_| u.arbitrary())
            .collect::<arbitrary::Result<_>>()?;
        Ok(Self {
            instruction_selectors,
            compressed_mask: u.arbitrary()?,
        })
    }
}

#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct BytecodeExpansionInvariant;

impl Invariant for BytecodeExpansionInvariant {
    type Setup = ();
    type Input = BytecodeExpansionInput;

    fn name(&self) -> &str {
        "bytecode_expansion"
    }

    fn description(&self) -> String {
        "jolt-program expansion and program preprocessing must be deterministic, and expanded \
         bytecode rows must map back to bytecode PCs consistently."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: BytecodeExpansionInput) -> Result<(), CheckError> {
        if input.instruction_selectors.is_empty() {
            return Err(CheckError::InvalidInput(
                "instruction_selectors must not be empty".into(),
            ));
        }

        let source = fixture_program(&input);
        let expanded_once = expand_program(source.clone()).map_err(|err| {
            CheckError::Violation(InvariantViolation::with_details(
                "expansion failed",
                err.to_string(),
            ))
        })?;
        let expanded_twice = expand_program(source).map_err(|err| {
            CheckError::Violation(InvariantViolation::with_details(
                "second expansion failed",
                err.to_string(),
            ))
        })?;

        if expanded_once != expanded_twice {
            return Err(CheckError::Violation(InvariantViolation::new(
                "expansion is not deterministic",
            )));
        }

        let preprocessing_once = preprocess(expanded_once.clone())?;
        let preprocessing_twice = preprocess(expanded_twice)?;
        if preprocessing_once != preprocessing_twice {
            return Err(CheckError::Violation(InvariantViolation::new(
                "program preprocessing is not deterministic",
            )));
        }

        if preprocessing_once.bytecode.entry_bytecode_index().is_none() {
            return Err(CheckError::Violation(InvariantViolation::new(
                "entry address is missing from bytecode PC map",
            )));
        }

        for instruction in expanded_once {
            if instruction.instruction_kind == InstructionKind::NoOp {
                continue;
            }
            if preprocessing_once.bytecode.get_pc(&instruction).is_none() {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "expanded instruction is missing from bytecode PC map",
                    format!("{instruction:?}"),
                )));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<BytecodeExpansionInput> {
        vec![
            BytecodeExpansionInput {
                instruction_selectors: vec![0, 1, 2, 3],
                compressed_mask: 0,
            },
            BytecodeExpansionInput {
                instruction_selectors: vec![4, 5, 6, 7, 8],
                compressed_mask: 0b10101,
            },
        ]
    }
}

fn preprocess(
    expanded: Vec<NormalizedInstruction>,
) -> Result<JoltProgramPreprocessing, CheckError> {
    let memory_config = MemoryConfig {
        program_size: Some(0x1000),
        ..MemoryConfig::default()
    };
    JoltProgramPreprocessing::new(
        expanded,
        vec![
            (RAM_START_ADDRESS, 0x13),
            (RAM_START_ADDRESS + 1, 0x01),
            (RAM_START_ADDRESS + 8, 0x37),
        ],
        MemoryLayout::new(&memory_config),
        RAM_START_ADDRESS,
        1 << 12,
    )
    .map_err(|err| {
        CheckError::Violation(InvariantViolation::with_details(
            "program preprocessing failed",
            err.to_string(),
        ))
    })
}

fn fixture_program(input: &BytecodeExpansionInput) -> Vec<NormalizedInstruction> {
    input
        .instruction_selectors
        .iter()
        .enumerate()
        .map(|(index, selector)| {
            let is_compressed = input.compressed_mask & (1 << (index % 8)) != 0;
            let address = RAM_START_ADDRESS as usize + index * 4;
            match selector % 8 {
                0 => instruction(
                    InstructionKind::ADDI,
                    address,
                    Some(1),
                    Some(2),
                    None,
                    3,
                    is_compressed,
                ),
                1 => instruction(
                    InstructionKind::ADDIW,
                    address,
                    Some(3),
                    Some(4),
                    None,
                    -7,
                    is_compressed,
                ),
                2 => instruction(
                    InstructionKind::ADDW,
                    address,
                    Some(5),
                    Some(6),
                    Some(7),
                    0,
                    is_compressed,
                ),
                3 => instruction(
                    InstructionKind::MULH,
                    address,
                    Some(8),
                    Some(9),
                    Some(10),
                    0,
                    is_compressed,
                ),
                4 => instruction(
                    InstructionKind::LB,
                    address,
                    Some(11),
                    Some(12),
                    None,
                    5,
                    is_compressed,
                ),
                5 => instruction(
                    InstructionKind::SW,
                    address,
                    None,
                    Some(13),
                    Some(14),
                    8,
                    is_compressed,
                ),
                6 => instruction(
                    InstructionKind::SLLI,
                    address,
                    Some(15),
                    Some(16),
                    None,
                    4,
                    is_compressed,
                ),
                _ => instruction(
                    InstructionKind::ECALL,
                    address,
                    Some(0),
                    Some(0),
                    Some(0),
                    0,
                    is_compressed,
                ),
            }
        })
        .collect()
}

fn instruction(
    instruction_kind: InstructionKind,
    address: usize,
    rd: Option<u8>,
    rs1: Option<u8>,
    rs2: Option<u8>,
    imm: i128,
    is_compressed: bool,
) -> NormalizedInstruction {
    NormalizedInstruction {
        instruction_kind,
        address,
        operands: NormalizedOperands { rd, rs1, rs2, imm },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed,
    }
}
