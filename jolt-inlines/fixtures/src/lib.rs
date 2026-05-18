extern crate jolt_inlines_bigint as _;
extern crate jolt_inlines_blake2 as _;
extern crate jolt_inlines_blake3 as _;
extern crate jolt_inlines_grumpkin as _;
extern crate jolt_inlines_keccak256 as _;
extern crate jolt_inlines_p256 as _;
extern crate jolt_inlines_secp256k1 as _;
extern crate jolt_inlines_sha2 as _;

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        fs, io,
        path::{Path, PathBuf},
    };

    use jolt_program::expand::{expand_instruction_with_provider, ExpansionAllocator};
    use jolt_riscv::{
        JoltInstructionRow, NormalizedOperands, SourceInlineKey, SourceInstruction,
        SourceInstructionKind, SourceInstructionRow, RV64IMAC_JOLT_ALL_INLINES,
    };
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use tracer::{InlineRegistration, TracerInlineExpansionProvider};

    const FIXTURE_PATH: &str = "fixtures/registered_inline_expand_parity_hashes.json";

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum ExpectedAdmissibility {
        PublicNoRequirements,
        InternalOnly,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct RegisteredInlineExpansionParityCase {
        name: String,
        inline_name: String,
        opcode: u8,
        funct3: u8,
        funct7: u8,
        extension: String,
        expected_admissibility: ExpectedAdmissibility,
        input: SourceInstruction,
        row_count: usize,
        output_sha256: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct RegisteredInlineSummary {
        inline_name: String,
        opcode: u8,
        funct3: u8,
        funct7: u8,
        extension: String,
        expected_admissibility: ExpectedAdmissibility,
    }

    #[derive(Debug, Clone, Copy)]
    struct FixtureScenario {
        name: &'static str,
        address: usize,
        rd: u8,
        rs1: u8,
        rs2: u8,
        is_compressed: bool,
    }

    const SCENARIOS: &[FixtureScenario] = &[
        FixtureScenario {
            name: "normal",
            address: 0x8000_0000,
            rd: 3,
            rs1: 1,
            rs2: 2,
            is_compressed: false,
        },
        FixtureScenario {
            name: "compressed",
            address: 0x8000_0010,
            rd: 3,
            rs1: 1,
            rs2: 2,
            is_compressed: true,
        },
        FixtureScenario {
            name: "rd_zero",
            address: 0x8000_0020,
            rd: 0,
            rs1: 1,
            rs2: 2,
            is_compressed: false,
        },
        FixtureScenario {
            name: "aliased_operands",
            address: 0x8000_0030,
            rd: 5,
            rs1: 5,
            rs2: 5,
            is_compressed: false,
        },
    ];

    fn registrations() -> Vec<&'static InlineRegistration> {
        let mut registrations = inventory::iter::<InlineRegistration>
            .into_iter()
            .collect::<Vec<_>>();
        registrations.sort_by_key(|registration| {
            (
                registration.opcode,
                registration.funct7,
                registration.funct3,
                registration.name,
            )
        });
        registrations
    }

    fn source_instruction(
        registration: &InlineRegistration,
        scenario: FixtureScenario,
    ) -> SourceInstruction {
        SourceInstruction::new(
            SourceInstructionKind::Inline,
            SourceInstructionRow {
                address: scenario.address,
                operands: NormalizedOperands {
                    rd: Some(scenario.rd),
                    rs1: Some(scenario.rs1),
                    rs2: Some(scenario.rs2),
                    imm: 0,
                },
                inline: Some(SourceInlineKey {
                    opcode: registration.opcode as u8,
                    funct3: registration.funct3 as u8,
                    funct7: registration.funct7 as u8,
                }),
                is_compressed: scenario.is_compressed,
            },
        )
    }

    fn compute_cases() -> Result<Vec<RegisteredInlineExpansionParityCase>, Box<dyn Error>> {
        let mut cases = Vec::new();
        for registration in registrations() {
            for scenario in SCENARIOS {
                let input = source_instruction(registration, *scenario);
                let mut provider = TracerInlineExpansionProvider::new();
                let mut allocator = ExpansionAllocator::new();
                let expanded = expand_instruction_with_provider(
                    &input,
                    &mut allocator,
                    &mut provider,
                    RV64IMAC_JOLT_ALL_INLINES,
                )?;
                let rows = expanded
                    .into_iter()
                    .map(JoltInstructionRow::from)
                    .collect::<Vec<_>>();
                let output_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&rows)?));
                cases.push(RegisteredInlineExpansionParityCase {
                    name: format!("{}::{}", registration.name, scenario.name),
                    inline_name: registration.name.to_string(),
                    opcode: registration.opcode as u8,
                    funct3: registration.funct3 as u8,
                    funct7: registration.funct7 as u8,
                    extension: format!("{:?}", registration.extension),
                    expected_admissibility: expected_admissibility(registration.name)?,
                    input,
                    row_count: rows.len(),
                    output_sha256,
                });
            }
        }
        Ok(cases)
    }

    fn expected_admissibility(name: &str) -> Result<ExpectedAdmissibility, Box<dyn Error>> {
        match name {
            "SHA256_INLINE"
            | "SHA256_INIT_INLINE"
            | "BLAKE2_INLINE"
            | "BLAKE3_INLINE"
            | "BLAKE3_KEYED64_INLINE"
            | "KECCAK256_INLINE"
            | "BIGINT256_MUL_INLINE" => Ok(ExpectedAdmissibility::PublicNoRequirements),
            "SECP256K1_MULQ" | "SECP256K1_SQUAREQ" | "SECP256K1_DIVQ" | "SECP256K1_MULR"
            | "SECP256K1_SQUARER" | "SECP256K1_DIVR" | "SECP256K1_GLVR_ADV"
            | "GRUMPKIN_DIVQ_ADV" | "GRUMPKIN_DIVR_ADV" | "P256_MULQ" | "P256_SQUAREQ"
            | "P256_DIVQ" | "P256_MULR" | "P256_SQUARER" | "P256_DIVR" | "P256_FAKE_GLV_ADV" => {
                Ok(ExpectedAdmissibility::InternalOnly)
            }
            other => Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown inline registration {other}"),
            ))),
        }
    }

    fn unique_summaries(
        cases: &[RegisteredInlineExpansionParityCase],
    ) -> Vec<RegisteredInlineSummary> {
        let mut summaries = cases
            .iter()
            .map(|case| RegisteredInlineSummary {
                inline_name: case.inline_name.clone(),
                opcode: case.opcode,
                funct3: case.funct3,
                funct7: case.funct7,
                extension: case.extension.clone(),
                expected_admissibility: case.expected_admissibility,
            })
            .collect::<Vec<_>>();
        summaries.sort_by(|left, right| {
            (
                left.opcode,
                left.funct7,
                left.funct3,
                left.inline_name.as_str(),
            )
                .cmp(&(
                    right.opcode,
                    right.funct7,
                    right.funct3,
                    right.inline_name.as_str(),
                ))
        });
        summaries.dedup();
        summaries
    }

    fn fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH)
    }

    #[test]
    fn registered_inline_expansions_match_golden_fixture() -> Result<(), Box<dyn Error>> {
        let actual = compute_cases()?;

        if std::env::var_os("JOLT_UPDATE_INLINE_EXPANSION_FIXTURES").is_some() {
            fs::write(fixture_path(), serde_json::to_string_pretty(&actual)?)?;
            return Ok(());
        }

        let expected: Vec<RegisteredInlineExpansionParityCase> = serde_json::from_str(
            include_str!("../fixtures/registered_inline_expand_parity_hashes.json"),
        )?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn linked_inline_registration_inventory_matches_fixture() -> Result<(), Box<dyn Error>> {
        let expected_cases: Vec<RegisteredInlineExpansionParityCase> = serde_json::from_str(
            include_str!("../fixtures/registered_inline_expand_parity_hashes.json"),
        )?;
        let expected = unique_summaries(&expected_cases);
        let actual = unique_summaries(&compute_cases()?);

        assert_eq!(actual.len(), 23);
        assert_eq!(actual, expected);
        Ok(())
    }
}
