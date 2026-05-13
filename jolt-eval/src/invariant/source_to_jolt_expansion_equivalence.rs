use arbitrary::{Arbitrary, Unstructured};
use jolt_program::expand::{expand_instruction, ExpansionAllocator};
use jolt_riscv::{JoltRow, SourceInstruction};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::invariant::{CheckError, Invariant, InvariantViolation};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SourceToJoltExpansionInput {
    pub case_index: u32,
}

impl<'a> Arbitrary<'a> for SourceToJoltExpansionInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            case_index: u.arbitrary()?,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpansionParityCase {
    pub name: String,
    pub input: SourceInstruction,
    pub output_sha256: String,
}

fn fixture_cases() -> Result<Vec<ExpansionParityCase>, serde_json::Error> {
    serde_json::from_str(include_str!(
        "../../../crates/jolt-program/src/expand/fixtures/main_expand_parity_hashes.json"
    ))
}

#[jolt_eval_macros::invariant(Test)]
#[derive(Default)]
pub struct SourceToJoltExpansionEquivalenceInvariant;

impl Invariant for SourceToJoltExpansionEquivalenceInvariant {
    type Setup = Result<Vec<ExpansionParityCase>, String>;
    type Input = SourceToJoltExpansionInput;

    fn name(&self) -> &str {
        "source_to_jolt_expansion_equivalence"
    }

    fn description(&self) -> String {
        "Decoded source instructions must expand to the same canonical Jolt rows \
         as the checked baseline parity corpus."
            .to_string()
    }

    fn setup(&self) -> Self::Setup {
        fixture_cases().map_err(|error| error.to_string())
    }

    fn check(
        &self,
        setup: &Self::Setup,
        input: SourceToJoltExpansionInput,
    ) -> Result<(), CheckError> {
        let cases = setup.as_ref().map_err(|error| {
            CheckError::Violation(InvariantViolation::with_details(
                "expansion parity fixture failed to deserialize",
                error,
            ))
        })?;

        if cases.is_empty() {
            return Err(CheckError::InvalidInput(
                "expansion parity corpus is empty".into(),
            ));
        }

        let case = &cases[input.case_index as usize % cases.len()];
        let mut allocator = ExpansionAllocator::new();
        let expanded = expand_instruction(&case.input, &mut allocator).map_err(|error| {
            CheckError::Violation(InvariantViolation::with_details(
                "source expansion failed",
                format!("case={}, error={error}", case.name),
            ))
        })?;
        let expanded_rows: Vec<JoltRow> = expanded.into_iter().map(JoltRow::from).collect();
        let encoded = serde_json::to_vec(&expanded_rows).map_err(|error| {
            CheckError::Violation(InvariantViolation::with_details(
                "expanded row serialization failed",
                format!("case={}, error={error}", case.name),
            ))
        })?;
        let output_sha256 = hex::encode(Sha256::digest(encoded));

        if output_sha256 != case.output_sha256 {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "source expansion parity hash mismatch",
                format!(
                    "case={}, got={}, expected={}",
                    case.name, output_sha256, case.output_sha256
                ),
            )));
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<SourceToJoltExpansionInput> {
        fixture_cases()
            .map(|cases| {
                (0..cases.len() as u32)
                    .map(|case_index| SourceToJoltExpansionInput { case_index })
                    .collect()
            })
            .unwrap_or_else(|_| vec![SourceToJoltExpansionInput { case_index: 0 }])
    }
}
