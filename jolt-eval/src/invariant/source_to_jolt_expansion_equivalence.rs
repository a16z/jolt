use arbitrary::{Arbitrary, Unstructured};
use jolt_program::expand::{expand_instruction, ExpansionAllocator, SourceExpansion};
use jolt_riscv::{JoltInstruction, JoltInstructionRow, SourceInstruction, RV64IMAC_JOLT};
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
    /// The single sequence, or a dual expansion's aligned one.
    pub output_sha256: String,
    /// A dual expansion's misaligned sequence (only `zicclsm` builds expand one).
    #[serde(default)]
    pub misaligned_sha256: Option<String>,
}

/// SHA-256 of `sequence`'s canonical rows, as the corpus pins them.
fn sequence_sha256(
    case: &ExpansionParityCase,
    sequence: Vec<JoltInstruction>,
) -> Result<String, CheckError> {
    let rows: Vec<JoltInstructionRow> =
        sequence.into_iter().map(JoltInstructionRow::from).collect();
    let encoded = serde_json::to_vec(&rows).map_err(|error| {
        CheckError::Violation(InvariantViolation::with_details(
            "expanded row serialization failed",
            format!("case={}, error={error}", case.name),
        ))
    })?;
    Ok(hex::encode(Sha256::digest(encoded)))
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
        "Decoded source instructions must expand to the same canonical Jolt rows as \
         the checked baseline parity corpus: the single sequence, or both sequences \
         of a dual load or store (zicclsm builds only)."
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
        let expanded =
            expand_instruction(&case.input, &mut allocator, RV64IMAC_JOLT).map_err(|error| {
                CheckError::Violation(InvariantViolation::with_details(
                    "source expansion failed",
                    format!("case={}, error={error}", case.name),
                ))
            })?;
        let (sequence, misaligned) = match expanded {
            SourceExpansion::Single(sequence) => (sequence, None),
            SourceExpansion::Dual(dual) => (dual.aligned, Some(dual.misaligned)),
        };
        let mismatch = |field: &str, got: String, expected: Option<&str>| {
            CheckError::Violation(InvariantViolation::with_details(
                "source expansion parity hash mismatch",
                format!(
                    "case={}, field={field}, got={got}, expected={}",
                    case.name,
                    expected.unwrap_or("<unpinned>")
                ),
            ))
        };
        let output_sha256 = sequence_sha256(case, sequence)?;
        if output_sha256 != case.output_sha256 {
            return Err(mismatch(
                "output_sha256",
                output_sha256,
                Some(case.output_sha256.as_str()),
            ));
        }
        // Only `zicclsm` builds expand a misaligned sequence to check.
        if let Some(misaligned) = misaligned {
            let misaligned_sha256 = sequence_sha256(case, misaligned)?;
            if case.misaligned_sha256.as_deref() != Some(misaligned_sha256.as_str()) {
                return Err(mismatch(
                    "misaligned_sha256",
                    misaligned_sha256,
                    case.misaligned_sha256.as_deref(),
                ));
            }
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
