use std::collections::{hash_map::Entry, HashMap};

use jolt_backends::CommitmentResult;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_openings::CommitmentScheme;
#[cfg(feature = "field-inline")]
use jolt_verifier::proof::{FieldInlineCommitments, FieldRegistersCommitments};
use jolt_verifier::proof::{JoltCommitments, JoltRaCommitments};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, OracleKind};

use crate::ProverError;

use super::input::CommitmentStageConfig;

type CommitmentOutput<PCS> = (
    <PCS as jolt_crypto::Commitment>::Output,
    <PCS as CommitmentScheme>::OpeningHint,
    usize,
);

#[derive(Clone)]
pub struct CommitmentStageOutput<PCS: CommitmentScheme> {
    pub commitments: JoltCommitments<PCS::Output>,
    pub trusted_advice_commitment: Option<PCS::Output>,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub opening_hints: HashMap<JoltCommittedPolynomial, PCS::OpeningHint>,
    #[cfg(feature = "field-inline")]
    pub field_inline_opening_hints: HashMap<FieldInlineCommittedPolynomial, PCS::OpeningHint>,
}

impl<PCS: CommitmentScheme> CommitmentStageOutput<PCS> {
    #[cfg(not(feature = "field-inline"))]
    pub fn from_backend_result(
        result: CommitmentResult<JoltVmNamespace, PCS>,
        config: CommitmentStageConfig,
    ) -> Result<Self, ProverError> {
        let mut outputs = collect_jolt_commitment_outputs(result)?;
        let base = build_base_jolt_commitments::<PCS>(&mut outputs, config)?;
        Ok(Self {
            commitments: JoltCommitments::new(base.rd_inc, base.ram_inc, base.ra),
            trusted_advice_commitment: base.trusted_advice_commitment,
            untrusted_advice_commitment: base.untrusted_advice_commitment,
            opening_hints: base.opening_hints,
        })
    }

    #[cfg(feature = "field-inline")]
    pub fn from_backend_result(
        result: CommitmentResult<JoltVmNamespace, PCS>,
        config: CommitmentStageConfig,
        field_inline_outputs: Vec<FieldInlineCommittedPolynomialOutput<PCS>>,
    ) -> Result<Self, ProverError> {
        let mut outputs = collect_jolt_commitment_outputs(result)?;
        let field_inline = collect_field_inline_commitment_outputs(field_inline_outputs)?;
        Self::from_backend_result_with_field_inline(&mut outputs, config, field_inline)
    }

    pub fn trusted_advice_commitment(&self) -> Option<&PCS::Output> {
        self.trusted_advice_commitment.as_ref()
    }

    pub fn untrusted_advice_commitment(&self) -> Option<&PCS::Output> {
        self.untrusted_advice_commitment.as_ref()
    }

    #[cfg(feature = "field-inline")]
    fn from_backend_result_with_field_inline(
        outputs: &mut HashMap<JoltCommittedPolynomial, CommitmentOutput<PCS>>,
        config: CommitmentStageConfig,
        mut field_inline: HashMap<FieldInlineCommittedPolynomial, CommitmentOutput<PCS>>,
    ) -> Result<Self, ProverError> {
        let base = build_base_jolt_commitments::<PCS>(outputs, config)?;
        let (field_rd_inc, field_rd_inc_hint, _) = take_field_inline_output::<PCS>(
            &mut field_inline,
            FieldInlineCommittedPolynomial::FieldRdInc,
        )?;
        if let Some(unexpected) = field_inline.keys().next().copied() {
            return Err(ProverError::InvalidCommitmentOutput {
                reason: format!("unexpected field-inline committed polynomial {unexpected:?}"),
            });
        }

        let mut field_inline_opening_hints = HashMap::new();
        let _ = field_inline_opening_hints.insert(
            FieldInlineCommittedPolynomial::FieldRdInc,
            field_rd_inc_hint,
        );

        Ok(Self {
            commitments: JoltCommitments::new(
                base.rd_inc,
                base.ram_inc,
                base.ra,
                FieldInlineCommitments::new(FieldRegistersCommitments::new(field_rd_inc)),
            ),
            trusted_advice_commitment: base.trusted_advice_commitment,
            untrusted_advice_commitment: base.untrusted_advice_commitment,
            opening_hints: base.opening_hints,
            field_inline_opening_hints,
        })
    }
}

struct BaseJoltCommitmentParts<PCS: CommitmentScheme> {
    rd_inc: PCS::Output,
    ram_inc: PCS::Output,
    ra: JoltRaCommitments<PCS::Output>,
    trusted_advice_commitment: Option<PCS::Output>,
    untrusted_advice_commitment: Option<PCS::Output>,
    opening_hints: HashMap<JoltCommittedPolynomial, PCS::OpeningHint>,
}

fn build_base_jolt_commitments<PCS>(
    outputs: &mut HashMap<JoltCommittedPolynomial, CommitmentOutput<PCS>>,
    config: CommitmentStageConfig,
) -> Result<BaseJoltCommitmentParts<PCS>, ProverError>
where
    PCS: CommitmentScheme,
{
    let mut opening_hints = HashMap::new();
    let rd_inc =
        take_jolt_output::<PCS>(outputs, JoltCommittedPolynomial::RdInc, &mut opening_hints)?;
    let ram_inc =
        take_jolt_output::<PCS>(outputs, JoltCommittedPolynomial::RamInc, &mut opening_hints)?;
    let instruction = take_ra_outputs::<PCS>(
        outputs,
        JoltCommittedPolynomial::InstructionRa,
        config.ra_layout.instruction(),
        &mut opening_hints,
    )?;
    let ram = take_ra_outputs::<PCS>(
        outputs,
        JoltCommittedPolynomial::RamRa,
        config.ra_layout.ram(),
        &mut opening_hints,
    )?;
    let bytecode = take_ra_outputs::<PCS>(
        outputs,
        JoltCommittedPolynomial::BytecodeRa,
        config.ra_layout.bytecode(),
        &mut opening_hints,
    )?;
    let trusted_advice_commitment = take_optional_advice_output::<PCS>(
        outputs,
        JoltCommittedPolynomial::TrustedAdvice,
        config.include_trusted_advice,
        &mut opening_hints,
    )?;
    let untrusted_advice_commitment = take_optional_advice_output::<PCS>(
        outputs,
        JoltCommittedPolynomial::UntrustedAdvice,
        config.include_untrusted_advice,
        &mut opening_hints,
    )?;

    if let Some(unexpected) = outputs.keys().next().copied() {
        return Err(ProverError::InvalidCommitmentOutput {
            reason: format!("unexpected Jolt committed polynomial {unexpected:?}"),
        });
    }

    Ok(BaseJoltCommitmentParts {
        rd_inc,
        ram_inc,
        ra: JoltRaCommitments::new(instruction, ram, bytecode),
        trusted_advice_commitment,
        untrusted_advice_commitment,
        opening_hints,
    })
}

fn collect_jolt_commitment_outputs<PCS>(
    result: CommitmentResult<JoltVmNamespace, PCS>,
) -> Result<HashMap<JoltCommittedPolynomial, CommitmentOutput<PCS>>, ProverError>
where
    PCS: CommitmentScheme,
{
    let mut by_polynomial = HashMap::with_capacity(result.commitments.len());
    let mut slots = HashMap::with_capacity(result.commitments.len());
    for output in result.commitments {
        let OracleKind::Committed(polynomial) = output.oracle.kind else {
            return Err(ProverError::InvalidCommitmentOutput {
                reason: "commitment backend emitted a virtual oracle output".to_owned(),
            });
        };
        if output.rows == 0 {
            return Err(ProverError::InvalidCommitmentOutput {
                reason: format!("commitment output for {polynomial:?} has zero rows"),
            });
        }
        match slots.entry(output.slot) {
            Entry::Occupied(_) => {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("duplicate commitment output slot {:?}", output.slot),
                });
            }
            Entry::Vacant(entry) => {
                let _ = entry.insert(polynomial);
            }
        }
        match by_polynomial.entry(polynomial) {
            Entry::Occupied(_) => {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("duplicate commitment output for {polynomial:?}"),
                });
            }
            Entry::Vacant(entry) => {
                let _ = entry.insert((output.commitment, output.opening_hint, output.rows));
            }
        }
    }
    Ok(by_polynomial)
}

#[cfg(feature = "field-inline")]
fn collect_field_inline_commitment_outputs<PCS>(
    outputs: Vec<FieldInlineCommittedPolynomialOutput<PCS>>,
) -> Result<HashMap<FieldInlineCommittedPolynomial, CommitmentOutput<PCS>>, ProverError>
where
    PCS: CommitmentScheme,
{
    let mut by_polynomial = HashMap::with_capacity(outputs.len());
    for output in outputs {
        if output.rows == 0 {
            return Err(ProverError::InvalidCommitmentOutput {
                reason: format!(
                    "field-inline commitment output for {:?} has zero rows",
                    output.polynomial
                ),
            });
        }
        match by_polynomial.entry(output.polynomial) {
            Entry::Occupied(_) => {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!(
                        "duplicate field-inline commitment output for {:?}",
                        output.polynomial
                    ),
                });
            }
            Entry::Vacant(entry) => {
                let _ = entry.insert((output.commitment, output.opening_hint, output.rows));
            }
        }
    }
    Ok(by_polynomial)
}

fn take_jolt_output<PCS>(
    outputs: &mut HashMap<JoltCommittedPolynomial, CommitmentOutput<PCS>>,
    polynomial: JoltCommittedPolynomial,
    opening_hints: &mut HashMap<JoltCommittedPolynomial, PCS::OpeningHint>,
) -> Result<PCS::Output, ProverError>
where
    PCS: CommitmentScheme,
{
    let (commitment, opening_hint, _) =
        outputs
            .remove(&polynomial)
            .ok_or_else(|| ProverError::InvalidCommitmentOutput {
                reason: format!("missing Jolt commitment output for {polynomial:?}"),
            })?;
    let _ = opening_hints.insert(polynomial, opening_hint);
    Ok(commitment)
}

fn take_ra_outputs<PCS>(
    outputs: &mut HashMap<JoltCommittedPolynomial, CommitmentOutput<PCS>>,
    polynomial_at: impl Fn(usize) -> JoltCommittedPolynomial,
    count: usize,
    opening_hints: &mut HashMap<JoltCommittedPolynomial, PCS::OpeningHint>,
) -> Result<Vec<PCS::Output>, ProverError>
where
    PCS: CommitmentScheme,
{
    let mut commitments = Vec::with_capacity(count);
    for index in 0..count {
        commitments.push(take_jolt_output::<PCS>(
            outputs,
            polynomial_at(index),
            opening_hints,
        )?);
    }
    Ok(commitments)
}

fn take_optional_advice_output<PCS>(
    outputs: &mut HashMap<JoltCommittedPolynomial, CommitmentOutput<PCS>>,
    polynomial: JoltCommittedPolynomial,
    expected: bool,
    opening_hints: &mut HashMap<JoltCommittedPolynomial, PCS::OpeningHint>,
) -> Result<Option<PCS::Output>, ProverError>
where
    PCS: CommitmentScheme,
{
    if expected {
        take_jolt_output::<PCS>(outputs, polynomial, opening_hints).map(Some)
    } else {
        Ok(None)
    }
}

#[cfg(feature = "field-inline")]
fn take_field_inline_output<PCS>(
    outputs: &mut HashMap<FieldInlineCommittedPolynomial, CommitmentOutput<PCS>>,
    polynomial: FieldInlineCommittedPolynomial,
) -> Result<CommitmentOutput<PCS>, ProverError>
where
    PCS: CommitmentScheme,
{
    outputs
        .remove(&polynomial)
        .ok_or_else(|| ProverError::InvalidCommitmentOutput {
            reason: format!("missing field-inline commitment output for {polynomial:?}"),
        })
}

#[cfg(feature = "field-inline")]
#[derive(Clone)]
pub struct FieldInlineCommittedPolynomialOutput<PCS: CommitmentScheme> {
    pub polynomial: FieldInlineCommittedPolynomial,
    pub rows: usize,
    pub commitment: PCS::Output,
    pub opening_hint: PCS::OpeningHint,
}

#[cfg(feature = "field-inline")]
impl<PCS: CommitmentScheme> FieldInlineCommittedPolynomialOutput<PCS> {
    pub const fn new(
        polynomial: FieldInlineCommittedPolynomial,
        rows: usize,
        commitment: PCS::Output,
        opening_hint: PCS::OpeningHint,
    ) -> Self {
        Self {
            polynomial,
            rows,
            commitment,
            opening_hint,
        }
    }
}
