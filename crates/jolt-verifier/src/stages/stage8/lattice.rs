use crate::{stages::PrecommittedSchedule, VerifierError};
#[cfg(feature = "field-inline")]
use jolt_akita::AkitaField;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{claim_reductions::increments as field_increments, lattice as field_lattice},
    FieldInlineOpeningId,
};
use jolt_claims::protocols::jolt::{
    byte_decode_terms, unsigned_inc_msb_lattice_view_formula, unsigned_inc_msb_opening,
    weighted_symbol_terms, JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId,
    JoltPolynomialId, JoltRelationId, LatticePackedFamilyId, LatticePackedValidityRequirement,
    LatticePackedViewFormula,
};
use jolt_field::Field;
#[cfg(feature = "field-inline")]
use jolt_field::FixedByteSize;
use jolt_openings::{
    PackingAdviceKind, PackingFamilyId, PackingViewError, PackingViewFormula, PackingViewTerm,
    PackingWitnessLayout,
};
use jolt_poly::EqPolynomial;

use super::outputs::{Stage8LogicalManifest, Stage8OpeningId, Stage8PhysicalManifest};

#[path = "lattice_layout.rs"]
mod layout;
pub use layout::{
    derive_lattice_packed_validity_requirements, derive_lattice_packed_witness_layout,
    lattice_protocol_config_for_packed_witness_layout,
    lattice_validity_requirements_for_packed_witness_layout,
    validate_lattice_packed_witness_layout_config, validate_lattice_packed_witness_validity_config,
};

#[path = "lattice_validity.rs"]
mod validity;
#[cfg(test)]
pub(crate) use validity::field_element_canonical_value_from_openings;
pub use validity::{
    build_lattice_packed_validity_batch, derive_lattice_packed_validity_statements,
    lattice_packed_validity_claims, lattice_packed_validity_opening_count,
    sample_lattice_packed_validity_eq_points, verify_lattice_packed_validity_proof,
    LatticePackedValidityBatch, LatticePackedValidityBatchStatement,
    LatticePackedValidityStatement, LatticePackedValidityStatementKind,
};
pub(crate) use validity::{field_element_canonical_factors, FieldCanonicalFactor};

#[path = "lattice_validity_coverage.rs"]
mod validity_coverage;
pub use validity_coverage::validate_lattice_view_validity_coverage;

pub type JoltLatticeViewFormulaWithRowPoint<F> =
    (Stage8OpeningId, LatticePackedViewFormula<F>, Vec<F>);

pub fn jolt_lattice_view_formulas<F>(
    logical: &Stage8LogicalManifest<F>,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<JoltLatticeViewFormulaWithRowPoint<F>>, VerifierError>
where
    F: Field,
{
    logical
        .openings
        .iter()
        .map(|opening| {
            let (formula, row_point) =
                stage8_lattice_view_formula(opening.id, &opening.point, log_k_chunk, precommitted)?;
            Ok((opening.id, formula, row_point))
        })
        .collect()
}

fn stage8_lattice_view_formula<F>(
    id: Stage8OpeningId,
    point: &[F],
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<(LatticePackedViewFormula<F>, Vec<F>), VerifierError>
where
    F: Field,
{
    match id {
        Stage8OpeningId::Jolt(id) => Ok((
            jolt_lattice_view_formula(id, point, log_k_chunk, precommitted)?,
            jolt_lattice_row_point(id, point, log_k_chunk, precommitted)?,
        )),
        #[cfg(feature = "field-inline")]
        Stage8OpeningId::FieldInline(id) => Ok((
            field_inline_lattice_view_formula(id)?,
            field_inline_lattice_row_point(id, point)?,
        )),
    }
}

pub fn jolt_lattice_physical_manifest<F>(
    logical: &Stage8LogicalManifest<F>,
    layout: &PackingWitnessLayout,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Stage8PhysicalManifest<F>, VerifierError>
where
    F: Field,
{
    let formulas = jolt_lattice_view_formulas(logical, log_k_chunk, precommitted)?;
    Stage8PhysicalManifest::from_jolt_lattice_view_formulas(logical, layout, formulas)
        .map_err(lattice_view_resolution_error)
}

pub fn jolt_lattice_physical_manifest_with_validity<F>(
    logical: &Stage8LogicalManifest<F>,
    layout: &PackingWitnessLayout,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
    validity_requirements: &[LatticePackedValidityRequirement],
) -> Result<Stage8PhysicalManifest<F>, VerifierError>
where
    F: Field,
{
    let formulas = jolt_lattice_view_formulas(logical, log_k_chunk, precommitted)?;
    validate_lattice_view_validity_coverage(&formulas, validity_requirements)?;
    Stage8PhysicalManifest::from_jolt_lattice_view_formulas(logical, layout, formulas)
        .map_err(lattice_view_resolution_error)
}

fn jolt_lattice_row_point<F>(
    id: JoltOpeningId,
    point: &[F],
    log_k_chunk: usize,
    _precommitted: &PrecommittedSchedule,
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(polynomial),
            relation,
        } => committed_lattice_row_point(polynomial, relation, point, log_k_chunk),
        JoltOpeningId::TrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Err(unsupported_lattice_view(
            "trusted advice uses a separate precommitted opening, not a packed witness row point",
        )),
        JoltOpeningId::UntrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(point.to_vec()),
        id if id == unsigned_inc_msb_opening() => Ok(point.to_vec()),
        id if unsigned_inc_chunk_index(id).is_some() => ra_row_point(point, log_k_chunk),
        _ => Err(unsupported_lattice_view(format!(
            "final opening {id:?} has no supported lattice packed row point"
        ))),
    }
}

fn committed_lattice_row_point<F>(
    polynomial: JoltCommittedPolynomial,
    relation: JoltRelationId,
    point: &[F],
    log_k_chunk: usize,
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    match (polynomial, relation) {
        (
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_),
            JoltRelationId::HammingWeightClaimReduction,
        ) => ra_row_point(point, log_k_chunk),
        (
            JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc,
            JoltRelationId::IncClaimReduction,
        ) => Err(unsupported_lattice_view(
            "lattice mode opens increment chunks/MSB through unsigned increment reconstruction, not dense IncClaimReduction polynomials",
        )),
        (JoltCommittedPolynomial::ProgramImageInit, JoltRelationId::ProgramImageClaimReduction) => {
            Err(unsupported_lattice_view(
                "ProgramImageInit uses a separate precommitted opening, not a packed witness row point",
            ))
        }
        (JoltCommittedPolynomial::TrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Err(unsupported_lattice_view(
                "TrustedAdvice uses a separate precommitted opening, not a packed witness row point",
            ))
        }
        (JoltCommittedPolynomial::UntrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(point.to_vec())
        }
        (JoltCommittedPolynomial::BytecodeChunk(index), JoltRelationId::BytecodeClaimReduction) => {
            Err(unsupported_lattice_view(format!(
                "BytecodeChunk({index}) uses a separate precommitted opening, not a packed witness row point"
            )))
        }
        _ => Err(unsupported_lattice_view(format!(
            "committed polynomial {polynomial:?} under relation {relation:?} has no supported lattice packed row point"
        ))),
    }
}

fn ra_row_point<F>(point: &[F], log_k_chunk: usize) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    if point.len() < log_k_chunk {
        return Err(unsupported_lattice_view(format!(
            "RA lattice opening point has {} variables but needs at least {log_k_chunk}",
            point.len()
        )));
    }
    Ok(point[log_k_chunk..].to_vec())
}

#[cfg(feature = "field-inline")]
fn field_inline_lattice_row_point<F>(
    id: FieldInlineOpeningId,
    point: &[F],
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    if id == field_increments::field_rd_inc_reduced_opening() {
        return Ok(point.to_vec());
    }
    Err(unsupported_lattice_view(format!(
        "field-inline opening {id:?} has no supported lattice packed row point"
    )))
}

#[cfg(feature = "field-inline")]
fn field_inline_lattice_view_formula<F>(
    id: FieldInlineOpeningId,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    if id == field_increments::field_rd_inc_reduced_opening() {
        return Ok(field_lattice::field_rd_inc_lattice_view_formula(
            AkitaField::NUM_BYTES,
        ));
    }
    Err(unsupported_lattice_view(format!(
        "field-inline opening {id:?} has no supported lattice packed view"
    )))
}

pub fn jolt_lattice_view_formula<F>(
    id: JoltOpeningId,
    point: &[F],
    log_k_chunk: usize,
    _precommitted: &PrecommittedSchedule,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(polynomial),
            relation,
        } => committed_lattice_view_formula(polynomial, relation, point, log_k_chunk),
        JoltOpeningId::TrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Err(unsupported_lattice_view(
            "trusted advice uses a separate precommitted opening, not a packed witness view",
        )),
        JoltOpeningId::UntrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted)),
        id if id == unsigned_inc_msb_opening() => Ok(unsigned_inc_msb_lattice_view_formula()),
        id => {
            if let Some(index) = unsigned_inc_chunk_index(id) {
                return unsigned_inc_chunk_lattice_view_formula(index, point, log_k_chunk);
            }
            Err(unsupported_lattice_view(format!(
                "final opening {id:?} has no supported lattice packed view"
            )))
        }
    }
}

pub fn lattice_packing_family_id(family: &LatticePackedFamilyId) -> PackingFamilyId {
    match family {
        LatticePackedFamilyId::InstructionRa { index } => {
            PackingFamilyId::InstructionRa { index: *index }
        }
        LatticePackedFamilyId::BytecodeRa { index } => {
            PackingFamilyId::BytecodeRa { index: *index }
        }
        LatticePackedFamilyId::RamRa { index } => PackingFamilyId::RamRa { index: *index },
        LatticePackedFamilyId::UnsignedIncChunk { index } => {
            PackingFamilyId::UnsignedIncChunk { index: *index }
        }
        LatticePackedFamilyId::UnsignedIncMsb => PackingFamilyId::UnsignedIncMsb,
        LatticePackedFamilyId::FieldRdIncByte { index } => {
            PackingFamilyId::FieldRdIncByte { index: *index }
        }
        LatticePackedFamilyId::FieldRdIncSign => PackingFamilyId::FieldRdIncSign,
        LatticePackedFamilyId::AdviceBytes { kind, index } => PackingFamilyId::AdviceBytes {
            kind: lattice_packing_advice_kind(*kind),
            index: *index,
        },
        LatticePackedFamilyId::BytecodeChunk { index } => {
            PackingFamilyId::BytecodeChunk { index: *index }
        }
        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            PackingFamilyId::BytecodeRegisterSelector {
                chunk: *chunk,
                selector: *selector,
            }
        }
        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            PackingFamilyId::BytecodeCircuitFlag {
                chunk: *chunk,
                flag: *flag,
            }
        }
        LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag } => {
            PackingFamilyId::BytecodeInstructionFlag {
                chunk: *chunk,
                flag: *flag,
            }
        }
        LatticePackedFamilyId::BytecodeLookupSelector { chunk } => {
            PackingFamilyId::BytecodeLookupSelector { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeRafFlag { chunk } => {
            PackingFamilyId::BytecodeRafFlag { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
            PackingFamilyId::BytecodeUnexpandedPcBytes { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeImmBytes { chunk } => {
            PackingFamilyId::BytecodeImmBytes { chunk: *chunk }
        }
        LatticePackedFamilyId::ProgramImageInit => PackingFamilyId::ProgramImageInit,
        LatticePackedFamilyId::Custom { namespace, index } => PackingFamilyId::Custom {
            namespace: *namespace,
            index: *index,
        },
    }
}

fn committed_lattice_view_formula<F>(
    polynomial: JoltCommittedPolynomial,
    relation: JoltRelationId,
    point: &[F],
    log_k_chunk: usize,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    match (polynomial, relation) {
        (
            JoltCommittedPolynomial::InstructionRa(index),
            JoltRelationId::HammingWeightClaimReduction,
        ) => ra_lattice_view_formula(
            LatticePackedFamilyId::InstructionRa { index },
            point,
            log_k_chunk,
        ),
        (
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltRelationId::HammingWeightClaimReduction,
        ) => {
            ra_lattice_view_formula(
                LatticePackedFamilyId::BytecodeRa { index },
                point,
                log_k_chunk,
            )
        }
        (JoltCommittedPolynomial::RamRa(index), JoltRelationId::HammingWeightClaimReduction) => {
            ra_lattice_view_formula(
                LatticePackedFamilyId::RamRa { index },
                point,
                log_k_chunk,
            )
        }
        (JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc, JoltRelationId::IncClaimReduction) => {
            Err(unsupported_lattice_view(
                "lattice mode opens increment chunks/MSB through unsigned increment reconstruction, not dense IncClaimReduction polynomials",
            ))
        }
        (JoltCommittedPolynomial::ProgramImageInit, JoltRelationId::ProgramImageClaimReduction) => {
            Err(unsupported_lattice_view(
                "ProgramImageInit uses a separate precommitted opening, not a packed witness view",
            ))
        }
        (JoltCommittedPolynomial::TrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Err(unsupported_lattice_view(
                "TrustedAdvice uses a separate precommitted opening, not a packed witness view",
            ))
        }
        (JoltCommittedPolynomial::UntrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted))
        }
        (JoltCommittedPolynomial::BytecodeChunk(index), JoltRelationId::BytecodeClaimReduction) => {
            Err(unsupported_lattice_view(format!(
                "BytecodeChunk({index}) uses a separate precommitted opening, not a packed witness view"
            )))
        }
        _ => Err(unsupported_lattice_view(format!(
            "committed polynomial {polynomial:?} under relation {relation:?} has no supported lattice packed view"
        ))),
    }
}

fn unsigned_inc_chunk_index(id: JoltOpeningId) -> Option<usize> {
    let JoltOpeningId::Lattice {
        relation: JoltRelationId::UnsignedIncChunkReconstruction,
        index,
    } = id
    else {
        return None;
    };
    Some(index)
}

fn unsigned_inc_chunk_lattice_view_formula<F>(
    index: usize,
    point: &[F],
    log_k_chunk: usize,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    if log_k_chunk == 0 {
        return Err(unsupported_lattice_view(
            "unsigned increment chunk view requires a nonzero chunk size",
        ));
    }
    if point.len() < log_k_chunk {
        return Err(unsupported_lattice_view(format!(
            "unsigned increment chunk opening point has {} variables but needs at least {log_k_chunk}",
            point.len()
        )));
    }
    ra_lattice_view_formula(
        LatticePackedFamilyId::UnsignedIncChunk { index },
        point,
        log_k_chunk,
    )
}

fn ra_lattice_view_formula<F>(
    family: LatticePackedFamilyId,
    point: &[F],
    log_k_chunk: usize,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    if log_k_chunk == 0 {
        return Err(unsupported_lattice_view(
            "RA lattice view requires a nonzero one-hot chunk size",
        ));
    }
    if point.len() < log_k_chunk {
        return Err(unsupported_lattice_view(format!(
            "RA lattice opening point has {} variables but needs at least {log_k_chunk}",
            point.len()
        )));
    }
    Ok(LatticePackedViewFormula::linear_decoded(
        weighted_symbol_terms(
            family,
            0,
            EqPolynomial::<F>::evals(&point[..log_k_chunk], None),
        ),
    ))
}

fn advice_lattice_view_formula<F>(kind: JoltAdviceKind) -> LatticePackedViewFormula<F>
where
    F: Field,
{
    LatticePackedViewFormula::linear_decoded(byte_decode_terms(
        LatticePackedFamilyId::AdviceBytes { kind, index: 0 },
        0,
    ))
}

pub fn lattice_packing_view_formula<F>(
    formula: &LatticePackedViewFormula<F>,
) -> Result<PackingViewFormula<F>, PackingViewError>
where
    F: Field,
{
    match formula {
        LatticePackedViewFormula::Direct {
            family,
            limb,
            symbol,
        } => Ok(PackingViewFormula::direct(
            lattice_packing_family_id(family),
            *limb,
            *symbol,
        )),
        LatticePackedViewFormula::LinearDecoded { terms } => {
            Ok(PackingViewFormula::linear_decoded(
                terms
                    .iter()
                    .map(|term| {
                        PackingViewTerm::new(
                            term.coefficient,
                            lattice_packing_family_id(&term.family),
                            term.limb,
                            term.symbol,
                        )
                    })
                    .collect(),
            ))
        }
        LatticePackedViewFormula::ReducedMasked { terms, .. } => {
            Ok(PackingViewFormula::reduced_masked(
                terms
                    .iter()
                    .map(|term| {
                        PackingViewTerm::new(
                            term.coefficient,
                            lattice_packing_family_id(&term.family),
                            term.limb,
                            term.symbol,
                        )
                    })
                    .collect(),
            ))
        }
        LatticePackedViewFormula::MaskedDecoded { .. } => {
            Err(PackingViewError::MaskedViewRequiresTranslation)
        }
    }
}

fn lattice_packing_advice_kind(kind: JoltAdviceKind) -> PackingAdviceKind {
    match kind {
        JoltAdviceKind::Trusted => PackingAdviceKind::Trusted,
        JoltAdviceKind::Untrusted => PackingAdviceKind::Untrusted,
    }
}

fn power_of_two_log(value: usize, name: &'static str) -> Result<usize, VerifierError> {
    if value.is_power_of_two() {
        Ok(value.trailing_zeros() as usize)
    } else {
        Err(invalid_lattice_config(format!(
            "{name} must be a power of two, got {value}",
        )))
    }
}

fn invalid_lattice_config(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidProtocolConfig {
        reason: reason.into(),
    }
}

fn invalid_precommitted_schedule(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidPrecommittedSchedule {
        reason: reason.into(),
    }
}

fn unsupported_lattice_view(reason: impl Into<String>) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: format!("unsupported lattice final opening view: {}", reason.into()),
    }
}

fn lattice_view_resolution_error(error: PackingViewError) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: format!("lattice packed view resolution failed: {error}"),
    }
}
