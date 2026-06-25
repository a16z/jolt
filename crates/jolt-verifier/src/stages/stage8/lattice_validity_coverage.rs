use jolt_claims::protocols::jolt::JoltPackingFamilyId;
use jolt_openings::{
    PackingFamilyId, PackingValidityKind, PackingValidityRequirement, PackingViewFormula,
};
use jolt_riscv::CircuitFlags;

use super::{
    unsupported_lattice_view, validity::canonical_field_byte_width,
    JoltLatticeViewFormulaWithRowPoint,
};
use crate::{stages::stage8::Stage8OpeningId, VerifierError};

pub fn validate_lattice_view_validity_coverage<F>(
    formulas: &[JoltLatticeViewFormulaWithRowPoint<F>],
    requirements: &[PackingValidityRequirement],
) -> Result<(), VerifierError> {
    for (id, formula, _) in formulas {
        validate_lattice_formula_validity_coverage(*id, formula, requirements)?;
    }
    Ok(())
}

fn validate_lattice_formula_validity_coverage<F>(
    id: Stage8OpeningId,
    formula: &PackingViewFormula<F>,
    requirements: &[PackingValidityRequirement],
) -> Result<(), VerifierError> {
    match formula {
        PackingViewFormula::Direct {
            family,
            limb,
            symbol,
        } => validate_lattice_term_validity_coverage(id, family, *limb, *symbol, requirements),
        PackingViewFormula::LinearDecoded { terms, .. }
        | PackingViewFormula::ReducedMasked { terms, .. } => {
            for term in terms {
                validate_lattice_term_validity_coverage(
                    id,
                    &term.family,
                    term.limb,
                    term.symbol,
                    requirements,
                )?;
            }
            Ok(())
        }
        PackingViewFormula::MaskedDecoded => Err(unsupported_lattice_view(format!(
            "opening {id:?} still has an unresolved masked view"
        ))),
    }
}

fn validate_lattice_term_validity_coverage(
    id: Stage8OpeningId,
    family: &PackingFamilyId,
    limb: usize,
    symbol: usize,
    requirements: &[PackingValidityRequirement],
) -> Result<(), VerifierError> {
    if core_jolt_ra_family(family) {
        return Ok(());
    }
    let has_value_validity = requirements
        .iter()
        .any(|requirement| requirement_covers_term(requirement, family, limb, symbol));
    if !has_value_validity {
        return Err(unsupported_lattice_view(format!(
            "opening {id:?} uses packed family {family:?} limb {limb} symbol {symbol} without a bound validity requirement"
        )));
    }
    if term_requires_canonical_bytes(family)
        && !requirements
            .iter()
            .any(|requirement| canonical_requirement_covers_term(requirement, family, limb))
    {
        return Err(unsupported_lattice_view(format!(
            "opening {id:?} uses field-byte packed family {family:?} limb {limb} without a bound canonical-byte validity requirement"
        )));
    }
    if term_requires_bytecode_store_rd_disjoint(family)
        && !requirements.iter().any(|requirement| {
            bytecode_store_rd_disjoint_requirement_covers_term(requirement, family)
        })
    {
        return Err(unsupported_lattice_view(format!(
            "opening {id:?} uses bytecode source family {family:?} without a bound Store/Rd disjointness requirement"
        )));
    }
    Ok(())
}

fn core_jolt_ra_family(family: &PackingFamilyId) -> bool {
    matches!(
        JoltPackingFamilyId::from_physical_id(family),
        Some(
            JoltPackingFamilyId::InstructionRa { .. }
                | JoltPackingFamilyId::BytecodeRa { .. }
                | JoltPackingFamilyId::RamRa { .. }
        )
    )
}

fn requirement_covers_term(
    requirement: &PackingValidityRequirement,
    family: &PackingFamilyId,
    limb: usize,
    symbol: usize,
) -> bool {
    if &requirement.family != family || limb >= requirement.limbs {
        return false;
    }
    match requirement.kind {
        PackingValidityKind::ExactOneHot | PackingValidityKind::OptionalOneHot => {
            symbol < requirement.alphabet_size
        }
        PackingValidityKind::BooleanIndicator { symbol: indicator } => {
            symbol == indicator && indicator < requirement.alphabet_size
        }
        PackingValidityKind::BytecodeStoreRdDisjoint
        | PackingValidityKind::FieldElementCanonicalBytes { .. } => false,
    }
}

fn term_requires_canonical_bytes(family: &PackingFamilyId) -> bool {
    matches!(
        JoltPackingFamilyId::from_physical_id(family),
        Some(
            JoltPackingFamilyId::FieldRdIncByte { .. }
                | JoltPackingFamilyId::BytecodeImmBytes { .. }
        )
    )
}

fn canonical_requirement_covers_term(
    requirement: &PackingValidityRequirement,
    family: &PackingFamilyId,
    limb: usize,
) -> bool {
    let Ok(byte_width) = canonical_field_byte_width(requirement) else {
        return false;
    };
    match (
        JoltPackingFamilyId::from_physical_id(&requirement.family),
        JoltPackingFamilyId::from_physical_id(family),
    ) {
        (
            Some(JoltPackingFamilyId::FieldRdIncByte { index: 0 }),
            Some(JoltPackingFamilyId::FieldRdIncByte { index }),
        ) => index < byte_width && limb == 0,
        (
            Some(JoltPackingFamilyId::BytecodeImmBytes { chunk: expected }),
            Some(JoltPackingFamilyId::BytecodeImmBytes { chunk }),
        ) => expected == chunk && limb < byte_width,
        _ => false,
    }
}

fn term_requires_bytecode_store_rd_disjoint(family: &PackingFamilyId) -> bool {
    match JoltPackingFamilyId::from_physical_id(family) {
        Some(JoltPackingFamilyId::BytecodeCircuitFlag { flag, .. }) => {
            flag == CircuitFlags::Store as usize
        }
        Some(JoltPackingFamilyId::BytecodeRegisterSelector { selector, .. }) => selector == 2,
        _ => false,
    }
}

fn bytecode_store_rd_disjoint_requirement_covers_term(
    requirement: &PackingValidityRequirement,
    family: &PackingFamilyId,
) -> bool {
    let PackingValidityKind::BytecodeStoreRdDisjoint = requirement.kind else {
        return false;
    };
    let Some(JoltPackingFamilyId::BytecodeCircuitFlag {
        chunk: requirement_chunk,
        flag,
    }) = JoltPackingFamilyId::from_physical_id(&requirement.family)
    else {
        return false;
    };
    if flag != CircuitFlags::Store as usize {
        return false;
    }
    match JoltPackingFamilyId::from_physical_id(family) {
        Some(JoltPackingFamilyId::BytecodeCircuitFlag { chunk, flag }) => {
            requirement_chunk == chunk && flag == CircuitFlags::Store as usize
        }
        Some(JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector }) => {
            requirement_chunk == chunk && selector == 2
        }
        _ => false,
    }
}
