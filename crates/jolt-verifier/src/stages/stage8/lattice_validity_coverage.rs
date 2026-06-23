use jolt_claims::protocols::jolt::{
    LatticePackedFamilyId, LatticePackedValidityKind, LatticePackedValidityRequirement,
    LatticePackedViewFormula,
};
use jolt_riscv::CircuitFlags;

use super::{
    unsupported_lattice_view, validity::canonical_field_byte_width,
    JoltLatticeViewFormulaWithRowPoint,
};
use crate::{stages::stage8::Stage8OpeningId, VerifierError};

pub fn validate_lattice_view_validity_coverage<F>(
    formulas: &[JoltLatticeViewFormulaWithRowPoint<F>],
    requirements: &[LatticePackedValidityRequirement],
) -> Result<(), VerifierError> {
    for (id, formula, _) in formulas {
        validate_lattice_formula_validity_coverage(*id, formula, requirements)?;
    }
    Ok(())
}

fn validate_lattice_formula_validity_coverage<F>(
    id: Stage8OpeningId,
    formula: &LatticePackedViewFormula<F>,
    requirements: &[LatticePackedValidityRequirement],
) -> Result<(), VerifierError> {
    match formula {
        LatticePackedViewFormula::Direct {
            family,
            limb,
            symbol,
        } => validate_lattice_term_validity_coverage(id, family, *limb, *symbol, requirements),
        LatticePackedViewFormula::LinearDecoded { terms }
        | LatticePackedViewFormula::ReducedMasked { terms, .. } => {
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
        LatticePackedViewFormula::MaskedDecoded { relation } => Err(unsupported_lattice_view(
            format!("opening {id:?} still has unresolved masked relation {relation:?}"),
        )),
    }
}

fn validate_lattice_term_validity_coverage(
    id: Stage8OpeningId,
    family: &LatticePackedFamilyId,
    limb: usize,
    symbol: usize,
    requirements: &[LatticePackedValidityRequirement],
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

fn core_jolt_ra_family(family: &LatticePackedFamilyId) -> bool {
    matches!(
        family,
        LatticePackedFamilyId::InstructionRa { .. }
            | LatticePackedFamilyId::BytecodeRa { .. }
            | LatticePackedFamilyId::RamRa { .. }
    )
}

fn requirement_covers_term(
    requirement: &LatticePackedValidityRequirement,
    family: &LatticePackedFamilyId,
    limb: usize,
    symbol: usize,
) -> bool {
    if &requirement.family != family || limb >= requirement.limbs {
        return false;
    }
    match requirement.kind {
        LatticePackedValidityKind::ExactOneHot | LatticePackedValidityKind::OptionalOneHot => {
            symbol < requirement.alphabet_size
        }
        LatticePackedValidityKind::BooleanIndicator { symbol: indicator } => {
            symbol == indicator && indicator < requirement.alphabet_size
        }
        LatticePackedValidityKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityKind::FieldElementCanonicalBytes { .. } => false,
    }
}

fn term_requires_canonical_bytes(family: &LatticePackedFamilyId) -> bool {
    matches!(
        family,
        LatticePackedFamilyId::FieldRdIncByte { .. }
            | LatticePackedFamilyId::BytecodeImmBytes { .. }
    )
}

fn canonical_requirement_covers_term(
    requirement: &LatticePackedValidityRequirement,
    family: &LatticePackedFamilyId,
    limb: usize,
) -> bool {
    let Ok(byte_width) = canonical_field_byte_width(requirement) else {
        return false;
    };
    match (&requirement.family, family) {
        (
            LatticePackedFamilyId::FieldRdIncByte { index: 0 },
            LatticePackedFamilyId::FieldRdIncByte { index },
        ) => *index < byte_width && limb == 0,
        (
            LatticePackedFamilyId::BytecodeImmBytes { chunk: expected },
            LatticePackedFamilyId::BytecodeImmBytes { chunk },
        ) => expected == chunk && limb < byte_width,
        _ => false,
    }
}

fn term_requires_bytecode_store_rd_disjoint(family: &LatticePackedFamilyId) -> bool {
    match family {
        LatticePackedFamilyId::BytecodeCircuitFlag { flag, .. } => {
            *flag == CircuitFlags::Store as usize
        }
        LatticePackedFamilyId::BytecodeRegisterSelector { selector, .. } => *selector == 2,
        _ => false,
    }
}

fn bytecode_store_rd_disjoint_requirement_covers_term(
    requirement: &LatticePackedValidityRequirement,
    family: &LatticePackedFamilyId,
) -> bool {
    let LatticePackedValidityKind::BytecodeStoreRdDisjoint = requirement.kind else {
        return false;
    };
    let LatticePackedFamilyId::BytecodeCircuitFlag {
        chunk: requirement_chunk,
        flag,
    } = &requirement.family
    else {
        return false;
    };
    if *flag != CircuitFlags::Store as usize {
        return false;
    }
    match family {
        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            requirement_chunk == chunk && *flag == CircuitFlags::Store as usize
        }
        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            requirement_chunk == chunk && *selector == 2
        }
        _ => false,
    }
}
