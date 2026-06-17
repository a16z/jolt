use crate::{
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    stages::PrecommittedSchedule,
    VerifierError,
};
use jolt_akita::{
    AkitaField, PackedAdviceKind, PackedAlphabet, PackedFactDomain, PackedFamilyId,
    PackedFamilySpec, PackedViewError, PackedViewFormula, PackedViewTerm, PackedWitnessLayout,
};
use jolt_claims::protocols::jolt::{
    byte_decode_terms,
    formulas::{dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    little_endian_byte_decode_terms, weighted_symbol_terms, AdviceClaimReductionLayout,
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltRelationId,
    LatticePackedFamilyId, LatticePackedViewFormula, ProgramImageClaimReductionLayout,
};
use jolt_field::{Field, FixedByteSize};
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use super::outputs::{Stage8LogicalManifest, Stage8OpeningId, Stage8PhysicalManifest};

pub fn derive_akita_packed_witness_layout(
    config: &JoltProtocolConfig,
    log_t: usize,
    log_k_chunk: usize,
    ra_layout: JoltRaPolynomialLayout,
    precommitted: &PrecommittedSchedule,
) -> Result<PackedWitnessLayout, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness layout derivation requires lattice PCS mode",
        ));
    }

    let trace = PackedFactDomain::TraceRows { log_t };
    let ra_alphabet = one_hot_alphabet(log_k_chunk)?;
    let mut specs = Vec::new();
    specs.extend((0..ra_layout.instruction()).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index },
            trace,
            1,
            ra_alphabet,
        )
    }));
    specs.extend((0..ra_layout.bytecode()).map(|index| {
        PackedFamilySpec::direct(PackedFamilyId::BytecodeRa { index }, trace, 1, ra_alphabet)
    }));
    specs.extend((0..ra_layout.ram()).map(|index| {
        PackedFamilySpec::direct(PackedFamilyId::RamRa { index }, trace, 1, ra_alphabet)
    }));

    specs.extend((0..8).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::IncByte { index },
            trace,
            1,
            PackedAlphabet::Byte,
        )
    }));
    specs.push(PackedFamilySpec::direct(
        PackedFamilyId::IncSign,
        trace,
        1,
        PackedAlphabet::Bit,
    ));

    if config.lattice.field_inline.enabled {
        specs.extend((0..8).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                trace,
                1,
                PackedAlphabet::Byte,
            )
        }));
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::FieldRdIncSign,
            trace,
            1,
            PackedAlphabet::Bit,
        ));
    }

    if config.lattice.advice.trusted {
        specs.push(advice_family(
            PackedAdviceKind::Trusted,
            require_advice_layout(precommitted, JoltAdviceKind::Trusted)?,
        ));
    }
    if config.lattice.advice.untrusted {
        specs.push(advice_family(
            PackedAdviceKind::Untrusted,
            require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?,
        ));
    }

    let bytecode_layout = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;
    for index in 0..bytecode_layout.chunk_count() {
        extend_bytecode_families(&mut specs, index, bytecode_layout.log_bytecode_chunk_size());
    }

    let program_image_layout = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;
    specs.push(program_image_family(program_image_layout)?);

    PackedWitnessLayout::new(specs).map_err(|error| invalid_lattice_config(error.to_string()))
}

pub fn validate_akita_packed_witness_layout_config(
    config: &JoltProtocolConfig,
    layout: &PackedWitnessLayout,
) -> Result<(), VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness layout validation requires lattice PCS mode",
        ));
    }
    if config.lattice.packed_witness.layout_digest != Some(layout.digest) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness layout digest does not match derived layout",
        ));
    }
    if config.lattice.packed_witness.d_pack != Some(layout.dimension) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness D_pack does not match derived layout",
        ));
    }
    Ok(())
}

pub fn jolt_lattice_view_formulas<F>(
    logical: &Stage8LogicalManifest<F>,
    log_k_chunk: usize,
) -> Result<Vec<(JoltOpeningId, LatticePackedViewFormula<F>)>, VerifierError>
where
    F: Field,
{
    logical
        .openings
        .iter()
        .map(|opening| {
            let id = stage8_jolt_opening_id(opening.id)?;
            Ok((
                id,
                jolt_lattice_view_formula(id, &opening.point, log_k_chunk)?,
            ))
        })
        .collect()
}

pub fn jolt_lattice_physical_manifest<F>(
    logical: &Stage8LogicalManifest<F>,
    layout: &PackedWitnessLayout,
    log_k_chunk: usize,
) -> Result<Stage8PhysicalManifest<F>, VerifierError>
where
    F: Field,
{
    let formulas = jolt_lattice_view_formulas(logical, log_k_chunk)?;
    Stage8PhysicalManifest::from_jolt_lattice_view_formulas(logical, layout, formulas)
        .map_err(lattice_view_resolution_error)
}

#[cfg(not(feature = "field-inline"))]
fn stage8_jolt_opening_id(id: Stage8OpeningId) -> Result<JoltOpeningId, VerifierError> {
    let Stage8OpeningId::Jolt(id) = id;
    Ok(id)
}

#[cfg(feature = "field-inline")]
fn stage8_jolt_opening_id(id: Stage8OpeningId) -> Result<JoltOpeningId, VerifierError> {
    match id {
        Stage8OpeningId::Jolt(id) => Ok(id),
        Stage8OpeningId::FieldInline(id) => Err(unsupported_lattice_view(format!(
            "field-inline opening {id:?} requires a field-inline lattice view policy"
        ))),
    }
}

pub fn jolt_lattice_view_formula<F>(
    id: JoltOpeningId,
    point: &[F],
    log_k_chunk: usize,
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
        } => Ok(advice_lattice_view_formula(JoltAdviceKind::Trusted)),
        JoltOpeningId::UntrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted)),
        _ => Err(unsupported_lattice_view(format!(
            "final opening {id:?} has no supported lattice packed view"
        ))),
    }
}

pub fn akita_packed_family_id(family: &LatticePackedFamilyId) -> PackedFamilyId {
    match family {
        LatticePackedFamilyId::InstructionRa { index } => {
            PackedFamilyId::InstructionRa { index: *index }
        }
        LatticePackedFamilyId::BytecodeRa { index } => PackedFamilyId::BytecodeRa { index: *index },
        LatticePackedFamilyId::RamRa { index } => PackedFamilyId::RamRa { index: *index },
        LatticePackedFamilyId::IncByte { index } => PackedFamilyId::IncByte { index: *index },
        LatticePackedFamilyId::IncSign => PackedFamilyId::IncSign,
        LatticePackedFamilyId::RamIncByte { index } => PackedFamilyId::RamIncByte { index: *index },
        LatticePackedFamilyId::RamIncSign => PackedFamilyId::RamIncSign,
        LatticePackedFamilyId::RdIncByte { index } => PackedFamilyId::RdIncByte { index: *index },
        LatticePackedFamilyId::RdIncSign => PackedFamilyId::RdIncSign,
        LatticePackedFamilyId::FieldRdIncByte { index } => {
            PackedFamilyId::FieldRdIncByte { index: *index }
        }
        LatticePackedFamilyId::FieldRdIncSign => PackedFamilyId::FieldRdIncSign,
        LatticePackedFamilyId::AdviceBytes { kind, index } => PackedFamilyId::AdviceBytes {
            kind: akita_advice_kind(*kind),
            index: *index,
        },
        LatticePackedFamilyId::BytecodeChunk { index } => {
            PackedFamilyId::BytecodeChunk { index: *index }
        }
        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: *chunk,
                selector: *selector,
            }
        }
        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            PackedFamilyId::BytecodeCircuitFlag {
                chunk: *chunk,
                flag: *flag,
            }
        }
        LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag } => {
            PackedFamilyId::BytecodeInstructionFlag {
                chunk: *chunk,
                flag: *flag,
            }
        }
        LatticePackedFamilyId::BytecodeLookupSelector { chunk } => {
            PackedFamilyId::BytecodeLookupSelector { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeRafFlag { chunk } => {
            PackedFamilyId::BytecodeRafFlag { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
            PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeImmBytes { chunk } => {
            PackedFamilyId::BytecodeImmBytes { chunk: *chunk }
        }
        LatticePackedFamilyId::ProgramImageInit => PackedFamilyId::ProgramImageInit,
        LatticePackedFamilyId::Custom { namespace, index } => PackedFamilyId::Custom {
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
        (JoltCommittedPolynomial::BytecodeRa(index), JoltRelationId::HammingWeightClaimReduction) => {
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
            Ok(LatticePackedViewFormula::MaskedDecoded)
        }
        (JoltCommittedPolynomial::ProgramImageInit, JoltRelationId::ProgramImageClaimReduction) => {
            Ok(LatticePackedViewFormula::linear_decoded(
                little_endian_byte_decode_terms(LatticePackedFamilyId::ProgramImageInit, 8),
            ))
        }
        (JoltCommittedPolynomial::TrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(advice_lattice_view_formula(JoltAdviceKind::Trusted))
        }
        (JoltCommittedPolynomial::UntrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted))
        }
        (JoltCommittedPolynomial::BytecodeChunk(index), JoltRelationId::BytecodeClaimReduction) => {
            Err(unsupported_lattice_view(format!(
                "BytecodeChunk({index}) lattice view requires committed-bytecode lane byte policy"
            )))
        }
        _ => Err(unsupported_lattice_view(format!(
            "committed polynomial {polynomial:?} under relation {relation:?} has no supported lattice packed view"
        ))),
    }
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

pub fn akita_packed_view_formula<F>(
    formula: &LatticePackedViewFormula<F>,
) -> Result<PackedViewFormula<F>, PackedViewError>
where
    F: Field,
{
    match formula {
        LatticePackedViewFormula::Direct {
            family,
            limb,
            symbol,
        } => Ok(PackedViewFormula::direct(
            akita_packed_family_id(family),
            *limb,
            *symbol,
        )),
        LatticePackedViewFormula::LinearDecoded { terms } => Ok(PackedViewFormula::linear_decoded(
            terms
                .iter()
                .map(|term| {
                    PackedViewTerm::new(
                        term.coefficient,
                        akita_packed_family_id(&term.family),
                        term.limb,
                        term.symbol,
                    )
                })
                .collect(),
        )),
        LatticePackedViewFormula::ReducedMasked { .. }
        | LatticePackedViewFormula::MaskedDecoded => {
            Err(PackedViewError::MaskedViewRequiresTranslation)
        }
    }
}

fn akita_advice_kind(kind: JoltAdviceKind) -> PackedAdviceKind {
    match kind {
        JoltAdviceKind::Trusted => PackedAdviceKind::Trusted,
        JoltAdviceKind::Untrusted => PackedAdviceKind::Untrusted,
    }
}

fn advice_family(kind: PackedAdviceKind, layout: &AdviceClaimReductionLayout) -> PackedFamilySpec {
    PackedFamilySpec::direct(
        PackedFamilyId::AdviceBytes { kind, index: 0 },
        PackedFactDomain::AdviceBytes {
            kind,
            log_bytes: layout.advice_shape().total_vars() + 3,
        },
        1,
        PackedAlphabet::Byte,
    )
}

fn extend_bytecode_families(specs: &mut Vec<PackedFamilySpec>, chunk: usize, log_bytecode: usize) {
    let domain = PackedFactDomain::BytecodeRows { log_bytecode };
    let register_alphabet = PackedAlphabet::Fixed {
        size: 1usize << REGISTER_ADDRESS_BITS,
    };
    for selector in 0..3 {
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRegisterSelector { chunk, selector },
            domain,
            1,
            register_alphabet,
        ));
    }
    for flag in 0..NUM_CIRCUIT_FLAGS {
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::BytecodeCircuitFlag { chunk, flag },
            domain,
            1,
            PackedAlphabet::Bit,
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::BytecodeInstructionFlag { chunk, flag },
            domain,
            1,
            PackedAlphabet::Bit,
        ));
    }
    specs.push(PackedFamilySpec::direct(
        PackedFamilyId::BytecodeLookupSelector { chunk },
        domain,
        1,
        PackedAlphabet::Fixed {
            size: LookupTableKind::<RISCV_XLEN>::COUNT,
        },
    ));
    specs.push(PackedFamilySpec::direct(
        PackedFamilyId::BytecodeRafFlag { chunk },
        domain,
        1,
        PackedAlphabet::Bit,
    ));
    specs.push(PackedFamilySpec::direct(
        PackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
        domain,
        8,
        PackedAlphabet::Byte,
    ));
    specs.push(PackedFamilySpec::direct(
        PackedFamilyId::BytecodeImmBytes { chunk },
        domain,
        AkitaField::NUM_BYTES,
        PackedAlphabet::Byte,
    ));
}

fn program_image_family(
    layout: &ProgramImageClaimReductionLayout,
) -> Result<PackedFamilySpec, VerifierError> {
    Ok(PackedFamilySpec::direct(
        PackedFamilyId::ProgramImageInit,
        PackedFactDomain::ProgramImageWords {
            log_words: power_of_two_log(layout.padded_len_words(), "program image length")?,
        },
        8,
        PackedAlphabet::Byte,
    ))
}

fn require_advice_layout(
    precommitted: &PrecommittedSchedule,
    kind: JoltAdviceKind,
) -> Result<&AdviceClaimReductionLayout, VerifierError> {
    precommitted.advice(kind).ok_or_else(|| {
        invalid_precommitted_schedule(format!(
            "lattice advice mode requires a {kind:?} advice claim-reduction layout",
        ))
    })
}

fn one_hot_alphabet(log_k_chunk: usize) -> Result<PackedAlphabet, VerifierError> {
    match log_k_chunk {
        0 => Err(invalid_lattice_config(
            "lattice one-hot chunk size must be nonzero",
        )),
        1 => Ok(PackedAlphabet::Bit),
        8 => Ok(PackedAlphabet::Byte),
        bits if bits < usize::BITS as usize => Ok(PackedAlphabet::Fixed {
            size: 1usize << bits,
        }),
        _ => Err(invalid_lattice_config(
            "lattice one-hot chunk size is too large",
        )),
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

fn lattice_view_resolution_error(error: PackedViewError) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: format!("lattice packed view resolution failed: {error}"),
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on setup errors")]

    use super::super::outputs::{Stage8LogicalManifest, Stage8LogicalOpening, Stage8OpeningId};
    use super::*;
    use crate::{
        config::{IncrementCommitmentMode, PackedWitnessConfig, ProgramMode},
        stages::CommittedProgramSchedule,
    };
    use jolt_claims::protocols::jolt::{
        byte_decode_terms, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
        LatticePackedFamilyId, LatticePackedViewFormula, TracePolynomialOrder,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::PhysicalView;
    use jolt_poly::{EqPolynomial, Point};

    fn lattice_config() -> JoltProtocolConfig {
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness = PackedWitnessConfig {
            layout_digest: Some([0; 32]),
            d_pack: Some(0),
            field_rd_inc_family: false,
            trusted_advice_family: false,
            untrusted_advice_family: false,
        };
        config
    }

    fn precommitted_schedule(trusted_max_advice_bytes: Option<usize>) -> PrecommittedSchedule {
        precommitted_schedule_with_advice(trusted_max_advice_bytes, None)
    }

    fn precommitted_schedule_with_advice(
        trusted_max_advice_bytes: Option<usize>,
        untrusted_max_advice_bytes: Option<usize>,
    ) -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            2,
            8,
            trusted_max_advice_bytes,
            untrusted_max_advice_bytes,
            Some(CommittedProgramSchedule {
                bytecode_len: 16,
                bytecode_chunk_count: 2,
                program_image_len_words: 4,
                program_image_start_index: 0,
            }),
        )
        .unwrap_or_else(|error| panic!("precommitted schedule should build: {error}"))
    }

    fn ra_layout() -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(2, 1, 1)
            .unwrap_or_else(|error| panic!("RA layout should build: {error}"))
    }

    fn logical_manifest(id: JoltOpeningId, point: Vec<Fr>) -> Stage8LogicalManifest<Fr> {
        Stage8LogicalManifest {
            openings: vec![Stage8LogicalOpening {
                id: Stage8OpeningId::from(id),
                point,
                claim: Some(Fr::from_u64(2)),
                scale: Fr::from_u64(3),
            }],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
        }
    }

    #[test]
    fn derive_layout_includes_base_lattice_families() {
        let config = lattice_config();
        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(layout
            .family(&PackedFamilyId::InstructionRa { index: 0 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::RamRa { index: 0 }).is_some());
        assert!(layout
            .family(&PackedFamilyId::IncByte { index: 7 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::IncSign).is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeChunk { index: 0 })
            .is_none());
        assert!(layout.family(&PackedFamilyId::ProgramImageInit).is_some());
        assert_eq!(layout.audit().d_pack, layout.dimension);

        let mut matching_config = lattice_config();
        matching_config.lattice.packed_witness.layout_digest = Some(layout.digest);
        matching_config.lattice.packed_witness.d_pack = Some(layout.dimension);

        validate_akita_packed_witness_layout_config(&matching_config, &layout)
            .unwrap_or_else(|error| panic!("layout config should validate: {error}"));
    }

    #[test]
    fn lattice_family_ids_convert_to_akita_family_ids() {
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Trusted,
                index: 3,
            }),
            PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 3,
            }
        );
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::Custom {
                namespace: 17,
                index: 5,
            }),
            PackedFamilyId::Custom {
                namespace: 17,
                index: 5,
            }
        );
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::BytecodeRegisterSelector {
                chunk: 2,
                selector: 1,
            }),
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: 2,
                selector: 1,
            }
        );
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::BytecodeImmBytes { chunk: 2 }),
            PackedFamilyId::BytecodeImmBytes { chunk: 2 }
        );
    }

    #[test]
    fn lattice_direct_view_converts_to_akita_view_formula() {
        let formula = LatticePackedViewFormula::<Fr>::direct(LatticePackedFamilyId::IncSign, 0, 1);

        assert_eq!(
            akita_packed_view_formula(&formula)
                .unwrap_or_else(|error| panic!("direct view should convert: {error}")),
            PackedViewFormula::direct(PackedFamilyId::IncSign, 0, 1)
        );
    }

    #[test]
    fn lattice_linear_view_converts_terms_to_akita_view_formula() {
        let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
            LatticePackedFamilyId::BytecodeChunk { index: 2 },
            4,
        ));

        let converted = akita_packed_view_formula(&formula)
            .unwrap_or_else(|error| panic!("linear view should convert: {error}"));

        assert!(matches!(
            converted,
            PackedViewFormula::LinearDecoded { terms, .. }
                if terms.len() == 256
                    && terms[7].coefficient == Fr::from_u64(7)
                    && terms[7].family == (PackedFamilyId::BytecodeChunk { index: 2 })
                    && terms[7].limb == 4
                    && terms[7].symbol == 7
        ));
    }

    #[test]
    fn lattice_masked_views_require_prior_translation() {
        assert!(matches!(
            akita_packed_view_formula::<Fr>(&LatticePackedViewFormula::MaskedDecoded),
            Err(PackedViewError::MaskedViewRequiresTranslation)
        ));
        assert!(matches!(
            akita_packed_view_formula::<Fr>(&LatticePackedViewFormula::reduced_masked(
                jolt_claims::protocols::jolt::JoltRelationId::IncClaimReduction,
                Vec::new(),
            )),
            Err(PackedViewError::MaskedViewRequiresTranslation)
        ));
    }

    #[test]
    fn jolt_lattice_resolver_weights_ra_symbols_by_address_point() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(1),
            JoltRelationId::HammingWeightClaimReduction,
        );
        let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let expected_weights = EqPolynomial::<Fr>::evals(&point[..2], None);
        let formulas = jolt_lattice_view_formulas(&logical_manifest(id, point), 2)
            .unwrap_or_else(|error| panic!("RA lattice formula should resolve: {error}"));

        assert_eq!(formulas[0].0, id);
        assert!(matches!(
            &formulas[0].1,
            LatticePackedViewFormula::LinearDecoded { terms }
                if terms.len() == 4
                    && terms[2].coefficient == expected_weights[2]
                    && terms[2].family == LatticePackedFamilyId::InstructionRa { index: 1 }
                    && terms[2].limb == 0
                    && terms[2].symbol == 2
        ));
    }

    #[test]
    fn jolt_lattice_resolver_decodes_advice_and_program_image_bytes() {
        let trusted = JoltOpeningId::trusted_advice(JoltRelationId::AdviceClaimReduction);
        let program_image = JoltOpeningId::committed(
            JoltCommittedPolynomial::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        );

        let trusted_formula = jolt_lattice_view_formula(trusted, &[Fr::from_u64(1)], 8)
            .unwrap_or_else(|error| panic!("trusted advice view should resolve: {error}"));
        assert!(matches!(
            trusted_formula,
            LatticePackedViewFormula::LinearDecoded { terms }
                if terms.len() == 256
                    && terms[7].coefficient == Fr::from_u64(7)
                    && terms[7].family == (LatticePackedFamilyId::AdviceBytes {
                        kind: JoltAdviceKind::Trusted,
                        index: 0,
                    })
                    && terms[7].limb == 0
                    && terms[7].symbol == 7
        ));

        let program_image_formula = jolt_lattice_view_formula(program_image, &[Fr::from_u64(1)], 8)
            .unwrap_or_else(|error| panic!("program image view should resolve: {error}"));
        assert!(matches!(
            program_image_formula,
            LatticePackedViewFormula::LinearDecoded { terms }
                if terms.len() == 8 * 256
                    && terms[256 + 7].coefficient == Fr::from_u64(256 * 7)
                    && terms[256 + 7].family == LatticePackedFamilyId::ProgramImageInit
                    && terms[256 + 7].limb == 1
                    && terms[256 + 7].symbol == 7
        ));
    }

    #[test]
    fn jolt_lattice_resolver_marks_increments_as_masked() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        );
        assert!(matches!(
            jolt_lattice_view_formula(id, &[Fr::from_u64(1)], 8)
                .unwrap_or_else(|error| panic!("increment view should resolve: {error}")),
            LatticePackedViewFormula::MaskedDecoded
        ));
    }

    #[test]
    fn jolt_lattice_resolver_rejects_bytecode_chunks_until_lane_policy_exists() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeChunk(0),
            JoltRelationId::BytecodeClaimReduction,
        );
        assert!(matches!(
            jolt_lattice_view_formula::<Fr>(id, &[Fr::from_u64(1)], 8),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("BytecodeChunk")
        ));
    }

    #[test]
    fn jolt_lattice_physical_manifest_lowers_supported_ra_view() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(1),
            JoltRelationId::HammingWeightClaimReduction,
        );
        let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let expected_weights = EqPolynomial::<Fr>::evals(&point[..2], None);
        let logical = logical_manifest(id, point);
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index: 1 },
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Fixed { size: 4 },
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

        let physical = jolt_lattice_physical_manifest(&logical, &layout, 2)
            .unwrap_or_else(|error| panic!("physical manifest should resolve: {error}"));

        assert_eq!(physical.layout_digest, layout.digest);
        assert_eq!(physical.openings[0].id, Stage8OpeningId::from(id));
        assert!(matches!(
            &physical.openings[0].view,
            PhysicalView::PackedLinear {
                layout_digest,
                terms
            } if *layout_digest == layout.digest
                && terms.len() == 4
                && terms[2].coefficient == expected_weights[2]
                && terms[2].family == (PackedFamilyId::InstructionRa { index: 1 }).physical_ref()
                && terms[2].limb == 0
                && terms[2].symbol == 2
        ));
    }

    #[test]
    fn jolt_lattice_physical_manifest_rejects_masked_increment_view() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        );
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Bit,
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

        assert!(matches!(
            jolt_lattice_physical_manifest(&logical_manifest(id, vec![Fr::from_u64(1)]), &layout, 8),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("masked packed view")
        ));
    }

    #[test]
    fn layout_config_mismatch_rejects() {
        let layout = derive_akita_packed_witness_layout(
            &lattice_config(),
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(matches!(
            validate_akita_packed_witness_layout_config(&lattice_config(), &layout),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));
    }

    #[test]
    fn advice_layout_requires_precommitted_schedule() {
        let mut config = lattice_config();
        config.lattice.advice.trusted = true;
        config.lattice.packed_witness.trusted_advice_family = true;

        assert!(matches!(
            derive_akita_packed_witness_layout(
                &config,
                2,
                8,
                ra_layout(),
                &precommitted_schedule(None),
            ),
            Err(VerifierError::InvalidPrecommittedSchedule { .. })
        ));

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(Some(64)),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
        assert!(layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            })
            .is_some());
    }

    #[test]
    fn field_inline_layout_uses_separate_rd_inc_families() {
        let mut config = lattice_config();
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.field_rd_inc_family = true;

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(layout.family(&PackedFamilyId::IncSign).is_some());
        assert!(layout.family(&PackedFamilyId::FieldRdIncSign).is_some());
        assert!(layout
            .family(&PackedFamilyId::FieldRdIncByte { index: 7 })
            .is_some());
    }

    #[test]
    fn advice_and_committed_program_use_non_trace_domains() {
        let mut config = lattice_config();
        config.lattice.advice.trusted = true;
        config.lattice.advice.untrusted = true;
        config.lattice.packed_witness.trusted_advice_family = true;
        config.lattice.packed_witness.untrusted_advice_family = true;

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule_with_advice(Some(64), Some(128)),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        let trusted = layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            })
            .unwrap_or_else(|| panic!("trusted advice family should be present"));
        assert!(matches!(
            trusted.domain,
            PackedFactDomain::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                ..
            }
        ));

        let untrusted = layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Untrusted,
                index: 0,
            })
            .unwrap_or_else(|| panic!("untrusted advice family should be present"));
        assert!(matches!(
            untrusted.domain,
            PackedFactDomain::AdviceBytes {
                kind: PackedAdviceKind::Untrusted,
                ..
            }
        ));

        let bytecode = layout
            .family(&PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            })
            .unwrap_or_else(|| panic!("bytecode register selector family should be present"));
        assert_eq!(
            bytecode.domain,
            PackedFactDomain::BytecodeRows { log_bytecode: 3 }
        );
        assert_eq!(
            bytecode.alphabet,
            PackedAlphabet::Fixed {
                size: 1usize << REGISTER_ADDRESS_BITS,
            }
        );

        let imm = layout
            .family(&PackedFamilyId::BytecodeImmBytes { chunk: 0 })
            .unwrap_or_else(|| panic!("bytecode immediate byte family should be present"));
        assert_eq!(
            imm.domain,
            PackedFactDomain::BytecodeRows { log_bytecode: 3 }
        );
        assert_eq!(imm.limbs, AkitaField::NUM_BYTES);
        assert_eq!(imm.alphabet, PackedAlphabet::Byte);

        let program_image = layout
            .family(&PackedFamilyId::ProgramImageInit)
            .unwrap_or_else(|| panic!("program-image family should be present"));
        assert_eq!(
            program_image.domain,
            PackedFactDomain::ProgramImageWords { log_words: 2 }
        );
        assert_eq!(program_image.limbs, 8);
    }
}
