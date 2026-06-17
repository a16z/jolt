use crate::{
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    stages::PrecommittedSchedule,
    VerifierError,
};
use jolt_akita::{
    PackedAdviceKind, PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec,
    PackedViewError, PackedViewFormula, PackedViewTerm, PackedWitnessLayout,
};
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::bytecode, ra::JoltRaPolynomialLayout},
    AdviceClaimReductionLayout, JoltAdviceKind, LatticePackedFamilyId, LatticePackedViewFormula,
    ProgramImageClaimReductionLayout,
};
use jolt_field::Field;

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
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::BytecodeChunk { index },
            PackedFactDomain::BytecodeRows {
                log_bytecode: bytecode_layout.log_bytecode_chunk_size(),
            },
            bytecode::committed_lanes(),
            PackedAlphabet::Byte,
        ));
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
        LatticePackedFamilyId::ProgramImageInit => PackedFamilyId::ProgramImageInit,
        LatticePackedFamilyId::Custom { namespace, index } => PackedFamilyId::Custom {
            namespace: *namespace,
            index: *index,
        },
    }
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

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on setup errors")]

    use super::*;
    use crate::{
        config::{IncrementCommitmentMode, PackedWitnessConfig, ProgramMode},
        stages::CommittedProgramSchedule,
    };
    use jolt_claims::protocols::jolt::{
        byte_decode_terms, LatticePackedFamilyId, LatticePackedViewFormula, TracePolynomialOrder,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

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
            .family(&PackedFamilyId::BytecodeChunk { index: 0 })
            .is_some());
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
            .family(&PackedFamilyId::BytecodeChunk { index: 0 })
            .unwrap_or_else(|| panic!("bytecode chunk family should be present"));
        assert_eq!(
            bytecode.domain,
            PackedFactDomain::BytecodeRows { log_bytecode: 3 }
        );

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
