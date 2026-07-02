#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    reason = "tests fail loudly on unexpected errors"
)]

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{CircuitFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use super::super::claim_reductions::bytecode as bytecode_reduction;
use super::super::claim_reductions::precommitted::PrecommittedClaimReduction;
use super::super::dimensions::{
    CommitmentMatrixShape, JoltFormulaPointError, JoltSumcheckSpec, TraceDimensions,
    TracePolynomialOrder, REGISTER_ADDRESS_BITS,
};
use super::super::ra::JoltRaPolynomialLayout;
use super::*;
use crate::protocols::jolt::{
    AdviceClaimReductionLayout, IncVirtualizationChallenge, IncVirtualizationPublic,
    JoltAdviceKind, JoltChallengeId, JoltCommittedPolynomial, JoltPublicId, JoltRelationId,
    UnsignedIncChunkReconstructionChallenge, UnsignedIncChunkReconstructionPublic,
};

fn physical(family: JoltPackingFamilyId) -> PackingFamilyId {
    family.into()
}

#[test]
fn final_opening_lattice_requirement_marks_increments_as_logical_only() {
    assert_eq!(
        final_opening_lattice_requirement(JoltCommittedPolynomial::RamInc),
        LatticeFinalOpeningRequirement::LogicalOnly
    );
    assert_eq!(
        final_opening_lattice_requirement(JoltCommittedPolynomial::RdInc),
        LatticeFinalOpeningRequirement::LogicalOnly
    );
}

#[test]
fn final_opening_lattice_requirement_names_packed_families() {
    assert_eq!(
        final_opening_lattice_requirement(JoltCommittedPolynomial::InstructionRa(2)),
        LatticeFinalOpeningRequirement::PackingLayoutFamily {
            family: JoltPackingFamilyId::InstructionRa { index: 2 },
            relation: JoltRelationId::HammingWeightClaimReduction,
        }
    );
    assert_eq!(
        final_opening_lattice_requirement(JoltCommittedPolynomial::ProgramImageInit),
        LatticeFinalOpeningRequirement::PackingLayoutFamily {
            family: JoltPackingFamilyId::ProgramImageInit,
            relation: JoltRelationId::ProgramImageClaimReduction,
        }
    );
}

#[test]
fn inc_virtualization_claim_exposes_expected_dependencies() {
    let claims = inc_virtualization_claim::<Fr>(TraceDimensions::new(5));

    assert_eq!(claims.id, JoltRelationId::IncVirtualization);
    assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(3));
    assert_eq!(
        claims.input.required_openings,
        inc_virtualization_input_openings()
    );
    assert_eq!(
        claims.output.required_openings,
        inc_virtualization_output_openings()
    );
    assert_eq!(
        claims.required_challenges(),
        vec![JoltChallengeId::from(IncVirtualizationChallenge::Gamma)]
    );
    assert_eq!(
        claims.required_publics(),
        vec![
            JoltPublicId::from(IncVirtualizationPublic::EqRamReadWrite),
            JoltPublicId::from(IncVirtualizationPublic::EqRamValCheck),
            JoltPublicId::from(IncVirtualizationPublic::EqRegistersReadWrite),
            JoltPublicId::from(IncVirtualizationPublic::EqRegistersValEvaluation),
        ]
    );
}

#[test]
fn inc_virtualization_claim_evaluates_store_selected_inc() {
    let claims = inc_virtualization_claim::<Fr>(TraceDimensions::new(5));
    let ram_rw = Fr::from_u64(3);
    let ram_val = Fr::from_u64(5);
    let rd_rw = Fr::from_u64(7);
    let rd_val = Fr::from_u64(11);
    let inc = Fr::from_u64(13);
    let store = Fr::from_u64(0);
    let eq_ram_rw = Fr::from_u64(17);
    let eq_ram_val = Fr::from_u64(19);
    let eq_rd_rw = Fr::from_u64(23);
    let eq_rd_val = Fr::from_u64(29);
    let gamma = Fr::from_u64(31);
    let zero = Fr::from_u64(0);

    let input = claims.input.expression().evaluate(
        |id| match *id {
            id if id == inc_virtualization_ram_read_write_opening() => ram_rw,
            id if id == inc_virtualization_ram_val_check_opening() => ram_val,
            id if id == inc_virtualization_rd_read_write_opening() => rd_rw,
            id if id == inc_virtualization_rd_val_evaluation_opening() => rd_val,
            _ => zero,
        },
        |id| match id {
            JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
            _ => zero,
        },
        |_| zero,
    );
    let output = claims.output.expression().evaluate(
        |id| match *id {
            id if id == inc_virtualization_inc_opening() => inc,
            id if id == inc_virtualization_store_opening() => store,
            _ => zero,
        },
        |id| match id {
            JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
            _ => zero,
        },
        |id| match *id {
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamReadWrite) => eq_ram_rw,
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamValCheck) => eq_ram_val,
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersReadWrite) => {
                eq_rd_rw
            }
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersValEvaluation) => {
                eq_rd_val
            }
            _ => zero,
        },
    );

    let gamma_2 = gamma * gamma;
    assert_eq!(
        input,
        ram_rw + gamma * ram_val + gamma_2 * rd_rw + gamma_2 * gamma * rd_val
    );
    assert_eq!(output, inc * gamma_2 * (eq_rd_rw + gamma * eq_rd_val));

    let store_output = claims.output.expression().evaluate(
        |id| match *id {
            id if id == inc_virtualization_inc_opening() => inc,
            id if id == inc_virtualization_store_opening() => Fr::from_u64(1),
            _ => zero,
        },
        |id| match id {
            JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
            _ => zero,
        },
        |id| match *id {
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamReadWrite) => eq_ram_rw,
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamValCheck) => eq_ram_val,
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersReadWrite) => {
                eq_rd_rw
            }
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersValEvaluation) => {
                eq_rd_val
            }
            _ => zero,
        },
    );
    assert_eq!(store_output, inc * (eq_ram_rw + gamma * eq_ram_val));
}

#[test]
fn unsigned_inc_claim_reduction_offsets_inc_by_two_to_64() {
    let claims = unsigned_inc_claim_reduction_claim::<Fr>(TraceDimensions::new(5));
    let inc = Fr::from_u64(13);
    let unsigned_inc = Fr::from_u128((1u128 << 64) + 13);
    let zero = Fr::from_u64(0);

    assert_eq!(claims.id, JoltRelationId::UnsignedIncClaimReduction);
    assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(2));
    assert_eq!(
        claims.input.required_openings,
        vec![inc_virtualization_inc_opening()]
    );
    assert_eq!(
        claims.output.required_openings,
        vec![unsigned_inc_opening()]
    );
    assert!(claims.required_challenges().is_empty());
    assert!(claims.required_publics().is_empty());

    let input = claims.input.expression().evaluate(
        |id| match *id {
            id if id == inc_virtualization_inc_opening() => inc,
            _ => zero,
        },
        |_| zero,
        |_| zero,
    );
    let output = claims.output.expression().evaluate(
        |id| match *id {
            id if id == unsigned_inc_opening() => unsigned_inc,
            _ => zero,
        },
        |_| zero,
        |_| zero,
    );

    assert_eq!(input, unsigned_inc);
    assert_eq!(output, unsigned_inc);
}

#[test]
fn unsigned_inc_msb_booleanity_claim_checks_cycle_bit() {
    let claims = unsigned_inc_msb_booleanity_claim::<Fr>(TraceDimensions::new(5));
    let zero = Fr::from_u64(0);
    let msb = Fr::from_u64(7);

    assert_eq!(claims.id, JoltRelationId::Booleanity);
    assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(2));
    assert!(claims.input.required_openings.is_empty());
    assert_eq!(
        claims.output.required_openings,
        vec![unsigned_inc_msb_opening()]
    );
    assert!(claims.required_challenges().is_empty());
    assert!(claims.required_publics().is_empty());

    let output = claims.output.expression().evaluate(
        |id| match *id {
            id if id == unsigned_inc_msb_opening() => msb,
            _ => zero,
        },
        |_| zero,
        |_| zero,
    );

    assert_eq!(output, msb * msb - msb);
}

#[test]
fn unsigned_inc_chunk_reconstruction_claim_batches_hamming_point_and_value() {
    let claims =
        unsigned_inc_chunk_reconstruction_claim::<Fr>(8).expect("8-bit chunking should be valid");

    assert_eq!(claims.id, JoltRelationId::UnsignedIncChunkReconstruction);
    assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(8, 3));
    assert!(claims
        .input
        .required_openings
        .contains(&unsigned_inc_opening()));
    assert!(claims
        .input
        .required_openings
        .contains(&unsigned_inc_msb_opening()));
    assert_eq!(
        claims.output.required_openings,
        (0..8).map(unsigned_inc_chunk_opening).collect::<Vec<_>>()
    );
    assert_eq!(
        claims.required_challenges(),
        vec![JoltChallengeId::from(
            UnsignedIncChunkReconstructionChallenge::Gamma,
        )]
    );
    assert_eq!(
        claims.required_publics(),
        vec![
            JoltPublicId::from(UnsignedIncChunkReconstructionPublic::EqBooleanityAddress),
            JoltPublicId::from(UnsignedIncChunkReconstructionPublic::IdentityAtAddress),
        ]
    );
}

#[test]
fn unsigned_inc_chunking_rejects_invalid_sizes() {
    assert_eq!(unsigned_inc_lower_chunk_count(0), None);
    assert_eq!(unsigned_inc_lower_chunk_count(3), None);
    assert_eq!(unsigned_inc_lower_chunk_count(64), None);
    assert!(unsigned_inc_chunk_reconstruction_claim::<Fr>(3).is_none());
    assert!(unsigned_inc_validity_requirements(3).is_none());
    assert!(unsigned_inc_lower_value_lattice_view_formula::<Fr>(3).is_none());
    assert!(unsigned_inc_chunk_reconstruction_claim::<Fr>(64).is_none());
    assert!(unsigned_inc_validity_requirements(64).is_none());
    assert!(unsigned_inc_lower_value_lattice_view_formula::<Fr>(64).is_none());
}

#[test]
fn unsigned_inc_decode_formulas_use_configured_chunks_and_msb() {
    assert_eq!(
        unsigned_inc_msb_lattice_view_formula::<Fr>(),
        PackingViewFormula::direct(physical(JoltPackingFamilyId::UnsignedIncMsb), 0, 1)
    );

    let lower = unsigned_inc_lower_value_lattice_view_formula::<Fr>(4)
        .expect("4-bit chunking should be valid");
    let terms = linear_decoded_terms(&lower);
    assert_eq!(terms.len(), 16 * 16);
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::UnsignedIncChunk { index: 0 }),
            0,
            7
        )
        .coefficient,
        Fr::from_u64(7)
    );
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::UnsignedIncChunk { index: 1 }),
            0,
            3
        )
        .coefficient,
        Fr::from_u64(16 * 3)
    );
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::UnsignedIncChunk { index: 15 }),
            0,
            2
        )
        .coefficient,
        Fr::from_u64(1u64 << 60) * Fr::from_u64(2)
    );
}

#[test]
fn unsigned_inc_validity_requirements_cover_chunks_and_msb() {
    let requirements =
        unsigned_inc_validity_requirements(4).expect("4-bit chunking should be valid");

    assert_eq!(requirements.len(), 17);
    assert_eq!(
        requirements[0],
        PackingValidityRequirement::exact_one_hot(
            physical(JoltPackingFamilyId::UnsignedIncChunk { index: 0 }),
            1,
            16,
        )
    );
    assert_eq!(
        requirements[15],
        PackingValidityRequirement::exact_one_hot(
            physical(JoltPackingFamilyId::UnsignedIncChunk { index: 15 }),
            1,
            16,
        )
    );
    assert_eq!(
        requirements[16],
        PackingValidityRequirement::boolean_indicator(
            physical(JoltPackingFamilyId::UnsignedIncMsb),
            1,
            2,
            1,
        )
    );
}

#[test]
fn lattice_packed_witness_layout_derives_ra_and_increment_families() {
    let ra_layout = JoltRaPolynomialLayout::new(2, 1, 1).unwrap();
    let layout = derive_jolt_lattice_packed_witness_layout(JoltLatticePackingInputs {
        log_t: 5,
        log_k_chunk: 4,
        ra_layout,
        field_rd_inc: None,
        untrusted_advice: None,
    })
    .unwrap();

    let trace = PackingFactDomain::TraceRows { log_t: 5 };
    let fixed_16 = PackingAlphabet::Fixed { size: 16 };
    assert_eq!(
        layout
            .family(&physical(JoltPackingFamilyId::InstructionRa { index: 1 }))
            .unwrap()
            .domain,
        trace
    );
    assert_eq!(
        layout
            .family(&physical(JoltPackingFamilyId::InstructionRa { index: 1 }))
            .unwrap()
            .alphabet,
        fixed_16
    );
    assert!(layout
        .family(&physical(JoltPackingFamilyId::BytecodeRa { index: 0 }))
        .is_some());
    assert!(layout
        .family(&physical(JoltPackingFamilyId::RamRa { index: 0 }))
        .is_some());
    assert!(layout
        .family(&physical(JoltPackingFamilyId::UnsignedIncChunk {
            index: 15
        }))
        .is_some());
    assert!(layout
        .family(&physical(JoltPackingFamilyId::UnsignedIncMsb))
        .is_some());
}

#[test]
fn lattice_packed_witness_layout_rejects_invalid_chunking() {
    let ra_layout = JoltRaPolynomialLayout::new(1, 1, 1).unwrap();
    let err = derive_jolt_lattice_packed_witness_layout(JoltLatticePackingInputs {
        log_t: 5,
        log_k_chunk: 3,
        ra_layout,
        field_rd_inc: None,
        untrusted_advice: None,
    })
    .unwrap_err();

    assert_eq!(
        err,
        JoltLatticeLayoutError::InvalidUnsignedIncChunking { log_k_chunk: 3 }
    );
}

#[test]
fn lattice_packed_witness_layout_includes_field_and_advice_families() {
    let ra_layout = JoltRaPolynomialLayout::new(1, 1, 1).unwrap();
    let advice_layout = advice_layout(5, 4, 64);
    let layout = derive_jolt_lattice_packed_witness_layout(JoltLatticePackingInputs {
        log_t: 5,
        log_k_chunk: 4,
        ra_layout,
        field_rd_inc: Some(FieldRdIncPacking {
            byte_width: 2,
            canonical_modulus: Some(257),
        }),
        untrusted_advice: Some(&advice_layout),
    })
    .unwrap();

    assert!(layout_has_field_rd_inc(&layout));
    assert!(layout_has_advice(&layout, PackingAdviceKind::Untrusted));
    assert_eq!(
        layout
            .family(&physical(JoltPackingFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Untrusted,
                index: 0,
            }))
            .unwrap()
            .domain,
        PackingFactDomain::AdviceBytes {
            kind: PackingAdviceKind::Untrusted,
            log_bytes: 6,
        }
    );

    let requirements =
        derive_jolt_lattice_packed_validity_requirements(JoltLatticeValidityInputs {
            log_k_chunk: 4,
            field_rd_inc: Some(FieldRdIncPacking {
                byte_width: 2,
                canonical_modulus: Some(257),
            }),
            untrusted_advice: Some(&advice_layout),
        })
        .unwrap();
    assert!(
        requirements.contains(&PackingValidityRequirement::field_element_canonical_bytes(
            physical(JoltPackingFamilyId::FieldRdIncByte { index: 0 }),
            2,
            257,
        ))
    );
    assert!(requirements.contains(&advice_bytes_validity_requirement(
        JoltAdviceKind::Untrusted,
    )));
}

#[test]
fn layout_derived_validity_marks_bytecode_store_rd_disjointness() {
    let layout = PackingWitnessLayout::new([
        PackingFamilySpec::direct(
            physical(JoltPackingFamilyId::BytecodeCircuitFlag {
                chunk: 0,
                flag: CircuitFlags::Store as usize,
            }),
            PackingFactDomain::BytecodeRows { log_bytecode: 4 },
            1,
            PackingAlphabet::Bit,
        ),
        PackingFamilySpec::direct(
            physical(JoltPackingFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 2,
            }),
            PackingFactDomain::BytecodeRows { log_bytecode: 4 },
            1,
            PackingAlphabet::Fixed {
                size: 1 << REGISTER_ADDRESS_BITS,
            },
        ),
    ])
    .unwrap();

    let requirements = lattice_validity_requirements_for_packed_witness_layout(&layout)
        .expect("layout validity requirements should derive");
    assert!(
        requirements.contains(&PackingValidityRequirement::bytecode_store_rd_disjoint(
            physical(JoltPackingFamilyId::BytecodeCircuitFlag {
                chunk: 0,
                flag: CircuitFlags::Store as usize,
            })
        ))
    );
}

#[test]
fn precommitted_family_classification_keeps_untrusted_advice_packable() {
    assert!(packed_family_is_precommitted(&physical(
        JoltPackingFamilyId::BytecodeChunk { index: 0 }
    )));
    assert!(packed_family_is_precommitted(&physical(
        JoltPackingFamilyId::AdviceBytes {
            kind: JoltAdviceKind::Trusted,
            index: 0,
        }
    )));
    assert!(!packed_family_is_precommitted(&physical(
        JoltPackingFamilyId::AdviceBytes {
            kind: JoltAdviceKind::Untrusted,
            index: 0,
        }
    )));
}

#[test]
fn base_increment_validity_requirements_do_not_use_field_inline_families() {
    let requirements =
        unsigned_inc_validity_requirements(8).expect("8-bit chunking should be valid");

    assert!(requirements.iter().all(|requirement| {
        !matches!(
            JoltPackingFamilyId::from_physical_id(&requirement.family),
            Some(JoltPackingFamilyId::FieldRdIncByte { .. } | JoltPackingFamilyId::FieldRdIncSign)
        )
    }));
}

#[test]
fn packed_validity_digest_is_order_stable_and_kind_sensitive() {
    let exact = PackingValidityRequirement::exact_one_hot(
        physical(JoltPackingFamilyId::UnsignedIncChunk { index: 0 }),
        1,
        256,
    );
    let msb = PackingValidityRequirement::boolean_indicator(
        physical(JoltPackingFamilyId::UnsignedIncMsb),
        1,
        2,
        1,
    );
    let optional = PackingValidityRequirement::optional_one_hot(
        physical(JoltPackingFamilyId::UnsignedIncChunk { index: 0 }),
        1,
        256,
    );

    assert_eq!(
        packing_validity_digest(&[exact.clone(), msb.clone()]),
        packing_validity_digest(&[msb, exact.clone()])
    );
    assert_ne!(
        packing_validity_digest(&[exact]),
        packing_validity_digest(&[optional])
    );
}

#[test]
fn bytecode_validity_requirements_cover_committed_program_facts() {
    let chunk = 2;
    let field_byte_width = 16;
    let requirements = bytecode_validity_requirements(chunk, field_byte_width);

    assert_eq!(
        requirements.len(),
        3 + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTION_FLAGS + 5
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::optional_one_hot(
            physical(JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 }),
            1,
            1 << REGISTER_ADDRESS_BITS,
        ))
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::boolean_indicator(
            physical(JoltPackingFamilyId::BytecodeCircuitFlag { chunk, flag: 0 }),
            1,
            2,
            1,
        ))
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::boolean_indicator(
            physical(JoltPackingFamilyId::BytecodeInstructionFlag {
                chunk,
                flag: NUM_INSTRUCTION_FLAGS - 1,
            }),
            1,
            2,
            1,
        ))
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::optional_one_hot(
            physical(JoltPackingFamilyId::BytecodeLookupSelector { chunk }),
            1,
            LookupTableKind::<XLEN>::COUNT.next_power_of_two(),
        ))
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::boolean_indicator(
            physical(JoltPackingFamilyId::BytecodeRafFlag { chunk }),
            1,
            2,
            1,
        ))
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::exact_one_hot(
            physical(JoltPackingFamilyId::BytecodeUnexpandedPcBytes { chunk }),
            8,
            256,
        ))
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::exact_one_hot(
            physical(JoltPackingFamilyId::BytecodeImmBytes { chunk }),
            field_byte_width,
            256,
        ))
    );
    assert!(
        requirements.contains(&PackingValidityRequirement::bytecode_store_rd_disjoint(
            physical(JoltPackingFamilyId::BytecodeCircuitFlag {
                chunk,
                flag: CircuitFlags::Store as usize,
            })
        ))
    );
}

#[test]
fn bytecode_imm_canonical_bytes_requirement_anchors_bytecode_immediates() {
    assert_eq!(
        bytecode_imm_canonical_bytes_requirement(2, 16, 97),
        PackingValidityRequirement::field_element_canonical_bytes(
            physical(JoltPackingFamilyId::BytecodeImmBytes { chunk: 2 }),
            16,
            97,
        )
    );
}

#[test]
fn advice_and_program_image_validity_requirements_are_byte_facts() {
    assert_eq!(
        advice_bytes_validity_requirement(JoltAdviceKind::Trusted),
        PackingValidityRequirement::exact_one_hot(
            physical(JoltPackingFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Trusted,
                index: 0,
            }),
            1,
            256,
        )
    );
    assert_eq!(
        program_image_validity_requirement(),
        PackingValidityRequirement::exact_one_hot(
            physical(JoltPackingFamilyId::ProgramImageInit),
            8,
            256,
        )
    );
}

#[test]
fn byte_decode_terms_are_little_endian_symbol_weights() {
    let terms =
        byte_decode_terms::<Fr>(physical(JoltPackingFamilyId::BytecodeChunk { index: 0 }), 3);

    assert_eq!(terms.len(), 256);
    assert_eq!(terms[7].coefficient, Fr::from_u64(7));
    assert_eq!(
        terms[7].family,
        physical(JoltPackingFamilyId::BytecodeChunk { index: 0 })
    );
    assert_eq!(terms[7].limb, 3);
    assert_eq!(terms[7].symbol, 7);
}

#[test]
fn committed_bytecode_lattice_family_ids_name_lane_classes() {
    assert_ne!(
        physical(JoltPackingFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 0,
        }),
        physical(JoltPackingFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 1,
        })
    );
    assert_ne!(
        physical(JoltPackingFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 }),
        physical(JoltPackingFamilyId::BytecodeImmBytes { chunk: 0 })
    );
    assert_ne!(
        physical(JoltPackingFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 }),
        physical(JoltPackingFamilyId::BytecodeInstructionFlag { chunk: 0, flag: 0 })
    );
    let circuit_flag = JoltPackingFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 1 };
    let other_circuit_flag = JoltPackingFamilyId::BytecodeCircuitFlag { chunk: 1, flag: 0 };
    assert_ne!(physical(circuit_flag), physical(other_circuit_flag));
    assert_eq!(
        JoltPackingFamilyId::from_physical_id(&physical(circuit_flag)),
        Some(circuit_flag)
    );
    assert_eq!(
        JoltPackingFamilyId::from_physical_id(&physical(other_circuit_flag)),
        Some(other_circuit_flag)
    );
}

#[test]
fn layout_derived_validity_rejects_non_jolt_family_ids() {
    let layout = PackingWitnessLayout::new([PackingFamilySpec::direct(
        PackingFamilyId::new(0x7465_7374, 1, 0),
        PackingFactDomain::TraceRows { log_t: 0 },
        1,
        PackingAlphabet::Bit,
    )])
    .unwrap();

    assert!(matches!(
        lattice_validity_requirements_for_packed_witness_layout(&layout),
        Err(JoltLatticeLayoutError::UnsupportedPackingFamily { .. })
    ));
}

#[test]
fn bytecode_chunk_lattice_view_formula_uses_cycle_major_lane_weights() {
    let lane_vars = bytecode_reduction::committed_lane_vars();
    let log_bytecode = 2;
    let lane_point = (1..=lane_vars as u64).map(Fr::from_u64).collect::<Vec<_>>();
    let mut opening_point = lane_point.clone();
    opening_point.extend([Fr::from_u64(101), Fr::from_u64(103)]);
    let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);
    let lane_layout = bytecode_reduction::BYTECODE_LANE_LAYOUT;

    let formula = bytecode_chunk_lattice_view_formula(
        2,
        &opening_point,
        TracePolynomialOrder::CycleMajor,
        log_bytecode,
        2,
    )
    .unwrap();
    let terms = linear_decoded_terms(&formula);

    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::BytecodeRegisterSelector {
                chunk: 2,
                selector: 2
            }),
            0,
            5
        )
        .coefficient,
        lane_weights[lane_layout.rd_start + 5]
    );
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::BytecodeCircuitFlag { chunk: 2, flag: 0 }),
            0,
            1
        )
        .coefficient,
        lane_weights[lane_layout.circuit_start]
    );
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::BytecodeLookupSelector { chunk: 2 }),
            0,
            3
        )
        .coefficient,
        lane_weights[lane_layout.lookup_start + 3]
    );
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::BytecodeUnexpandedPcBytes { chunk: 2 }),
            1,
            7
        )
        .coefficient,
        lane_weights[lane_layout.unexp_pc_idx] * Fr::from_u64(256 * 7)
    );
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::BytecodeImmBytes { chunk: 2 }),
            1,
            9
        )
        .coefficient,
        lane_weights[lane_layout.imm_idx] * Fr::from_u64(256 * 9)
    );
    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::BytecodeRafFlag { chunk: 2 }),
            0,
            1
        )
        .coefficient,
        lane_weights[lane_layout.raf_flag_idx]
    );
}

#[test]
fn bytecode_chunk_lattice_view_formula_uses_address_major_lane_suffix() {
    let lane_vars = bytecode_reduction::committed_lane_vars();
    let log_bytecode = 2;
    let lane_point = (11..11 + lane_vars as u64)
        .map(Fr::from_u64)
        .collect::<Vec<_>>();
    let mut opening_point = vec![Fr::from_u64(101), Fr::from_u64(103)];
    opening_point.extend(lane_point.iter().copied());
    let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);

    let formula = bytecode_chunk_lattice_view_formula(
        1,
        &opening_point,
        TracePolynomialOrder::AddressMajor,
        log_bytecode,
        1,
    )
    .unwrap();
    let terms = linear_decoded_terms(&formula);

    assert_eq!(
        find_term(
            terms,
            physical(JoltPackingFamilyId::BytecodeRegisterSelector {
                chunk: 1,
                selector: 0
            }),
            0,
            1
        )
        .coefficient,
        lane_weights[bytecode_reduction::BYTECODE_LANE_LAYOUT.rs1_start + 1]
    );
}

#[test]
fn bytecode_chunk_lattice_view_formula_rejects_bad_point_length() {
    let expected = bytecode_reduction::committed_lane_vars() + 3;

    let err = bytecode_chunk_lattice_view_formula::<Fr>(
        0,
        &vec![Fr::from_u64(0); expected - 1],
        TracePolynomialOrder::CycleMajor,
        3,
        1,
    )
    .unwrap_err();

    assert_eq!(
        err,
        JoltFormulaPointError::OpeningPointLengthMismatch {
            expected,
            got: expected - 1
        }
    );
}

#[test]
fn symbol_decode_terms_support_non_byte_alphabets() {
    let family = physical(JoltPackingFamilyId::RamRa { index: 1 });
    let terms = symbol_decode_terms::<Fr>(family, 0, 4);

    assert_eq!(terms.len(), 4);
    assert_eq!(terms[3].coefficient, Fr::from_u64(3));
    assert_eq!(terms[3].family, family);
    assert_eq!(terms[3].limb, 0);
    assert_eq!(terms[3].symbol, 3);
}

#[test]
fn weighted_symbol_terms_use_supplied_coefficients() {
    let terms = weighted_symbol_terms(
        physical(JoltPackingFamilyId::InstructionRa { index: 0 }),
        2,
        [Fr::from_u64(11), Fr::from_u64(13), Fr::from_u64(17)],
    );

    assert_eq!(terms.len(), 3);
    assert_eq!(terms[1].coefficient, Fr::from_u64(13));
    assert_eq!(
        terms[1].family,
        physical(JoltPackingFamilyId::InstructionRa { index: 0 })
    );
    assert_eq!(terms[1].limb, 2);
    assert_eq!(terms[1].symbol, 1);
}

#[test]
fn weighted_byte_decode_terms_scale_symbols_by_limb_weights() {
    let terms = weighted_byte_decode_terms(
        physical(JoltPackingFamilyId::BytecodeChunk { index: 2 }),
        [(3, Fr::from_u64(5)), (8, Fr::from_u64(7))],
    );

    assert_eq!(terms.len(), 512);
    assert_eq!(terms[9].coefficient, Fr::from_u64(45));
    assert_eq!(terms[9].limb, 3);
    assert_eq!(terms[9].symbol, 9);
    assert_eq!(terms[256 + 9].coefficient, Fr::from_u64(63));
    assert_eq!(terms[256 + 9].limb, 8);
    assert_eq!(
        terms[256 + 9].family,
        physical(JoltPackingFamilyId::BytecodeChunk { index: 2 })
    );
}

#[test]
fn little_endian_byte_decode_terms_weight_limbs_by_place_value() {
    let terms =
        little_endian_byte_decode_terms::<Fr>(physical(JoltPackingFamilyId::ProgramImageInit), 2);

    assert_eq!(terms.len(), 512);
    assert_eq!(terms[7].coefficient, Fr::from_u64(7));
    assert_eq!(terms[7].limb, 0);
    assert_eq!(terms[7].symbol, 7);
    assert_eq!(terms[256 + 7].coefficient, Fr::from_u64(256 * 7));
    assert_eq!(terms[256 + 7].limb, 1);
    assert_eq!(terms[256 + 7].symbol, 7);
    assert_eq!(
        terms[256 + 7].family,
        physical(JoltPackingFamilyId::ProgramImageInit)
    );
}

fn advice_layout(
    log_t: usize,
    log_k_chunk: usize,
    max_advice_size_bytes: usize,
) -> AdviceClaimReductionLayout {
    let advice_vars =
        CommitmentMatrixShape::advice_from_max_bytes(max_advice_size_bytes).total_vars();
    let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
        log_t + log_k_chunk,
        &[advice_vars],
        log_k_chunk,
    );
    AdviceClaimReductionLayout::balanced(
        TracePolynomialOrder::CycleMajor,
        log_t,
        scheduling_reference,
        max_advice_size_bytes,
    )
    .unwrap()
}

fn linear_decoded_terms(formula: &PackingViewFormula<Fr>) -> &[PackingViewTerm<Fr>] {
    match formula {
        PackingViewFormula::LinearDecoded { terms, .. } => terms,
        _ => panic!("expected linear decoded formula"),
    }
}

fn find_term(
    terms: &[PackingViewTerm<Fr>],
    family: PackingFamilyId,
    limb: usize,
    symbol: usize,
) -> &PackingViewTerm<Fr> {
    terms
        .iter()
        .find(|term| term.family == family && term.limb == limb && term.symbol == symbol)
        .unwrap()
}
