
#![expect(clippy::panic, reason = "tests fail loudly on setup errors")]

use super::super::outputs::{Stage8LogicalManifest, Stage8LogicalOpening, Stage8OpeningId};
use super::*;
use crate::{
    config::{IncrementCommitmentMode, PackedWitnessConfig, ProgramMode},
    stages::CommittedProgramSchedule,
};
use jolt_akita::AKITA_FIELD_MODULUS;
use jolt_claims::protocols::jolt::formulas::claim_reductions::bytecode;
use jolt_claims::protocols::jolt::{
    byte_decode_terms, bytecode_imm_canonical_bytes_requirement, bytecode_validity_requirements,
    formulas::dimensions::REGISTER_ADDRESS_BITS, program_image_validity_requirement,
    unsigned_inc_chunk_opening, unsigned_inc_lower_chunk_count,
    unsigned_inc_lower_value_lattice_view_formula, JoltCommittedPolynomial, JoltOpeningId,
    JoltRelationId, LatticePackedFamilyId, LatticePackedViewFormula, LatticePackedViewTerm,
    TracePolynomialOrder,
};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{PackedLinearTerm, PhysicalView};
use jolt_poly::{EqPolynomial, Point};
use jolt_riscv::CircuitFlags;
use jolt_sumcheck::{BatchedEvaluationClaim, EvaluationClaim};

fn lattice_config() -> JoltProtocolConfig {
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice.program_mode = ProgramMode::Committed;
    config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
    config.lattice.packed_witness = PackedWitnessConfig {
        layout_digest: Some([0; 32]),
        d_pack: Some(0),
        validity_digest: Some([0; 32]),
    };
    #[cfg(feature = "field-inline")]
    {
        config.lattice.field_inline.enabled = true;
    }
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

fn precommitted_schedule_without_committed_program() -> PrecommittedSchedule {
    PrecommittedSchedule::new(TracePolynomialOrder::CycleMajor, 2, 8, None, None, None)
        .unwrap_or_else(|error| panic!("precommitted schedule should build: {error}"))
}

fn ra_layout() -> JoltRaPolynomialLayout {
    JoltRaPolynomialLayout::new(2, 1, 1)
        .unwrap_or_else(|error| panic!("RA layout should build: {error}"))
}

fn logical_manifest_for_stage8(id: Stage8OpeningId, point: Vec<Fr>) -> Stage8LogicalManifest<Fr> {
    Stage8LogicalManifest {
        openings: vec![Stage8LogicalOpening {
            id,
            point,
            claim: Some(Fr::from_u64(2)),
            scale: Fr::from_u64(3),
        }],
        pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
    }
}

fn logical_manifest(id: JoltOpeningId, point: Vec<Fr>) -> Stage8LogicalManifest<Fr> {
    logical_manifest_for_stage8(Stage8OpeningId::from(id), point)
}

fn bytecode_chunk_opening_point() -> (Vec<Fr>, Vec<Fr>) {
    let lane_vars = bytecode::committed_lane_vars();
    let lane_point = (1..=lane_vars as u64).map(Fr::from_u64).collect::<Vec<_>>();
    let mut point = lane_point.clone();
    point.extend([Fr::from_u64(101), Fr::from_u64(103), Fr::from_u64(107)]);
    let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);
    (point, lane_weights)
}

fn linear_decoded_terms(formula: &LatticePackedViewFormula<Fr>) -> &[LatticePackedViewTerm<Fr>] {
    match formula {
        LatticePackedViewFormula::LinearDecoded { terms } => terms,
        _ => panic!("expected linear decoded formula"),
    }
}

fn find_lattice_term(
    terms: &[LatticePackedViewTerm<Fr>],
    family: LatticePackedFamilyId,
    limb: usize,
    symbol: usize,
) -> &LatticePackedViewTerm<Fr> {
    terms
        .iter()
        .find(|term| term.family == family && term.limb == limb && term.symbol == symbol)
        .unwrap_or_else(|| panic!("missing lattice term"))
}

fn find_physical_term(
    terms: &[PackedLinearTerm<Fr>],
    family: PackedFamilyId,
    limb: usize,
    symbol: usize,
) -> &PackedLinearTerm<Fr> {
    let family = family.physical_ref();
    terms
        .iter()
        .find(|term| term.family == family && term.limb == limb && term.symbol == symbol)
        .unwrap_or_else(|| panic!("missing physical term"))
}

fn unsigned_increment_validity_layout(log_t: usize, log_k_chunk: usize) -> PackedWitnessLayout {
    let trace = PackedFactDomain::TraceRows { log_t };
    let requirements = unsigned_inc_validity_requirements(log_k_chunk)
        .unwrap_or_else(|| panic!("unsigned increment requirements should derive"));
    let specs = requirements
        .into_iter()
        .map(|index| {
            let alphabet = packed_alphabet_with_size(index.alphabet_size)
                .unwrap_or_else(|error| panic!("packed alphabet should derive: {error}"));
            PackedFamilySpec::direct(
                lattice_packing_family_id(&index.family),
                trace,
                index.limbs,
                alphabet,
            )
        })
        .collect::<Vec<_>>();
    PackedWitnessLayout::new(specs)
        .unwrap_or_else(|error| panic!("unsigned increment layout should build: {error}"))
}

fn bytecode_source_validity_layout(chunk: usize, log_bytecode: usize) -> PackedWitnessLayout {
    let domain = PackedFactDomain::BytecodeRows { log_bytecode };
    PackedWitnessLayout::new([
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeCircuitFlag {
                chunk,
                flag: CircuitFlags::Store as usize,
            },
            domain,
            1,
            PackedAlphabet::Bit,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
            domain,
            1,
            PackedAlphabet::Fixed {
                size: 1 << REGISTER_ADDRESS_BITS,
            },
        ),
    ])
    .unwrap_or_else(|error| panic!("bytecode source layout should build: {error}"))
}

#[test]
fn derive_layout_includes_base_lattice_families() {
    let config = lattice_config();
    let layout = derive_lattice_packed_witness_layout(
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
        .family(&PackedFamilyId::UnsignedIncChunk { index: 7 })
        .is_some());
    assert!(layout.family(&PackedFamilyId::UnsignedIncMsb).is_some());
    assert!(layout
        .family(&PackedFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 0,
        })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeLookupSelector { chunk: 0 })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeImmBytes { chunk: 0 })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeChunk { index: 0 })
        .is_none());
    assert!(layout.family(&PackedFamilyId::ProgramImageInit).is_none());
    assert_eq!(layout.audit().d_pack, layout.dimension);

    let mut matching_config = lattice_config();
    matching_config.lattice.packed_witness.layout_digest = Some(layout.digest);
    matching_config.lattice.packed_witness.d_pack = Some(layout.dimension);

    validate_lattice_packed_witness_layout_config(&matching_config, &layout)
        .unwrap_or_else(|error| panic!("layout config should validate: {error}"));
}

#[test]
fn validate_validity_config_rejects_mismatched_digest() {
    let mut config = lattice_config();
    let schedule = precommitted_schedule(None);
    let requirements = derive_lattice_packed_validity_requirements(&config, 8, &schedule)
        .unwrap_or_else(|error| panic!("validity requirements should derive: {error}"));
    let digest = lattice_packed_validity_digest(&requirements);
    config.lattice.packed_witness.validity_digest = Some(digest);

    validate_lattice_packed_witness_validity_config(&config, 8, &schedule).unwrap_or_else(
        |error| panic!("validity config should match derived requirements: {error}"),
    );

    let mut wrong_digest = digest;
    wrong_digest[0] ^= 1;
    config.lattice.packed_witness.validity_digest = Some(wrong_digest);

    assert!(matches!(
        validate_lattice_packed_witness_validity_config(&config, 8, &schedule),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("validity digest")
    ));
}

#[test]
fn derive_validity_statements_matches_requirement_semantics() {
    let layout = PackedWitnessLayout::new([
        PackedFamilySpec::direct(
            PackedFamilyId::ProgramImageInit,
            PackedFactDomain::ProgramImageWords { log_words: 2 },
            8,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeLookupSelector { chunk: 0 },
            PackedFactDomain::BytecodeRows { log_bytecode: 3 },
            1,
            PackedAlphabet::Fixed { size: 8 },
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::UnsignedIncMsb,
            PackedFactDomain::TraceRows { log_t: 4 },
            1,
            PackedAlphabet::Bit,
        ),
    ])
    .unwrap_or_else(|error| panic!("layout should build: {error}"));
    let requirements = vec![
        LatticePackedValidityRequirement::exact_one_hot(
            LatticePackedFamilyId::ProgramImageInit,
            8,
            256,
        ),
        LatticePackedValidityRequirement::optional_one_hot(
            LatticePackedFamilyId::BytecodeLookupSelector { chunk: 0 },
            1,
            8,
        ),
        LatticePackedValidityRequirement::boolean_indicator(
            LatticePackedFamilyId::UnsignedIncMsb,
            1,
            2,
            1,
        ),
    ];

    let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
        .unwrap_or_else(|error| panic!("validity statements should derive: {error}"));

    assert_eq!(statements.len(), 5);
    assert_eq!(
        statements[0].kind,
        LatticePackedValidityStatementKind::CellBooleanity
    );
    assert_eq!(statements[0].num_vars, 2 + 3 + 8);
    assert_eq!(
        statements[1].kind,
        LatticePackedValidityStatementKind::ExactOneHotRowSum
    );
    assert_eq!(statements[1].num_vars, 2 + 3);
    assert_eq!(
        statements[2].kind,
        LatticePackedValidityStatementKind::CellBooleanity
    );
    assert_eq!(statements[2].num_vars, 3 + 3);
    assert_eq!(
        statements[3].kind,
        LatticePackedValidityStatementKind::OptionalOneHotRowSum
    );
    assert_eq!(statements[3].num_vars, 3);
    assert_eq!(
        statements[4].kind,
        LatticePackedValidityStatementKind::BooleanIndicator
    );
    assert_eq!(statements[4].num_vars, 4);
    assert!(statements.iter().all(|statement| statement.degree == 3));
}

#[test]
fn derive_validity_statements_adds_unsigned_increment_chunk_and_msb_checks() {
    let layout = unsigned_increment_validity_layout(4, 4);
    let requirements = unsigned_inc_validity_requirements(4)
        .unwrap_or_else(|| panic!("unsigned increment requirements should derive"));

    let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
        .unwrap_or_else(|error| {
            panic!("unsigned increment validity statements should derive: {error}")
        });

    let chunk_count =
        unsigned_inc_lower_chunk_count(4).unwrap_or_else(|| panic!("chunk count should derive"));
    assert_eq!(requirements.len(), chunk_count + 1);
    assert_eq!(statements.len(), chunk_count * 2 + 1);
    assert_eq!(
        statements[0].kind,
        LatticePackedValidityStatementKind::CellBooleanity
    );
    assert_eq!(
        statements[1].kind,
        LatticePackedValidityStatementKind::ExactOneHotRowSum
    );
    assert_eq!(
        statements.last().map(|statement| statement.kind),
        Some(LatticePackedValidityStatementKind::BooleanIndicator)
    );
    assert!(statements.iter().all(|statement| statement.degree == 3));
}

#[test]
fn derive_validity_statements_adds_bytecode_store_rd_disjointness() {
    let chunk = 2;
    let layout = bytecode_source_validity_layout(chunk, 5);
    let requirement = LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk);

    let statements =
        derive_lattice_packed_validity_statements(&layout, std::slice::from_ref(&requirement))
            .unwrap_or_else(|error| {
                panic!("Store/Rd disjointness validity statement should derive: {error}")
            });

    assert_eq!(statements.len(), 1);
    assert_eq!(statements[0].requirement, requirement);
    assert_eq!(
        statements[0].kind,
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
    );
    assert_eq!(statements[0].num_vars, 5);
    assert_eq!(statements[0].degree, 3);
    assert_eq!(lattice_packed_validity_opening_count(&statements), 2);
}

#[test]
fn derive_validity_statements_adds_field_element_canonical_bytes() {
    let layout = PackedWitnessLayout::new((0..2).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::FieldRdIncByte { index },
            PackedFactDomain::TraceRows { log_t: 3 },
            1,
            PackedAlphabet::Byte,
        )
    }))
    .unwrap_or_else(|error| panic!("layout should build: {error}"));
    let requirement = LatticePackedValidityRequirement::field_element_canonical_bytes(
        LatticePackedFamilyId::FieldRdIncByte { index: 0 },
        2,
        257,
    );

    let statements =
        derive_lattice_packed_validity_statements(&layout, std::slice::from_ref(&requirement))
            .unwrap_or_else(|error| {
                panic!("field canonical-byte validity statement should derive: {error}")
            });

    assert_eq!(statements.len(), 1);
    assert_eq!(statements[0].requirement, requirement);
    assert_eq!(
        statements[0].kind,
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes
    );
    assert_eq!(statements[0].num_vars, 3);
    assert_eq!(statements[0].degree, 2);
    assert_eq!(lattice_packed_validity_opening_count(&statements), 3);
}

#[test]
fn derive_validity_statements_adds_bytecode_imm_canonical_bytes() {
    let chunk = 2;
    let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::BytecodeImmBytes { chunk },
        PackedFactDomain::BytecodeRows { log_bytecode: 3 },
        2,
        PackedAlphabet::Byte,
    )])
    .unwrap_or_else(|error| panic!("layout should build: {error}"));
    let requirement = bytecode_imm_canonical_bytes_requirement(chunk, 2, 257);

    let statements =
        derive_lattice_packed_validity_statements(&layout, std::slice::from_ref(&requirement))
            .unwrap_or_else(|error| {
                panic!("bytecode imm canonical-byte validity statement should derive: {error}")
            });

    assert_eq!(statements.len(), 1);
    assert_eq!(statements[0].requirement, requirement);
    assert_eq!(
        statements[0].kind,
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes
    );
    assert_eq!(statements[0].num_vars, 3);
    assert_eq!(statements[0].degree, 2);
    assert_eq!(lattice_packed_validity_opening_count(&statements), 3);
}

#[test]
fn validity_batch_builder_lowers_cell_booleanity_to_packed_terms() {
    let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::ProgramImageInit,
        PackedFactDomain::ProgramImageWords { log_words: 1 },
        2,
        PackedAlphabet::Fixed { size: 4 },
    )])
    .unwrap_or_else(|error| panic!("layout should build: {error}"));
    let requirement = LatticePackedValidityRequirement::exact_one_hot(
        LatticePackedFamilyId::ProgramImageInit,
        2,
        4,
    );
    let statement = LatticePackedValidityStatement {
        requirement,
        kind: LatticePackedValidityStatementKind::CellBooleanity,
        num_vars: 4,
        degree: 3,
    };
    let point = vec![
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(7),
    ];
    let eq_point = vec![
        Fr::from_u64(11),
        Fr::from_u64(13),
        Fr::from_u64(17),
        Fr::from_u64(19),
    ];
    let batching_coefficient = Fr::from_u64(23);
    let opening_claim = Fr::from_u64(29);
    let reduction = BatchedEvaluationClaim {
        reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(31)),
        batching_coefficients: vec![batching_coefficient],
        max_num_vars: 4,
        max_degree: 3,
    };

    let batch = build_lattice_packed_validity_batch(
        &layout,
        std::slice::from_ref(&statement),
        99_u64,
        std::slice::from_ref(&eq_point),
        &reduction,
        &[opening_claim],
    )
    .unwrap_or_else(|error| panic!("validity batch should build: {error}"));

    let expected_eq = try_eq_mle(&point, &eq_point)
        .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
    assert_eq!(
        batch.expected_final_claim,
        batching_coefficient * expected_eq * opening_claim * (opening_claim - Fr::from_u64(1))
    );
    assert_eq!(batch.statement.claims.len(), 1);
    assert_eq!(batch.statement.claims[0].commitment, 99);
    assert_eq!(batch.statement.claims[0].claim, opening_claim);

    let PhysicalView::PackedLinear {
        layout_digest,
        terms,
    } = &batch.statement.claims[0].view
    else {
        panic!("validity opening should use a packed linear view");
    };
    assert_eq!(layout_digest, &layout.digest);
    assert_eq!(terms.len(), 8);
    let limb_weights = EqPolynomial::<Fr>::evals(&point[1..2], None);
    let symbol_weights = EqPolynomial::<Fr>::evals(&point[2..], None);
    let term = find_physical_term(terms, PackedFamilyId::ProgramImageInit, 1, 3);
    assert_eq!(term.row_point, vec![Fr::from_u64(2)]);
    assert_eq!(term.coefficient, limb_weights[1] * symbol_weights[3]);
}

#[test]
fn validity_batch_builder_lowers_bytecode_store_rd_disjointness_factors() {
    let chunk = 2;
    let layout = bytecode_source_validity_layout(chunk, 3);
    let statement = LatticePackedValidityStatement {
        requirement: LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk),
        kind: LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint,
        num_vars: 3,
        degree: 3,
    };
    let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
    let eq_point = vec![Fr::from_u64(11), Fr::from_u64(13), Fr::from_u64(17)];
    let batching_coefficient = Fr::from_u64(19);
    let opening_claims = [Fr::from_u64(23), Fr::from_u64(29)];
    let reduction = BatchedEvaluationClaim {
        reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(31)),
        batching_coefficients: vec![batching_coefficient],
        max_num_vars: 3,
        max_degree: 3,
    };

    let batch = build_lattice_packed_validity_batch(
        &layout,
        std::slice::from_ref(&statement),
        99_u64,
        std::slice::from_ref(&eq_point),
        &reduction,
        &opening_claims,
    )
    .unwrap_or_else(|error| panic!("Store/Rd disjointness batch should build: {error}"));

    let expected_eq = try_eq_mle(&point, &eq_point)
        .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
    assert_eq!(
        batch.expected_final_claim,
        batching_coefficient * expected_eq * opening_claims[0] * opening_claims[1]
    );
    assert_eq!(batch.statement.claims.len(), 2);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[0].view else {
        panic!("Store factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 1);
    assert_eq!(
        terms[0].family,
        PackedFamilyId::BytecodeCircuitFlag {
            chunk,
            flag: CircuitFlags::Store as usize
        }
        .physical_ref()
    );
    assert_eq!(terms[0].symbol, 1);
    assert_eq!(terms[0].row_point, point);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[1].view else {
        panic!("Rd-present factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 1 << REGISTER_ADDRESS_BITS);
    let term = find_physical_term(
        terms,
        PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
        0,
        (1 << REGISTER_ADDRESS_BITS) - 1,
    );
    assert_eq!(term.coefficient, Fr::from_u64(1));
    assert_eq!(term.row_point, point);
}

#[test]
fn validity_batch_builder_lowers_field_element_canonical_byte_factors() {
    let layout = PackedWitnessLayout::new((0..2).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::FieldRdIncByte { index },
            PackedFactDomain::TraceRows { log_t: 2 },
            1,
            PackedAlphabet::Byte,
        )
    }))
    .unwrap_or_else(|error| panic!("layout should build: {error}"));
    let requirement = LatticePackedValidityRequirement::field_element_canonical_bytes(
        LatticePackedFamilyId::FieldRdIncByte { index: 0 },
        2,
        257,
    );
    let statement = LatticePackedValidityStatement {
        requirement,
        kind: LatticePackedValidityStatementKind::FieldElementCanonicalBytes,
        num_vars: 2,
        degree: 2,
    };
    let point = vec![Fr::from_u64(2), Fr::from_u64(3)];
    let eq_point = vec![Fr::from_u64(5), Fr::from_u64(7)];
    let batching_coefficient = Fr::from_u64(11);
    let opening_claims = [Fr::from_u64(13), Fr::from_u64(17), Fr::from_u64(19)];
    let reduction = BatchedEvaluationClaim {
        reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(23)),
        batching_coefficients: vec![batching_coefficient],
        max_num_vars: 2,
        max_degree: 2,
    };

    let batch = build_lattice_packed_validity_batch(
        &layout,
        std::slice::from_ref(&statement),
        99_u64,
        std::slice::from_ref(&eq_point),
        &reduction,
        &opening_claims,
    )
    .unwrap_or_else(|error| panic!("field canonical-byte batch should build: {error}"));

    let expected_eq = try_eq_mle(&point, &eq_point)
        .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
    let expected_invalid_indicator = opening_claims[0] + opening_claims[1] * opening_claims[2];
    assert_eq!(
        batch.expected_final_claim,
        batching_coefficient * expected_eq * expected_invalid_indicator
    );
    assert_eq!(batch.statement.claims.len(), 3);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[0].view else {
        panic!("high-byte range factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 254);
    assert_eq!(
        terms[0].family,
        PackedFamilyId::FieldRdIncByte { index: 1 }.physical_ref()
    );
    assert_eq!(terms[0].symbol, 2);
    assert_eq!(terms[253].symbol, 255);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[1].view else {
        panic!("high-byte equality factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 1);
    assert_eq!(
        terms[0].family,
        PackedFamilyId::FieldRdIncByte { index: 1 }.physical_ref()
    );
    assert_eq!(terms[0].symbol, 1);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[2].view else {
        panic!("low-byte range factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 255);
    assert_eq!(
        terms[0].family,
        PackedFamilyId::FieldRdIncByte { index: 0 }.physical_ref()
    );
    assert_eq!(terms[0].symbol, 1);
    assert_eq!(terms[254].symbol, 255);
}

#[test]
fn validity_batch_builder_lowers_bytecode_imm_canonical_byte_factors() {
    let chunk = 2;
    let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::BytecodeImmBytes { chunk },
        PackedFactDomain::BytecodeRows { log_bytecode: 2 },
        2,
        PackedAlphabet::Byte,
    )])
    .unwrap_or_else(|error| panic!("layout should build: {error}"));
    let requirement = bytecode_imm_canonical_bytes_requirement(chunk, 2, 257);
    let statement = LatticePackedValidityStatement {
        requirement,
        kind: LatticePackedValidityStatementKind::FieldElementCanonicalBytes,
        num_vars: 2,
        degree: 2,
    };
    let point = vec![Fr::from_u64(2), Fr::from_u64(3)];
    let eq_point = vec![Fr::from_u64(5), Fr::from_u64(7)];
    let batching_coefficient = Fr::from_u64(11);
    let opening_claims = [Fr::from_u64(13), Fr::from_u64(17), Fr::from_u64(19)];
    let reduction = BatchedEvaluationClaim {
        reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(23)),
        batching_coefficients: vec![batching_coefficient],
        max_num_vars: 2,
        max_degree: 2,
    };

    let batch = build_lattice_packed_validity_batch(
        &layout,
        std::slice::from_ref(&statement),
        99_u64,
        std::slice::from_ref(&eq_point),
        &reduction,
        &opening_claims,
    )
    .unwrap_or_else(|error| panic!("bytecode imm canonical-byte batch should build: {error}"));

    let expected_eq = try_eq_mle(&point, &eq_point)
        .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
    let expected_invalid_indicator = opening_claims[0] + opening_claims[1] * opening_claims[2];
    assert_eq!(
        batch.expected_final_claim,
        batching_coefficient * expected_eq * expected_invalid_indicator
    );
    assert_eq!(batch.statement.claims.len(), 3);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[0].view else {
        panic!("high-byte range factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 254);
    assert_eq!(
        terms[0].family,
        PackedFamilyId::BytecodeImmBytes { chunk }.physical_ref()
    );
    assert_eq!(terms[0].limb, 1);
    assert_eq!(terms[0].symbol, 2);
    assert_eq!(terms[253].limb, 1);
    assert_eq!(terms[253].symbol, 255);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[1].view else {
        panic!("high-byte equality factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 1);
    assert_eq!(
        terms[0].family,
        PackedFamilyId::BytecodeImmBytes { chunk }.physical_ref()
    );
    assert_eq!(terms[0].limb, 1);
    assert_eq!(terms[0].symbol, 1);

    let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[2].view else {
        panic!("low-byte range factor should use a packed linear view");
    };
    assert_eq!(terms.len(), 255);
    assert_eq!(
        terms[0].family,
        PackedFamilyId::BytecodeImmBytes { chunk }.physical_ref()
    );
    assert_eq!(terms[0].limb, 0);
    assert_eq!(terms[0].symbol, 1);
    assert_eq!(terms[254].limb, 0);
    assert_eq!(terms[254].symbol, 255);
}

#[test]
fn derive_validity_statements_rejects_layout_requirement_mismatch() {
    let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::UnsignedIncMsb,
        PackedFactDomain::TraceRows { log_t: 4 },
        1,
        PackedAlphabet::Bit,
    )])
    .unwrap_or_else(|error| panic!("layout should build: {error}"));
    let requirements = [LatticePackedValidityRequirement::exact_one_hot(
        LatticePackedFamilyId::UnsignedIncMsb,
        8,
        256,
    )];

    assert!(matches!(
        derive_lattice_packed_validity_statements(&layout, &requirements),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("limb count mismatch")
    ));
}

#[test]
fn validity_coverage_accepts_bound_decoded_families() {
    let id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::ProgramImageInit,
        JoltRelationId::ProgramImageClaimReduction,
    ));
    let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
        LatticePackedFamilyId::ProgramImageInit,
        0,
    ));

    validate_lattice_view_validity_coverage(
        &[(id, formula, vec![Fr::from_u64(1)])],
        &[program_image_validity_requirement()],
    )
    .unwrap_or_else(|error| panic!("bound decoded family should validate: {error}"));
}

#[test]
fn validity_coverage_rejects_unbound_decoded_families() {
    let id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::ProgramImageInit,
        JoltRelationId::ProgramImageClaimReduction,
    ));
    let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
        LatticePackedFamilyId::ProgramImageInit,
        0,
    ));

    assert!(matches!(
        validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &[]),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("without a bound validity requirement")
    ));
}

#[test]
fn validity_coverage_requires_canonical_byte_requirements_for_field_bytes() {
    let id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeChunk(0),
        JoltRelationId::BytecodeClaimReduction,
    ));
    let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
        LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 },
        0,
    ));
    let one_hot = LatticePackedValidityRequirement::exact_one_hot(
        LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 },
        AkitaField::NUM_BYTES,
        256,
    );
    let canonical =
        bytecode_imm_canonical_bytes_requirement(0, AkitaField::NUM_BYTES, AKITA_FIELD_MODULUS);

    assert!(matches!(
        validate_lattice_view_validity_coverage(
            &[(id, formula.clone(), Vec::new())],
            std::slice::from_ref(&one_hot),
        ),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("without a bound canonical-byte validity requirement")
    ));

    validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &[one_hot, canonical])
        .unwrap_or_else(|error| panic!("canonical byte family should validate: {error}"));
}

#[test]
fn validity_coverage_accepts_unsigned_increment_validity_requirements() {
    let id = Stage8OpeningId::from(unsigned_inc_chunk_opening(0));
    let formula = unsigned_inc_lower_value_lattice_view_formula::<Fr>(8)
        .unwrap_or_else(|| panic!("unsigned increment lower-value view should derive"));
    let requirements = unsigned_inc_validity_requirements(8)
        .unwrap_or_else(|| panic!("unsigned increment requirements should derive"));

    validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &requirements)
        .unwrap_or_else(|error| {
            panic!("unsigned increment validity coverage should validate: {error}")
        });
}

#[test]
fn validity_coverage_requires_bytecode_store_rd_disjointness() {
    let id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeChunk(0),
        JoltRelationId::BytecodeClaimReduction,
    ));
    let formula = LatticePackedViewFormula::<Fr>::direct(
        LatticePackedFamilyId::BytecodeCircuitFlag {
            chunk: 0,
            flag: CircuitFlags::Store as usize,
        },
        0,
        1,
    );
    let store_flag = LatticePackedValidityRequirement::boolean_indicator(
        LatticePackedFamilyId::BytecodeCircuitFlag {
            chunk: 0,
            flag: CircuitFlags::Store as usize,
        },
        1,
        2,
        1,
    );
    let disjoint = LatticePackedValidityRequirement::bytecode_store_rd_disjoint(0);

    assert!(matches!(
        validate_lattice_view_validity_coverage(
            &[(id, formula.clone(), Vec::new())],
            std::slice::from_ref(&store_flag),
        ),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("without a bound Store/Rd disjointness requirement")
    ));

    validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &[store_flag, disjoint])
        .unwrap_or_else(|error| {
            panic!("bytecode Store/Rd disjointness coverage should validate: {error}")
        });
}

#[test]
fn validity_coverage_allows_core_ra_families() {
    let id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::InstructionRa(0),
        JoltRelationId::HammingWeightClaimReduction,
    ));
    let formula = LatticePackedViewFormula::linear_decoded(vec![LatticePackedViewTerm::new(
        Fr::from_u64(1),
        LatticePackedFamilyId::InstructionRa { index: 0 },
        0,
        0,
    )]);

    validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &[])
        .unwrap_or_else(|error| panic!("core RA family should not need lattice validity: {error}"));
}

#[test]
fn validity_coverage_checks_boolean_indicator_symbol() {
    let id = Stage8OpeningId::from(unsigned_inc_msb_opening());
    let requirement = LatticePackedValidityRequirement::boolean_indicator(
        LatticePackedFamilyId::UnsignedIncMsb,
        1,
        2,
        1,
    );

    validate_lattice_view_validity_coverage(
        &[(
            id,
            LatticePackedViewFormula::<Fr>::direct(LatticePackedFamilyId::UnsignedIncMsb, 0, 1),
            Vec::new(),
        )],
        std::slice::from_ref(&requirement),
    )
    .unwrap_or_else(|error| panic!("covered boolean indicator should validate: {error}"));

    assert!(matches!(
        validate_lattice_view_validity_coverage(
            &[(
                id,
                LatticePackedViewFormula::<Fr>::direct(
                    LatticePackedFamilyId::UnsignedIncMsb,
                    0,
                    0,
                ),
                Vec::new(),
            )],
            &[requirement],
        ),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("without a bound validity requirement")
    ));
}

#[test]
fn derive_layout_uses_unsigned_increment_validity_requirements() {
    let config = lattice_config();
    let log_t = 3;
    let layout = derive_lattice_packed_witness_layout(
        &config,
        log_t,
        8,
        ra_layout(),
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

    for requirement in unsigned_inc_validity_requirements(8)
        .unwrap_or_else(|| panic!("unsigned increment requirements should derive"))
    {
        let family_id = lattice_packing_family_id(&requirement.family);
        let family = layout
            .family(&family_id)
            .unwrap_or_else(|| panic!("validity family {family_id:?} should be present"));

        assert_eq!(family.domain, PackedFactDomain::TraceRows { log_t });
        assert_eq!(family.limbs, requirement.limbs);
        assert_eq!(family.alphabet.size(), requirement.alphabet_size);
    }
}

#[test]
fn derive_layout_excludes_committed_bytecode_source_requirements() {
    let config = lattice_config();
    let precommitted = precommitted_schedule(Some(8));
    let layout = derive_lattice_packed_witness_layout(&config, 2, 8, ra_layout(), &precommitted)
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
    let validity_requirements =
        derive_lattice_packed_validity_requirements(&config, 8, &precommitted)
            .unwrap_or_else(|error| panic!("validity requirements should derive: {error}"));

    for requirement in bytecode_validity_requirements(0, AkitaField::NUM_BYTES) {
        let family_id = lattice_packing_family_id(&requirement.family);
        assert!(layout.family(&family_id).is_none());
        assert!(!validity_requirements.contains(&requirement));
    }
    assert!(
        !validity_requirements.contains(&bytecode_imm_canonical_bytes_requirement(
            0,
            AkitaField::NUM_BYTES,
            AKITA_FIELD_MODULUS,
        ))
    );
}

#[test]
fn lattice_family_ids_convert_to_packing_family_ids() {
    assert_eq!(
        lattice_packing_family_id(&LatticePackedFamilyId::AdviceBytes {
            kind: JoltAdviceKind::Trusted,
            index: 3,
        }),
        PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Trusted,
            index: 3,
        }
    );
    assert_eq!(
        lattice_packing_family_id(&LatticePackedFamilyId::Custom {
            namespace: 17,
            index: 5,
        }),
        PackedFamilyId::Custom {
            namespace: 17,
            index: 5,
        }
    );
    assert_eq!(
        lattice_packing_family_id(&LatticePackedFamilyId::BytecodeRegisterSelector {
            chunk: 2,
            selector: 1,
        }),
        PackedFamilyId::BytecodeRegisterSelector {
            chunk: 2,
            selector: 1,
        }
    );
    assert_eq!(
        lattice_packing_family_id(&LatticePackedFamilyId::BytecodeImmBytes { chunk: 2 }),
        PackedFamilyId::BytecodeImmBytes { chunk: 2 }
    );
}

#[test]
fn lattice_direct_view_converts_to_packing_view_formula() {
    let formula =
        LatticePackedViewFormula::<Fr>::direct(LatticePackedFamilyId::UnsignedIncMsb, 0, 1);

    assert_eq!(
        lattice_packing_view_formula(&formula)
            .unwrap_or_else(|error| panic!("direct view should convert: {error}")),
        PackedViewFormula::direct(PackedFamilyId::UnsignedIncMsb, 0, 1)
    );
}

#[test]
fn lattice_linear_view_converts_terms_to_packing_view_formula() {
    let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
        LatticePackedFamilyId::BytecodeChunk { index: 2 },
        4,
    ));

    let converted = lattice_packing_view_formula(&formula)
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
        lattice_packing_view_formula::<Fr>(&LatticePackedViewFormula::masked_decoded(
            JoltRelationId::UnsignedIncClaimReduction,
        )),
        Err(PackedViewError::MaskedViewRequiresTranslation)
    ));
}

#[test]
fn lattice_reduced_masked_view_converts_terms_to_packing_formula() {
    let formula = LatticePackedViewFormula::reduced_masked(
        JoltRelationId::UnsignedIncClaimReduction,
        vec![jolt_claims::protocols::jolt::LatticePackedViewTerm::new(
            Fr::from_u64(9),
            LatticePackedFamilyId::UnsignedIncMsb,
            0,
            1,
        )],
    );
    assert!(matches!(
        lattice_packing_view_formula::<Fr>(&formula),
        Ok(PackedViewFormula::ReducedMasked { terms })
            if terms.len() == 1
                && terms[0].coefficient == Fr::from_u64(9)
                && terms[0].family == PackedFamilyId::UnsignedIncMsb
                && terms[0].limb == 0
                && terms[0].symbol == 1
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
    let formulas = jolt_lattice_view_formulas(
        &logical_manifest(id, point),
        2,
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("RA lattice formula should resolve: {error}"));

    assert_eq!(formulas[0].0, Stage8OpeningId::from(id));
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

#[cfg(feature = "field-inline")]
#[test]
fn field_inline_rd_inc_resolves_to_packed_byte_families() {
    let id = field_increments::field_rd_inc_reduced_opening();
    let point = vec![Fr::from_u64(2), Fr::from_u64(3)];
    let formulas = jolt_lattice_view_formulas(
        &logical_manifest_for_stage8(Stage8OpeningId::from(id), point.clone()),
        8,
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("field rd inc lattice formula should resolve: {error}"));

    assert_eq!(formulas[0].0, Stage8OpeningId::from(id));
    assert_eq!(formulas[0].2, point);

    let terms = linear_decoded_terms(&formulas[0].1);
    assert_eq!(terms.len(), AkitaField::NUM_BYTES * 256);
    let byte_1_symbol_3 = find_lattice_term(
        terms,
        LatticePackedFamilyId::FieldRdIncByte { index: 1 },
        0,
        3,
    );
    assert_eq!(byte_1_symbol_3.coefficient, Fr::from_u64(3 * 256));
}

#[test]
fn jolt_lattice_resolver_keeps_precommitted_finals_out_of_w_pack() {
    let trusted = JoltOpeningId::trusted_advice(JoltRelationId::AdviceClaimReduction);
    let untrusted = JoltOpeningId::untrusted_advice(JoltRelationId::AdviceClaimReduction);
    let program_image = JoltOpeningId::committed(
        JoltCommittedPolynomial::ProgramImageInit,
        JoltRelationId::ProgramImageClaimReduction,
    );

    let schedule = precommitted_schedule(None);
    assert!(matches!(
        jolt_lattice_view_formula::<Fr>(trusted, &[Fr::from_u64(1)], 8, &schedule),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("trusted advice uses a separate precommitted opening")
    ));
    assert!(matches!(
        jolt_lattice_view_formula::<Fr>(program_image, &[Fr::from_u64(1)], 8, &schedule),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("ProgramImageInit uses a separate precommitted opening")
    ));

    let untrusted_formula = jolt_lattice_view_formula(untrusted, &[Fr::from_u64(1)], 8, &schedule)
        .unwrap_or_else(|error| panic!("untrusted advice view should resolve: {error}"));
    assert!(matches!(
        untrusted_formula,
        LatticePackedViewFormula::LinearDecoded { terms }
            if terms.len() == 256
                && terms[7].coefficient == Fr::from_u64(7)
                && terms[7].family == (LatticePackedFamilyId::AdviceBytes {
                    kind: JoltAdviceKind::Untrusted,
                    index: 0,
                })
                && terms[7].limb == 0
                && terms[7].symbol == 7
    ));
}

#[test]
fn jolt_lattice_resolver_rejects_dense_increment_openings() {
    let id = JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::IncClaimReduction,
    );
    assert!(matches!(
        jolt_lattice_view_formula(id, &[Fr::from_u64(1)], 8, &precommitted_schedule(None)),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("not dense IncClaimReduction polynomials")
    ));
}

#[test]
fn jolt_lattice_resolver_lowers_unsigned_increment_chunk_and_msb_outputs() {
    let point = (1..=9).map(Fr::from_u64).collect::<Vec<_>>();
    let chunk = jolt_lattice_view_formula(
        unsigned_inc_chunk_opening(7),
        &point,
        8,
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("chunk view should resolve: {error}"));
    let chunk_terms = linear_decoded_terms(&chunk);
    let expected_weights = EqPolynomial::<Fr>::evals(&point[..8], None);
    assert_eq!(
        find_lattice_term(
            chunk_terms,
            LatticePackedFamilyId::UnsignedIncChunk { index: 7 },
            0,
            3,
        )
        .coefficient,
        expected_weights[3]
    );

    assert!(matches!(
        jolt_lattice_view_formula(
            unsigned_inc_msb_opening(),
            &[Fr::from_u64(1)],
            8,
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("sign view should resolve: {error}")),
        LatticePackedViewFormula::Direct {
            family: LatticePackedFamilyId::UnsignedIncMsb,
            limb: 0,
            symbol: 1
        }
    ));
}

#[test]
fn jolt_lattice_resolver_rejects_bytecode_chunks_as_packed_views() {
    let schedule = precommitted_schedule(None);
    let id = JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeChunk(0),
        JoltRelationId::BytecodeClaimReduction,
    );
    let (point, _) = bytecode_chunk_opening_point();

    assert!(matches!(
        jolt_lattice_view_formula(id, &point, 8, &schedule),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("BytecodeChunk(0) uses a separate precommitted opening")
    ));
}

#[test]
fn jolt_lattice_resolver_rejects_bytecode_chunks_without_layout() {
    let id = JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeChunk(0),
        JoltRelationId::BytecodeClaimReduction,
    );
    assert!(matches!(
        jolt_lattice_view_formula::<Fr>(
            id,
            &[Fr::from_u64(1)],
            8,
            &precommitted_schedule_without_committed_program(),
        ),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("BytecodeChunk(0) uses a separate precommitted opening")
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

    let physical =
        jolt_lattice_physical_manifest(&logical, &layout, 2, &precommitted_schedule(None))
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
fn jolt_lattice_physical_manifest_rejects_bytecode_chunk_packed_view() {
    let schedule = precommitted_schedule(None);
    let layout =
        derive_lattice_packed_witness_layout(&lattice_config(), 2, 8, ra_layout(), &schedule)
            .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
    let id = JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeChunk(0),
        JoltRelationId::BytecodeClaimReduction,
    );
    let (point, _) = bytecode_chunk_opening_point();
    let logical = logical_manifest(id, point);

    assert!(matches!(
        jolt_lattice_physical_manifest(&logical, &layout, 8, &schedule),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("BytecodeChunk(0) uses a separate precommitted opening")
    ));
}

#[test]
fn jolt_lattice_physical_manifest_rejects_dense_increment_view() {
    let id = JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::IncClaimReduction,
    );
    let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::UnsignedIncMsb,
        PackedFactDomain::TraceRows { log_t: 0 },
        1,
        PackedAlphabet::Bit,
    )])
    .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

    assert!(matches!(
        jolt_lattice_physical_manifest(
            &logical_manifest(id, vec![Fr::from_u64(1)]),
            &layout,
            8,
            &precommitted_schedule(None),
        ),
        Err(VerifierError::FinalOpeningBatchFailed { reason })
            if reason.contains("not dense IncClaimReduction polynomials")
    ));
}

#[cfg(feature = "field-inline")]
#[test]
fn jolt_lattice_physical_manifest_resolves_field_inline_rd_inc() {
    let mut config = lattice_config();
    config.lattice.field_inline.enabled = true;
    let layout = derive_lattice_packed_witness_layout(
        &config,
        2,
        8,
        ra_layout(),
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
    let id = field_increments::field_rd_inc_reduced_opening();
    let row_point = vec![Fr::from_u64(11), Fr::from_u64(13)];

    let manifest = jolt_lattice_physical_manifest(
        &logical_manifest_for_stage8(Stage8OpeningId::from(id), row_point.clone()),
        &layout,
        8,
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("field rd inc physical manifest should resolve: {error}"));

    let PhysicalView::PackedLinear { terms, .. } = &manifest.openings[0].view else {
        panic!("field rd inc should lower to a packed linear view");
    };
    let term = find_physical_term(terms, PackedFamilyId::FieldRdIncByte { index: 1 }, 0, 3);
    assert_eq!(term.coefficient, Fr::from_u64(3 * 256));
    assert_eq!(term.row_point, row_point);
}

#[test]
fn layout_config_mismatch_rejects() {
    let layout = derive_lattice_packed_witness_layout(
        &lattice_config(),
        2,
        8,
        ra_layout(),
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

    assert!(matches!(
        validate_lattice_packed_witness_layout_config(&lattice_config(), &layout),
        Err(VerifierError::InvalidProtocolConfig { .. })
    ));
}

#[test]
fn layout_config_rejects_precommitted_packed_families() {
    let precommitted_specs = [
        PackedFamilySpec::direct(
            PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            },
            PackedFactDomain::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                log_bytes: 1,
            },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeChunk { index: 0 },
            PackedFactDomain::BytecodeRows { log_bytecode: 1 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeCircuitFlag {
                chunk: 0,
                flag: CircuitFlags::Store as usize,
            },
            PackedFactDomain::BytecodeRows { log_bytecode: 1 },
            1,
            PackedAlphabet::Bit,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 2,
            },
            PackedFactDomain::BytecodeRows { log_bytecode: 1 },
            1,
            PackedAlphabet::Fixed {
                size: 1 << REGISTER_ADDRESS_BITS,
            },
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::ProgramImageInit,
            PackedFactDomain::ProgramImageWords { log_words: 1 },
            8,
            PackedAlphabet::Byte,
        ),
    ];

    for spec in precommitted_specs {
        let layout = PackedWitnessLayout::new([spec])
            .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let mut config = lattice_config();
        config.lattice.packed_witness.layout_digest = Some(layout.digest);
        config.lattice.packed_witness.d_pack = Some(layout.dimension);

        assert!(matches!(
            validate_lattice_packed_witness_layout_config(&config, &layout),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("cannot be included in the Akita packed witness layout")
        ));
    }
}

#[test]
fn untrusted_advice_layout_requires_precommitted_schedule() {
    let mut config = lattice_config();
    config.lattice.advice.untrusted = true;

    assert!(matches!(
        derive_lattice_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        ),
        Err(VerifierError::InvalidPrecommittedSchedule { .. })
    ));

    let layout = derive_lattice_packed_witness_layout(
        &config,
        2,
        8,
        ra_layout(),
        &precommitted_schedule_with_advice(None, Some(64)),
    )
    .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
    assert!(layout
        .family(&PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Untrusted,
            index: 0,
        })
        .is_some());
}

#[cfg(feature = "field-inline")]
#[test]
fn field_inline_layout_uses_separate_rd_inc_families() {
    let mut config = lattice_config();
    config.lattice.field_inline.enabled = true;

    let layout = derive_lattice_packed_witness_layout(
        &config,
        2,
        8,
        ra_layout(),
        &precommitted_schedule(None),
    )
    .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

    assert!(layout.family(&PackedFamilyId::UnsignedIncMsb).is_some());
    assert!(layout.family(&PackedFamilyId::FieldRdIncSign).is_none());
    assert!(layout
        .family(&PackedFamilyId::FieldRdIncByte { index: 7 })
        .is_some());
    assert!(layout
        .family(&PackedFamilyId::FieldRdIncByte {
            index: AkitaField::NUM_BYTES - 1
        })
        .is_some());
}

#[test]
fn untrusted_advice_uses_non_trace_domain() {
    let mut config = lattice_config();
    config.lattice.advice.untrusted = true;

    let layout = derive_lattice_packed_witness_layout(
        &config,
        2,
        8,
        ra_layout(),
        &precommitted_schedule_with_advice(None, Some(128)),
    )
    .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

    assert!(layout
        .family(&PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Trusted,
            index: 0,
        })
        .is_none());

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

    assert!(layout
        .family(&PackedFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 0,
        })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeLookupSelector { chunk: 0 })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeImmBytes { chunk: 0 })
        .is_none());
    assert!(layout.family(&PackedFamilyId::ProgramImageInit).is_none());
}

#[cfg(feature = "field-inline")]
#[test]
fn single_packed_witness_layout_includes_all_supported_lattice_families() {
    let mut config = lattice_config();
    config.lattice.field_inline.enabled = true;
    config.lattice.advice.trusted = true;
    config.lattice.advice.untrusted = true;

    let layout = derive_lattice_packed_witness_layout(
        &config,
        2,
        8,
        ra_layout(),
        &precommitted_schedule_with_advice(Some(64), Some(128)),
    )
    .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
    let audit = layout.audit();

    assert_eq!(audit.d_pack, layout.dimension);
    assert!(audit.cells_by_domain.trace_rows > 0);
    assert_eq!(audit.cells_by_domain.bytecode_rows, 0);
    assert_eq!(audit.cells_by_domain.program_image_words, 0);
    assert!(audit.cells_by_domain.advice_bytes > 0);
    assert!(layout.family(&PackedFamilyId::UnsignedIncMsb).is_some());
    assert!(layout
        .family(&PackedFamilyId::UnsignedIncChunk { index: 7 })
        .is_some());
    assert!(layout
        .family(&PackedFamilyId::FieldRdIncByte { index: 0 })
        .is_some());
    assert!(layout
        .family(&PackedFamilyId::FieldRdIncByte {
            index: AkitaField::NUM_BYTES - 1,
        })
        .is_some());
    assert!(layout
        .family(&PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Trusted,
            index: 0,
        })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Untrusted,
            index: 0,
        })
        .is_some());
    assert!(layout
        .family(&PackedFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 2,
        })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeLookupSelector { chunk: 0 })
        .is_none());
    assert!(layout
        .family(&PackedFamilyId::BytecodeImmBytes { chunk: 0 })
        .is_none());
    assert!(layout.family(&PackedFamilyId::ProgramImageInit).is_none());
}
