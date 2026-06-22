
#![expect(
    clippy::expect_used,
    reason = "tests assert successful layout construction"
)]

use super::*;
use jolt_field::{Fr, FromPrimitiveInt};

fn trace(log_t: usize) -> PackedFactDomain {
    PackedFactDomain::TraceRows { log_t }
}

fn byte_family(id: PackedFamilyId, log_t: usize) -> PackedFamilySpec {
    PackedFamilySpec::direct(id, trace(log_t), 1, PackedAlphabet::Byte)
}

fn bit_family(id: PackedFamilyId, log_t: usize) -> PackedFamilySpec {
    PackedFamilySpec::direct(id, trace(log_t), 1, PackedAlphabet::Bit)
}

fn base_ra_specs(log_t: usize) -> Vec<PackedFamilySpec> {
    let mut specs = Vec::new();
    specs.extend((0..16).map(|index| byte_family(PackedFamilyId::InstructionRa { index }, log_t)));
    specs.extend((0..3).map(|index| byte_family(PackedFamilyId::BytecodeRa { index }, log_t)));
    specs.extend((0..4).map(|index| byte_family(PackedFamilyId::RamRa { index }, log_t)));
    specs
}

fn unsigned_increment_specs(log_t: usize) -> Vec<PackedFamilySpec> {
    let mut specs = (0..8)
        .map(|index| byte_family(PackedFamilyId::UnsignedIncChunk { index }, log_t))
        .collect::<Vec<_>>();
    specs.push(bit_family(PackedFamilyId::UnsignedIncMsb, log_t));
    specs
}

#[test]
fn packed_witness_layout_digest_stable() {
    let mut specs = vec![
        byte_family(PackedFamilyId::RamRa { index: 0 }, 4),
        bit_family(PackedFamilyId::UnsignedIncMsb, 4),
        byte_family(PackedFamilyId::InstructionRa { index: 0 }, 4),
    ];
    let layout_a = PackedWitnessLayout::new(specs.clone()).expect("layout should build");
    specs.reverse();
    let layout_b = PackedWitnessLayout::new(specs).expect("layout should build");

    assert_eq!(layout_a.digest, layout_b.digest);
    assert_eq!(layout_a.families, layout_b.families);
}

#[test]
fn packed_witness_layout_rejects_duplicate_ranges() {
    let specs = vec![
        byte_family(PackedFamilyId::RamRa { index: 0 }, 3),
        byte_family(PackedFamilyId::RamRa { index: 0 }, 3),
    ];
    assert!(matches!(
        PackedWitnessLayout::new(specs),
        Err(PackedLayoutError::DuplicateFamily { .. })
    ));
}

#[test]
fn large_trace_base_cells_are_5888_per_row() {
    let layout = PackedWitnessLayout::new(base_ra_specs(20)).expect("layout should build");
    let audit = layout.audit();

    assert_eq!(audit.trace_cells_per_row, Some(5_888));
    assert_eq!(layout.dimension, 33);
}

#[test]
fn unsigned_increment_budget_is_n_plus_13() {
    let log_t = 20;
    let mut specs = base_ra_specs(log_t);
    specs.extend(unsigned_increment_specs(log_t));

    let layout = PackedWitnessLayout::new(specs).expect("layout should build");
    let audit = layout.audit();

    assert_eq!(audit.trace_cells_per_row, Some(7_938));
    assert_eq!(layout.dimension, log_t + 13);
}

#[test]
fn bit_fact_costs_two_cells_per_row() {
    let layout = PackedWitnessLayout::new([bit_family(PackedFamilyId::UnsignedIncMsb, 5)])
        .expect("layout should build");
    let audit = layout.audit();

    assert_eq!(layout.cells, 64);
    assert_eq!(audit.trace_cells_per_row, Some(2));
    assert_eq!(audit.fact_count_by_alphabet.bit, 1);
}

#[test]
fn rank_unrank_roundtrip() {
    let layout = PackedWitnessLayout::new([
        byte_family(PackedFamilyId::RamRa { index: 0 }, 1),
        bit_family(PackedFamilyId::UnsignedIncMsb, 1),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeChunk { index: 0 },
            PackedFactDomain::BytecodeRows { log_bytecode: 1 },
            2,
            PackedAlphabet::Fixed { size: 3 },
        ),
    ])
    .expect("layout should build");

    for rank in 0..layout.cells {
        let address = layout.unrank(rank).expect("non-dummy rank should unrank");
        assert_eq!(layout.rank(&address).expect("address should rank"), rank);
    }
}

#[test]
fn committed_bytecode_lane_families_are_distinct() {
    let bytecode = PackedFactDomain::BytecodeRows { log_bytecode: 1 };
    let families = [
        PackedFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 0,
        },
        PackedFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 1,
        },
        PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 },
        PackedFamilyId::BytecodeInstructionFlag { chunk: 0, flag: 0 },
        PackedFamilyId::BytecodeLookupSelector { chunk: 0 },
        PackedFamilyId::BytecodeRafFlag { chunk: 0 },
        PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
        PackedFamilyId::BytecodeImmBytes { chunk: 0 },
    ];
    let layout = PackedWitnessLayout::new([
        PackedFamilySpec::direct(
            families[0].clone(),
            bytecode,
            1,
            PackedAlphabet::Fixed { size: 32 },
        ),
        PackedFamilySpec::direct(
            families[1].clone(),
            bytecode,
            1,
            PackedAlphabet::Fixed { size: 32 },
        ),
        PackedFamilySpec::direct(families[2].clone(), bytecode, 1, PackedAlphabet::Bit),
        PackedFamilySpec::direct(families[3].clone(), bytecode, 1, PackedAlphabet::Bit),
        PackedFamilySpec::direct(
            families[4].clone(),
            bytecode,
            1,
            PackedAlphabet::Fixed { size: 4 },
        ),
        PackedFamilySpec::direct(families[5].clone(), bytecode, 1, PackedAlphabet::Bit),
        PackedFamilySpec::direct(families[6].clone(), bytecode, 8, PackedAlphabet::Byte),
        PackedFamilySpec::direct(families[7].clone(), bytecode, 16, PackedAlphabet::Byte),
    ])
    .expect("layout should build");

    for family in &families {
        assert!(layout.family(family).is_some());
    }
    for left in 0..families.len() {
        for right in left + 1..families.len() {
            assert_ne!(
                families[left].physical_ref(),
                families[right].physical_ref()
            );
        }
    }

    let address = PackedCellAddress {
        family: PackedFamilyId::BytecodeImmBytes { chunk: 0 },
        row: 1,
        limb: 15,
        symbol: 7,
    };
    let rank = layout.rank(&address).expect("bytecode byte address ranks");
    assert_eq!(layout.unrank(rank), Some(address));
}

#[test]
fn dummy_cells_are_zero_and_unreferenced() {
    let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        },
        trace(0),
        1,
        PackedAlphabet::Fixed { size: 3 },
    )])
    .expect("layout should build");

    assert_eq!(layout.cells, 3);
    assert_eq!(layout.dimension, 2);
    assert_eq!(layout.dummy_cell_count(), 1);
    assert!(layout.unrank(layout.cells).is_none());

    let source = SparsePackedWitness::<Fr>::try_new(layout.clone(), Vec::new())
        .expect("empty source should build");
    let zero_address = PackedCellAddress {
        family: PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        },
        row: 0,
        limb: 0,
        symbol: 2,
    };
    assert_eq!(
        source
            .eval_direct_fact(&zero_address)
            .expect("address is in range"),
        Fr::from_u64(0)
    );
}

#[test]
fn layout_sort_order_is_stable() {
    let layout = PackedWitnessLayout::new([
        byte_family(PackedFamilyId::RamRa { index: 3 }, 2),
        byte_family(PackedFamilyId::InstructionRa { index: 1 }, 2),
        byte_family(PackedFamilyId::BytecodeRa { index: 2 }, 2),
    ])
    .expect("layout should build");

    assert_eq!(
        layout
            .families
            .iter()
            .map(|family| &family.id)
            .collect::<Vec<_>>(),
        vec![
            &PackedFamilyId::InstructionRa { index: 1 },
            &PackedFamilyId::BytecodeRa { index: 2 },
            &PackedFamilyId::RamRa { index: 3 },
        ]
    );
}

#[test]
fn committed_program_families_use_non_trace_domains() {
    let layout = PackedWitnessLayout::new([
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeChunk { index: 0 },
            PackedFactDomain::BytecodeRows { log_bytecode: 4 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::ProgramImageInit,
            PackedFactDomain::ProgramImageWords { log_words: 3 },
            8,
            PackedAlphabet::Byte,
        ),
    ])
    .expect("layout should build");
    let audit = layout.audit();

    assert_eq!(audit.cells_by_domain.trace_rows, 0);
    assert_eq!(audit.cells_by_domain.bytecode_rows, 16 * 256);
    assert_eq!(audit.cells_by_domain.program_image_words, 8 * 8 * 256);
    assert_eq!(audit.trace_cells_per_row, None);
}

#[test]
fn planner_audit_fields_are_reported() {
    let layout = PackedWitnessLayout::new([
        byte_family(PackedFamilyId::InstructionRa { index: 0 }, 2),
        bit_family(PackedFamilyId::UnsignedIncMsb, 2),
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
            PackedAlphabet::Fixed { size: 4 },
        ),
    ])
    .expect("layout should build");
    let audit = layout.audit();

    assert_eq!(
        audit.fact_count_by_alphabet,
        PackedAlphabetCounts {
            bit: 1,
            byte: 1,
            fixed: 1,
        }
    );
    assert_eq!(audit.cells_by_domain.trace_rows, 4 * 256 + 4 * 2);
    assert_eq!(audit.cells_by_domain.advice_bytes, 2 * 4);
    assert_eq!(audit.d_pack, layout.dimension);
    assert!(audit.rectangular_lane_equivalent >= layout.cells);
}

#[test]
fn packed_witness_source_respects_layout() {
    let layout = PackedWitnessLayout::new([
        byte_family(PackedFamilyId::RamRa { index: 0 }, 1),
        bit_family(PackedFamilyId::UnsignedIncMsb, 1),
    ])
    .expect("layout should build");
    let one_address = PackedCellAddress {
        family: PackedFamilyId::RamRa { index: 0 },
        row: 1,
        limb: 0,
        symbol: 17,
    };
    let sign_address = PackedCellAddress {
        family: PackedFamilyId::UnsignedIncMsb,
        row: 0,
        limb: 0,
        symbol: 1,
    };
    let source = SparsePackedWitness::try_from_cells(
        layout.clone(),
        [
            (one_address.clone(), Fr::from_u64(11)),
            (sign_address.clone(), Fr::from_u64(1)),
        ],
    )
    .expect("source should build");

    let mut streamed = Vec::new();
    source.for_each_nonzero(|rank, value| streamed.push((rank, value)));

    assert_eq!(source.layout().digest, layout.digest);
    assert_eq!(streamed.len(), 2);
    assert!(streamed.iter().all(|(rank, _)| *rank < layout.cells));
    assert_eq!(
        source
            .eval_direct_fact(&one_address)
            .expect("address is in range"),
        Fr::from_u64(11)
    );
    assert_eq!(
        source
            .eval_direct_fact(&sign_address)
            .expect("address is in range"),
        Fr::from_u64(1)
    );
}

#[test]
fn view_catalog_references_existing_families() {
    let layout = PackedWitnessLayout::new([byte_family(PackedFamilyId::RamRa { index: 0 }, 2)])
        .expect("layout should build");

    layout
        .validate_view_families(&[PackedFamilyId::RamRa { index: 0 }])
        .expect("existing family should validate");
    assert!(matches!(
        layout.validate_view_families(&[PackedFamilyId::UnsignedIncMsb]),
        Err(PackedLayoutError::MissingViewFamily { .. })
    ));
}

#[test]
fn sparse_source_rejects_out_of_layout_ranks() {
    let layout = PackedWitnessLayout::new([bit_family(PackedFamilyId::UnsignedIncMsb, 0)])
        .expect("layout should build");

    assert!(matches!(
        SparsePackedWitness::try_new(layout.clone(), vec![(layout.cells, Fr::from_u64(1))]),
        Err(PackedLayoutError::RankOutOfRange { .. })
    ));
}
