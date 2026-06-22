#![expect(
    clippy::expect_used,
    reason = "tests assert successful packed-view setup"
)]

use super::*;
use crate::{
    PackedAlphabet, PackedFactDomain, PackedFamilySpec, PackedWitnessLayout, SparsePackedWitness,
};
use jolt_field::{Fr, FromPrimitiveInt};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OpeningId {
    A,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RelationId {
    First,
    Second,
}

fn f(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn byte_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new([
        PackedFamilySpec::direct(
            PackedFamilyId::RamRa { index: 0 },
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::UnsignedIncMsb,
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Bit,
        ),
    ])
    .expect("layout should build")
}

fn byte_decode_terms(family: PackedFamilyId) -> Vec<PackedViewTerm<Fr>> {
    (0..256)
        .map(|symbol| PackedViewTerm::new(f(symbol as u64), family.clone(), 0, symbol))
        .collect()
}

#[test]
fn direct_view_translation_matches_packed_eval() {
    let layout = byte_layout();
    let address = PackedCellAddress {
        family: PackedFamilyId::UnsignedIncMsb,
        row: 1,
        limb: 0,
        symbol: 1,
    };
    let source = SparsePackedWitness::try_from_cells(layout.clone(), [(address, f(1))])
        .expect("source should build");
    let formula = PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1);

    assert_eq!(
        formula.eval_row(&source, 1).expect("view should evaluate"),
        f(1)
    );
    assert!(matches!(
        formula
            .physical_view(&layout)
            .expect("view should lower to the opening API"),
        PhysicalView::PackedLinear { terms, .. }
            if terms.len() == 1
                && terms[0].coefficient == f(1)
                && terms[0].family == PackedFamilyId::UnsignedIncMsb.physical_ref()
                && terms[0].symbol == 1
    ));
}

#[test]
fn linear_decode_translation_matches_direct_sum() {
    let layout = byte_layout();
    let address = PackedCellAddress {
        family: PackedFamilyId::RamRa { index: 0 },
        row: 0,
        limb: 0,
        symbol: 7,
    };
    let source = SparsePackedWitness::try_from_cells(layout.clone(), [(address, f(1))])
        .expect("source should build");
    let formula =
        PackedViewFormula::linear_decoded(byte_decode_terms(PackedFamilyId::RamRa { index: 0 }));

    assert_eq!(
        formula.eval_row(&source, 0).expect("view should evaluate"),
        f(7)
    );
    assert!(matches!(
        formula
            .physical_view(&layout)
            .expect("view should lower to the opening API"),
        PhysicalView::PackedLinear { layout_digest, terms }
            if layout_digest == layout.digest
                && terms.len() == 256
                && terms[7].coefficient == f(7)
                && terms[7].family == (PackedFamilyId::RamRa { index: 0 }).physical_ref()
                && terms[7].symbol == 7
    ));
}

#[test]
fn direct_view_point_eval_interpolates_rows() {
    let layout = byte_layout();
    let address = PackedCellAddress {
        family: PackedFamilyId::UnsignedIncMsb,
        row: 1,
        limb: 0,
        symbol: 1,
    };
    let source = SparsePackedWitness::try_from_cells(layout, [(address, f(1))])
        .expect("source should build");
    let formula = PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1);
    let point = [f(3)];

    assert_eq!(
        formula
            .eval_row_point(&source, &point)
            .expect("view should evaluate at point"),
        point[0]
    );
}

#[test]
fn linear_decode_point_eval_interpolates_rows() {
    let layout = byte_layout();
    let source = SparsePackedWitness::try_from_cells(
        layout,
        [
            (
                PackedCellAddress {
                    family: PackedFamilyId::RamRa { index: 0 },
                    row: 0,
                    limb: 0,
                    symbol: 7,
                },
                f(1),
            ),
            (
                PackedCellAddress {
                    family: PackedFamilyId::RamRa { index: 0 },
                    row: 1,
                    limb: 0,
                    symbol: 11,
                },
                f(1),
            ),
        ],
    )
    .expect("source should build");
    let formula =
        PackedViewFormula::linear_decoded(byte_decode_terms(PackedFamilyId::RamRa { index: 0 }));
    let point = [f(5)];
    let expected = (f(1) - point[0]) * f(7) + point[0] * f(11);

    assert_eq!(
        formula
            .eval_row_point(&source, &point)
            .expect("view should evaluate at point"),
        expected
    );
}

#[test]
fn row_point_dimension_mismatch_rejects() {
    let layout = byte_layout();
    let source = SparsePackedWitness::try_from_cells(
        layout,
        [(
            PackedCellAddress {
                family: PackedFamilyId::UnsignedIncMsb,
                row: 1,
                limb: 0,
                symbol: 1,
            },
            f(1),
        )],
    )
    .expect("source should build");
    let formula = PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1);

    assert!(matches!(
        formula.eval_row_point(&source, &[]),
        Err(PackedViewError::InvalidRowPointDimension {
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn masked_view_requires_translation_sumcheck() {
    let layout = byte_layout();
    let formula = PackedViewFormula::<Fr>::MaskedDecoded;

    assert!(matches!(
        formula.physical_view(&layout),
        Err(PackedViewError::MaskedViewRequiresTranslation)
    ));
}

#[test]
fn translation_layout_digest_mismatch_rejects() {
    let layout = byte_layout();
    let catalog_a = PackedViewCatalog::new(
        &layout,
        [PackedViewEntry::new(
            OpeningId::A,
            RelationId::First,
            PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1),
        )],
    )
    .expect("catalog should build");
    let catalog_b = PackedViewCatalog::new(
        &layout,
        [PackedViewEntry::new(
            OpeningId::A,
            RelationId::First,
            PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 0),
        )],
    )
    .expect("catalog should build");

    assert_ne!(catalog_a.digest, catalog_b.digest);
    assert!(matches!(
        catalog_a.verify_digest(&catalog_b.digest),
        Err(PackedViewError::CatalogDigestMismatch { .. })
    ));
}

#[test]
fn decoded_view_without_validity_rejects_or_is_not_enabled() {
    let layout = byte_layout();
    let formula =
        PackedViewFormula::unchecked_linear_decoded(byte_decode_terms(PackedFamilyId::RamRa {
            index: 0,
        }));

    assert!(matches!(
        formula.physical_view(&layout),
        Err(PackedViewError::DecodedViewNeedsValidity)
    ));
}

#[test]
fn same_polynomial_different_relation_ids_distinct() {
    let layout = byte_layout();
    let catalog = PackedViewCatalog::new(
        &layout,
        [
            PackedViewEntry::new(
                OpeningId::A,
                RelationId::Second,
                PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 0),
            ),
            PackedViewEntry::new(
                OpeningId::A,
                RelationId::First,
                PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1),
            ),
        ],
    )
    .expect("catalog should build");

    assert_eq!(
        catalog
            .lookup(&OpeningId::A, &RelationId::First)
            .expect("first relation should exist"),
        &PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1)
    );
    assert_eq!(
        catalog
            .lookup(&OpeningId::A, &RelationId::Second)
            .expect("second relation should exist"),
        &PackedViewFormula::<Fr>::direct(PackedFamilyId::UnsignedIncMsb, 0, 0)
    );
}

#[test]
fn bound_precommitted_program_view_formula_validates_against_supplied_layout() {
    let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::ProgramImageInit,
        PackedFactDomain::ProgramImageWords { log_words: 2 },
        8,
        PackedAlphabet::Byte,
    )])
    .expect("layout should build");
    let formula =
        PackedViewFormula::linear_decoded(byte_decode_terms(PackedFamilyId::ProgramImageInit));

    formula.validate(&layout).expect("formula should validate");
    assert_eq!(
        layout
            .family(&PackedFamilyId::ProgramImageInit)
            .expect("program family should exist")
            .domain,
        PackedFactDomain::ProgramImageWords { log_words: 2 }
    );
}
