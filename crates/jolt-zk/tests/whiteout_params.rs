//! The Σ-protocol's segment table and parameter derivation (Whiteout §6.2).

#![expect(clippy::unwrap_used, reason = "tests unwrap well-formed inputs")]

use jolt_zk::whiteout::{
    Segment, SegmentMask, SegmentRole, SegmentShape, SegmentTable, SigmaParams, WhiteoutError,
};

fn segment(
    role: SegmentRole,
    len: usize,
    log_basis: u32,
    shape: SegmentShape,
    mask: SegmentMask,
) -> Segment {
    Segment {
        role,
        len,
        native_dim: 256,
        log_basis,
        shape,
        mask,
    }
}

/// The headline residual vector at T = 2^24, k = 2 (`scripts/whiteout-costs.py`),
/// each segment rounded up to whole R_256 elements.
fn headline(operand_shape: SegmentShape) -> SegmentTable {
    SegmentTable::new(vec![
        segment(
            SegmentRole::Witness,
            98_304,
            4,
            SegmentShape::Dense,
            SegmentMask::Public,
        ),
        segment(
            SegmentRole::Witness,
            85_480,
            4,
            SegmentShape::Dense,
            SegmentMask::Gaussian,
        ),
        segment(
            SegmentRole::InnerDigits,
            9_216,
            4,
            SegmentShape::Dense,
            SegmentMask::Public,
        ),
        segment(
            SegmentRole::InnerDigits,
            8_064,
            4,
            SegmentShape::Dense,
            SegmentMask::Gaussian,
        ),
        segment(
            SegmentRole::ResidualRandomness,
            388,
            4,
            SegmentShape::Dense,
            SegmentMask::Gaussian,
        ),
        segment(
            SegmentRole::PadDigits,
            948,
            4,
            SegmentShape::Dense,
            SegmentMask::Gaussian,
        ),
        segment(
            SegmentRole::OperandDigits,
            1_904,
            8,
            operand_shape,
            SegmentMask::Gaussian,
        ),
        segment(
            SegmentRole::PadRandomness,
            420,
            4,
            SegmentShape::Dense,
            SegmentMask::Gaussian,
        ),
    ])
    .unwrap()
}

#[test]
fn headline_parameters_match_the_closed_form() {
    let params = SigmaParams::derive(&headline(SegmentShape::ConstantPoly)).unwrap();
    let widths = params
        .blocks()
        .iter()
        .map(|block| block.width)
        .collect::<Vec<_>>();
    // σ_b = αΓ·8·√ΣN for radix-16 segments, αΓ·16·√ΣN for the operand
    // digit polynomials, k_b = σ_b √(2 ln 2), rounded up, plus one.
    assert_eq!(
        widths,
        [6_766_161, 6_766_161, 6_766_161, 6_766_161, 13_532_321, 6_766_161]
    );
    assert_eq!(params.response_digits(), [3, 7, 3, 7, 7, 7, 7, 7]);
    assert!(params
        .blocks()
        .iter()
        .all(|block| block.box_bound == 125_269_879));
    assert_eq!(params.max_attempts(), 163);
    let _rejection = params.rejection().unwrap();
}

#[test]
fn width_check_rejects_narrower_widths() {
    let table = headline(SegmentShape::ConstantPoly);
    let derived = SigmaParams::derive(&table).unwrap();
    let narrow = derived
        .blocks()
        .iter()
        .map(|block| block.width - block.width / 100)
        .collect::<Vec<_>>();
    assert!(matches!(
        SigmaParams::new(&table, &narrow),
        Err(WhiteoutError::Parameters(_))
    ));
}

#[test]
fn width_check_depends_on_the_constant_polynomial_shape() {
    // Operand digit polynomials are constant, so their ℓ₂ norm is 8× below
    // their ∞-norm bound: widths derived for that shape fail for a dense
    // operand segment.
    let widths = SigmaParams::derive(&headline(SegmentShape::ConstantPoly))
        .unwrap()
        .blocks()
        .iter()
        .map(|block| block.width)
        .collect::<Vec<_>>();
    assert!(SigmaParams::new(&headline(SegmentShape::Dense), &widths).is_err());
}

fn small_table() -> SegmentTable {
    SegmentTable::new(vec![
        Segment {
            role: SegmentRole::PadDigits,
            len: 4,
            native_dim: 256,
            log_basis: 4,
            shape: SegmentShape::Dense,
            mask: SegmentMask::Gaussian,
        },
        Segment {
            role: SegmentRole::OperandDigits,
            len: 4,
            native_dim: 256,
            log_basis: 8,
            shape: SegmentShape::ConstantPoly,
            mask: SegmentMask::Gaussian,
        },
    ])
    .unwrap()
}

fn honest_vector() -> Vec<i64> {
    let mut s = (0..256).map(|i| (i % 16) - 8).collect::<Vec<i64>>();
    // Constant R_64 polynomials packed by stride: native positions 0..4 of
    // the one R_256 element.
    s.extend([-128, 127, 5, -1]);
    s.extend([0; 252]);
    s
}

#[test]
fn private_vector_check_accepts_the_declared_alphabets() {
    small_table()
        .check_private_vector(&honest_vector())
        .unwrap();
}

#[test]
fn private_vector_check_rejects_alphabet_and_shape_violations() {
    let table = small_table();
    let mut digit = honest_vector();
    digit[3] = 8;
    assert_eq!(
        table.check_private_vector(&digit),
        Err(WhiteoutError::WitnessOutsideAlphabet)
    );
    let mut constant = honest_vector();
    constant[256 + 4] = 1;
    assert_eq!(
        table.check_private_vector(&constant),
        Err(WhiteoutError::WitnessOutsideAlphabet)
    );
    assert!(matches!(
        table.check_private_vector(&honest_vector()[..511]),
        Err(WhiteoutError::WitnessLength { .. })
    ));
}

#[test]
fn segments_must_hold_whole_native_elements() {
    let mut bad = small_table().segments()[0];
    bad.len = 3;
    assert!(matches!(
        SegmentTable::new(vec![bad]),
        Err(WhiteoutError::Segment { index: 0, .. })
    ));
}
