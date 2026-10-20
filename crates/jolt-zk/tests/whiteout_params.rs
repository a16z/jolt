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
    assert_eq!(params.max_attempts(), Ok(163));
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
        Err(WhiteoutError::WidthsTooNarrow)
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
    assert_eq!(
        SigmaParams::new(&headline(SegmentShape::Dense), &widths),
        Err(WhiteoutError::WidthsTooNarrow)
    );
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
    // Above and below the radix-16 alphabet, the i64 extremes, a constant
    // coefficient outside the radix-256 alphabet, and a nonzero coefficient
    // off the constant positions.
    for (index, value) in [
        (3, 8),
        (3, -9),
        (3, i64::MIN),
        (3, i64::MAX),
        (256 + 2, 128),
        (256 + 2, -129),
        (256 + 4, 1),
        (511, i64::MIN),
    ] {
        let mut s = honest_vector();
        s[index] = value;
        assert_eq!(
            table.check_private_vector(&s),
            Err(WhiteoutError::WitnessOutsideAlphabet),
            "s[{index}] = {value}"
        );
    }
    assert!(matches!(
        table.check_private_vector(&honest_vector()[..511]),
        Err(WhiteoutError::WitnessLength { .. })
    ));
}

#[test]
fn segments_must_hold_whole_native_elements() {
    let mut bad = *small_table().segments().next().unwrap();
    bad.len = 3;
    assert!(matches!(
        SegmentTable::new(vec![bad]),
        Err(WhiteoutError::Segment { index: 0, .. })
    ));
}

#[test]
fn widths_must_match_the_gaussian_segments() {
    let table = small_table();
    assert_eq!(
        SigmaParams::new(&table, &[1 << 20]),
        Err(WhiteoutError::WidthCount {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        SigmaParams::new(&table, &[0, 1 << 20]),
        Err(WhiteoutError::WidthOutOfRange { width: 0 })
    );
    assert_eq!(
        SigmaParams::new(&table, &[1 << 20, (1 << 31) + 1]),
        Err(WhiteoutError::WidthOutOfRange {
            width: (1 << 31) + 1
        })
    );
}

#[test]
fn parameters_outside_the_rejection_arithmetic_are_rejected() {
    // Three distinct widths near 2^24 make the product of the distinct k_b²
    // exceed 2^126; the widths themselves are wide enough.
    let table = SegmentTable::new(
        (0..3)
            .map(|_| *small_table().segments().next().unwrap())
            .collect(),
    )
    .unwrap();
    assert_eq!(
        SigmaParams::new(&table, &[1 << 24, (1 << 24) - 1, (1 << 24) - 2]),
        Err(WhiteoutError::RejectionArithmetic)
    );
    let accepted = SigmaParams::new(&table, &[1 << 24, 1 << 24, (1 << 24) - 1]).unwrap();
    let _rejection = accepted.rejection().unwrap();
    // Widths of 2^31 need nine response digits, past the Ajtai map's i32
    // lift: rejected before any proof could fail on it.
    assert_eq!(
        SigmaParams::new(&table, &[1 << 31; 3]),
        Err(WhiteoutError::ResponseBoundTooLarge)
    );
}
