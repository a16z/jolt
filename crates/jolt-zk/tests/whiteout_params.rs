//! The Σ-protocol's segment table and parameter derivation (Whiteout §6.2).

#![expect(clippy::unwrap_used, reason = "tests unwrap well-formed inputs")]

use jolt_crypto::lattice::required_randomness_columns;
use jolt_field::Prime128OffsetA7F7;
use jolt_zk::whiteout::{
    binding_collision_radius, Segment, SegmentMask, SegmentRole, SegmentShape, SegmentTable,
    SigmaParams, WhiteoutError, BINDING_RESPONSE_DIGITS,
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

/// Rank over R_256 of the residual commitment's A and [B|B′] and of the pad
/// commitment at the headline: a setup input (Akita's planner sizes it for
/// the eight-digit collision radius, `specs/whiteout-sigma.md` §5.2), owned
/// here for the whole table.
const RANK: usize = 5;
/// Inner-digit R_64 elements per unit of A's rank: Gaussian and public.
const INNER_DIGITS_PER_RANK: [usize; 2] = [2_688, 3_072];

/// R_64 elements of hiding randomness for an image of `RANK` R_256 elements
/// plus `extra` coefficients, at the radix-16 digits and 8 bits of slack of
/// `specs/whiteout-sigma.md` §1.5.
fn randomness_len(extra: usize) -> usize {
    let columns =
        required_randomness_columns::<Prime128OffsetA7F7>(RANK * 256, extra, 256, 4, 8).unwrap();
    columns * 4
}

/// A residual vector at T = 2^24, k = 2 sizes (`scripts/whiteout-costs.py`),
/// each segment rounded up to whole R_256 elements, at commitment rank
/// [`RANK`].
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
            INNER_DIGITS_PER_RANK[1] * RANK,
            4,
            SegmentShape::Dense,
            SegmentMask::Public,
        ),
        segment(
            SegmentRole::InnerDigits,
            INNER_DIGITS_PER_RANK[0] * RANK,
            4,
            SegmentShape::Dense,
            SegmentMask::Gaussian,
        ),
        segment(
            SegmentRole::ResidualRandomness,
            randomness_len(0),
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
            // The pad commitment's randomness also hides the row L.
            randomness_len(64),
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
    // σ = αΓ·√(Σ_b γ_b²), k = σ √(2 ln 2), rounded up, plus one.
    assert_eq!(params.width(), WIDTH);
    assert_eq!(params.response_digits(), [3, 8, 3, 8, 8, 8, 8, 8]);
    assert!(params
        .blocks()
        .iter()
        .all(|block| block.box_bound == 2_004_318_071));
    // Per-attempt rejection about 2^{−5.23}: 25 attempts reach 2^{−128}.
    assert_eq!(params.max_attempts(), Ok(25));
    let _rejection = params.rejection().unwrap();
}

/// The binding radius is the paper's coefficientwise `4Γ_Δ B` with
/// `Γ_Δ = 2κ₁ = 106` (paper: preliminaries) and `B` the eight-digit
/// radix-16 box magnitude `8(16^8 − 1)/15`.
#[test]
fn binding_radius_is_four_gamma_delta_times_the_eight_digit_box() {
    assert_eq!(BINDING_RESPONSE_DIGITS, 8);
    let magnitude: u128 = 8 * (16u128.pow(8) - 1) / 15;
    assert_eq!(binding_collision_radius(), 4 * 106 * magnitude);
    assert_eq!(binding_collision_radius(), 971_235_270_976);
}

/// A table whose responses need a ninth digit leaves the box the
/// commitments were priced for, and is rejected.
#[test]
fn responses_above_the_binding_digits_are_rejected() {
    let headline = headline(SegmentShape::ConstantPoly);
    let mut segments = headline.segments().copied().collect::<Vec<_>>();
    segments[1].len *= 400;
    let wide = SegmentTable::new(segments).unwrap();
    assert_eq!(
        SigmaParams::derive(&wide),
        Err(WhiteoutError::BindingDigitsExceeded { digits: 9 })
    );
}

/// The headline's derived width multiplier.
const WIDTH: u64 = 229_072_314;

#[test]
fn width_check_rejects_narrower_widths() {
    let table = headline(SegmentShape::ConstantPoly);
    let width = SigmaParams::derive(&table).unwrap().width();
    assert!(matches!(
        SigmaParams::new(&table, width - width / 100),
        Err(WhiteoutError::WidthTooNarrow)
    ));
}

#[test]
fn width_check_depends_on_the_constant_polynomial_shape() {
    // Operand digit polynomials are constant, so their ℓ₂ norm is 8× below
    // their ∞-norm bound: the width derived for that shape fails for a dense
    // operand segment.
    let width = SigmaParams::derive(&headline(SegmentShape::ConstantPoly))
        .unwrap()
        .width();
    assert_eq!(
        SigmaParams::new(&headline(SegmentShape::Dense), width),
        Err(WhiteoutError::WidthTooNarrow)
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
fn width_must_be_a_sampler_width_and_the_table_must_mask_something() {
    let table = small_table();
    assert_eq!(
        SigmaParams::new(&table, 0),
        Err(WhiteoutError::WidthOutOfRange { width: 0 })
    );
    let public = SegmentTable::new(vec![Segment {
        mask: SegmentMask::Public,
        ..*table.segments().next().unwrap()
    }])
    .unwrap();
    assert_eq!(
        SigmaParams::new(&public, 1 << 20),
        Err(WhiteoutError::NoGaussianSegment)
    );
}
