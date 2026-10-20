#![expect(clippy::indexing_slicing, clippy::unwrap_used, reason = "test code")]

use jolt_field::{CanonicalBytes, CyclotomicRing, Field, Prime128OffsetA7F7, Ring};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;

use super::*;
use crate::CryptoError;

type F = Prime128OffsetA7F7;

fn stream(len: usize, seed: u8) -> Vec<F> {
    derive_public_matrix_prefix::<F>(len, &PublicMatrixSeed::shake256_paged_v1([seed; 32])).unwrap()
}

fn prepared<const D: usize>(shape: MatrixShape, seed: u8) -> (Vec<F>, PreparedMatrix<F, D>) {
    let data = stream(shape.num_field_elements().unwrap(), seed);
    let matrix = PreparedMatrix::new(&MatrixView::new(shape, &data).unwrap()).unwrap();
    (data, matrix)
}

/// `M · x` by schoolbook negacyclic products in `F`, the ground truth.
fn schoolbook<const D: usize>(
    shape: MatrixShape,
    data: &[F],
    columns: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let view = MatrixView::new(shape, data).unwrap();
    (0..shape.rows)
        .map(|row| {
            let mut out = CyclotomicRing::zero();
            for (col, x) in columns.iter().enumerate() {
                CyclotomicRing::<F, D>::from_slice(view.entry(row, col).unwrap())
                    .mul_accumulate_into(x, &mut out);
            }
            out
        })
        .collect()
}

fn lift<const D: usize>(input: &[i64]) -> Vec<CyclotomicRing<F, D>> {
    input
        .chunks_exact(D)
        .map(|column| {
            CyclotomicRing::from_slice(&column.iter().map(|&v| F::from_i64(v)).collect::<Vec<_>>())
        })
        .collect()
}

/// Centered inputs at `bound`, including both extremes.
fn centered_input(len: usize, bound: u32, rng: &mut ChaCha20Rng) -> Vec<i64> {
    let bound = i64::from(bound);
    (0..len)
        .map(|index| match index % 11 {
            0 => bound,
            1 => -bound,
            _ => (rng.next_u64() % (2 * bound as u64 + 1)) as i64 - bound,
        })
        .collect()
}

fn digest<const D: usize>(rings: &[CyclotomicRing<F, D>]) -> [u8; 32] {
    let mut xof = Shake256::default();
    let mut bytes = [0u8; 16];
    for coefficient in rings.iter().flat_map(|ring| ring.coefficients()) {
        coefficient.to_bytes_le(&mut bytes);
        xof.update(&bytes);
    }
    let mut out = [0u8; 32];
    xof.finalize_xof().read(&mut out);
    out
}

fn check_apply_centered<const D: usize>(bound: u32) {
    let shape = MatrixShape {
        rows: 3,
        width: 130,
        ring_dim: D,
    };
    let (data, matrix) = prepared::<D>(shape, 11);
    let mut rng = ChaCha20Rng::seed_from_u64(D as u64 + u64::from(bound));
    let input = centered_input(shape.width * D, bound, &mut rng);
    let expected = schoolbook(shape, &data, &lift::<D>(&input));
    assert_eq!(matrix.apply_centered(&input, bound).unwrap(), expected);
}

#[test]
fn apply_centered_matches_schoolbook_across_chunks() {
    check_apply_centered::<64>(1 << 28);
    check_apply_centered::<64>(i32::MAX as u32);
    check_apply_centered::<256>(1 << 28);
    check_apply_centered::<256>(8);
}

#[test]
fn apply_matches_schoolbook_on_field_columns() {
    let shape = MatrixShape {
        rows: 2,
        width: 5,
        ring_dim: 64,
    };
    let (data, matrix) = prepared::<64>(shape, 12);
    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let columns = (0..shape.width)
        .map(|_| {
            CyclotomicRing::from_slice(&(0..64).map(|_| F::random(&mut rng)).collect::<Vec<_>>())
        })
        .collect::<Vec<CyclotomicRing<F, 64>>>();
    assert_eq!(
        matrix.apply(&columns).unwrap(),
        schoolbook(shape, &data, &columns)
    );
}

#[test]
fn apply_centered_equals_apply_of_the_lift() {
    let shape = MatrixShape {
        rows: 2,
        width: 4,
        ring_dim: 256,
    };
    let (_, matrix) = prepared::<256>(shape, 13);
    let mut rng = ChaCha20Rng::seed_from_u64(6);
    let input = centered_input(shape.width * 256, 1 << 20, &mut rng);
    assert_eq!(
        matrix.apply_centered(&input, 1 << 20).unwrap(),
        matrix.apply(&lift::<256>(&input)).unwrap()
    );
}

#[test]
fn apply_centered_rejects_inputs_above_the_bound() {
    let shape = MatrixShape {
        rows: 1,
        width: 2,
        ring_dim: 64,
    };
    let (_, matrix) = prepared::<64>(shape, 14);
    let mut input = vec![0i64; 128];
    input[77] = -9;
    assert!(matches!(
        matrix.apply_centered(&input, 8),
        Err(CryptoError::InvalidInput(_))
    ));
    assert!(matrix.apply_centered(&input[..64], 8).is_err());
}

fn level<const D: usize>(
    log_basis: u32,
    levels: usize,
    message_rings: usize,
    randomness_columns: usize,
) -> AjtaiLevel {
    AjtaiLevel {
        matrix: MatrixShape {
            rows: 2,
            width: message_rings * levels + randomness_columns,
            ring_dim: D,
        },
        digits: GadgetDigits { log_basis, levels },
        message_rings,
        randomness_columns,
    }
}

/// The commitment's columns are the message's digit planes, ring-major and
/// digit-minor, then the randomness; the planes reconstruct the message.
fn check_commit<const D: usize>(
    log_basis: u32,
    levels: usize,
    randomness_columns: usize,
) -> [u8; 32] {
    let level = level::<D>(log_basis, levels, 3, randomness_columns);
    let (data, matrix) = prepared::<D>(level.matrix, 15);
    let mut rng = ChaCha20Rng::seed_from_u64(u64::from(log_basis));
    let message = (0..level.message_rings)
        .map(|_| {
            CyclotomicRing::from_slice(&(0..D).map(|_| F::random(&mut rng)).collect::<Vec<_>>())
        })
        .collect::<Vec<CyclotomicRing<F, D>>>();
    let randomness = sample_box_digits(&mut rng, randomness_columns * D, 4).unwrap();
    let commitment = commit(&level, &matrix, &message, &randomness).unwrap();

    let params =
        jolt_field::ring::cyclotomic::BalancedDecomposePow2Params::<F>::new(levels, log_basis);
    let radix = F::from_u64(1 << log_basis);
    let mut columns = Vec::new();
    for ring in &message {
        let mut digits = vec![0i8; levels * D];
        jolt_field::balanced_decompose_coefficients_pow2_i8_into(
            ring.coefficients(),
            &mut digits,
            &params,
        );
        let planes = digits
            .chunks_exact(D)
            .map(|plane| {
                CyclotomicRing::from_slice(
                    &plane
                        .iter()
                        .map(|&d| F::from_i64(d.into()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<CyclotomicRing<F, D>>>();
        let rebuilt = planes
            .iter()
            .rev()
            .fold(CyclotomicRing::zero(), |acc, plane| {
                acc.scale(&radix) + *plane
            });
        assert_eq!(&rebuilt, ring);
        columns.extend(planes);
    }
    columns.extend(lift::<D>(
        &randomness.iter().map(|&r| i64::from(r)).collect::<Vec<_>>(),
    ));
    assert_eq!(commitment, schoolbook(level.matrix, &data, &columns));
    digest(&commitment)
}

/// `commit` against its column definition, with frozen digests: a change
/// here changes commitment bytes.
#[test]
fn commit_digests_are_frozen() {
    let hex = |bytes: [u8; 32]| {
        bytes
            .iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect::<String>()
    };
    assert_eq!(
        hex(check_commit::<64>(4, 32, 2)),
        "310ebca88dbd3fe51d9c5eed0140036d211707d0b84fe0c871238d2af458eacc"
    );
    assert_eq!(
        hex(check_commit::<256>(8, 16, 0)),
        "f02de342c6e62ffd35617e6a9c6f384f9e2da18612da0fd40538495f3439aaac"
    );
    assert_eq!(
        hex(check_commit::<256>(4, 32, 1)),
        "7becc5408f96577627857ff16ad0582183a2c6ee903381a3f5aa89b150efa8a6"
    );
}

#[test]
fn views_reject_truncated_streams() {
    let shape = MatrixShape {
        rows: 2,
        width: 3,
        ring_dim: 64,
    };
    let data = stream(2 * 3 * 64 - 1, 16);
    assert!(matches!(
        MatrixView::new(shape, &data),
        Err(CryptoError::InvalidSize {
            expected: 384,
            actual: 383
        })
    ));
    let data = stream(384, 16);
    let view = MatrixView::new(shape, &data).unwrap();
    assert!(view.entry(2, 0).is_err());
    assert!(view.entry(0, 3).is_err());
    assert_eq!(view.entry(1, 2).unwrap(), &data[320..384]);
}

#[test]
fn commit_rejects_inconsistent_levels() {
    let mut bad = level::<64>(4, 32, 3, 1);
    let (_, matrix) = prepared::<64>(bad.matrix, 17);
    bad.matrix.width += 1;
    assert!(commit(&bad, &matrix, &[], &[]).is_err());
    let short = level::<64>(4, 16, 1, 0);
    let (_, matrix) = prepared::<64>(short.matrix, 17);
    assert!(commit(&short, &matrix, &[CyclotomicRing::zero()], &[]).is_err());
}

#[test]
fn randomness_columns_match_the_headline_sizes() {
    // Whiteout's pad commitment (rank 3 over R_256, plus the 64-coefficient
    // row L) and residual commitment, at 8 bits of slack (§6.2).
    assert_eq!(
        required_randomness_columns::<F>(768, 64, 256, 4, 8).unwrap(),
        105
    );
    assert_eq!(
        required_randomness_columns::<F>(768, 0, 256, 4, 8).unwrap(),
        97
    );
}

#[test]
fn box_digits_cover_exactly_the_balanced_alphabet() {
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let digits = sample_box_digits(&mut rng, 1 << 14, 4).unwrap();
    assert!(digits.iter().all(|d| (-8..=7).contains(d)));
    assert!((-8..=7).all(|v| digits.contains(&v)));
    assert!(sample_box_digits(&mut rng, 1, 9).is_err());
}

fn headline_two_level(outer_dim: usize) -> TwoLevelLayout {
    let blocks = BlockGeometry {
        live_rings: 45 * 1024,
        rings_per_block: 1024,
        live_blocks: 45,
    };
    let inner = AjtaiLevel {
        matrix: MatrixShape {
            rows: 3,
            width: 1024,
            ring_dim: 256,
        },
        digits: GadgetDigits {
            log_basis: 4,
            levels: 1,
        },
        message_rings: 1024,
        randomness_columns: 0,
    };
    let message_rings = 3 * (256 / outer_dim) * 45;
    let outer = AjtaiLevel {
        matrix: MatrixShape {
            rows: 3,
            width: message_rings * 32 + 97,
            ring_dim: outer_dim,
        },
        digits: GadgetDigits {
            log_basis: 4,
            levels: 32,
        },
        message_rings,
        randomness_columns: 97,
    };
    TwoLevelLayout {
        blocks,
        num_polys: 1,
        inner,
        outer,
        slices: SliceCount::One,
    }
}

#[test]
fn two_level_layout_checks_shape_and_module_structure() {
    let layout = headline_two_level(256);
    layout.validate().unwrap();
    assert!(layout.is_module_over_challenge_ring());
    let split = headline_two_level(64);
    split.validate().unwrap();
    assert!(!split.is_module_over_challenge_ring());
    let mut bad = layout;
    bad.outer.message_rings += 1;
    assert!(bad.validate().is_err());
}
