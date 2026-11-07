#![expect(clippy::indexing_slicing, clippy::unwrap_used, reason = "test code")]

use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use jolt_algebra::{CanonicalBytes, CyclotomicRing, Field, One, Prime128OffsetA7F7, Ring};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;
use std::sync::Arc;

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
fn centered_input(len: usize, bound: u64, rng: &mut ChaCha20Rng) -> Vec<i64> {
    let bound = bound as i64;
    (0..len)
        .map(|index| match index % 11 {
            0 => bound,
            1 => -bound,
            _ => (rng.next_u64() % (2 * bound as u64 + 1)) as i64 - bound,
        })
        .collect()
}

fn hex(bytes: [u8; 32]) -> String {
    bytes
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
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

fn check_apply_centered<const D: usize>(bound: u64) -> [u8; 32] {
    let shape = MatrixShape {
        rows: 3,
        width: 130,
        ring_dim: D,
    };
    let (data, matrix) = prepared::<D>(shape, 11);
    let mut rng = ChaCha20Rng::seed_from_u64(D as u64 + bound);
    let input = centered_input(shape.width * D, bound, &mut rng);
    let expected = schoolbook(shape, &data, &lift::<D>(&input));
    let actual = matrix.apply_centered(&input, bound).unwrap();
    assert_eq!(actual, expected);
    digest(&actual)
}

/// `apply_centered` against schoolbook products across chunk boundaries,
/// with frozen digests: a change here changes the bytes of `N·y`.
#[test]
fn apply_centered_matches_schoolbook_across_chunks() {
    for (digest, expected) in [
        (
            check_apply_centered::<64>(1 << 28),
            "9c24c5494f15ec6403971f437bb2905d48f163cb9d69e1ff828977a7727ff9a5",
        ),
        (
            check_apply_centered::<64>(i32::MAX as u64),
            "3d17b525db1c7b4b5fac09f19b4155999d7c55b6ae1323444ca56b247a12c147",
        ),
        (
            check_apply_centered::<256>(1 << 28),
            "cbf8fca64e5f1bc15ae014162b6e3ce33980345b6402d714cc3ba8deeb43840a",
        ),
        (
            check_apply_centered::<256>(8),
            "5dd65520b7eb26f2f9ebdb291fe2e77408c707397e61f264a3c5d9c7e0d49da5",
        ),
    ] {
        assert_eq!(hex(digest), expected);
    }
}

/// `Mᵀ·w` on full-width field weights against schoolbook products of the
/// stream's entries.
fn check_apply_transposed<const D: usize>(rows: usize, width: usize) {
    let shape = MatrixShape {
        rows,
        width,
        ring_dim: D,
    };
    let (data, matrix) = prepared::<D>(shape, 15);
    let mut rng = ChaCha20Rng::seed_from_u64(16);
    let mut weights: Vec<CyclotomicRing<F, D>> = (0..rows)
        .map(|_| CyclotomicRing::from_coefficients(std::array::from_fn(|_| F::random(&mut rng))))
        .collect();
    // The largest weight, and limb boundaries of the radix-2^32 split.
    weights[0].coefficients_mut()[1] = -F::one();
    weights[0].coefficients_mut()[2] = F::from_u64(u32::MAX.into());
    weights[0].coefficients_mut()[3] = F::from_u128(1 << 96);
    weights[0].coefficients_mut()[4] = F::from_u64(0);
    if let Some(last) = weights.last_mut() {
        *last = CyclotomicRing::from_coefficients([-F::one(); D]);
    }
    let view = MatrixView::new(shape, &data).unwrap();
    let expected: Vec<CyclotomicRing<F, D>> = (0..width)
        .map(|col| {
            let mut out = CyclotomicRing::zero();
            for (row, weight) in weights.iter().enumerate() {
                CyclotomicRing::<F, D>::from_slice(view.entry(row, col).unwrap())
                    .mul_accumulate_into(weight, &mut out);
            }
            out
        })
        .collect();
    assert_eq!(matrix.apply_transposed(&weights).unwrap(), expected);
    assert!(matrix.apply_transposed(&weights[1..]).is_err());
}

/// `Σ_col s_col·M[row][col]` against the stream's entries.
#[test]
fn combine_columns_matches_the_stream() {
    let shape = MatrixShape {
        rows: 2,
        width: 3,
        ring_dim: 64,
    };
    let (data, matrix) = prepared::<64>(shape, 17);
    let mut rng = ChaCha20Rng::seed_from_u64(18);
    let scales: Vec<F> = (0..shape.width).map(|_| F::random(&mut rng)).collect();
    let view = MatrixView::new(shape, &data).unwrap();
    let expected: Vec<CyclotomicRing<F, 64>> = (0..shape.rows)
        .map(|row| {
            let mut out = CyclotomicRing::zero();
            for (col, scale) in scales.iter().enumerate() {
                out +=
                    CyclotomicRing::<F, 64>::from_slice(view.entry(row, col).unwrap()).scale(scale);
            }
            out
        })
        .collect();
    assert_eq!(matrix.combine_columns(&scales).unwrap(), expected);
    assert!(matrix.combine_columns(&scales[1..]).is_err());
}

#[test]
fn apply_transposed_matches_schoolbook() {
    check_apply_transposed::<64>(3, 5);
    check_apply_transposed::<128>(10, 4);
    check_apply_transposed::<256>(2, 3);
}

/// Bounds past the `i32` lift take the limb split; schoolbook is the oracle,
/// at the largest bound and at one needing two limbs.
#[test]
fn apply_centered_splits_wide_inputs_into_limbs() {
    for bound in [(1 << 40) + 3, MAX_CENTERED_BOUND] {
        let _ = check_apply_centered::<64>(bound);
        let _ = check_apply_centered::<256>(bound);
    }
    let (_, matrix) = prepared::<64>(
        MatrixShape {
            rows: 1,
            width: 1,
            ring_dim: 64,
        },
        15,
    );
    assert!(matrix
        .apply_centered(&[0; 64], MAX_CENTERED_BOUND + 1)
        .is_err());
}

/// Limb edges against schoolbook: the smallest bound that takes the limb
/// path (2^31), and inputs whose low limb is exactly −2^29 or 2^29 − 1, with
/// carries into the next limb.
#[test]
fn apply_centered_limb_edges() {
    let shape = MatrixShape {
        rows: 2,
        width: 1,
        ring_dim: 64,
    };
    let (data, matrix) = prepared::<64>(shape, 16);
    let low = 1i64 << 29;
    for (bound, top) in [(1u64 << 31, 1i64 << 31), (MAX_CENTERED_BOUND, 1 << 62)] {
        let edges = [
            -low,
            low - 1,
            -low + (1 << 30),
            low - 1 + (1 << 30),
            -low - (1 << 30),
            top,
            -top,
            top - low,
            -top + low,
            0,
        ];
        let input: Vec<i64> = (0..64).map(|i| edges[i % edges.len()]).collect();
        assert_eq!(
            matrix.apply_centered(&input, bound).unwrap(),
            schoolbook(shape, &data, &lift::<64>(&input)),
            "bound {bound}"
        );
    }
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
    let randomness = (0..randomness_columns * D)
        .map(|_| (rng.next_u32() % 16) as i8 - 8)
        .collect::<Vec<_>>();
    let commitment = commit(&level, &matrix, &message, &randomness).unwrap();

    let params = BalancedDecomposePow2Params::<F>::new(levels, log_basis);
    let radix = F::from_u64(1 << log_basis);
    let mut columns = Vec::new();
    for ring in &message {
        let mut digits = vec![0i8; levels * D];
        jolt_algebra::balanced_decompose_coefficients_pow2_i8_into(
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
    assert_eq!(
        hex(check_commit::<64>(4, 32, 2)),
        "2f053786aad127aa3023f93e4eaad5c855002ad8f4398a44c5120003bcbeb0fb"
    );
    assert_eq!(
        hex(check_commit::<256>(8, 16, 0)),
        "f02de342c6e62ffd35617e6a9c6f384f9e2da18612da0fd40538495f3439aaac"
    );
    assert_eq!(
        hex(check_commit::<256>(4, 32, 1)),
        "74812bea54b3446061a0607cc2323c25fb99228cf62a61984fe55960aede6960"
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

#[cfg(feature = "sample")]
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
fn two_level_layout_checks_its_shape() {
    let layout = headline_two_level(256);
    layout.validate::<F>().unwrap();
    headline_two_level(64).validate::<F>().unwrap();
    let mut bad = layout;
    bad.outer.message_rings += 1;
    assert!(bad.validate::<F>().is_err());
    let mut wide = layout;
    wide.outer.matrix.width += 1;
    assert!(wide.validate::<F>().is_err());
}

#[test]
fn offset_view_products_use_only_the_bounded_entries() {
    const D: usize = 64;
    let data = stream(12 * D, 42);
    let matrix = PreparedMatrix::<F, D>::from_shared(
        MatrixShape {
            rows: 3,
            width: 4,
            ring_dim: D,
        },
        Arc::new(data.clone()),
    )
    .unwrap();
    let shape = MatrixShape {
        rows: 1,
        width: 2,
        ring_dim: D,
    };
    let offset = 10 * D;
    let view = matrix.view(shape, offset).unwrap();
    let input: Vec<_> = (0..2 * D).map(|index| (index % 7) as i64 - 3).collect();
    assert_eq!(
        view.apply_centered(&input, 3).unwrap(),
        schoolbook(shape, &data[offset..], &lift::<D>(&input))
    );
    let scales = [F::from_u64(3), F::from_u64(5)];
    let expected = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
        data[offset + index] * scales[0] + data[offset + D + index] * scales[1]
    }));
    assert_eq!(view.combine_columns(&scales).unwrap(), vec![expected]);
    assert!(view.view(shape, D).is_err());
    assert!(matrix.view(shape, usize::MAX).is_err());
}

#[test]
fn public_stream_rejects_an_unallocatable_length() {
    let error =
        derive_public_matrix_prefix::<F>(usize::MAX, &PublicMatrixSeed::shake256_paged_v1([0; 32]))
            .unwrap_err();
    assert!(matches!(error, CryptoError::InvalidSetup(_)));
}
