//! Fast commitments to binary columns over a shared BN254 G1 base set.

use ark_bn254::{Fq, G1Affine, G1Projective};
use ark_ff::{Field, One, Zero};
use rayon::prelude::*;

use super::{Bn254G1, Bn254G1Affine};

const BASES_PER_GROUP: usize = 6;
const SUBSETS_PER_GROUP: usize = 1 << BASES_PER_GROUP;
const GROUPS_PER_CHUNK: usize = 256;
const GROUPED_THRESHOLD_PER_BASE: usize = 24;

/// Commits binary columns as affine subset sums over `bases`.
///
/// Zero bytes skip the corresponding base; nonzero bytes select it. Every
/// column must have the same length as `bases`.
pub fn g1_bit_columns_msm(bases: &[Bn254G1Affine], columns: &[&[u8]]) -> Vec<Bn254G1> {
    assert!(
        columns.iter().all(|column| column.len() == bases.len()),
        "bit column/base length mismatch"
    );

    let bases = Bn254G1Affine::as_inner_slice(bases);
    let selected_per_column: Vec<usize> = columns
        .par_iter()
        .map(|column| column.iter().filter(|&&bit| bit != 0).count())
        .collect();
    let selected = selected_per_column.iter().sum::<usize>();
    if selected <= bases.len().saturating_mul(GROUPED_THRESHOLD_PER_BASE) {
        columns
            .par_iter()
            .zip(selected_per_column)
            .map(|(column, selected)| commit_column(bases, column, selected))
            .collect()
    } else {
        commit_grouped(bases, columns)
    }
}

#[expect(
    clippy::indexing_slicing,
    reason = "pair and carry indices are bounded by the active scratch length"
)]
fn commit_column(bases: &[G1Affine], column: &[u8], selected: usize) -> Bn254G1 {
    let mut points = Vec::with_capacity(selected);
    points.extend(
        column
            .iter()
            .zip(bases)
            .filter_map(|(&bit, &base)| (bit != 0).then_some(base)),
    );
    if points.is_empty() {
        return Bn254G1::from(G1Projective::zero());
    }

    let mut denominators = Vec::with_capacity(points.len() / 2);
    let mut prefixes = Vec::with_capacity(points.len() / 2);
    let mut len = points.len();
    while len > 1 {
        let pairs = len / 2;
        denominators.clear();
        denominators.extend((0..pairs).map(|i| pair_denominator(points[2 * i], points[2 * i + 1])));
        batch_invert(&mut denominators, &mut prefixes);

        for i in 0..pairs {
            points[i] = add_affine(points[2 * i], points[2 * i + 1], denominators[i]);
        }
        if len % 2 == 1 {
            points[pairs] = points[len - 1];
        }
        len = len.div_ceil(2);
    }

    Bn254G1::from(G1Projective::from(points[0]))
}

fn commit_grouped(bases: &[G1Affine], columns: &[&[u8]]) -> Vec<Bn254G1> {
    let group_count = bases.len().div_ceil(BASES_PER_GROUP);
    let chunk_count = group_count.div_ceil(GROUPS_PER_CHUNK);
    let partials: Vec<Vec<G1Projective>> = (0..chunk_count)
        .into_par_iter()
        .map(|chunk| commit_group_chunk(bases, columns, chunk))
        .collect();

    #[expect(
        clippy::indexing_slicing,
        reason = "every chunk result has one entry per input column"
    )]
    (0..columns.len())
        .into_par_iter()
        .map(|column| {
            Bn254G1::from(
                partials
                    .iter()
                    .fold(G1Projective::zero(), |sum, chunk| sum + chunk[column]),
            )
        })
        .collect()
}

#[expect(
    clippy::indexing_slicing,
    reason = "group, subset, column, and pair indices are bounded by their flat layouts"
)]
fn commit_group_chunk(bases: &[G1Affine], columns: &[&[u8]], chunk: usize) -> Vec<G1Projective> {
    let first_group = chunk * GROUPS_PER_CHUNK;
    let group_count = bases
        .len()
        .div_ceil(BASES_PER_GROUP)
        .saturating_sub(first_group)
        .min(GROUPS_PER_CHUNK);
    let identity = G1Affine::identity();
    let mut subsets = vec![identity; group_count * SUBSETS_PER_GROUP];

    for group in 0..group_count {
        let first_base = (first_group + group) * BASES_PER_GROUP;
        let bases_in_group = (bases.len() - first_base).min(BASES_PER_GROUP);
        for bit in 0..bases_in_group {
            subsets[group * SUBSETS_PER_GROUP + (1 << bit)] = bases[first_base + bit];
        }
    }

    let mut denominators = Vec::with_capacity(group_count * 20);
    let mut prefixes = Vec::with_capacity(group_count * 20);
    for weight in 2..=BASES_PER_GROUP {
        denominators.clear();
        for group in 0..group_count {
            let first_base = (first_group + group) * BASES_PER_GROUP;
            let bases_in_group = (bases.len() - first_base).min(BASES_PER_GROUP);
            for subset in 1usize..(1 << bases_in_group) {
                if subset.count_ones() as usize == weight {
                    let singleton = 1 << subset.trailing_zeros();
                    let offset = group * SUBSETS_PER_GROUP;
                    denominators.push(pair_denominator(
                        subsets[offset + (subset ^ singleton)],
                        subsets[offset + singleton],
                    ));
                }
            }
        }
        batch_invert(&mut denominators, &mut prefixes);
        let mut inverse = 0;
        for group in 0..group_count {
            let first_base = (first_group + group) * BASES_PER_GROUP;
            let bases_in_group = (bases.len() - first_base).min(BASES_PER_GROUP);
            for subset in 1usize..(1 << bases_in_group) {
                if subset.count_ones() as usize == weight {
                    let singleton = 1 << subset.trailing_zeros();
                    let offset = group * SUBSETS_PER_GROUP;
                    subsets[offset + subset] = add_affine(
                        subsets[offset + (subset ^ singleton)],
                        subsets[offset + singleton],
                        denominators[inverse],
                    );
                    inverse += 1;
                }
            }
        }
    }

    let mut points = vec![identity; columns.len() * group_count];
    let mut lengths = vec![0usize; columns.len()];
    for (column_index, column) in columns.iter().enumerate() {
        let point_offset = column_index * group_count;
        for group in 0..group_count {
            let first_base = (first_group + group) * BASES_PER_GROUP;
            let bases_in_group = (column.len() - first_base).min(BASES_PER_GROUP);
            let mut subset = 0;
            for bit in 0..bases_in_group {
                subset |= usize::from(column[first_base + bit] != 0) << bit;
            }
            if subset != 0 {
                points[point_offset + lengths[column_index]] =
                    subsets[group * SUBSETS_PER_GROUP + subset];
                lengths[column_index] += 1;
            }
        }
    }

    reduce_columns(&mut points, &mut lengths, group_count);
    lengths
        .into_iter()
        .enumerate()
        .map(|(column, len)| {
            if len == 0 {
                G1Projective::zero()
            } else {
                G1Projective::from(points[column * group_count])
            }
        })
        .collect()
}

#[expect(
    clippy::indexing_slicing,
    reason = "pair and carry indices are bounded by each column's active scratch length"
)]
fn reduce_columns(points: &mut [G1Affine], lengths: &mut [usize], stride: usize) {
    let mut denominators = Vec::with_capacity(points.len() / 2);
    let mut prefixes = Vec::with_capacity(points.len() / 2);
    loop {
        denominators.clear();
        for (column, &len) in lengths.iter().enumerate() {
            let offset = column * stride;
            for pair in 0..len / 2 {
                denominators.push(pair_denominator(
                    points[offset + 2 * pair],
                    points[offset + 2 * pair + 1],
                ));
            }
        }
        if denominators.is_empty() {
            return;
        }
        batch_invert(&mut denominators, &mut prefixes);

        let mut inverse = 0;
        for (column, len) in lengths.iter_mut().enumerate() {
            let offset = column * stride;
            let pairs = *len / 2;
            for pair in 0..pairs {
                points[offset + pair] = add_affine(
                    points[offset + 2 * pair],
                    points[offset + 2 * pair + 1],
                    denominators[inverse],
                );
                inverse += 1;
            }
            if *len % 2 == 1 {
                points[offset + pairs] = points[offset + *len - 1];
            }
            *len = (*len).div_ceil(2);
        }
    }
}

#[expect(
    clippy::expect_used,
    reason = "pair_denominator replaces every exceptional zero denominator with one"
)]
fn batch_invert(values: &mut [Fq], prefixes: &mut Vec<Fq>) {
    if values.is_empty() {
        return;
    }
    prefixes.clear();
    let mut product = Fq::one();
    for value in values.iter() {
        prefixes.push(product);
        product *= value;
    }
    let mut inverse = product.inverse().expect("denominator product is nonzero");
    for (value, prefix) in values.iter_mut().zip(prefixes.iter()).rev() {
        let original = *value;
        *value = inverse * prefix;
        inverse *= original;
    }
}

#[inline]
fn pair_denominator(p: G1Affine, q: G1Affine) -> Fq {
    if p.infinity || q.infinity {
        Fq::one()
    } else if p.x == q.x {
        if p.y == q.y {
            p.y + p.y
        } else {
            Fq::one()
        }
    } else {
        q.x - p.x
    }
}

#[inline]
fn add_affine(p: G1Affine, q: G1Affine, denominator_inverse: Fq) -> G1Affine {
    if p.infinity {
        return q;
    }
    if q.infinity {
        return p;
    }
    if p.x == q.x && p.y != q.y {
        return G1Affine::identity();
    }

    let numerator = if p == q {
        let x_squared = p.x.square();
        x_squared + x_squared + x_squared
    } else {
        q.y - p.y
    };
    let lambda = numerator * denominator_inverse;
    let x = lambda.square() - p.x - q.x;
    let y = lambda * (p.x - x) - p.y;
    G1Affine::new_unchecked(x, y)
}

#[cfg(test)]
#[expect(clippy::indexing_slicing, reason = "tests index fixture columns")]
mod tests {
    use ark_ec::CurveGroup;
    use ark_ff::Zero;
    use ark_std::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    fn expected(bases: &[Bn254G1Affine], column: &[u8]) -> Bn254G1 {
        let sum: G1Projective = Bn254G1Affine::as_inner_slice(bases)
            .iter()
            .zip(column)
            .filter_map(|(&base, &bit)| (bit != 0).then_some(base))
            .map(G1Projective::from)
            .sum();
        Bn254G1::from(sum)
    }

    #[test]
    fn random_columns_match_projective_sums() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        for len in [0, 1, 2, 3, 31, 128] {
            let bases: Vec<Bn254G1Affine> = (0..len)
                .map(|_| Bn254G1Affine(G1Projective::rand(&mut rng).into_affine()))
                .collect();
            let columns: Vec<Vec<u8>> = (0..64)
                .map(|_| (0..len).map(|_| (rng.next_u32() & 1) as u8).collect())
                .collect();
            let refs: Vec<&[u8]> = columns.iter().map(Vec::as_slice).collect();
            let got = g1_bit_columns_msm(&bases, &refs);
            for (got, column) in got.iter().zip(&columns) {
                assert_eq!(*got, expected(&bases, column));
            }
        }
    }

    #[test]
    fn handles_empty_singleton_duplicate_inverse_and_identity_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let p = G1Projective::rand(&mut rng).into_affine();
        let q = G1Projective::rand(&mut rng).into_affine();
        let bases = [
            Bn254G1Affine(G1Affine::identity()),
            Bn254G1Affine(p),
            Bn254G1Affine(p),
            Bn254G1Affine(-p),
            Bn254G1Affine(q),
        ];
        let mut columns = vec![
            vec![0, 0, 0, 0, 0],
            vec![0, 1, 0, 0, 0],
            vec![1, 1, 1, 1, 1],
            vec![0, 1, 0, 1, 0],
            vec![0, 1, 1, 0, 0],
        ];
        columns.extend((0..48).map(|_| vec![1, 1, 1, 1, 1]));
        let refs: Vec<&[u8]> = columns.iter().map(Vec::as_slice).collect();
        let got = g1_bit_columns_msm(&bases, &refs);
        for (got, column) in got.iter().zip(&columns) {
            assert_eq!(*got, expected(&bases, column));
        }
        assert_eq!(got[0], Bn254G1::from(G1Projective::zero()));
    }
}
